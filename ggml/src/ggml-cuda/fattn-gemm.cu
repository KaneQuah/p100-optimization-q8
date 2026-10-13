// Flash attention via cuBLAS GEMMs for Pascal (no tensor cores), where flash_attn_tile reaches only a
// fraction of the cuBLAS hgemm rate at long KV. Ported from Kmic-68/llama.cpp (738022bda and follow-ups).
//
// Online softmax over KV chunks, with the two matmuls done by cuBLAS:
//   S = K^T Q (f16 accumulate), m/l/P from S + mask, Otmp = V P (f16, ALGO4), O = O*corr + Otmp (f32)
// S is computed transposed ([nkv_chunk x n_tokens]) so one query's scores are contiguous.
// K/V that are not f16 (q8_0) are dequantized to f16 one chunk at a time.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-gemm.cuh"
#include "convert.cuh"

#include <cublas_v2.h>

// One block per (query token, head). Applies the mask, advances the running softmax statistics,
// writes the chunk's probabilities to P (never in place over S: the in-place write raced), and
// reports the rescale factor for O.
template <int block_size>
static __global__ void fattn_gemm_softmax(
        const half   * __restrict__ S,          // [nkv_c x nt] per head, column-major: scores
        half         * __restrict__ P,          // out, same layout: probabilities
        const half   * __restrict__ mask,       // rows of nkv keys, stride s_mask
        const float  * __restrict__ mask_first, // [nt] first nonzero mask column of each row
        float        * __restrict__ m_state,    // [nt x nh]
        float        * __restrict__ l_state,    // [nt x nh]
        float        * __restrict__ corr_out,   // [nt x nh]
        const int nkv_c,      // keys in this chunk
        const int nkv_off,    // offset of this chunk within the full KV
        const int nt,
        const int64_t s_mask, // mask row stride in elements
        const int64_t s_head) // per-head stride of S and P in elements
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const half * Sh = S + h*s_head + (int64_t) t*nkv_c;
    half       * Ph = P + h*s_head + (int64_t) t*nkv_c;
    // a chunk that ends before the row's first nonzero mask entry sees only +-0: skip the mask loads
    const half * mh = (float) (nkv_off + nkv_c) > mask_first[t] ? mask + (int64_t) t*s_mask + nkv_off : nullptr;

    __shared__ float red[block_size/WARP_SIZE];

    // pass 1: row max of (4*S + mask); Q already carried scale*0.25 into the GEMM
    float vmax = -FLT_MAX/2.0f;
    for (int j = tid; j < nkv_c; j += block_size) {
        float v = 4.0f*__half2float(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        vmax = fmaxf(vmax, v + FATTN_KQ_MAX_OFFSET);
    }
    vmax = warp_reduce_max(vmax);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = vmax;
        }
        __syncthreads();
        vmax = tid < block_size/WARP_SIZE ? red[tid] : -FLT_MAX/2.0f;
        vmax = warp_reduce_max(vmax);
        if (tid == 0) {
            red[0] = vmax;
        }
        __syncthreads();
        vmax = red[0];
        // all threads must have read red[0] before pass 2 reuses red for the row sum (data race otherwise)
        __syncthreads();
    }

    const float m_old = m_state[h*nt + t];
    const float m_new = fmaxf(m_old, vmax);
    const float corr  = m_old <= -FLT_MAX/4.0f ? 0.0f : expf(m_old - m_new);

    // pass 2: P = exp(v - m_new), and its row sum
    float sum = 0.0f;
    for (int j = tid; j < nkv_c; j += block_size) {
        float v = 4.0f*__half2float(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        const float p = (v <= -FLT_MAX/4.0f || m_new <= -FLT_MAX/4.0f) ? 0.0f : expf(v - m_new);
        Ph[j] = __float2half(p);
        sum += p;
    }
    sum = warp_reduce_sum(sum);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = sum;
        }
        __syncthreads();
        sum = tid < block_size/WARP_SIZE ? red[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            red[0] = sum;
        }
        __syncthreads();
        sum = red[0];
    }

    if (tid == 0) {
        m_state[h*nt + t]  = m_new;
        l_state[h*nt + t]  = l_state[h*nt + t]*corr + sum;
        corr_out[h*nt + t] = corr;
    }
}

// first[t] = smallest key index j with mask[t][j] != 0, or nkv if the row has none (-0.0 counts as zero)
template <int block_size>
static __global__ void fattn_gemm_mask_first_nz(
        const half * __restrict__ mask, float * __restrict__ first,
        const int64_t s_mask, const int nkv) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const half * mh = mask + (int64_t) t*s_mask;

    __shared__ float red[block_size/WARP_SIZE];

    // track max(-j) over nonzero entries, an exact float (j < 2^24)
    float neg = -(float) nkv;
    for (int j = tid; j < nkv; j += block_size) {
        if (__half2float(mh[j]) != 0.0f) {
            neg = fmaxf(neg, -(float) j);
        }
    }
    neg = warp_reduce_max(neg);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = neg;
        }
        __syncthreads();
        neg = tid < block_size/WARP_SIZE ? red[tid] : -(float) nkv;
        neg = warp_reduce_max(neg);
        if (tid == 0) {
            red[0] = neg;
        }
        __syncthreads();
        neg = red[0];
    }
    if (tid == 0) {
        first[t] = -neg;
    }
}

static __global__ void fattn_gemm_fill(float * __restrict__ p, const float v, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

// O_out = O_in*corr + Otmp: f16 sums span one chunk, accumulation across chunks is f32. Out of place.
static __global__ void fattn_gemm_accum_O(
        const float * __restrict__ O_in, float * __restrict__ O_out, const half * __restrict__ Otmp,
        const float * __restrict__ corr, const int DV, const int nt) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float c = corr[h*nt + t];
    const int64_t off = ((int64_t) h*nt + t)*DV;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        O_out[off + d] = O_in[off + d]*c + __half2float(Otmp[off + d]);
    }
}

// dst[d, h, t] = O[d, t, h] / l[t, h]; dst is [DV, n_head, n_tokens, n_seq]
static __global__ void fattn_gemm_finalize(
        const float * __restrict__ O, const float * __restrict__ l_state,
        float * __restrict__ dst,
        const int DV, const int nt, const int head0,
        const int64_t dst_s1, const int64_t dst_s2) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float l = l_state[h*nt + t];
    const float inv = l > 0.0f ? 1.0f/l : 0.0f;
    const float * Oh = O + ((int64_t) h*nt + t)*DV;
    float * dh = dst + (int64_t) t*dst_s2 + (int64_t)(head0 + h)*dst_s1;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        dh[d] = Oh[d]*inv;
    }
}

// Q (strided f32) -> contiguous f16 [D x nt] per head, pre-scaled by scale*0.25 before the f16 QK^T
// accumulation (overflow headroom, as in fattn-tile); the softmax multiplies the logits back by 4.
static __global__ void fattn_gemm_q_to_f16(
        const char * __restrict__ Q, half * __restrict__ Qf16,
        const int D, const int nt, const int64_t nbq1, const int64_t nbq2,
        const int head0, const float qscale) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float * q = (const float *) (Q + (int64_t) t*nbq1 + (int64_t)(head0 + h)*nbq2);
    half * o = Qf16 + ((int64_t) h*nt + t)*D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        o[d] = __float2half(q[d]*qscale);
    }
}

bool ggml_cuda_flash_attn_ext_gemm_supported(int cc, const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FA_GEMM");
        return !s || atoi(s) != 0;
    }();
    if (!enabled || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc >= GGML_CUDA_CC_VOLTA) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!mask || sinks || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || K->ne[0] != V->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    // only worth it once attention dominates
    if (Q->ne[1] < 128 || K->ne[1] < 4096) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    // a single flat mask pointer: it must be shared by all heads and sequences, and cover all keys
    if (mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1]) {
        return false;
    }
    for (const ggml_tensor * t : {K, V}) {
        if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0) {
            return false;
        }
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % sizeof(half) != 0) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t D    = K->ne[0];
    const int64_t DV   = V->ne[0];
    const int64_t nt   = Q->ne[1];
    const int64_t nh   = Q->ne[2];
    const int64_t nkv  = K->ne[1];
    const int64_t nhkv = K->ne[2];
    const int64_t ns   = Q->ne[3];
    const int64_t gqa  = nh / nhkv;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();
    cublasHandle_t cublas = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // the score matrix is chunk*nt*gqa halves
    const int64_t chunk = 2048;

    ggml_cuda_pool & pool = ctx.pool();
    // the pool may require frees in reverse order of allocation; destructors run in reverse declaration order
    ggml_cuda_pool_alloc<half>  Qf16(pool, D*nt*gqa);
    ggml_cuda_pool_alloc<half>  Kf16(pool);
    ggml_cuda_pool_alloc<half>  Vf16(pool);
    if (K->type != GGML_TYPE_F16) { Kf16.alloc(D*chunk); }
    if (V->type != GGML_TYPE_F16) { Vf16.alloc(DV*chunk); }
    ggml_cuda_pool_alloc<half>  S(pool, chunk*nt*gqa);
    ggml_cuda_pool_alloc<half>  P(pool, chunk*nt*gqa);
    ggml_cuda_pool_alloc<float> O(pool, DV*nt*gqa);
    ggml_cuda_pool_alloc<float> O2(pool, DV*nt*gqa);
    ggml_cuda_pool_alloc<half>  Otmp(pool, DV*nt*gqa);
    ggml_cuda_pool_alloc<float> m_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> l_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> corr(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> mask_first(pool, nt);

    const bool K_is_f16 = K->type == GGML_TYPE_F16;
    const bool V_is_f16 = V->type == GGML_TYPE_F16;
    const to_fp16_nc_cuda_t to_fp16_K = K_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(K->type);
    const to_fp16_nc_cuda_t to_fp16_V = V_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(V->type);
    GGML_ASSERT(K_is_f16 || to_fp16_K);
    GGML_ASSERT(V_is_f16 || to_fp16_V);

    const int64_t dst_s1 = dst->nb[1]/sizeof(float);
    const int64_t dst_s2 = dst->nb[2]/sizeof(float);
    const int64_t s_mask = mask->nb[1]/sizeof(half);

    {
        fattn_gemm_mask_first_nz<256><<<dim3(nt, 1, 1), 256, 0, stream>>>(
            (const half *) mask->data, mask_first.ptr, s_mask, (int) nkv);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int64_t s = 0; s < ns; ++s) {
        for (int64_t kvh = 0; kvh < nhkv; ++kvh) {
            const int64_t head0 = kvh*gqa;

            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_q_to_f16<<<grid, 256, 0, stream>>>(
                    (const char *) Q->data + s*Q->nb[3], Qf16.ptr,
                    D, nt, Q->nb[1], Q->nb[2], head0, scale*0.25f);
                CUDA_CHECK(cudaGetLastError());
            }

            float * O_cur = O.ptr;
            float * O_nxt = O2.ptr;
            CUDA_CHECK(cudaMemsetAsync(O_cur, 0, DV*nt*gqa*sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(l_state.ptr, 0, nt*gqa*sizeof(float), stream));
            {
                const int64_t n = nt*gqa;
                fattn_gemm_fill<<<(n + 255)/256, 256, 0, stream>>>(m_state.ptr, -INFINITY, n);
                CUDA_CHECK(cudaGetLastError());
            }

            for (int64_t c = 0; c < nkv; c += chunk) {
                const int64_t nkv_c = std::min(chunk, nkv - c);

                // f16 is used in place via the cuBLAS leading dimension; q8_0 is dequantized per chunk
                const char * Kp = (const char *) K->data + s*K->nb[3] + kvh*K->nb[2] + c*K->nb[1];
                const char * Vp = (const char *) V->data + s*V->nb[3] + kvh*V->nb[2] + c*V->nb[1];
                const half * Kmat;
                const half * Vmat;
                int64_t ldK;
                int64_t ldV;
                if (K_is_f16) {
                    Kmat = (const half *) Kp;
                    ldK  = K->nb[1]/sizeof(half);
                } else {
                    to_fp16_K(Kp, Kf16.ptr, D, nkv_c, 1, 1, K->nb[1]/ggml_type_size(K->type), 0, 0, stream);
                    Kmat = Kf16.ptr;
                    ldK  = D;
                }
                if (V_is_f16) {
                    Vmat = (const half *) Vp;
                    ldV  = V->nb[1]/sizeof(half);
                } else {
                    to_fp16_V(Vp, Vf16.ptr, DV, nkv_c, 1, 1, V->nb[1]/ggml_type_size(V->type), 0, 0, stream);
                    Vmat = Vf16.ptr;
                    ldV  = DV;
                }

                // S = K^T Q -> [nkv_c x nt] per head, column-major. alpha/beta are half for COMPUTE_16F.
                {
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nkv_c, nt*gqa, D,
                        &alpha,
                        Kmat,     CUDA_R_16F, ldK,
                        Qf16.ptr, CUDA_R_16F, D,
                        &beta,
                        S.ptr,    CUDA_R_16F, nkv_c,
                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                }

                {
                    dim3 grid(nt, gqa, 1);
                    fattn_gemm_softmax<256><<<grid, 256, 0, stream>>>(
                        S.ptr, P.ptr, (const half *) mask->data, mask_first.ptr,
                        m_state.ptr, l_state.ptr, corr.ptr, nkv_c, c, nt, s_mask, nkv_c*nt);
                    CUDA_CHECK(cudaGetLastError());
                }

                // Otmp = V P. ALGO4 accumulates in blocks (NMSE ~10x lower than the DEFAULT long
                // fp16 chains at large nt*gqa); fall back to DEFAULT if it is unavailable.
                {
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    for (const cublasGemmAlgo_t algo : {CUBLAS_GEMM_ALGO4, CUBLAS_GEMM_DEFAULT}) {
                        const cublasStatus_t st = cublasGemmEx(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            DV, nt*gqa, nkv_c,
                            &alpha,
                            Vmat,     CUDA_R_16F, ldV,
                            P.ptr,    CUDA_R_16F, nkv_c,
                            &beta,
                            Otmp.ptr, CUDA_R_16F, DV,
                            CUBLAS_COMPUTE_16F, algo);
                        if (st == CUBLAS_STATUS_SUCCESS) {
                            break;
                        }
                        if (algo == CUBLAS_GEMM_DEFAULT) {
                            CUBLAS_CHECK(st);
                        }
                    }
                }

                {
                    dim3 grid(nt, gqa, 1);
                    fattn_gemm_accum_O<<<grid, 256, 0, stream>>>(O_cur, O_nxt, Otmp.ptr, corr.ptr, DV, nt);
                    CUDA_CHECK(cudaGetLastError());
                    std::swap(O_cur, O_nxt);
                }
            }

            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_finalize<<<grid, 256, 0, stream>>>(
                    O_cur, l_state.ptr, (float *) dst->data + s*(dst->nb[3]/sizeof(float)),
                    DV, nt, head0, dst_s1, dst_s2);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}
