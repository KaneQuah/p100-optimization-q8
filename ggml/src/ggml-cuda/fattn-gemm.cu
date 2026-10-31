// Flash attention via cuBLAS GEMMs for Pascal (no tensor cores), where flash_attn_tile reaches only a
// fraction of the cuBLAS hgemm rate at long KV. Ported from Kmic-68/llama.cpp (738022bda and follow-ups).
//
// Online softmax over KV chunks, with the two matmuls done by cuBLAS:
//   S = K^T Q (f16 accumulate), m/l/P from S + mask, Otmp = V P (f16, ALGO4), O = O*corr + Otmp (f32)
// S is computed transposed ([nkv_chunk x n_tokens]) so one query's scores are contiguous.
// K/V that are not f16 (q8_0) are dequantized to f16 one chunk at a time.
// q8_0 K/V (and f16 with GGML_CUDA_FA_FOLD=2) take the fold kernels below instead.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-gemm.cuh"
#include "convert.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// NVIDIA-only at runtime, no ALGO4 in hipBLAS/muBLAS
#define CUBLAS_GEMM_ALGO4 CUBLAS_GEMM_DEFAULT
#endif

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

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// ---------------------------------------------------------------------------------------------
// Fold path (GGML_CUDA_FA_FOLD, default on for a q4_0 / q8_0 K/V cache; 2 also takes f16), ported from
// Kmic-68/llama.cpp 7461904db (q8_0 / f16 templating from our PR #3): two hand-written GEMM kernels
// (fp16 HFMA2 products, half2 chains folded into fp32) replace dequant + cuBLAS QK + softmax +
// cuBLAS PV + rescale. Not taken: his bank-fixed SASS, the diagnostics, the first PV kernel and the
// compact I32 causal mask (prefix stays false here).
//
//   fa_fold_qk2: S tile = K Q^T for 128 keys x 128 query columns, k = D. The epilogue applies the
//                mask, takes the max of the tile per column (m_t), writes P = exp(4S + mask - m_t)
//                in f16 and the tile's fp32 row sum l_t. S never reaches memory.
//   fa_fold_pv2: O = O*corr + sum_t exp(m_t - M) * (V P_t), with M the new running max. The fp16
//                chains restart at every 128-key tile and are folded into fp32 with the tile's
//                factor, so the rescale costs one FMUL per fold.
//
// Precision against the cuBLAS path: QK^T chains are 128 (not 256) long and summed in fp32; P is
// relative to the tile max instead of the chunk max (more bits); PV accumulates fp16 over 128 keys
// (not over the 2048-key chunk) and fp32 across tiles.
namespace fa_fold {

// The KV range is cut into nsplit independent streams of L keys each (own O, m, l), merged at the
// end: the PV grid is only (DV/128) x (N/128) x heads, 3.4 waves on 56 SMs with one stream.
// Virtual head z = kv_head*nsplit + split; chunk c of every stream runs in the same launch.
struct fa_fold_split {
    int nsplit, L, c, nkv;
};
static __device__ __forceinline__ int fa_fold_kv_off(const fa_fold_split & g, const int z) {
    return (z % g.nsplit)*g.L + g.c;
}
static __device__ __forceinline__ int fa_fold_nkv_c(const fa_fold_split & g, const int z, const int C) {
    const int off = fa_fold_kv_off(g, z);
    return max(0, min(min(C, g.L - g.c), g.nkv - off));
}

constexpr int BM  = 128;
constexpr int BN  = 128;
constexpr int BK2 = 16;   // k2 steps per smem tile = 32 values of K
constexpr int TK  = 128;  // keys per softmax tile (= QK BM, and the PV fold period)
constexpr int MAXTILE = 16; // tiles per chunk: chunk <= 2048 keys

static __device__ __forceinline__ float fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));  // high lane = lo + hi
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);                 // = (lo + hi) * 2^-112
}

// 32 consecutive values (block b of a row) of a q4_0, q8_0 or f16 K/V row as exact fp32:
// q4_0 d*(q-8) needs 15 significant bits, q8_0 d*q 19, f16 is f16; all fit the 24 of a float.
static_assert(QK4_0 == QK8_0, "fold dequant works on 32-value blocks");
template <ggml_type T>
static __device__ __forceinline__ void fa_fold_row32(const char * __restrict__ row, const int b, float * __restrict__ v) {
    if constexpr (T == GGML_TYPE_Q4_0) {
        const block_q4_0 * blk = (const block_q4_0 *) row + b;
        const float d = __half2float(blk->d);
#pragma unroll
        for (int j = 0; j < QK4_0/2; j++) {
            const int q = blk->qs[j];
            v[j]           = d*(float) ((q & 0xF) - 8);
            v[j + QK4_0/2] = d*(float) ((q >> 4)  - 8);
        }
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const block_q8_0 * blk = (const block_q8_0 *) row + b;
        const float d = __half2float(blk->d);
        const uint16_t * q2 = (const uint16_t *) blk->qs;   // 34-byte blocks: qs is only 2-byte aligned
#pragma unroll
        for (int j = 0; j < QK8_0/2; j++) {
            const int w = q2[j];
            v[2*j + 0] = d*(float) (int8_t) (w & 0xFF);
            v[2*j + 1] = d*(float) (int8_t) (w >> 8);
        }
    } else {
        static_assert(T == GGML_TYPE_F16, "unsupported fold KV type");
        const half2 * p = (const half2 *) row + b*(QK4_0/2);
#pragma unroll
        for (int j = 0; j < QK4_0/2; j++) {
            const float2 f = __half22float2(p[j]);
            v[2*j + 0] = f.x;
            v[2*j + 1] = f.y;
        }
    }
}

// Dequantize one chunk of a q4_0 / q8_0 / f16 K and V cache for all KV heads:
//   K16[h][key][D]              (row per key, D contiguous)
//   Vp [h][key/2][DV] as half2  (lo lane = even key, hi lane = odd key: the PV A operand)
// Keys from nkv_c up to the padded chunk are written as zeros (P is zero there too, and 0*garbage
// could be NaN). Exact values (times the block scale) rounded once to half: identical to
// the generic dequantizer.
template <ggml_type TK, ggml_type TV>
static __global__ void fa_fold_dequant(
        const char * __restrict__ K, const char * __restrict__ V,
        half * __restrict__ K16, half2 * __restrict__ Vp,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
        const int D, const fa_fold_split g, const int C) {
    const int nb   = D/QK4_0;
    const int b    = threadIdx.x % nb;          // block within a row
    const int kp   = blockIdx.x*(blockDim.x/nb) + threadIdx.x/nb;  // key pair
    const int z    = blockIdx.y;
    const int h    = z / g.nsplit;
    if (2*kp >= C) {
        return;
    }
    const int nkv_c  = fa_fold_nkv_c(g, z, C);
    const int kv_off = fa_fold_kv_off(g, z);
    half * k16 = K16 + ((int64_t) z*C + 2*kp)*D + b*QK4_0;
    half2 * vp = Vp  + ((int64_t) z*(C/2) + kp)*D + b*QK4_0;
    float vlo[QK4_0], vhi[QK4_0];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int key = 2*kp + r;
        float * vv = r == 0 ? vlo : vhi;
        if (key < nkv_c) {
            float kf[QK4_0];
            fa_fold_row32<TK>(K + h*nbk2 + (int64_t) (kv_off + key)*nbk1, b, kf);
            fa_fold_row32<TV>(V + h*nbv2 + (int64_t) (kv_off + key)*nbv1, b, vv);
            half kk[QK4_0];
#pragma unroll
            for (int j = 0; j < QK4_0; j++) {
                kk[j] = __float2half(kf[j]);
            }
#pragma unroll
            for (int j = 0; j < QK4_0; j += 8) {
                *(uint4 *) (k16 + r*D + j) = *(const uint4 *) (kk + j);
            }
        } else {
#pragma unroll
            for (int j = 0; j < QK4_0; j++) {
                vv[j] = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < QK4_0; j += 8) {
                *(uint4 *) (k16 + r*D + j) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    half2 pv[QK4_0];
#pragma unroll
    for (int j = 0; j < QK4_0; j++) {
        pv[j] = __halves2half2(__float2half(vlo[j]), __float2half(vhi[j]));
    }
#pragma unroll
    for (int j = 0; j < QK4_0; j += 4) {
        *(uint4 *) (vp + j) = *(const uint4 *) (pv + j);
    }
}

template <ggml_type TK, typename... A>
static void fa_fold_dequant_v(const ggml_type tv, const dim3 grid, const int thr, cudaStream_t st, A... a) {
    switch (tv) {
        case GGML_TYPE_Q4_0: fa_fold_dequant<TK, GGML_TYPE_Q4_0><<<grid, thr, 0, st>>>(a...); break;
        case GGML_TYPE_Q8_0: fa_fold_dequant<TK, GGML_TYPE_Q8_0><<<grid, thr, 0, st>>>(a...); break;
        case GGML_TYPE_F16:  fa_fold_dequant<TK, GGML_TYPE_F16> <<<grid, thr, 0, st>>>(a...); break;
        default: GGML_ABORT("fold attention: unsupported V type");
    }
}

template <typename... A>
static void fa_fold_dequant_launch(const ggml_type tk, const ggml_type tv, const dim3 grid, const int thr, cudaStream_t st, A... a) {
    switch (tk) {
        case GGML_TYPE_Q4_0: fa_fold_dequant_v<GGML_TYPE_Q4_0>(tv, grid, thr, st, a...); break;
        case GGML_TYPE_Q8_0: fa_fold_dequant_v<GGML_TYPE_Q8_0>(tv, grid, thr, st, a...); break;
        case GGML_TYPE_F16:  fa_fold_dequant_v<GGML_TYPE_F16> (tv, grid, thr, st, a...); break;
        default: GGML_ABORT("fold attention: unsupported K type");
    }
}

// Row (A / m) and column (B / n) of a thread's i-th / j-th output.
static __device__ __forceinline__ int fa_fold_row(const int i) { return (i < 4 ? 0 : 64) + (threadIdx.x >> 4)*4 + (i & 3); }
static __device__ __forceinline__ int fa_fold_col(const int j) { return (j < 4 ? 0 : 64) + (threadIdx.x & 15)*4 + (j & 3); }

// QK + softmax, two CTAs per SM. k = D is short (8 smem tiles at D = 256), so with one CTA per SM
// the prologue load and the softmax epilogue sat exposed (main loop 72% of peak, epilogue +20%).
// Here the fp16 chains run over the whole of D (two lanes of D/4 products each, then one fp32
// fold -- still shorter chains than cuBLAS COMPUTE_16F's single D-long one), so no fp32
// accumulators live through the loop, and the smem tiles are single-buffered: <= 128 registers
// and ~17 KiB of smem, and a second CTA covers each one's latency.
static __global__ void __launch_bounds__(256, 2) fa_fold_qk2(
        const half * __restrict__ K16, const half * __restrict__ Q16,
        const half * __restrict__ mask, const float * __restrict__ mask_first, const int64_t s_mask,
        half * __restrict__ P, float * __restrict__ mt, float * __restrict__ lt,
        const int D, const int N, const int nt, const int C, const fa_fold_split g, const bool prefix = false) {
    union smem_t {
        struct { uint32_t As[BK2][BM]; uint32_t Bs[BK2][BN]; } t;
        float red[16][BN];
        uint2 Ps[BN*16];
    };
    __shared__ __align__(16) smem_t sm;
    __shared__ float colv[BN];
    __shared__ int   colq[BN];

    const int h   = blockIdx.z;
    const int m0  = blockIdx.x*TK;
    const int n0  = blockIdx.y*BN;
    const int t   = threadIdx.x;
    const int tx  = t & 15;
    const int ty  = t >> 4;
    const int nkv_c  = fa_fold_nkv_c(g, h, C);
    const int kv_off = fa_fold_kv_off(g, h);

    int cq = -1;
    if (t < BN) {
        const int n  = n0 + t;
        const int tq = n < N ? n % nt : 0;
        cq = n < N && (float) (kv_off + m0 + TK) > mask_first[tq] ? tq : -1;
        colq[t] = cq;
    }
    // block-uniform: does any column need its mask here, is this the ragged last tile?
    // (under a causal mask only the diagonal chunk does; everything else takes the plain path)
    const bool any_mask = __syncthreads_or(cq >= 0);
    const bool ragged   = m0 + TK > nkv_c;

    // prefix mask (compact causal): a key at or past mask_first[t] is masked for query t, so a tile
    // whose first key is past every column's mask_first computes to exactly P = 0, m_t = -inf,
    // l_t = 0 (see the end of this kernel). Write those without the GEMM.
    if (prefix) {
        bool cskip = true;
        if (t < BN) {
            const int n = n0 + t;
            cskip = n >= N || mask_first[n % nt] <= (float) (kv_off + m0);
        }
        if (__syncthreads_and(cskip)) {
#pragma unroll
            for (int ih = 0; ih < 2; ih++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int idx = t + 256*k;
                    const int kbl = idx >> 9;
                    const int r   = (idx >> 2) & 127;
                    *(uint4 *) (P + (((int64_t) (h*gridDim.y + blockIdx.y)*(C/32) + m0/32 + ih*2 + kbl)*BN + r)*32 + (idx & 3)*8) =
                        make_uint4(0, 0, 0, 0);
                }
            }
            if (t < BN && n0 + t < N) {
                const int64_t o = ((int64_t) h*(C/TK) + blockIdx.x)*N + n0 + t;
                mt[o] = -INFINITY;
                lt[o] = 0.0f;
            }
            return;
        }
    }

    const half * A = K16 + ((int64_t) h*C + m0)*D;
    const half * B = Q16 + ((int64_t) (h / g.nsplit)*N + n0)*D;
    const int nB = N - n0;

    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            hh[i][j] = make_half2(0.0f, 0.0f);
        }
    }
    uint4 ra[2], rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = *(const uint4 *) (A + (int64_t) r*D + k0 + c*8);
            rb[i] = r < nB ? *(const uint4 *) (B + (int64_t) r*D + k0 + c*8) : make_uint4(0, 0, 0, 0);
        }
    };
    const int nit = D / (2*BK2);
    gload(0);
    for (int it = 0; it < nit; it++) {
        if (it > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l  = t + 256*i;
            const int r  = l >> 2;
            const int c  = l & 3;
            const int rs = r ^ (c << 3);
            sm.t.As[c*4 + 0][rs] = ra[i].x; sm.t.As[c*4 + 1][rs] = ra[i].y;
            sm.t.As[c*4 + 2][rs] = ra[i].z; sm.t.As[c*4 + 3][rs] = ra[i].w;
            sm.t.Bs[c*4 + 0][rs] = rb[i].x; sm.t.Bs[c*4 + 1][rs] = rb[i].y;
            sm.t.Bs[c*4 + 2][rs] = rb[i].z; sm.t.Bs[c*4 + 3][rs] = rb[i].w;
        }
        __syncthreads();
        if (it + 1 < nit) {
            gload((it + 1)*2*BK2);
        }
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &sm.t.As[k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &sm.t.As[k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &sm.t.Bs[k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &sm.t.Bs[k2][(64 + tx*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            // j outer and b in the first slot: same products per chain (bit-identical), but ptxas then leaves
            // 163 instead of 300 same-bank source pairs after bankfix (qk2 181.8 -> 177.7 ms on the model shape)
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    hh[i][j] = __hfma2(bj, ai, hh[i][j]);
                }
            }
        }
    }

    // logits in log2 units: log2(e) * (4*S + mask) (Q carried scale/4), from the one fold; keys
    // past nkv_c are -inf. The whole softmax state (m_t, M, the running max) is kept in log2
    // units so every exponential is a bare exp2f.
    float acc[8][8];
    float cmax[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        cmax[j] = -FLT_MAX/2.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i][j] = fold_h(hh[i][j]) * (0x1p114f*1.44269504088896341f);
        }
        if (any_mask || ragged) {
            const int tq = colq[fa_fold_col(j)];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int key = fa_fold_row(i);
                if (m0 + key >= nkv_c) {
                    acc[i][j] = -INFINITY;
                } else if (tq >= 0) {
                    acc[i][j] += 1.44269504088896341f*__half2float(mask[(int64_t) tq*s_mask + kv_off + m0 + key]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            cmax[j] = fmaxf(cmax[j], acc[i][j]);
        }
    }
    __syncthreads();   // all reads of the tiles are done: red aliases them
#pragma unroll
    for (int j = 0; j < 8; j++) {
        sm.red[ty][fa_fold_col(j)] = cmax[j];
    }
    __syncthreads();
    if (t < BN) {
        float m = -FLT_MAX/2.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            m = fmaxf(m, sm.red[r][t]);
        }
        colv[t] = m + 3.0f;   // FATTN_KQ_MAX_OFFSET (3 ln 2) in log2 units: every p <= 1/8
    }
    __syncthreads();
    float csum[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        csum[j] = 0.0f;
    }
    // P = exp(v - m_t) in f16, staged through smem in two 64-key halves and written as whole
    // 8 KiB blocks of the [h][n/128][key/32][128][32] layout that fa_fold_pv reads
#pragma unroll
    for (int ih = 0; ih < 2; ih++) {
        if (ih > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int c = fa_fold_col(j);
            const float m = colv[c];
            half pp[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                // v = -inf gives 0; a fully masked column has m ~ -FLT_MAX/2 and v = -inf, also 0
                const float p = exp2f(acc[ih*4 + q][j] - m);
                pp[q] = __float2half(p);
                csum[j] += p;
            }
            sm.Ps[c*16 + (ty ^ (((c >> 2) & 7) << 1))] = *(const uint2 *) pp;
        }
        __syncthreads();
        const uint4 * Ps = (const uint4 *) sm.Ps;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int idx = t + 256*k;
            const int kbl = idx >> 9;
            const int r   = (idx >> 2) & 127;
            const int q   = kbl*4 + (idx & 3);
            *(uint4 *) (P + (((int64_t) (h*gridDim.y + blockIdx.y)*(C/32) + m0/32 + ih*2 + kbl)*BN + r)*32 + (idx & 3)*8) =
                Ps[r*8 + (q ^ ((r >> 2) & 7))];
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
        sm.red[ty][fa_fold_col(j)] = csum[j];
    }
    __syncthreads();
    if (t < BN) {
        const int n = n0 + t;
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            s += sm.red[r][t];
        }
        if (n < N) {
            const int64_t o = ((int64_t) h*(C/TK) + blockIdx.x)*N + n;
            mt[o] = colv[t] <= -FLT_MAX/8.0f ? -INFINITY : colv[t];
            lt[o] = s;
        }
    }
}

// PV with gemm_fold_kernel_u2's main loop (Kmic OPTLOG 238): load/store offsets computed once, the
// tile loop unrolled by two with compile-time smem buffers, chains restarted by HMUL2, and the fold as
// HADD2 -> f32 without the 2^-112 bit trick (subnormal chain sums and tiny products are kept instead of
// flushed). Chains of 128 keys; -14% PV time against his first PV kernel (OPTLOG 256).
static __global__ void __launch_bounds__(256, 1) fa_fold_pv2(
        const half2 * __restrict__ Vp, const half * __restrict__ P,
        const float * __restrict__ mt, const float * __restrict__ lt,
        const float * __restrict__ m_in, float * __restrict__ m_out, float * __restrict__ l_state,
        const float * __restrict__ O_in, float * __restrict__ O_out,
        const int DV, const int N, const int C, const int ntile,
        const float * __restrict__ mask_first = nullptr, const int nt = 0, const fa_fold_split g = {}, const bool prefix = false) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];
    __shared__ float corr[BN];
    __shared__ float fac[MAXTILE][BN];   // exp(m_t - M) per tile and column

    const int h  = blockIdx.z;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;

    auto prologue = [&]() {
        if (t < BN) {
            const int n = n0 + t;
            float M = -INFINITY, c = 0.0f;
            if (n < N) {
                // all tile maxima loaded at once (independent loads; ntile <= MAXTILE)
                float mm[MAXTILE];
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    mm[tt] = tt < ntile ? mt[((int64_t) h*(C/TK) + tt)*N + n] : -INFINITY;
                }
                const float mo = m_in[(int64_t) h*N + n];
                M = mo;
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    M = fmaxf(M, mm[tt]);
                }
                c = mo == -INFINITY ? 0.0f : exp2f(mo - M);
                if (blockIdx.x == 0) {
                    float l = l_state[(int64_t) h*N + n]*c;
                    if (M != -INFINITY) {
                        float lv[MAXTILE];
#pragma unroll
                        for (int tt = 0; tt < MAXTILE; tt++) {
                            lv[tt] = tt < ntile ? lt[((int64_t) h*(C/TK) + tt)*N + n] : 0.0f;
                        }
                        for (int tt = 0; tt < ntile; tt++) {
                            l += mm[tt] == -INFINITY ? 0.0f : lv[tt]*exp2f(mm[tt] - M);
                        }
                    }
                    l_state[(int64_t) h*N + n] = l;
                    m_out[(int64_t) h*N + n]   = M;
                }
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    if (tt < ntile) {
                        fac[tt][t] = (mm[tt] == -INFINITY || M == -INFINITY) ? 0.0f : exp2f(mm[tt] - M);
                    }
                }
            } else {
                for (int tt = 0; tt < ntile; tt++) {
                    fac[tt][t] = 0.0f;
                }
            }
            corr[t] = c;
        }
    };

    int ntile_run = ntile;
    if (prefix) {
        __shared__ int kmax;
        if (t == 0) {
            kmax = 0;
        }
        __syncthreads();
        if (t < BN && n0 + t < N) {
            atomicMax(&kmax, (int) mask_first[(n0 + t) % nt]);
        }
        __syncthreads();
        const int vis = kmax - fa_fold_kv_off(g, h);
        ntile_run = max(1, min(ntile, (vis + TK - 1)/TK));
    }

    float acc[8][8];
    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            hh[i][j]  = make_half2(0.0f, 0.0f);
        }
    }

    // A: V as half2 key pairs [k2][DV] (DV words per k2 row); B: P blocked [key/32][128][32]
    const uint32_t * A = (const uint32_t *) (Vp + (int64_t) h*(C/2)*DV + m0);
    const half     * B = P + ((int64_t) h*gridDim.y + blockIdx.y)*C*BN;
    const uint4 * pa[2];
    const uint4 * pb[2];
    int sa[2], sb[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256*i;
        const int ra_ = l >> 5, ca = l & 31;
        pa[i] = (const uint4 *) (A + (int64_t) ra_*DV + ca*4);
        sa[i] = ra_*BM + ((ca*4) ^ ((ra_ >> 2) << 3));
        const int rb_ = l >> 2, cb = l & 3;
        pb[i] = (const uint4 *) (B + rb_*32 + cb*8);
        sb[i] = (cb*4)*BN + (rb_ ^ (cb << 3));
    }
    const int64_t da = (int64_t) BK2*DV/4;   // uint4 per tile along A
    constexpr int db = BN*32/8;              // uint4 per tile along B
    int ao[4], bo[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        ao[q] = (ty*4) ^ (q << 3);
        bo[q] = (tx*4) ^ (q << 3);
    }

    uint4 ra[2], rb[2];
    auto gload = [&](const int it) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            ra[i] = pa[i][it*da];
            rb[i] = pb[i][it*db];
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            *(uint4 *) (&As[buf][0][0] + sa[i]) = ra[i];
            uint32_t * b = &Bs[buf][0][0] + sb[i];
            b[0] = rb[i].x; b[BN] = rb[i].y; b[2*BN] = rb[i].z; b[3*BN] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const uint4 a0 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2]];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2] + 64];
            const uint4 b0 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2]];
            const uint4 b1 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2] + 64];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    hh[i][j] = k2 == 0 && restart ? __hmul2(ai, bj) : __hfma2(ai, bj, hh[i][j]);
                }
            }
        }
    };
    auto fold = [&](const int tt) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float f = fac[tt][fa_fold_col(j)];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                acc[i][j] += __half2float(__hadd(__low2half(hh[i][j]), __high2half(hh[i][j]))) * f;
            }
        }
    };

    constexpr int TPF = TK/2/BK2;         // tiles per fold (128 keys)
    const int ntl = ntile_run*TPF;        // tiles of 32 keys, a multiple of 4
    gload(0);
    prologue();
    sstore(0);
    __syncthreads();
    for (int it = 0; it < ntl; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < ntl) {
            gload(it + 2);
        }
        tile(1, false);
        if ((it + 2) % TPF == 0) {
            fold((it + 2)/TPF - 1);
        }
        if (it + 2 < ntl) {
            sstore(0);
        }
        __syncthreads();
    }

    float4 oin[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = min(n0 + fa_fold_col(j), N - 1);
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            oin[j][ih] = *(const float4 *) (O_in + ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int c = fa_fold_col(j);
        const int n = n0 + c;
        if (n >= N) {
            continue;
        }
        const float cr = corr[c];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int64_t o = ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4;
            const float4 oi = oin[j][ih];
            float4 v;
            v.x = oi.x*cr + acc[ih*4 + 0][j];
            v.y = oi.y*cr + acc[ih*4 + 1][j];
            v.z = oi.z*cr + acc[ih*4 + 2][j];
            v.w = oi.w*cr + acc[ih*4 + 3][j];
            *(float4 *) (O_out + o) = v;
        }
    }
}

// dst[d, head, t] = sum_split w*O / sum_split w*l, w = exp(m_split - max m): the split merge
static __global__ void fa_fold_finalize(
        const float * __restrict__ O, const float * __restrict__ m, const float * __restrict__ l,
        float * __restrict__ dst, const int DV, const int nt, const int N, const int gqa, const int nsplit,
        const int64_t dst_s1, const int64_t dst_s2) {
    const int t   = blockIdx.x;
    const int hq  = blockIdx.y;
    const int kvh = blockIdx.z;
    const int n   = hq*nt + t;
    float M = -INFINITY;
    for (int sp = 0; sp < nsplit; sp++) {
        M = fmaxf(M, m[(int64_t) (kvh*nsplit + sp)*N + n]);
    }
    float w[8];
    float L = 0.0f;
    for (int sp = 0; sp < nsplit; sp++) {
        const int64_t o = (int64_t) (kvh*nsplit + sp)*N + n;
        w[sp] = m[o] == -INFINITY ? 0.0f : exp2f(m[o] - M);
        L += w[sp]*l[o];
    }
    const float inv = L > 0.0f ? 1.0f/L : 0.0f;
    float * dh = dst + (int64_t) t*dst_s2 + (int64_t) (kvh*gqa + hq)*dst_s1;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float v = 0.0f;
        for (int sp = 0; sp < nsplit; sp++) {
            v += w[sp]*O[((int64_t) (kvh*nsplit + sp)*N + n)*DV + d];
        }
        dh[d] = v*inv;
    }
}

} // namespace fa_fold

static int ggml_cuda_fa_fold_env(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

// GGML_CUDA_FA_FOLD: 0 off (q4_0 back to the tile kernel), 1 (default) q4_0 and q8_0 K/V in any combination,
// 2 also f16 K and/or V.
static bool fa_fold_type_ok(const ggml_tensor * t, const int mode) {
    switch (t->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0: return true;
        case GGML_TYPE_F16:  // half2 loads
            return mode >= 2 && t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && (uintptr_t) t->data % 4 == 0;
        default: return false;
    }
}

static bool ggml_cuda_fa_fold_usable(const ggml_tensor * dst) {
    static const int mode = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD", 1);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return mode != 0 &&
        fa_fold_type_ok(K, mode) && fa_fold_type_ok(V, mode) &&
        K->ne[0] == V->ne[0] && K->ne[0] % 128 == 0 && K->ne[0] <= 512 &&
        K->ne[2] == V->ne[2] && Q->ne[1]*(Q->ne[2]/K->ne[2]) <= 65535*fa_fold::BN;
}

static void ggml_cuda_flash_attn_ext_fold(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fa_fold;
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
    const int64_t N    = nt*gqa;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();

    // GGML_CUDA_FA_FOLD_CHUNK keys per chunk (default 2048, at most MAXTILE*TK; +2.2% prefill at 262k over 1024),
    // GGML_CUDA_FA_FOLD_SPLIT independent KV streams (default 2, at most 8)
    static const int chunk_env  = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD_CHUNK", 2048);
    static const int nsplit_env = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD_SPLIT", 2);
    const int64_t nsplit = std::max(1, std::min(8, nsplit_env));
    const int64_t Lsp = ((nkv + nsplit - 1)/nsplit + TK - 1)/TK*TK;   // keys per split
    const int64_t C = std::max<int64_t>(TK, (std::min<int64_t>(std::min(chunk_env, MAXTILE*TK), Lsp) + TK - 1)/TK*TK);
    const int64_t nz = nhkv*nsplit;   // virtual heads

    ggml_cuda_pool & pool = ctx.pool();
    // the pool may require frees in reverse order of allocation; destructors run in reverse declaration order
    ggml_cuda_pool_alloc<half>  Qf16(pool, D*N*nhkv);
    ggml_cuda_pool_alloc<half>  K16(pool, D*C*nz);
    ggml_cuda_pool_alloc<half2> Vp(pool, DV*(C/2)*nz);
    ggml_cuda_pool_alloc<half>  P(pool, C*((N + BN - 1)/BN*BN)*nz);
    ggml_cuda_pool_alloc<float> mt(pool, (C/TK)*N*nz);
    ggml_cuda_pool_alloc<float> lt(pool, (C/TK)*N*nz);
    ggml_cuda_pool_alloc<float> O(pool, DV*N*nz);
    ggml_cuda_pool_alloc<float> O2(pool, DV*N*nz);
    ggml_cuda_pool_alloc<float> mA(pool, N*nz);
    ggml_cuda_pool_alloc<float> mB(pool, N*nz);
    ggml_cuda_pool_alloc<float> l_state(pool, N*nz);
    ggml_cuda_pool_alloc<float> mask_first(pool, nt);

    const int64_t dst_s1 = dst->nb[1]/sizeof(float);
    const int64_t dst_s2 = dst->nb[2]/sizeof(float);
    const int64_t s_mask = mask->nb[1]/sizeof(half);
    const half *  mask_d = (const half *) mask->data;

    fattn_gemm_mask_first_nz<256><<<nt, 256, 0, stream>>>(mask_d, mask_first.ptr, s_mask, (int) nkv);
    CUDA_CHECK(cudaGetLastError());

    for (int64_t s = 0; s < ns; ++s) {
        for (int64_t kvh = 0; kvh < nhkv; ++kvh) {
            fattn_gemm_q_to_f16<<<dim3(nt, gqa, 1), 256, 0, stream>>>(
                (const char *) Q->data + s*Q->nb[3], Qf16.ptr + kvh*N*D,
                D, nt, Q->nb[1], Q->nb[2], kvh*gqa, scale*0.25f);
            CUDA_CHECK(cudaGetLastError());
        }
        float * O_cur = O.ptr;
        float * O_nxt = O2.ptr;
        float * m_cur = mA.ptr;
        float * m_nxt = mB.ptr;
        CUDA_CHECK(cudaMemsetAsync(O_cur, 0, DV*N*nz*sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(l_state.ptr, 0, N*nz*sizeof(float), stream));
        fattn_gemm_fill<<<(N*nz + 255)/256, 256, 0, stream>>>(m_cur, -INFINITY, N*nz);
        CUDA_CHECK(cudaGetLastError());

        for (int64_t c = 0; c < Lsp; c += C) {
            const fa_fold_split g = {(int) nsplit, (int) Lsp, (int) c, (int) nkv};
            // the longest stream this chunk (split 0 is always full-length)
            const int nkv_c = (int) std::min(C, Lsp - c);
            const int ntile = (nkv_c + TK - 1)/TK;
            {
                const int nb  = (int) (D/QK4_0);
                const int kpb = 256/nb;   // key pairs per block
                const int npairs = ntile*TK/2;
                fa_fold_dequant_launch(K->type, V->type, dim3((npairs + kpb - 1)/kpb, nz, 1), kpb*nb, stream,
                    (const char *) K->data + s*K->nb[3], (const char *) V->data + s*V->nb[3],
                    K16.ptr, Vp.ptr, K->nb[1], K->nb[2], V->nb[1], V->nb[2], (int) D, g, (int) C);
                CUDA_CHECK(cudaGetLastError());
            }
            fa_fold_qk2<<<dim3(ntile, (N + BN - 1)/BN, nz), 256, 0, stream>>>(
                K16.ptr, Qf16.ptr, mask_d, mask_first.ptr, s_mask, P.ptr, mt.ptr, lt.ptr,
                (int) D, (int) N, (int) nt, (int) C, g);
            CUDA_CHECK(cudaGetLastError());
            fa_fold_pv2<<<dim3(DV/BM, (N + BN - 1)/BN, nz), 256, 0, stream>>>(
                Vp.ptr, P.ptr, mt.ptr, lt.ptr, m_cur, m_nxt, l_state.ptr, O_cur, O_nxt,
                (int) DV, (int) N, (int) C, ntile, mask_first.ptr, (int) nt, g);
            CUDA_CHECK(cudaGetLastError());
            std::swap(O_cur, O_nxt);
            std::swap(m_cur, m_nxt);
        }

        fa_fold_finalize<<<dim3(nt, gqa, nhkv), 256, 0, stream>>>(
            O_cur, m_cur, l_state.ptr, (float *) dst->data + s*(dst->nb[3]/sizeof(float)),
            (int) DV, (int) nt, (int) N, (int) gqa, (int) nsplit, dst_s1, dst_s2);
        CUDA_CHECK(cudaGetLastError());
    }
}
#else
static bool ggml_cuda_fa_fold_usable(const ggml_tensor *) { return false; }
static void ggml_cuda_flash_attn_ext_fold(ggml_backend_cuda_context &, ggml_tensor *) { GGML_ABORT("fatal error"); }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

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
    if (ggml_cuda_fa_fold_usable(dst)) {
        ggml_cuda_flash_attn_ext_fold(ctx, dst);
        return;
    }

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
    // the handle is already bound to this stream, cublasSetStream would also reset its workspace
    cublasHandle_t cublas = ctx.cublas_handle();

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
                    for (const auto algo : {CUBLAS_GEMM_ALGO4, CUBLAS_GEMM_DEFAULT}) {
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
