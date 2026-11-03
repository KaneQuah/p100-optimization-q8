#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/unary.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

// l2 norm of one 128-wide head in the S_v = 128 layout (a lane holds the columns r * 8 + lane_in_col), bit-exact with
// rms_norm_f32<256> + scale of build_gdn_l2_norm: there a warp sums 32 columns with xor 16, 8, 4, 2, 1 (here r ^ 2, r ^ 1
// and the lanes) and the block adds the 4 warp sums as (w0 + w2) + (w1 + w3).
template <int rows_per_lane, int lanes_per_col>
static __device__ __forceinline__ void gdn_l2_norm_128(float * x, const float eps, const float scale_out, const int ncols) {
    static_assert(rows_per_lane == 16 && lanes_per_col == 8, "S_v = 128 with 32-wide warps only");
    float w[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float a0 = __fadd_rn(__fmul_rn(x[4*j + 0], x[4*j + 0]), __fmul_rn(x[4*j + 2], x[4*j + 2]));
        const float a1 = __fadd_rn(__fmul_rn(x[4*j + 1], x[4*j + 1]), __fmul_rn(x[4*j + 3], x[4*j + 3]));
        w[j] = warp_reduce_sum<lanes_per_col>(__fadd_rn(a0, a1));
    }
    const float sum   = __fadd_rn(__fadd_rn(w[0], w[2]), __fadd_rn(w[1], w[3]));
    const float mean  = sum / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        x[r] = scale_out * (scale * x[r]);
    }
}

// fuse_in: q and k come before their l2 norm, g is alpha with g = softplus(alpha + dt) * A, beta is taken before its sigmoid
template <int S_v, bool KDA, bool keep_rs_t, bool fuse_in = false, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int *   state_ids,
                                     int64_t       state_row_stride,
                                     const float * dt,
                                     const float * A,
                                     float         l2_eps,
                                     float         l2_scale,
                                     int           l2_ncols) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    // with state_ids the input state of a sequence is row state_ids[sequence] of the cache
    const int64_t state_in_offset = (state_ids ? state_ids[sequence] * state_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = fuse_in ? ggml_cuda_op_sigmoid_single(*beta_t) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (fuse_in) {
            if constexpr (rows_per_lane == 16 && lanes_per_col == 8) {
                gdn_l2_norm_128<rows_per_lane, lanes_per_col>(k_reg, l2_eps, l2_scale, l2_ncols);
                gdn_l2_norm_128<rows_per_lane, lanes_per_col>(q_reg, l2_eps, l2_scale, l2_ncols);
            } else {
                NO_DEVICE_CODE;
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(fuse_in ? ggml_cuda_op_softplus_single(*g_t + dt[h_idx]) * A[h_idx] : *g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Prompt-processing variant (non-KDA, S_v = 128, not fused) of the kernel above, ported from Mrinohk/qwen36-p100-patches
// patch 0027 (MIT). Same 4 columns per warp and 8 lanes x 16 rows per column, but rows are contiguous per lane and the
// per-token q / k / v / g / beta are staged in shared memory in chunks of CH tokens with coalesced loads, the next
// chunk's loads in flight while the current one is computed (the kernel above reads them from global memory per token).
// fp32 throughout; the row sums are taken in a different order than in the kernel above.
template <bool keep_rs_t>
__global__ void __launch_bounds__(128, 4)
gated_delta_net_chunk_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic, float scale, int64_t state_slot_stride, int K,
        const int * state_ids, int64_t state_row_stride) {
    constexpr int S_v = 128, CH = 8, NC = 16, R = 16; // NC columns per block, R rows per lane
    __shared__ __align__(16) float sk[2][CH][S_v];
    __shared__ __align__(16) float sq[2][CH][S_v];
    __shared__ float svv[2][CH][NC], sg[2][CH], sbt[2][CH];
    const uint32_t h_idx = blockIdx.x, sequence = blockIdx.y;
    const int lane = threadIdx.x, wy = threadIdx.y, tid = wy * 32 + lane;
    const int sub = lane >> 3, r0 = (lane & 7) * R;      // column within the warp, first row of this lane
    const int cb = wy * 4 + sub;                         // column within the block
    const int col0 = blockIdx.z * NC, col = col0 + cb;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += (state_ids ? (int64_t) state_ids[sequence] * state_row_stride + h_idx * S_v * S_v : state_in_offset) + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s_shard[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        s_shard[r] = curr_state[r0 + r];
    }

    float lk[8], lq[8], lv = 0.0f, lgb = 0.0f;
    static_assert(CH * NC == 128, "one v value per thread");
    auto load = [&](int t0) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int idx = tid + 128 * j, tt = idx / S_v, i = idx % S_v, t = t0 + tt;
            const int64_t o = iq3 * sq3 + (int64_t) t * sq2 + iq1 * sq1 + i;
            lk[j] = t < n_tokens ? k[o] : 0.0f;
            lq[j] = t < n_tokens ? q[o] : 0.0f;
        }
        {
            const int t = t0 + tid / NC;
            lv = t < n_tokens ? v[sequence * sv3 + (int64_t) t * sv2 + h_idx * sv1 + col0 + tid % NC] : 0.0f;
        }
        if (tid < 2 * CH) {
            const int t = t0 + tid % CH;
            const int64_t gb_offset = sequence * sb3 + (int64_t) t * sb2 + h_idx * sb1;
            lgb = 0.0f;
            if (t < n_tokens) {
                lgb = tid < CH ? expf(g[gb_offset]) : beta[gb_offset];
            }
        }
    };
    auto store = [&](int b) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int idx = tid + 128 * j;
            sk[b][idx / S_v][idx % S_v] = lk[j];
            sq[b][idx / S_v][idx % S_v] = lq[j];
        }
        svv[b][tid / NC][tid % NC] = lv;
        if (tid < CH) {
            sg[b][tid] = lgb;
        } else if (tid < 2 * CH) {
            sbt[b][tid - CH] = lgb;
        }
    };
    auto red8 = [](float x) {
        x += __shfl_xor_sync(0xffffffff, x, 4);
        x += __shfl_xor_sync(0xffffffff, x, 2);
        x += __shfl_xor_sync(0xffffffff, x, 1);
        return x;
    };

    ggml_cuda_pdl_sync();
    load(0);
    store(0);
    __syncthreads();
    for (int t0 = 0, b = 0; t0 < n_tokens; t0 += CH, b ^= 1) {
        if (t0 + CH < n_tokens) {
            load(t0 + CH);
        }
        const int nt = n_tokens - t0 < CH ? (int) (n_tokens - t0) : CH;
        for (int tt = 0; tt < nt; tt++) {
            const int t = t0 + tt;
            float k_reg[R], q_reg[R];
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                const float4 kk = *(const float4 *) &sk[b][tt][r0 + r];
                const float4 qq = *(const float4 *) &sq[b][tt][r0 + r];
                k_reg[r] = kk.x; k_reg[r + 1] = kk.y; k_reg[r + 2] = kk.z; k_reg[r + 3] = kk.w;
                q_reg[r] = qq.x; q_reg[r + 1] = qq.y; q_reg[r + 2] = qq.z; q_reg[r + 3] = qq.w;
            }
            const float g_val = sg[b][tt], beta_val = sbt[b][tt];
            float kv0 = 0.0f, kv1 = 0.0f, kv2 = 0.0f, kv3 = 0.0f;
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                kv0 += s_shard[r] * k_reg[r]; kv1 += s_shard[r + 1] * k_reg[r + 1];
                kv2 += s_shard[r + 2] * k_reg[r + 2]; kv3 += s_shard[r + 3] * k_reg[r + 3];
            }
            const float kv_col = red8((kv0 + kv1) + (kv2 + kv3));
            const float delta_col = (svv[b][tt][cb] - g_val * kv_col) * beta_val;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int r = 0; r < R; r += 4) {
                s_shard[r]     = g_val * s_shard[r]     + k_reg[r]     * delta_col; a0 += s_shard[r]     * q_reg[r];
                s_shard[r + 1] = g_val * s_shard[r + 1] + k_reg[r + 1] * delta_col; a1 += s_shard[r + 1] * q_reg[r + 1];
                s_shard[r + 2] = g_val * s_shard[r + 2] + k_reg[r + 2] * delta_col; a2 += s_shard[r + 2] * q_reg[r + 2];
                s_shard[r + 3] = g_val * s_shard[r + 3] + k_reg[r + 3] * delta_col; a3 += s_shard[r + 3] * q_reg[r + 3];
            }
            const float attn_col = red8((a0 + a1) + (a2 + a3));
            if ((lane & 7) == 0) {
                attn_data[(int64_t) t * S_v * H + col] = attn_col * scale;
            }
            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * cs = state + target_slot * state_slot_stride + col * S_v + r0;
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        cs[r] = s_shard[r];
                    }
                }
            }
        }
        if (t0 + CH < n_tokens) {
            store(b ^ 1);
        }
        __syncthreads();
    }
    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            state[col * S_v + r0 + r] = s_shard[r];
        }
    }
}


// optional in-kernel pre-ops (fuse_in), see ggml_cuda_gated_delta_net_fused_cache
struct gdn_fused_in_args {
    const float * dt       = nullptr;
    const float * A        = nullptr;
    float         l2_eps   = 0.0f;
    float         l2_scale = 0.0f;
    int           l2_ncols = 0;
};

template <bool KDA, bool keep_rs_t, bool fuse_in>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int * state_ids, int64_t state_row_stride,
        const gdn_fused_in_args & fin, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // GGML_CUDA_P60_GDN_CHUNK=1 (opt-in): the staged prompt-processing kernel on Pascal for 32+ tokens. Under -sm layer
    // the chunked kernel (gdn-chunked.cu) takes 64+ tokens first.
    if constexpr (!KDA && !fuse_in) {
        static const bool p60_chunk = getenv("GGML_CUDA_P60_GDN_CHUNK") != nullptr && atoi(getenv("GGML_CUDA_P60_GDN_CHUNK")) != 0;
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (p60_chunk && S_v == 128 && n_tokens >= 32 && warp_size == 32 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA) {
            gated_delta_net_chunk_cuda<keep_rs_t><<<dim3(H, n_seqs, S_v / 16), dim3(32, 4), 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    if constexpr (fuse_in) {
        GGML_ASSERT(S_v == 128 && warp_size == 32);
        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, true>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride,
            fin.dt, fin.A, fin.l2_eps, fin.l2_scale, fin.l2_ncols);
        return;
    }
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride,
                fin.dt, fin.A, fin.l2_eps, fin.l2_scale, fin.l2_ncols);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride,
                fin.dt, fin.A, fin.l2_eps, fin.l2_scale, fin.l2_ncols);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride,
                fin.dt, fin.A, fin.l2_eps, fin.l2_scale, fin.l2_ncols);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride,
                fin.dt, fin.A, fin.l2_eps, fin.l2_scale, fin.l2_ncols);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    // with fuse_in the kernel takes q, k, alpha and beta before their pre-ops (same layout)
    const bool fuse_in = cache != nullptr && cache->q != nullptr;

    const ggml_tensor * src_q     = fuse_in ? cache->q : dst->src[0];
    const ggml_tensor * src_k     = fuse_in ? cache->k : dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) (fuse_in ? cache->alpha : src_g)->data;
    const float * b_d = (const float *) (fuse_in ? cache->b : src_beta)->data;

    gdn_fused_in_args fin;
    if (fuse_in) {
        GGML_ASSERT(!kda && S_v == 128 && neq0 == 128);
        GGML_ASSERT(ggml_nelements(cache->alpha) == ggml_nelements(src_g) && ggml_is_contiguous(cache->alpha));
        GGML_ASSERT(ggml_are_same_shape(cache->b, src_beta) && ggml_is_contiguous(cache->b));
        fin.dt       = cache->dt;
        fin.A        = cache->A;
        fin.l2_eps   = cache->l2_eps;
        fin.l2_scale = cache->l2_scale;
        fin.l2_ncols = (int) neq0;
    }

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // read the input state straight from the cache (see ggml_cuda_try_gdn_state_fusion)
    const int * state_ids        = nullptr;
    int64_t     state_row_stride = 0;
    if (cache != nullptr && cache->state_ids != nullptr) {
        s_d              = cache->state_src;
        state_ids        = cache->state_ids;
        state_row_stride = cache->state_row_stride;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr && cache->data != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // the MMA path reads q, k, g, beta and the state as plain tensors, not the fused pre-ops
    if (!fuse_in && state_ids == nullptr) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (fuse_in) {
        if (keep_rs) {
            launch_gated_delta_net<false, true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        } else {
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        }
    } else if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        } else {
            launch_gated_delta_net<true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        }
    } else {
        if (S_v == 128 && state_ids == nullptr) {
            const ggml_cuda_gdn_chunked_args a = { q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, neqk1, rq3, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                scale, state_slot_stride, K };
            if (ggml_cuda_gdn_chunked(ctx, a)) {
                return;
            }
        }
        if (keep_rs) {
            launch_gated_delta_net<false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        } else {
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, fin, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
