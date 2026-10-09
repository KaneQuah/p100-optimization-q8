#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "active-tokens.cuh"

// Flash attention for pre-Volta GPUs over an f16 K/V cache at head size 128 or 256, for the
// verify widths of speculative decoding (three to five tokens, a GQA group of 1 to 8 heads shared
// by one block). The f16 counterpart of fattn-q8p.cuh, on the same skeleton: lazy softmax max,
// parallel denominator, no chunk-end barrier. The tile kernel is issue-bound here, not memory-bound.
//
//   * K rows are 16-byte aligned, so K is read with 16-byte loads: NSPLIT adjacent lanes share one
//     position and take interleaved 16-byte chunks of its row. The lanes of a load sit on
//     consecutive bytes of a row, not one row per lane.
//   * QK runs on half2 FMAs straight from the loaded K registers, with Q rounded to fp16 as the tile
//     kernel does. A lane sums its dim pairs in one fp16 chain, then widens to fp32 for the sum over
//     the NSPLIT lanes, the mask and the softmax.
//   * P is rounded to fp16 and PV runs on half2 FMAs with V as stored. Each thread folds its fp16
//     chain into fp32 accumulators every PFOLD positions and at the end of each chunk. The lazy max
//     keeps P <= 2^LAZY, so a chain stays far from the fp16 maximum for |V| up to a few hundred.
//   * V of the next VD positions is requested before the current ones are used: at one block per SM
//     there are too few warps to hide an L2 round trip behind the FMAs.
//   * Not faster than the tile and vec kernels at one or two tokens (measured), so those stay there.
//
// FATTN_F16P_QKH and FATTN_F16P_PVH set to 0 give the fp32 arithmetic of the first prototype; it is
// the accuracy reference for the KLD checks, not a tuned path.

// Tuning of the loops below, measured on a P100 at the real (token-major) cache stride.
#ifndef FATTN_F16P_QKH
#define FATTN_F16P_QKH 1
#endif
#ifndef FATTN_F16P_PVH
#define FATTN_F16P_PVH 1
#endif
#define FATTN_F16P_PFOLD 16   // positions per thread whose fp16 PV chain is folded into fp32 together
#define FATTN_F16P_LAZY 4.0f  // the running max moves when a score beats it by more than this (log2 units)

static __device__ __forceinline__ uint32_t fattn_f16p_h2u(const half2 h) {
    return *(const uint32_t *) &h;
}

template <int D, int R>
struct fattn_f16p_cfg {
    static constexpr int NT     = 256;
    static constexpr int RQP    = (R + 3) & ~3;                // rows padded for float4 access
    static constexpr int NSPLIT = D == 128 ? 4 : 8;            // lanes per position
    static constexpr int NI     = D/8/NSPLIT;                  // 16-byte chunks per lane and position
    static constexpr int NQP    = NT/NSPLIT;                   // position slots per pass
    // positions per thread: as many as keep the PT*R score registers within a budget
    static constexpr int PT0    = R*8 <= 80 ? 8 : R*4 <= 80 ? 4 : R*2 <= 80 ? 2 : 1;
    // keep P_s (C*RQP words) within 20 KB
    static constexpr int PT     = NQP*PT0*RQP <= 5120 ? PT0 : NQP*PT0/2*RQP <= 5120 ? PT0/2 : NQP*PT0/4*RQP <= 5120 ? PT0/4 : 1;
    static constexpr int C      = NQP*PT;                      // KV positions per chunk
    static constexpr bool QKH   = FATTN_F16P_QKH;
    static constexpr bool PVH   = FATTN_F16P_PVH;
    static constexpr int PFOLD  = FATTN_F16P_PFOLD;
    static constexpr int FOLDI  = 4;                           // K chunks per fp16 QK chain
    static constexpr float LAZY = FATTN_F16P_LAZY;
    static constexpr int VD     = 4;                           // positions whose V is requested ahead
    // Q_s floats: [dim][row] (fp32) or [dim pair][row] (half2), plus 4 words per 8 dims to spread banks,
    // and at least R*D for the output reduction
    static constexpr int QSZ    = QKH ? (D/2*RQP + D/8*4 > R*D ? D/2*RQP + D/8*4 : R*D) : D*RQP + D/8*4;
    static constexpr int DPT    = (R <= 8 && D >= 256) ? 8 : 4;
    static constexpr int NDG    = D/DPT;
    static constexpr int NPG    = NT/NDG;
    static constexpr int LRP    = R <= 16 ? 16 : 32;           // rows (padded) of the parallel denominator sum
    static constexpr int LPPS   = C / (NT/LRP);                // positions per thread in it

    static_assert(NI >= 1 && NI*NSPLIT*8 == D, "bad K split");
    static_assert(NSPLIT <= WARP_SIZE, "bad K split");
    static_assert(QSZ >= R*D, "Q_s is reused for the output reduction");
    static_assert(NPG <= 8 && C % NPG == 0, "bad DPT");
    static_assert(C % (NT/LRP) == 0, "bad denominator split");
};

template <int D, int ncols1, int ncols2>
__launch_bounds__(256, 1)
static __global__ void flash_attn_ext_f16p(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
                            const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    constexpr int R = ncols1*ncols2;
    using cfg = fattn_f16p_cfg<D, R>;
    constexpr int NT = cfg::NT, RQP = cfg::RQP, C = cfg::C, PT = cfg::PT, NI = cfg::NI;
    constexpr int NSPLIT = cfg::NSPLIT, NQP = cfg::NQP;
    constexpr int DPT = cfg::DPT, NDG = cfg::NDG, NPG = cfg::NPG;
    constexpr int LRP = cfg::LRP, LPPS = cfg::LPPS;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;

    const int tid  = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int ic0      = blockIdx.x*ncols1;
    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0    = blockIdx.z*ncols2 - sequence*ne02;
    const int gqa      = ne02/ne12;
    const int n_tok    = int(ne01.z);

    const char * Kb = K + nb13*sequence + int64_t(nb12)*(head0/gqa);
    const char * Vb = V + nb23*sequence + int64_t(nb22)*(head0/gqa);
    const char * Mb = mask ? mask + nb33*(sequence % ne33) + int64_t(nb31)*ic0 : nullptr;

    __shared__ __align__(16) float Q_s[cfg::QSZ];        // Q, then the output reduction
    __shared__ __align__(16) float P_s[C*RQP];
    __shared__ float m_s[R];                              // running max, log2 units
    __shared__ float a_s[R];                              // rescale factor
    __shared__ float red_s[NT/WARP_SIZE][R];
    __shared__ float lsum_s[NT/WARP_SIZE][LRP];
    __shared__ float l_s[R];

    ggml_cuda_pdl_sync();

    // Q, pre-scaled into log2 units so the softmax is a bare exp2. Layout [dim][row], rows padded
    // to RQP, with 4 floats of padding per 8 dims.
    const float qscale = scale*LOG2E;
    for (int idx = tid; idx < RQP*D; idx += NT) {
        const int rl = idx / D, dim = idx % D;
        const int t = rl / ncols2, g = rl % ncols2;
        float q = 0.0f;
        if (rl < R && ic0 + t < n_tok) {
            q = ((const float *) (Q + nb03*sequence + int64_t(nb02)*(head0 + g) + int64_t(nb01)*(ic0 + t)))[dim]*qscale;
        }
        if constexpr (cfg::QKH) {
            // pairs of dims as half2, written by the even dim's thread
            const float q1 = __shfl_down_sync(0xFFFFFFFF, q, 1, WARP_SIZE);
            if ((dim & 1) == 0) {
                ((half2 *) Q_s)[(dim/2)*RQP + (dim/8)*4 + rl] = __floats2half2_rn(q, q1);
            }
        } else {
            Q_s[dim*RQP + (dim/8)*4 + rl] = q;
        }
    }
    if (tid < R) {
        m_s[tid] = -1e30f;
    }

    // QK role: split (fastest), then position slot
    const int qsplit = tid % NSPLIT;
    const int qslot  = tid / NSPLIT;

    // PV role: dim group (fastest), then position group
    const int dg = tid % NDG;
    const int pg = tid / NDG;

    float acc[R][DPT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    // Softmax denominator of row tid, for tid < R: summed once per chunk from P_s, so the PV loop
    // neither spends registers on it nor diverges to add it.
    float l_row = 0.0f;
    auto add_lsum = [&]() {
        if (tid < R) {
            float ls = 0.0f;
#pragma unroll
            for (int w = 0; w < NT/WARP_SIZE; ++w) {
                ls += lsum_s[w][tid];
            }
            l_row += ls;
        }
    };

    // The mask halves of a chunk's positions, one per token. Loaded ahead of the QK math.
    uint32_t mraw[PT][ncols1];
    auto load_mask = [&](const int kc) {
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = kc + qslot + NQP*pt;
#pragma unroll
            for (int t = 0; t < ncols1; ++t) {
                mraw[pt][t] = Mb && pos < ne11 && ic0 + t < n_tok ?
                    __ldg((const unsigned short *) (Mb + int64_t(nb31)*t) + pos) : 0;
            }
        }
    };

    __syncthreads();

    for (int k0 = blockIdx.y*C; k0 < ne11; k0 += gridDim.y*C) {
        const int p_end = min(C, ne11 - k0);
        load_mask(k0);

        // ---- S = Q K^T for this thread's PT positions and R rows, over its chunks of the dims ----
        const uint4 * krow[PT];
        bool kvalid[PT];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = k0 + qslot + NQP*pt;
            kvalid[pt] = pos < ne11;
            krow[pt] = (const uint4 *) (Kb + int64_t(kvalid[pt] ? pos : 0)*nb11) + qsplit;
        }

        float S[PT][R];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                S[pt][r] = 0.0f;
            }
        }

        if constexpr (cfg::QKH) {
            half2 hs[PT][R];
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int c = i*NSPLIT + qsplit;                  // this lane's 16-byte chunk: dims 8c..8c+7
                if (i % cfg::FOLDI == 0) {
#pragma unroll
                    for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            hs[pt][r] = __float2half2_rn(0.0f);
                        }
                    }
                }
                half2 kh[PT][4];
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
                    const uint4 kw = __ldg(krow[pt] + i*NSPLIT);
                    const uint32_t kk[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        kh[pt][h] = *(const half2 *) &kk[h];
                    }
                }
                const half2 * Qd = (const half2 *) Q_s + (4*c)*RQP + 4*c;
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const uint4 * q4 = (const uint4 *) (Qd + h*RQP);
#pragma unroll
                    for (int rq = 0; rq < RQP/4; ++rq) {
                        const uint4 q = q4[rq];
                        const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                        for (int cc = 0; cc < 4; ++cc) {
                            if (4*rq + cc < R) {
#pragma unroll
                                for (int pt = 0; pt < PT; ++pt) {
                                    hs[pt][4*rq + cc] = __hfma2(kh[pt][h], *(const half2 *) &qw[cc], hs[pt][4*rq + cc]);
                                }
                            }
                        }
                    }
                }
                if (i % cfg::FOLDI == cfg::FOLDI - 1 || i == NI - 1) {
#pragma unroll
                    for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const float2 f = __half22float2(hs[pt][r]);
                            S[pt][r] += f.x + f.y;
                        }
                    }
                }
            }
        } else {
    #pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int c = i*NSPLIT + qsplit;                  // this lane's 16-byte chunk: dims 8c..8c+7
                float kf[PT][8];
    #pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
                    const uint4 kw = __ldg(krow[pt] + i*NSPLIT);
                    const uint32_t kk[4] = {kw.x, kw.y, kw.z, kw.w};
    #pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float2 f = __half22float2(*(const half2 *) &kk[h]);
                        kf[pt][2*h] = f.x;
                        kf[pt][2*h + 1] = f.y;
                    }
                }
                const float * Qd = Q_s + (8*c)*RQP + 4*c;
    #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float4 * q4 = (const float4 *) (Qd + e*RQP);
    #pragma unroll
                    for (int rq = 0; rq < RQP/4; ++rq) {
                        const float4 q = q4[rq];
                        const float qv[4] = {q.x, q.y, q.z, q.w};
    #pragma unroll
                        for (int cc = 0; cc < 4; ++cc) {
                            if (4*rq + cc < R) {
    #pragma unroll
                                for (int pt = 0; pt < PT; ++pt) {
                                    S[pt][4*rq + cc] = fmaf(qv[cc], kf[pt][e], S[pt][4*rq + cc]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // combine the NSPLIT partial dot products, then mask
        float lmax[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            lmax[r] = -INFINITY;
        }
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int off = 1; off < NSPLIT; off <<= 1) {
                    S[pt][r] += __shfl_xor_sync(0xFFFFFFFF, S[pt][r], off, WARP_SIZE);
                }
            }
#pragma unroll
            for (int rl = 0; rl < R; ++rl) {
                const int t = rl / ncols2;
                float mv = 0.0f;
                if (!kvalid[pt]) {
                    mv = -INFINITY;
                } else {
                    mv = __half2float(__ushort_as_half((unsigned short) mraw[pt][t]))*LOG2E;
                }
                S[pt][rl] += mv;
                lmax[rl] = fmaxf(lmax[rl], S[pt][rl]);
            }
        }

        // lazy max: only rescale when a score beats the running max by more than 2^8
        bool grow = false;
#pragma unroll
        for (int rl = 0; rl < R; ++rl) {
            grow |= lmax[rl] > m_s[rl] + cfg::LAZY;
        }
        const bool any_grow = __syncthreads_or(grow);
        // No barrier at the chunk end: this vote already orders the previous chunk's P_s reads and
        // lsum_s writes before this chunk's publishes. The previous chunk's denominators are summed here.
        if (k0 != blockIdx.y*C) {
            add_lsum();
        }
        if (any_grow) {
            // per warp, reduce over the lanes that share a row (the position-slot bits)
#pragma unroll
            for (int rl = 0; rl < R; ++rl) {
                float v = lmax[rl];
#pragma unroll
                for (int off = WARP_SIZE/2; off >= NSPLIT; off >>= 1) {
                    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE));
                }
                if (lane == 0) {
                    red_s[warp][rl] = v;
                }
            }
            __syncthreads();
            if (tid < R) {
                float mx = m_s[tid];
#pragma unroll
                for (int w = 0; w < NT/WARP_SIZE; ++w) {
                    mx = fmaxf(mx, red_s[w][tid]);
                }
                a_s[tid] = exp2f(m_s[tid] - mx);
                m_s[tid] = mx;
                l_row *= a_s[tid];
            }
            __syncthreads();
#pragma unroll
            for (int rl = 0; rl < R; ++rl) {
                const float a = a_s[rl];
#pragma unroll
                for (int j = 0; j < DPT; ++j) {
                    acc[rl][j] *= a;
                }
            }
        }

        // P for this chunk. After the reduction every split holds the same S; with at least as many
        // splits as positions per thread, split j publishes position j, else split 0 publishes all.
        constexpr bool PSPL = NSPLIT >= PT;
        if (PSPL ? qsplit < PT : qsplit == 0) {
            if constexpr (PSPL) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int pt = 1; pt < PT; ++pt) {
                        if (qsplit == pt) {
                            S[0][r] = S[pt][r];
                        }
                    }
                }
            }
#pragma unroll
            for (int pt = 0; pt < (PSPL ? 1 : PT); ++pt) {
                float * Pp = P_s + (qslot + NQP*(PSPL ? qsplit : pt))*RQP;
#pragma unroll
                for (int rq = 0; rq < RQP/4; ++rq) {
                    float pv[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int rl = 4*rq + c;
                        pv[c] = rl < R ? exp2f(S[pt][rl] - m_s[rl]) : 0.0f;
                    }
                    if constexpr (cfg::PVH) {
                        // P rounded to fp16 and doubled into a half2, ready to multiply a V pair
                        ((uint4 *) Pp)[rq] = make_uint4(
                            fattn_f16p_h2u(__float2half2_rn(pv[0])), fattn_f16p_h2u(__float2half2_rn(pv[1])),
                            fattn_f16p_h2u(__float2half2_rn(pv[2])), fattn_f16p_h2u(__float2half2_rn(pv[3])));
                    } else {
                        ((float4 *) Pp)[rq] = make_float4(pv[0], pv[1], pv[2], pv[3]);
                    }
                }
            }
        }
        __syncthreads();

        // This chunk's softmax denominators: thread tid owns row tid % LRP and positions
        // [seg*LPPS, (seg+1)*LPPS); per-warp partials go to lsum_s, summed after the next vote barrier.
        {
            const int lr = tid % LRP, seg = tid / LRP;
            float ls = 0.0f;
            if (lr < R) {
#pragma unroll
                for (int j = 0; j < LPPS; ++j) {
                    const int p = seg*LPPS + j;
                    if (p < p_end) {
                        if constexpr (cfg::PVH) {
                            ls += __low2float(((const half2 *) P_s)[p*RQP + lr]);
                        } else {
                            ls += P_s[p*RQP + lr];
                        }
                    }
                }
            }
            if constexpr (LRP == 16) {
                ls += __shfl_xor_sync(0xFFFFFFFF, ls, 16, WARP_SIZE);
            }
            if (lane < LRP) {
                lsum_s[warp][lane] = ls;
            }
        }

        // ---- O += P V ----
        // V is requested one position ahead: at one block per SM there are too few warps to hide
        // an L2 round trip behind the FMAs of the others
        constexpr int VW = DPT/2;                              // 32-bit words (half2) per thread and position
        auto load_v = [&](const int p, uint32_t (&w)[VW]) {
            const uint32_t * vp = (const uint32_t *) (Vb + int64_t(k0 + min(p, p_end - 1))*nb21) + VW*dg;
            if constexpr (VW == 2) {
                const uint2 v = __ldg((const uint2 *) vp);
                w[0] = v.x; w[1] = v.y;
            } else {
                const uint4 v = __ldg((const uint4 *) vp);
                w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
            }
        };
        // fp16 chain: half2 products of P and V, folded into the fp32 accumulators every PFOLD positions
        half2 ah[cfg::PVH ? R : 1][cfg::PVH ? VW : 1];
        if constexpr (cfg::PVH) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < VW; ++j) {
                    ah[r][j] = __float2half2_rn(0.0f);
                }
            }
        }
        auto fold = [&]() {
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < VW; ++j) {
                    const float2 f = __half22float2(ah[r][j]);
                    acc[r][2*j] += f.x;
                    acc[r][2*j + 1] += f.y;
                    ah[r][j] = __float2half2_rn(0.0f);
                }
            }
        };
        auto pv_pos = [&](const int p, const uint32_t (&wv)[VW]) {
            if constexpr (cfg::PVH) {
                half2 Ph[RQP];
#pragma unroll
                for (int rq = 0; rq < RQP/4; ++rq) {
                    const uint4 pq = ((const uint4 *) (P_s + p*RQP))[rq];
                    const uint32_t pw[4] = {pq.x, pq.y, pq.z, pq.w};
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        Ph[4*rq + c] = *(const half2 *) &pw[c];
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int j = 0; j < VW; ++j) {
                        ah[r][j] = __hfma2(Ph[r], *(const half2 *) &wv[j], ah[r][j]);
                    }
                }
            } else {
                float y[DPT];
#pragma unroll
                for (int j = 0; j < VW; ++j) {
                    const float2 f = __half22float2(*(const half2 *) &wv[j]);
                    y[2*j] = f.x;
                    y[2*j + 1] = f.y;
                }
                float P[RQP];
#pragma unroll
                for (int rq = 0; rq < RQP/4; ++rq) {
                    const float4 pq = ((const float4 *) (P_s + p*RQP))[rq];
                    P[4*rq + 0] = pq.x; P[4*rq + 1] = pq.y; P[4*rq + 2] = pq.z; P[4*rq + 3] = pq.w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        acc[r][j] = fmaf(P[r], y[j], acc[r][j]);
                    }
                }
            }
        };
        // V words of the next VD positions are requested before the current VD are used
        constexpr int VD = cfg::VD;
        uint32_t nw[VD][VW];
#pragma unroll
        for (int j = 0; j < VD; ++j) {
            load_v(pg + NPG*j, nw[j]);
        }
        int it = 0;
        for (int p0 = pg; p0 < p_end; p0 += NPG*VD) {
            uint32_t cw[VD][VW];
#pragma unroll
            for (int j = 0; j < VD; ++j) {
#pragma unroll
                for (int w = 0; w < VW; ++w) {
                    cw[j][w] = nw[j][w];
                }
                load_v(p0 + NPG*(VD + j), nw[j]);
            }
#pragma unroll
            for (int j = 0; j < VD; ++j) {
                const int p = p0 + NPG*j;
                if (p < p_end) {
                    pv_pos(p, cw[j]);
                    if constexpr (cfg::PVH) {
                        if (++it == cfg::PFOLD) {
                            fold();
                            it = 0;
                        }
                    }
                }
            }
        }
        if constexpr (cfg::PVH) {
            fold();
        }
    }
    __syncthreads();
    add_lsum();
    if (tid < R) {
        l_s[tid] = l_row;
    }

    // ---- reduce the NPG partial outputs ----
    float * O_s = Q_s;
    const int pv_d0 = DPT*dg;
#pragma unroll
    for (int g0 = 0; g0 < NPG; ++g0) {
        if (pg == g0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < DPT; ++j) {
                    float & o = O_s[r*D + pv_d0 + j];
                    o = g0 == 0 ? acc[r][j] : o + acc[r][j];
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < R*D; idx += NT) {
        const int r = idx / D, dim = idx % D;
        const int t = r / ncols2, g = r % ncols2;
        if (ic0 + t >= n_tok) {
            continue;
        }
        float val = O_s[idx];
        if (gridDim.y == 1) {
            val /= l_s[r];
        }
        dst[(((int64_t(sequence)*n_tok + ic0 + t)*ne02 + head0 + g)*gridDim.y + blockIdx.y)*D + dim] = val;
    }
    if (gridDim.y != 1 && tid < R) {
        const int t = tid / ncols2, g = tid % ncols2;
        if (ic0 + t < n_tok) {
            dst_meta[((int64_t(sequence)*n_tok + ic0 + t)*ne02 + head0 + g)*gridDim.y + blockIdx.y] =
                make_float2(m_s[tid]*LN2, l_s[tid]);
        }
    }
    GGML_UNUSED_VARS(sinks, KV_max, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne12, ne13,
        ne31, ne32, nb32);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2,
        logit_softcap, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

// Query heads per block for a GQA group of g heads and nb tokens (0: not handled). Whole groups
// where they fit in 24 rows, half of one beyond that.
static int fattn_f16p_ncols2(const int64_t g, const int64_t nb) {
    int nc2;
    switch (g) {
        case 1: nc2 = 1; break;
        case 2: nc2 = 2; break;
        case 3: nc2 = 3; break;
        case 4: nc2 = 4; break;
        case 6: nc2 = 6; break;
        case 8: nc2 = 4; break;
        default: return 0;
    }
    while (nb*nc2 > 24 && nc2 > 1) {
        nc2 = nc2 % 2 == 0 ? nc2/2 : 1;
    }
    return nb*nc2 > 24 ? 0 : nc2;
}

// Whether flash_attn_ext_f16p handles this op. Kept in one place because the alloc-size path must
// agree with the dispatch.
static bool ggml_cuda_fattn_f16p_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FA_F16P");
        return !s || atoi(s) != 0;
    }();
    if (!enabled) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32 || sinks) {
        return false;
    }
    if ((Q->ne[0] != 128 && Q->ne[0] != 256) || K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0]) {
        return false;
    }
    // nb 1 and 2 are not faster than the tile and vec kernels on every shape (measured), nb 3 to 5 are
    if (Q->ne[1] < 3 || Q->ne[1] > 5 || K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 ||
            fattn_f16p_ncols2(Q->ne[2]/K->ne[2], Q->ne[1]) == 0) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(dst, 4) > 0) {
        return false; // no ALiBi, softcap or sparse KV hint
    }
    if (mask && (mask->ne[2] != 1 || mask->ne[1] < Q->ne[1] || mask->ne[0] < K->ne[1])) {
        return false;
    }
    // 16-byte loads need 16-aligned rows, heads and sequences
    for (const ggml_tensor * t : {K, V}) {
        if (t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0 || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return true;
}

template <int D, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_f16p_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    launch_fattn<D, ncols1, ncols2>(ctx, dst, flash_attn_ext_f16p<D, ncols1, ncols2>, 256/WARP_SIZE, 0,
        fattn_f16p_cfg<D, ncols1*ncols2>::C, false, false, false, false);
}

template <int D, int ncols2>
static void ggml_cuda_flash_attn_ext_f16p_nc2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[1]) {
        case 1: ggml_cuda_flash_attn_ext_f16p_case<D, 1, ncols2>(ctx, dst); break;
        case 2: ggml_cuda_flash_attn_ext_f16p_case<D, 2, ncols2>(ctx, dst); break;
        case 3: ggml_cuda_flash_attn_ext_f16p_case<D, 3, ncols2>(ctx, dst); break;
        case 4: if constexpr (ncols2 <= 6) { ggml_cuda_flash_attn_ext_f16p_case<D, 4, ncols2>(ctx, dst); } break;
        case 5: if constexpr (ncols2 <= 4) { ggml_cuda_flash_attn_ext_f16p_case<D, 5, ncols2>(ctx, dst); } break;
        default: GGML_ABORT("fatal error");
    }
}

template <int D>
static void ggml_cuda_flash_attn_ext_f16p_d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (fattn_f16p_ncols2(Q->ne[2]/dst->src[1]->ne[2], Q->ne[1])) {
        case 1: ggml_cuda_flash_attn_ext_f16p_nc2<D, 1>(ctx, dst); break;
        case 2: ggml_cuda_flash_attn_ext_f16p_nc2<D, 2>(ctx, dst); break;
        case 3: ggml_cuda_flash_attn_ext_f16p_nc2<D, 3>(ctx, dst); break;
        case 4: ggml_cuda_flash_attn_ext_f16p_nc2<D, 4>(ctx, dst); break;
        case 6: ggml_cuda_flash_attn_ext_f16p_nc2<D, 6>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_f16p(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // a fixed-width verify graph may have only its first tokens real (active-tokens.cuh): run the
    // kernel on views of Q and the output that stop there. The mask keeps its rows.
    const int64_t n_act = ggml_cuda_active_cols(dst->src[0]->ne[1]);
    ggml_tensor q_act, dst_act;
    if (n_act < dst->src[0]->ne[1]) {
        q_act          = *dst->src[0];
        q_act.ne[1]    = n_act;
        dst_act        = *dst;
        dst_act.ne[2]  = n_act;
        dst_act.src[0] = &q_act;
        dst = &dst_act;
    }
    if (dst->src[0]->ne[0] == 128) {
        ggml_cuda_flash_attn_ext_f16p_d<128>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_f16p_d<256>(ctx, dst);
    }
}
