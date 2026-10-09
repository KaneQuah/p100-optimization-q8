#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// Flash attention for pre-Volta GPUs over a q8_0 K/V cache at head size 128 or 256, for the
// few-token batches of decode and speculative verify (one to five tokens, a GQA group of 1 to 8
// heads shared by one block).
// The q8_0 counterpart of fattn-q4p.cuh, built the same way and for the same reason: the tile and
// vec kernels are issue-bound on these GPUs, not memory-bound, and spend their instructions on
// fp16 conversions and shared-memory round trips.
//
//   * Dequantization is 2 instructions per value, straight into fp32: the int8 is biased to
//     unsigned (xor 0x80), PRMT places it into the low mantissa bits of 2^23 (0x4B0000nn), and one
//     FADD of -(2^23 + 128) leaves the exact value. Every K/V byte is read once.
//   * Every dequantized value is reused by the query rows of the block: the whole GQA group of
//     every token, so a 5-token verify amortises one dequant over up to 30 FMAs.
//   * Q stays fp32 (the tile and vec kernels round it to fp16), and products and sums are fp32. An
//     int8 times an fp16 block scale has 19 significant bits, so folding the scale into a
//     dequantized value is exact in fp32.
//
// q8_0 blocks are 34 bytes (fp16 scale, 32 int8), so they alternate between word-aligned and two
// bytes past it. Rows are read as aligned words and the odd/even alignment is resolved with PRMT.
//
// At head size 128 with up to 8 rows, K goes through shared memory: the next chunk's rows are
// requested once this chunk's QK pass is done (a warp on consecutive words of one row), held in
// registers through its PV pass, then stored for the next QK pass to read. Reading K straight from
// the cache puts a 32-row gather behind every load instruction, which is slow when the rows are a
// KV cache's token-major rows, nh*136 bytes apart.
//
// Softmax uses a lazily updated running max: the max only moves when some score exceeds it by more
// than 2^8, so P can reach 256, which fp32 accumulation absorbs without loss.

// R = query rows per block (tokens x GQA heads). One thread block of 256 threads works through the
// KV cache in chunks of C positions.
//   QK: NSPLIT adjacent threads share one KV position, each over DPS dimensions; PT positions each.
//   PV: a thread owns DPT output dimensions for all R rows, over every NPG-th position.
template <int D, int R>
struct fattn_q8p_cfg {
    static constexpr int NT     = 256;
    static constexpr int RQP    = (R + 3) & ~3;                // rows padded for float4 access
    static constexpr int NSPLIT = (D == 128 || R <= 18) ? 2 : 4;
    static constexpr int PT     = R <= 24 ? 2 : 1;
    static constexpr int NQP    = NT/NSPLIT;                   // position slots per pass
    static constexpr int C      = NQP*PT;                      // KV positions per chunk
    static constexpr int BPS    = D/32/NSPLIT;                 // q8_0 blocks per split (even)
    static constexpr int DPS    = D/NSPLIT;                    // dimensions per split
    static constexpr int QREG   = DPS*RQP + 4;                 // one split's region of Q_s; +4 spreads banks
    static constexpr bool BLOCK_T = R <= 15;                   // scale per 32-dim partial sum, not per value
    static constexpr int DPT    = (R <= 12 && D >= 256) ? 8 : 4;
    static constexpr int NDG    = D/DPT;
    static constexpr int NPG    = NT/NDG;
    static constexpr bool FOLD  = R <= DPT;                    // V block scale into P (R multiplies) or V (DPT)
    static constexpr int LRP    = R <= 16 ? 16 : 32;           // rows (padded) of the parallel denominator sum
    static constexpr int LPPS   = C / (NT/LRP);                // positions per thread in it
    // K rows of a chunk staged through shared memory: a coalesced load into registers issued before
    // the PV pass, stored once it is done. Only where the rows fit next to everything else.
    static constexpr bool KST   = D == 128 && R <= 8;
    static constexpr int KRW    = D/32*34/4;                   // words per K row
    static constexpr int KNW    = KST ? C*KRW/NT : 1;          // words per thread of a chunk

    static_assert(NSPLIT*QREG >= R*D, "Q_s is reused for the output reduction");
    static_assert(!KST || C*KRW % NT == 0, "bad K staging split");
    static_assert(NPG <= 8 && C % NPG == 0, "bad DPT");
    static_assert(C % (NT/LRP) == 0, "bad denominator split");
    static_assert(BPS % 2 == 0, "q8_0 block alignment pattern needs an even block count per split");
};

// exact value of byte e of x, whose bytes hold the int8 values xor 0x80
static __device__ __forceinline__ float fattn_q8p_val(const uint32_t x, const int e) {
    return __int_as_float(__byte_perm(x, 0x4B000000u, 0x7440u | e)) - 8388736.0f; // 2^23 + 128
}

static __device__ __forceinline__ uint32_t fattn_q8p_ld16(const char * p) {
    return __ldg((const unsigned short *) p);
}

template <int D, int ncols1, int ncols2>
__launch_bounds__(256, 1)
static __global__ void flash_attn_ext_q8p(
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
    using cfg = fattn_q8p_cfg<D, R>;
    constexpr int NT = cfg::NT, RQP = cfg::RQP, C = cfg::C, PT = cfg::PT;
    constexpr int NSPLIT = cfg::NSPLIT, NQP = cfg::NQP, BPS = cfg::BPS, DPS = cfg::DPS, QREG = cfg::QREG;
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

    __shared__ __align__(16) float Q_s[NSPLIT*QREG];     // Q, then the output reduction
    __shared__ __align__(16) float P_s[C*RQP];
    __shared__ float m_s[R];                              // running max, log2 units
    __shared__ float a_s[R];                              // rescale factor
    __shared__ float red_s[NT/WARP_SIZE][R];
    __shared__ float lsum_s[NT/WARP_SIZE][LRP];
    __shared__ float l_s[R];
    __shared__ uint32_t K_s[cfg::KST ? C*cfg::KRW : 1];   // this chunk's K rows, row-major in words

    ggml_cuda_pdl_sync();

    // Q, pre-scaled into log2 units so the softmax is a bare exp2. Layout: region per split, then
    // [dim][row] with rows padded to RQP.
    const float qscale = scale*LOG2E;
    for (int idx = tid; idx < RQP*D; idx += NT) {
        const int rl = idx / D, dim = idx % D;
        const int t = rl / ncols2, g = rl % ncols2;
        float q = 0.0f;
        if (rl < R && ic0 + t < n_tok) {
            q = ((const float *) (Q + nb03*sequence + int64_t(nb02)*(head0 + g) + int64_t(nb01)*(ic0 + t)))[dim]*qscale;
        }
        Q_s[(dim / DPS)*QREG + (dim % DPS)*RQP + rl] = q;
    }
    if (tid < R) {
        m_s[tid] = -1e30f;
    }

    // QK role: split (fastest), then position slot
    const int qsplit = tid % NSPLIT;
    const int qslot  = tid / NSPLIT;
    const float * Qt = Q_s + qsplit*QREG;

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

    // this thread's V bytes: DPT consecutive dims inside one block; the 8 or 4 bytes start 2 bytes
    // past a word in even blocks and on a word in odd blocks, so read the aligned words around them
    const int      pv_d0  = DPT*dg;
    const int      pv_b   = pv_d0/32;
    const int      pv_o   = 34*pv_b + 2 + pv_d0 % 32;
    const int      pv_oa  = pv_o & ~3;
    const uint32_t pv_sel = (pv_o & 2) ? 0x5432u : 0x3210u;
    constexpr int  NWV    = DPT == 8 ? 3 : 2;

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

    // K rows of the chunk at kc, a warp reading consecutive words of a row, into registers
    uint32_t kreg[cfg::KNW];
    auto load_k = [&](const int kc) {
        if constexpr (cfg::KST) {
#pragma unroll
            for (int i = 0; i < cfg::KNW; ++i) {
                const int e = tid + NT*i, row = e / cfg::KRW;
                kreg[i] = __ldg((const uint32_t *) (Kb + int64_t(min(kc + row, ne11 - 1))*nb11) + (e - row*cfg::KRW));
            }
        }
    };
    auto store_k = [&]() {
        if constexpr (cfg::KST) {
#pragma unroll
            for (int i = 0; i < cfg::KNW; ++i) {
                K_s[tid + NT*i] = kreg[i];
            }
        }
    };
    if constexpr (cfg::KST) {
        load_k(blockIdx.y*C);
        store_k();
    }

    __syncthreads();

    for (int k0 = blockIdx.y*C; k0 < ne11; k0 += gridDim.y*C) {
        const int p_end = min(C, ne11 - k0);
        const int kn    = k0 + gridDim.y*C;               // the next chunk
        load_mask(k0);

        // ---- S = Q K^T for this thread's PT positions and R rows, over its split of the dims ----
        const char * krow[PT];
        bool kvalid[PT];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = k0 + qslot + NQP*pt;
            kvalid[pt] = pos < ne11;
            krow[pt] = Kb + int64_t(kvalid[pt] ? pos : 0)*nb11 + qsplit*(34*BPS);
        }

        float S[PT][R];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                S[pt][r] = 0.0f;
            }
        }

#pragma unroll
        for (int bl = 0; bl < BPS; ++bl) {
            // block bl starts at byte 34*bl: word-aligned when bl is even, 2 bytes past when odd
            const int o = 34*bl;
            float    d[PT];
            uint32_t w[PT][8];
#pragma unroll
            for (int pt = 0; pt < PT; ++pt) {
                uint32_t v[9];
                if constexpr (cfg::KST) {
                    const uint32_t * bp = K_s + (qslot + NQP*pt)*cfg::KRW + qsplit*(34*BPS/4) + (o >> 2);
#pragma unroll
                    for (int j = 0; j < 9; ++j) {
                        v[j] = bp[j];
                    }
                } else {
                    const uint32_t * bp = (const uint32_t *) (krow[pt] + (o & ~3));
#pragma unroll
                    for (int j = 0; j < 9; ++j) {
                        v[j] = __ldg(bp + j);
                    }
                }
                if ((bl & 1) == 0) {
                    d[pt] = __half2float(__ushort_as_half((unsigned short) (v[0] & 0xFFFF)));
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        w[pt][k] = __byte_perm(v[k], v[k + 1], 0x5432);
                    }
                } else {
                    d[pt] = __half2float(__ushort_as_half((unsigned short) (v[0] >> 16)));
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        w[pt][k] = v[k + 1];
                    }
                }
            }

            float t[cfg::BLOCK_T ? PT : 1][cfg::BLOCK_T ? R : 1];
            if constexpr (cfg::BLOCK_T) {
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        t[pt][r] = 0.0f;
                    }
                }
            }

#pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t x[PT];
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
                    x[pt] = w[pt][k] ^ 0x80808080u;
                }
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float y[PT];
#pragma unroll
                    for (int pt = 0; pt < PT; ++pt) {
                        y[pt] = fattn_q8p_val(x[pt], e);
                        if constexpr (!cfg::BLOCK_T) {
                            y[pt] *= d[pt]; // exact
                        }
                    }
                    const float4 * q4 = (const float4 *) (Qt + (32*bl + 4*k + e)*RQP);
#pragma unroll
                    for (int rq = 0; rq < RQP/4; ++rq) {
                        const float4 q = q4[rq];
                        const float qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            if (4*rq + c < R) {
#pragma unroll
                                for (int pt = 0; pt < PT; ++pt) {
                                    if constexpr (cfg::BLOCK_T) {
                                        t[pt][4*rq + c] = fmaf(qv[c], y[pt], t[pt][4*rq + c]);
                                    } else {
                                        S[pt][4*rq + c] = fmaf(qv[c], y[pt], S[pt][4*rq + c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (cfg::BLOCK_T) {
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        S[pt][r] = fmaf(d[pt], t[pt][r], S[pt][r]);
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
            grow |= lmax[rl] > m_s[rl] + 8.0f;
        }
        const bool any_grow = __syncthreads_or(grow);
        // every thread is past its K reads, so the next chunk's rows can be requested now and
        // land while the PV pass runs
        if (kn < ne11) {
            load_k(kn);
        }
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

        // P for this chunk, published by the first thread of each position; with as many splits as
        // positions per thread, split j publishes position j (the reduction left S in both)
        constexpr bool PSPL = NSPLIT == PT;
        if (PSPL || qsplit == 0) {
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
                    ((float4 *) Pp)[rq] = make_float4(pv[0], pv[1], pv[2], pv[3]);
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
                        ls += P_s[p*RQP + lr];
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
        // V words are loaded one position ahead: at one block per SM there are too few warps to
        // hide an L2 round trip behind the FMAs of the others
        const char * vrow0 = Vb + int64_t(k0 + min(pg, p_end - 1))*nb21;
        uint32_t nd = fattn_q8p_ld16(vrow0 + 34*pv_b);
        uint32_t nw[NWV];
#pragma unroll
        for (int j = 0; j < NWV; ++j) {
            nw[j] = __ldg((const uint32_t *) (vrow0 + pv_oa) + j);
        }
#pragma unroll 2
        for (int p = pg; p < p_end; p += NPG) {
            const float dv = __half2float(__ushort_as_half((unsigned short) nd));
            uint32_t wv[DPT/4];
#pragma unroll
            for (int j = 0; j < DPT/4; ++j) {
                wv[j] = __byte_perm(nw[j], nw[j + 1], pv_sel) ^ 0x80808080u;
            }
            {
                const char * vrow = Vb + int64_t(k0 + min(p + NPG, p_end - 1))*nb21;
                nd = fattn_q8p_ld16(vrow + 34*pv_b);
#pragma unroll
                for (int j = 0; j < NWV; ++j) {
                    nw[j] = __ldg((const uint32_t *) (vrow + pv_oa) + j);
                }
            }

            float y[DPT];
#pragma unroll
            for (int j = 0; j < DPT; ++j) {
                y[j] = fattn_q8p_val(wv[j/4], j % 4);
            }

            float P[RQP];
#pragma unroll
            for (int rq = 0; rq < RQP/4; ++rq) {
                const float4 pq = ((const float4 *) (P_s + p*RQP))[rq];
                P[4*rq + 0] = pq.x; P[4*rq + 1] = pq.y; P[4*rq + 2] = pq.z; P[4*rq + 3] = pq.w;
            }
            if constexpr (cfg::FOLD) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float pd = P[r]*dv;
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        acc[r][j] = fmaf(pd, y[j], acc[r][j]);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < DPT; ++j) {
                    y[j] *= dv; // exact
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        acc[r][j] = fmaf(P[r], y[j], acc[r][j]);
                    }
                }
            }
        }
        if constexpr (cfg::KST) {
            if (kn < ne11) {
                store_k();
                __syncthreads();
            }
        }
    }
    __syncthreads();
    add_lsum();
    if (tid < R) {
        l_s[tid] = l_row;
    }

    // ---- reduce the NPG partial outputs ----
    float * O_s = Q_s;
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
// where they fit in 24 rows, half of one beyond that, so a block always holds 24 rows or fewer
// (30 rows would need more registers than a thread has).
static int fattn_q8p_ncols2(const int64_t g, const int64_t nb) {
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
    if (nb*nc2 > 24 && nc2 % 2 == 0) {
        nc2 /= 2;
    }
    return nb*nc2 > 24 ? 0 : nc2;
}

// Whether flash_attn_ext_q8p handles this op. Kept in one place because the alloc-size path must
// agree with the dispatch: it reserves no f16 staging when this kernel runs.
static bool ggml_cuda_fattn_q8p_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FA_Q8P");
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
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0 || Q->type != GGML_TYPE_F32 || sinks) {
        return false;
    }
    if ((Q->ne[0] != 128 && Q->ne[0] != 256) || K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > 5 || K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 ||
            fattn_q8p_ncols2(Q->ne[2]/K->ne[2], Q->ne[1]) == 0) {
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
    // 32-bit loads of the aligned words need 4-aligned rows
    if (K->nb[1] % 4 != 0 || V->nb[1] % 4 != 0 || (uintptr_t) K->data % 4 != 0 || (uintptr_t) V->data % 4 != 0) {
        return false;
    }
    return true;
}

template <int D, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_q8p_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    launch_fattn<D, ncols1, ncols2>(ctx, dst, flash_attn_ext_q8p<D, ncols1, ncols2>, 256/WARP_SIZE, 0,
        fattn_q8p_cfg<D, ncols1*ncols2>::C, false, false, false, false);
}

template <int D, int ncols2>
static void ggml_cuda_flash_attn_ext_q8p_nc2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[1]) {
        case 1: ggml_cuda_flash_attn_ext_q8p_case<D, 1, ncols2>(ctx, dst); break;
        case 2: ggml_cuda_flash_attn_ext_q8p_case<D, 2, ncols2>(ctx, dst); break;
        case 3: ggml_cuda_flash_attn_ext_q8p_case<D, 3, ncols2>(ctx, dst); break;
        case 4: if constexpr (ncols2 <= 6) { ggml_cuda_flash_attn_ext_q8p_case<D, 4, ncols2>(ctx, dst); } break;
        case 5: if constexpr (ncols2 <= 4) { ggml_cuda_flash_attn_ext_q8p_case<D, 5, ncols2>(ctx, dst); } break;
        default: GGML_ABORT("fatal error");
    }
}

template <int D>
static void ggml_cuda_flash_attn_ext_q8p_d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (fattn_q8p_ncols2(Q->ne[2]/dst->src[1]->ne[2], Q->ne[1])) {
        case 1: ggml_cuda_flash_attn_ext_q8p_nc2<D, 1>(ctx, dst); break;
        case 2: ggml_cuda_flash_attn_ext_q8p_nc2<D, 2>(ctx, dst); break;
        case 3: ggml_cuda_flash_attn_ext_q8p_nc2<D, 3>(ctx, dst); break;
        case 4: ggml_cuda_flash_attn_ext_q8p_nc2<D, 4>(ctx, dst); break;
        case 6: ggml_cuda_flash_attn_ext_q8p_nc2<D, 6>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_q8p(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[0]->ne[0] == 128) {
        ggml_cuda_flash_attn_ext_q8p_d<128>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_q8p_d<256>(ctx, dst);
    }
}
