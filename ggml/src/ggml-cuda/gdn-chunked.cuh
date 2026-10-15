// Chunked Gated DeltaNet for prefill, scalar gate (non-KDA), S_v = 128, fp32.
//
// Per head and chunk of C = 64 tokens starting from state S0 (S is [i=key][j=value]):
//   G_t   = sum_{u<=t} g_u                      in-chunk log-decay prefix (fp64)
//   A     = strict-lower( beta_t * exp(G_t - G_s) * k_t.k_s )
//   R     = beta * (V - diag(exp(G)) K S0)        (the recurrence's own "v - g*kv" residual)
//   Delta = (I + A)^-1 R                          blocked forward substitution, in-kernel
//   O     = scale * ( diag(exp(G)) Q S0 + P Delta ),  P = lower-incl( exp(G_t - G_s) q_t.k_s )
//   S1    = exp(G_C) S0 + (diag(exp(G_C - G)) K)^T Delta
//
// This is the "forward substitution on the residual" form rather than the WY form
// (Delta = U - W S0 with U = T beta V, W = T beta e^G K). The WY form cancels two large
// T-amplified terms and loses precision with correlated keys; this form only ever subtracts
// what the recurrence subtracts.
//
// Only ratios exp(G_t - G_s) with t >= s are exponentiated, so nothing can overflow. G is
// summed and differenced in fp64 so the ratio is exact to fp32 rounding. All other arithmetic
// is fp32 FMA; every long dot product is accumulated as fresh 16-term partials that are then
// summed (two-level summation), which keeps the error at or below the recurrence's butterfly.
//
// Phase 1 (gdn2_prep): per (chunk, k-head, seq): K K^T and Q K^T once per k-head, K^T/Q^T
//   copies for phase 2, then per value head A, P^T and the gate factors.
// Phase 2 (gdn2_main): per (value head, 64-column half, seq), chunk-serial.
//   The trailing K-1 tokens (keep_rs snapshots) run through the plain recurrence.

#pragma once

#include "common.cuh"

#ifndef GCH
#define GCH 64
#endif

#ifndef TG
#define TG float
#endif
#ifndef TA
#define TA float
#endif
#ifndef TB
#define TB float
#endif
#ifndef TC
#define TC float
#endif
#ifndef TF
#define TF float
#endif
static __device__ __forceinline__ float  xfma(float a, float b, float c) { return fmaf(a, b, c); }
static __device__ __forceinline__ double xfma(float a, float b, double c) { return fma((double) a, (double) b, c); }
static __device__ __forceinline__ double xfma(float a, double b, double c) { return fma((double) a, b, c); }


struct gdn_ch_params {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int H, n_tokens, n_seqs, L, nch, neqk1, rq3, K;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride;
    bool keep;
    // scratch
    float * KT; float * QT; float * Am; float * Pt; float * gx;
};

static __device__ __forceinline__ float4 ldg4(const float * p) { return __ldg((const float4 *) p); }

static __device__ __forceinline__ float f4get(const float4 & a, int i) {
    return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w;
}

// ---------------------------------------------------------------------------------------------
// Phase 1
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) gdn2_prep(const gdn_ch_params p) {
    const int c   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int seq = blockIdx.z;
    const int tid = threadIdx.x;
    const int tx  = tid & 15;
    const int ty  = tid >> 4;
    const int t0  = c * GCH;
    const int len = min(GCH, p.L - t0);
    const int iq3 = seq / p.rq3;

    const float * kb = p.k + iq3 * p.sq3 + kh * p.sq1 + (int64_t) t0 * p.sq2;
    const float * qb = p.q + iq3 * p.sq3 + kh * p.sq1 + (int64_t) t0 * p.sq2;

    __shared__ __align__(16) float Ks[16 * 64];
    __shared__ __align__(16) float Qs[16 * 64];
    __shared__ double Gs[64];
    __shared__ float  bs[64];

    TG kk[4][4], qk[4][4];
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int cc = 0; cc < 4; cc++) { kk[r][cc] = 0.0f; qk[r][cc] = 0.0f; }
    }

    const int64_t khc = (int64_t) (seq * p.neqk1 + kh) * p.nch + c;
    float * KTc = p.KT + khc * 8192;
    float * QTc = p.QT + khc * 8192;

    const int lt = tid >> 2;
    const int li = (tid & 3) * 4;
    float4 kv4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 qv4 = kv4;
    if (lt < len) {
        kv4 = ldg4(kb + (int64_t) lt * p.sq2 + li);
        qv4 = ldg4(qb + (int64_t) lt * p.sq2 + li);
    }
#pragma unroll 1
    for (int i0 = 0; i0 < 128; i0 += 16) {
        __syncthreads();
        Ks[(li + 0) * 64 + lt] = kv4.x; Ks[(li + 1) * 64 + lt] = kv4.y;
        Ks[(li + 2) * 64 + lt] = kv4.z; Ks[(li + 3) * 64 + lt] = kv4.w;
        Qs[(li + 0) * 64 + lt] = qv4.x; Qs[(li + 1) * 64 + lt] = qv4.y;
        Qs[(li + 2) * 64 + lt] = qv4.z; Qs[(li + 3) * 64 + lt] = qv4.w;
        __syncthreads();
        if (i0 + 16 < 128 && lt < len) {
            kv4 = ldg4(kb + (int64_t) lt * p.sq2 + i0 + 16 + li);
            qv4 = ldg4(qb + (int64_t) lt * p.sq2 + i0 + 16 + li);
        }
        // K^T / Q^T slabs out (i-major), coalesced
        *(float4 *) (KTc + (i0 + (tid >> 4)) * 64 + (tid & 15) * 4) = *(const float4 *) (Ks + (tid >> 4) * 64 + (tid & 15) * 4);
        *(float4 *) (QTc + (i0 + (tid >> 4)) * 64 + (tid & 15) * 4) = *(const float4 *) (Qs + (tid >> 4) * 64 + (tid & 15) * 4);
        TG pk[4][4], pq[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++) { pk[r][cc] = 0.0f; pq[r][cc] = 0.0f; }
        }
#pragma unroll
        for (int ii = 0; ii < 16; ii++) {
            const float4 a  = *(const float4 *) (Ks + ii * 64 + 4 * ty);
            const float4 aq = *(const float4 *) (Qs + ii * 64 + 4 * ty);
            const float4 b  = *(const float4 *) (Ks + ii * 64 + 4 * tx);
#pragma unroll
            for (int r = 0; r < 4; r++) {
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    pk[r][cc] = xfma(f4get(a,  r), f4get(b, cc), pk[r][cc]);
                    pq[r][cc] = xfma(f4get(aq, r), f4get(b, cc), pq[r][cc]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++) { kk[r][cc] += pk[r][cc]; qk[r][cc] += pq[r][cc]; }
        }
    }

    const int rep = p.H / p.neqk1;
#pragma unroll 1
    for (int m = 0; m < rep; m++) {
        const int h = kh + m * p.neqk1;
        const int64_t hc = (int64_t) (seq * p.H + h) * p.nch + c;
        __syncthreads();
        if (tid < 64) {
            float gv = 0.0f, bv = 0.0f;
            if (tid < len) {
                const int64_t o = (int64_t) seq * p.sb3 + (int64_t) (t0 + tid) * p.sb2 + (int64_t) h * p.sb1;
                gv = p.g[o];
                bv = p.beta[o];
            }
            bs[tid] = bv;
            Gs[tid] = (double) gv;
        }
        __syncthreads();
        if (tid < 32) {
            const double a = Gs[2 * tid];
            const double b = a + Gs[2 * tid + 1];
            double incl = b;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const double o = __shfl_up_sync(0xffffffff, incl, off, 32);
                if (tid >= off) incl += o;
            }
            const double excl = incl - b;
            Gs[2 * tid]     = excl + a;
            Gs[2 * tid + 1] = incl;
        }
        __syncthreads();
        float * gxc = p.gx + hc * 256;
        if (tid < 64) {
            const double G  = Gs[tid];
            const double GC = Gs[63];
            gxc[tid]       = (float) exp(G);
            gxc[64 + tid]  = (float) exp(GC - G);
            gxc[128 + tid] = bs[tid];
            if (tid == 0) gxc[192] = (float) exp(GC);
        }
        float * Amc = p.Am + hc * 4096;
        float * Ptc = p.Pt + hc * 4096;
        float av[4][4];
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const int s = 4 * tx + cc;
            float pv[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int t = 4 * ty + r;
                const double e = (s <= t) ? exp(Gs[t] - Gs[s]) : 0.0;
                av[r][cc] = (s < t) ? (float) ((double) bs[t] * e * (double) kk[r][cc]) : 0.0f;
                pv[r]     = (float) (e * (double) qk[r][cc]);
            }
            *(float4 *) (Ptc + s * 64 + 4 * ty) = make_float4(pv[0], pv[1], pv[2], pv[3]);
        }
#pragma unroll
        for (int r = 0; r < 4; r++) {
            *(float4 *) (Amc + (4 * ty + r) * 64 + 4 * tx) = make_float4(av[r][0], av[r][1], av[r][2], av[r][3]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 2
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1) gdn2_main(const gdn_ch_params p) {
    const int jg   = blockIdx.x;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int tx   = tid & 15;
    const int ty   = tid >> 4;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int j0   = jg * 64;
    const int kh   = h % p.neqk1;
    const int iq3  = seq / p.rq3;

    __shared__ __align__(16) float sm[12288];
    float * X  = sm;         // S [128][64] during step A; afterwards D [64][64] | A or P^T [64][64]
    float * Y  = sm + 8192;  // 2 x 2048 staging
    float * D  = X;
    float * X2 = X + 4096;

    const float * cs = p.curr_state + (int64_t) (seq * p.H + h) * 16384;
    float s[8][4];
#pragma unroll
    for (int cc = 0; cc < 4; cc++) {
        const int j = j0 + 4 * tx + cc;
        const float4 a = ldg4(cs + j * 128 + 4 * ty);
        const float4 b = ldg4(cs + j * 128 + 64 + 4 * ty);
        s[0][cc] = a.x; s[1][cc] = a.y; s[2][cc] = a.z; s[3][cc] = a.w;
        s[4][cc] = b.x; s[5][cc] = b.y; s[6][cc] = b.z; s[7][cc] = b.w;
    }
    auto store_S = [&]() {
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const int row = (r < 4) ? 4 * ty + r : 64 + 4 * ty + (r - 4);
            *(float4 *) (X + row * 64 + 4 * tx) = make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
        }
    };
    store_S();

    const float * kb = p.k + iq3 * p.sq3 + kh * p.sq1;
    const float * qb = p.q + iq3 * p.sq3 + kh * p.sq1;
    const float * vb = p.v + (int64_t) seq * p.sv3 + (int64_t) h * p.sv1 + j0 + 4 * tx;
    float * dstb = p.dst + ((int64_t) seq * p.n_tokens * p.H + h) * 128 + j0 + 4 * tx;

#pragma unroll 1
    for (int c = 0; c < p.nch; c++) {
        const int t0  = c * GCH;
        const int len = min(GCH, p.L - t0);
        const int64_t hc  = (int64_t) (seq * p.H + h) * p.nch + c;
        const int64_t khc = (int64_t) (seq * p.neqk1 + kh) * p.nch + c;
        const float * KTc = p.KT + khc * 8192;
        const float * QTc = p.QT + khc * 8192;
        const float * Amc = p.Am + hc * 4096;
        const float * Ptc = p.Pt + hc * 4096;
        const float * gxc = p.gx + hc * 256;

        // ---- step A: acc = [K; Q] S0 ----
        float4 la, lb;
        auto load_slab = [&](int i0) {
            la = ldg4(KTc + (i0 + (tid >> 4)) * 64 + (tid & 15) * 4);
            lb = ldg4(QTc + (i0 + (tid >> 4)) * 64 + (tid & 15) * 4);
        };
        auto store_slab = [&](float * buf) {
            *(float4 *) (buf + (tid >> 4) * 128 + (tid & 15) * 4)      = la;
            *(float4 *) (buf + (tid >> 4) * 128 + 64 + (tid & 15) * 4) = lb;
        };
        TA acc[8][4];
#pragma unroll
        for (int r = 0; r < 8; r++) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++) acc[r][cc] = 0.0f;
        }
        load_slab(0);
        float4 vv[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            vv[r] = (4 * ty + r < len) ? ldg4(vb + (int64_t) (t0 + 4 * ty + r) * p.sv2) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        const float4 eG4 = ldg4(gxc + 4 * ty);
        const float4 bt4 = ldg4(gxc + 128 + 4 * ty);
        float4 pre[4];

#pragma unroll 1
        for (int sl = 0; sl < 8; sl++) {
            float * buf = Y + (sl & 1) * 2048;
            store_slab(buf);
            __syncthreads();
            if (sl < 7) {
                load_slab((sl + 1) * 16);
            } else {
#pragma unroll
                for (int q4 = 0; q4 < 4; q4++) pre[q4] = ldg4(Amc + (tid + 256 * q4) * 4);
            }
            const float * Sx = X + sl * 16 * 64 + 4 * tx;
            TA pa[8][4];
#pragma unroll
            for (int r = 0; r < 8; r++) {
#pragma unroll
                for (int cc = 0; cc < 4; cc++) pa[r][cc] = 0.0f;
            }
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                const float4 a0 = *(const float4 *) (buf + kk * 128 + 4 * ty);
                const float4 a1 = *(const float4 *) (buf + kk * 128 + 64 + 4 * ty);
                const float4 b  = *(const float4 *) (Sx + kk * 64);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const float bc = f4get(b, cc);
                    pa[0][cc] = xfma(a0.x, bc, pa[0][cc]);
                    pa[1][cc] = xfma(a0.y, bc, pa[1][cc]);
                    pa[2][cc] = xfma(a0.z, bc, pa[2][cc]);
                    pa[3][cc] = xfma(a0.w, bc, pa[3][cc]);
                    pa[4][cc] = xfma(a1.x, bc, pa[4][cc]);
                    pa[5][cc] = xfma(a1.y, bc, pa[5][cc]);
                    pa[6][cc] = xfma(a1.z, bc, pa[6][cc]);
                    pa[7][cc] = xfma(a1.w, bc, pa[7][cc]);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
#pragma unroll
                for (int cc = 0; cc < 4; cc++) acc[r][cc] += pa[r][cc];
            }
        }

        // K slab loader for step C: 16 tokens x 128 keys, rows scaled by exp(G_C - G_t)
        float4 lk0, lk1;
        auto load_kslab = [&](int tt0) {
#pragma unroll
            for (int q2 = 0; q2 < 2; q2++) {
                const int f  = tid + 256 * q2;
                const int tt = tt0 + (f >> 5);
                const int i4 = (f & 31) * 4;
                float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
                if (tt < len) {
                    val = ldg4(kb + (int64_t) (t0 + tt) * p.sq2 + i4);
                    const float e = __ldg(gxc + 64 + tt);
                    val.x *= e; val.y *= e; val.z *= e; val.w *= e;
                }
                if (q2 == 0) lk0 = val; else lk1 = val;
            }
        };
        auto store_kslab = [&](float * buf) {
            *(float4 *) (buf + (tid >> 5) * 128 + (tid & 31) * 4)       = lk0;
            *(float4 *) (buf + (8 + (tid >> 5)) * 128 + (tid & 31) * 4) = lk1;
        };
        load_kslab(0);

        __syncthreads();  // all reads of S (X) and staging (Y) done

        // ---- residual R = beta * (v - exp(G) * K S0) -> D ; A -> X2 ----
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float e = f4get(eG4, r), bt = f4get(bt4, r);
            const float4 d = make_float4((float)((vv[r].x - e * acc[r][0]) * bt), (float)((vv[r].y - e * acc[r][1]) * bt),
                                         (float)((vv[r].z - e * acc[r][2]) * bt), (float)((vv[r].w - e * acc[r][3]) * bt));
            *(float4 *) (D + (4 * ty + r) * 64 + 4 * tx) = d;
        }
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) *(float4 *) (X2 + (tid + 256 * q4) * 4) = pre[q4];
        store_kslab(Y);
        TB o[4][4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float e = f4get(eG4, r);
#pragma unroll
            for (int cc = 0; cc < 4; cc++) o[r][cc] = e * acc[4 + r][cc];
        }
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) pre[q4] = ldg4(Ptc + (tid + 256 * q4) * 4);
        __syncthreads();

        // ---- blocked forward substitution: Delta = (I + A)^-1 R, in place in D ----
        {
            const int l  = lane & 15;
            const int fc = 8 * wid + 4 * (lane >> 4);
#pragma unroll 1
            for (int b = 0; b < 4; b++) {
                const int row = 16 * b + l;
                float4 x = *(const float4 *) (D + row * 64 + fc);
                const float * Ar = X2 + row * 64 + 16 * b;
#pragma unroll
                for (int s4 = 0; s4 < 4; s4++) {
                    const float4 a4 = *(const float4 *) (Ar + 4 * s4);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int ss = 4 * s4 + e;
                        const int src = ss + (lane & 16);
                        const float d0 = __shfl_sync(0xffffffff, x.x, src, 32);
                        const float d1 = __shfl_sync(0xffffffff, x.y, src, 32);
                        const float d2 = __shfl_sync(0xffffffff, x.z, src, 32);
                        const float d3 = __shfl_sync(0xffffffff, x.w, src, 32);
                        if (l > ss) {
                            const float a = f4get(a4, e);
                            x.x = fmaf(-a, d0, x.x);
                            x.y = fmaf(-a, d1, x.y);
                            x.z = fmaf(-a, d2, x.z);
                            x.w = fmaf(-a, d3, x.w);
                        }
                    }
                }
                *(float4 *) (D + row * 64 + fc) = x;
                __syncthreads();
                if (b < 3) {
                    // rows below the block: R_t -= sum_{s in block b} A[t][s] Delta_s
                    for (int t = 16 * (b + 1) + ty; t < 64; t += 16) {
                        TF pr[4] = {0.f, 0.f, 0.f, 0.f};
                        const float * At = X2 + t * 64 + 16 * b;
#pragma unroll
                        for (int s4 = 0; s4 < 4; s4++) {
                            const float4 a4 = *(const float4 *) (At + 4 * s4);
#pragma unroll
                            for (int e = 0; e < 4; e++) {
                                const float4 dd = *(const float4 *) (D + (16 * b + 4 * s4 + e) * 64 + 4 * tx);
                                const float a = f4get(a4, e);
                                pr[0] = xfma(a, dd.x, pr[0]);
                                pr[1] = xfma(a, dd.y, pr[1]);
                                pr[2] = xfma(a, dd.z, pr[2]);
                                pr[3] = xfma(a, dd.w, pr[3]);
                            }
                        }
                        float4 * dp = (float4 *) (D + t * 64 + 4 * tx);
                        float4 cur = *dp;
                        cur.x = (float)(cur.x - pr[0]); cur.y = (float)(cur.y - pr[1]); cur.z = (float)(cur.z - pr[2]); cur.w = (float)(cur.w - pr[3]);
                        *dp = cur;
                    }
                    __syncthreads();
                }
            }
        }

        // ---- step B: O = scale * (exp(G) Q S0 + P Delta) ----
#pragma unroll
        for (int q4 = 0; q4 < 4; q4++) *(float4 *) (X2 + (tid + 256 * q4) * 4) = pre[q4];
        __syncthreads();
        {
            const int nsb = min(4, ((8 * wid + 8) + 15) / 16);   // rows of this warp are < 8*wid+8
#pragma unroll 1
            for (int sb = 0; sb < nsb; sb++) {
                TB po[4][4];
#pragma unroll
                for (int r = 0; r < 4; r++) {
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) po[r][cc] = 0.0f;
                }
#pragma unroll
                for (int e = 0; e < 16; e++) {
                    const int ss = 16 * sb + e;
                    const float4 a = *(const float4 *) (X2 + ss * 64 + 4 * ty);
                    const float4 bq = *(const float4 *) (D + ss * 64 + 4 * tx);
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) {
                        const float bc = f4get(bq, cc);
                        po[0][cc] = xfma(a.x, bc, po[0][cc]);
                        po[1][cc] = xfma(a.y, bc, po[1][cc]);
                        po[2][cc] = xfma(a.z, bc, po[2][cc]);
                        po[3][cc] = xfma(a.w, bc, po[3][cc]);
                    }
                }
#pragma unroll
                for (int r = 0; r < 4; r++) {
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) o[r][cc] += po[r][cc];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const int t = 4 * ty + r;
            if (t < len) {
                *(float4 *) (dstb + (int64_t) (t0 + t) * p.H * 128) =
                    make_float4((float)(o[r][0] * p.scale), (float)(o[r][1] * p.scale), (float)(o[r][2] * p.scale), (float)(o[r][3] * p.scale));
            }
        }

        // ---- step C: S1 = eGC * S0 + Kd^T Delta ----
        const float eGC = __ldg(gxc + 192);
#pragma unroll
        for (int r = 0; r < 8; r++) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++) s[r][cc] *= eGC;
        }
#pragma unroll 1
        for (int sl = 0; sl < 4; sl++) {
            const float * buf = Y + (sl & 1) * 2048;
            if (sl < 3) load_kslab((sl + 1) * 16);
            const float * Dx = D + sl * 16 * 64 + 4 * tx;
            TC pc[8][4];
#pragma unroll
            for (int r = 0; r < 8; r++) {
#pragma unroll
                for (int cc = 0; cc < 4; cc++) pc[r][cc] = 0.0f;
            }
#pragma unroll
            for (int tt = 0; tt < 16; tt++) {
                const float4 a0 = *(const float4 *) (buf + tt * 128 + 4 * ty);
                const float4 a1 = *(const float4 *) (buf + tt * 128 + 64 + 4 * ty);
                const float4 b  = *(const float4 *) (Dx + tt * 64);
#pragma unroll
                for (int cc = 0; cc < 4; cc++) {
                    const float bc = f4get(b, cc);
                    pc[0][cc] = xfma(a0.x, bc, pc[0][cc]);
                    pc[1][cc] = xfma(a0.y, bc, pc[1][cc]);
                    pc[2][cc] = xfma(a0.z, bc, pc[2][cc]);
                    pc[3][cc] = xfma(a0.w, bc, pc[3][cc]);
                    pc[4][cc] = xfma(a1.x, bc, pc[4][cc]);
                    pc[5][cc] = xfma(a1.y, bc, pc[5][cc]);
                    pc[6][cc] = xfma(a1.z, bc, pc[6][cc]);
                    pc[7][cc] = xfma(a1.w, bc, pc[7][cc]);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
#pragma unroll
                for (int cc = 0; cc < 4; cc++) s[r][cc] = (float)(s[r][cc] + pc[r][cc]);
            }
            if (sl < 3) {
                store_kslab(Y + ((sl + 1) & 1) * 2048);
            }
            __syncthreads();
        }
        store_S();
    }

    // ---- final state / snapshots ----
    float * st = p.state + (int64_t) (seq * p.H + h) * 16384;
    auto write_state = [&](float * base) {
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const int j = j0 + 4 * tx + cc;
            *(float4 *) (base + j * 128 + 4 * ty)      = make_float4(s[0][cc], s[1][cc], s[2][cc], s[3][cc]);
            *(float4 *) (base + j * 128 + 64 + 4 * ty) = make_float4(s[4][cc], s[5][cc], s[6][cc], s[7][cc]);
        }
    };
    if (!p.keep) {
        write_state(st);
        return;
    }
    write_state(st + (int64_t) (p.n_tokens - p.L) * p.state_slot_stride);

    // plain recurrence for the trailing tokens, one snapshot after each
    float * red = Y;
    float * kvs = Y + 1024;
    __syncthreads();
    for (int t = p.L; t < p.n_tokens; t++) {
        float kr[8], qr[8];
        const float * kt = kb + (int64_t) t * p.sq2;
        const float * qt = qb + (int64_t) t * p.sq2;
        {
            const float4 a = ldg4(kt + 4 * ty), b = ldg4(kt + 64 + 4 * ty);
            kr[0] = a.x; kr[1] = a.y; kr[2] = a.z; kr[3] = a.w; kr[4] = b.x; kr[5] = b.y; kr[6] = b.z; kr[7] = b.w;
            const float4 c = ldg4(qt + 4 * ty), d = ldg4(qt + 64 + 4 * ty);
            qr[0] = c.x; qr[1] = c.y; qr[2] = c.z; qr[3] = c.w; qr[4] = d.x; qr[5] = d.y; qr[6] = d.z; qr[7] = d.w;
        }
        const int64_t gbo = (int64_t) seq * p.sb3 + (int64_t) t * p.sb2 + (int64_t) h * p.sb1;
        const float eg = expf(p.g[gbo]);
        const float bt = p.beta[gbo];
        const float * vt = p.v + (int64_t) seq * p.sv3 + (int64_t) t * p.sv2 + (int64_t) h * p.sv1 + j0 + 4 * tx;
        float part[4];
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            float a = 0.0f;
#pragma unroll
            for (int r = 0; r < 8; r++) a = fmaf(s[r][cc], kr[r], a);
            part[cc] = a;
        }
        *(float4 *) (red + ty * 64 + 4 * tx) = make_float4(part[0], part[1], part[2], part[3]);
        __syncthreads();
        if (tid < 64) {
            float a = 0.0f;
            for (int y = 0; y < 16; y++) a += red[y * 64 + tid];
            kvs[tid] = a;
        }
        __syncthreads();
#pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            const float delta = (vt[cc] - eg * kvs[4 * tx + cc]) * bt;
            float a = 0.0f;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                s[r][cc] = fmaf(eg, s[r][cc], kr[r] * delta);
                a = fmaf(s[r][cc], qr[r], a);
            }
            part[cc] = a;
        }
        *(float4 *) (red + ty * 64 + 4 * tx) = make_float4(part[0], part[1], part[2], part[3]);
        __syncthreads();
        if (tid < 64) {
            float a = 0.0f;
            for (int y = 0; y < 16; y++) a += red[y * 64 + tid];
            p.dst[((int64_t) seq * p.n_tokens * p.H + (int64_t) t * p.H + h) * 128 + j0 + tid] = a * p.scale;
        }
        write_state(st + (int64_t) (p.n_tokens - 1 - t) * p.state_slot_stride);
        __syncthreads();
    }
}

// scratch floats needed (upper bound; L <= n_tokens)
static size_t gdn_ch_scratch_floats(int H, int neqk1, int n_seqs, int L) {
    const size_t nch = (L + GCH - 1) / GCH;
    return nch * n_seqs * ((size_t) H * (4096 + 4096 + 256) + (size_t) neqk1 * (8192 + 8192));
}

static void gdn_ch_launch(gdn_ch_params p, float * scratch, cudaStream_t stream) {
    p.L   = p.keep ? p.n_tokens - (p.K - 1) : p.n_tokens;
    p.nch = (p.L + GCH - 1) / GCH;
    const size_t nch = p.nch;
    float * ptr = scratch;
    p.Am = ptr; ptr += nch * p.n_seqs * p.H * 4096;
    p.Pt = ptr; ptr += nch * p.n_seqs * p.H * 4096;
    p.gx = ptr; ptr += nch * p.n_seqs * p.H * 256;
    p.KT = ptr; ptr += nch * p.n_seqs * p.neqk1 * 8192;
    p.QT = ptr;
    gdn2_prep<<<dim3(p.nch, p.neqk1, p.n_seqs), 256, 0, stream>>>(p);
    gdn2_main<<<dim3(2, p.H, p.n_seqs), 256, 0, stream>>>(p);
}
