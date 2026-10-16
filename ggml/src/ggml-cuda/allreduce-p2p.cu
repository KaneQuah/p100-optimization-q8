#include "allreduce-p2p.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// AllReduce over peer-to-peer access, any number of GPUs (2..8), F32 only.
// One kernel launch per GPU and no host sync, which is what token generation needs.
//
// Reduce-scatter and all-gather in one kernel, with no staging buffers for f32: rank r reads part r of the partial sums
// directly from the tensors of all peers, sums in rank order, and writes the result to its own tensor and to the
// tensors of all peers. Each GPU moves 2*(n-1)/n times the tensor over PCIe in each direction instead of n-1 times.
// Only rank r computes part r, so all GPUs get bit-identical results.
//
// Two flags per AR and block: 2*token when the partial sum is ready, 2*token+1 when the block pushed its part.
// A peer reads our partial sums before it pushes, so when we saw the second flag of all peers our tensor is complete
// and no peer still reads it: the kernels after the AR may overwrite it.

static constexpr int GGML_CUDA_AR_P2P_MAX_RANKS  = 8;
static constexpr int GGML_CUDA_AR_P2P_MAX_BLOCKS = 32;
static constexpr int GGML_CUDA_AR_P2P_FLAG_INTS  = 16; // 64 bytes per flag, one flag per cache line

struct ggml_cuda_ar_p2p_args {
    float * data[GGML_CUDA_AR_P2P_MAX_RANKS];  // tensor of each rank
    uint2 * stage[GGML_CUDA_AR_P2P_MAX_RANKS]; // bf16 only: packed partial sums, then the inbox for the results
    int   * flags[GGML_CUDA_AR_P2P_MAX_RANKS]; // flag area of each rank
    int     rank;
    int     n_ranks;
    int     count_vec; // number of float4
    int     part_vec;  // number of float4 reduced by each rank
    unsigned token;
};

// set our flag in all peers, then wait for the flag of all peers; per block
static __device__ __forceinline__ void ggml_cuda_ar_p2p_sync(const ggml_cuda_ar_p2p_args & args, const unsigned val) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    if (tid < args.n_ranks && tid != args.rank) {
        volatile int * flag = args.flags[tid] + (bid * GGML_CUDA_AR_P2P_MAX_RANKS + args.rank) * GGML_CUDA_AR_P2P_FLAG_INTS;
        *flag = val;
        const volatile int * wait = args.flags[args.rank] + (bid * GGML_CUDA_AR_P2P_MAX_RANKS + tid) * GGML_CUDA_AR_P2P_FLAG_INTS;
        while ((int) ((unsigned) *wait - val) < 0) {
        }
    }
    __syncthreads();
}

static __device__ __forceinline__ uint2 ggml_cuda_ar_p2p_pack_bf16(const float4 v) {
    const nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
    const nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
    return make_uint2(*(const unsigned *) &a, *(const unsigned *) &b);
}

static __device__ __forceinline__ float4 ggml_cuda_ar_p2p_unpack_bf16(const uint2 u) {
    const float2 a = __bfloat1622float2(*(const nv_bfloat162 *) &u.x);
    const float2 b = __bfloat1622float2(*(const nv_bfloat162 *) &u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

// f32: peers read and write the tensors directly.
// bf16: peers read our packed partial sums from our staging buffer and push the packed results into its inbox.
template <bool bf16>
static __global__ void ggml_cuda_ar_p2p_kernel(const ggml_cuda_ar_p2p_args args) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt  = gridDim.x * blockDim.x;

    float4 * data = (float4 *) args.data[args.rank];
    uint2  * mine = args.stage[args.rank];

    if constexpr (bf16) {
        // same index to block mapping as the reduce loop of the peers, so the per-block flags cover it
        for (int j = gtid; j < args.part_vec; j += gnt) {
            for (int r = 0; r < args.n_ranks; ++r) {
                const int i = r * args.part_vec + j;
                if (r != args.rank && i < args.count_vec) {
                    mine[i] = ggml_cuda_ar_p2p_pack_bf16(data[i]);
                }
            }
        }
        __threadfence_system();
        __syncthreads();
    }

    ggml_cuda_ar_p2p_sync(args, 2*args.token);
    __threadfence();

    const int i0 = args.rank * args.part_vec;
    const int i1 = min(i0 + args.part_vec, args.count_vec);

    for (int i = i0 + gtid; i < i1; i += gnt) {
        float4 v[GGML_CUDA_AR_P2P_MAX_RANKS];
        #pragma unroll
        for (int r = 0; r < GGML_CUDA_AR_P2P_MAX_RANKS; ++r) {
            if (r < args.n_ranks) {
                if (r == args.rank) {
                    v[r] = data[i];
                } else if constexpr (bf16) {
                    v[r] = ggml_cuda_ar_p2p_unpack_bf16(__ldcv(args.stage[r] + i));
                } else {
                    v[r] = __ldcv(((const float4 *) args.data[r]) + i);
                }
            }
        }
        // Same add order as the meta-backend butterfly, so results are identical to the fallback
        int m = 1;
        while (2*m <= args.n_ranks) {
            m *= 2;
        }
        #pragma unroll
        for (int r = 1; r < GGML_CUDA_AR_P2P_MAX_RANKS; ++r) {
            if (r >= m && r < args.n_ranks) {
                v[r - m].x += v[r].x;
                v[r - m].y += v[r].y;
                v[r - m].z += v[r].z;
                v[r - m].w += v[r].w;
            }
        }
        #pragma unroll
        for (int stride = GGML_CUDA_AR_P2P_MAX_RANKS/2; stride >= 1; stride /= 2) {
            #pragma unroll
            for (int r = 0; r < GGML_CUDA_AR_P2P_MAX_RANKS/2; ++r) {
                if (stride < m && r < stride) {
                    v[r].x += v[r + stride].x;
                    v[r].y += v[r + stride].y;
                    v[r].z += v[r + stride].z;
                    v[r].w += v[r + stride].w;
                }
            }
        }
        if constexpr (bf16) {
            // keep the rounded value too, so all ranks get the same result
            const uint2 w = ggml_cuda_ar_p2p_pack_bf16(v[0]);
            data[i] = ggml_cuda_ar_p2p_unpack_bf16(w);
            #pragma unroll
            for (int r = 0; r < GGML_CUDA_AR_P2P_MAX_RANKS; ++r) {
                if (r < args.n_ranks && r != args.rank) {
                    args.stage[r][args.count_vec + i] = w;
                }
            }
        } else {
            data[i] = v[0];
            #pragma unroll
            for (int r = 0; r < GGML_CUDA_AR_P2P_MAX_RANKS; ++r) {
                if (r < args.n_ranks && r != args.rank) {
                    ((float4 *) args.data[r])[i] = v[0];
                }
            }
        }
    }
    __threadfence_system();
    __syncthreads();

    ggml_cuda_ar_p2p_sync(args, 2*args.token + 1);

    if constexpr (bf16) {
        __threadfence();
        for (int j = gtid; j < args.part_vec; j += gnt) {
            for (int r = 0; r < args.n_ranks; ++r) {
                const int i = r * args.part_vec + j;
                if (r != args.rank && i < args.count_vec) {
                    data[i] = ggml_cuda_ar_p2p_unpack_bf16(__ldcv(mine + args.count_vec + i));
                }
            }
        }
    }
}

struct ggml_cuda_ar_p2p {
    int    n_devices = 0;
    int    devices[GGML_CUDA_AR_P2P_MAX_RANKS] = {};
    int  * flags[GGML_CUDA_AR_P2P_MAX_RANKS]   = {};
    uint2 * stage[GGML_CUDA_AR_P2P_MAX_RANKS]  = {};
    size_t stage_bytes = 0; // > 0: large tensors go over the wire as bf16
    unsigned token    = 0;
    int    threads    = 32;
    int    max_blocks = 0; // 0: automatic
};

static size_t ggml_cuda_ar_p2p_env(const char * name, size_t def) {
    const char * s = getenv(name);
    return s ? (size_t) strtoull(s, nullptr, 0) : def;
}

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p) {
    if (p == nullptr) {
        return;
    }
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cudaFree(p->flags[i]);
        cudaFree(p->stage[i]);
    }
    delete p;
}

ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices) {
    // peers read and write the tensors directly, which must not be managed memory
    if (n_devices < 2 || n_devices > GGML_CUDA_AR_P2P_MAX_RANKS || getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        return nullptr;
    }

    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(devices[i]);
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]) != cudaSuccess || !can_access) {
                (void) cudaGetLastError();
                return nullptr;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
                return nullptr;
            }
            (void) cudaGetLastError();
        }
    }

    auto * p = new ggml_cuda_ar_p2p{};
    p->n_devices  = (int) n_devices;
    p->threads    = (int) std::max<size_t>(ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_THREADS", 32), GGML_CUDA_AR_P2P_MAX_RANKS);
    p->max_blocks = (int) std::min<size_t>(ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_BLOCKS", 0), GGML_CUDA_AR_P2P_MAX_BLOCKS);
    const size_t flag_bytes = (size_t) GGML_CUDA_AR_P2P_MAX_BLOCKS * GGML_CUDA_AR_P2P_MAX_RANKS * GGML_CUDA_AR_P2P_FLAG_INTS * sizeof(int);

    // opt-in, changes results: tensors of 32768 values and more go over the wire as bf16, which halves the PCIe traffic (prefill)
    if (ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_BF16", 0) != 0) {
        p->stage_bytes = 16 * 1024 * 1024;
    }

    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
        p->n_devices  = (int) i + 1;
        ggml_cuda_set_device(devices[i]);
        if (cudaMalloc(&p->flags[i], flag_bytes) != cudaSuccess ||
            cudaMemset(p->flags[i], 0, flag_bytes) != cudaSuccess ||
            (p->stage_bytes > 0 && cudaMalloc(&p->stage[i], p->stage_bytes) != cudaSuccess)) {
            (void) cudaGetLastError();
            ggml_cuda_ar_p2p_free(p);
            return nullptr;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return p;
}

bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p * p, ggml_backend_t * backends, ggml_tensor ** tensors) {
    const int64_t ne     = ggml_nelements(tensors[0]);
    const size_t  nbytes = ggml_nbytes(tensors[0]);
    if (tensors[0]->type != GGML_TYPE_F32 || ne % 4 != 0 || ne / 4 > INT_MAX / 2) {
        return false;
    }
    for (int i = 0; i < p->n_devices; ++i) {
        if (tensors[i] == nullptr || ggml_nelements(tensors[i]) != ne || tensors[i]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguously_allocated(tensors[i]) || ((uintptr_t) tensors[i]->data & 0xF) != 0) {
            return false;
        }
    }

    ggml_cuda_ar_p2p_args args = {};
    for (int r = 0; r < p->n_devices; ++r) {
        args.data[r]  = (float *) tensors[r]->data;
        args.stage[r] = p->stage[r];
        args.flags[r] = p->flags[r];
    }
    args.n_ranks   = p->n_devices;
    args.count_vec = (int) (ne / 4);
    args.part_vec  = (args.count_vec + p->n_devices - 1) / p->n_devices;
    args.token     = ++p->token;

    // the staging holds the packed partial sums and the packed results: 2 x 2 bytes per value
    const bool bf16 = ne >= 32768 && nbytes <= p->stage_bytes;

    // one float4 per thread while it fits into 16 blocks (latency bound), else 8 blocks: more loads in flight only add PCIe contention
    // bf16 also packs and unpacks locally, which wants more threads
    const int threads = bf16 ? 2*p->threads : p->threads;
    int blocks = std::max(1, (args.part_vec + threads - 1) / threads);
    if (p->max_blocks > 0) {
        blocks = std::min(blocks, p->max_blocks);
    } else if (blocks > 16) {
        blocks = bf16 ? GGML_CUDA_AR_P2P_MAX_BLOCKS : 8;
    }

    for (int r = 0; r < p->n_devices; ++r) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[r]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[r]);
        ggml_cuda_set_device(p->devices[r]);
        if ((tensors[r]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            CUDA_CHECK(cudaMemsetAsync(tensors[r]->data, 0, nbytes, cuda_ctx->stream()));
        }
        args.rank = r;
        if (bf16) {
            ggml_cuda_ar_p2p_kernel<true><<<blocks, threads, 0, cuda_ctx->stream()>>>(args);
        } else {
            ggml_cuda_ar_p2p_kernel<false><<<blocks, threads, 0, cuda_ctx->stream()>>>(args);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

#else // !GGML_USE_HIP && !GGML_USE_MUSA

ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int *, size_t) {
    return nullptr;
}

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p *) {
}

bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA
