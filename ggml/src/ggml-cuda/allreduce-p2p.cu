#include "allreduce-p2p.cuh"

#include <algorithm>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// One-shot AllReduce over peer-to-peer access, any number of GPUs (2..8), F32 only.
// One kernel launch per GPU and no host sync, which is what token generation needs. Tensors above GGML_CUDA_AR_P2P_MAX_BYTES (16 MB) use the fallback.
//
// Every GPU copies its partial sum to a staging buffer in its own memory and tells all peers
// through a flag in the peer's memory. Then every GPU reads the staging buffers of all
// peers, sums in rank order (so all GPUs get bit-identical results) and writes the result in place.
//
// Staging is a ring of 2 slots. A GPU can only write slot s of AR n+2 after it saw
// the flags of AR n+1 from all peers, and a peer sets that flag only after its AR n kernel
// ended, so nobody still reads slot s of AR n.

static constexpr int GGML_CUDA_AR_P2P_MAX_RANKS  = 8;
static constexpr int GGML_CUDA_AR_P2P_MAX_BLOCKS = 32;
static constexpr int GGML_CUDA_AR_P2P_FLAG_INTS  = 16; // 64 bytes per flag, one flag per cache line

struct ggml_cuda_ar_p2p_args {
    float * stage[GGML_CUDA_AR_P2P_MAX_RANKS]; // staging buffer of each rank for this slot
    int   * flags[GGML_CUDA_AR_P2P_MAX_RANKS]; // flag area of each rank
    float * data;
    int     rank;
    int     n_ranks;
    int     count_vec; // number of float4
    unsigned token;
};

static __global__ void ggml_cuda_ar_p2p_kernel(const ggml_cuda_ar_p2p_args args) {
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;

    float4 * data = (float4 *) args.data;
    float4 * mine = (float4 *) args.stage[args.rank];

    for (int i = gtid; i < args.count_vec; i += gnt) {
        mine[i] = data[i];
    }
    __threadfence();
    __syncthreads();

    if (tid < args.n_ranks && tid != args.rank) {
        volatile int * flag = args.flags[tid] + (bid * GGML_CUDA_AR_P2P_MAX_RANKS + args.rank) * GGML_CUDA_AR_P2P_FLAG_INTS;
        *flag = args.token;
        const volatile int * wait = args.flags[args.rank] + (bid * GGML_CUDA_AR_P2P_MAX_RANKS + tid) * GGML_CUDA_AR_P2P_FLAG_INTS;
        while ((int) ((unsigned) *wait - args.token) < 0) {
        }
    }
    __syncthreads();
    __threadfence();

    for (int i = gtid; i < args.count_vec; i += gnt) {
        float4 v[GGML_CUDA_AR_P2P_MAX_RANKS];
        #pragma unroll
        for (int r = 0; r < GGML_CUDA_AR_P2P_MAX_RANKS; ++r) {
            if (r < args.n_ranks) {
                v[r] = r == args.rank ? data[i] : __ldcv(((const float4 *) args.stage[r]) + i);
            }
        }
        // Same add order as the meta-backend butterfly, so results are identical on all ranks and to the fallback
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
        data[i] = v[0];
    }
}

struct ggml_cuda_ar_p2p {
    int    n_devices = 0;
    int    devices[GGML_CUDA_AR_P2P_MAX_RANKS] = {};
    float * stage[GGML_CUDA_AR_P2P_MAX_RANKS]  = {}; // 2 slots each
    int   * flags[GGML_CUDA_AR_P2P_MAX_RANKS]  = {};
    size_t slot_bytes = 0;
    unsigned token    = 0;
    int    threads    = 128;
    int    max_blocks = 32;
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
        cudaFree(p->stage[i]);
        cudaFree(p->flags[i]);
    }
    delete p;
}

ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices) {
    if (n_devices < 2 || n_devices > GGML_CUDA_AR_P2P_MAX_RANKS) {
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
    p->slot_bytes = ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_MAX_BYTES", 16 * 1024 * 1024);
    p->threads    = (int) ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_THREADS", 128);
    p->max_blocks = (int) std::min<size_t>(ggml_cuda_ar_p2p_env("GGML_CUDA_AR_P2P_BLOCKS", 32), GGML_CUDA_AR_P2P_MAX_BLOCKS);
    const size_t flag_bytes = (size_t) GGML_CUDA_AR_P2P_MAX_BLOCKS * GGML_CUDA_AR_P2P_MAX_RANKS * GGML_CUDA_AR_P2P_FLAG_INTS * sizeof(int);

    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
        p->n_devices  = (int) i + 1;
        ggml_cuda_set_device(devices[i]);
        if (cudaMalloc(&p->stage[i], 2 * p->slot_bytes) != cudaSuccess ||
            cudaMalloc(&p->flags[i], flag_bytes) != cudaSuccess ||
            cudaMemset(p->flags[i], 0, flag_bytes) != cudaSuccess) {
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
    if (tensors[0]->type != GGML_TYPE_F32 || ne % 4 != 0 || nbytes > p->slot_bytes) {
        return false;
    }
    for (int i = 0; i < p->n_devices; ++i) {
        if (tensors[i] == nullptr || ggml_nelements(tensors[i]) != ne || tensors[i]->type != GGML_TYPE_F32 ||
                !ggml_is_contiguously_allocated(tensors[i]) || ((uintptr_t) tensors[i]->data & 0xF) != 0) {
            return false;
        }
    }

    const unsigned token = ++p->token;
    const int      slot  = token & 1;

    ggml_cuda_ar_p2p_args args = {};
    for (int r = 0; r < p->n_devices; ++r) {
        args.stage[r] = p->stage[r] + slot * (p->slot_bytes / sizeof(float));
        args.flags[r] = p->flags[r];
    }
    args.n_ranks   = p->n_devices;
    args.count_vec = (int) (ne / 4);
    args.token     = token;

    const int blocks = std::max(1, std::min(p->max_blocks, (args.count_vec + p->threads - 1) / p->threads));

    for (int r = 0; r < p->n_devices; ++r) {
        auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[r]->context);
        GGML_ASSERT(cuda_ctx->device == p->devices[r]);
        ggml_cuda_set_device(p->devices[r]);
        if ((tensors[r]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            CUDA_CHECK(cudaMemsetAsync(tensors[r]->data, 0, nbytes, cuda_ctx->stream()));
        }
        args.rank = r;
        args.data = (float *) tensors[r]->data;
        ggml_cuda_ar_p2p_kernel<<<blocks, p->threads, 0, cuda_ctx->stream()>>>(args);
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
