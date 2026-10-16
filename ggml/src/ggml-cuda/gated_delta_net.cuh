#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data        = nullptr; // rollback slot 0 (nullptr: state goes to the dst tail)
    int64_t slot_stride = 0;       // between rollback slots (0 when K==1)

    // optional input: read the state of sequence s from row state_ids[s] of state_src (nullptr: use src[5])
    const float * state_src        = nullptr;
    const int *   state_ids        = nullptr;
    int64_t       state_row_stride = 0; // in floats

    // optional pre-ops done in the kernel (S_v = 128): q, k before l2_scale * rms_norm(x, l2_eps),
    // alpha with g = softplus(alpha + dt) * A, b with beta = sigmoid(b)
    const ggml_tensor * q        = nullptr;
    const ggml_tensor * k        = nullptr;
    const ggml_tensor * alpha    = nullptr;
    const ggml_tensor * b        = nullptr;
    const float *       dt       = nullptr;
    const float *       A        = nullptr;
    float               l2_eps   = 0.0f;
    float               l2_scale = 0.0f;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// chunked prefill form for scalar-gate S_v = 128 (gdn-chunked.cu); false = run the plain recurrence
struct ggml_cuda_gdn_chunked_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride; int K;
};
bool ggml_cuda_gdn_chunked(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunked_args & a);
