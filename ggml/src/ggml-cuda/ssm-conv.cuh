#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// ssm_conv + silu for one sequence that reads the conv state from the cache row ids[0] and writes the new state
// to state_dst (replaces get_rows + concat + cpy + ssm_conv + silu)
void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * conv_w,
                                 const ggml_tensor * states, const ggml_tensor * ids, float * state_dst, ggml_tensor * dst);
