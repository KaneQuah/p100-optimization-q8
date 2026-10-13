#pragma once

#include "common.cuh"

// cuBLAS-GEMM flash attention for long-KV prefill on Pascal (cc 6.x). GGML_CUDA_FA_GEMM=0 disables.
// Only the dispatch gates on this; the alloc-size path keeps the tile fallback's f16 staging.
bool ggml_cuda_flash_attn_ext_gemm_supported(int cc, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
