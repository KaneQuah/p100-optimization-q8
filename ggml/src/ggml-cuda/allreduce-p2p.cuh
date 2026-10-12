#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Single-kernel peer-to-peer AllReduce for F32 tensors on 2..8 GPUs.
// Default when there are exactly 4 devices, all cc 6.0 (P100). Select with GGML_CUDA_ALLREDUCE=p2p, turn off with GGML_CUDA_ALLREDUCE=none.
struct ggml_cuda_ar_p2p;

// Returns nullptr if peer access is not available between all devices.
ggml_cuda_ar_p2p * ggml_cuda_ar_p2p_init(const int * devices, size_t n_devices);

void ggml_cuda_ar_p2p_free(ggml_cuda_ar_p2p * p);

// In-place sum over tensors[0..n_devices-1], enqueued on the stream of each backend.
// Returns false if the tensors are not supported, nothing is enqueued then.
bool ggml_cuda_ar_p2p_allreduce(ggml_cuda_ar_p2p * p, ggml_backend_t * backends, ggml_tensor ** tensors);
