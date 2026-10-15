// Chunked gated delta net for prefill (scalar gate, S_v = 128), see gdn-chunked.cuh.
//   GGML_CUDA_GDN_CHUNKED: 0 = plain recurrence, 1 (default) = chunked for n_tokens >= 64 on
//   Pascal and not under tensor split (no gain there), 2 = chunked for n_tokens >= 64 on any device
#include "gated_delta_net.cuh"
#include "gdn-chunked.cuh"

bool ggml_cuda_gdn_chunked(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunked_args & a) {
    static const int mode = getenv("GGML_CUDA_GDN_CHUNKED") ? atoi(getenv("GGML_CUDA_GDN_CHUNKED")) : 1;
    const bool keep = a.K > 1;
    const int64_t L = keep ? a.n_tokens - (a.K - 1) : a.n_tokens;
    const bool gated_off = mode == 1 && (ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_VOLTA || ctx.tensor_split);
    if (mode == 0 || gated_off || L < GCH) {
        return false;
    }
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), gdn_ch_scratch_floats(a.H, a.neqk1, a.n_seqs, L));
    gdn_ch_params p = {};
    p.q = a.q; p.k = a.k; p.v = a.v; p.g = a.g; p.beta = a.beta;
    p.curr_state = a.curr_state; p.dst = a.dst; p.state = a.state;
    p.H = a.H; p.n_tokens = a.n_tokens; p.n_seqs = a.n_seqs; p.neqk1 = a.neqk1; p.rq3 = a.rq3; p.K = a.K;
    p.sq1 = a.sq1; p.sq2 = a.sq2; p.sq3 = a.sq3; p.sv1 = a.sv1; p.sv2 = a.sv2; p.sv3 = a.sv3;
    p.sb1 = a.sb1; p.sb2 = a.sb2; p.sb3 = a.sb3;
    p.scale = a.scale; p.state_slot_stride = a.state_slot_stride;
    p.keep = keep;
    gdn_ch_launch(p, scratch.get(), ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    return true;
}
