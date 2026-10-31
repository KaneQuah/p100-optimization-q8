// MoE FFN with the routed experts split across the devices of a meta backend (expert parallelism),
// compared with the same graph on one backend. The meta backend is made of CPU devices.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

struct moe_shape {
    int64_t   n_embd;
    int64_t   n_ff;
    int64_t   n_expert;
    int64_t   n_used;
    int64_t   n_tokens;
    ggml_type type;
    bool      gate_up; // fused gate and up experts
};

struct moe_tensors {
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * w;
    ggml_tensor * up;
    ggml_tensor * gate;
    ggml_tensor * down;
};

struct split_ud {
    size_t n_dev;
};

// whole experts per device for the "exps." tensors, everything else mirrored
static ggml_backend_meta_split_state get_split_state(const ggml_tensor * t, void * userdata) {
    const size_t n_dev = ((const split_ud *) userdata)->n_dev;
    ggml_backend_meta_split_state ss;
    memset(&ss, 0, sizeof(ss));
    ss.nr[0]      = 1;
    ss.n_segments = 1;
    if (strncmp(t->name, "exps.", 5) == 0) {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_2;
        int64_t low = 0;
        for (size_t j = 0; j < n_dev; j++) {
            const int64_t high = t->ne[2]*int64_t(j + 1)/int64_t(n_dev);
            ss.ne[j] = high - low;
            low = high;
        }
    } else {
        ss.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return ss;
}

static moe_tensors new_moe_tensors(ggml_context * ctx, const moe_shape & s) {
    moe_tensors t = {};
    t.x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n_embd, s.n_tokens);
    t.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.n_used, s.n_tokens);
    t.w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, s.n_used, s.n_tokens);
    if (s.gate_up) {
        t.up = ggml_new_tensor_3d(ctx, s.type, s.n_embd, 2*s.n_ff, s.n_expert);
        ggml_set_name(t.up, "exps.gate_up");
    } else {
        t.up   = ggml_new_tensor_3d(ctx, s.type, s.n_embd, s.n_ff, s.n_expert);
        t.gate = ggml_new_tensor_3d(ctx, s.type, s.n_embd, s.n_ff, s.n_expert);
        ggml_set_name(t.up,   "exps.up");
        ggml_set_name(t.gate, "exps.gate");
    }
    t.down = ggml_new_tensor_3d(ctx, s.type, s.n_ff, s.n_embd, s.n_expert);
    ggml_set_name(t.x,    "x");
    ggml_set_name(t.ids,  "ids");
    ggml_set_name(t.w,    "w");
    ggml_set_name(t.down, "exps.down");
    return t;
}

// same op order as llm_graph_context::build_moe_ffn
static ggml_tensor * build_moe(ggml_context * ctx, ggml_cgraph * gf, const moe_tensors & t, const moe_shape & s) {
    ggml_tensor * cur = ggml_reshape_3d(ctx, t.x, s.n_embd, 1, s.n_tokens);
    ggml_tensor * gate;
    ggml_tensor * up;
    if (s.gate_up) {
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx, t.up, cur, t.ids);
        gate = ggml_view_3d(ctx, gate_up, s.n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        up   = ggml_view_3d(ctx, gate_up, s.n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], s.n_ff*gate_up->nb[0]);
    } else {
        up   = ggml_mul_mat_id(ctx, t.up,   cur, t.ids);
        gate = ggml_mul_mat_id(ctx, t.gate, cur, t.ids);
    }
    ggml_tensor * act     = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, t.down, act, t.ids);
    experts = ggml_mul(ctx, experts, t.w);
    ggml_build_forward_expand(gf, experts);

    std::vector<ggml_tensor *> views(s.n_used);
    for (int64_t i = 0; i < s.n_used; i++) {
        views[i] = ggml_view_2d(ctx, experts, s.n_embd, s.n_tokens, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, views[i]);
    }
    ggml_tensor * out = views[0];
    for (int64_t i = 1; i < s.n_used; i++) {
        out = ggml_add(ctx, out, views[i]);
        ggml_build_forward_expand(gf, out);
    }
    // residual add: the MoE sum needs a consumer for the meta backend to delay the AllReduce past the adds
    out = ggml_add(ctx, out, t.x);
    ggml_build_forward_expand(gf, out);
    return out;
}

static void set_random(ggml_tensor * t, std::mt19937 & rng, float amax) {
    std::uniform_real_distribution<float> dist(-amax, amax);
    std::vector<float> f(ggml_nelements(t));
    for (float & v : f) {
        v = dist(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// fill the inputs and weights from a fixed seed, so both runs see the same data
static void set_moe_data(const moe_tensors & t, const moe_shape & s, uint32_t seed) {
    std::mt19937 rng(seed);
    set_random(t.x, rng, 1.0f);
    set_random(t.w, rng, 1.0f);
    set_random(t.up, rng, 0.1f);
    if (t.gate) {
        set_random(t.gate, rng, 0.1f);
    }
    set_random(t.down, rng, 0.1f);

    std::vector<int32_t> ids(s.n_used*s.n_tokens);
    std::vector<int32_t> perm(s.n_expert);
    for (int64_t it = 0; it < s.n_tokens; it++) {
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::copy(perm.begin(), perm.begin() + s.n_used, ids.begin() + it*s.n_used);
    }
    ggml_backend_tensor_set(t.ids, ids.data(), 0, ggml_nbytes(t.ids));
}

static std::vector<float> run_moe(ggml_backend_t backend, ggml_backend_buffer_type_t buft, const moe_shape & s, uint32_t seed) {
    const ggml_init_params params_w = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx_w(ggml_init(params_w));
    const moe_tensors t = new_moe_tensors(ctx_w.get(), s);
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors_from_buft(ctx_w.get(), buft));
    set_moe_data(t, s, seed);

    const ggml_init_params params_g = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx_g(ggml_init(params_g));
    ggml_cgraph * gf  = ggml_new_graph(ctx_g.get());
    ggml_tensor * out = build_moe(ctx_g.get(), gf, t, s);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    return result;
}

int main() {
    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev_cpu != nullptr);
    ggml_backend_ptr backend_cpu(ggml_backend_dev_init(dev_cpu, nullptr));

    int n_fail = 0;
    int n_test = 0;
    for (size_t n_dev : {4, 3, 2}) {
        split_ud ud = { n_dev };
        std::vector<ggml_backend_dev_t> devs(n_dev, dev_cpu);
        ggml_backend_dev_t dev_meta = ggml_backend_meta_device(devs.data(), n_dev, get_split_state, &ud);
        ggml_backend_ptr backend_meta(ggml_backend_dev_init(dev_meta, nullptr));

        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K}) {
            for (int64_t n_tokens : {1, 7, 64}) {
                for (bool gate_up : {false, true}) {
                    const moe_shape s = { 256, 512, 16, 4, n_tokens, type, gate_up };
                    const uint32_t seed = 1234 + uint32_t(n_tokens);

                    const std::vector<float> ref = run_moe(backend_cpu.get(),  ggml_backend_dev_buffer_type(dev_cpu),  s, seed);
                    const std::vector<float> res = run_moe(backend_meta.get(), ggml_backend_dev_buffer_type(dev_meta), s, seed);

                    double err = 0.0;
                    double amax = 0.0;
                    for (size_t i = 0; i < ref.size(); i++) {
                        err  = std::max(err,  (double) std::fabs(ref[i] - res[i]));
                        amax = std::max(amax, (double) std::fabs(ref[i]));
                    }
                    const bool ok = std::isfinite(err) && err <= 1e-5*amax;
                    printf("  n_dev=%zu type=%s n_tokens=%3lld gate_up=%d: max_err=%.3e max_abs=%.3e %s\n",
                        n_dev, ggml_type_name(type), (long long) n_tokens, gate_up, err, amax, ok ? "OK" : "FAIL");
                    n_fail += ok ? 0 : 1;
                    n_test++;
                }
            }
        }
    }
    printf("%d/%d tests passed\n", n_test - n_fail, n_test);
    return n_fail == 0 ? 0 : 1;
}
