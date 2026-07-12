// mk_dispatch.cpp — R5 layer 1: on the k=0 decode graph, extract llama's raw
// per-GPU device pointers (the R2 mirror) for every weight and KV/state tensor,
// keyed by GGUF name, so the persistent megakernel can later run over them in
// place. This layer BUILDS and LOGS the pointer map and returns false (the caller
// falls through to the R3 forward). The dual_setup/dual_run_pass re-host (packing
// the split program against this map, launching the two persistent kernels,
// per-pass run + logit writeback) is the next layer.
//
// mk_dispatch(cgraph) returns true only once it actually runs the megakernel;
// until then it is a no-op probe that leaves execution to the forward path.

#include "ggml.h"
#include "ggml-impl.h"          // full ggml_cgraph struct (n_nodes/n_leafs/nodes/leafs)
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace {

// Mirror enough of ggml_backend_meta_buffer_context to reach simple_tensors
// (R2-proven: sizeof(std::map) is layout-independent, so the second member sits
// at the correct offset regardless of the value types).
struct meta_ctx_mirror {
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<int, int>> split_state_cache;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>>           simple_tensors;
};

struct dev_ptrs { void * p[2]; };

// Per-GPU device pointers behind a meta tensor, or false if not a meta tensor
// with 2 simple tensors carrying data.
static bool extract(const ggml_tensor * t, dev_ptrs & out) {
    if (!t || !t->buffer || !ggml_backend_buffer_is_meta(t->buffer)) return false;
    auto * mc = (meta_ctx_mirror *) t->buffer->context;
    auto it = mc->simple_tensors.find(t);
    if (it == mc->simple_tensors.end() || it->second.size() < 2) return false;
    out.p[0] = it->second[0] ? it->second[0]->data : nullptr;
    out.p[1] = it->second[1] ? it->second[1]->data : nullptr;
    return out.p[0] && out.p[1];
}

// Walk the graph, collect every named tensor (leaves = weights + KV/state, and
// any named node) that resolves to a meta tensor, into name -> per-GPU pointers.
static void build_ptr_map(const ggml_cgraph * g, std::map<std::string, dev_ptrs> & m) {
    auto consider = [&](const ggml_tensor * t) {
        if (!t || !t->name[0]) return;
        dev_ptrs d;
        if (extract(t, d)) m.emplace(t->name, d);
    };
    for (int i = 0; i < g->n_leafs; ++i) consider(g->leafs[i]);
    for (int i = 0; i < g->n_nodes; ++i) {
        consider(g->nodes[i]);
        for (int s = 0; s < GGML_MAX_SRC; ++s) consider(g->nodes[i]->src[s]);
    }
}

} // namespace

// R5 dispatch entry. Returns true iff it ran the megakernel (then the caller
// skips the forward). Layer 1: probe + log the pointer map, return false.
bool mk_dispatch(struct ggml_cgraph * cgraph) {
    static int logged = 0;
    std::map<std::string, dev_ptrs> ptrs;
    build_ptr_map(cgraph, ptrs);

    // Coverage of the load-bearing families the split program binds by name.
    if (logged < 2 && getenv("MK_DISPATCH_LOG")) {
        int qkv = 0, ffn_down = 0, ssm_out = 0, cache_k = 0, cache_r = 0, cache_s = 0, out_w = 0;
        for (auto & kv : ptrs) {
            const std::string & n = kv.first;
            if (n.find("attn_qkv.weight")  != std::string::npos) qkv++;
            if (n.find("ffn_down.weight")  != std::string::npos) ffn_down++;
            if (n.find("ssm_out.weight")   != std::string::npos) ssm_out++;
            if (n.rfind("cache_k_l", 0) == 0) cache_k++;
            if (n.rfind("cache_r_l", 0) == 0) cache_r++;
            if (n.rfind("cache_s_l", 0) == 0) cache_s++;
            if (n == "output.weight") out_w++;
        }
        fprintf(stderr,
            "[MK dispatch probe] graph nodes=%d leafs=%d ; meta-resolved tensors=%zu ; "
            "attn_qkv=%d ffn_down=%d ssm_out=%d cache_k=%d cache_r=%d cache_s=%d output.weight=%d\n",
            cgraph->n_nodes, cgraph->n_leafs, ptrs.size(),
            qkv, ffn_down, ssm_out, cache_k, cache_r, cache_s, out_w);
        ++logged;
    }

    // Layer 1: never claims the graph. Fall through to the R3 forward.
    return false;
}
