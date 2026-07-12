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
#include <cstring>

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

// Simple (per-GPU) tensors behind a meta tensor, for geometry inspection.
static bool simple_pair(const ggml_tensor * t, const ggml_tensor * out[2]) {
    if (!t || !t->buffer || !ggml_backend_buffer_is_meta(t->buffer)) return false;
    auto * mc = (meta_ctx_mirror *) t->buffer->context;
    auto it = mc->simple_tensors.find(t);
    if (it == mc->simple_tensors.end() || it->second.size() < 2) return false;
    out[0] = it->second[0]; out[1] = it->second[1];
    return out[0] && out[1];
}

static void log_geom1(const char * tag, const ggml_tensor * t) {
    if (!t) { fprintf(stderr, "  %-22s <null>\n", tag); return; }
    fprintf(stderr, "  %-22s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
            tag, (int) t->type,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}

// For a meta tensor: log the meta geometry + each per-GPU simple tensor's geometry.
// This is the L3 layout de-risk: the megakernel's fattn/ssm ops assume the harness
// cache layout; here we read llama's actual cache_k/r/s (and a weight slice) shape.
static void log_geom(const ggml_cgraph * g, const char * name) {
    auto find = [&](const char * nm) -> const ggml_tensor * {
        for (int i = 0; i < g->n_leafs; ++i)
            if (g->leafs[i] && strcmp(g->leafs[i]->name, nm) == 0) return g->leafs[i];
        for (int i = 0; i < g->n_nodes; ++i) {
            if (g->nodes[i] && strcmp(g->nodes[i]->name, nm) == 0) return g->nodes[i];
            for (int s = 0; s < GGML_MAX_SRC; ++s)
                if (g->nodes[i]->src[s] && strcmp(g->nodes[i]->src[s]->name, nm) == 0)
                    return g->nodes[i]->src[s];
        }
        return nullptr;
    };
    const ggml_tensor * t = find(name);
    if (!t) { fprintf(stderr, "[MK geom] %-18s NOT FOUND\n", name); return; }
    fprintf(stderr, "[MK geom] %s\n", name);
    log_geom1("meta", t);
    const ggml_tensor * s[2];
    if (simple_pair(t, s)) { log_geom1("gpu0", s[0]); log_geom1("gpu1", s[1]); }
    else fprintf(stderr, "  (not a 2-way meta tensor)\n");
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
        // L3 layout de-risk: dump llama's actual KV + weight-slice geometry so we
        // can compare against the harness's assumed cache layout (cache_k =
        // n_ctx x N_EMBD_GQA/2 f16; conv/ssm state; qkv/ssm_out/output slices).
        for (const char * nm : { "cache_k_l3", "cache_v_l3", "cache_r_l0", "cache_s_l0",
                                 "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight",
                                 "output.weight", "blk.0.ssm_in.weight", "token_embd.weight" })
            log_geom(cgraph, nm);
        // Embed-seed question: dump the leaves (inputs) and node[0..2] so we can
        // see whether MK's graph starts from the token (embed inside) or the
        // CPU-computed residual (embed excluded -> seed the residual).
        fprintf(stderr, "[MK inputs] %d leafs:\n", cgraph->n_leafs);
        for (int i = 0; i < cgraph->n_leafs && i < 40; ++i) {
            const ggml_tensor * t = cgraph->leafs[i];
            fprintf(stderr, "  leaf[%d] %-28s op=%d type=%d ne=[%lld,%lld] %s\n", i,
                    t->name, (int) t->op, (int) t->type,
                    (long long) t->ne[0], (long long) t->ne[1],
                    (t->buffer && ggml_backend_buffer_is_meta(t->buffer)) ? "META" : "host?");
        }
        for (int i = 0; i < 3 && i < cgraph->n_nodes; ++i) {
            const ggml_tensor * t = cgraph->nodes[i];
            fprintf(stderr, "[MK node%d] %-28s op=%d src0=%s src1=%s\n", i, t->name, (int) t->op,
                    t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
        }
        // Position sourcing (risk #4): find any tensor whose name mentions pos /
        // inp / idx, plus the input residual, and report its type+shape+location.
        { std::map<std::string,const ggml_tensor*> hits;
          auto note=[&](const ggml_tensor*t){ if(!t||!t->name[0])return; std::string n=t->name;
            if(n.find("pos")!=std::string::npos||n.rfind("inp",0)==0||n.find("idx")!=std::string::npos
               ||n.find("input_embed")!=std::string::npos) hits.emplace(n,t); };
          for(int i=0;i<cgraph->n_leafs;++i) note(cgraph->leafs[i]);
          for(int i=0;i<cgraph->n_nodes;++i){ note(cgraph->nodes[i]);
            for(int s=0;s<GGML_MAX_SRC;++s) note(cgraph->nodes[i]->src[s]); }
          for(auto&kv:hits){ const ggml_tensor*t=kv.second;
            fprintf(stderr,"[MK posrc] %-30s op=%d type=%d ne=[%lld,%lld] %s\n",
              kv.first.c_str(),(int)t->op,(int)t->type,(long long)t->ne[0],(long long)t->ne[1],
              (t->buffer?(ggml_backend_buffer_is_meta(t->buffer)?"META":"nonmeta"):"nobuf")); } }
        ++logged;
    }

    // Layer 1: never claims the graph. Fall through to the R3 forward.
    return false;
}
