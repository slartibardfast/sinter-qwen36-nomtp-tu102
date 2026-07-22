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
#include "dual_core.h"          // mk_dual_setup / mk_dual_step (k0/harness.cpp)

#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

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

static const ggml_tensor * find_named(const ggml_cgraph * g, const char * nm) {
    for (int i = 0; i < g->n_leafs; ++i)
        if (g->leafs[i] && strcmp(g->leafs[i]->name, nm) == 0) return g->leafs[i];
    for (int i = 0; i < g->n_nodes; ++i) {
        if (g->nodes[i] && strcmp(g->nodes[i]->name, nm) == 0) return g->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s)
            if (g->nodes[i]->src[s] && strcmp(g->nodes[i]->src[s]->name, nm) == 0)
                return g->nodes[i]->src[s];
    }
    return nullptr;
}

// Read the first index of a (meta or plain) index tensor to host: the decode
// position lives in the SET_ROWS k_idxs (= [n_past]).
static int64_t read_index0(const ggml_tensor * t) {
    if (!t) return -1;
    void * p = nullptr; dev_ptrs d;
    if (extract(t, d)) p = d.p[0]; else p = t->data;
    if (!p) return -1;
    if (t->type == GGML_TYPE_I64) {
        int64_t v = -1; return cudaMemcpy(&v, p, 8, cudaMemcpyDeviceToHost) == cudaSuccess ? v : -1;
    }
    int32_t v = -1; return cudaMemcpy(&v, p, 4, cudaMemcpyDeviceToHost) == cudaSuccess ? (int64_t) v : -1;
}

// v2 count-class structural fingerprint (call/0024, cgraph/v2). A STRUCTURAL
// signature over op name, tensor type, op_params, and src topology, with tensor
// EXTENTS (ne) DELIBERATELY EXCLUDED. Because ne is excluded, this one hash is
// invariant to (a) sequence depth / n_kv, (b) the token/batch count U, and
// (c) n_ctx -- so decode (U=1) and prefill (U>1) of the served model share ONE
// fingerprint. The two masked axes that DO bind are checked separately by the
// caller: n_ctx as the KV-leaf extent, and the batch class (U==1 decode vs U>1
// prefill). op_params are position-invariant here (this fork writes KV via
// SET_ROWS with the row index as tensor data, never a view offset -- LG0 NOTES),
// so including them is safe. Node index = cgraph order; leaves keyed by
// first-encounter, type only. FNV-1a/64; a local exact-match key, not a digest.
static inline uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

static uint64_t mk_graph_fingerprint(const ggml_cgraph * g) {
    uint64_t h = 0xcbf29ce484222325ULL;
    std::map<const ggml_tensor *, int> node_idx;
    std::map<const ggml_tensor *, int> leaf_idx;
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * t = g->nodes[i];
        const char * opn = ggml_op_name(t->op);
        h = fnv1a(h, opn, strlen(opn));
        int ty = (int) t->type;
        h = fnv1a(h, &ty, sizeof ty);
        h = fnv1a(h, t->op_params, sizeof t->op_params);   // ne NOT hashed
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = t->src[s];
            if (!src) continue;
            const auto it = node_idx.find(src);
            if (it != node_idx.end()) {
                char c = 'n'; int r = it->second;
                h = fnv1a(h, &c, 1); h = fnv1a(h, &r, sizeof r);
            } else {
                const auto lt = leaf_idx.find(src);
                int k;
                if (lt != leaf_idx.end()) {
                    k = lt->second;
                } else {
                    k = (int) leaf_idx.size();
                    leaf_idx.emplace(src, k);
                    char c = 'L'; int lty = (int) src->type;   // leaf: type only, no ne
                    h = fnv1a(h, &c, 1); h = fnv1a(h, &lty, sizeof lty);
                }
                char c = 'l'; h = fnv1a(h, &c, 1); h = fnv1a(h, &k, sizeof k);
            }
        }
        node_idx.emplace(t, i);
    }
    return h;
}

} // namespace

// Diagnostic (MK_ZERO_STATE): zero llama's cache_k/v/r/s so a decode starts from
// genuinely zero recurrent state (isolates the meta-prefill->MK state handoff).
// Default stream — safe only with no resident MK kernel (the stock/compare path).
void mk_zero_state(struct ggml_cgraph * cgraph) {
    std::map<std::string, const ggml_tensor *> caches;
    auto consider = [&](const ggml_tensor * t) {
        if (!t || !t->name[0]) return;
        std::string n = t->name;
        if (n.find(' ') != std::string::npos) return;
        if (n.rfind("cache_k_l", 0) == 0 || n.rfind("cache_v_l", 0) == 0 ||
            n.rfind("cache_r_l", 0) == 0 || n.rfind("cache_s_l", 0) == 0)
            caches.emplace(n, t);
    };
    for (int i = 0; i < cgraph->n_leafs; ++i) consider(cgraph->leafs[i]);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        consider(cgraph->nodes[i]);
        for (int s = 0; s < GGML_MAX_SRC; ++s) consider(cgraph->nodes[i]->src[s]);
    }
    size_t n = 0;
    for (auto & kv : caches) {
        const ggml_tensor * s[2];
        if (!simple_pair(kv.second, s)) continue;
        for (int g = 0; g < 2; g++) { cudaSetDevice(g); cudaMemset(s[g]->data, 0, ggml_nbytes(s[g])); }
        n++;
    }
    cudaDeviceSynchronize();
    fprintf(stderr, "[MK zero-state] zeroed %zu cache tensors\n", n);
}

// Diagnostic: after a STOCK forward, read result_output's two vocab halves and
// print the global argmax, so it can be compared to the MK argmax on the same
// decode. Safe on the default stream (no resident MK kernel in compare mode).
void mk_stock_argmax(struct ggml_cgraph * cgraph) {
    if (cgraph->n_nodes < 3000) return;
    const ggml_tensor * st = find_named(cgraph, "MK#model.input_embed#0");
    if (!st || st->ne[1] != 1) return;
    int64_t pos = -1;
    for (int i = 0; i < cgraph->n_nodes; ++i)
        if (cgraph->nodes[i]->op == GGML_OP_SET_ROWS) { pos = read_index0(cgraph->nodes[i]->src[1]); break; }
    const ggml_tensor * out = cgraph->nodes[cgraph->n_nodes - 1];
    dev_ptrs od; if (!extract(out, od)) return;
    int64_t nv = out->ne[0], half = nv / 2;
    std::vector<float> h0(half), h1(half);
    if (cudaMemcpy(h0.data(), od.p[0], half * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    if (cudaMemcpy(h1.data(), od.p[1], half * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    int bg = 0; int64_t bi = 0; float bv = -1e30f;
    for (int64_t i = 0; i < half; i++) if (h0[i] > bv) { bv = h0[i]; bi = i; bg = 0; }
    for (int64_t i = 0; i < half; i++) if (h1[i] > bv) { bv = h1[i]; bi = i; bg = 1; }
    fprintf(stderr, "[stock argmax] pos=%lld tok=%lld val=%.3f\n",
            (long long) pos, (long long) (bg * half + bi), bv);
}

// R5 dispatch entry. Returns true iff it ran the megakernel (then the caller
// skips the forward). Layer 1: probe + log the pointer map, return false.
bool mk_dispatch(struct ggml_cgraph * cgraph) {
    static int logged = 0;
    std::map<std::string, dev_ptrs> ptrs;
    build_ptr_map(cgraph, ptrs);

    // v2 fingerprint measurement (MK_FP_LOG): log the structural hash + the two
    // separately-bound axes (n_ctx = KV-leaf extent, U = batch class) for every
    // graph that flows -- decode, prefill, K-shift, warmup. Non-destructive; the
    // gate below is unchanged in this phase.
    if (getenv("MK_FP_LOG")) {
        uint64_t fp = mk_graph_fingerprint(cgraph);
        const ggml_tensor * st = find_named(cgraph, "MK#model.input_embed#0");
        const ggml_tensor * ck = find_named(cgraph, "cache_k_l3");
        fprintf(stderr, "[MK fp] fp=%016llx nodes=%d n_ctx=%lld U=%lld\n",
                (unsigned long long) fp, cgraph->n_nodes,
                (long long) (ck ? ck->ne[1] : -1),
                (long long) (st ? st->ne[1] : -1));
    }

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
                                 "output.weight", "blk.0.ssm_in.weight", "token_embd.weight",
                                 "result_output", "MK#model.input_embed#0" })
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

    // Default (and any non-decode graph): forward. MK_DISPATCH_RUN opts into the
    // megakernel; the k=0 decode graph is the transformer stack (~3703 nodes).
    if (!getenv("MK_DISPATCH_RUN") || cgraph->n_nodes < 3000) return false;

    // Residual seed (CPU embed output), position (SET_ROWS k_idxs), output halves.
    auto seed = ptrs.find("MK#model.input_embed#0");
    if (seed == ptrs.end()) return false;
    // DECODE ONLY: the k=0 megakernel processes exactly one token. The residual
    // seed's ne[1] is the token count; forward prefill (ne[1] > 1) to stock so it
    // populates the KV the decode then reads.
    const ggml_tensor * seed_t = find_named(cgraph, "MK#model.input_embed#0");
    if (!seed_t || seed_t->ne[1] != 1) return false;
    int64_t pos = -1;
    for (int i = 0; i < cgraph->n_nodes; ++i)
        if (cgraph->nodes[i]->op == GGML_OP_SET_ROWS) {
            pos = read_index0(cgraph->nodes[i]->src[1]); break;
        }
    if (pos < 0) return false;
    const ggml_tensor * out = cgraph->nodes[cgraph->n_nodes - 1];
    dev_ptrs od;
    if (!extract(out, od)) return false;
    const ggml_tensor * ck = find_named(cgraph, "cache_k_l3");
    int64_t n_ctx   = ck ? ck->ne[1] : 8192;
    int64_t n_vocab = out->ne[0];

    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "[MK dispatch RUN] pos=%lld n_ctx=%lld n_vocab=%lld out=%s "
                "out.ne=[%lld,%lld] seed(%p,%p)\n", (long long) pos, (long long) n_ctx,
                (long long) n_vocab, out->name, (long long) out->ne[0], (long long) out->ne[1],
                seed->second.p[0], seed->second.p[1]);
        warned = true;
    }

    if (!mk_dual_ready()) {
        MkPtrMap pm;
        for (auto & kv : ptrs) pm.p[kv.first] = { kv.second.p[0], kv.second.p[1] };
        const char * prog = getenv("MK_PROGRAM");
        mk_dual_setup(pm, prog ? prog : "k0/program-split.json", n_ctx, n_vocab);
    }
    return mk_dual_step(seed->second.p[0], seed->second.p[1], pos, od.p[0], od.p[1]);
}
