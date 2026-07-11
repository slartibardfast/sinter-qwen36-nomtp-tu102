// oracle.cpp - G13 numerical-parity oracle dumper (plan/0135)
//
// Extends capture/g13-envelope/probe.cpp: loads the model at a given split
// mode, prefills exactly --ctx-tokens tokens (the probes' fixed base text,
// tiled/truncated, BOS once), then runs --steps greedy (fp32 first-max argmax,
// no sampler, no RNG) single-token decode steps. During each decode step
// (never the prefill) the backend-scheduler eval callback observes EVERY
// graph node (ask==true -> true) and, at ask==false, fetches the computed
// tensor with ggml_backend_tensor_get. Graph structure is not modified;
// observation only changes scheduler batching (one sync per node).
//
// Outputs, all under --out DIR:
//   logits.bin   raw f32, (steps+1) rows x n_vocab. Row 0 is the prefill
//                output row; row s+1 is the row produced by decode step s.
//                Token s is chosen greedily from row s.
//   tokens.txt   the steps emitted token ids, one per line.
//   nodes.csv    one row per observed node per decode step:
//                step,idx,name,op,type,ne0..ne3,summary,rms,mean,min,max,v0..v7
//                summary: ok = fp64 RMS/mean/min/max + first 8 values (flatten
//                order, i0 fastest); big = nbytes > 32 MiB, data not fetched
//                (the 262144-row KV-cache SET_ROWS nodes: fetching them would
//                read ~512 MB of mostly never-written cache per node); skip =
//                non-float type.
//   full/s<S>/<name>.bin    full raw f32 dump of the residual stream nodes
//                           l_out-<il> at decode step S.
//   state/s<S>/<name>.bin   full raw dump of every DeltaNet recurrent-state
//                           write at decode step S: the CPY nodes whose dst is
//                           a view of cache_s_l<il> (DeltaNet state) or
//                           cache_r_l<il> (conv state). Bytes as written by
//                           the engine, for bit-exact comparison.
//   files.csv    inventory of the .bin files: step,kind,name,path,op,type,
//                ne0..ne3,nbytes
//   index.json   deterministic run record: config, prompt tokens, emitted
//                tokens, layouts. No timestamps (trees must be byte-
//                comparable across runs).
//
// Serving config mirrored from capture.sh / fingerprint.cpp: -ngl 999, flash
// attention enabled, default (f16) KV cache; split mode and n_ctx from args.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

static void print_usage(const char * prog) {
    printf("\nusage:\n");
    printf("\n    %s --model PATH --split none|layer|tensor --out DIR [--n-ctx N] [--ctx-tokens N] [--steps N] [--tag STR]\n", prog);
    printf("\n");
}

// nodes larger than this are recorded metadata-only (summary = big). 8 MiB
// sits safely above every legitimate batch-1 activation (largest: the 3 MiB
// cache_s state write; logits ~1 MiB) and below the whole-KV-cache SET_ROWS
// nodes (16 MiB at n_ctx 8192, 512 MiB at 262144), whose bulk is never-written
// cache padding: summarizing it would be meaningless and a run-to-run
// determinism hazard.
static const size_t SUMMARY_CAP = 8u * 1024u * 1024u;

struct dump_ctx {
    int    step  = -1;      // -1 = not capturing (prefill)
    int    idx   = 0;       // node index within the current step
    bool   targeted = false; // true at multi-GPU splits: fetch only the safe
                             // node families (see oracle_cb_eval)
    FILE * nodes = nullptr; // nodes.csv
    FILE * files = nullptr; // files.csv
    std::string dir;        // out dir
    std::vector<uint8_t> buf;
    std::unordered_map<std::string, int> wcount; // per-step dup counter for dump names
    long   n_full  = 0;     // total full/ files written
    long   n_state = 0;     // total state/ files written
};

// residual stream node: name exactly l_out-<digits>
static bool is_l_out(const char * n) {
    if (strncmp(n, "l_out-", 6) != 0 || n[6] == '\0') {
        return false;
    }
    for (const char * p = n + 6; *p; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }
    return true;
}

// DeltaNet state writes (measured on the smoke run, engine 546eca8dc):
//   - ssm state:  CPY named "cache_s_l<il> (view) (copy of new_state-<il>)"
//                 (ggml_cpy_impl naming; dst view of cache_s_l<il>).
//   - conv state: CPY named "state_update_target-<il> (copy of last_conv_states-<il>)"
//                 — build_conv_state cb()-renames the cache_r_l<il> dst view
//                 to state_update_target-<il> before the cpy, so the cache
//                 name does NOT appear; the layer id in the name is il.
// Zero-extent bookkeeping CPYs ("cache_{s,r}_l<il> (view) (copy of )", ne1=0,
// the batch-absent-sequence copy path) are excluded via nelem > 0.
// Returns the dump base name in `base` (the physical cache bank).
static bool is_state_write(const struct ggml_tensor * t, char * base, size_t cap) {
    if (ggml_nelements(t) <= 0 || strstr(t->name, "(copy") == NULL) {
        return false;
    }
    if (strncmp(t->name, "cache_s_l", 9) == 0 || strncmp(t->name, "cache_r_l", 9) == 0) {
        size_t j = 0;
        for (const char * p = t->name; *p && *p != ' ' && j + 1 < cap; p++) {
            base[j++] = *p;
        }
        base[j] = '\0';
        return true;
    }
    if (strncmp(t->name, "state_update_target-", 20) == 0) {
        const int il = atoi(t->name + 20);
        snprintf(base, cap, "cache_r_l%d", il);
        return true;
    }
    return false;
}

static void csv_safe(const char * in, char * out, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < cap; i++) {
        char c = in[i];
        out[j++] = (c == ',' || c == '"' || c == '\n' || c == '\r') ? '_' : c;
    }
    out[j] = '\0';
}

// Nodes the tensor-split meta backend skips when planning subgraphs (views of
// host-buffer input tensors, ggml-backend-meta.cpp compute loop). A compute
// range must never END at one: the final subgraph would not be closed and
// GGML_ASSERT(i_start == cgraph->n_nodes) fires. So these are recorded
// metadata-only at ask time (summary = hostview) and never observed. They are
// views of inputs, not computed activations; no parity value is lost.
static bool is_meta_skipped(const struct ggml_tensor * t) {
    return t->view_src != NULL && t->view_src->op == GGML_OP_NONE &&
           t->view_src->buffer != NULL && ggml_backend_buffer_is_host(t->view_src->buffer);
}

// one nodes.csv row; stats may be null (no data fetched)
static void write_node_row(dump_ctx * d, const struct ggml_tensor * t, int idx,
                           const char * status,
                           const double * stats /* rms,mean,min,max */,
                           const float * first8, int n8) {
    char name[512];
    csv_safe(t->name, name, sizeof(name));
    fprintf(d->nodes, "%d,%d,%s,%s,%s,%lld,%lld,%lld,%lld,%s",
            d->step, idx, name, ggml_op_name(t->op), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3], status);
    if (stats != NULL) {
        fprintf(d->nodes, ",%.17g,%.17g,%.17g,%.17g", stats[0], stats[1], stats[2], stats[3]);
        for (int k = 0; k < 8; k++) {
            if (k < n8) {
                fprintf(d->nodes, ",%.9g", first8[k]);
            } else {
                fprintf(d->nodes, ",");
            }
        }
    } else {
        fprintf(d->nodes, ",,,,,,,,,,,,");
    }
    fprintf(d->nodes, "\n");
}

// the node families whose data the oracle needs (and which the g13-envelope
// acts probe already fetched safely under the tensor-split meta backend)
static bool is_family(const struct ggml_tensor * t, char * base, size_t cap) {
    return is_l_out(t->name) ||
           strncmp(t->name, "result_norm",  11) == 0 ||
           strncmp(t->name, "result_output", 13) == 0 ||
           is_state_write(t, base, cap);
}

// ggml_backend_sched_eval_callback. Fetch policy:
//   - split none (single device, plain CUDA buffers): fetch every float node
//     <= SUMMARY_CAP, except the meta-skipped host-view class.
//   - tensor split (targeted): the meta backend cannot service get_tensor on
//     arbitrary nodes (GGML_ASSERT(ggml_is_contiguous) at
//     ggml-backend-meta.cpp:1311; GGML_ABORT on SPLIT_AXIS_PARTIAL nodes, and
//     the split state is not queryable from outside). Fetch only the needed
//     families: l_out-*, result_norm, result_output and the DeltaNet state
//     writes; everything else is recorded metadata-only at ask time
//     (summary = meta), the fingerprint.cpp pattern.
static bool oracle_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;

    if (ask) {
        if (d->step < 0) {
            return false;
        }
        const int64_t nelem  = ggml_nelements(t);
        const size_t  nbytes = ggml_nbytes(t);
        const bool    is_f   = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
        char base[128];
        const bool fetchable = is_f && nelem > 0 && nbytes <= SUMMARY_CAP && !is_meta_skipped(t);
        const bool fetch     = fetchable && (!d->targeted || is_family(t, base, sizeof(base)));
        if (fetch) {
            return true; // row written at ask==false, with data
        }
        const char * status = is_meta_skipped(t) ? "hostview"
                            : nelem <= 0         ? "empty"
                            : !is_f              ? "skip"
                            : nbytes > SUMMARY_CAP ? "big"
                            : "meta";
        write_node_row(d, t, d->idx++, status, NULL, NULL, 0);
        return false;
    }
    if (d->step < 0) {
        return true;
    }

    // everything delivered here passed the fetch policy above
    const int     idx    = d->idx++;
    const size_t  nbytes = ggml_nbytes(t);
    const int64_t nelem  = ggml_nelements(t);

    double rms = 0.0, mean = 0.0, vmin = 0.0, vmax = 0.0;
    float  first8[8];
    int    n8 = 0;

    {
        d->buf.resize(nbytes);
        ggml_backend_tensor_get(t, d->buf.data(), 0, nbytes);
        const uint8_t * data = d->buf.data();

        double sum = 0.0, sum2 = 0.0;
        vmin =  INFINITY;
        vmax = -INFINITY;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        const size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
                        float v;
                        if (t->type == GGML_TYPE_F32) {
                            v = *(const float *) &data[i];
                        } else if (t->type == GGML_TYPE_F16) {
                            v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
                        } else {
                            v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
                        }
                        sum  += (double) v;
                        sum2 += (double) v * (double) v;
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                        if (n8 < 8) first8[n8++] = v;
                    }
                }
            }
        }
        rms  = sqrt(sum2 / (double) nelem);
        mean = sum / (double) nelem;
    }

    const double stats[4] = { rms, mean, vmin, vmax };
    write_node_row(d, t, idx, "ok", stats, first8, n8);

    // full-data dumps: residual stream + DeltaNet state writes
    char base[128];
    const char * kind = NULL;
    if (is_l_out(t->name)) {
        kind = "full";
        snprintf(base, sizeof(base), "%s", t->name);
    } else if (is_state_write(t, base, sizeof(base))) {
        kind = "state";
    }
    if (kind != NULL) {

        char fname[192];
        const std::string key = std::string(kind) + "/" + base;
        const int w = d->wcount[key]++;
        if (w == 0) {
            snprintf(fname, sizeof(fname), "%s.bin", base);
        } else {
            snprintf(fname, sizeof(fname), "%s-w%d.bin", base, w);
        }

        char rel[256];
        snprintf(rel, sizeof(rel), "%s/s%d/%s", kind, d->step, fname);
        const std::string path = d->dir + "/" + rel;
        FILE * f = fopen(path.c_str(), "wb");
        if (f == NULL) {
            fprintf(stderr, "error: cannot open %s\n", path.c_str());
            exit(1);
        }
        fwrite(d->buf.data(), 1, nbytes, f);
        fclose(f);

        fprintf(d->files, "%d,%s,%s,%s,%s,%s,%lld,%lld,%lld,%lld,%zu\n",
                d->step, kind, base, rel, ggml_op_name(t->op), ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3], nbytes);
        if (strcmp(kind, "full") == 0) d->n_full++; else d->n_state++;
    }

    return true;
}

// fixed deterministic base text, shared with probe.cpp / fingerprint.cpp
static const char * BASE_TEXT =
    "The measurement of numerical drift in large language models requires a fixed, "
    "deterministic input, and this paragraph serves as that input. It describes, in "
    "plain English, the reasons a probe might compare logits across hardware "
    "configurations: floating point addition is not associative, reduction order "
    "differs between kernels, and split strategies change the order in which partial "
    "sums are combined across devices. A careful engineer therefore records every "
    "output row, compares the rows pairwise between runs, and reports the largest "
    "divergence found anywhere in the sequence.";

static void mkdir_or_die(const std::string & p) {
    if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: cannot create dir %s\n", p.c_str());
        exit(1);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string split_str;
    std::string out_dir;
    std::string tag;
    int n_ctx      = 262144;
    int ctx_tokens = 64;
    int n_steps    = 32;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--split") == 0 && i + 1 < argc) {
            split_str = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--n-ctx") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-tokens") == 0 && i + 1 < argc) {
            ctx_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            n_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tag") == 0 && i + 1 < argc) {
            tag = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || split_str.empty() || out_dir.empty() || n_steps <= 0 || ctx_tokens < 1) {
        print_usage(argv[0]);
        return 1;
    }
    if (ctx_tokens + n_steps > n_ctx) {
        fprintf(stderr, "error: ctx-tokens + steps (%d) exceeds n-ctx (%d)\n", ctx_tokens + n_steps, n_ctx);
        return 1;
    }

    llama_split_mode split_mode;
    if (split_str == "none") {
        split_mode = LLAMA_SPLIT_MODE_NONE;
    } else if (split_str == "layer") {
        split_mode = LLAMA_SPLIT_MODE_LAYER;
    } else if (split_str == "tensor") {
        split_mode = LLAMA_SPLIT_MODE_TENSOR;
    } else {
        fprintf(stderr, "error: --split must be none|layer|tensor\n");
        return 1;
    }

    mkdir_or_die(out_dir);
    mkdir_or_die(out_dir + "/full");
    mkdir_or_die(out_dir + "/state");

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    model_params.split_mode   = split_mode;
    model_params.main_gpu     = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize the base text without specials, then tile: BOS once, then the
    // base tokens cyclically, truncated to exactly ctx_tokens (fingerprint.cpp)
    const int n_base = -llama_tokenize(vocab, BASE_TEXT, strlen(BASE_TEXT), NULL, 0, false, false);
    std::vector<llama_token> base(n_base);
    if (llama_tokenize(vocab, BASE_TEXT, strlen(BASE_TEXT), base.data(), base.size(), false, false) < 0) {
        fprintf(stderr, "error: failed to tokenize the base text\n");
        return 1;
    }
    std::vector<llama_token> prompt;
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        prompt.push_back(bos);
    }
    for (int i = 0; (int) prompt.size() < ctx_tokens; i++) {
        prompt.push_back(base[i % n_base]);
    }

    dump_ctx d;
    d.dir      = out_dir;
    d.targeted = split_mode != LLAMA_SPLIT_MODE_NONE;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx             = n_ctx;
    ctx_params.n_threads         = 8;
    ctx_params.n_threads_batch   = 8;
    ctx_params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED; // capture.sh: -fa on
    // type_k/type_v left at default (f16), as in capture.sh
    ctx_params.cb_eval           = oracle_cb_eval;
    ctx_params.cb_eval_user_data = &d;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "error: failed to create the llama_context\n");
        return 1;
    }

    printf("split=%s n_ctx=%d n_vocab=%d ctx_tokens=%d steps=%d\n",
           split_str.c_str(), n_ctx, n_vocab, (int) prompt.size(), n_steps);

    FILE * f_logits = fopen((out_dir + "/logits.bin").c_str(), "wb");
    FILE * f_tokens = fopen((out_dir + "/tokens.txt").c_str(), "w");
    d.nodes = fopen((out_dir + "/nodes.csv").c_str(), "w");
    d.files = fopen((out_dir + "/files.csv").c_str(), "w");
    if (!f_logits || !f_tokens || !d.nodes || !d.files) {
        fprintf(stderr, "error: failed to open output files in %s\n", out_dir.c_str());
        return 1;
    }
    fprintf(d.nodes, "step,idx,name,op,type,ne0,ne1,ne2,ne3,summary,rms,mean,min,max,v0,v1,v2,v3,v4,v5,v6,v7\n");
    fprintf(d.files, "step,kind,name,path,op,type,ne0,ne1,ne2,ne3,nbytes\n");

    // prefill in n_batch-sized chunks (capture off: d.step == -1)
    const int n_batch = 2048; // llama_context_default_params().n_batch
    for (int i = 0; i < (int) prompt.size(); i += n_batch) {
        const int n = std::min(n_batch, (int) prompt.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(prompt.data() + i, n))) {
            fprintf(stderr, "error: llama_decode failed on prefill at %d\n", i);
            return 1;
        }
    }

    // logits row 0: the prefill output row
    {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == NULL) {
            fprintf(stderr, "error: no logits after prefill\n");
            return 1;
        }
        fwrite(logits, sizeof(float), n_vocab, f_logits);
        fflush(f_logits);
    }

    std::vector<llama_token> emitted;
    for (int step = 0; step < n_steps; step++) {
        const float * logits = llama_get_logits_ith(ctx, -1); // row `step`
        // plain fp32 argmax, first max wins (probe.cpp)
        llama_token tok = 0;
        float max = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > max) {
                max = logits[i];
                tok = i;
            }
        }
        emitted.push_back(tok);
        fprintf(f_tokens, "%d\n", tok);
        fflush(f_tokens);

        char piece[128];
        const int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        printf("step %3d: token %6d '%.*s'\n", step, tok, n < 0 ? 0 : n, piece);
        fflush(stdout);

        char sub[512];
        snprintf(sub, sizeof(sub), "%s/full/s%d", out_dir.c_str(), step);
        mkdir_or_die(sub);
        snprintf(sub, sizeof(sub), "%s/state/s%d", out_dir.c_str(), step);
        mkdir_or_die(sub);

        d.step = step;
        d.idx  = 0;
        d.wcount.clear();

        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            fprintf(stderr, "error: llama_decode failed at step %d\n", step);
            return 1;
        }

        fprintf(stderr, "step %d: nodes=%d\n", step, d.idx);
        d.step = -1;

        // row step+1: the logits produced by this decode
        const float * out = llama_get_logits_ith(ctx, -1);
        if (out == NULL) {
            fprintf(stderr, "error: no logits at step %d\n", step);
            return 1;
        }
        fwrite(out, sizeof(float), n_vocab, f_logits);
        fflush(f_logits);
    }

    fclose(f_logits);
    fclose(f_tokens);
    fclose(d.nodes);
    fclose(d.files);

    // deterministic index.json: config + tokens only, no timestamps/paths
    FILE * f_idx = fopen((out_dir + "/index.json").c_str(), "w");
    if (f_idx == NULL) {
        fprintf(stderr, "error: cannot open index.json\n");
        return 1;
    }
    fprintf(f_idx, "{\n");
    fprintf(f_idx, "  \"format\": \"mk-oracle/v1\",\n");
    fprintf(f_idx, "  \"tag\": \"%s\",\n", tag.c_str());
    fprintf(f_idx, "  \"config\": {\n");
    fprintf(f_idx, "    \"split\": \"%s\",\n", split_str.c_str());
    fprintf(f_idx, "    \"n_ctx\": %d,\n", n_ctx);
    fprintf(f_idx, "    \"ctx_tokens\": %d,\n", (int) prompt.size());
    fprintf(f_idx, "    \"steps\": %d,\n", n_steps);
    fprintf(f_idx, "    \"n_vocab\": %d,\n", n_vocab);
    fprintf(f_idx, "    \"flash_attn\": true,\n");
    fprintf(f_idx, "    \"kv_type\": \"f16\",\n");
    fprintf(f_idx, "    \"n_gpu_layers\": 999,\n");
    fprintf(f_idx, "    \"capture\": \"%s\"\n", d.targeted ? "targeted" : "full");
    fprintf(f_idx, "  },\n");
    fprintf(f_idx, "  \"logits\": { \"file\": \"logits.bin\", \"dtype\": \"f32\", \"rows\": %d, \"cols\": %d,\n", n_steps + 1, n_vocab);
    fprintf(f_idx, "    \"note\": \"row 0 = prefill output; row s+1 = output of decode step s; token s chosen greedily from row s\" },\n");
    fprintf(f_idx, "  \"nodes_csv\": \"nodes.csv\",\n");
    fprintf(f_idx, "  \"files_csv\": \"files.csv\",\n");
    fprintf(f_idx, "  \"n_full_files\": %ld,\n", d.n_full);
    fprintf(f_idx, "  \"n_state_files\": %ld,\n", d.n_state);
    fprintf(f_idx, "  \"prompt_tokens\": [");
    for (size_t i = 0; i < prompt.size(); i++) {
        fprintf(f_idx, "%s%d", i ? "," : "", prompt[i]);
    }
    fprintf(f_idx, "],\n");
    fprintf(f_idx, "  \"emitted_tokens\": [");
    for (size_t i = 0; i < emitted.size(); i++) {
        fprintf(f_idx, "%s%d", i ? "," : "", emitted[i]);
    }
    fprintf(f_idx, "]\n");
    fprintf(f_idx, "}\n");
    fclose(f_idx);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
