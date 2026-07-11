// mk-harness — standalone parity/bench harness for the k=0 persistent
// megakernel. No llama-server, no ggml: a GGUF mmap loader (k0/gguf.h), a
// single-GPU model-residency uploader, the program packer shell, and the
// two drivers (--parity against a tests/oracle mk-oracle/v1 tree, --bench).
//
// Modes:
//   mk-harness [--validate]            loader validation (default mode):
//                                      enumerate + type totals + upload +
//                                      readback checksums + pack dry-run
//   mk-harness --parity REF_DIR        decode-only parity vs an oracle tree
//                                      (e.g. /var/tmp/mk-oracle/ref-tensor-ctx64)
//   mk-harness --bench N               sustained decode of N tokens, tok/s
// Options: --model PATH (default /opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf)
//          --n-ctx N (default 8192)  --gpu I (default 0)
//          --program PATH (default k0/program.json)  --out DIR (parity dump)
// Run from the repo root (relative defaults assume it).
//
// Coordination notes (parallel agents; state as of writing, 2026-07-11):
//   - core/host.h did NOT exist yet. The "LAUNCHER STAND-IN" section below
//     is this harness's own minimal upload/run_pass notion and duplicates
//     what core/host.cpp will own. When core/host.h lands, port the
//     stand-in to its API and delete the duplication.
//   - k0/ops/*.cuh did NOT exist yet: every per-kind packer in the PACKER
//     table is a stub that fails loudly at pack time naming the kind.
//     Replace each stub with a real packer against the op's Args struct.
//   - k0/compile_schedule.py / k0/program.json did NOT exist yet. If
//     program.json is absent the harness tries to generate it by running
//     python3 k0/compile_schedule.py; failing that it stubs the buffer
//     table (one 64 MiB "scratch" arena + the named IO buffers below) and
//     skips packing with a note. Expected program.json schema:
//       { "epoch_stride": N,
//         "buffers": [ {"name": "...", "bytes": N}, ... ],
//         "instrs":  [ {"kind": "MMVQ_Q4_0", "block_lo": 0, "block_hi": 72,
//                       "flags": 0, "dbg_node": 123, "args": {...}}, ... ] }
//
// Semantics pinned from the docs:
//   - Positions: the graph feeds ROPE an i32[4] (4 M-RoPE position ids per
//     token; docs/BLOCKS.md l57). For text-only decode all four are set to
//     the sequence position: docs/SEMANTICS.md §11 — sections are
//     [11,11,10,0] and the imrope sector%3 routing only ever reads the
//     first three axes (the 4th id is dead weight with sections[3]=0).
//   - Mask: f16, n_kv = pad(n_past+1, 256) (the FA classes of 256,
//     SEMANTICS.md §15); entry j = 0.0 for j <= n_past, -inf for the
//     padding tail. Rebuilt per pass into the "mask_f16" buffer.
//   - KV append row: SET_ROWS carries the destination row as tensor DATA
//     (i64, shared k_idxs/v_idxs leaf) = the token position ("kv_row").
//   - Recurrent state: one row per bank (ring slot 0). The graph's rs row
//     indices (l6 = 0, l7 = empty) always select row 0, so the banks are
//     allocated as exactly one snapshot row: conv_state_l<il> 30720 f32,
//     ssm_state_l<il> 786432 f32 (leaves.csv).
//   - blk.64.* (the nextn/MTP head) is enumerated and counted but NOT
//     uploaded: it is absent from the k=0 decode graph (BLOCKS.md, MTP off).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>

#include "../core/isa.cuh"
#include "gguf.h"

#define CUDA_CHECK(expr) do { \
    cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(2); \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Model shape constants (Qwen3.6-27B hybrid; docs/BLOCKS.md)
// ---------------------------------------------------------------------------

static const int   N_LAYER      = 64;          // decode-graph blocks (blk.64 = MTP, excluded)
static const int   N_EMBD       = 5120;
static const int   N_EMBD_GQA   = 1024;        // 4 kv heads x 256 per attention layer
static const int   CONV_STATE_N = 30720;       // f32 per DeltaNet block (10240 x 3)
static const int   SSM_STATE_N  = 786432;      // f32 per DeltaNet block (128 x 128 x 48)
static const int   N_LOUT       = 63;          // l_out-0..62 (block 63 has no l_out node)

static bool is_attn_layer(int il) { return il % 4 == 3; }

// Loader-validation ground truth (k0 inventory; decimal MB over ALL
// tensors including blk.64). NOTE: the briefed figure was 867 tensors, but
// the GGUF header itself records 866 (independently confirmed by a raw
// struct read of the header, 2026-07-11) and all four per-type byte totals
// match the briefed figures exactly at 866; the 851 non-blk.64 tensors
// also exactly match the 851 distinct weight names k0/leaves.csv
// references. 866 is the verified ground truth.
static const uint64_t EXPECT_N_TENSORS = 866;
struct ExpectTotal { uint32_t type; double mb; };
static const ExpectTotal EXPECT_TOTALS[] = {
    { gguf::T_Q4_0,      13043.96 },
    { gguf::T_Q4_0_AR16,   943.72 },
    { gguf::T_F16,        5237.64 },
    { gguf::T_F32,          10.69 },
};

// ---------------------------------------------------------------------------
// Tiny JSON parser (for oracle index.json and k0/program.json)
// ---------------------------------------------------------------------------

struct Jv {
    enum K { NUL, BOO, NUM, STR, ARR, OBJ } k = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Jv> arr;
    std::vector<std::pair<std::string, Jv>> obj;

    const Jv *get(const std::string &key) const {
        if (k != OBJ) return nullptr;
        for (auto &p : obj) if (p.first == key) return &p.second;
        return nullptr;
    }
    const Jv &at(const std::string &key) const {
        const Jv *v = get(key);
        if (!v) throw std::runtime_error("json: missing key '" + key + "'");
        return *v;
    }
    int64_t as_i() const {
        if (k != NUM) throw std::runtime_error("json: not a number");
        return (int64_t) num;
    }
};

struct JsonParser {
    const char *p, *end;
    explicit JsonParser(const std::string &s) : p(s.data()), end(s.data() + s.size()) {}
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++; }
    char peek() { ws(); if (p >= end) throw std::runtime_error("json: unexpected EOF"); return *p; }
    void expect(char c) { if (peek() != c) throw std::runtime_error(std::string("json: expected '") + c + "'"); p++; }
    std::string str() {
        expect('"');
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') {
                p++;
                if (p >= end) break;
                switch (*p) {
                    case 'n': s += '\n'; break; case 't': s += '\t'; break;
                    case 'r': s += '\r'; break; case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'u': { // ASCII-only decode; non-ASCII becomes '?'
                        if (end - p >= 5) {
                            unsigned v = (unsigned) strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16);
                            s += (v < 128) ? (char) v : '?';
                            p += 4;
                        }
                        break;
                    }
                    default: s += *p; break;
                }
                p++;
            } else s += *p++;
        }
        if (p >= end) throw std::runtime_error("json: unterminated string");
        p++;
        return s;
    }
    Jv value() {
        char c = peek();
        Jv v;
        if (c == '{') {
            v.k = Jv::OBJ; p++;
            if (peek() == '}') { p++; return v; }
            while (true) {
                std::string key = str();
                expect(':');
                v.obj.emplace_back(key, value());
                char d = peek();
                if (d == ',') { p++; continue; }
                expect('}');
                break;
            }
        } else if (c == '[') {
            v.k = Jv::ARR; p++;
            if (peek() == ']') { p++; return v; }
            while (true) {
                v.arr.push_back(value());
                char d = peek();
                if (d == ',') { p++; continue; }
                expect(']');
                break;
            }
        } else if (c == '"') {
            v.k = Jv::STR; v.str = str();
        } else if (c == 't') { v.k = Jv::BOO; v.b = true;  p += 4; }
        else if (c == 'f')   { v.k = Jv::BOO; v.b = false; p += 5; }
        else if (c == 'n')   { v.k = Jv::NUL; p += 4; }
        else {
            v.k = Jv::NUM;
            char *q = nullptr;
            v.num = strtod(p, &q);
            if (q == p) throw std::runtime_error("json: bad number");
            p = q;
        }
        return v;
    }
};

static bool read_file(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    size_t rd = fread(&out[0], 1, (size_t) n, f);
    fclose(f);
    return rd == (size_t) n;
}

static Jv json_load(const std::string &path) {
    std::string text;
    if (!read_file(path, text)) throw std::runtime_error("cannot read " + path);
    JsonParser jp(text);
    return jp.value();
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static uint64_t fnv1a(uint64_t h, const void *data, size_t n) {
    const uint8_t *b = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static const uint64_t FNV_BASIS = 1469598103934665603ull;

static void mkdir_p(const std::string &path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty()) mkdir(cur.c_str(), 0755);
            if (i < path.size()) cur += '/';
        } else cur += path[i];
    }
}

static int64_t pad_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

static const uint16_t F16_ZERO    = 0x0000;
static const uint16_t F16_NEG_INF = 0xFC00;

struct Stats { double rms, mean, mn, mx; float v[8]; size_t n; };
static Stats stats_of(const float *x, size_t n) {
    Stats s{};
    double sq = 0, sum = 0, mn = x[0], mx = x[0];
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        sq += v * v; sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    s.rms = std::sqrt(sq / (double) n);
    s.mean = sum / (double) n;
    s.mn = mn; s.mx = mx; s.n = n;
    for (int i = 0; i < 8; i++) s.v[i] = (size_t) i < n ? x[i] : 0.0f;
    return s;
}

// ---------------------------------------------------------------------------
// Device residency: weights + runtime buffers + name resolver
// ---------------------------------------------------------------------------

struct DevBuf { void *ptr = nullptr; size_t bytes = 0; };

struct Resolver {
    std::map<std::string, DevBuf> table;
    void add(const std::string &name, void *ptr, size_t bytes) {
        if (table.count(name)) throw std::runtime_error("resolver: duplicate name " + name);
        table[name] = { ptr, bytes };
    }
    // "name" or "name+<byte offset>"
    void *resolve(const std::string &spec) const {
        size_t plus = spec.rfind('+');
        std::string name = spec;
        size_t off = 0;
        if (plus != std::string::npos && plus > 0 &&
            spec.find_first_not_of("0123456789", plus + 1) == std::string::npos) {
            name = spec.substr(0, plus);
            off = (size_t) strtoull(spec.c_str() + plus + 1, nullptr, 10);
        }
        auto it = table.find(name);
        if (it == table.end()) throw std::runtime_error("resolver: unknown name '" + name + "'");
        if (off >= it->second.bytes && it->second.bytes > 0)
            throw std::runtime_error("resolver: offset past end of '" + name + "'");
        return (char *) it->second.ptr + off;
    }
};

struct Residency {
    gguf::File gg;
    void *arena = nullptr;
    size_t arena_bytes = 0;
    std::map<std::string, DevBuf> weights;      // GGUF name -> device
    uint64_t n_skipped = 0, skipped_bytes = 0;  // blk.64.*
    int64_t n_vocab = 0;

    // Enumerate + validate against the k0 inventory. Returns false on any
    // count/total mismatch (message already printed).
    bool enumerate_and_validate(const std::string &model_path) {
        gg.open(model_path);
        printf("model: %s\n", model_path.c_str());
        printf("  gguf v%u, %zu tensors, %zu kv entries, alignment %llu, data %.2f MB\n",
               gg.version, gg.tensors.size(), gg.kv.size(),
               (unsigned long long) gg.alignment, gg.data_size / 1e6);

        std::string arch = "?";
        if (const gguf::Value *a = gg.find_kv("general.architecture")) arch = a->s;
        printf("  general.architecture = %s\n", arch.c_str());
        if (const gguf::Value *v = gg.find_kv(arch + ".rope.dimension_sections")) {
            printf("  %s.rope.dimension_sections = [", arch.c_str());
            for (size_t i = 0; i < v->arr_i.size(); i++)
                printf("%s%lld", i ? "," : "", (long long) v->arr_i[i]);
            printf("]\n");
        }
        for (const char *k : { "attention.layer_norm_rms_epsilon", "rope.freq_base", "context_length" }) {
            if (const gguf::Value *v = gg.find_kv(arch + "." + k)) {
                if (v->kind == gguf::VK_F32 || v->kind == gguf::VK_F64)
                    printf("  %s.%s = %g\n", arch.c_str(), k, v->f);
                else
                    printf("  %s.%s = %lld\n", arch.c_str(), k, (long long) v->i);
            }
        }

        const gguf::TensorInfo *emb = gg.find("token_embd.weight");
        if (!emb) { fprintf(stderr, "FAIL: token_embd.weight not found\n"); return false; }
        n_vocab = (int64_t) emb->ne[1];
        printf("  n_vocab = %lld (token_embd.weight ne1)\n", (long long) n_vocab);

        std::map<uint32_t, std::pair<uint64_t, uint64_t>> per_type; // type -> (count, bytes)
        for (auto &t : gg.tensors) {
            auto &e = per_type[t.type];
            e.first++;
            e.second += t.nbytes;
        }
        printf("  per-type totals (all tensors incl. blk.64):\n");
        bool ok = true;
        for (auto &pt : per_type) {
            const gguf::TypeTraits *tt = gguf::type_traits(pt.first);
            double mb = pt.second.second / 1e6;
            const ExpectTotal *exp = nullptr;
            for (auto &e : EXPECT_TOTALS) if (e.type == pt.first) exp = &e;
            if (exp) {
                bool match = std::fabs(mb - exp->mb) < 0.01;
                printf("    %-10s %4llu tensors  %14llu B  %10.2f MB  (expect %10.2f MB) %s\n",
                       tt->name, (unsigned long long) pt.second.first,
                       (unsigned long long) pt.second.second, mb, exp->mb,
                       match ? "OK" : "MISMATCH");
                if (!match) ok = false;
            } else {
                printf("    %-10s %4llu tensors  %14llu B  %10.2f MB  (UNEXPECTED TYPE)\n",
                       tt->name, (unsigned long long) pt.second.first,
                       (unsigned long long) pt.second.second, mb);
                ok = false;
            }
        }
        if (gg.tensors.size() != EXPECT_N_TENSORS) {
            printf("  tensor count %zu != expected %llu: MISMATCH\n",
                   gg.tensors.size(), (unsigned long long) EXPECT_N_TENSORS);
            ok = false;
        } else {
            printf("  tensor count %zu == expected %llu: OK\n",
                   gg.tensors.size(), (unsigned long long) EXPECT_N_TENSORS);
        }
        return ok;
    }

    size_t upload_bytes_needed() const {
        size_t total = 0;
        for (auto &t : gg.tensors) {
            if (t.name.rfind("blk.64.", 0) == 0) continue;
            total += pad_up((int64_t) t.nbytes, 256);
        }
        return total;
    }

    void upload(Resolver &R) {
        arena_bytes = upload_bytes_needed();
        printf("upload: decode-graph weights %.2f GB to one GPU (blk.64.* = MTP head skipped: "
               "absent from the k=0 decode graph, BLOCKS.md)\n", arena_bytes / 1e9);
        CUDA_CHECK(cudaMalloc(&arena, arena_bytes));
        size_t cursor = 0;
        for (auto &t : gg.tensors) {
            if (t.name.rfind("blk.64.", 0) == 0) {
                n_skipped++;
                skipped_bytes += t.nbytes;
                continue;
            }
            void *dst = (char *) arena + cursor;
            CUDA_CHECK(cudaMemcpy(dst, t.data, t.nbytes, cudaMemcpyHostToDevice));
            weights[t.name] = { dst, t.nbytes };
            R.add(t.name, dst, t.nbytes);
            cursor += pad_up((int64_t) t.nbytes, 256);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  uploaded %zu tensors (%.2f GB); skipped %llu blk.64 tensors (%.2f MB)\n",
               weights.size(), arena_bytes / 1e9,
               (unsigned long long) n_skipped, skipped_bytes / 1e6);
    }

    // Readback checksum spot-checks: FNV-1a64 of the mmap bytes vs the
    // device copy fetched back in 32 MiB chunks.
    bool checksum_spot_checks() {
        const char *names[] = {
            "token_embd.weight",        // f16, 2.5 GB, file head
            "blk.0.attn_qkv.weight",    // q4_0
            "blk.0.ssm_out.weight",     // q4_0_ar16
            "output_norm.weight",       // f32
            "blk.63.ffn_down.weight",   // q4_0, near file tail (offset math check)
        };
        const size_t CHUNK = 32ull << 20;
        std::vector<uint8_t> host(CHUNK);
        bool all_ok = true;
        printf("checksum spot-checks (FNV-1a64, host mmap vs device readback):\n");
        for (const char *name : names) {
            const gguf::TensorInfo *t = gg.find(name);
            if (!t || !weights.count(name)) {
                printf("  %-26s MISSING\n", name);
                all_ok = false;
                continue;
            }
            uint64_t h_host = fnv1a(FNV_BASIS, t->data, t->nbytes);
            uint64_t h_dev = FNV_BASIS;
            const char *src = (const char *) weights[name].ptr;
            for (size_t off = 0; off < t->nbytes; off += CHUNK) {
                size_t n = t->nbytes - off < CHUNK ? t->nbytes - off : CHUNK;
                CUDA_CHECK(cudaMemcpy(host.data(), src + off, n, cudaMemcpyDeviceToHost));
                h_dev = fnv1a(h_dev, host.data(), n);
            }
            bool match = h_host == h_dev;
            printf("  %-26s %-9s %12llu B  host %016llx dev %016llx  %s\n",
                   name, gguf::type_traits(t->type)->name, (unsigned long long) t->nbytes,
                   (unsigned long long) h_host, (unsigned long long) h_dev,
                   match ? "OK" : "MISMATCH");
            if (!match) all_ok = false;
        }
        return all_ok;
    }
};

// Per-pass host-visible context, mirrored to the "pass_ctx" device buffer
// every pass (how ops get per-pass scalars like n_kv that cannot live in
// the once-packed Instr payloads).
struct PassCtx {
    int32_t token;
    int32_t pos[4];
    int32_t n_past;
    int32_t n_kv;
    int64_t kv_row;
};

struct Runtime {
    int64_t n_ctx = 8192;
    int64_t n_vocab = 0;
    int64_t mask_cap = 0;   // pad(n_ctx, 256) f16 entries
    std::map<std::string, DevBuf> bufs;
    bool buffer_table_stubbed = false;

    void *alloc(Resolver &R, const std::string &name, size_t bytes, bool zero = true) {
        void *p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes ? bytes : 4));
        if (zero) CUDA_CHECK(cudaMemset(p, 0, bytes ? bytes : 4));
        bufs[name] = { p, bytes };
        R.add(name, p, bytes);
        return p;
    }

    size_t bytes_needed() const {
        size_t kv = 16ull * 2 * (size_t) n_ctx * N_EMBD_GQA * 2;
        size_t st = 48ull * (CONV_STATE_N + SSM_STATE_N) * 4;
        size_t io = (size_t) pad_up(n_ctx, 256) * 2 + (size_t) n_vocab * 4
                  + (size_t) N_LOUT * N_EMBD * 4 + N_EMBD * 4 + 4096;
        return kv + st + io + (64ull << 20);
    }

    void allocate(Resolver &R, const Jv *program) {
        // KV caches: 16 attention layers, [n_ctx rows x 1024] f16 per K and V.
        for (int il = 0; il < N_LAYER; il++) {
            if (!is_attn_layer(il)) continue;
            size_t b = (size_t) n_ctx * N_EMBD_GQA * 2;
            alloc(R, "cache_k_l" + std::to_string(il), b);
            alloc(R, "cache_v_l" + std::to_string(il), b);
        }
        // DeltaNet state banks, one snapshot row each (ring slot 0).
        for (int il = 0; il < N_LAYER; il++) {
            if (is_attn_layer(il)) continue;
            alloc(R, "conv_state_l" + std::to_string(il), (size_t) CONV_STATE_N * 4);
            alloc(R, "ssm_state_l" + std::to_string(il), (size_t) SSM_STATE_N * 4);
        }
        // Per-pass IO.
        alloc(R, "token_id", 4);
        alloc(R, "positions", 16);          // i32[4] M-RoPE ids, all = seq pos
        alloc(R, "kv_row", 8);              // i64[1], shared k_idxs/v_idxs
        alloc(R, "rs_row", 4);              // i32[1] = 0 (ring slot 0)
        alloc(R, "rs_clear", 4);            // i32[0] clear list (allocated, count 0)
        alloc(R, "out_row", 4);             // i32[1] = 0 (batch-1)
        alloc(R, "pass_ctx", sizeof(PassCtx));
        mask_cap = pad_up(n_ctx, 256);
        alloc(R, "mask_f16", (size_t) mask_cap * 2);
        alloc(R, "logits", (size_t) n_vocab * 4);
        alloc(R, "result_norm", (size_t) N_EMBD * 4);
        // Parity dual-write target: row il = residual after block il
        // (0..62). The schedule must alias/copy each block's residual add
        // into this buffer for the parity dump to carry real data.
        alloc(R, "dbg_lout", (size_t) N_LOUT * N_EMBD * 4);

        if (program && program->get("buffers")) {
            const Jv &tbl = program->at("buffers");
            for (auto &b : tbl.arr)
                alloc(R, b.at("name").str, (size_t) b.at("bytes").as_i());
            printf("buffers: %zu scratch buffers from program.json buffer table\n", tbl.arr.size());
        } else {
            // STUB: k0/program.json's buffer table was not available when
            // this ran; one generic scratch arena stands in for it.
            alloc(R, "scratch", 64ull << 20, false);
            buffer_table_stubbed = true;
            printf("buffers: program.json buffer table ABSENT — stubbed with one 64 MiB "
                   "\"scratch\" arena (regenerate with python3 k0/compile_schedule.py)\n");
        }
    }

    void set_pass_inputs(int32_t token, int64_t pos) {
        int32_t tok = token;
        int32_t p4[4] = { (int32_t) pos, (int32_t) pos, (int32_t) pos, (int32_t) pos };
        int64_t row = pos;
        CUDA_CHECK(cudaMemcpy(bufs["token_id"].ptr, &tok, 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bufs["positions"].ptr, p4, 16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bufs["kv_row"].ptr, &row, 8, cudaMemcpyHostToDevice));
        // Mask: n_kv = pad(n_past+1, 256); 0 for j <= n_past, -inf padding tail.
        int64_t n_past = pos;
        int64_t n_kv = pad_up(n_past + 1, 256);
        if (n_kv > mask_cap) throw std::runtime_error("mask: n_kv past n_ctx padding cap");
        std::vector<uint16_t> mask((size_t) n_kv);
        for (int64_t j = 0; j < n_kv; j++)
            mask[(size_t) j] = j <= n_past ? F16_ZERO : F16_NEG_INF;
        CUDA_CHECK(cudaMemcpy(bufs["mask_f16"].ptr, mask.data(), (size_t) n_kv * 2, cudaMemcpyHostToDevice));
        PassCtx ctx{};
        ctx.token = tok;
        for (int i = 0; i < 4; i++) ctx.pos[i] = (int32_t) pos;
        ctx.n_past = (int32_t) n_past;
        ctx.n_kv = (int32_t) n_kv;
        ctx.kv_row = row;
        CUDA_CHECK(cudaMemcpy(bufs["pass_ctx"].ptr, &ctx, sizeof(ctx), cudaMemcpyHostToDevice));
    }

    void read_f32(const std::string &name, std::vector<float> &out, size_t n_elems, size_t byte_off = 0) {
        out.resize(n_elems);
        CUDA_CHECK(cudaMemcpy(out.data(), (char *) bufs[name].ptr + byte_off,
                              n_elems * 4, cudaMemcpyDeviceToHost));
    }
};

// ---------------------------------------------------------------------------
// PACKER — program.json instrs -> mk::Instr[]
//
// PACKER STUBS: k0/ops/*.cuh (the Args structs, parallel agents) did not
// exist when this harness was written, so every per-kind packer below
// fails loudly at pack time naming its kind. To land a kind: include its
// ops header, write pack_<kind>(args, R, out) filling out.payload with the
// op's Args struct (static_assert(sizeof(Args) <= sizeof(out.payload))),
// and point the table entry at it.
// ---------------------------------------------------------------------------

using PackFn = void (*)(const Jv &args, const Resolver &R, mk::Instr &out);

[[noreturn]] static void stub_fail(const char *kind) {
    throw std::runtime_error(std::string("packer stub: ") + kind +
        " — no Args struct (k0/ops/*.cuh absent at harness-writing time); implement pack_" + kind);
}
#define PACK_STUB(KIND) \
    static void pack_##KIND(const Jv &, const Resolver &, mk::Instr &) { stub_fail(#KIND); }

PACK_STUB(EMBED_LOOKUP)     PACK_STUB(RMSNORM)          PACK_STUB(QUANT_Q8_1)
PACK_STUB(HEAD_GEMV_F16)    PACK_STUB(LOGITS_EMIT)      PACK_STUB(MMVQ_Q4_0)
PACK_STUB(MMVQ_Q4_0_FUSED)  PACK_STUB(MMVQ_AR16)        PACK_STUB(GEMV_F16)
PACK_STUB(CONV_SHIFT_CONCAT) PACK_STUB(SSM_CONV_SILU)   PACK_STUB(QK_L2NORM)
PACK_STUB(GDN_GATES)        PACK_STUB(GDN_STEP)         PACK_STUB(GATED_RMSNORM)
PACK_STUB(QK_NORM_ROPE)     PACK_STUB(KV_APPEND)        PACK_STUB(FATTN_DECODE)
PACK_STUB(FATTN_REDUCE)     PACK_STUB(ATTN_GATE)        PACK_STUB(RESIDUAL_ADD)
PACK_STUB(STATE_LOAD)       PACK_STUB(STATE_STORE)      PACK_STUB(XCHG_PUSH)
PACK_STUB(XCHG_REDUCE)

static void pack_NOP(const Jv &, const Resolver &, mk::Instr &) {}
static void pack_BOUNDARY(const Jv &, const Resolver &, mk::Instr &) {}

struct KindEntry { mk::MacroKind kind; const char *name; PackFn pack; };
static const KindEntry KIND_TABLE[] = {
    { mk::OP_NOP,               "NOP",               pack_NOP },
    { mk::OP_BOUNDARY,          "BOUNDARY",          pack_BOUNDARY },
    { mk::OP_EMBED_LOOKUP,      "EMBED_LOOKUP",      pack_EMBED_LOOKUP },
    { mk::OP_RMSNORM,           "RMSNORM",           pack_RMSNORM },
    { mk::OP_QUANT_Q8_1,        "QUANT_Q8_1",        pack_QUANT_Q8_1 },
    { mk::OP_HEAD_GEMV_F16,     "HEAD_GEMV_F16",     pack_HEAD_GEMV_F16 },
    { mk::OP_LOGITS_EMIT,       "LOGITS_EMIT",       pack_LOGITS_EMIT },
    { mk::OP_MMVQ_Q4_0,         "MMVQ_Q4_0",         pack_MMVQ_Q4_0 },
    { mk::OP_MMVQ_Q4_0_FUSED,   "MMVQ_Q4_0_FUSED",   pack_MMVQ_Q4_0_FUSED },
    { mk::OP_MMVQ_AR16,         "MMVQ_AR16",         pack_MMVQ_AR16 },
    { mk::OP_GEMV_F16,          "GEMV_F16",          pack_GEMV_F16 },
    { mk::OP_CONV_SHIFT_CONCAT, "CONV_SHIFT_CONCAT", pack_CONV_SHIFT_CONCAT },
    { mk::OP_SSM_CONV_SILU,     "SSM_CONV_SILU",     pack_SSM_CONV_SILU },
    { mk::OP_QK_L2NORM,         "QK_L2NORM",         pack_QK_L2NORM },
    { mk::OP_GDN_GATES,         "GDN_GATES",         pack_GDN_GATES },
    { mk::OP_GDN_STEP,          "GDN_STEP",          pack_GDN_STEP },
    { mk::OP_GATED_RMSNORM,     "GATED_RMSNORM",     pack_GATED_RMSNORM },
    { mk::OP_QK_NORM_ROPE,      "QK_NORM_ROPE",      pack_QK_NORM_ROPE },
    { mk::OP_KV_APPEND,         "KV_APPEND",         pack_KV_APPEND },
    { mk::OP_FATTN_DECODE,      "FATTN_DECODE",      pack_FATTN_DECODE },
    { mk::OP_FATTN_REDUCE,      "FATTN_REDUCE",      pack_FATTN_REDUCE },
    { mk::OP_ATTN_GATE,         "ATTN_GATE",         pack_ATTN_GATE },
    { mk::OP_RESIDUAL_ADD,      "RESIDUAL_ADD",      pack_RESIDUAL_ADD },
    { mk::OP_STATE_LOAD,        "STATE_LOAD",        pack_STATE_LOAD },
    { mk::OP_STATE_STORE,       "STATE_STORE",       pack_STATE_STORE },
    { mk::OP_XCHG_PUSH,         "XCHG_PUSH",         pack_XCHG_PUSH },
    { mk::OP_XCHG_REDUCE,       "XCHG_REDUCE",       pack_XCHG_REDUCE },
};

static const KindEntry *kind_by_name(const std::string &raw) {
    std::string name = raw.rfind("OP_", 0) == 0 ? raw.substr(3) : raw;
    for (auto &e : KIND_TABLE) if (name == e.name) return &e;
    return nullptr;
}

struct PackedProgram {
    std::vector<mk::Instr> instrs;
    uint32_t epoch_stride = 0;
    bool complete = false;
    std::string first_failure;
    size_t packed_before_failure = 0;
};

static PackedProgram pack_program(const Jv &pj, const Resolver &R) {
    PackedProgram out;
    out.epoch_stride = pj.get("epoch_stride") ? (uint32_t) pj.at("epoch_stride").as_i() : 0;
    const Jv &instrs = pj.at("instrs");
    out.instrs.reserve(instrs.arr.size());
    for (size_t i = 0; i < instrs.arr.size(); i++) {
        const Jv &ij = instrs.arr[i];
        const std::string kname = ij.at("kind").str;
        const KindEntry *ke = kind_by_name(kname);
        if (!ke) {
            out.first_failure = "instr " + std::to_string(i) + ": unknown kind '" + kname + "'";
            out.packed_before_failure = i;
            return out;
        }
        mk::Instr ins{};
        ins.kind = (uint16_t) ke->kind;
        ins.block_lo = (uint16_t) ij.at("block_lo").as_i();
        ins.block_hi = (uint16_t) ij.at("block_hi").as_i();
        ins.flags = ij.get("flags") ? (uint16_t) ij.at("flags").as_i() : 0;
        ins.dbg_node = ij.get("dbg_node") ? (uint32_t) ij.at("dbg_node").as_i() : 0;
        static const Jv empty_args;
        const Jv *args = ij.get("args");
        try {
            ke->pack(args ? *args : empty_args, R, ins);
        } catch (const std::exception &e) {
            out.first_failure = "instr " + std::to_string(i) + " (" + kname + "): " + e.what();
            out.packed_before_failure = i;
            return out;
        }
        out.instrs.push_back(ins);
    }
    out.complete = true;
    out.packed_before_failure = out.instrs.size();
    return out;
}

// ---------------------------------------------------------------------------
// LAUNCHER STAND-IN (duplication, noted)
//
// core/host.h (the persistent-kernel upload/launch/run_pass library a
// parallel agent owns) did not exist when this harness was written, so
// this is the harness's own minimal notion of the same contract:
//   upload_program(instrs, epoch_stride)  — device Instr[] + Program header
//                                           + zeroed y02 counter (real)
//   run_pass()                            — one decode pass; UNAVAILABLE
//                                           here: no interpreter kernel is
//                                           linked into this binary.
// When core/host.h lands, port parity_run/bench_run to its API and delete
// this struct.
// ---------------------------------------------------------------------------

struct Launcher {
    mk::Instr *d_instr = nullptr;
    unsigned *d_y02 = nullptr;
    mk::Program hdr{};
    bool program_uploaded = false;
    static constexpr bool kernel_available = false;

    void upload_program(const std::vector<mk::Instr> &prog, uint32_t epoch_stride) {
        CUDA_CHECK(cudaMalloc(&d_instr, prog.size() * sizeof(mk::Instr)));
        CUDA_CHECK(cudaMemcpy(d_instr, prog.data(), prog.size() * sizeof(mk::Instr),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_y02, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(d_y02, 0, sizeof(unsigned)));
        hdr.n_instr = (uint32_t) prog.size();
        hdr.epoch_stride = epoch_stride;
        hdr.y02_counter = d_y02;
        program_uploaded = true;
    }

    bool run_pass() {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "mk-harness: run_pass UNAVAILABLE — the persistent interpreter "
                    "kernel (core/host.cpp + k0/ops/*.cuh) is not built into this binary "
                    "(core/host.h was absent at harness-writing time)\n");
            warned = true;
        }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Parity mode: mk-oracle/v1 dump-tree writer + driver
// ---------------------------------------------------------------------------

struct DumpTree {
    std::string dir;
    FILE *nodes = nullptr, *files = nullptr;
    std::vector<float> logits_rows;   // (steps+1) x n_vocab, appended in order
    int64_t n_vocab = 0;
    int n_full = 0, n_state = 0;

    void open(const std::string &d, int64_t nv) {
        dir = d;
        n_vocab = nv;
        mkdir_p(dir);
        nodes = fopen((dir + "/nodes.csv").c_str(), "w");
        files = fopen((dir + "/files.csv").c_str(), "w");
        if (!nodes || !files) throw std::runtime_error("cannot create dump tree at " + dir);
        fprintf(nodes, "step,idx,name,op,type,ne0,ne1,ne2,ne3,summary,rms,mean,min,max,v0,v1,v2,v3,v4,v5,v6,v7\n");
        fprintf(files, "step,kind,name,path,op,type,ne0,ne1,ne2,ne3,nbytes\n");
    }

    void node_row(int step, int idx, const char *name, const char *op, size_t ne0, const Stats &s) {
        fprintf(nodes, "%d,%d,%s,%s,f32,%zu,1,1,1,ok,%.17g,%.17g,%.17g,%.17g", step, idx, name, op, ne0,
                s.rms, s.mean, s.mn, s.mx);
        for (int i = 0; i < 8; i++) fprintf(nodes, ",%.9g", s.v[i]);
        fprintf(nodes, "\n");
    }

    void write_bin(int step, const char *kind, const std::string &name, const char *op,
                   const std::string &rel, const float *data, size_t n) {
        std::string full_path = dir + "/" + rel;
        size_t slash = full_path.rfind('/');
        mkdir_p(full_path.substr(0, slash));
        FILE *f = fopen(full_path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + full_path);
        fwrite(data, 4, n, f);
        fclose(f);
        fprintf(files, "%d,%s,%s,%s,%s,f32,%zu,1,1,1,%zu\n", step, kind, name.c_str(), rel.c_str(), op, n, n * 4);
        if (strcmp(kind, "full") == 0) n_full++; else n_state++;
    }

    void add_logits_row(const std::vector<float> &row) {
        logits_rows.insert(logits_rows.end(), row.begin(), row.end());
    }

    void finish(const Jv &ref_idx, int64_t n_ctx, const std::vector<int32_t> &emitted) {
        FILE *f = fopen((dir + "/logits.bin").c_str(), "wb");
        fwrite(logits_rows.data(), 4, logits_rows.size(), f);
        fclose(f);
        f = fopen((dir + "/tokens.txt").c_str(), "w");
        for (int32_t t : emitted) fprintf(f, "%d\n", t);
        fclose(f);
        fclose(nodes); nodes = nullptr;
        fclose(files); files = nullptr;

        const Jv &cfg = ref_idx.at("config");
        const Jv &prompt = ref_idx.at("prompt_tokens");
        int steps = (int) cfg.at("steps").as_i();
        f = fopen((dir + "/index.json").c_str(), "w");
        fprintf(f, "{\n  \"format\": \"mk-oracle/v1\",\n  \"tag\": \"mk-k0-harness\",\n");
        fprintf(f, "  \"config\": {\n    \"split\": \"mk-k0\",\n    \"n_ctx\": %lld,\n"
                   "    \"ctx_tokens\": %lld,\n    \"steps\": %d,\n    \"n_vocab\": %lld,\n"
                   "    \"flash_attn\": true,\n    \"kv_type\": \"f16\",\n"
                   "    \"n_gpu_layers\": 999,\n    \"capture\": \"targeted\"\n  },\n",
                (long long) n_ctx, (long long) cfg.at("ctx_tokens").as_i(), steps, (long long) n_vocab);
        fprintf(f, "  \"logits\": { \"file\": \"logits.bin\", \"dtype\": \"f32\", \"rows\": %d, \"cols\": %lld,\n"
                   "    \"note\": \"row 0 = prefill output; row s+1 = output of decode step s; "
                   "token s chosen greedily from row s\" },\n", steps + 1, (long long) n_vocab);
        fprintf(f, "  \"nodes_csv\": \"nodes.csv\",\n  \"files_csv\": \"files.csv\",\n");
        fprintf(f, "  \"n_full_files\": %d,\n  \"n_state_files\": %d,\n", n_full, n_state);
        fprintf(f, "  \"prompt_tokens\": [");
        for (size_t i = 0; i < prompt.arr.size(); i++)
            fprintf(f, "%s%lld", i ? "," : "", (long long) prompt.arr[i].as_i());
        fprintf(f, "],\n  \"emitted_tokens\": [");
        for (size_t i = 0; i < emitted.size(); i++)
            fprintf(f, "%s%d", i ? "," : "", emitted[i]);
        fprintf(f, "]\n}\n");
        fclose(f);
    }
};

static int32_t argmax_f32_first(const std::vector<float> &row) {
    // fp32 first-max greedy (ties resolve to the lowest index), matching the
    // oracle's "greedy fp32 first-max argmax".
    float best = row[0];
    int32_t arg = 0;
    for (size_t i = 1; i < row.size(); i++)
        if (row[i] > best) { best = row[i]; arg = (int32_t) i; }
    return arg;
}

static int parity_run(Residency &res, Runtime &rt, Launcher &ln, const std::string &ref_dir,
                      const std::string &out_dir) {
    Jv ref_idx = json_load(ref_dir + "/index.json");
    if (ref_idx.at("format").str != "mk-oracle/v1")
        throw std::runtime_error("ref tree is not mk-oracle/v1");
    const Jv &cfg = ref_idx.at("config");
    const Jv &prompt = ref_idx.at("prompt_tokens");
    int steps = (int) cfg.at("steps").as_i();
    int64_t nv_ref = cfg.at("n_vocab").as_i();
    if (nv_ref != res.n_vocab)
        throw std::runtime_error("ref n_vocab != model n_vocab");
    printf("parity: ref %s — %zu prompt tokens, %d greedy steps, n_vocab %lld\n",
           ref_dir.c_str(), prompt.arr.size(), steps, (long long) nv_ref);
    if ((int64_t) prompt.arr.size() + steps > rt.n_ctx)
        throw std::runtime_error("prompt + steps exceed --n-ctx");

    DumpTree dump;
    dump.open(out_dir, res.n_vocab);

    std::vector<float> logits, lout, rnorm, state;
    std::vector<int32_t> emitted;
    int64_t pos = 0;

    // Prefill: one decode pass per prompt token, logits discarded until the
    // last (decode-only prefill, position from 0).
    for (size_t i = 0; i < prompt.arr.size(); i++, pos++) {
        rt.set_pass_inputs((int32_t) prompt.arr[i].as_i(), pos);
        if (!ln.run_pass()) return 3;
    }
    rt.read_f32("logits", logits, (size_t) res.n_vocab);
    dump.add_logits_row(logits);   // row 0 = prefill output

    for (int s = 0; s < steps; s++, pos++) {
        int32_t tok = argmax_f32_first(logits);   // token s from row s
        emitted.push_back(tok);
        rt.set_pass_inputs(tok, pos);
        if (!ln.run_pass()) return 3;
        rt.read_f32("logits", logits, (size_t) res.n_vocab);
        dump.add_logits_row(logits);   // row s+1

        char rel[128], name[64];
        // Residual stream: dbg_lout row il = residual after block il
        // (schedule contract; see Runtime::allocate).
        rt.read_f32("dbg_lout", lout, (size_t) N_LOUT * N_EMBD);
        for (int il = 0; il < N_LOUT; il++) {
            Stats st = stats_of(lout.data() + (size_t) il * N_EMBD, N_EMBD);
            snprintf(name, sizeof(name), "l_out-%d", il);
            snprintf(rel, sizeof(rel), "full/s%d/l_out-%d.bin", s, il);
            dump.node_row(s, il, name, "ADD", N_EMBD, st);
            dump.write_bin(s, "full", name, "ADD", rel, lout.data() + (size_t) il * N_EMBD, N_EMBD);
        }
        rt.read_f32("result_norm", rnorm, N_EMBD);
        dump.node_row(s, 3702, "result_norm", "MUL", N_EMBD, stats_of(rnorm.data(), N_EMBD));
        dump.node_row(s, 3703, "result_output", "MUL_MAT", (size_t) res.n_vocab,
                      stats_of(logits.data(), logits.size()));
        // DeltaNet state banks, raw bytes for the bit-exact lane.
        for (int il = 0; il < N_LAYER; il++) {
            if (is_attn_layer(il)) continue;
            rt.read_f32("ssm_state_l" + std::to_string(il), state, SSM_STATE_N);
            snprintf(name, sizeof(name), "cache_s_l%d", il);
            snprintf(rel, sizeof(rel), "state/s%d/cache_s_l%d.bin", s, il);
            dump.write_bin(s, "state", name, "CPY", rel, state.data(), SSM_STATE_N);
            rt.read_f32("conv_state_l" + std::to_string(il), state, CONV_STATE_N);
            snprintf(name, sizeof(name), "cache_r_l%d", il);
            snprintf(rel, sizeof(rel), "state/s%d/cache_r_l%d.bin", s, il);
            dump.write_bin(s, "state", name, "CPY", rel, state.data(), CONV_STATE_N);
        }
    }

    dump.finish(ref_idx, rt.n_ctx, emitted);
    printf("parity: dump tree written to %s\n", out_dir.c_str());
    printf("compare with:\n  python3 tests/oracle/compare.py %s %s --allow-config-mismatch --report %s/parity-report.json\n",
           ref_dir.c_str(), out_dir.c_str(), out_dir.c_str());
    printf("  (--allow-config-mismatch: the harness runs single-GPU at n_ctx %lld vs the ref's "
           "split/n_ctx; prompt, steps and n_vocab still bind)\n", (long long) rt.n_ctx);
    return 0;
}

// ---------------------------------------------------------------------------
// Bench mode
// ---------------------------------------------------------------------------

static bool gpu_looks_idle(int gpu) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i %d 2>/dev/null", gpu);
    FILE *p = popen(cmd, "r");
    if (!p) return false;
    char line[64] = {0};
    bool got = fgets(line, sizeof(line), p) != nullptr;
    pclose(p);
    return got && atoi(line) <= 2;
}

static int bench_run(Runtime &rt, Launcher &ln, int gpu, int64_t n_tokens) {
    bool idle = gpu_looks_idle(gpu);
    const int WARMUP = 32;
    if ((int64_t) WARMUP + n_tokens > rt.n_ctx)
        throw std::runtime_error("warmup + N exceed --n-ctx");
    printf("bench: %d warmup + %lld timed decode passes (token id fixed, positions advancing)\n",
           WARMUP, (long long) n_tokens);
    int64_t pos = 0;
    for (int i = 0; i < WARMUP; i++, pos++) {
        rt.set_pass_inputs(11, pos);
        if (!ln.run_pass()) return 3;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n_tokens; i++, pos++) {
        rt.set_pass_inputs(11, pos);
        if (!ln.run_pass()) return 3;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("bench: %lld tokens in %.3f s = %.2f tok/s [%s]\n",
           (long long) n_tokens, sec, n_tokens / sec,
           idle ? "GPU otherwise idle" : "CONTENDED/INDICATIVE: GPU not idle at start");
    printf("bench: per-pass ms breakdown unavailable (the spine's on-device timing is not "
           "built into this binary)\n");
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static std::string basename_of(const std::string &p) {
    std::string s = p;
    while (!s.empty() && s.back() == '/') s.pop_back();
    size_t slash = s.rfind('/');
    return slash == std::string::npos ? s : s.substr(slash + 1);
}

int main(int argc, char **argv) {
    std::string model = "/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf";
    std::string program_path = "k0/program.json";
    std::string parity_ref, out_dir;
    int64_t n_ctx = 8192, bench_n = -1;
    int gpu = 0;
    enum { M_VALIDATE, M_PARITY, M_BENCH } mode = M_VALIDATE;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs an argument (%s)\n", a.c_str(), what); exit(2); }
            return argv[++i];
        };
        if (a == "--validate")     mode = M_VALIDATE;
        else if (a == "--parity")  { mode = M_PARITY; parity_ref = need("oracle ref dir"); }
        else if (a == "--bench")   { mode = M_BENCH; bench_n = strtoll(need("token count"), nullptr, 10); }
        else if (a == "--model")   model = need("gguf path");
        else if (a == "--program") program_path = need("program.json path");
        else if (a == "--out")     out_dir = need("dump dir");
        else if (a == "--n-ctx")   n_ctx = strtoll(need("context size"), nullptr, 10);
        else if (a == "--gpu")     gpu = atoi(need("device index"));
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    try {
        CUDA_CHECK(cudaSetDevice(gpu));

        Residency res;
        bool inventory_ok = res.enumerate_and_validate(model);

        Runtime rt;
        rt.n_ctx = n_ctx;
        rt.n_vocab = res.n_vocab;

        // GPU etiquette: the ~15.6 GB upload needs a mostly-free device.
        size_t need = res.upload_bytes_needed() + rt.bytes_needed() + (256ull << 20);
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        printf("vram: GPU %d free %.2f / %.2f GB; need %.2f GB (weights %.2f + runtime %.2f + margin)\n",
               gpu, free_b / 1e9, total_b / 1e9, need / 1e9,
               res.upload_bytes_needed() / 1e9, rt.bytes_needed() / 1e9);
        if (free_b < need) {
            fprintf(stderr, "aborting: not enough free VRAM on GPU %d (other agents may be running "
                    "tests; retry when the device is mostly free)\n", gpu);
            return 2;
        }

        Resolver R;
        res.upload(R);
        bool checksums_ok = res.checksum_spot_checks();

        // Program: k0/program.json, regenerating via compile_schedule.py if
        // absent; stub the buffer table if both are missing.
        Jv program;
        bool have_program = false;
        {
            std::string text;
            if (!read_file(program_path, text)) {
                struct stat st{};
                if (stat("k0/compile_schedule.py", &st) == 0) {
                    printf("program: %s absent — running python3 k0/compile_schedule.py\n", program_path.c_str());
                    int rc = system("python3 k0/compile_schedule.py");
                    if (rc != 0) printf("program: compile_schedule.py exited %d\n", rc);
                }
            }
            if (read_file(program_path, text)) {
                JsonParser jp(text);
                program = jp.value();
                have_program = true;
            }
        }
        rt.allocate(R, have_program ? &program : nullptr);
        printf("runtime: KV 16 layers x 2 x %lld x %d f16 (%.2f GB), state 48 x (%d + %d) f32 (%.2f MB), "
               "mask cap %lld f16\n",
               (long long) n_ctx, N_EMBD_GQA, 16.0 * 2 * n_ctx * N_EMBD_GQA * 2 / 1e9,
               SSM_STATE_N, CONV_STATE_N, 48.0 * (SSM_STATE_N + CONV_STATE_N) * 4 / 1e6,
               (long long) rt.mask_cap);

        Launcher ln;
        PackedProgram packed;
        if (have_program) {
            packed = pack_program(program, R);
            if (packed.complete) {
                printf("pack: %zu instrs packed\n", packed.instrs.size());
                ln.upload_program(packed.instrs, packed.epoch_stride);
                printf("pack: program uploaded (%u instrs, epoch_stride %u)\n",
                       ln.hdr.n_instr, ln.hdr.epoch_stride);
            } else {
                printf("pack: FAILED after %zu instrs: %s\n",
                       packed.packed_before_failure, packed.first_failure.c_str());
            }
        } else {
            printf("pack: SKIPPED — no %s and no k0/compile_schedule.py to generate it\n",
                   program_path.c_str());
        }

        printf("pipeline status: inventory %s, checksums %s, buffers %s, program %s, kernel %s\n",
               inventory_ok ? "OK" : "MISMATCH",
               checksums_ok ? "OK" : "MISMATCH",
               rt.buffer_table_stubbed ? "STUBBED (no program.json buffer table)" : "from program.json",
               !have_program ? "ABSENT" : (packed.complete ? "PACKED" : "PACK-FAILED (op Args stubs)"),
               Launcher::kernel_available ? "linked" : "NOT LINKED (core/host.h absent at build)");

        if (mode == M_VALIDATE)
            return (inventory_ok && checksums_ok) ? 0 : 1;

        if (!inventory_ok || !checksums_ok) {
            fprintf(stderr, "refusing to run %s: loader validation failed\n",
                    mode == M_PARITY ? "parity" : "bench");
            return 1;
        }
        if (!have_program || !packed.complete) {
            fprintf(stderr, "cannot run %s: no packed program (see pack status above)\n",
                    mode == M_PARITY ? "parity" : "bench");
            return 3;
        }

        if (mode == M_PARITY) {
            if (out_dir.empty())
                out_dir = "/var/tmp/mk-harness/cand-" + basename_of(parity_ref);
            return parity_run(res, rt, ln, parity_ref, out_dir);
        }
        return bench_run(rt, ln, gpu, bench_n);
    } catch (const std::exception &e) {
        fprintf(stderr, "mk-harness: %s\n", e.what());
        return 2;
    }
}
