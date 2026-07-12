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

#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include "../core/host.h"          // mk::Host control-plane API (the launcher)
#include "ops/glue.cuh"            // Args structs for the packer (compiled by nvcc)
#include "ops/gdn.cuh"
#include "ops/gemv.cuh"
#include "ops/attn.cuh"
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

struct Runtime {
    int64_t n_ctx = 8192;
    int64_t n_vocab = 0;
    int64_t mask_cap = 0;   // pad(n_ctx, 256) f16 entries
    std::map<std::string, DevBuf> bufs;
    bool buffer_table_stubbed = false;
    // Pass-input copies ride a non-blocking stream so they never serialize
    // against the persistent kernel (which lives on its own non-blocking
    // stream and never returns); the legacy default stream would be a hazard.
    cudaStream_t pstream = nullptr;

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
        CUDA_CHECK(cudaStreamCreateWithFlags(&pstream, cudaStreamNonBlocking));
        // KV caches: 16 attention layers, [n_ctx rows x 1024] f16 per K and V.
        // Zero-filled: padded/unwritten rows read as finite 0.0 f16 (the FATTN
        // op loads padded rows but the -inf mask zeroes their weight).
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
        // Per-pass IO. cell:token is bound to the Host's d_token elsewhere.
        alloc(R, "positions", 16);          // i32[4] M-RoPE ids, all = seq pos
        alloc(R, "kv_row", 8);              // i64[1] KV append row = position
        alloc(R, "rs_row", 8);              // i64[1] = 0 (single-seq ring slot 0)
        mask_cap = pad_up(n_ctx, 256);
        alloc(R, "mask_f16", (size_t) mask_cap * 2);
        // Output cells the ops write but the harness reads via buf:logits /
        // the kernel's pass-done: result_output (LOGITS_EMIT copy), done_flag
        // (op-set flag, unused), fattn_error (FATTN_REDUCE sentinel count).
        alloc(R, "result_output", (size_t) n_vocab * 4);
        alloc(R, "done_flag", 4);
        alloc(R, "fattn_error", 4);
        // Parity residual-stream buffer: row il = residual after block il
        // (l_out-il, blocks 0..62), written by each attn_norm RMSNORM's dbg.
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
            alloc(R, "logits", (size_t) n_vocab * 4);   // table absent: own it
            buffer_table_stubbed = true;
            printf("buffers: program.json buffer table ABSENT — stubbed with one 64 MiB "
                   "\"scratch\" arena (regenerate with python3 k0/compile_schedule.py)\n");
        }
    }

    // Write positions / kv_row / mask / zero fattn_error for this token; the
    // token id itself rides host_run_pass -> h.d_token. Returns padded n_kv so
    // the driver can patch the FATTN_DECODE payloads. All copies land (synced)
    // before the caller rings the doorbell.
    int64_t set_pass_inputs(int64_t pos) {
        int32_t p4[4] = { (int32_t) pos, (int32_t) pos, (int32_t) pos, (int32_t) pos };
        int64_t row = pos;
        CUDA_CHECK(cudaMemcpyAsync(bufs["positions"].ptr, p4, 16, cudaMemcpyHostToDevice, pstream));
        CUDA_CHECK(cudaMemcpyAsync(bufs["kv_row"].ptr, &row, 8, cudaMemcpyHostToDevice, pstream));
        // Mask: n_kv = pad(n_past+1, 256); 0 for j <= n_past, -inf padding tail.
        int64_t n_past = pos;
        int64_t n_kv = pad_up(n_past + 1, 256);
        if (n_kv > mask_cap) throw std::runtime_error("mask: n_kv past n_ctx padding cap");
        static std::vector<uint16_t> mask;
        mask.resize((size_t) n_kv);
        for (int64_t j = 0; j < n_kv; j++)
            mask[(size_t) j] = j <= n_past ? F16_ZERO : F16_NEG_INF;
        CUDA_CHECK(cudaMemcpyAsync(bufs["mask_f16"].ptr, mask.data(), (size_t) n_kv * 2,
                                   cudaMemcpyHostToDevice, pstream));
        CUDA_CHECK(cudaMemsetAsync(bufs["fattn_error"].ptr, 0, 4, pstream));
        CUDA_CHECK(cudaStreamSynchronize(pstream));
        return n_kv;
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

// ---- arg accessors + namespace-aware pointer resolution --------------------
// program.json args carry namespaced names ("gguf:NAME", "buf:NAME",
// "cell:NAME", "cache:NAME") and runtime symbols ("sym:$NAME"). dev_ptr maps a
// namespaced name to a device pointer via the Resolver; the resolver is keyed
// on BARE names (weights by GGUF name, buffers/cells by name, caches under the
// harness's own cache names), so dev_ptr strips the namespace and applies the
// cache-name mapping (state_r_l<i> -> conv_state_l<i>, state_s_l<i> ->
// ssm_state_l<i>). "sym:" is a per-pass scalar and is NOT a pointer.

static std::string arg_str(const Jv &a, const char *k) { return a.at(k).str; }
static int64_t arg_i(const Jv &a, const char *k) { return a.at(k).as_i(); }
static int64_t arg_i_def(const Jv &a, const char *k, int64_t d) {
    const Jv *v = a.get(k); return v ? v->as_i() : d;
}
static bool arg_has(const Jv &a, const char *k) { return a.get(k) != nullptr; }
static bool is_sym(const Jv &a, const char *k) {
    const Jv *v = a.get(k);
    return v && v->k == Jv::STR && v->str.rfind("sym:", 0) == 0;
}

static void *dev_ptr(const Resolver &R, const std::string &spec) {
    size_t colon = spec.find(':');
    if (colon == std::string::npos) return R.resolve(spec);
    std::string ns = spec.substr(0, colon), name = spec.substr(colon + 1);
    if (ns == "sym")
        throw std::runtime_error("dev_ptr: '" + spec + "' is a runtime symbol, not a pointer");
    if (ns == "cache") {
        if (name.rfind("state_r_l", 0) == 0) name = "conv_state_l" + name.substr(9);
        else if (name.rfind("state_s_l", 0) == 0) name = "ssm_state_l" + name.substr(9);
        // cache_k_l<i>/cache_v_l<i> pass through to the harness's own names
    }
    return R.resolve(name);           // gguf/buf/cell/cache resolve by bare name
}
// buffer/weight base + element offset (offsets in program.json are f32 elems).
static float *dev_f32(const Resolver &R, const Jv &a, const char *k, int64_t off_elems = 0) {
    return reinterpret_cast<float *>(dev_ptr(R, arg_str(a, k))) + off_elems;
}

// Everything a pack fn needs: the resolver, the proto instruction (kind /
// block range / flags / dbg_node prefilled), and a hook to record the emitted
// FATTN_DECODE positions so the per-pass driver can patch their $n_kv field.
struct PackCtx {
    const Resolver &R;
    const mk::Instr &proto;
    std::vector<size_t> &out_idx_fattn;  // indices (into `out`) of FATTN_DECODEs
};

template <class Args>
static mk::Instr with_payload(const mk::Instr &proto, const Args &args) {
    static_assert(sizeof(Args) <= sizeof(proto.payload), "payload overflow");
    mk::Instr ins = proto;
    memset(ins.payload, 0, sizeof(ins.payload));
    memcpy(ins.payload, &args, sizeof(Args));
    return ins;
}
template <class Args>
static void emit(std::vector<mk::Instr> &out, const mk::Instr &proto, const Args &args) {
    out.push_back(with_payload(proto, args));
}

// l_out node index for a residual-fold RMSNORM: block M's attn_norm folds
// block (M-1)'s FFN output, so residual after it is l_out-(M-1). Returns -1
// for the first norm (no fold), post_attention_norm folds, and output_norm.
static int lout_index_of(const Jv &a) {
    if (!arg_has(a, "add_src")) return -1;
    const std::string w = arg_str(a, "weight");         // "gguf:blk.M.attn_norm.weight"
    size_t b = w.find("blk.");
    if (b == std::string::npos) return -1;               // output_norm: no l_out
    if (w.find(".attn_norm.weight") == std::string::npos) return -1; // post_attention: no l_out
    int m = atoi(w.c_str() + b + 4);
    return m - 1;                                         // blk.1 -> l_out-0
}

using PackFn = void (*)(const Jv &args, PackCtx &c, std::vector<mk::Instr> &out);

static void pack_NOP(const Jv &, PackCtx &c, std::vector<mk::Instr> &out) { out.push_back(c.proto); }
static void pack_BOUNDARY(const Jv &, PackCtx &c, std::vector<mk::Instr> &out) { out.push_back(c.proto); }

static void pack_EMBED_LOOKUP(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::EmbedLookupArgs e{};
    e.emb        = reinterpret_cast<const __half *>(dev_ptr(c.R, arg_str(a, "weight")));
    e.token      = reinterpret_cast<const int32_t *>(dev_ptr(c.R, arg_str(a, "token")));
    e.y          = dev_f32(c.R, a, "dst");
    e.ncols      = (uint32_t) arg_i(a, "n_embd");
    e.row_stride = e.ncols;
    emit(out, c.proto, e);
}

static void pack_RMSNORM(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::RmsnormArgs r{};
    r.x     = dev_f32(c.R, a, "src");
    r.w     = dev_f32(c.R, a, "weight");
    r.add   = arg_has(a, "add_src") ? dev_f32(c.R, a, "add_src") : nullptr;
    r.y     = dev_f32(c.R, a, "dst");
    r.ncols = (uint32_t) arg_i(a, "width");
    r.nrows = 1;                                     // batch-1 trunk (row_select = 0)
    r.sum   = arg_has(a, "write_sum") ? dev_f32(c.R, a, "write_sum") : nullptr;
    int il  = lout_index_of(a);
    r.dbg   = il >= 0 ? reinterpret_cast<float *>(c.R.resolve("dbg_lout")) + (size_t) il * N_EMBD
                      : nullptr;
    emit(out, c.proto, r);
}

static void pack_QUANT_Q8_1(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::QuantQ8_1Args q{};
    q.x          = dev_f32(c.R, a, "src");
    q.y          = dev_ptr(c.R, arg_str(a, "dst"));
    q.ne00       = (uint32_t) arg_i(a, "elems");
    q.ne0_padded = (uint32_t) pad_up(q.ne00, 512);  // MATRIX_ROW_PADDING
    emit(out, c.proto, q);
}

static void pack_gemv_f16_common(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out,
                                 uint32_t ncols_default) {
    mk::GemvF16Args g{};
    g.w      = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "weight")));
    g.x      = dev_f32(c.R, a, "src");
    g.dst    = dev_f32(c.R, a, "dst");
    g.ncols  = (uint32_t) arg_i_def(a, "src_elems", ncols_default);
    g.row_lo = (uint32_t) arg_i(a, "row_lo");
    g.row_hi = (uint32_t) arg_i(a, "row_hi");
    emit(out, c.proto, g);
}
static void pack_GEMV_F16(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    pack_gemv_f16_common(a, c, out, N_EMBD);
}
static void pack_HEAD_GEMV_F16(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    pack_gemv_f16_common(a, c, out, N_EMBD);       // src = xn (5120), no src_elems arg
}

static void pack_MMVQ_Q4_0(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::MmvqQ40Args m{};
    m.w      = dev_ptr(c.R, arg_str(a, "weight"));
    m.y      = dev_ptr(c.R, arg_str(a, "src"));
    m.dst    = dev_f32(c.R, a, "dst", arg_i_def(a, "dst_off", 0));
    m.ncols  = (uint32_t) arg_i(a, "src_elems");
    m.row_lo = (uint32_t) arg_i(a, "row_lo");
    m.row_hi = (uint32_t) arg_i(a, "row_hi");
    emit(out, c.proto, m);
}
static void pack_MMVQ_Q4_0_FUSED(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::MmvqQ40FusedArgs m{};
    m.w_up   = dev_ptr(c.R, arg_str(a, "weight_up"));
    m.w_gate = dev_ptr(c.R, arg_str(a, "weight_gate"));
    m.y      = dev_ptr(c.R, arg_str(a, "src"));
    m.dst    = dev_f32(c.R, a, "dst");
    m.ncols  = (uint32_t) arg_i(a, "src_elems");
    m.row_lo = (uint32_t) arg_i(a, "row_lo");
    m.row_hi = (uint32_t) arg_i(a, "row_hi");
    emit(out, c.proto, m);
}
static void pack_MMVQ_AR16(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::MmvqAr16Args m{};
    m.w      = dev_ptr(c.R, arg_str(a, "weight"));
    m.y      = dev_ptr(c.R, arg_str(a, "src"));
    m.dst    = dev_f32(c.R, a, "dst", arg_i_def(a, "dst_off", 0));
    m.ncols  = (uint32_t) arg_i(a, "src_elems");
    m.row_lo = (uint32_t) arg_i(a, "row_lo");
    m.row_hi = (uint32_t) arg_i(a, "row_hi");
    emit(out, c.proto, m);
}

static void pack_CONV_SHIFT_CONCAT(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::ConvShiftConcatArgs s{};
    // rs_row is always 0 (single sequence), so the history row = cache base.
    s.hist       = dev_f32(c.R, a, "conv_state");
    s.xnew       = dev_f32(c.R, a, "token_col");
    s.win        = dev_f32(c.R, a, "window_dst");
    s.state      = dev_f32(c.R, a, "state_writeback");
    s.row        = reinterpret_cast<const int64_t *>(dev_ptr(c.R, "rs_row"));
    s.channels   = (int32_t) arg_i(a, "channels");
    s.row_stride = (int64_t) CONV_STATE_N;          // one conv-cache row
    emit(out, c.proto, s);
}
static void pack_SSM_CONV_SILU(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::SsmConvSiluArgs s{};
    s.win      = dev_f32(c.R, a, "window");
    s.weight   = dev_f32(c.R, a, "kernel");
    s.dst      = dev_f32(c.R, a, "dst");
    s.channels = (int32_t) arg_i(a, "channels");
    emit(out, c.proto, s);
}
static void pack_QK_L2NORM(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    // q at q_off and k at k_off are contiguous 128-wide heads: normalize both
    // in one call (per-head l2 norm is independent). Assert contiguity.
    const int64_t q_off = arg_i(a, "q_off"), k_off = arg_i(a, "k_off");
    const int64_t heads = arg_i(a, "heads"), hd = arg_i(a, "head_dim");
    if (k_off != q_off + heads * hd)
        throw std::runtime_error("QK_L2NORM: q|k not contiguous");
    mk::QkL2NormArgs n{};
    n.src     = dev_f32(c.R, a, "buf", q_off);
    n.dst     = dev_f32(c.R, a, "buf", q_off);   // in place
    n.n_heads = (int32_t)(2 * heads);
    n.eps     = (float) a.at("eps").num;
    emit(out, c.proto, n);
}
static void pack_GDN_GATES(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::GdnGatesArgs g{};
    g.alpha_raw = dev_f32(c.R, a, "alpha");
    g.beta_raw  = dev_f32(c.R, a, "beta");
    g.dt_bias   = dev_f32(c.R, a, "dt_bias");
    g.a         = dev_f32(c.R, a, "a");
    g.g         = dev_f32(c.R, a, "g_dst");
    g.beta      = dev_f32(c.R, a, "beta_dst");
    g.n_heads   = (int32_t) arg_i(a, "heads");
    emit(out, c.proto, g);
}
static void pack_GDN_STEP(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::GdnStepArgs s{};
    s.q         = dev_f32(c.R, a, "qkv", arg_i(a, "q_off"));
    s.k         = dev_f32(c.R, a, "qkv", arg_i(a, "k_off"));
    s.v         = dev_f32(c.R, a, "qkv", arg_i(a, "v_off"));
    s.g         = dev_f32(c.R, a, "g");
    s.beta      = dev_f32(c.R, a, "beta");
    s.state_in  = dev_f32(c.R, a, "state");
    s.state_out = dev_f32(c.R, a, "state");   // in place (may alias state_in)
    s.attn_out  = dev_f32(c.R, a, "dst");
    s.n_heads   = (int32_t) arg_i(a, "v_heads");
    s.n_k_heads = (int32_t) arg_i(a, "k_heads");
    s.scale     = (float) a.at("scale").num;
    emit(out, c.proto, s);
}
static void pack_GATED_RMSNORM(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::GatedRmsNormArgs g{};
    g.x       = dev_f32(c.R, a, "src");
    g.w       = dev_f32(c.R, a, "weight");
    g.z       = dev_f32(c.R, a, "gate");
    g.dst     = dev_f32(c.R, a, "dst");
    g.n_heads = (int32_t) arg_i(a, "heads");
    g.eps     = (float) a.at("eps").num;
    emit(out, c.proto, g);
}
static void pack_STATE_LOAD(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::StateLoadArgs s{};
    s.src        = dev_f32(c.R, a, "src");
    s.rows       = reinterpret_cast<const int64_t *>(dev_ptr(c.R, "rs_row"));
    s.dst        = dev_f32(c.R, a, "dst");
    s.n_elems    = (int32_t) arg_i(a, "elems");
    s.row_stride = (int64_t) s.n_elems;
    s.n_rows     = 1;
    emit(out, c.proto, s);
}
static void pack_STATE_STORE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::StateStoreArgs s{};
    s.src        = dev_f32(c.R, a, "src");
    s.dst        = dev_f32(c.R, a, "dst");
    s.rows       = reinterpret_cast<const int64_t *>(dev_ptr(c.R, "rs_row"));
    s.n_elems    = (int32_t) arg_i(a, "elems");
    s.row_stride = (int64_t) s.n_elems;
    s.n_rows     = 1;
    emit(out, c.proto, s);
}

static void pack_QK_NORM_ROPE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    // program.json fuses the q and k norm+rope into one node; the op does one
    // tensor, so emit two independent instructions (disjoint writes, no
    // boundary needed between them).
    const int32_t *pos = reinterpret_cast<const int32_t *>(dev_ptr(c.R, arg_str(a, "positions")));
    mk::QkNormRopeArgs q{};
    q.src        = dev_f32(c.R, a, "q_src", arg_i_def(a, "q_off", 0));
    q.norm_w     = dev_f32(c.R, a, "q_norm_weight");
    q.pos        = pos;
    q.dst        = dev_f32(c.R, a, "q_dst");
    q.n_heads    = (uint32_t) arg_i(a, "q_heads");
    q.src_stride = (uint32_t) arg_i(a, "q_head_stride");
    emit(out, c.proto, q);

    mk::QkNormRopeArgs k{};
    k.src        = dev_f32(c.R, a, "k_src");
    k.norm_w     = dev_f32(c.R, a, "k_norm_weight");
    k.pos        = pos;
    k.dst        = dev_f32(c.R, a, "k_dst");
    k.n_heads    = (uint32_t) arg_i(a, "k_heads");
    k.src_stride = (uint32_t) arg_i(a, "head_dim");   // k rows are dense (256)
    emit(out, c.proto, k);
}
static void pack_KV_APPEND(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    // K and V SET_ROWS as two independent instructions (one cache each).
    const long long *row = reinterpret_cast<const long long *>(dev_ptr(c.R, "kv_row"));
    const uint32_t rw = (uint32_t) arg_i(a, "row_width");
    mk::KvAppendArgs kk{};
    kk.src = dev_f32(c.R, a, "k_src"); kk.row_idx = row;
    kk.cache = reinterpret_cast<half *>(dev_ptr(c.R, arg_str(a, "cache_k")));
    kk.row_width = rw;
    emit(out, c.proto, kk);
    mk::KvAppendArgs vv{};
    vv.src = dev_f32(c.R, a, "v_src"); vv.row_idx = row;
    vv.cache = reinterpret_cast<half *>(dev_ptr(c.R, arg_str(a, "cache_v")));
    vv.row_width = rw;
    emit(out, c.proto, vv);
}
static void pack_FATTN_DECODE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::FattnDecodeArgs f{};
    f.q          = dev_f32(c.R, a, "q");
    f.k_cache    = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "cache_k")));
    f.v_cache    = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "cache_v")));
    f.mask       = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "mask")));
    f.partials   = dev_f32(c.R, a, "partials");
    f.n_kv       = 0;                                  // $n_kv, patched per pass
    f.n_q        = (uint32_t) arg_i(a, "q_heads");
    f.n_kv_heads = (uint32_t) arg_i(a, "kv_heads");
    f.row_width  = (uint32_t)(arg_i(a, "kv_heads") * arg_i(a, "head_dim"));
    c.out_idx_fattn.push_back(out.size());             // record for $n_kv patch
    emit(out, c.proto, f);
}
static void pack_FATTN_REDUCE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::FattnReduceArgs f{};
    f.partials = dev_f32(c.R, a, "partials");
    f.dst      = dev_f32(c.R, a, "dst");
    f.error    = reinterpret_cast<unsigned *>(dev_ptr(c.R, "fattn_error"));
    f.n_q      = (uint32_t) arg_i(a, "q_heads");
    f.n_chunks = (uint32_t) arg_i(a, "n_splits");
    emit(out, c.proto, f);
}
static void pack_ATTN_GATE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::AttnGateArgs g{};
    g.attn        = dev_f32(c.R, a, "attn");
    g.gate        = dev_f32(c.R, a, "gate_src", arg_i_def(a, "gate_off", 0));
    g.dst         = dev_f32(c.R, a, "dst");
    g.n_q         = (uint32_t) arg_i(a, "heads");
    g.gate_stride = (uint32_t) arg_i(a, "gate_head_stride");
    emit(out, c.proto, g);
}
static void pack_LOGITS_EMIT(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::LogitsEmitArgs e{};
    e.src        = dev_f32(c.R, a, "src");
    e.dst        = reinterpret_cast<float *>(dev_ptr(c.R, arg_str(a, "dst")));
    e.n          = (uint32_t) arg_i(a, "elems");
    e.flag_value = 0;
    e.flag       = nullptr;   // multi-block copy: rely on the kernel's pass done
    emit(out, c.proto, e);
}
[[noreturn]] static void pack_unsupported(const Jv &, PackCtx &c, std::vector<mk::Instr> &) {
    throw std::runtime_error(std::string("packer: kind ") +
        std::to_string(c.proto.kind) + " not supported at the single-GPU binding");
}

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
    { mk::OP_RESIDUAL_ADD,      "RESIDUAL_ADD",      pack_unsupported },
    { mk::OP_STATE_LOAD,        "STATE_LOAD",        pack_STATE_LOAD },
    { mk::OP_STATE_STORE,       "STATE_STORE",       pack_STATE_STORE },
    { mk::OP_XCHG_PUSH,         "XCHG_PUSH",         pack_unsupported },
    { mk::OP_XCHG_REDUCE,       "XCHG_REDUCE",       pack_unsupported },
};

static const KindEntry *kind_by_name(const std::string &raw) {
    std::string name = raw.rfind("OP_", 0) == 0 ? raw.substr(3) : raw;
    for (auto &e : KIND_TABLE) if (name == e.name) return &e;
    return nullptr;
}

struct PackedProgram {
    std::vector<mk::Instr> instrs;
    std::vector<size_t> fattn_idx;     // FATTN_DECODE positions ($n_kv patch)
    uint32_t epoch_stride = 0;
    bool complete = false;
    std::string first_failure;
    size_t packed_before_failure = 0;
};

static PackedProgram pack_program(const Jv &pj, const Resolver &R) {
    PackedProgram out;
    // epoch_stride lives under meta.per_pass (boundaries per pass); the kernel
    // crosses boundaries by counter, so this is informational (G15 accounting).
    if (const Jv *m = pj.get("meta"))
        if (const Jv *pp = m->get("per_pass"))
            if (const Jv *es = pp->get("epoch_stride")) out.epoch_stride = (uint32_t) es->as_i();
    const Jv &instrs = pj.get("instructions") ? pj.at("instructions") : pj.at("instrs");
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
        mk::Instr proto{};
        proto.kind = (uint16_t) ke->kind;
        proto.block_lo = (uint16_t) ij.at("block_lo").as_i();
        proto.block_hi = (uint16_t) ij.at("block_hi").as_i();
        proto.flags = ij.get("flags") ? (uint16_t) ij.at("flags").as_i() : 0;
        proto.dbg_node = ij.get("dbg_node") && ij.get("dbg_node")->k == Jv::NUM
                             ? (uint32_t) ij.at("dbg_node").as_i() : 0;
        static const Jv empty_args;
        const Jv *args = ij.get("args");
        PackCtx ctx{R, proto, out.fattn_idx};
        try {
            ke->pack(args ? *args : empty_args, ctx, out.instrs);
        } catch (const std::exception &e) {
            out.first_failure = "instr " + std::to_string(i) + " (" + kname + "): " + e.what();
            out.packed_before_failure = i;
            return out;
        }
    }
    out.complete = true;
    out.packed_before_failure = out.instrs.size();
    return out;
}

// ---------------------------------------------------------------------------
// LAUNCHER — the persistent interpreter, driven through mk::Host (core/host.h).
// host_init allocates the control cells (incl. h.d_token, the per-pass token
// input); host_upload uploads the packed Instr[] and wires the y02 counter;
// host_launch does the cooperative launch (72x384, 60 KiB slab); run_pass ->
// host_run_pass. The FATTN_DECODE payloads carry a per-pass $n_kv (padded KV
// window) patched into device memory before each pass.
// ---------------------------------------------------------------------------

struct Launcher {
    mk::Host h;
    std::vector<size_t> fattn_idx;     // instr indices needing $n_kv patched
    bool program_uploaded = false;
    static constexpr bool kernel_available = true;
    // byte offset of FattnDecodeArgs::n_kv within mk::Instr (payload + field).
    static constexpr size_t N_KV_OFF =
        offsetof(mk::Instr, payload) + offsetof(mk::FattnDecodeArgs, n_kv);

    void init(int device) {
        if (!mk::host_init(h, device, /*pass_cycles_cap=*/1024))
            throw std::runtime_error("mk::host_init failed (G11 envelope / coop launch)");
    }
    // d_token is valid only after host_init; the resolver binds cell:token to it.
    int32_t *d_token() { return h.d_token; }

    void upload_program(const PackedProgram &p) {
        if (!mk::host_upload(h, p.instrs.data(), (uint32_t) p.instrs.size(), p.epoch_stride))
            throw std::runtime_error("mk::host_upload failed");
        fattn_idx = p.fattn_idx;
        if (!mk::host_launch(h))
            throw std::runtime_error("mk::host_launch failed (cooperative launch)");
        program_uploaded = true;
    }

    // Patch $n_kv into every FATTN_DECODE payload for this pass.
    void patch_n_kv(uint32_t n_kv) {
        for (size_t idx : fattn_idx)
            CUDA_CHECK(cudaMemcpy((char *) h.d_program + idx * sizeof(mk::Instr) + N_KV_OFF,
                                  &n_kv, sizeof(n_kv), cudaMemcpyHostToDevice));
    }

    bool run_pass(int32_t token) {
        mk::RunStatus st = mk::host_run_pass(h, token, /*timeout_ms=*/60000.0);
        if (st != mk::RUN_OK) {
            fprintf(stderr, "mk-harness: run_pass status %d (pass %u)\n", (int) st, h.pass);
            return false;
        }
        return true;
    }
    void shutdown() {
        if (program_uploaded) mk::host_shutdown(h, 5000.0);
        mk::host_destroy(h);
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
        ln.patch_n_kv((uint32_t) rt.set_pass_inputs(pos));
        if (!ln.run_pass((int32_t) prompt.arr[i].as_i())) return 3;
    }
    rt.read_f32("logits", logits, (size_t) res.n_vocab);
    dump.add_logits_row(logits);   // row 0 = prefill output

    for (int s = 0; s < steps; s++, pos++) {
        int32_t tok = argmax_f32_first(logits);   // token s from row s
        emitted.push_back(tok);
        ln.patch_n_kv((uint32_t) rt.set_pass_inputs(pos));
        if (!ln.run_pass(tok)) return 3;
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
        // result_norm = the final output_norm result (buf:xn after the last
        // RMSNORM, before the head GEMV). Informational in compare.py.
        rt.read_f32("xn", rnorm, N_EMBD);
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
        ln.patch_n_kv((uint32_t) rt.set_pass_inputs(pos));
        if (!ln.run_pass(11)) return 3;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n_tokens; i++, pos++) {
        ln.patch_n_kv((uint32_t) rt.set_pass_inputs(pos));
        if (!ln.run_pass(11)) return 3;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("bench: %lld tokens in %.3f s = %.2f tok/s [%s]\n",
           (long long) n_tokens, sec, n_tokens / sec,
           idle ? "GPU otherwise idle" : "CONTENDED/INDICATIVE: GPU not idle at start");
    // On-device per-pass cycle deltas (block-0 clock64) from the G15 ring.
    unsigned cap = ln.h.pass_cycles_cap;
    if (cap && n_tokens > 0) {
        std::vector<long long> cyc(cap);
        if (mk::host_read_pass_cycles(ln.h, cyc.data(), cap)) {
            unsigned cnt = (unsigned) std::min<int64_t>(n_tokens, cap);
            double sum = 0; long long mn = cyc[0], mx = cyc[0];
            for (unsigned k = 0; k < cnt; k++) {
                long long v = cyc[(ln.h.pass - 1 - k) % cap];
                sum += (double) v; if (v < mn) mn = v; if (v > mx) mx = v;
            }
            // TU102 boost 1455 MHz; cycles -> ms.
            const double gHz = 1.455;
            printf("bench: per-pass on-device clock64: mean %.3f ms, min %.3f, max %.3f (%u samples)\n",
                   sum / cnt / gHz / 1e6, mn / gHz / 1e6, mx / gHz / 1e6, cnt);
        }
    }
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
        // The launcher owns the persistent kernel's control plane. host_init
        // allocates h.d_token (the per-pass token cell); bind cell:token to it
        // BEFORE the packer resolves names. Validate mode skips the launch.
        Launcher ln;
        if (mode != M_VALIDATE) {
            ln.init(gpu);
            R.add("token", ln.d_token(), 128);   // cell:token -> Host d_token
        }

        rt.allocate(R, have_program ? &program : nullptr);
        printf("runtime: KV 16 layers x 2 x %lld x %d f16 (%.2f GB), state 48 x (%d + %d) f32 (%.2f MB), "
               "mask cap %lld f16\n",
               (long long) n_ctx, N_EMBD_GQA, 16.0 * 2 * n_ctx * N_EMBD_GQA * 2 / 1e9,
               SSM_STATE_N, CONV_STATE_N, 48.0 * (SSM_STATE_N + CONV_STATE_N) * 4 / 1e6,
               (long long) rt.mask_cap);

        PackedProgram packed;
        if (have_program && mode != M_VALIDATE) {
            packed = pack_program(program, R);
            if (packed.complete) {
                printf("pack: %zu instrs packed (%zu FATTN_DECODE need $n_kv patched)\n",
                       packed.instrs.size(), packed.fattn_idx.size());
                ln.upload_program(packed);
                printf("pack: program uploaded + cooperative kernel launched "
                       "(%zu instrs, epoch_stride %u)\n",
                       packed.instrs.size(), packed.epoch_stride);
            } else {
                printf("pack: FAILED after %zu instrs: %s\n",
                       packed.packed_before_failure, packed.first_failure.c_str());
            }
        }

        printf("pipeline status: inventory %s, checksums %s, buffers %s, program %s, kernel %s\n",
               inventory_ok ? "OK" : "MISMATCH",
               checksums_ok ? "OK" : "MISMATCH",
               rt.buffer_table_stubbed ? "STUBBED (no program.json buffer table)" : "from program.json",
               !have_program ? "ABSENT"
                   : (mode == M_VALIDATE ? "PRESENT (not packed in validate mode)"
                      : (packed.complete ? "PACKED" : "PACK-FAILED")),
               Launcher::kernel_available ? "linked" : "NOT LINKED");

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

        int rc;
        if (mode == M_PARITY) {
            if (out_dir.empty())
                out_dir = "/var/tmp/mk-harness/cand-" + basename_of(parity_ref);
            rc = parity_run(res, rt, ln, parity_ref, out_dir);
        } else {
            rc = bench_run(rt, ln, gpu, bench_n);
        }
        ln.shutdown();   // HALT the persistent kernel + free cleanly
        return rc;
    } catch (const std::exception &e) {
        fprintf(stderr, "mk-harness: %s\n", e.what());
        return 2;
    }
}
