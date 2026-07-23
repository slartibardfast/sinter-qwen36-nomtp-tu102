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
// Options: --model PATH (default /opt/models/Qwen3.6-27B-AR16asF16-probe.gguf, the F16-ssm_out base)
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
#include <atomic>
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
#include <unistd.h>                 // sysconf(_SC_PAGESIZE) for the soak RSS watermark

#include <cuda_runtime.h>
#include <cudaProfiler.h>        // DRIVER cuProfilerStart/Stop: the RANGE markers ncu
                                 // intercepts (it does NOT hook the runtime variants) for
                                 // --replay-mode app-range on the persistent cooperative
                                 // megakernel (per-kernel replay can't profile it).

#include "../core/isa.cuh"
#include "../core/host.h"          // mk::Host control-plane API (the launcher)
#include "ops/glue.cuh"            // Args structs for the packer (compiled by nvcc)
#include "ops/gdn.cuh"
#include "ops/gemv.cuh"
#include "ops/attn.cuh"
#include "ops/xchg.cuh"           // dual-GPU cross-GPU reduce Args structs
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
// Ground truth for the DEFAULT base = the F16-ssm_out file (call/0020): the 48
// ssm_out projections are F16, not Q4_0_AR16, so f16 rises 5237.64 -> 8257.54 MB
// (+~3020 the ssm_out weights) and q4_0_ar16 is absent. The AR16 file (943.72 MB
// ar16, 5237.64 f16) is the flagged case -> run it with --allow-inventory-mismatch.
static const uint64_t EXPECT_N_TENSORS = 866;
struct ExpectTotal { uint32_t type; double mb; };
static const ExpectTotal EXPECT_TOTALS[] = {
    { gguf::T_Q4_0,      13043.96 },
    { gguf::T_F16,        8257.54 },
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
        alloc(R, "n_kv", 4);               // u32[1] padded KV window, read strong by FATTN
        alloc(R, "n_tok", 4);              // u32[1] per-pass tile width (U-loop live bound)
        { uint32_t one = 1;
          CUDA_CHECK(cudaMemcpy(R.resolve("n_tok"), &one, 4, cudaMemcpyHostToDevice)); }
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
        alloc(R, "dbg_mid", (size_t) (N_LAYER + 1) * N_EMBD * 4); // mid + embed bisect

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

    // Write positions / kv_row / n_kv cell / mask / zero fattn_error for this
    // token; the token id itself rides host_run_pass -> h.d_token. Writes the
    // padded KV window into the "n_kv" cell (read STRONG by OP_FATTN_DECODE) and
    // returns it. All copies land (synced) before the caller rings the doorbell.
    int64_t set_pass_inputs(int64_t pos) {
        int32_t p4[4] = { (int32_t) pos, (int32_t) pos, (int32_t) pos, (int32_t) pos };
        int64_t row = pos;
        CUDA_CHECK(cudaMemcpyAsync(bufs["positions"].ptr, p4, 16, cudaMemcpyHostToDevice, pstream));
        CUDA_CHECK(cudaMemcpyAsync(bufs["kv_row"].ptr, &row, 8, cudaMemcpyHostToDevice, pstream));
        // Mask: n_kv = pad(n_past+1, 256); 0 for j <= n_past, -inf padding tail.
        int64_t n_past = pos;
        int64_t n_kv = pad_up(n_past + 1, 256);
        if (n_kv > mask_cap) throw std::runtime_error("mask: n_kv past n_ctx padding cap");
        uint32_t nkv32 = (uint32_t) n_kv;
        CUDA_CHECK(cudaMemcpyAsync(bufs["n_kv"].ptr, &nkv32, 4, cudaMemcpyHostToDevice, pstream));
        static const uint32_t one_tok = 1;
        CUDA_CHECK(cudaMemcpyAsync(bufs["n_tok"].ptr, &one_tok, 4, cudaMemcpyHostToDevice, pstream));
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

// Per-instr U (batched prefill, meta.prefill): compute instrs carry
// n_tokens=U; a decode program has no such key -> 1.
static uint32_t arg_ntok(const Jv &a) { return (uint32_t) arg_i_def(a, "n_tokens", 1); }

// Everything a pack fn needs: the resolver, the proto instruction (kind /
// block range / flags / dbg_node prefilled), and a hook to record the emitted
// FATTN_DECODE positions (reporting count).
struct PackCtx {
    const Resolver &R;
    const mk::Instr &proto;
    std::vector<size_t> &out_idx_fattn;  // indices (into `out`) of FATTN_DECODEs
    // Dual-GPU (--tensor) extras; nullptr/-1 in the single-GPU path.
    const std::map<std::string, void *> *mbox = nullptr;  // "peer_payload:0"->ptr
    int gpu_index = -1;                  // 0/1, selects the p0+p1 fold order
    std::vector<size_t> *out_idx_xchg = nullptr;  // OP_XCHG_REDUCE ($seqno patch)
    uint32_t prog_u = 1;                 // meta.prefill.n_tokens (1 = decode program)
};

// Per-token slot (f32 elems) of a U-scaled scratch buffer: table bytes/4 / U.
// meta.prefill scales every per-token transient uniformly by U (decode: U=1,
// slot = whole buffer). Only f32 buffers are queried; q8_act's per-token
// stride is width-derived inside the ops instead.
static uint32_t slot_elems(const PackCtx &c, const Jv &a, const char *k) {
    std::string spec = arg_str(a, k);
    size_t colon = spec.find(':');
    std::string name = colon == std::string::npos ? spec : spec.substr(colon + 1);
    auto it = c.R.table.find(name);
    if (it == c.R.table.end())
        throw std::runtime_error("slot_elems: unknown buffer '" + spec + "'");
    return (uint32_t) (it->second.bytes / 4 / c.prog_u);
}

// The per-pass tile-width cell every U-looping op's live bound reads
// (min(payload capacity, cell); the n_kv-cell pattern).
static const uint32_t *ntok_cell(const PackCtx &c) {
    return reinterpret_cast<const uint32_t *>(c.R.resolve("n_tok"));
}

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
    e.n_tokens   = arg_ntok(a);
    e.ntok_cell  = ntok_cell(c);
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
    // BISECT: post_attention_norm folds the mixer output -> capture the mid-block
    // residual (embed + mixer, before the FFN) into dbg_mid[block]. And capture
    // block-0's attn_norm INPUT (= the embedding, no add) into dbg_mid[N_LAYER].
    {
        const std::string w = arg_str(a, "weight");
        size_t b = w.find("blk.");
        float *mid = reinterpret_cast<float *>(c.R.resolve("dbg_mid"));
        if (b != std::string::npos && arg_has(a, "add_src") &&
            w.find(".post_attention_norm.weight") != std::string::npos)
            r.dbg = mid + (size_t) atoi(w.c_str() + b + 4) * N_EMBD;
        else if (w == "gguf:blk.0.attn_norm.weight")
            r.dbg = mid + (size_t) N_LAYER * N_EMBD;   // = the embedding (no add)
    }
    r.n_tokens = arg_ntok(a);
    // Epilogue row select is RUNTIME (row nt-1 from the n_tok cell): a
    // pack-time literal cannot know the live width, and a per-token pass
    // through a prefill program (the parity reference leg) or a remainder
    // tile would norm an unwritten residual row. Both the decode "sym:
    // $out_row" form (nt=1 -> row 0) and the prefill literal U-1 (a full
    // tile's nt-1 = U-1) are realized by the same runtime select.
    r.row_select_last = arg_has(a, "row_select") ? 1u : 0u;
    r.ntok_cell = (r.n_tokens == 1 && !r.row_select_last) ? nullptr : ntok_cell(c);
    emit(out, c.proto, r);
}

static void pack_QUANT_Q8_1(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::QuantQ8_1Args q{};
    q.x          = dev_f32(c.R, a, "src");
    q.y          = dev_ptr(c.R, arg_str(a, "dst"));
    q.ne00       = (uint32_t) arg_i(a, "elems");
    q.ne0_padded = (uint32_t) pad_up(q.ne00, 512);  // MATRIX_ROW_PADDING
    q.n_tokens   = arg_ntok(a);
    q.ntok_cell  = ntok_cell(c);
    emit(out, c.proto, q);
}

static void pack_gemv_f16_common(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out,
                                 uint32_t ncols_default, bool single_row = false) {
    mk::GemvF16Args g{};
    g.w      = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "weight")));
    g.x      = dev_f32(c.R, a, "src");
    g.dst    = dev_f32(c.R, a, "dst");
    g.ncols  = (uint32_t) arg_i_def(a, "src_elems", ncols_default);
    g.row_lo = (uint32_t) arg_i(a, "row_lo");
    g.row_hi = (uint32_t) arg_i(a, "row_hi");
    g.n_tokens    = single_row ? 1 : arg_ntok(a);
    g.dst_tstride = single_row ? 0 : slot_elems(c, a, "dst");
    g.ntok_cell   = single_row ? nullptr : ntok_cell(c);
    emit(out, c.proto, g);
}
static void pack_GEMV_F16(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    pack_gemv_f16_common(a, c, out, N_EMBD);
}
static void pack_HEAD_GEMV_F16(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    // src = xn (5120), no src_elems arg. Downstream of the epilogue row
    // select: single-row at any U (meta.prefill), so loop count 1.
    pack_gemv_f16_common(a, c, out, N_EMBD, /*single_row=*/true);
}

static void pack_MMVQ_Q4_0(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::MmvqQ40Args m{};
    m.w      = dev_ptr(c.R, arg_str(a, "weight"));
    m.y      = dev_ptr(c.R, arg_str(a, "src"));
    m.dst    = dev_f32(c.R, a, "dst", arg_i_def(a, "dst_off", 0));
    m.ncols  = (uint32_t) arg_i(a, "src_elems");
    m.row_lo = (uint32_t) arg_i(a, "row_lo");
    m.row_hi = (uint32_t) arg_i(a, "row_hi");
    m.n_tokens    = arg_ntok(a);
    m.dst_tstride = slot_elems(c, a, "dst");
    m.ntok_cell   = ntok_cell(c);
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
    m.n_tokens    = arg_ntok(a);
    m.dst_tstride = slot_elems(c, a, "dst");
    m.ntok_cell   = ntok_cell(c);
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
    m.n_tokens    = arg_ntok(a);
    m.dst_tstride = slot_elems(c, a, "dst");
    m.ntok_cell   = ntok_cell(c);
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
    s.n_tokens     = (int32_t) arg_ntok(a);
    s.xnew_tstride = (int32_t) slot_elems(c, a, "token_col");
    s.ntok_cell    = ntok_cell(c);
    emit(out, c.proto, s);
}
static void pack_SSM_CONV_SILU(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::SsmConvSiluArgs s{};
    s.win      = dev_f32(c.R, a, "window");
    s.weight   = dev_f32(c.R, a, "kernel");
    s.dst      = dev_f32(c.R, a, "dst");
    s.channels = (int32_t) arg_i(a, "channels");
    s.n_tokens    = (int32_t) arg_ntok(a);
    s.dst_tstride = (int32_t) slot_elems(c, a, "dst");
    s.ntok_cell   = ntok_cell(c);
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
    n.n_tokens    = (int32_t) arg_ntok(a);
    n.src_tstride = (int32_t) slot_elems(c, a, "buf");
    n.dst_tstride = n.src_tstride;               // in place
    n.ntok_cell   = ntok_cell(c);
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
    g.n_tokens  = (int32_t) arg_ntok(a);
    g.tstride   = (int32_t) slot_elems(c, a, "alpha");
    g.ntok_cell = ntok_cell(c);
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
    s.n_tokens    = (int32_t) arg_ntok(a);
    s.qkv_tstride = (int32_t) slot_elems(c, a, "qkv");
    s.out_tstride = (int32_t) slot_elems(c, a, "dst");
    s.ntok_cell   = ntok_cell(c);
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
    g.n_tokens    = (int32_t) arg_ntok(a);
    g.x_tstride   = (int32_t) slot_elems(c, a, "src");
    g.z_tstride   = (int32_t) slot_elems(c, a, "gate");
    g.dst_tstride = (int32_t) slot_elems(c, a, "dst");
    g.ntok_cell   = ntok_cell(c);
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
    q.n_tokens    = arg_ntok(a);
    q.src_tstride = slot_elems(c, a, "q_src");
    q.dst_tstride = slot_elems(c, a, "q_dst");
    q.ntok_cell   = ntok_cell(c);
    emit(out, c.proto, q);

    mk::QkNormRopeArgs k{};
    k.src        = dev_f32(c.R, a, "k_src");
    k.norm_w     = dev_f32(c.R, a, "k_norm_weight");
    k.pos        = pos;
    k.dst        = dev_f32(c.R, a, "k_dst");
    k.n_heads    = (uint32_t) arg_i(a, "k_heads");
    k.src_stride = (uint32_t) arg_i(a, "head_dim");   // k rows are dense (256)
    k.n_tokens    = arg_ntok(a);
    k.src_tstride = slot_elems(c, a, "k_src");
    k.dst_tstride = slot_elems(c, a, "k_dst");
    k.ntok_cell   = ntok_cell(c);
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
    kk.n_tokens = arg_ntok(a);
    kk.src_tstride = slot_elems(c, a, "k_src");
    kk.ntok_cell   = ntok_cell(c);
    emit(out, c.proto, kk);
    mk::KvAppendArgs vv{};
    vv.src = dev_f32(c.R, a, "v_src"); vv.row_idx = row;
    vv.cache = reinterpret_cast<half *>(dev_ptr(c.R, arg_str(a, "cache_v")));
    vv.row_width = rw;
    vv.n_tokens = arg_ntok(a);
    vv.src_tstride = slot_elems(c, a, "v_src");
    vv.ntok_cell   = ntok_cell(c);
    emit(out, c.proto, vv);
}
static void pack_FATTN_DECODE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::FattnDecodeArgs f{};
    f.q          = dev_f32(c.R, a, "q");
    f.k_cache    = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "cache_k")));
    f.v_cache    = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "cache_v")));
    f.mask       = reinterpret_cast<const half *>(dev_ptr(c.R, arg_str(a, "mask")));
    f.partials   = dev_f32(c.R, a, "partials");
    f.n_kv_cell  = reinterpret_cast<const uint32_t *>(dev_ptr(c.R, arg_str(a, "n_kv")));
    f.n_q        = (uint32_t) arg_i(a, "q_heads");
    f.n_kv_heads = (uint32_t) arg_i(a, "kv_heads");
    f.row_width  = (uint32_t)(arg_i(a, "kv_heads") * arg_i(a, "head_dim"));
    f.n_tokens        = arg_ntok(a);
    f.q_tstride       = slot_elems(c, a, "q");
    f.partial_tstride = slot_elems(c, a, "partials");
    f.ntok_cell       = ntok_cell(c);
    c.out_idx_fattn.push_back(out.size());             // record for reporting count
    emit(out, c.proto, f);
}
static void pack_FATTN_REDUCE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::FattnReduceArgs f{};
    f.partials = dev_f32(c.R, a, "partials");
    f.dst      = dev_f32(c.R, a, "dst");
    f.error    = reinterpret_cast<unsigned *>(dev_ptr(c.R, "fattn_error"));
    f.n_q      = (uint32_t) arg_i(a, "q_heads");
    f.n_chunks = (uint32_t) arg_i(a, "n_splits");
    f.n_tokens        = arg_ntok(a);
    f.partial_tstride = slot_elems(c, a, "partials");
    f.dst_tstride     = slot_elems(c, a, "dst");
    f.ntok_cell       = ntok_cell(c);
    emit(out, c.proto, f);
}
static void pack_ATTN_GATE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::AttnGateArgs g{};
    g.attn        = dev_f32(c.R, a, "attn");
    g.gate        = dev_f32(c.R, a, "gate_src", arg_i_def(a, "gate_off", 0));
    g.dst         = dev_f32(c.R, a, "dst");
    g.n_q         = (uint32_t) arg_i(a, "heads");
    g.gate_stride = (uint32_t) arg_i(a, "gate_head_stride");
    g.n_tokens     = arg_ntok(a);
    g.attn_tstride = slot_elems(c, a, "attn");
    g.gate_tstride = slot_elems(c, a, "gate_src");
    g.ntok_cell    = ntok_cell(c);
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
// mailbox arg "mbox:peer_payload:0" -> key "peer_payload:0" in the per-GPU table.
static void *mbox_ptr(PackCtx &c, const Jv &a, const char *k) {
    if (!c.mbox) throw std::runtime_error("XCHG op packed without a mailbox table (single-GPU?)");
    std::string s = arg_str(a, k);                 // "mbox:peer_payload:0"
    if (s.rfind("mbox:", 0) != 0) throw std::runtime_error("XCHG: bad mailbox spec " + s);
    auto it = c.mbox->find(s.substr(5));
    if (it == c.mbox->end()) throw std::runtime_error("XCHG: unknown mailbox " + s);
    return it->second;
}
static void pack_XCHG_PUSH(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::XchgPushArgs x{};
    x.local_partial = dev_f32(c.R, a, "local_partial");
    x.peer_payload  = reinterpret_cast<float *>(mbox_ptr(c, a, "peer_payload"));
    x.n_elems       = (int) arg_i(a, "n_elems");
    x.n_tokens      = (int) arg_ntok(a);
    x.lp_tstride    = (int) slot_elems(c, a, "local_partial");
    x.ntok_cell     = ntok_cell(c);
    emit(out, c.proto, x);
}
static void pack_XCHG_REDUCE(const Jv &a, PackCtx &c, std::vector<mk::Instr> &out) {
    mk::XchgReduceArgs x{};
    x.local_partial = dev_f32(c.R, a, "local_partial");
    x.my_payload    = reinterpret_cast<const float *>(mbox_ptr(c, a, "my_payload"));
    x.out           = dev_f32(c.R, a, "out");
    x.peer_seqno    = reinterpret_cast<unsigned *>(mbox_ptr(c, a, "peer_seqno"));
    x.my_seqno      = reinterpret_cast<const unsigned *>(mbox_ptr(c, a, "my_seqno"));
    x.n_elems       = (int) arg_i(a, "n_elems");
    x.seqno         = 0;                 // $seqno, patched per pass (= pass number)
    x.gpu_index     = c.gpu_index;       // fixed p0+p1 fold order
    x.n_tokens      = (int) arg_ntok(a);
    x.lp_tstride    = (int) slot_elems(c, a, "local_partial");
    if (x.lp_tstride != (int) slot_elems(c, a, "out"))
        throw std::runtime_error("XCHG_REDUCE: local_partial/out slot mismatch");
    x.ntok_cell = ntok_cell(c);
    if (c.out_idx_xchg) c.out_idx_xchg->push_back(out.size());
    emit(out, c.proto, x);
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
    std::vector<uint32_t> src_json;    // packed idx -> originating program.json instr idx
                                       // (one json op expands to >=1 packed instrs; plan/0144
                                       // telemetry maps a packed op back to its schedule entry)
    std::vector<size_t> fattn_idx;     // FATTN_DECODE positions (reporting count)
    std::vector<size_t> xchg_idx;      // OP_XCHG_REDUCE positions ($seqno patch)
    uint32_t epoch_stride = 0;
    bool complete = false;
    std::string first_failure;
    size_t packed_before_failure = 0;
};

static PackedProgram pack_program(const Jv &pj, const Resolver &R,
                                  const std::map<std::string, void *> *mbox = nullptr,
                                  int gpu_index = -1) {
    PackedProgram out;
    // epoch_stride lives under meta.per_pass (boundaries per pass); the kernel
    // crosses boundaries by counter, so this is informational (G15 accounting).
    if (const Jv *m = pj.get("meta"))
        if (const Jv *pp = m->get("per_pass"))
            if (const Jv *es = pp->get("epoch_stride")) out.epoch_stride = (uint32_t) es->as_i();
    // Batched prefill: the buffer table is U-scaled uniformly (meta.prefill),
    // so per-token slot strides divide by the program's U. Decode: 1.
    uint32_t prog_u = 1;
    if (const Jv *m = pj.get("meta"))
        if (const Jv *pf = m->get("prefill"))
            if (const Jv *nt = pf->get("n_tokens")) prog_u = (uint32_t) nt->as_i();
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
        PackCtx ctx{R, proto, out.fattn_idx, mbox, gpu_index, &out.xchg_idx, prog_u};
        size_t before = out.instrs.size();
        try {
            ke->pack(args ? *args : empty_args, ctx, out.instrs);
        } catch (const std::exception &e) {
            out.first_failure = "instr " + std::to_string(i) + " (" + kname + "): " + e.what();
            out.packed_before_failure = i;
            return out;
        }
        for (size_t k = before; k < out.instrs.size(); k++)
            out.src_json.push_back((uint32_t) i);  // align packed -> json origin
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
// host_run_pass. The per-pass padded KV window rides the host-written "n_kv"
// cell (set_pass_inputs), read STRONG by OP_FATTN_DECODE — no payload patch.
// ---------------------------------------------------------------------------

struct Launcher {
    mk::Host h;
    bool program_uploaded = false;
    static constexpr bool kernel_available = true;

    void init(int device) {
        if (!mk::host_init(h, device, /*pass_cycles_cap=*/1024))
            throw std::runtime_error("mk::host_init failed (G11 envelope / coop launch)");
    }
    // d_token is valid only after host_init; the resolver binds cell:token to it.
    int32_t *d_token() { return h.d_token; }

    // Upload the packed program and set the dynamic-smem opt-in, but do NOT launch.
    // Split out so a caller can hoist the cuFuncSetAttribute out of an ncu profiler
    // range (which forbids it) and place only launch() inside the range.
    void stage_program(const PackedProgram &p) {
        if (!mk::host_upload(h, p.instrs.data(), (uint32_t) p.instrs.size(), p.epoch_stride))
            throw std::runtime_error("mk::host_upload failed");
        if (!mk::host_smem_optin(h))
            throw std::runtime_error("mk::host_smem_optin failed");
    }
    void launch() {
        if (!mk::host_launch(h))
            throw std::runtime_error("mk::host_launch failed (cooperative launch)");
        program_uploaded = true;
    }
    void upload_program(const PackedProgram &p) {
        stage_program(p);
        launch();
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
        rt.set_pass_inputs(pos);
        if (!ln.run_pass((int32_t) prompt.arr[i].as_i())) return 3;
    }
    rt.read_f32("logits", logits, (size_t) res.n_vocab);
    dump.add_logits_row(logits);   // row 0 = prefill output

    for (int s = 0; s < steps; s++, pos++) {
        int32_t tok = argmax_f32_first(logits);   // token s from row s
        emitted.push_back(tok);
        rt.set_pass_inputs(pos);
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
        rt.set_pass_inputs(pos);
        if (!ln.run_pass(11)) return 3;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n_tokens; i++, pos++) {
        rt.set_pass_inputs(pos);
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

// ===========================================================================
// DUAL-GPU tensor-parallel path (--parity-tensor / --bench-tensor)
//
// Two cooperative kernels (one per TU102), one split program packed per GPU.
// The op implementations are the verified single-GPU ops UNCHANGED; the split
// lives entirely in (1) the weight slice each GPU uploads, (2) the halved
// ranges/counts the compiler emits, (3) the 128 cross-GPU reduce sites. See
// core/DUAL-GPU-DESIGN.md, docs/INTEGRATION-FINDINGS.md (per-weight axis),
// k0/PARITY-TENSOR.md.
// ===========================================================================

// -- per-weight split axis (docs/INTEGRATION-FINDINGS.md, section A) ----------
//
// Simple contiguous axes for the attention/FFN/norm weights, plus a STRIDED
// mode for the DeltaNet head weights. The GDN step maps v-head h to q/k-head
// (h % n_k_heads) — a MODULO grouping (k0/ops/gdn.cuh:256) — so a contiguous
// v-head split would pair v-heads with k-heads on the wrong GPU. The v-head
// dimension (and the fused qkv / conv1d that carry it) must instead be split
// in blocks of n_k_heads(16) heads, taking this GPU's half of each block. In
// row/channel units that is a period of 16*head_dim=2048; for the per-v-head
// scalar params it is a period of 16. q and k (16 heads each, one block) fall
// out as their contiguous first half under the same period-16 rule.
enum WAxis { WA_MIRROR, WA_ROW, WA_COL };

static bool name_ends(const std::string &name, const char *suf) {
    size_t n = strlen(suf);
    return name.size() >= n && name.compare(name.size() - n, n, suf) == 0;
}

// Strided DeltaNet split descriptor. dim: 0 none, 1 ROW(split ne[1]),
// 2 COL(split ne[0]/input columns), 3 ROW1D(split the flat ne[0] vector).
// period is in units of that dimension (heads*head_dim, or bare heads).
struct DnStride { int dim; int64_t period; };
static DnStride dn_stride(const std::string &name) {
    if (name_ends(name, ".attn_qkv.weight"))  return { 1, 2048 };  // q|k|v rows
    if (name_ends(name, ".ssm_conv1d.weight")) return { 1, 2048 }; // q|k|v channels
    if (name_ends(name, ".attn_gate.weight"))  return { 1, 2048 }; // v-head z-gate rows
    if (name_ends(name, ".ssm_alpha.weight"))  return { 1, 16 };   // per-v-head rows
    if (name_ends(name, ".ssm_beta.weight"))   return { 1, 16 };
    if (name_ends(name, ".ssm_a"))             return { 3, 16 };   // per-v-head vector
    if (name_ends(name, ".ssm_dt.bias"))       return { 3, 16 };
    if (name_ends(name, ".ssm_out.weight"))    return { 2, 2048 }; // v-head input cols
    return { 0, 0 };
}

static WAxis weight_axis(const std::string &name) {
    if (name == "token_embd.weight" || name == "output_norm.weight" ||
        name_ends(name, ".attn_norm.weight") ||
        name_ends(name, ".post_attention_norm.weight") ||
        name_ends(name, ".attn_q_norm.weight") ||
        name_ends(name, ".attn_k_norm.weight") ||
        name_ends(name, ".ssm_norm.weight"))
        return WA_MIRROR;
    if (name_ends(name, ".attn_output.weight") || name_ends(name, ".ffn_down.weight"))
        return WA_COL;                                   // attention/FFN contract
    // ROW (AXIS_1): attention expand projections (contiguous head split).
    if (name_ends(name, ".attn_q.weight") || name_ends(name, ".attn_k.weight") ||
        name_ends(name, ".attn_v.weight") ||
        name_ends(name, ".ffn_gate.weight") || name_ends(name, ".ffn_up.weight") ||
        name == "output.weight")
        return WA_ROW;
    return WA_MIRROR;
}

// This GPU's slice of a weight into `stage` (or a direct pointer for the
// contiguous cases); returns the local byte count.  NGPU==2 exact halves.
static size_t weight_slice(const gguf::TensorInfo &t, int g,
                           std::vector<uint8_t> &stage, const uint8_t *&src) {
    const gguf::TypeTraits *tt = gguf::type_traits(t.type);
    const size_t rowbytes = (size_t)(t.ne[0] / tt->block_elems) * tt->block_bytes;

    // DeltaNet head weights: strided (period-block) split.
    DnStride ds = dn_stride(t.name);
    if (ds.dim == 1) {                 // split ne[1] rows in period blocks
        const int64_t nblk = t.ne[1] / ds.period, half = ds.period / 2;
        stage.resize((size_t)(nblk * half) * rowbytes);
        size_t cur = 0;
        for (int64_t b = 0; b < nblk; b++) {
            const size_t off = (size_t)(b * ds.period + (int64_t) g * half) * rowbytes;
            memcpy(stage.data() + cur, t.data + off, (size_t) half * rowbytes);
            cur += (size_t) half * rowbytes;
        }
        src = stage.data();
        return stage.size();
    }
    if (ds.dim == 3) {                 // split the flat ne[0] vector in period blocks
        const size_t es = tt->block_bytes;             // f32 scalar per head
        const int64_t nblk = t.ne[0] / ds.period, half = ds.period / 2;
        stage.resize((size_t)(nblk * half) * es);
        size_t cur = 0;
        for (int64_t b = 0; b < nblk; b++) {
            memcpy(stage.data() + cur, t.data + (size_t)(b * ds.period + (int64_t) g * half) * es,
                   (size_t) half * es);
            cur += (size_t) half * es;
        }
        src = stage.data();
        return stage.size();
    }
    if (ds.dim == 2) {                 // COL: per row, strided input-column blocks
        const size_t M = t.ne[1];
        const size_t colblk = tt->block_bytes;         // bytes per block_elems columns
        const int64_t nblk = t.ne[0] / ds.period, half = ds.period / 2;
        const size_t runbytes = (size_t)(half / tt->block_elems) * colblk;   // per row-block
        const size_t rowlocal = (size_t) nblk * runbytes;
        stage.resize(M * rowlocal);
        for (size_t r = 0; r < M; r++)
            for (int64_t b = 0; b < nblk; b++) {
                const size_t soff = r * rowbytes +
                    (size_t)((b * ds.period + (int64_t) g * half) / tt->block_elems) * colblk;
                memcpy(stage.data() + r * rowlocal + (size_t) b * runbytes,
                       t.data + soff, runbytes);
            }
        src = stage.data();
        return stage.size();
    }

    // Simple contiguous axes (attention / FFN / norms).
    const WAxis ax = weight_axis(t.name);
    if (ax == WA_MIRROR) { src = t.data; return t.nbytes; }
    if (ax == WA_ROW) { src = t.data + (size_t) g * (t.nbytes / 2); return t.nbytes / 2; }
    // WA_COL: each of ne[1] rows, this GPU's contiguous half of the columns.
    const size_t M = t.ne[1], half = rowbytes / 2;
    stage.resize(M * half);
    for (size_t r = 0; r < M; r++)
        memcpy(stage.data() + r * half, t.data + r * rowbytes + (size_t) g * half, half);
    src = stage.data();
    return stage.size();
}

// -- per-GPU context ---------------------------------------------------------
struct GpuCtx {
    int device = -1, gpu_index = -1;
    Resolver R;
    void *warena = nullptr;
    size_t warena_bytes = 0;
    std::map<std::string, DevBuf> bufs;
    float    *mbox_payload = nullptr;   // my inbox payloads [n_sites][5120] f32
    unsigned *mbox_seqno = nullptr;     // my inbox seqnos, one 128 B line/site
    std::map<std::string, void *> mtab; // "peer_payload:0" -> ptr (for the packer)
    Launcher ln;
    PackedProgram staged;              // packed program held between stage_program and launch
    std::vector<size_t> xchg_idx;      // OP_XCHG_REDUCE positions ($seqno patch)
    int64_t n_ctx = 8192, mask_cap = 0, n_vocab_full = 0;
    cudaStream_t pstream = nullptr;
    // seqno field offset inside a packed OP_XCHG_REDUCE instruction.
    static constexpr size_t SEQNO_OFF =
        offsetof(mk::Instr, payload) + offsetof(mk::XchgReduceArgs, seqno);
};

static const int MBOX_PAYLOAD_ELEMS = 5120;   // full residual width
static const int MBOX_SEQNO_STRIDE  = 32;      // u32 per 128 B line

static void gpu_upload_weights(GpuCtx &c, gguf::File &gg) {
    CUDA_CHECK(cudaSetDevice(c.device));
    std::vector<uint8_t> stage;
    const uint8_t *src;
    size_t total = 0, n_mirror = 0, n_row = 0, n_col = 0, n_dn = 0;
    for (auto &t : gg.tensors) {
        if (t.name.rfind("blk.64.", 0) == 0) continue;
        total += pad_up((int64_t) weight_slice(t, c.gpu_index, stage, src), 256);
    }
    CUDA_CHECK(cudaMalloc(&c.warena, total));
    c.warena_bytes = total;
    size_t cursor = 0;
    for (auto &t : gg.tensors) {
        if (t.name.rfind("blk.64.", 0) == 0) continue;
        size_t lb = weight_slice(t, c.gpu_index, stage, src);
        if (dn_stride(t.name).dim) n_dn++;
        else switch (weight_axis(t.name)) {
            case WA_MIRROR: n_mirror++; break; case WA_ROW: n_row++; break;
            case WA_COL: n_col++; break;
        }
        void *dst = (char *) c.warena + cursor;
        CUDA_CHECK(cudaMemcpy(dst, src, lb, cudaMemcpyHostToDevice));
        c.R.add(t.name, dst, lb);
        cursor += pad_up((int64_t) lb, 256);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  gpu %d (index %d): %.2f GB sliced weights "
           "(mirror %zu, row %zu, col %zu, dn-strided %zu)\n",
           c.device, c.gpu_index, c.warena_bytes / 1e9, n_mirror, n_row, n_col, n_dn);
}

// u_max sizes the per-pass input staging for a prefill U-tile (positions,
// kv_row, mask_f16 grow by the factor); the default 1 keeps every decode
// caller byte-identical to the pre-tile layout.
static void gpu_alloc_buffers(GpuCtx &c, const Jv &program, bool skip_kv = false,
                              int64_t u_max = 1) {
    CUDA_CHECK(cudaSetDevice(c.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.pstream, cudaStreamNonBlocking));
    auto alloc = [&](const std::string &nm, size_t bytes, bool zero = true) {
        void *p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes ? bytes : 4));
        if (zero) CUDA_CHECK(cudaMemset(p, 0, bytes ? bytes : 4));
        c.bufs[nm] = { p, bytes };
        c.R.add(nm, p, bytes);
        return p;
    };
    // skip_kv: the MK backend binds llama's KV/state pointers in place instead,
    // so the cache_k/v/conv_state/ssm_state buffers are not MK-owned.
    const size_t kv_row = (size_t)(N_EMBD_GQA / 2);       // 512 f16 elems (2 kv heads)
    if (!skip_kv) {
        for (int il = 0; il < N_LAYER; il++) if (is_attn_layer(il)) {
            size_t b = (size_t) c.n_ctx * kv_row * 2;
            alloc("cache_k_l" + std::to_string(il), b);
            alloc("cache_v_l" + std::to_string(il), b);
        }
        for (int il = 0; il < N_LAYER; il++) if (!is_attn_layer(il)) {
            alloc("conv_state_l" + std::to_string(il), (size_t)(CONV_STATE_N / 2) * 4);
            alloc("ssm_state_l" + std::to_string(il), (size_t)(SSM_STATE_N / 2) * 4);
        }
    }
    alloc("positions", 16 * (size_t) u_max);              // i32[4] M-RoPE quad per token
    alloc("kv_row", 8 * (size_t) u_max);                  // i64 KV append row per token
    alloc("rs_row", 8);
    alloc("n_kv", 4);                                     // u32 padded KV window (strong read)
    { void *p = alloc("n_tok", 4);                        // u32 per-pass tile width (U-loop live bound)
      uint32_t one = 1;
      CUDA_CHECK(cudaMemcpy(p, &one, 4, cudaMemcpyHostToDevice)); }
    c.mask_cap = pad_up(c.n_ctx, 256);
    alloc("mask_f16", (size_t) c.mask_cap * 2 * (size_t) u_max);   // u_max causal rows
    alloc("result_output", (size_t)(c.n_vocab_full / 2) * 4);  // this GPU's vocab half
    alloc("done_flag", 4);
    alloc("fattn_error", 4);
    alloc("dbg_lout", (size_t) N_LOUT * N_EMBD * 4);           // mirrored residual
    alloc("dbg_mid", (size_t) (N_LAYER + 1) * N_EMBD * 4);   // mid + embed bisect
    const Jv &tbl = program.at("buffers");
    for (auto &b : tbl.arr) alloc(b.at("name").str, (size_t) b.at("bytes").as_i());
}

// u_max widens each site's payload slot to MBOX_PAYLOAD_ELEMS * u_max: a
// prefill U-tile's OP_XCHG_PUSH lands U dense per-token slices per site
// (xchg.cuh, inbox packed at t*n_elems). One seqno per site either way.
static void gpu_alloc_mailboxes(GpuCtx &c, int n_sites, int64_t u_max = 1) {
    CUDA_CHECK(cudaSetDevice(c.device));
    const size_t slot = (size_t) MBOX_PAYLOAD_ELEMS * (size_t) u_max;
    CUDA_CHECK(cudaMalloc(&c.mbox_payload, (size_t) n_sites * slot * 4));
    CUDA_CHECK(cudaMemset(c.mbox_payload, 0, (size_t) n_sites * slot * 4));
    CUDA_CHECK(cudaMalloc(&c.mbox_seqno, (size_t) n_sites * MBOX_SEQNO_STRIDE * 4));
    CUDA_CHECK(cudaMemset(c.mbox_seqno, 0, (size_t) n_sites * MBOX_SEQNO_STRIDE * 4));
}

// Wire the packer's mailbox table: my inbox (peer writes here), and the peer's
// inbox (I push/publish there). Peer access is enabled both ways beforehand.
// u_max must match gpu_alloc_mailboxes (per-site payload slot stride).
static void gpu_wire_mailboxes(GpuCtx &c, GpuCtx &peer, int n_sites, int64_t u_max = 1) {
    const size_t slot = (size_t) MBOX_PAYLOAD_ELEMS * (size_t) u_max;
    for (int s = 0; s < n_sites; s++) {
        std::string ss = std::to_string(s);
        c.mtab["my_payload:" + ss]   = c.mbox_payload + (size_t) s * slot;
        c.mtab["my_seqno:" + ss]     = c.mbox_seqno + (size_t) s * MBOX_SEQNO_STRIDE;
        c.mtab["peer_payload:" + ss] = peer.mbox_payload + (size_t) s * slot;
        c.mtab["peer_seqno:" + ss]   = peer.mbox_seqno + (size_t) s * MBOX_SEQNO_STRIDE;
    }
}

// Write positions / kv_row / mask / zero fattn_error for this token on GPU c.
static int64_t gpu_set_inputs(GpuCtx &c, int64_t pos) {
    CUDA_CHECK(cudaSetDevice(c.device));
    int32_t p4[4] = { (int32_t) pos, (int32_t) pos, (int32_t) pos, (int32_t) pos };
    int64_t row = pos, n_past = pos, n_kv = pad_up(n_past + 1, 256);
    if (n_kv > c.mask_cap) throw std::runtime_error("mask: n_kv past n_ctx padding cap");
    uint32_t nkv32 = (uint32_t) n_kv;
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["positions"].ptr, p4, 16, cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["kv_row"].ptr, &row, 8, cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["n_kv"].ptr, &nkv32, 4, cudaMemcpyHostToDevice, c.pstream));
    static const uint32_t one_tok = 1;
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["n_tok"].ptr, &one_tok, 4, cudaMemcpyHostToDevice, c.pstream));
    static std::vector<uint16_t> mask;
    mask.resize((size_t) n_kv);
    for (int64_t j = 0; j < n_kv; j++) mask[(size_t) j] = j <= n_past ? F16_ZERO : F16_NEG_INF;
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["mask_f16"].ptr, mask.data(), (size_t) n_kv * 2,
                               cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemsetAsync(c.bufs["fattn_error"].ptr, 0, 4, c.pstream));
    CUDA_CHECK(cudaStreamSynchronize(c.pstream));
    return n_kv;
}

// Stage a U-token prefill tile starting at position pos0 on GPU c: one M-RoPE
// quad and one KV append row PER TOKEN, ONE padded window n_kv covering the
// whole tile, and the 2-D causal mask, query-major with the KV index fastest
// (row t at mask_f16 + t*n_kv), matching the fork's (n_kv, N, 1, 1) mask
// layout, idst = n_kv*i (llama-kv-cache.cpp:1477,1542-1573 via
// docs/PREFILL-SEMANTICS.md, batched causal attention). Row t masks j > pos0+t
// to -inf: token t attends its own and earlier positions ONLY. At U=1 every
// byte written here equals gpu_set_inputs(c, pos0); --prefill-parity binds
// that equivalence on device against the untouched decode staging above.
// Deliberately a SEPARATE implementation from gpu_set_inputs: the parity
// memcmp compares two independently written staging paths, so a bug in this
// one cannot hide by also steering the reference leg.
static int64_t gpu_set_inputs_tile(GpuCtx &c, int64_t pos0, int64_t U) {
    CUDA_CHECK(cudaSetDevice(c.device));
    int64_t n_kv = pad_up(pos0 + U, 256);
    if (n_kv > c.mask_cap) throw std::runtime_error("tile: n_kv past n_ctx padding cap");
    if ((size_t) U * 16 > c.bufs["positions"].bytes ||
        (size_t) U * 8 > c.bufs["kv_row"].bytes ||
        (size_t) U * (size_t) n_kv * 2 > c.bufs["mask_f16"].bytes)
        throw std::runtime_error("tile: U exceeds the u_max the buffers were sized for "
                                 "(re-run dual_setup with u_max >= U)");
    static std::vector<int32_t> p4;       // [U][4] M-RoPE ids, quad t all = pos0+t
    static std::vector<int64_t> rows;     // [U] KV append rows, row t = pos0+t
    static std::vector<uint16_t> mask;    // [U][n_kv] causal, row stride n_kv
    p4.resize((size_t) U * 4);
    rows.resize((size_t) U);
    mask.resize((size_t) U * (size_t) n_kv);
    for (int64_t t = 0; t < U; t++) {
        for (int q = 0; q < 4; q++) p4[(size_t) t * 4 + q] = (int32_t)(pos0 + t);
        rows[(size_t) t] = pos0 + t;
        uint16_t *mrow = mask.data() + (size_t) t * (size_t) n_kv;
        for (int64_t j = 0; j < n_kv; j++)
            mrow[j] = j <= pos0 + t ? F16_ZERO : F16_NEG_INF;
    }
    uint32_t nkv32 = (uint32_t) n_kv;
    static uint32_t u32_tok;   // stable source for the async copy
    u32_tok = (uint32_t) U;
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["n_tok"].ptr, &u32_tok, 4, cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["positions"].ptr, p4.data(), (size_t) U * 16,
                               cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["kv_row"].ptr, rows.data(), (size_t) U * 8,
                               cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["n_kv"].ptr, &nkv32, 4, cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemcpyAsync(c.bufs["mask_f16"].ptr, mask.data(),
                               (size_t) U * (size_t) n_kv * 2, cudaMemcpyHostToDevice, c.pstream));
    CUDA_CHECK(cudaMemsetAsync(c.bufs["fattn_error"].ptr, 0, 4, c.pstream));
    CUDA_CHECK(cudaStreamSynchronize(c.pstream));
    return n_kv;
}

static void gpu_read_f32(GpuCtx &c, const std::string &name, std::vector<float> &out,
                         size_t n_elems, size_t byte_off = 0) {
    CUDA_CHECK(cudaSetDevice(c.device));
    out.resize(n_elems);
    CUDA_CHECK(cudaMemcpy(out.data(), (char *) c.bufs[name].ptr + byte_off,
                          n_elems * 4, cudaMemcpyDeviceToHost));
}

// Reset the recurrent stream to its post-alloc initial condition so a second
// parity leg replays the same prompt from a pristine sequence: DeltaNet
// conv/ssm banks back to zero (gpu_alloc_buffers zero-fills at alloc) and the
// first pad_up(kv_rows, 256) KV rows of every attention layer back to zero
// (covers every window the replay can open over those rows, so the two legs
// see identical bytes even in the masked padded tail).
static void gpu_reset_stream_state(GpuCtx &c, int64_t kv_rows) {
    CUDA_CHECK(cudaSetDevice(c.device));
    for (int il = 0; il < N_LAYER; il++) {
        if (is_attn_layer(il)) {
            size_t kb = (size_t) pad_up(kv_rows, 256) * (size_t)(N_EMBD_GQA / 2) * 2;
            DevBuf &k = c.bufs["cache_k_l" + std::to_string(il)];
            DevBuf &v = c.bufs["cache_v_l" + std::to_string(il)];
            CUDA_CHECK(cudaMemset(k.ptr, 0, std::min(kb, k.bytes)));
            CUDA_CHECK(cudaMemset(v.ptr, 0, std::min(kb, v.bytes)));
        } else {
            DevBuf &s = c.bufs["ssm_state_l" + std::to_string(il)];
            DevBuf &r = c.bufs["conv_state_l" + std::to_string(il)];
            CUDA_CHECK(cudaMemset(s.ptr, 0, s.bytes));
            CUDA_CHECK(cudaMemset(r.ptr, 0, r.bytes));
        }
    }
}

// Ring both doorbells, then wait both done (bounded; never kills). The padded
// KV window is already in each GPU's "n_kv" cell (gpu_set_inputs); FATTN reads
// it STRONG, so only $seqno + the token span are patched here. n_tok tokens
// (n_tok = 1 for every decode pass; a prefill U-tile passes its whole tile)
// land in consecutive d_token slots; the doorbell rings ONCE either way, so
// the SingleLaunch / NoTeardown discipline is untouched.
static bool dual_run_pass(GpuCtx g2[2], const int32_t *tokens, int64_t n_tok,
                          unsigned seqno, double timeout_ms) {
    if (n_tok < 1 || (size_t) n_tok * 4 > 512) {
        // d_token is a 512 B cell (core/host.cpp): 128 i32 slots, the U=128
        // gate config's whole tile. Refuse rather than overrun it.
        fprintf(stderr, "dual_run_pass: n_tok %lld exceeds the 512 B d_token cell\n",
                (long long) n_tok);
        return false;
    }
    // Per-pass device patch ($seqno) and the token ride the copy stream ASYNC;
    // one sync per GPU collapses the 4-byte copies/pass into 2 stream syncs. The
    // persistent kernel spins at the doorbell between passes, so patching
    // d_program in place is safe.
    static thread_local unsigned s_seq;   // stable source for async copies
    s_seq = seqno;
    for (int g = 0; g < 2; g++) {
        GpuCtx &c = g2[g];
        CUDA_CHECK(cudaSetDevice(c.device));
        if (c.ln.h.h_err[0] != 0) {
            fprintf(stderr, "gpu %d device error %u (aux %u) already set\n",
                    c.device, c.ln.h.h_err[0], c.ln.h.h_err[1]);
            return false;
        }
        for (size_t idx : c.xchg_idx)
            CUDA_CHECK(cudaMemcpyAsync((char *) c.ln.h.d_program + idx * sizeof(mk::Instr) +
                                       GpuCtx::SEQNO_OFF, &s_seq, 4, cudaMemcpyHostToDevice,
                                       c.ln.h.cstream));
        c.ln.h.pass += 1;
        CUDA_CHECK(cudaMemcpyAsync(c.ln.h.d_token, tokens, (size_t) n_tok * 4,
                                   cudaMemcpyHostToDevice, c.ln.h.cstream));
    }
    for (int g = 0; g < 2; g++) {
        CUDA_CHECK(cudaSetDevice(g2[g].device));
        CUDA_CHECK(cudaStreamSynchronize(g2[g].ln.h.cstream));
        std::atomic_thread_fence(std::memory_order_release);
        *g2[g].ln.h.h_doorbell = g2[g].ln.h.pass;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double, std::milli>(timeout_ms);
    for (;;) {
        for (int g = 0; g < 2; g++)
            if (g2[g].ln.h.h_err[0] != 0) {
                fprintf(stderr, "gpu %d device error %u (aux %u) on pass %u\n",
                        g2[g].device, g2[g].ln.h.h_err[0], g2[g].ln.h.h_err[1], g2[g].ln.h.pass);
                return false;
            }
        bool d0 = g2[0].ln.h.h_done[0] == g2[0].ln.h.pass;
        bool d1 = g2[1].ln.h.h_done[0] == g2[1].ln.h.pass;
        if (d0 && d1) return true;
        if (std::chrono::steady_clock::now() > deadline) {
            fprintf(stderr, "dual_run_pass TIMEOUT pass %u: done g0=%d g1=%d "
                    "(not killing — context teardown at exit reclaims)\n",
                    g2[0].ln.h.pass, d0, d1);
            for (int g = 0; g < 2; g++) {
                cudaSetDevice(g2[g].device);
                cudaError_t q = cudaStreamQuery(g2[g].ln.h.kstream);
                fprintf(stderr, "  gpu %d kstream %s\n", g2[g].device,
                        q == cudaSuccess ? "idle" : q == cudaErrorNotReady ? "running"
                                                    : cudaGetErrorString(q));
            }
            return false;
        }
    }
}

// Set up both GPUs from one mmap'd GGUF + the split program. n_sites from the
// program (count of OP_XCHG_REDUCE == count of OP_XCHG_PUSH). u_max sizes the
// per-pass staging buffers for prefill U-tiles (default 1 = decode layout).
static void dual_setup(GpuCtx g2[2], gguf::File &gg, const Jv &program, int64_t n_ctx,
                       int64_t n_vocab_full, int64_t u_max = 1) {
    // peer access both ways (NVLink); tolerate already-enabled.
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            if (a != b) {
                cudaSetDevice(a);
                cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    throw std::runtime_error(std::string("enable peer ") +
                                             std::to_string(a) + "->" + std::to_string(b) +
                                             ": " + cudaGetErrorString(e));
            }
    int n_sites = 0;
    for (auto &I : program.at("instructions").arr)
        if (I.at("kind").str == "OP_XCHG_REDUCE") n_sites++;

    for (int g = 0; g < 2; g++) {
        g2[g].device = g;
        g2[g].gpu_index = g;
        g2[g].n_ctx = n_ctx;
        g2[g].n_vocab_full = n_vocab_full;
    }
    printf("dual: %d cross-GPU reduce sites/pass; uploading sliced weights...\n", n_sites);
    for (int g = 0; g < 2; g++) gpu_upload_weights(g2[g], gg);
    for (int g = 0; g < 2; g++) gpu_alloc_mailboxes(g2[g], n_sites, u_max);
    gpu_wire_mailboxes(g2[0], g2[1], n_sites, u_max);
    gpu_wire_mailboxes(g2[1], g2[0], n_sites, u_max);

    // Phase 1 (BEFORE the ncu range): init, allocate, pack, upload the program, and set
    // the dynamic-smem opt-in on each device. Everything but the launch. The smem opt-in
    // (cuFuncSetAttribute) is an API ncu forbids inside a profiler range, so it must land
    // here; stage_program() sets it, launch() below then skips it.
    for (int g = 0; g < 2; g++) {
        GpuCtx &c = g2[g];
        c.ln.init(g);                                   // host_init on device g
        c.R.add("token", c.ln.d_token(), 512);          // cell:token -> d_token
        gpu_alloc_buffers(c, program, /*skip_kv=*/false, u_max);
        PackedProgram p = pack_program(program, c.R, &c.mtab, c.gpu_index);
        if (!p.complete)
            throw std::runtime_error("gpu " + std::to_string(g) + " pack failed after " +
                                     std::to_string(p.packed_before_failure) + ": " + p.first_failure);
        c.xchg_idx  = p.xchg_idx;
        c.staged    = p;
        c.ln.stage_program(p);
    }

    // ncu RANGE start: after the 21 GB upload + smem opt-in, BEFORE the cooperative
    // mk_interp launch, so the persistent kernel is INSIDE the profiled range (a range
    // excluding the launch is empty; an opt-in inside it is rejected). Driver API = the
    // variant ncu intercepts. Inert without a profiler. Closed by cuProfilerStop() after
    // the timed loop.
    cuProfilerStart();
    for (int g = 0; g < 2; g++) {
        GpuCtx &c = g2[g];
        cudaSetDevice(c.device);                        // launch() targets the current device
        c.ln.launch();
        printf("  gpu %d: packed %zu instrs (%zu FATTN, %zu XCHG_REDUCE), kernel launched\n",
               g, c.staged.instrs.size(), c.staged.fattn_idx.size(), c.staged.xchg_idx.size());
    }
}

static void dual_shutdown(GpuCtx g2[2]) {
    for (int g = 0; g < 2; g++) {
        cudaSetDevice(g2[g].device);
        if (g2[g].ln.program_uploaded) mk::host_shutdown(g2[g].ln.h, 5000.0);
    }
}

// -- tensor parity: match the 2-GPU reference oracle -------------------------
static int parity_run_tensor(gguf::File &gg, const Jv &program, int64_t n_ctx,
                             int64_t n_vocab, const std::string &ref_dir,
                             const std::string &out_dir) {
    Jv ref_idx = json_load(ref_dir + "/index.json");
    if (ref_idx.at("format").str != "mk-oracle/v1")
        throw std::runtime_error("ref tree is not mk-oracle/v1");
    const Jv &cfg = ref_idx.at("config");
    const Jv &prompt = ref_idx.at("prompt_tokens");
    int steps = (int) cfg.at("steps").as_i();
    if (cfg.at("n_vocab").as_i() != n_vocab)
        throw std::runtime_error("ref n_vocab != model n_vocab");
    if ((int64_t) prompt.arr.size() + steps > n_ctx)
        throw std::runtime_error("prompt + steps exceed --n-ctx");
    printf("parity-tensor: ref %s — %zu prompt tokens, %d greedy steps, n_vocab %lld\n",
           ref_dir.c_str(), prompt.arr.size(), steps, (long long) n_vocab);

    GpuCtx g2[2];
    dual_setup(g2, gg, program, n_ctx, n_vocab);

    DumpTree dump;
    dump.open(out_dir, n_vocab);
    const size_t HALF = (size_t) n_vocab / 2;

    std::vector<float> lo0, lo1, logits(n_vocab), lout, rnorm, s0, s1, state;
    std::vector<int32_t> emitted;
    int64_t pos = 0;
    unsigned pass = 0;

    auto gather_logits = [&]() {
        gpu_read_f32(g2[0], "logits", lo0, HALF);
        gpu_read_f32(g2[1], "logits", lo1, HALF);
        memcpy(logits.data(), lo0.data(), HALF * 4);
        memcpy(logits.data() + HALF, lo1.data(), HALF * 4);
    };

    // prefill: one pass per prompt token
    for (size_t i = 0; i < prompt.arr.size(); i++, pos++) {
        gpu_set_inputs(g2[0], pos);
        gpu_set_inputs(g2[1], pos);
        ++pass;
        int32_t ptok = (int32_t) prompt.arr[i].as_i();
        if (!dual_run_pass(g2, &ptok, 1, pass, 60000.0))
            return 3;
    }
    gather_logits();
    dump.add_logits_row(logits);

    for (int s = 0; s < steps; s++, pos++) {
        int32_t tok = argmax_f32_first(logits);
        emitted.push_back(tok);
        gpu_set_inputs(g2[0], pos);
        gpu_set_inputs(g2[1], pos);
        ++pass;
        if (!dual_run_pass(g2, &tok, 1, pass, 60000.0)) return 3;
        gather_logits();
        dump.add_logits_row(logits);

        char rel[128], name[64];
        gpu_read_f32(g2[0], "dbg_lout", lout, (size_t) N_LOUT * N_EMBD);  // mirrored
        for (int il = 0; il < N_LOUT; il++) {
            Stats st = stats_of(lout.data() + (size_t) il * N_EMBD, N_EMBD);
            snprintf(name, sizeof(name), "l_out-%d", il);
            snprintf(rel, sizeof(rel), "full/s%d/l_out-%d.bin", s, il);
            dump.node_row(s, il, name, "ADD", N_EMBD, st);
            dump.write_bin(s, "full", name, "ADD", rel, lout.data() + (size_t) il * N_EMBD, N_EMBD);
        }
        gpu_read_f32(g2[0], "xn", rnorm, N_EMBD);
        dump.node_row(s, 3702, "result_norm", "MUL", N_EMBD, stats_of(rnorm.data(), N_EMBD));
        dump.node_row(s, 3703, "result_output", "MUL_MAT", (size_t) n_vocab,
                      stats_of(logits.data(), logits.size()));
        // DeltaNet state: gather both GPUs' head halves to full size (bytes
        // will not match the fork's fold — the state lane fails definitionally,
        // as single-GPU; keys and sizes match so compare.py is not structural).
        for (int il = 0; il < N_LAYER; il++) {
            if (is_attn_layer(il)) continue;
            gpu_read_f32(g2[0], "ssm_state_l" + std::to_string(il), s0, SSM_STATE_N / 2);
            gpu_read_f32(g2[1], "ssm_state_l" + std::to_string(il), s1, SSM_STATE_N / 2);
            state.resize(SSM_STATE_N);
            memcpy(state.data(), s0.data(), (SSM_STATE_N / 2) * 4);
            memcpy(state.data() + SSM_STATE_N / 2, s1.data(), (SSM_STATE_N / 2) * 4);
            snprintf(name, sizeof(name), "cache_s_l%d", il);
            snprintf(rel, sizeof(rel), "state/s%d/cache_s_l%d.bin", s, il);
            dump.write_bin(s, "state", name, "CPY", rel, state.data(), SSM_STATE_N);
            gpu_read_f32(g2[0], "conv_state_l" + std::to_string(il), s0, CONV_STATE_N / 2);
            gpu_read_f32(g2[1], "conv_state_l" + std::to_string(il), s1, CONV_STATE_N / 2);
            state.resize(CONV_STATE_N);
            memcpy(state.data(), s0.data(), (CONV_STATE_N / 2) * 4);
            memcpy(state.data() + CONV_STATE_N / 2, s1.data(), (CONV_STATE_N / 2) * 4);
            snprintf(name, sizeof(name), "cache_r_l%d", il);
            snprintf(rel, sizeof(rel), "state/s%d/cache_r_l%d.bin", s, il);
            dump.write_bin(s, "state", name, "CPY", rel, state.data(), CONV_STATE_N);
        }
    }

    dump.finish(ref_idx, n_ctx, emitted);
    dual_shutdown(g2);
    printf("parity-tensor: dump tree written to %s\n", out_dir.c_str());
    printf("compare with:\n  python3 tests/oracle/compare.py %s %s --allow-config-mismatch --report %s/parity-report.json\n",
           ref_dir.c_str(), out_dir.c_str(), out_dir.c_str());
    return 0;
}

// Resident set size of this process (MB), for the soak leak watermark.
static double rss_mb() {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0.0;
    long total_pages = 0, rss_pages = 0;
    if (fscanf(f, "%ld %ld", &total_pages, &rss_pages) != 2) rss_pages = 0;
    fclose(f);
    return rss_pages * (double)(sysconf(_SC_PAGESIZE)) / 1e6;
}

// One watermark line: free VRAM on each GPU + host RSS. A persistent-kernel
// leak shows as any of these drifting monotonically over the soak.
static void watermark(GpuCtx g2[2], const char *tag, int64_t pos) {
    size_t f0 = 0, f1 = 0, tt = 0;
    cudaSetDevice(g2[0].device); cudaMemGetInfo(&f0, &tt);
    cudaSetDevice(g2[1].device); cudaMemGetInfo(&f1, &tt);
    printf("watermark %-6s pos %8lld: gpu0 free %.1f MB, gpu1 free %.1f MB, host RSS %.1f MB\n",
           tag, (long long) pos, f0 / 1e6, f1 / 1e6, rss_mb());
    fflush(stdout);
}

// REDLINE itemization: per-op-kind clock64 breakdown from block 0. Only an
// MK_PROFILE-built kernel fills op_cycles (otherwise all zero -> no print). The
// caller resets after warmup; here we read back, convert cycles->ms per pass at
// 1.455 GHz, and print ms + %-of-pass per kind, the boundary total, the reduce
// total, and the on-device sum cross-checked against the pass_cycles wall.
static const char *KIND_NAME[mk::OP_KIND_COUNT] = {
    "NOP", "BOUNDARY", "EMBED_LOOKUP", "RMSNORM", "QUANT_Q8_1",
    "HEAD_GEMV_F16", "LOGITS_EMIT", "MMVQ_Q4_0", "MMVQ_Q4_0_FUSED",
    "MMVQ_AR16", "GEMV_F16", "CONV_SHIFT_CONCAT", "SSM_CONV_SILU",
    "QK_L2NORM", "GDN_GATES", "GDN_STEP", "GATED_RMSNORM", "QK_NORM_ROPE",
    "KV_APPEND", "FATTN_DECODE", "FATTN_REDUCE", "ATTN_GATE", "RESIDUAL_ADD",
    "STATE_LOAD", "STATE_STORE", "XCHG_PUSH", "XCHG_REDUCE"};

static const char *FATTN_PHASE_NAME[mk::MK_FATTN_NPHASE] = {
    "KLOAD", "QK_HMMA", "SOFTMAX", "VLOAD", "PV_HMMA", "SETUP+TAIL"};

static void print_op_breakdown(mk::Host &h, int gpu, int64_t n_tokens,
                               double pass_ms) {
    long long oc[mk::MK_OP_CYCLES_LEN] = {0};
    if (!mk::host_read_op_cycles(h, oc, mk::MK_OP_CYCLES_LEN)) return;
    long long sum = 0;
    for (int k = 0; k < mk::OP_KIND_COUNT; k++) sum += oc[k];
    if (sum == 0) return;  // non-profile kernel: nothing recorded
    const double gHz = 1.455;
    auto ms = [&](long long c) { return (double) c / n_tokens / gHz / 1e6; };
    const double total_ms = ms(sum);
    printf("PROFILE gpu %d: block-0 itemization over %lld passes "
           "(on-device sum %.3f ms/pass vs pass_cycles wall %.3f ms)\n",
           gpu, (long long) n_tokens, total_ms, pass_ms);
    printf("  %-18s %10s %8s\n", "kind", "ms/pass", "%pass");
    // sort by descending cycles
    int order[mk::OP_KIND_COUNT];
    for (int k = 0; k < mk::OP_KIND_COUNT; k++) order[k] = k;
    for (int a = 0; a < mk::OP_KIND_COUNT; a++)
        for (int b = a + 1; b < mk::OP_KIND_COUNT; b++)
            if (oc[order[b]] > oc[order[a]]) { int t = order[a]; order[a] = order[b]; order[b] = t; }
    double reduce_ms = ms(oc[mk::OP_XCHG_PUSH] + oc[mk::OP_XCHG_REDUCE]);
    for (int i = 0; i < mk::OP_KIND_COUNT; i++) {
        int k = order[i];
        if (oc[k] == 0) continue;
        printf("  %-18s %10.4f %7.2f%%\n", KIND_NAME[k], ms(oc[k]),
               100.0 * oc[k] / sum);
    }
    printf("  ---- boundary total %.4f ms (%.2f%%), reduce total (push+reduce) "
           "%.4f ms (%.2f%%)\n",
           ms(oc[mk::OP_BOUNDARY]), 100.0 * oc[mk::OP_BOUNDARY] / sum,
           reduce_ms, 100.0 * (oc[mk::OP_XCHG_PUSH] + oc[mk::OP_XCHG_REDUCE]) / sum);
    // FATTN_DECODE sub-phase decomposition (block-0 thread-0 clock64 laps in
    // op_fattn_decode; slots OP_KIND_COUNT..). % is of the FATTN_DECODE total so
    // it reads as "where the dominant op's time goes". Sums to ~FATTN_DECODE
    // modulo the one-thread sample vs whole-op-wrapper timing.
    long long fsum = 0;
    for (int p = 0; p < mk::MK_FATTN_NPHASE; p++) fsum += oc[mk::OP_KIND_COUNT + p];
    if (fsum > 0) {
        long long fdec = oc[mk::OP_FATTN_DECODE];
        printf("  FATTN_DECODE sub-phases (thread-0 laps; %% of FATTN_DECODE %.4f ms):\n",
               ms(fdec));
        for (int p = 0; p < mk::MK_FATTN_NPHASE; p++) {
            long long c = oc[mk::OP_KIND_COUNT + p];
            printf("    %-12s %10.4f %7.2f%%\n", FATTN_PHASE_NAME[p], ms(c),
                   fdec > 0 ? 100.0 * c / fdec : 0.0);
        }
    }
    fflush(stdout);
}

// plan/0144 milestone 2: per-op schedule-known DRAM bytes + compute ops, attached
// host-side (the spec keeps rates out of the device record). Covers the DRAM-dominant
// ops: weight matmuls (weight tensor size from GGUF, scaled by the row split) and
// FATTN_DECODE (KV read = kv_heads*head_dim*n_kv*2(K+V)*2(f16); ops = QK+PV MACs).
// Other ops return 0 = "column absent" (the ingest emits no MemoryBw/Pipe demand).
static std::pair<uint64_t, uint64_t> op_bytes_ops(const Jv &in, const gguf::File &gg,
                                                  int64_t n_kv) {
    const Jv *ka = in.get("args");
    if (!ka) return {0, 0};
    const Jv &a = *ka;
    const std::string &kind = in.at("kind").str;
    auto ai = [&](const char *k, int64_t d) -> int64_t {
        const Jv *v = a.get(k);
        return v && v->k == Jv::NUM ? (int64_t) v->num : d;
    };
    if (kind == "OP_MMVQ_Q4_0" || kind == "OP_MMVQ_Q4_0_FUSED" ||
        kind == "OP_GEMV_F16" || kind == "OP_HEAD_GEMV_F16") {
        const Jv *w = a.get("weight");
        if (!w || w->k != Jv::STR) return {0, 0};
        std::string name = w->str;
        auto c = name.find(':');
        if (c != std::string::npos) name = name.substr(c + 1);  // strip "gguf:"
        const gguf::TensorInfo *t = gg.find(name);
        if (!t) return {0, 0};
        int64_t rl = ai("row_lo", -1), rh = ai("row_hi", -1);
        uint64_t bytes = t->nbytes;
        if (rl >= 0 && rh > rl && t->ne[1] > 0)  // per-GPU row slice
            bytes = (uint64_t)((double) t->nbytes * (double)(rh - rl) / (double) t->ne[1]);
        return {bytes, 0};
    }
    if (kind == "OP_FATTN_DECODE") {
        int64_t kvh = ai("kv_heads", 0), qh = ai("q_heads", 0), hd = ai("head_dim", 0);
        uint64_t kv_bytes = (uint64_t) kvh * hd * (uint64_t) n_kv * 2ull * 2ull;
        uint64_t ops = 2ull * (uint64_t) qh * hd * (uint64_t) n_kv;  // QK + PV MACs
        return {kv_bytes, ops};
    }
    return {0, 0};
}

// plan/0144 primitive telemetry dump: the last timed pass's per-op {gt_start_ns,
// gt_end_ns, cycles} (block 0) + the block->SM residency census. Writes two TSV
// files at $MK_TELE.gpu<g>.{tele,smid}; calx-mill's `telemetry` mode ingests them
// as measured-SteadyState anchors. bytes/ops are attached in a later pass (0 here =
// "column absent", which the ingest treats as no MemoryBw/Pipe demand). op_kind and
// lane come from the packed program; op_index keys the record to the schedule.
static void dump_telemetry(GpuCtx &c, int gpu, const char *prefix, const Jv &program,
                           const gguf::File &gg, int64_t n_kv) {
    mk::Host &h = c.ln.h;
    unsigned n = h.op_tele_cap;
    if (!n) return;
    std::vector<long long> te((size_t)n * 3, 0);
    if (!mk::host_read_op_tele(h, te.data(), n)) return;
    std::vector<unsigned> smid(mk::GRID_BLOCKS, 0xffffffffu);
    mk::host_read_smid_census(h, smid.data(), mk::GRID_BLOCKS);
    const Jv &pins = program.at("instructions");
    // A json op expands to >=1 packed instrs (splits). Share its bytes/ops evenly
    // across the packed instrs that came from it, so each packed record gets its slice.
    const auto &src = c.staged.src_json;
    std::map<uint32_t, uint32_t> share;
    for (uint32_t s : src) share[s]++;

    char path[600];
    snprintf(path, sizeof path, "%s.gpu%d.tele", prefix, gpu);
    FILE *f = fopen(path, "w");
    if (!f) { perror("MK_TELE open"); return; }
    fprintf(f, "op_index\top_kind\tlane\tgt_start_ns\tgt_end_ns\tcycles\tbytes\tops\n");
    const auto &instrs = c.staged.instrs;
    int emitted = 0;
    long long tele_cyc_sum = 0;
    for (unsigned i = 0; i < n; i++) {
        long long gs = te[(size_t)i * 3 + 0], ge = te[(size_t)i * 3 + 1],
                  cy = te[(size_t)i * 3 + 2];
        if (gs == 0 && ge == 0 && cy == 0) continue;  // op did not run on block 0
        unsigned kind = i < instrs.size() ? instrs[i].kind : 0xffffu;
        const char *kn = kind < mk::OP_KIND_COUNT ? KIND_NAME[kind] : "?";
        const char *lane = (kind == mk::OP_FATTN_DECODE) ? "compute" : "mem";
        uint64_t bytes = 0, ops = 0;
        if (i < src.size() && src[i] < pins.arr.size()) {
            uint32_t j = src[i];
            auto bo = op_bytes_ops(pins.arr[j], gg, n_kv);
            uint32_t sh = share[j] ? share[j] : 1;
            bytes = bo.first / sh;  // this packed instr's slice of the json op
            ops = bo.second / sh;
        }
        fprintf(f, "%u\t%s\t%s\t%lld\t%lld\t%lld\t%llu\t%llu\n", i, kn, lane, gs, ge, cy,
                (unsigned long long) bytes, (unsigned long long) ops);
        emitted++;
        tele_cyc_sum += cy;
    }
    fclose(f);

    snprintf(path, sizeof path, "%s.gpu%d.smid", prefix, gpu);
    int distinct = 0;
    bool used[256] = {false};
    if ((f = fopen(path, "w"))) {
        fprintf(f, "block_id\tsmid\n");
        for (unsigned b = 0; b < mk::GRID_BLOCKS; b++) {
            fprintf(f, "%u\t%u\n", b, smid[b]);
            if (smid[b] < 256 && !used[smid[b]]) { used[smid[b]] = true; distinct++; }
        }
        fclose(f);
    }
    printf("TELE gpu %d: %d op records (cyc-sum %lld) + %u-block census (%d distinct SMs)"
           " -> %s.gpu%d.{tele,smid}\n",
           gpu, emitted, tele_cyc_sum, mk::GRID_BLOCKS, distinct, prefix, gpu);
    fflush(stdout);
}

// -- tensor bench / soak: dual-GPU decode floor + leak watermark --------------
// pos0 seeds the starting decode position; at pos0 near n_ctx the KV read (~8.6
// GB/GPU) dominates, giving the true deep-context floor. Long n_tokens with the
// periodic watermark is the 256K soak (device-free + RSS must be flat).
static int bench_run_tensor(gguf::File &gg, const Jv &program, int64_t n_ctx,
                            int64_t n_vocab, int64_t n_tokens, int64_t pos0) {
    GpuCtx g2[2];
    dual_setup(g2, gg, program, n_ctx, n_vocab);  // opens the ncu range internally (post-upload, pre-launch)
    const int WARMUP = 32;
    if (pos0 + (int64_t) WARMUP + n_tokens > n_ctx)
        throw std::runtime_error("pos0 + warmup + N exceed --n-ctx");
    printf("bench-tensor: %d warmup + %lld timed dual-GPU decode passes from pos0 %lld\n",
           WARMUP, (long long) n_tokens, (long long) pos0);
    int64_t pos = pos0;
    unsigned pass = 0;
    int32_t btok = 11;
    for (int i = 0; i < WARMUP; i++, pos++) {
        gpu_set_inputs(g2[0], pos);
        gpu_set_inputs(g2[1], pos);
        ++pass;
        if (!dual_run_pass(g2, &btok, 1, pass, 60000.0)) return 3;
    }
    watermark(g2, "warmed", pos);
    // REDLINE: zero the per-kind accumulator so the itemization covers only the
    // timed region (safe here — both kernels are spinning at the doorbell).
    for (int g = 0; g < 2; g++) mk::host_reset_op_cycles(g2[g].ln.h);
    const int64_t sample_every = n_tokens > 20 ? n_tokens / 20 : 1;
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < n_tokens; i++, pos++) {
        gpu_set_inputs(g2[0], pos);
        gpu_set_inputs(g2[1], pos);
        ++pass;
        if (!dual_run_pass(g2, &btok, 1, pass, 60000.0)) return 3;
        if ((i + 1) % sample_every == 0) watermark(g2, "soak", pos);
    }
    auto t1 = std::chrono::steady_clock::now();
    cuProfilerStop();   // close the ncu range
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("bench-tensor: %lld tokens in %.3f s = %.2f tok/s "
           "[contended-indicative: host-coordinated 2-GPU]\n",
           (long long) n_tokens, sec, n_tokens / sec);
    // per-pass on-device clock64 from each GPU's block-0 G15 ring.
    for (int g = 0; g < 2; g++) {
        unsigned cap = g2[g].ln.h.pass_cycles_cap;
        if (!cap || n_tokens <= 0) continue;
        std::vector<long long> cyc(cap);
        double pass_ms = 0;
        if (mk::host_read_pass_cycles(g2[g].ln.h, cyc.data(), cap)) {
            unsigned cnt = (unsigned) std::min<int64_t>(n_tokens, cap);
            double sum = 0; long long mn = cyc[0], mx = cyc[0];
            for (unsigned k = 0; k < cnt; k++) {
                long long v = cyc[(g2[g].ln.h.pass - 1 - k) % cap];
                sum += (double) v; if (v < mn) mn = v; if (v > mx) mx = v;
            }
            const double gHz = 1.455;
            pass_ms = sum / cnt / gHz / 1e6;
            printf("bench-tensor: gpu %d per-pass on-device clock64: mean %.3f ms, "
                   "min %.3f, max %.3f (%u samples)\n",
                   g, pass_ms, mn / gHz / 1e6, mx / gHz / 1e6, cnt);
        }
        print_op_breakdown(g2[g].ln.h, g, n_tokens, pass_ms);
        if (const char *tp = getenv("MK_TELE"))
            dump_telemetry(g2[g], g, tp, program, gg, pos0);  // n_kv ~= pos0 at deep decode
    }
    dual_shutdown(g2);
    return 0;
}

// The program's batched-prefill width: meta.prefill.n_tokens (the key
// compile_schedule.py --prefill actually emits; the same one pack_program
// divides buffer slots by). A decode program has no prefill block -> 1.
static int64_t program_prefill_u(const Jv &program) {
    if (const Jv *m = program.get("meta"))
        if (const Jv *pf = m->get("prefill"))
            if (const Jv *nt = pf->get("n_tokens")) return nt->as_i();
    return 1;
}

// -- prefill U-tile self-consistency parity (plan/0143 driver scaffold) ------
// Same resident kernel, same program, same prompt, run twice: the reference
// leg through the proven per-token staging (gpu_set_inputs, one doorbell per
// token), the tile leg through the tile staging (gpu_set_inputs_tile, one
// doorbell per U-tile). At U=1 the two legs are the same computation op for
// op, so logits, DeltaNet conv/ssm state, and the appended KV rows must be
// BIT-IDENTICAL (tol 0.0); a mismatch is a staging bug in the tile path, not
// an op regression. U>1 needs a program whose ops consume n_tokens
// (meta.prefill.n_tokens, compile_schedule.py --prefill U); with it the same
// memcmp binds the U-loop ops to the per-token fold, and the changed-tail
// causality micro-check arms (a transposed or last-row-broadcast mask passes
// every U=1 check and fails only there).
static int prefill_parity_run(gguf::File &gg, const Jv &program, int64_t n_ctx,
                              int64_t n_vocab, int64_t U) {
    // 128 tokens so the GATE config (U=128, one full tile) binds bit-exactly
    // against 128 per-token decode passes, not a shallow stand-in (drift
    // tripwire: shallow-as-deep). The first 8 ids are the original scaffold
    // prompt (capture/prefill-parity-u1.txt continuity); the tail walks odd
    // ids so no two tokens repeat and a row-swap cannot alias.
    static int32_t PROMPT[128];
    PROMPT[0] = 11;
    for (int i = 1; i < 8; i++) PROMPT[i] = 500 * i;
    for (int i = 8; i < 128; i++) PROMPT[i] = 4001 + 2 * i;
    const int64_t P = 128;
    if (U < 1 || U > P)
        throw std::runtime_error("--prefill-parity: U must be in 1.." + std::to_string(P));
    if (U > 1 && program_prefill_u(program) < U)
        throw std::runtime_error("--prefill-parity: U>1 needs a prefill program "
                                 "(python3 k0/compile_schedule.py --prefill U --split) whose "
                                 "meta.prefill.n_tokens >= U; the decode ops pack n_tokens=1");
    if (P + 1 > n_ctx) throw std::runtime_error("prompt exceeds --n-ctx");

    GpuCtx g2[2];
    dual_setup(g2, gg, program, n_ctx, n_vocab, U);
    const size_t HALF = (size_t) n_vocab / 2;
    unsigned pass = 0;

    std::vector<float> lo0, lo1, tmp;
    auto gather_logits = [&](std::vector<float> &dst) {
        gpu_read_f32(g2[0], "logits", lo0, HALF);
        gpu_read_f32(g2[1], "logits", lo1, HALF);
        dst.resize((size_t) n_vocab);
        memcpy(dst.data(), lo0.data(), HALF * 4);
        memcpy(dst.data() + HALF, lo1.data(), HALF * 4);
    };
    // KV snapshot: the first n_rows appended rows of each attention layer's K
    // and V cache, both GPUs (f16 rows read raw as 4-byte words: 512 f16 =
    // 256 words/row). Bit-compare only, never arithmetic.
    auto snap_kv = [&](std::vector<float> &dst, int64_t n_rows) {
        dst.clear();
        const size_t row_words = (size_t)(N_EMBD_GQA / 2) / 2;
        for (int g = 0; g < 2; g++)
            for (int il = 0; il < N_LAYER; il++) {
                if (!is_attn_layer(il)) continue;
                gpu_read_f32(g2[g], "cache_k_l" + std::to_string(il), tmp,
                             (size_t) n_rows * row_words);
                dst.insert(dst.end(), tmp.begin(), tmp.end());
                gpu_read_f32(g2[g], "cache_v_l" + std::to_string(il), tmp,
                             (size_t) n_rows * row_words);
                dst.insert(dst.end(), tmp.begin(), tmp.end());
            }
    };
    // recurrent snapshot: DeltaNet ssm + conv banks, both GPUs' halves.
    auto snap_rs = [&](std::vector<float> &dst) {
        dst.clear();
        for (int g = 0; g < 2; g++)
            for (int il = 0; il < N_LAYER; il++) {
                if (is_attn_layer(il)) continue;
                gpu_read_f32(g2[g], "ssm_state_l" + std::to_string(il), tmp, SSM_STATE_N / 2);
                dst.insert(dst.end(), tmp.begin(), tmp.end());
                gpu_read_f32(g2[g], "conv_state_l" + std::to_string(il), tmp, CONV_STATE_N / 2);
                dst.insert(dst.end(), tmp.begin(), tmp.end());
            }
    };
    auto run_tiles = [&](const int32_t *toks) -> bool {
        for (int64_t base = 0; base < P; base += U) {
            int64_t n_tok = std::min(U, P - base);
            gpu_set_inputs_tile(g2[0], base, n_tok);
            gpu_set_inputs_tile(g2[1], base, n_tok);
            ++pass;
            if (!dual_run_pass(g2, toks + base, n_tok, pass, 60000.0)) return false;
            fprintf(stderr, "  tile +%lld done (base %lld)\n", (long long) n_tok, (long long) base);
        }
        return true;
    };
    // bit-compare as raw 4-byte words (memcmp, NaN-safe); -1 = identical,
    // else the first differing word index (or min size on length mismatch).
    auto first_diff = [](const std::vector<float> &a, const std::vector<float> &b) -> long long {
        if (a.size() != b.size()) return (long long) std::min(a.size(), b.size());
        for (size_t i = 0; i < a.size(); i++)
            if (memcmp(&a[i], &b[i], 4) != 0) return (long long) i;
        return -1;
    };

    // reference leg: P per-token passes through the decode staging.
    printf("prefill-parity: U=%lld, prompt %lld tokens, reference leg (per-token)...\n",
           (long long) U, (long long) P);
    for (int64_t i = 0; i < P; i++) {
        gpu_set_inputs(g2[0], i);
        gpu_set_inputs(g2[1], i);
        ++pass;
        int32_t tk = PROMPT[i];
        if (!dual_run_pass(g2, &tk, 1, pass, 60000.0)) return 3;
        if ((i & 15) == 15)   // unbuffered heartbeat: a killed run localizes itself
            fprintf(stderr, "  ref pass %lld/%lld\n", (long long)(i + 1), (long long) P);
    }
    std::vector<float> ref_logits, ref_kv, ref_rs;
    gather_logits(ref_logits);
    snap_kv(ref_kv, P);
    snap_rs(ref_rs);

    // tile leg: same prompt in U-tiles from a re-zeroed stream.
    printf("prefill-parity: tile leg (%lld-token tiles, 1 doorbell per tile)...\n",
           (long long) U);
    gpu_reset_stream_state(g2[0], P);
    gpu_reset_stream_state(g2[1], P);
    if (!run_tiles(PROMPT)) return 3;
    std::vector<float> tile_logits, tile_kv, tile_rs, tile_kv_head;
    gather_logits(tile_logits);
    snap_kv(tile_kv, P);
    snap_rs(tile_rs);
    if (U >= 2) snap_kv(tile_kv_head, P - 1);

    long long dl = first_diff(ref_logits, tile_logits);
    long long dkv = first_diff(ref_kv, tile_kv);
    long long drs = first_diff(ref_rs, tile_rs);

    // causality micro-check (arms at U>=2): change ONLY the last prompt token
    // and rerun the tiles; the KV rows of tokens 0..P-2 must not move (token t
    // may depend on tokens 0..t ONLY). The final states and logits legitimately
    // differ and are not compared here.
    long long dc = -1;
    if (U >= 2) {
        std::vector<int32_t> prompt2(PROMPT, PROMPT + P);
        prompt2[(size_t)(P - 1)] = 42;
        gpu_reset_stream_state(g2[0], P);
        gpu_reset_stream_state(g2[1], P);
        if (!run_tiles(prompt2.data())) return 3;
        std::vector<float> pert_kv_head;
        snap_kv(pert_kv_head, P - 1);
        dc = first_diff(tile_kv_head, pert_kv_head);
    }

    bool ok = dl < 0 && dkv < 0 && drs < 0 && dc < 0;
    printf("PREFILL-PARITY U=%lld P=%lld: logits %s, kv-rows %s, rs-state %s, causality %s -> %s\n",
           (long long) U, (long long) P,
           dl < 0 ? "BIT-IDENTICAL" : "DIFF",
           dkv < 0 ? "BIT-IDENTICAL" : "DIFF",
           drs < 0 ? "BIT-IDENTICAL" : "DIFF",
           U >= 2 ? (dc < 0 ? "HOLDS" : "VIOLATED") : "n/a (needs U>=2)",
           ok ? "PASS" : "FAIL");
    if (dl >= 0) fprintf(stderr, "  logits first diff at word %lld\n", dl);
    if (dkv >= 0) fprintf(stderr, "  kv rows first diff at word %lld\n", dkv);
    if (drs >= 0) fprintf(stderr, "  rs state first diff at word %lld\n", drs);
    if (dc >= 0) fprintf(stderr, "  causality: earlier token's KV moved at word %lld\n", dc);
    dual_shutdown(g2);
    return ok ? 0 : 4;
}

// -- prefill U-tile bench: n_tokens from an empty context, one doorbell per
// tile. No warmup: prefill is a ramp, not a steady state. At U=1 tok/s must
// match --bench-tensor with --pos0 0 within noise (the degeneracy check in
// the plan/0143 prefill build plan). At U>1, once the U-loop program lands,
// this is the first measured prefill wall; MK_TELE dumps the last pass's
// per-op telemetry as the calx-mill prefill anchor.
static int prefill_bench_run(gguf::File &gg, const Jv &program, int64_t n_ctx,
                             int64_t n_vocab, int64_t U, int64_t n_tokens) {
    if (U < 1) throw std::runtime_error("--prefill-bench: U must be >= 1");
    if (n_tokens < U) throw std::runtime_error("--prefill-bench: token count < U");
    if (n_tokens > n_ctx) throw std::runtime_error("--prefill-bench: n_tokens exceed --n-ctx");
    if (U > 1 && program_prefill_u(program) < U)
        throw std::runtime_error("--prefill-bench: U>1 needs a prefill program "
                                 "(python3 k0/compile_schedule.py --prefill U --split) whose "
                                 "meta.prefill.n_tokens >= U; the decode ops pack n_tokens=1");
    GpuCtx g2[2];
    dual_setup(g2, gg, program, n_ctx, n_vocab, U);
    const int64_t n_tiles = (n_tokens + U - 1) / U;
    printf("prefill-bench: %lld tokens in %lld tiles of %lld (1 doorbell per tile)\n",
           (long long) n_tokens, (long long) n_tiles, (long long) U);
    std::vector<int32_t> toks((size_t) U, 11);   // fixed id, as --bench-tensor
    unsigned pass = 0;
    const int64_t sample_every = n_tiles > 20 ? n_tiles / 20 : 1;
    int64_t last_n_kv = 0, tile_i = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t base = 0; base < n_tokens; base += U, tile_i++) {
        int64_t n_tok = std::min(U, n_tokens - base);
        last_n_kv = gpu_set_inputs_tile(g2[0], base, n_tok);
        gpu_set_inputs_tile(g2[1], base, n_tok);
        ++pass;
        if (!dual_run_pass(g2, toks.data(), n_tok, pass, 60000.0)) return 3;
        if ((tile_i + 1) % sample_every == 0) watermark(g2, "tile", base + n_tok);
    }
    auto t1 = std::chrono::steady_clock::now();
    cuProfilerStop();   // close the ncu range opened by dual_setup
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("prefill-bench: %lld tokens (%lld passes) in %.3f s = %.2f tok/s "
           "[ramp 0..%lld; host-staged mask, see the diff-plan staging note]\n",
           (long long) n_tokens, (long long) n_tiles, sec, n_tokens / sec,
           (long long) n_tokens);
    for (int g = 0; g < 2; g++) {
        unsigned cap = g2[g].ln.h.pass_cycles_cap;
        double pass_ms = 0;
        if (cap && n_tiles > 0) {
            std::vector<long long> cyc(cap);
            if (mk::host_read_pass_cycles(g2[g].ln.h, cyc.data(), cap)) {
                unsigned cnt = (unsigned) std::min<int64_t>(n_tiles, cap);
                double sum = 0;
                for (unsigned k = 0; k < cnt; k++)
                    sum += (double) cyc[(g2[g].ln.h.pass - 1 - k) % cap];
                pass_ms = sum / cnt / 1.455 / 1e6;
                printf("prefill-bench: gpu %d per-pass on-device clock64 mean %.3f ms "
                       "(%u samples, whole ramp, no warmup cut)\n", g, pass_ms, cnt);
            }
        }
        print_op_breakdown(g2[g].ln.h, g, n_tiles, pass_ms);
        if (const char *tp = getenv("MK_TELE"))
            dump_telemetry(g2[g], g, tp, program, gg, last_n_kv);
    }
    dual_shutdown(g2);
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

// ============================================================================
// MK backend entry points (see integration/dual_core.h). Compiled into
// libggml-mk.so with -DMK_NO_MAIN. Reuse the static dual core above verbatim;
// only the weight/KV source changes: bind llama's extracted per-GPU pointers
// (weights + KV/state, in place) into the Resolver, scratch stays MK-owned.
// ============================================================================
#include "../integration/dual_core.h"

static GpuCtx   g_mk[2];
static bool     g_mk_up = false;
static int64_t  g_mk_nvocab = 0;
static unsigned g_mk_seqno = 0;

// Bind llama's per-GPU pointers into c's Resolver under the program's expected
// names (name-map cache_r->conv_state, cache_s->ssm_state). Skip the view /
// reshaped variants (names containing a space); bind each canonical name once.
static void gpu_bind_llama(GpuCtx &c, const MkPtrMap &pm) {
    for (auto &kv : pm.p) {
        const std::string &n = kv.first;
        if (n.find(' ') != std::string::npos) continue;
        // Bind ONLY GGUF model tensors (blk.* / output* / token_embd*) + KV/state;
        // intermediate/output nodes (result_output, norm-*, Vcur-*, MK#...) are
        // MK-owned scratch or the gather target, not inputs. GGUF names carry a
        // stable prefix that computed-node names never do.
        bool is_gguf  = n.rfind("blk.", 0) == 0 || n.rfind("output", 0) == 0 ||
                        n.rfind("token_embd", 0) == 0;
        bool is_cache = n.rfind("cache_k_l", 0) == 0 || n.rfind("cache_v_l", 0) == 0 ||
                        n.rfind("cache_r_l", 0) == 0 || n.rfind("cache_s_l", 0) == 0;
        if (!is_gguf && !is_cache) continue;
        void *ptr = kv.second[c.gpu_index];
        if (!ptr) continue;
        std::string bind = n;
        if      (n.rfind("cache_r_l", 0) == 0) bind = "conv_state_l" + n.substr(9);
        else if (n.rfind("cache_s_l", 0) == 0) bind = "ssm_state_l"  + n.substr(9);
        if (!c.R.table.count(bind)) c.R.add(bind, ptr, 0);
    }
}

void mk_dual_setup(const MkPtrMap &pm, const char *program_path,
                   int64_t n_ctx, int64_t n_vocab) {
    if (g_mk_up) return;
    Jv program = json_load(program_path);
    // Strip the leading OP_EMBED_LOOKUP: token_embd is not in MK's split-1 graph
    // (llama does the embed on the CPU side), so pack would fail on it. MK seeds
    // buf:residual from MK#model.input_embed#0 each pass instead.
    // MK_OWNEMBED: keep the embed op (bind token_embd below) to test the seed/strip
    // path; default strips it and seeds buf:residual from MK#model.input_embed#0.
    if (!getenv("MK_OWNEMBED"))
      for (auto &kv : program.obj)
        if (kv.first == "instructions" && !kv.second.arr.empty()) {
            const Jv *knd = kv.second.arr.front().get("kind");
            if (knd && knd->str == "OP_EMBED_LOOKUP") {
                kv.second.arr.erase(kv.second.arr.begin());
                fprintf(stderr, "[MK dual] stripped leading OP_EMBED_LOOKUP; "
                                "residual seeded from the CPU embed\n");
            }
            break;
        }
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            if (a != b) {
                cudaSetDevice(a);
                cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    throw std::runtime_error("mk enable peer access");
            }
        if (program_prefill_u(program) > 1)
        throw std::runtime_error("mk backend: prefill programs are not servable through "
                                 "the .so path (its mailboxes/staging are u_max=1); "
                                 "decode program only");
    int n_sites = 0;
    for (auto &I : program.at("instructions").arr)
        if (I.at("kind").str == "OP_XCHG_REDUCE") n_sites++;
    for (int g = 0; g < 2; g++) {
        g_mk[g].device = g; g_mk[g].gpu_index = g;
        g_mk[g].n_ctx = n_ctx; g_mk[g].n_vocab_full = n_vocab;
    }
    for (int g = 0; g < 2; g++) gpu_alloc_mailboxes(g_mk[g], n_sites);
    gpu_wire_mailboxes(g_mk[0], g_mk[1], n_sites);
    gpu_wire_mailboxes(g_mk[1], g_mk[0], n_sites);
    for (int g = 0; g < 2; g++) {
        GpuCtx &c = g_mk[g];
        c.ln.init(g);
        c.R.add("token", c.ln.d_token(), 512);
        gpu_bind_llama(c, pm);
        if (const char *mp = getenv("MK_OWNEMBED")) {   // bind the full mirrored token_embd
            gguf::File gg; gg.open(mp);
            const gguf::TensorInfo *te = gg.find("token_embd.weight");
            void *dev = nullptr; CUDA_CHECK(cudaMalloc(&dev, te->nbytes));
            CUDA_CHECK(cudaMemcpy(dev, te->data, te->nbytes, cudaMemcpyHostToDevice));
            if (!c.R.table.count("token_embd.weight")) c.R.add("token_embd.weight", dev, te->nbytes);
            fprintf(stderr, "[MK ownembed] bound token_embd %.2f GB on gpu %d\n", te->nbytes / 1e9, g);
        }
        // Layout diagnostic (MK_WCHECK=<model>): compare llama's bound per-GPU
        // weight bytes against the harness weight_slice for the same tensor. Runs
        // before upload_program (no resident kernel -> D2H is safe).
        if (const char *mp = getenv("MK_WCHECK")) {
            gguf::File gg; gg.open(mp);
            for (auto &ti : gg.tensors) {
                // check every blk.0.* (DeltaNet layer 0) + blk.3.* (first attn) tensor
                if (ti.name.rfind("blk.0.", 0) != 0 && ti.name.rfind("blk.3.", 0) != 0) continue;
                if (!c.R.table.count(ti.name)) { fprintf(stderr, "[wcheck g%d] %s NOT BOUND\n", g, ti.name.c_str()); continue; }
                std::vector<uint8_t> stage; const uint8_t *src;
                size_t lb = weight_slice(ti, g, stage, src);
                std::vector<uint8_t> got(lb);
                CUDA_CHECK(cudaMemcpy(got.data(), c.R.table[ti.name].ptr, lb, cudaMemcpyDeviceToHost));
                size_t nd = 0, first = lb; for (size_t i = 0; i < lb; i++) if (src[i] != got[i]) { if (nd == 0) first = i; nd++; }
                if (nd) fprintf(stderr, "[MK wcheck g%d] %-28s %zu B, %zu DIFFER (%.1f%%) first@%zu\n",
                                g, ti.name.c_str(), lb, nd, 100.0 * nd / lb, first);
            }
            if (g == 1) fprintf(stderr, "[MK wcheck] blk.0 + blk.3 scan done\n");
        }
        gpu_alloc_buffers(c, program, /*skip_kv=*/true);
        PackedProgram p = pack_program(program, c.R, &c.mtab, c.gpu_index);
        if (!p.complete)
            throw std::runtime_error("mk gpu " + std::to_string(g) + " pack failed after " +
                std::to_string(p.packed_before_failure) + ": " + p.first_failure);
        c.xchg_idx = p.xchg_idx;
        c.ln.upload_program(p);
        fprintf(stderr, "[MK dual] gpu %d packed %zu instrs (%zu XCHG), kernel launched\n",
                g, p.instrs.size(), p.xchg_idx.size());
    }
    g_mk_up = true; g_mk_nvocab = n_vocab;
}

bool mk_dual_step(const void *seed0, const void *seed1, int64_t pos,
                  void *out0, void *out1) {
    // All copies ride c.pstream: the persistent kernel spins on kstream, so a
    // default-stream copy (or cudaDeviceSynchronize) would serialize against a
    // kernel that never exits and deadlock.
    const void *seeds[2] = { seed0, seed1 };
    void *outs[2] = { out0, out1 };
    if (getenv("MK_ZERO_STATE")) {   // diagnostic: zero the MK-read KV/state (pstream)
        for (int g = 0; g < 2; g++) {
            CUDA_CHECK(cudaSetDevice(g_mk[g].device));
            for (auto &e : g_mk[g].R.table) {
                const std::string &n = e.first; size_t b = 0;
                if      (n.rfind("conv_state_l", 0) == 0) b = (size_t)(CONV_STATE_N / 2) * 4;
                else if (n.rfind("ssm_state_l", 0) == 0)  b = (size_t)(SSM_STATE_N / 2) * 4;
                else if (n.rfind("cache_k_l", 0) == 0 || n.rfind("cache_v_l", 0) == 0)
                    b = (size_t) g_mk[g].n_ctx * (N_EMBD_GQA / 2) * 2;
                if (b) CUDA_CHECK(cudaMemsetAsync(e.second.ptr, 0, b, g_mk[g].pstream));
            }
            CUDA_CHECK(cudaStreamSynchronize(g_mk[g].pstream));
        }
    }
    const char *oe = getenv("MK_OWNEMBED");
    for (int g = 0; g < 2; g++) {
        CUDA_CHECK(cudaSetDevice(g_mk[g].device));
        // seed the mirrored residual trunk (5120 f32) from the CPU embed output
        // (skipped under MK_OWNEMBED: the kept embed op writes buf:residual itself)
        if (!oe)
            CUDA_CHECK(cudaMemcpyAsync(g_mk[g].bufs["residual"].ptr, seeds[g],
                                       (size_t) N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                       g_mk[g].pstream));
        CUDA_CHECK(cudaStreamSynchronize(g_mk[g].pstream));
        gpu_set_inputs(g_mk[g], pos);
    }
    if (getenv("MK_ARGMAX")) {   // one-time seed-content sanity (before the pass runs)
        static bool once = false;
        if (!once) { once = true;
            float r[8]; CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(r, g_mk[0].bufs["residual"].ptr, 32,
                                       cudaMemcpyDeviceToHost, g_mk[0].pstream));
            CUDA_CHECK(cudaStreamSynchronize(g_mk[0].pstream));
            fprintf(stderr, "[MK seed] buf:residual[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
        }
    }
    int32_t tok = oe ? (int32_t) atoi(getenv("MK_TOKEN") ? getenv("MK_TOKEN") : "0") : 0;
    if (!dual_run_pass(g_mk, &tok, 1, ++g_mk_seqno, 60000.0)) return false;
    size_t half = (size_t)(g_mk_nvocab / 2) * 4;
    for (int g = 0; g < 2; g++) {
        CUDA_CHECK(cudaSetDevice(g_mk[g].device));
        CUDA_CHECK(cudaMemcpyAsync(outs[g], g_mk[g].bufs["result_output"].ptr, half,
                                   cudaMemcpyDeviceToDevice, g_mk[g].pstream));
        CUDA_CHECK(cudaStreamSynchronize(g_mk[g].pstream));
    }
    if (getenv("MK_ARGMAX")) {   // global argmax over the two vocab halves
        static std::vector<float> h0, h1;
        h0.resize(g_mk_nvocab / 2); h1.resize(g_mk_nvocab / 2);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpyAsync(h0.data(), g_mk[0].bufs["result_output"].ptr, half, cudaMemcpyDeviceToHost, g_mk[0].pstream));
        CUDA_CHECK(cudaStreamSynchronize(g_mk[0].pstream));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpyAsync(h1.data(), g_mk[1].bufs["result_output"].ptr, half, cudaMemcpyDeviceToHost, g_mk[1].pstream));
        CUDA_CHECK(cudaStreamSynchronize(g_mk[1].pstream));
        int bg = 0; size_t bi = 0; float bv = -1e30f;
        for (size_t i = 0; i < h0.size(); i++) if (h0[i] > bv) { bv = h0[i]; bi = i; bg = 0; }
        for (size_t i = 0; i < h1.size(); i++) if (h1[i] > bv) { bv = h1[i]; bi = i; bg = 1; }
        fprintf(stderr, "[MK argmax] pos=%lld tok=%lld val=%.3f\n",
                (long long) pos, (long long) (bg * (g_mk_nvocab / 2) + bi), bv);
    }
    if (getenv("MK_DBGMID")) {   // per-layer residual L2 norm + GPU0-vs-GPU1 mirror diff
        static bool once = false;
        if (!once) { once = true;
            size_t n = (size_t)(N_LAYER + 1) * N_EMBD;
            std::vector<float> d0(n), d1(n);
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(d0.data(), g_mk[0].bufs["dbg_mid"].ptr, n * 4, cudaMemcpyDeviceToHost, g_mk[0].pstream));
            CUDA_CHECK(cudaStreamSynchronize(g_mk[0].pstream));
            CUDA_CHECK(cudaSetDevice(1));
            CUDA_CHECK(cudaMemcpyAsync(d1.data(), g_mk[1].bufs["dbg_mid"].ptr, n * 4, cudaMemcpyDeviceToHost, g_mk[1].pstream));
            CUDA_CHECK(cudaStreamSynchronize(g_mk[1].pstream));
            for (int il = 0; il <= N_LAYER; il++) {
                double s = 0, md = 0; int nnan = 0;
                for (int j = 0; j < N_EMBD; j++) { float v = d0[(size_t) il * N_EMBD + j];
                    if (v != v) nnan++; else s += (double) v * v;
                    double diff = fabs((double) v - d1[(size_t) il * N_EMBD + j]); if (diff > md) md = diff; }
                fprintf(stderr, "[dbgmid] L%02d |x0|=%.4f  max|x0-x1|=%.5f%s\n",
                        il, sqrt(s), md, nnan ? " HAS-NAN" : "");
            }
        }
    }
    return true;
}

bool mk_dual_ready() { return g_mk_up; }

// Full teardown so the next mk_dual_setup starts clean (mk_backend calls this
// before any forward, to free the SMs the resident kernel hogs). Frees MK-owned
// scratch + mailboxes + streams and destroys the Host; llama's weight/KV pointers
// live in the Resolver with bytes=0 and are NOT freed (they are llama's).
void mk_dual_shutdown() {
    if (!g_mk_up) return;
    for (int g = 0; g < 2; g++) {
        GpuCtx &c = g_mk[g];
        cudaSetDevice(c.device);
        if (c.ln.program_uploaded) mk::host_shutdown(c.ln.h, 5000.0);
        mk::host_destroy(c.ln.h);
        for (auto &e : c.bufs) cudaFree(e.second.ptr);   // MK scratch only
        if (c.mbox_payload) cudaFree(c.mbox_payload);
        if (c.mbox_seqno)   cudaFree(c.mbox_seqno);
        if (c.pstream) cudaStreamDestroy(c.pstream);
        c = GpuCtx{};   // reset R.table / bufs / mtab / xchg_idx / ln to fresh
    }
    g_mk_up = false;
    g_mk_seqno = 0;
}

#ifndef MK_NO_MAIN
int main(int argc, char **argv) {
    // Line-buffer stdout even when redirected to a file: a leased run killed
    // by a watchdog or outer timeout must not take its progress to the grave
    // (block-buffered stdio never flushes on SIGKILL; one silent 10-minute
    // window run was unattributable for exactly this reason).
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model = "/opt/models/Qwen3.6-27B-AR16asF16-probe.gguf";  // F16-ssm_out base (call/0020)
    std::string program_path;
    std::string parity_ref, out_dir, diag_out;
    int64_t n_ctx = 8192, bench_n = -1, bench_pos0 = 0, prefill_u = 1;
    int gpu = 0;
    bool allow_inv_mismatch = false;   // per-type inventory totals are hardcoded for
                                       // the AR16 base; the F16-ssm_out base has a
                                       // legit different type mix. Checksums stay the
                                       // hard gate; this only downgrades inventory.
    enum { M_VALIDATE, M_PARITY, M_BENCH, M_PARITY_TENSOR, M_BENCH_TENSOR,
           M_PREFILL_PARITY, M_PREFILL_BENCH } mode = M_VALIDATE;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs an argument (%s)\n", a.c_str(), what); exit(2); }
            return argv[++i];
        };
        if (a == "--validate")     mode = M_VALIDATE;
        else if (a == "--parity")  { mode = M_PARITY; parity_ref = need("oracle ref dir"); }
        else if (a == "--bench")   { mode = M_BENCH; bench_n = strtoll(need("token count"), nullptr, 10); }
        else if (a == "--parity-tensor") { mode = M_PARITY_TENSOR; parity_ref = need("oracle ref dir"); }
        else if (a == "--bench-tensor")  { mode = M_BENCH_TENSOR; bench_n = strtoll(need("token count"), nullptr, 10); }
        else if (a == "--prefill-parity") { mode = M_PREFILL_PARITY;
                                            prefill_u = strtoll(need("tile width U"), nullptr, 10); }
        else if (a == "--prefill-bench")  { mode = M_PREFILL_BENCH;
                                            prefill_u = strtoll(need("tile width U"), nullptr, 10);
                                            bench_n = strtoll(need("token count"), nullptr, 10); }
        else if (a == "--model")   model = need("gguf path");
        else if (a == "--program") program_path = need("program.json path");
        else if (a == "--out")     out_dir = need("dump dir");
        else if (a == "--diag")    diag_out = need("diag out file");
        else if (a == "--n-ctx")   n_ctx = strtoll(need("context size"), nullptr, 10);
        else if (a == "--pos0")    bench_pos0 = strtoll(need("start decode position"), nullptr, 10);
        else if (a == "--gpu")     gpu = atoi(need("device index"));
        else if (a == "--allow-inventory-mismatch") allow_inv_mismatch = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const bool prefill_mode = (mode == M_PREFILL_PARITY || mode == M_PREFILL_BENCH);
    const bool tensor_mode = (mode == M_PARITY_TENSOR || mode == M_BENCH_TENSOR || prefill_mode);
    if (program_path.empty())
        program_path = (prefill_mode && prefill_u > 1) ? "k0/program-split-prefill.json"
                     : tensor_mode ? "k0/program-split.json" : "k0/program.json";

    try {
        // ---- dual-GPU tensor-parallel path -------------------------------
        if (tensor_mode) {
            int ndev = 0;
            CUDA_CHECK(cudaGetDeviceCount(&ndev));
            if (ndev < 2) { fprintf(stderr, "tensor mode needs 2 GPUs, have %d\n", ndev); return 2; }

            Residency res;
            bool inventory_ok = res.enumerate_and_validate(model);
            if (!inventory_ok && !allow_inv_mismatch) { fprintf(stderr, "loader validation failed; refusing tensor run\n"); return 1; }
            if (!inventory_ok) fprintf(stderr, "note: inventory MISMATCH accepted (--allow-inventory-mismatch); "
                                               "the F16-ssm_out base has a legit different type mix\n");

            std::string text;
            if (!read_file(program_path, text)) {
                struct stat st{};
                // auto-regen covers the decode split program only; a U>1
                // prefill program is generated explicitly (--prefill U).
                if (!(prefill_mode && prefill_u > 1) && stat("k0/compile_schedule.py", &st) == 0) {
                    printf("program: %s absent — running python3 k0/compile_schedule.py --split\n",
                           program_path.c_str());
                    if (system("python3 k0/compile_schedule.py --split") != 0)
                        printf("program: compile_schedule.py --split failed\n");
                }
            }
            if (!read_file(program_path, text))
                throw std::runtime_error("cannot read program " + program_path +
                                         " (run: python3 k0/compile_schedule.py --split, "
                                         "or --prefill U for a U>1 prefill program)");
            JsonParser jp(text);
            Jv program = jp.value();

            // per-GPU VRAM guard (weights half + full token_embd + KV/state/mailboxes).
            for (int g = 0; g < 2; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                size_t free_b = 0, total_b = 0;
                CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
                printf("vram: GPU %d free %.2f / %.2f GB\n", g, free_b / 1e9, total_b / 1e9);
                if (free_b < (10ull << 30)) {
                    fprintf(stderr, "aborting: GPU %d has < 10 GB free (need the device mostly free)\n", g);
                    return 2;
                }
            }

            if (!diag_out.empty()) {   // 1-pass pos-0 residual bisect (dual)
                GpuCtx g2[2];
                dual_setup(g2, res.gg, program, n_ctx, res.n_vocab);
                gpu_set_inputs(g2[0], 0);
                gpu_set_inputs(g2[1], 0);
                int32_t dtok = 11;
                if (!dual_run_pass(g2, &dtok, 1, 1, 60000.0)) return 3;
                std::vector<float> lout, lout1, lmid, lo0, lo1;
                gpu_read_f32(g2[0], "dbg_lout", lout, (size_t) N_LOUT * N_EMBD);
                gpu_read_f32(g2[1], "dbg_lout", lout1, (size_t) N_LOUT * N_EMBD);
                gpu_read_f32(g2[0], "dbg_mid", lmid, (size_t) (N_LAYER + 1) * N_EMBD);
                gpu_read_f32(g2[0], "logits", lo0, (size_t) res.n_vocab / 2);
                gpu_read_f32(g2[1], "logits", lo1, (size_t) res.n_vocab / 2);
                FILE *f = fopen(diag_out.c_str(), "wb");
                fwrite(lout.data(), 4, lout.size(), f);   // gpu0 residual (l_out)
                fwrite(lout1.data(), 4, lout1.size(), f); // gpu1 residual (mirror check)
                fwrite(lmid.data(), 4, lmid.size(), f);   // gpu0 mid-block residual
                fwrite(lo0.data(), 4, lo0.size(), f);
                fwrite(lo1.data(), 4, lo1.size(), f);
                fclose(f);
                printf("diag(dual): wrote %s (dbg_lout %d x %d + logits halves)\n",
                       diag_out.c_str(), N_LOUT, N_EMBD);
                dual_shutdown(g2);
                return 0;
            }

            int rc;
            if (mode == M_PARITY_TENSOR) {
                if (out_dir.empty())
                    out_dir = "/var/tmp/mk-harness/cand-tensor-" + basename_of(parity_ref);
                rc = parity_run_tensor(res.gg, program, n_ctx, res.n_vocab, parity_ref, out_dir);
            } else if (mode == M_PREFILL_PARITY) {
                rc = prefill_parity_run(res.gg, program, n_ctx, res.n_vocab, prefill_u);
            } else if (mode == M_PREFILL_BENCH) {
                rc = prefill_bench_run(res.gg, program, n_ctx, res.n_vocab, prefill_u, bench_n);
            } else {
                rc = bench_run_tensor(res.gg, program, n_ctx, res.n_vocab, bench_n, bench_pos0);
            }
            return rc;
        }

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
            R.add("token", ln.d_token(), 512);   // cell:token -> Host d_token
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
                printf("pack: %zu instrs packed (%zu FATTN_DECODE ops, n_kv via cell)\n",
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

        if ((!inventory_ok && !allow_inv_mismatch) || !checksums_ok) {
            fprintf(stderr, "refusing to run %s: loader validation failed\n",
                    mode == M_PARITY ? "parity" : "bench");
            return 1;
        }
        if (!inventory_ok && allow_inv_mismatch)
            fprintf(stderr, "note: inventory MISMATCH accepted (--allow-inventory-mismatch); "
                            "checksums OK is the binding gate\n");
        if (!have_program || !packed.complete) {
            fprintf(stderr, "cannot run %s: no packed program (see pack status above)\n",
                    mode == M_PARITY ? "parity" : "bench");
            return 3;
        }

        if (!diag_out.empty()) {   // 1-pass pos-0 residual bisect (single GPU)
            rt.set_pass_inputs(0);
            if (!ln.run_pass(11)) return 3;
            std::vector<float> lout, lmid, logits;
            rt.read_f32("dbg_lout", lout, (size_t) N_LOUT * N_EMBD);
            rt.read_f32("dbg_mid", lmid, (size_t) (N_LAYER + 1) * N_EMBD);
            rt.read_f32("logits", logits, (size_t) res.n_vocab);
            FILE *f = fopen(diag_out.c_str(), "wb");
            fwrite(lout.data(), 4, lout.size(), f);
            fwrite(lmid.data(), 4, lmid.size(), f);
            fwrite(logits.data(), 4, logits.size(), f);
            fclose(f);
            printf("diag(single): wrote %s (dbg_lout %d x %d + logits %lld)\n",
                   diag_out.c_str(), N_LOUT, N_EMBD, (long long) res.n_vocab);
            ln.shutdown();
            return 0;
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
#endif // MK_NO_MAIN
