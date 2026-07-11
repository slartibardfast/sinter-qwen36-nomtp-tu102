// Interpreter spine test: glue-op correctness against a host reference,
// the G15 boundary-crossing measurement, and the error-flag path.
//
// G15: a synthetic program of N_BOUND boundaries (with and without a trivial
// op between them), M passes; per-pass wall measured on-device (block-0
// clock64 deltas, ring in Control.pass_cycles) and from the host around
// run_pass. Reported as ns per boundary crossing against the ~0.4 us Y02
// estimate and the measured 825 ns grid.sync comparator
// (reference/tu102 sync_protocol).
//
// Error path: a program with an unwired kind (and one with an out-of-range
// kind) must set the error cell and EXIT the grid — never hang.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include "../core/host.h"
#include "../k0/ops/glue.cuh"   // the glue Args structs the program packs

static int g_fail = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                   \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
            g_fail = 1;                                                        \
        }                                                                      \
    } while (0)

static mk::Instr make(uint16_t kind, uint16_t lo, uint16_t hi,
                      const void *args = nullptr, size_t n = 0) {
    mk::Instr in{};
    in.kind = kind;
    in.block_lo = lo;
    in.block_hi = hi;
    if (args)
        std::memcpy(in.payload, args, n);
    return in;
}

static int pick_device() {
    int n = 0;
    cudaGetDeviceCount(&n);
    int best = 0;
    size_t best_free = 0;
    for (int d = 0; d < n; d++) {
        cudaSetDevice(d);
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        if (f > best_free) {
            best_free = f;
            best = d;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Part 1: glue-op program vs host reference.
// EMBED_LOOKUP -> RMSNORM -> RESIDUAL_ADD -> LOGITS_EMIT(flag), two passes
// with different tokens (exercises the per-pass token cell and epoch carry).
// ---------------------------------------------------------------------------
static void test_glue(int dev) {
    constexpr uint32_t NC = 5120, VOCAB = 4096;
    mk::Host h;
    if (!mk::host_init(h, dev, 16)) {
        CHECK(false, "host_init");
        return;
    }

    std::vector<__half> emb((size_t)VOCAB * NC);
    std::vector<float> w(NC), resid(NC);
    for (size_t i = 0; i < emb.size(); i++)
        emb[i] = __float2half(((int)((i * 7 + 13) % 997) - 498) / 997.0f);
    for (uint32_t i = 0; i < NC; i++) {
        w[i] = 0.5f + (float)(i % 61) / 61.0f;
        resid[i] = ((int)(i % 89) - 44) / 89.0f;
    }

    __half *d_emb;
    float *d_w, *d_resid, *d_x0, *d_x1, *d_x2, *d_out;
    cudaMalloc(&d_emb, emb.size() * 2);
    cudaMalloc(&d_w, NC * 4);
    cudaMalloc(&d_resid, NC * 4);
    cudaMalloc(&d_x0, NC * 4);
    cudaMalloc(&d_x1, NC * 4);
    cudaMalloc(&d_x2, NC * 4);
    cudaMalloc(&d_out, NC * 4);
    cudaMemcpy(d_emb, emb.data(), emb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), NC * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_resid, resid.data(), NC * 4, cudaMemcpyHostToDevice);

    unsigned *flag_h;
    cudaHostAlloc(&flag_h, 4, cudaHostAllocMapped);
    *flag_h = 0;
    unsigned *flag_d;
    cudaHostGetDevicePointer(&flag_d, flag_h, 0);

    mk::EmbedLookupArgs embed{d_emb, h.d_token, d_x0, NC, NC};
    mk::RmsnormArgs norm{d_x0, d_w, nullptr, d_x1, NC, 1};
    mk::ResidualAddArgs add{d_x1, d_resid, d_x2, NC};
    mk::LogitsEmitArgs emit{d_x2, d_out, NC, 0xC0FFEEu, flag_d};

    std::vector<mk::Instr> prog;
    prog.push_back(make(mk::OP_EMBED_LOOKUP, 0, 72, &embed, sizeof(embed)));
    prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    prog.push_back(make(mk::OP_RMSNORM, 0, 1, &norm, sizeof(norm)));
    prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    prog.push_back(make(mk::OP_RESIDUAL_ADD, 0, 72, &add, sizeof(add)));
    prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    prog.push_back(make(mk::OP_LOGITS_EMIT, 0, 1, &emit, sizeof(emit)));

    if (!mk::host_upload(h, prog.data(), (uint32_t)prog.size(), 3) ||
        !mk::host_launch(h)) {
        CHECK(false, "glue upload/launch");
        return;
    }

    const int32_t tokens[2] = {1234, 7};
    for (int t = 0; t < 2; t++) {
        const mk::RunStatus st = mk::host_run_pass(h, tokens[t], 5000.0);
        CHECK(st == mk::RUN_OK, "glue pass %d status %d", t, (int)st);
        if (st != mk::RUN_OK)
            break;
        CHECK(*flag_h == 0xC0FFEEu, "logits flag %#x", *flag_h);
        *flag_h = 0;

        std::vector<float> out(NC);
        cudaMemcpyAsync(out.data(), d_out, NC * 4, cudaMemcpyDeviceToHost,
                        h.cstream);
        cudaStreamSynchronize(h.cstream);

        // host reference
        std::vector<float> x0(NC);
        double sumsq = 0.0;
        for (uint32_t i = 0; i < NC; i++) {
            x0[i] = __half2float(emb[(size_t)tokens[t] * NC + i]);
            sumsq += (double)x0[i] * x0[i];
        }
        const float scale = (float)(1.0 / std::sqrt(sumsq / NC + 1e-6));
        float maxrel = 0.0f;
        for (uint32_t i = 0; i < NC; i++) {
            const float ref = scale * x0[i] * w[i] + resid[i];
            const float rel = std::fabs(out[i] - ref) /
                              std::fmax(1.0f, std::fabs(ref));
            maxrel = std::fmax(maxrel, rel);
        }
        CHECK(maxrel < 1e-3f, "glue pass %d maxrel %g", t, maxrel);
        std::printf("glue pass %d (token %d): maxrel %.3g %s\n", t, tokens[t],
                    maxrel, maxrel < 1e-3f ? "PASS" : "FAIL");
    }

    CHECK(mk::host_shutdown(h, 5000.0), "glue shutdown");
    mk::host_destroy(h);
    cudaFree(d_emb);
    cudaFree(d_w);
    cudaFree(d_resid);
    cudaFree(d_x0);
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_out);
    cudaFreeHost(flag_h);
}

// ---------------------------------------------------------------------------
// Part 2: G15. N_BOUND boundaries per pass, with_nop interleaves one OP_NOP
// (full range) before each boundary to price the fetch+dispatch crossing.
// ---------------------------------------------------------------------------
struct G15Result {
    double dev_ns_per_pass;
    double host_ns_per_pass;
    unsigned boundaries; // per pass, incl. the interpreter's epilogue one
};

static bool run_g15(int dev, bool with_nop, unsigned n_bound, unsigned warmup,
                    unsigned measure, double sm_ghz, G15Result &r) {
    mk::Host h;
    if (!mk::host_init(h, dev, warmup + measure))
        return false;

    std::vector<mk::Instr> prog;
    for (unsigned i = 0; i < n_bound; i++) {
        if (with_nop)
            prog.push_back(make(mk::OP_NOP, 0, 72));
        prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    }
    if (!mk::host_upload(h, prog.data(), (uint32_t)prog.size(), n_bound) ||
        !mk::host_launch(h))
        return false;

    for (unsigned i = 0; i < warmup; i++)
        if (mk::host_run_pass(h, 0, 5000.0) != mk::RUN_OK)
            return false;

    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < measure; i++)
        if (mk::host_run_pass(h, 0, 5000.0) != mk::RUN_OK)
            return false;
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<long long> cyc(warmup + measure);
    if (!mk::host_read_pass_cycles(h, cyc.data(), warmup + measure))
        return false;
    double sum = 0.0;
    for (unsigned i = warmup; i < warmup + measure; i++)
        sum += (double)cyc[i];

    r.dev_ns_per_pass = sum / measure / sm_ghz;
    r.host_ns_per_pass =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / measure;
    r.boundaries = n_bound + 1; // + the interpreter's epilogue boundary

    const bool ok = mk::host_shutdown(h, 5000.0);
    mk::host_destroy(h);
    return ok;
}

// ---------------------------------------------------------------------------
// Part 3: the error-flag path. A program with an unwired (or out-of-range)
// kind must set the error cell and exit the grid, not hang.
// ---------------------------------------------------------------------------
static void test_error(int dev, uint16_t kind, unsigned want_code,
                       const char *label) {
    mk::Host h;
    if (!mk::host_init(h, dev, 0)) {
        CHECK(false, "err host_init");
        return;
    }
    std::vector<mk::Instr> prog;
    prog.push_back(make(mk::OP_NOP, 0, 72));
    prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    prog.push_back(make(kind, 0, 72)); // instruction index 2
    prog.push_back(make(mk::OP_BOUNDARY, 0, 0));
    if (!mk::host_upload(h, prog.data(), (uint32_t)prog.size(), 2) ||
        !mk::host_launch(h)) {
        CHECK(false, "err upload/launch");
        return;
    }

    const mk::RunStatus st = mk::host_run_pass(h, 0, 5000.0);
    CHECK(st == mk::RUN_ERR, "%s: status %d (want RUN_ERR, never a hang)",
          label, (int)st);
    CHECK(h.h_err[0] == want_code, "%s: err code %u want %u", label,
          h.h_err[0], want_code);
    CHECK(h.h_err[1] == 2, "%s: err aux (instr idx) %u want 2", label,
          h.h_err[1]);

    // The grid must have exited on its own.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    cudaError_t q;
    while ((q = cudaStreamQuery(h.kstream)) == cudaErrorNotReady &&
           std::chrono::steady_clock::now() < deadline) {
    }
    CHECK(q == cudaSuccess, "%s: kernel did not exit (%s)", label,
          cudaGetErrorString(q));
    if (q == cudaSuccess)
        h.launched = false;
    std::printf("error path (%s): status %d code %u aux %u exit %s\n", label,
                (int)st, h.h_err[0], h.h_err[1],
                q == cudaSuccess ? "clean" : "WEDGED");
    mk::host_destroy(h);
}

int main(int argc, char **argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : pick_device();
    cudaSetDevice(dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    int khz = 0;
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev);
    const double sm_ghz = khz / 1e6;
    std::printf("device %d: %s, %d SMs, SM clock (attr) %.0f MHz\n", dev,
                p.name, p.multiProcessorCount, sm_ghz * 1e3);

    test_glue(dev);

    constexpr unsigned N_BOUND = 500, WARMUP = 32, MEASURE = 256;
    G15Result bare{}, nop{};
    if (!run_g15(dev, false, N_BOUND, WARMUP, MEASURE, sm_ghz, bare) ||
        !run_g15(dev, true, N_BOUND, WARMUP, MEASURE, sm_ghz, nop)) {
        CHECK(false, "G15 run");
    } else {
        const double bare_b = bare.dev_ns_per_pass / bare.boundaries;
        const double nop_b = nop.dev_ns_per_pass / nop.boundaries;
        const double dispatch = (nop.dev_ns_per_pass - bare.dev_ns_per_pass) /
                                N_BOUND;
        // G15 budget: interpreter overhead <= 2% of a 13.7 ms pass for ~500
        // boundary crossings.
        const double pct = nop.dev_ns_per_pass / 13.7e6 * 100.0;
        std::printf(
            "G15 (%u boundaries+epilogue, %u passes, clock %.0f MHz):\n"
            "  boundaries only : dev %.1f us/pass host %.1f us/pass -> "
            "%.0f ns/boundary\n"
            "  nop + boundary  : dev %.1f us/pass host %.1f us/pass -> "
            "%.0f ns/boundary (fetch+dispatch adds %.0f ns/instr)\n"
            "  vs Y02 estimate 400 ns, grid.sync comparator 825 ns\n"
            "  pass overhead %.3f%% of the 13.7 ms budget -> %s (gate: 2%%)\n",
            N_BOUND, MEASURE, sm_ghz * 1e3, bare.dev_ns_per_pass / 1e3,
            bare.host_ns_per_pass / 1e3, bare_b, nop.dev_ns_per_pass / 1e3,
            nop.host_ns_per_pass / 1e3, nop_b, dispatch, pct,
            pct <= 2.0 ? "PASS" : "FAIL");
        CHECK(pct <= 2.0, "G15 overhead %.3f%% > 2%%", pct);
    }

    // OP_XCHG_PUSH: the Y06 family is deferred to the dual-GPU milestone,
    // so it stays unwired while the other op families land in parallel.
    test_error(dev, mk::OP_XCHG_PUSH, mk::ERR_UNWIRED_KIND, "unwired kind");
    test_error(dev, (uint16_t)(mk::OP_KIND_COUNT + 5), mk::ERR_BAD_KIND,
               "bad kind");

    std::printf("test_interp: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
