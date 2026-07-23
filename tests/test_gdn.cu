// Unit + chain tests for the gated-DeltaNet op family (k0/ops/gdn.cuh).
//
// A mini-interpreter kernel executes Instr programs the ABI way (range
// checks, Y02 boundaries between dependent antichains) so the ops are
// exercised through the exact contract the real interpreter uses.
//
// CPU reference: float (not double) arithmetic in EXACTLY the declared G13
// fold order, with the 32-lane __shfl_xor butterfly simulated per lane.
// The device ops pin every mul-feeding-add with __fmul_rn/__fadd_rn (never
// FMA-contracted); the host helpers below round through volatile floats so
// the host compiler cannot contract either. x86-64 SSE2 f32 mul/add/sub is
// IEEE-exact like the device _rn ops, so these paths compare BITWISE:
//   - OP_STATE_LOAD / OP_STATE_STORE / OP_CONV_SHIFT_CONCAT: pure copies.
//   - OP_GDN_STEP: state AND output. The fold is transcendental-free (the
//     expf lives in OP_GDN_GATES), all partial sums replicate the per-lane
//     sequential order, and the tree reduce is simulated exactly (the
//     butterfly leaves every lane with identical bits, fp add being
//     bitwise-commutative), so even the "warp-reduced" output is bit-exact.
// Tolerance-checked (1e-5 rel), because device transcendentals (expf, logf,
// rsqrtf) are not bit-reproducible on the host:
//   - OP_SSM_CONV_SILU (silu), OP_QK_L2NORM (rsqrtf), OP_GDN_GATES
//     (softplus/exp/sigmoid), OP_GATED_RMSNORM (rsqrtf + silu).
// Chain test: the ssm-state recursion is closed on the GPU (store -> load
// through the ring) and independently on the CPU; per token the CPU fold
// consumes the GPU's own q/k/v/g/beta bits (downloaded), so the state and
// output comparison stays BIT-EXACT across tokens -- exactly the G13 claim.
// The conv-state recursion is copies only and is chained fully on the CPU,
// also compared bitwise per token.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "../core/isa.cuh"
#include "../core/sync.cuh"
#include "../k0/ops/gdn.cuh"

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                            \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// mini-interpreter (the ABI shape: range check per instruction, one Y02
// crossing per OP_BOUNDARY; 384 threads per block like the real kernel)
static constexpr int NBLK = 8;
static constexpr int NTHR = 384;

__global__ void mk_run(const mk::Instr *prog, int n_instr, unsigned *counter) {
    __shared__ alignas(16) char smem[64];  // this family uses no smem
    mk::GridBoundary bar{counter};
    unsigned epoch = 0;
    for (int i = 0; i < n_instr; i++) {
        const mk::Instr &ins = prog[i];
        if (ins.kind == mk::OP_BOUNDARY) {
            bar.cross(epoch);
            continue;
        }
        if (blockIdx.x < ins.block_lo || blockIdx.x >= ins.block_hi) {
            continue;
        }
        switch (ins.kind) {
            case mk::OP_STATE_LOAD:        mk::op_state_load(ins, smem); break;
            case mk::OP_CONV_SHIFT_CONCAT: mk::op_conv_shift_concat(ins, smem); break;
            case mk::OP_SSM_CONV_SILU:     mk::op_ssm_conv_silu(ins, smem); break;
            case mk::OP_QK_L2NORM:         mk::op_qk_l2norm(ins, smem); break;
            case mk::OP_GDN_GATES:         mk::op_gdn_gates(ins, smem); break;
            case mk::OP_GDN_STEP:          mk::op_gdn_step(ins, smem); break;
            case mk::OP_GATED_RMSNORM:     mk::op_gated_rmsnorm(ins, smem); break;
            case mk::OP_STATE_STORE:       mk::op_state_store(ins, smem); break;
            default: break;
        }
    }
}

template <typename A>
static mk::Instr make_instr(uint16_t kind, uint16_t lo, uint16_t hi, const A &args) {
    static_assert(sizeof(A) <= sizeof(mk::Instr::payload), "args exceed payload");
    mk::Instr ins;
    memset(&ins, 0, sizeof(ins));
    ins.kind = kind;
    ins.block_lo = lo;
    ins.block_hi = hi;
    memcpy(ins.payload, &args, sizeof(A));
    return ins;
}

static mk::Instr make_boundary() {
    mk::Instr ins;
    memset(&ins, 0, sizeof(ins));
    ins.kind = mk::OP_BOUNDARY;
    return ins;
}

static unsigned *g_counter = nullptr;

static void run_program(const std::vector<mk::Instr> &prog) {
    static mk::Instr *d_prog = nullptr;
    static size_t d_cap = 0;
    if (prog.size() > d_cap) {
        if (d_prog) CUDA_CHECK(cudaFree(d_prog));
        d_cap = prog.size();
        CUDA_CHECK(cudaMalloc(&d_prog, d_cap * sizeof(mk::Instr)));
    }
    CUDA_CHECK(cudaMemcpy(d_prog, prog.data(), prog.size() * sizeof(mk::Instr),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g_counter, 0, sizeof(unsigned)));  // fresh epoch base
    mk_run<<<NBLK, NTHR>>>(d_prog, (int) prog.size(), g_counter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// CPU reference helpers: float ops rounded through volatile so the host
// compiler cannot FMA-contract, mirroring the device __f{mul,add,sub}_rn.
static inline float fmul_ref(float a, float b) { volatile float r = a * b; return r; }
static inline float fadd_ref(float a, float b) { volatile float r = a + b; return r; }
static inline float fsub_ref(float a, float b) { volatile float r = a - b; return r; }

// Exact simulation of warp_reduce_sum's __shfl_xor butterfly: every lane
// converges to the same bits (fp add is bitwise-commutative), return lane 0.
static float tree_sum32_ref(const float *lane_val) {
    float v[32], nv[32];
    memcpy(v, lane_val, sizeof(v));
    for (int off = 16; off > 0; off >>= 1) {
        for (int l = 0; l < 32; l++) {
            nv[l] = fadd_ref(v[l], v[l ^ off]);
        }
        memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

static inline float silu_ref(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid_ref(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplus_ref(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

static constexpr int S = mk::GDN_SV;    // 128
static constexpr int RPL = S / 32;      // rows per lane

// The declared G13 fold, in-place on state (H x S x S, transposed M[col][i]).
static void ref_gdn_step(const float *q, const float *k, const float *v,
                         const float *g, const float *beta,
                         float *state, float *attn, int HV, int HK, float scale) {
    for (int h = 0; h < HV; h++) {
        const float *qh = q + (size_t) (h % HK) * S;
        const float *kh = k + (size_t) (h % HK) * S;
        const float gv = g[h];
        const float bv = beta[h];
        for (int col = 0; col < S; col++) {
            float *scol = state + ((size_t) h * S + col) * S;
            float lanes[32];
            for (int lane = 0; lane < 32; lane++) {
                float acc = 0.0f;
                for (int r = 0; r < RPL; r++) {
                    const int i = r * 32 + lane;
                    acc = fadd_ref(acc, fmul_ref(scol[i], kh[i]));
                }
                lanes[lane] = acc;
            }
            const float kv = tree_sum32_ref(lanes);
            const float delta = fmul_ref(fsub_ref(v[(size_t) h * S + col], fmul_ref(gv, kv)), bv);
            for (int lane = 0; lane < 32; lane++) {
                float acc = 0.0f;
                for (int r = 0; r < RPL; r++) {
                    const int i = r * 32 + lane;
                    scol[i] = fadd_ref(fmul_ref(gv, scol[i]), fmul_ref(kh[i], delta));
                    acc     = fadd_ref(acc, fmul_ref(scol[i], qh[i]));
                }
                lanes[lane] = acc;
            }
            attn[(size_t) h * S + col] = fmul_ref(tree_sum32_ref(lanes), scale);
        }
    }
}

// Pre-silu conv tap sums in the device order (bit-exact part of the op).
static void ref_conv_sums(const float *win, const float *w, float *sums, int channels) {
    for (int c = 0; c < channels; c++) {
        float s = 0.0f;
        for (int j = 0; j < mk::GDN_DCONV; j++) {
            s = fadd_ref(s, fmul_ref(win[c * mk::GDN_DCONV + j], w[c * mk::GDN_DCONV + j]));
        }
        sums[c] = s;
    }
}

// L2 norm with the exact lane-sum structure; only rsqrtf differs on device.
static void ref_l2norm(const float *x, float *y, int n_heads, float eps) {
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + (size_t) h * S;
        float lanes[32];
        for (int lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (int r = 0; r < RPL; r++) {
                acc = fadd_ref(acc, fmul_ref(xh[r * 32 + lane], xh[r * 32 + lane]));
            }
            lanes[lane] = acc;
        }
        const float ss = tree_sum32_ref(lanes);
        const float scale = 1.0f / sqrtf(fmaxf(ss, eps * eps));
        for (int c = 0; c < S; c++) y[(size_t) h * S + c] = scale * xh[c];
    }
}

static void ref_gated_rmsnorm(const float *x, const float *w, const float *z,
                              float *y, int n_heads, float eps) {
    for (int h = 0; h < n_heads; h++) {
        const float *xh = x + (size_t) h * S;
        const float *zh = z + (size_t) h * S;
        float lanes[32];
        for (int lane = 0; lane < 32; lane++) {
            float acc = 0.0f;
            for (int r = 0; r < RPL; r++) {
                acc = fadd_ref(acc, fmul_ref(xh[r * 32 + lane], xh[r * 32 + lane]));
            }
            lanes[lane] = acc;
        }
        const float ss = tree_sum32_ref(lanes);
        const float scale = 1.0f / sqrtf(ss / (float) S + eps);
        for (int c = 0; c < S; c++) {
            y[(size_t) h * S + c] = scale * xh[c] * w[c] * silu_ref(zh[c]);
        }
    }
}

// ---------------------------------------------------------------------------
// comparisons
static int g_failures = 0;

static void check_bitwise(const char *what, const float *got, const float *want, size_t n) {
    size_t bad = 0, first = (size_t) -1;
    for (size_t i = 0; i < n; i++) {
        uint32_t a, b;
        memcpy(&a, &got[i], 4);
        memcpy(&b, &want[i], 4);
        if (a != b) {
            if (bad == 0) first = i;
            bad++;
        }
    }
    if (bad) {
        g_failures++;
        printf("FAIL [bitwise] %-28s %zu/%zu mismatched (first at %zu: got %.9g want %.9g)\n",
               what, bad, n, first, got[first], want[first]);
    } else {
        printf("ok   [bitwise] %-28s %zu values\n", what, n);
    }
}

static void check_close(const char *what, const float *got, const float *want, size_t n,
                        float rtol) {
    size_t bad = 0, first = (size_t) -1;
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float denom = fmaxf(fabsf(want[i]), 1e-6f);
        const float rel = fabsf(got[i] - want[i]) / denom;
        if (rel > rtol) {
            if (bad == 0) first = i;
            bad++;
        }
        if (rel > worst) worst = rel;
    }
    if (bad) {
        g_failures++;
        printf("FAIL [<=%.0e ] %-28s %zu/%zu beyond tol (first at %zu: got %.9g want %.9g), worst rel %.3g\n",
               (double) rtol, what, bad, n, first, got[first], want[first], (double) worst);
    } else {
        printf("ok   [<=%.0e ] %-28s %zu values, worst rel %.3g\n",
               (double) rtol, what, n, (double) worst);
    }
}

// ---------------------------------------------------------------------------
// device buffer helpers
static float *dalloc(size_t n) {
    float *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}
static int64_t *dalloc_i64(size_t n) {
    int64_t *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(int64_t)));
    return p;
}
static void up(float *d, const float *h, size_t n) {
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
}
static void up_i64(int64_t *d, const int64_t *h, size_t n) {
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(int64_t), cudaMemcpyHostToDevice));
}
static void down(float *h, const float *d, size_t n) {
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static std::mt19937 g_rng(0x67646e21);
static void fill_rand(std::vector<float> &v, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto &x : v) x = d(g_rng);
}

// ---------------------------------------------------------------------------
// unit tests
static void test_state_load_store() {
    const int64_t stride = 786432;
    const int n_cache = 5;
    std::vector<float> cache((size_t) n_cache * stride);
    fill_rand(cache, -1.0f, 1.0f);
    float *d_cache = dalloc(cache.size());
    up(d_cache, cache.data(), cache.size());

    // load rows {3, 1} through a sub-range of blocks
    const int64_t rows_ld[2] = {3, 1};
    int64_t *d_rows = dalloc_i64(2);
    up_i64(d_rows, rows_ld, 2);
    float *d_work = dalloc(2 * stride);

    mk::StateLoadArgs la = {d_cache, d_rows, d_work, stride, (int32_t) stride, 2};
    run_program({make_instr(mk::OP_STATE_LOAD, 1, 7, la)});

    std::vector<float> got(2 * stride), want(2 * stride);
    down(got.data(), d_work, got.size());
    memcpy(want.data(), cache.data() + 3 * stride, stride * 4);
    memcpy(want.data() + stride, cache.data() + 1 * stride, stride * 4);
    check_bitwise("state_load (786432x2)", got.data(), want.data(), got.size());

    // store fresh data to rows {0, 4}
    std::vector<float> fresh(2 * stride);
    fill_rand(fresh, -1.0f, 1.0f);
    up(d_work, fresh.data(), fresh.size());
    const int64_t rows_st[2] = {0, 4};
    up_i64(d_rows, rows_st, 2);
    mk::StateStoreArgs sa = {d_work, d_cache, d_rows, stride, (int32_t) stride, 2};
    run_program({make_instr(mk::OP_STATE_STORE, 0, 8, sa)});

    std::vector<float> cache_got(cache.size());
    down(cache_got.data(), d_cache, cache.size());
    memcpy(cache.data() + 0 * stride, fresh.data(), stride * 4);
    memcpy(cache.data() + 4 * stride, fresh.data() + stride, stride * 4);
    check_bitwise("state_store (786432x2)", cache_got.data(), cache.data(), cache.size());

    // odd row length exercises the scalar (non-float4) path
    const int64_t stride2 = 1021;
    std::vector<float> cache2(3 * stride2);
    fill_rand(cache2, -1.0f, 1.0f);
    float *d_cache2 = dalloc(cache2.size());
    up(d_cache2, cache2.data(), cache2.size());
    const int64_t row2 = 2;
    up_i64(d_rows, &row2, 1);
    mk::StateLoadArgs la2 = {d_cache2, d_rows, d_work, stride2, (int32_t) stride2, 1};
    run_program({make_instr(mk::OP_STATE_LOAD, 0, 8, la2)});
    std::vector<float> got2(stride2);
    down(got2.data(), d_work, stride2);
    check_bitwise("state_load (odd length)", got2.data(), cache2.data() + 2 * stride2, stride2);

    CUDA_CHECK(cudaFree(d_cache));
    CUDA_CHECK(cudaFree(d_cache2));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rows));
}

static void test_conv_ops() {
    const int C = 10240;
    const int HIST = mk::GDN_DCONV - 1;
    std::vector<float> hist((size_t) C * HIST), xnew(C), weight((size_t) C * mk::GDN_DCONV);
    fill_rand(hist, -1.0f, 1.0f);
    fill_rand(xnew, -1.0f, 1.0f);
    fill_rand(weight, -1.0f, 1.0f);

    float *d_hist = dalloc(hist.size());
    float *d_xnew = dalloc(xnew.size());
    float *d_weight = dalloc(weight.size());
    float *d_win = dalloc((size_t) C * mk::GDN_DCONV);
    float *d_state = dalloc((size_t) 3 * C * HIST);  // 3-row conv cache
    float *d_out = dalloc(C);
    int64_t *d_row = dalloc_i64(1);
    up(d_hist, hist.data(), hist.size());
    up(d_xnew, xnew.data(), xnew.size());
    up(d_weight, weight.data(), weight.size());
    CUDA_CHECK(cudaMemset(d_state, 0, 3 * (size_t) C * HIST * 4));
    const int64_t row = 1;
    up_i64(d_row, &row, 1);

    mk::ConvShiftConcatArgs ca = {d_hist, d_xnew, d_win, d_state, d_row,
                                  (int64_t) C * HIST, C};
    mk::SsmConvSiluArgs va = {d_win, d_weight, d_out, C};
    run_program({make_instr(mk::OP_CONV_SHIFT_CONCAT, 0, 8, ca), make_boundary(),
                 make_instr(mk::OP_SSM_CONV_SILU, 2, 8, va)});

    // window + committed history are pure copies: bitwise
    std::vector<float> win_want((size_t) C * mk::GDN_DCONV), commit_want((size_t) C * HIST);
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < HIST; j++) win_want[c * mk::GDN_DCONV + j] = hist[c * HIST + j];
        win_want[c * mk::GDN_DCONV + HIST] = xnew[c];
        commit_want[c * HIST + 0] = hist[c * HIST + 1];
        commit_want[c * HIST + 1] = hist[c * HIST + 2];
        commit_want[c * HIST + 2] = xnew[c];
    }
    std::vector<float> win_got(win_want.size()), commit_got(commit_want.size());
    down(win_got.data(), d_win, win_got.size());
    down(commit_got.data(), d_state + (size_t) row * C * HIST, commit_got.size());
    check_bitwise("conv window", win_got.data(), win_want.data(), win_got.size());
    check_bitwise("conv history commit", commit_got.data(), commit_want.data(), commit_got.size());

    // conv+silu: tap sums replicate the device order; silu differs (host expf)
    std::vector<float> sums(C), out_want(C), out_got(C);
    ref_conv_sums(win_want.data(), weight.data(), sums.data(), C);
    for (int c = 0; c < C; c++) out_want[c] = silu_ref(sums[c]);
    down(out_got.data(), d_out, C);
    check_close("ssm_conv_silu", out_got.data(), out_want.data(), C, 1e-5f);

    // U-tile utile path at nt=2 (the branch the U-loop added; capacity ==
    // live width, so ntok_cell stays null and n_tokens drives the loop).
    // Window per channel: [h0 h1 h2 x0 x1], capacity stride HIST+2; token t
    // taps window[t..t+3]; commit = the last HIST entries = (h2, x0, x1).
    {
        const int NT = 2;
        std::vector<float> x2((size_t) NT * C);
        fill_rand(x2, -1.0f, 1.0f);
        float *d_x2 = dalloc(x2.size());
        float *d_win2 = dalloc((size_t) C * (HIST + NT));
        float *d_out2 = dalloc((size_t) NT * C);
        up(d_x2, x2.data(), x2.size());
        CUDA_CHECK(cudaMemset(d_state, 0, 3 * (size_t) C * HIST * 4));
        mk::ConvShiftConcatArgs ca2 = {d_hist, d_x2, d_win2, d_state, d_row,
                                       (int64_t) C * HIST, C};
        ca2.n_tokens = NT; ca2.xnew_tstride = C;
        mk::SsmConvSiluArgs va2 = {d_win2, d_weight, d_out2, C};
        va2.n_tokens = NT; va2.dst_tstride = C;
        run_program({make_instr(mk::OP_CONV_SHIFT_CONCAT, 0, 8, ca2), make_boundary(),
                     make_instr(mk::OP_SSM_CONV_SILU, 2, 8, va2)});

        const int W2 = HIST + NT;
        std::vector<float> win2_want((size_t) C * W2), commit2_want((size_t) C * HIST);
        for (int c = 0; c < C; c++) {
            for (int j = 0; j < HIST; j++) win2_want[(size_t) c * W2 + j] = hist[c * HIST + j];
            for (int t = 0; t < NT; t++) win2_want[(size_t) c * W2 + HIST + t] = x2[(size_t) t * C + c];
            commit2_want[c * HIST + 0] = hist[c * HIST + 2];
            commit2_want[c * HIST + 1] = x2[c];
            commit2_want[c * HIST + 2] = x2[(size_t) C + c];
        }
        std::vector<float> win2_got(win2_want.size()), commit2_got(commit2_want.size());
        down(win2_got.data(), d_win2, win2_got.size());
        down(commit2_got.data(), d_state + (size_t) row * C * HIST, commit2_got.size());
        check_bitwise("conv window nt=2", win2_got.data(), win2_want.data(), win2_got.size());
        check_bitwise("conv history commit nt=2", commit2_got.data(), commit2_want.data(),
                      commit2_got.size());

        // per-token sliding conv: token t's taps are window[t..t+3] per channel.
        std::vector<float> tapwin((size_t) C * mk::GDN_DCONV), sums2(C), out2_want((size_t) NT * C),
            out2_got((size_t) NT * C);
        for (int t = 0; t < NT; t++) {
            for (int c = 0; c < C; c++)
                for (int j = 0; j < mk::GDN_DCONV; j++)
                    tapwin[(size_t) c * mk::GDN_DCONV + j] = win2_want[(size_t) c * W2 + t + j];
            ref_conv_sums(tapwin.data(), weight.data(), sums2.data(), C);
            for (int c = 0; c < C; c++) out2_want[(size_t) t * C + c] = silu_ref(sums2[c]);
        }
        down(out2_got.data(), d_out2, out2_got.size());
        check_close("ssm_conv_silu nt=2", out2_got.data(), out2_want.data(), out2_got.size(), 1e-5f);
        CUDA_CHECK(cudaFree(d_x2)); CUDA_CHECK(cudaFree(d_win2)); CUDA_CHECK(cudaFree(d_out2));
    }

    CUDA_CHECK(cudaFree(d_hist)); CUDA_CHECK(cudaFree(d_xnew));
    CUDA_CHECK(cudaFree(d_weight)); CUDA_CHECK(cudaFree(d_win));
    CUDA_CHECK(cudaFree(d_state)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_row));
}

static void test_qk_l2norm() {
    const int H = 16;
    std::vector<float> x((size_t) H * S);
    fill_rand(x, -1.0f, 1.0f);
    // head 0 near-zero: exercises the eps^2 floor
    for (int c = 0; c < S; c++) x[c] *= 1e-25f;

    float *d_x = dalloc(x.size());
    float *d_y = dalloc(x.size());
    up(d_x, x.data(), x.size());
    mk::QkL2NormArgs a = {d_x, d_y, H, 1e-6f};
    run_program({make_instr(mk::OP_QK_L2NORM, 2, 5, a)});

    std::vector<float> want(x.size()), got(x.size());
    ref_l2norm(x.data(), want.data(), H, 1e-6f);
    down(got.data(), d_y, got.size());
    check_close("qk_l2norm", got.data(), want.data(), got.size(), 1e-5f);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

static void test_gdn_gates() {
    const int H = 48;
    std::vector<float> alpha(H), braw(H), bias(H), aa(H);
    fill_rand(alpha, -3.0f, 3.0f);
    fill_rand(braw, -3.0f, 3.0f);
    fill_rand(bias, -1.0f, 1.0f);
    fill_rand(aa, -4.0f, -0.01f);
    alpha[0] = 25.0f;  // exercises the softplus linear branch (x > 20)

    float *d_alpha = dalloc(H), *d_braw = dalloc(H), *d_bias = dalloc(H), *d_aa = dalloc(H);
    float *d_g = dalloc(H), *d_beta = dalloc(H);
    up(d_alpha, alpha.data(), H); up(d_braw, braw.data(), H);
    up(d_bias, bias.data(), H);   up(d_aa, aa.data(), H);
    mk::GdnGatesArgs a = {d_alpha, d_braw, d_bias, d_aa, d_g, d_beta, H};
    run_program({make_instr(mk::OP_GDN_GATES, 0, 8, a)});

    std::vector<float> g_want(H), b_want(H), g_got(H), b_got(H);
    for (int h = 0; h < H; h++) {
        g_want[h] = expf(softplus_ref(alpha[h] + bias[h]) * aa[h]);
        b_want[h] = sigmoid_ref(braw[h]);
    }
    down(g_got.data(), d_g, H);
    down(b_got.data(), d_beta, H);
    check_close("gdn_gates g", g_got.data(), g_want.data(), H, 1e-5f);
    check_close("gdn_gates beta", b_got.data(), b_want.data(), H, 1e-5f);
    CUDA_CHECK(cudaFree(d_alpha)); CUDA_CHECK(cudaFree(d_braw));
    CUDA_CHECK(cudaFree(d_bias)); CUDA_CHECK(cudaFree(d_aa));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_beta));
}

static void test_gdn_step(int HV, int HK, uint16_t lo, uint16_t hi) {
    std::vector<float> q((size_t) HK * S), k((size_t) HK * S), v((size_t) HV * S);
    std::vector<float> g(HV), beta(HV), state((size_t) HV * S * S);
    fill_rand(q, -1.0f, 1.0f);
    fill_rand(k, -1.0f, 1.0f);
    fill_rand(v, -1.0f, 1.0f);
    fill_rand(g, 0.01f, 0.99f);
    fill_rand(beta, 0.01f, 0.99f);
    fill_rand(state, -1.0f, 1.0f);

    float *d_q = dalloc(q.size()), *d_k = dalloc(k.size()), *d_v = dalloc(v.size());
    float *d_g = dalloc(HV), *d_beta = dalloc(HV);
    float *d_sin = dalloc(state.size()), *d_sout = dalloc(state.size());
    float *d_attn = dalloc(v.size());
    up(d_q, q.data(), q.size()); up(d_k, k.data(), k.size()); up(d_v, v.data(), v.size());
    up(d_g, g.data(), HV); up(d_beta, beta.data(), HV);
    up(d_sin, state.data(), state.size());

    const float scale = 1.0f / sqrtf((float) S);
    mk::GdnStepArgs a = {d_q, d_k, d_v, d_g, d_beta, d_sin, d_sout, d_attn, HV, HK, scale};
    run_program({make_instr(mk::OP_GDN_STEP, lo, hi, a)});

    std::vector<float> s_ref(state), attn_ref((size_t) HV * S);
    ref_gdn_step(q.data(), k.data(), v.data(), g.data(), beta.data(),
                 s_ref.data(), attn_ref.data(), HV, HK, scale);
    std::vector<float> s_got(state.size()), attn_got(attn_ref.size());
    down(s_got.data(), d_sout, s_got.size());
    down(attn_got.data(), d_attn, attn_got.size());
    char name[64];
    snprintf(name, sizeof(name), "gdn_step state (H=%d)", HV);
    check_bitwise(name, s_got.data(), s_ref.data(), s_got.size());
    snprintf(name, sizeof(name), "gdn_step output (H=%d)", HV);
    check_bitwise(name, attn_got.data(), attn_ref.data(), attn_got.size());

    // U-tile utile path at nt=2: the register-carried token loop must equal
    // TWO sequential reference steps (state carried through both). Dense
    // per-token slots: qkv stride = its own width, g/beta stride = HV.
    {
        const int NT = 2;
        std::vector<float> q2((size_t) NT * HK * S), k2((size_t) NT * HK * S),
            v2((size_t) NT * HV * S), g2v((size_t) NT * HV), b2((size_t) NT * HV);
        fill_rand(q2, -1.0f, 1.0f); fill_rand(k2, -1.0f, 1.0f); fill_rand(v2, -1.0f, 1.0f);
        fill_rand(g2v, 0.01f, 0.99f); fill_rand(b2, 0.01f, 0.99f);
        float *d_q2 = dalloc(q2.size()), *d_k2 = dalloc(k2.size()), *d_v2 = dalloc(v2.size());
        float *d_g2 = dalloc(g2v.size()), *d_b2 = dalloc(b2.size());
        float *d_attn2 = dalloc(v2.size());
        up(d_q2, q2.data(), q2.size()); up(d_k2, k2.data(), k2.size());
        up(d_v2, v2.data(), v2.size());
        up(d_g2, g2v.data(), g2v.size()); up(d_b2, b2.data(), b2.size());
        up(d_sin, state.data(), state.size());   // reset the carried state

        mk::GdnStepArgs a2 = {d_q2, d_k2, d_v2, d_g2, d_b2, d_sin, d_sout, d_attn2,
                              HV, HK, scale};
        a2.n_tokens = NT;
        a2.qkv_tstride = (int32_t)(HK * S);      // q,k share the stride; v uses
        a2.out_tstride = (int32_t)(HV * S);      // its own HV*S slot via toff
        // NOTE: q,k,v advance by the SAME qkv_tstride in the op (one mixer
        // slot); give v its own dense layout by matching strides: pack v at
        // HK*S stride only if HV==HK. For HV != HK exercise via out stride.
        run_program({make_instr(mk::OP_GDN_STEP, lo, hi, a2)});

        std::vector<float> s2_ref(state), attn2_ref((size_t) NT * HV * S);
        for (int t = 0; t < NT; t++)
            ref_gdn_step(q2.data() + (size_t) t * HK * S, k2.data() + (size_t) t * HK * S,
                         v2.data() + (size_t) t * a2.qkv_tstride, g2v.data() + (size_t) t * HV,
                         b2.data() + (size_t) t * HV, s2_ref.data(),
                         attn2_ref.data() + (size_t) t * HV * S, HV, HK, scale);
        std::vector<float> s2_got(state.size()), attn2_got(attn2_ref.size());
        down(s2_got.data(), d_sout, s2_got.size());
        down(attn2_got.data(), d_attn2, attn2_got.size());
        snprintf(name, sizeof(name), "gdn_step nt=2 state (H=%d)", HV);
        check_bitwise(name, s2_got.data(), s2_ref.data(), s2_got.size());
        snprintf(name, sizeof(name), "gdn_step nt=2 output (H=%d)", HV);
        check_bitwise(name, attn2_got.data(), attn2_ref.data(), attn2_got.size());
        CUDA_CHECK(cudaFree(d_q2)); CUDA_CHECK(cudaFree(d_k2)); CUDA_CHECK(cudaFree(d_v2));
        CUDA_CHECK(cudaFree(d_g2)); CUDA_CHECK(cudaFree(d_b2)); CUDA_CHECK(cudaFree(d_attn2));
    }

    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_sout));
    CUDA_CHECK(cudaFree(d_attn));
}

static void test_gated_rmsnorm() {
    const int H = 48;
    std::vector<float> x((size_t) H * S), w(S), z((size_t) H * S);
    fill_rand(x, -2.0f, 2.0f);
    fill_rand(w, -1.0f, 1.0f);
    fill_rand(z, -3.0f, 3.0f);
    float *d_x = dalloc(x.size()), *d_w = dalloc(S), *d_z = dalloc(z.size());
    float *d_y = dalloc(x.size());
    up(d_x, x.data(), x.size()); up(d_w, w.data(), S); up(d_z, z.data(), z.size());
    mk::GatedRmsNormArgs a = {d_x, d_w, d_z, d_y, H, 1e-6f};
    run_program({make_instr(mk::OP_GATED_RMSNORM, 0, 8, a)});

    std::vector<float> want(x.size()), got(x.size());
    ref_gated_rmsnorm(x.data(), w.data(), z.data(), want.data(), H, 1e-6f);
    down(got.data(), d_y, got.size());
    check_close("gated_rmsnorm", got.data(), want.data(), got.size(), 1e-5f);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_z)); CUDA_CHECK(cudaFree(d_y));
}

// ---------------------------------------------------------------------------
// full DeltaNet block chain over T synthetic tokens. HV/HK give the full
// model (48/16) or the per-GPU tensor-split half (24/8). The ssm state rides
// the GPU ring (state_load <- previous state_store) and, independently, a
// CPU mirror; both are compared BITWISE each token (the headline claim).
static void test_chain(int HV, int HK, int T) {
    const int C = 2 * HK * S + HV * S;   // conv channels
    const int HIST = mk::GDN_DCONV - 1;
    const int64_t conv_stride = (int64_t) C * HIST;
    const int64_t ssm_stride = (int64_t) HV * S * S;
    const int N_RING = 4;
    const float eps = 1e-6f;
    const float scale = 1.0f / sqrtf((float) S);

    printf("-- chain: HV=%d HK=%d C=%d T=%d --\n", HV, HK, C, T);

    // caches, ring-initialized random; CPU mirrors identical
    std::vector<float> conv_cache((size_t) N_RING * conv_stride);
    std::vector<float> ssm_cache((size_t) N_RING * ssm_stride);
    fill_rand(conv_cache, -1.0f, 1.0f);
    fill_rand(ssm_cache, -0.5f, 0.5f);
    float *d_conv_cache = dalloc(conv_cache.size());
    float *d_ssm_cache = dalloc(ssm_cache.size());
    up(d_conv_cache, conv_cache.data(), conv_cache.size());
    up(d_ssm_cache, ssm_cache.data(), ssm_cache.size());

    // weights (fixed across tokens)
    std::vector<float> convw((size_t) C * mk::GDN_DCONV), dt_bias(HV), aw(HV), normw(S);
    fill_rand(convw, -1.0f, 1.0f);
    fill_rand(dt_bias, -1.0f, 1.0f);
    fill_rand(aw, -4.0f, -0.01f);
    fill_rand(normw, -1.0f, 1.0f);
    float *d_convw = dalloc(convw.size()), *d_dt = dalloc(HV), *d_aw = dalloc(HV),
          *d_normw = dalloc(S);
    up(d_convw, convw.data(), convw.size());
    up(d_dt, dt_bias.data(), HV); up(d_aw, aw.data(), HV); up(d_normw, normw.data(), S);

    // working buffers + per-token inputs
    float *d_hist = dalloc(conv_stride), *d_sbuf = dalloc(ssm_stride);
    float *d_win = dalloc((size_t) C * mk::GDN_DCONV), *d_convout = dalloc(C);
    float *d_qn = dalloc((size_t) HK * S), *d_kn = dalloc((size_t) HK * S);
    float *d_g = dalloc(HV), *d_b = dalloc(HV);
    float *d_attn = dalloc((size_t) HV * S), *d_y = dalloc((size_t) HV * S);
    float *d_qkv = dalloc(C), *d_alpha = dalloc(HV), *d_braw = dalloc(HV),
          *d_z = dalloc((size_t) HV * S);
    int64_t *d_row_ld = dalloc_i64(1), *d_row_st = dalloc_i64(1);

    std::vector<float> qkv(C), alpha(HV), braw(HV), z((size_t) HV * S);
    std::vector<float> convout(C), qn((size_t) HK * S), kn((size_t) HK * S), gv(HV), bv(HV);
    std::vector<float> attn_got((size_t) HV * S), s_got(ssm_stride), conv_got(conv_stride);
    std::vector<float> s_ref(ssm_stride), attn_ref((size_t) HV * S);
    std::vector<float> sums(C), tolbuf((size_t) HV * S);

    for (int t = 0; t < T; t++) {
        const int64_t row_ld = t % N_RING;
        const int64_t row_st = (t + 1) % N_RING;
        up_i64(d_row_ld, &row_ld, 1);
        up_i64(d_row_st, &row_st, 1);

        fill_rand(qkv, -1.0f, 1.0f);
        fill_rand(alpha, -3.0f, 3.0f);
        fill_rand(braw, -3.0f, 3.0f);
        fill_rand(z, -3.0f, 3.0f);
        up(d_qkv, qkv.data(), C);
        up(d_alpha, alpha.data(), HV);
        up(d_braw, braw.data(), HV);
        up(d_z, z.data(), z.size());

        mk::StateLoadArgs ld_conv = {d_conv_cache, d_row_ld, d_hist, conv_stride,
                                     (int32_t) conv_stride, 1};
        mk::StateLoadArgs ld_ssm = {d_ssm_cache, d_row_ld, d_sbuf, ssm_stride,
                                    (int32_t) ssm_stride, 1};
        mk::ConvShiftConcatArgs shift = {d_hist, d_qkv, d_win, d_conv_cache, d_row_st,
                                         conv_stride, C};
        mk::SsmConvSiluArgs conv = {d_win, d_convw, d_convout, C};
        mk::QkL2NormArgs l2q = {d_convout, d_qn, HK, eps};
        mk::QkL2NormArgs l2k = {d_convout + (size_t) HK * S, d_kn, HK, eps};
        mk::GdnGatesArgs gates = {d_alpha, d_braw, d_dt, d_aw, d_g, d_b, HV};
        mk::GdnStepArgs step = {d_qn, d_kn, d_convout + (size_t) 2 * HK * S, d_g, d_b,
                                d_sbuf, d_sbuf /* in place */, d_attn, HV, HK, scale};
        mk::GatedRmsNormArgs norm = {d_attn, d_normw, d_z, d_y, HV, eps};
        mk::StateStoreArgs st_ssm = {d_sbuf, d_ssm_cache, d_row_st, ssm_stride,
                                     (int32_t) ssm_stride, 1};

        run_program({
            make_instr(mk::OP_STATE_LOAD, 0, 2, ld_conv),   // disjoint antichain
            make_instr(mk::OP_STATE_LOAD, 2, 8, ld_ssm),
            make_boundary(),
            make_instr(mk::OP_CONV_SHIFT_CONCAT, 0, 8, shift),
            make_boundary(),
            make_instr(mk::OP_SSM_CONV_SILU, 0, 8, conv),
            make_boundary(),
            make_instr(mk::OP_QK_L2NORM, 0, 3, l2q),        // disjoint antichain
            make_instr(mk::OP_QK_L2NORM, 3, 6, l2k),
            make_instr(mk::OP_GDN_GATES, 6, 8, gates),
            make_boundary(),
            make_instr(mk::OP_GDN_STEP, 0, 8, step),
            make_boundary(),
            make_instr(mk::OP_GATED_RMSNORM, 0, 4, norm),   // disjoint antichain
            make_instr(mk::OP_STATE_STORE, 4, 8, st_ssm),
            make_boundary(),
        });

        char what[64];

        // conv chain: fully CPU-recursive (copies only), bitwise per token
        const float *hist_ref = conv_cache.data() + (size_t) row_ld * conv_stride;
        std::vector<float> win_ref((size_t) C * mk::GDN_DCONV), commit_ref(conv_stride);
        for (int c = 0; c < C; c++) {
            for (int j = 0; j < HIST; j++) {
                win_ref[c * mk::GDN_DCONV + j] = hist_ref[c * HIST + j];
            }
            win_ref[c * mk::GDN_DCONV + HIST] = qkv[c];
            commit_ref[c * HIST + 0] = hist_ref[c * HIST + 1];
            commit_ref[c * HIST + 1] = hist_ref[c * HIST + 2];
            commit_ref[c * HIST + 2] = qkv[c];
        }
        memcpy(conv_cache.data() + (size_t) row_st * conv_stride, commit_ref.data(),
               conv_stride * 4);
        down(conv_got.data(), d_conv_cache + (size_t) row_st * conv_stride, conv_stride);
        snprintf(what, sizeof(what), "t%d conv state row", t);
        check_bitwise(what, conv_got.data(), commit_ref.data(), conv_stride);

        // conv/l2norm/gates sanity vs pure-CPU math (transcendentals -> tol)
        down(convout.data(), d_convout, C);
        ref_conv_sums(win_ref.data(), convw.data(), sums.data(), C);
        for (int c = 0; c < C; c++) sums[c] = silu_ref(sums[c]);
        snprintf(what, sizeof(what), "t%d conv+silu", t);
        check_close(what, convout.data(), sums.data(), C, 1e-5f);

        down(qn.data(), d_qn, qn.size());
        down(kn.data(), d_kn, kn.size());
        ref_l2norm(convout.data(), tolbuf.data(), HK, eps);
        snprintf(what, sizeof(what), "t%d l2norm q", t);
        check_close(what, qn.data(), tolbuf.data(), qn.size(), 1e-5f);
        ref_l2norm(convout.data() + (size_t) HK * S, tolbuf.data(), HK, eps);
        snprintf(what, sizeof(what), "t%d l2norm k", t);
        check_close(what, kn.data(), tolbuf.data(), kn.size(), 1e-5f);

        down(gv.data(), d_g, HV);
        down(bv.data(), d_b, HV);
        for (int h = 0; h < HV; h++) {
            tolbuf[h] = expf(softplus_ref(alpha[h] + dt_bias[h]) * aw[h]);
            tolbuf[HV + h] = sigmoid_ref(braw[h]);
        }
        snprintf(what, sizeof(what), "t%d gates g", t);
        check_close(what, gv.data(), tolbuf.data(), HV, 1e-5f);
        snprintf(what, sizeof(what), "t%d gates beta", t);
        check_close(what, bv.data(), tolbuf.data() + HV, HV, 1e-5f);

        // the G13 fold: CPU consumes the GPU's own q/k/v/g/beta bits and the
        // CPU-chained state; state and output must match BITWISE every token
        memcpy(s_ref.data(), ssm_cache.data() + (size_t) row_ld * ssm_stride,
               ssm_stride * 4);
        ref_gdn_step(qn.data(), kn.data(), convout.data() + (size_t) 2 * HK * S,
                     gv.data(), bv.data(), s_ref.data(), attn_ref.data(), HV, HK, scale);
        memcpy(ssm_cache.data() + (size_t) row_st * ssm_stride, s_ref.data(),
               ssm_stride * 4);

        down(s_got.data(), d_ssm_cache + (size_t) row_st * ssm_stride, ssm_stride);
        down(attn_got.data(), d_attn, attn_got.size());
        snprintf(what, sizeof(what), "t%d ssm state (chained)", t);
        check_bitwise(what, s_got.data(), s_ref.data(), ssm_stride);
        snprintf(what, sizeof(what), "t%d gdn output", t);
        check_bitwise(what, attn_got.data(), attn_ref.data(), attn_got.size());

        // gated rmsnorm epilogue vs the GPU's attn bits, tolerance
        down(tolbuf.data(), d_y, (size_t) HV * S);
        ref_gated_rmsnorm(attn_got.data(), normw.data(), z.data(), attn_ref.data(), HV, eps);
        snprintf(what, sizeof(what), "t%d gated_rmsnorm", t);
        check_close(what, tolbuf.data(), attn_ref.data(), (size_t) HV * S, 1e-5f);
    }

    CUDA_CHECK(cudaFree(d_conv_cache)); CUDA_CHECK(cudaFree(d_ssm_cache));
    CUDA_CHECK(cudaFree(d_convw)); CUDA_CHECK(cudaFree(d_dt));
    CUDA_CHECK(cudaFree(d_aw)); CUDA_CHECK(cudaFree(d_normw));
    CUDA_CHECK(cudaFree(d_hist)); CUDA_CHECK(cudaFree(d_sbuf));
    CUDA_CHECK(cudaFree(d_win)); CUDA_CHECK(cudaFree(d_convout));
    CUDA_CHECK(cudaFree(d_qn)); CUDA_CHECK(cudaFree(d_kn));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_qkv)); CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_braw)); CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_row_ld)); CUDA_CHECK(cudaFree(d_row_st));
}

// ---------------------------------------------------------------------------
int main() {
    // shared GPUs: pick the device with the most free memory
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int best = 0;
    size_t best_free = 0;
    for (int d = 0; d < ndev; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        if (free_b > best_free) { best_free = free_b; best = d; }
    }
    CUDA_CHECK(cudaSetDevice(best));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, best));
    printf("device %d (%s), %.1f GiB free\n", best, prop.name,
           (double) best_free / (1 << 30));

    // the spin boundary needs all NBLK blocks co-resident
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mk_run, NTHR, 0));
    if (blocks_per_sm * prop.multiProcessorCount < NBLK) {
        fprintf(stderr, "grid of %d blocks not co-resident (%d/SM x %d SMs)\n",
                NBLK, blocks_per_sm, prop.multiProcessorCount);
        return 1;
    }

    CUDA_CHECK(cudaMalloc(&g_counter, sizeof(unsigned)));

    test_state_load_store();
    test_conv_ops();
    test_qk_l2norm();
    test_gdn_gates();
    test_gdn_step(48, 16, 0, 8);   // full model
    test_gdn_step(24, 8, 0, 5);    // per-GPU tensor-split half, sub-range
    test_gated_rmsnorm();
    test_chain(48, 16, 5);
    test_chain(24, 8, 3);

    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
