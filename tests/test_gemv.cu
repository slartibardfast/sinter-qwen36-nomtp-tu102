// Unit tests for the GEMV op family (k0/ops/gemv.cuh): activation
// quantization (q8_1), the Q4_0 / Q4_0_AR16 MMVQ dots (plain and SwiGLU-
// fused), and the F16 GEMVs (ssm_alpha/beta class + the lm_head vocab stream).
//
// A wrapper kernel (72 blocks x 384 threads, dynamic smem slab sized per the
// op's declared smem table) executes one Instr per launch the ABI way (range
// check on blockIdx.x). Each launch is a single antichain instruction, so no
// Y02 boundary/cooperative-launch is needed: every block independently stages
// its own copy of the mutable activations and computes its assigned rows.
//
// CPU reference: fp32 arithmetic that reproduces each op's declared fold order
// EXACTLY -- per-lane sequential fmaf accumulation (std::fmaf == device
// __fmaf_rn, correctly-rounded IEEE fma), scale muls pinned through volatile
// (== __fmul_rn, no host FMA contraction), and the 32-lane __shfl_xor butterfly
// warp_sum simulated per lane. f16<->f32 conversions go through the SAME
// __half2float / __float2half_rn used device-side (bit-identical). So:
//   - the dp4a integer sub-sums are exact by construction (integer add is
//     order-free); the MMVQ Q4_0 / AR16 / F16-GEMV end-to-end results come out
//     BITWISE-exact vs the host fold (reported, and asserted <= 1e-5 rel per
//     the f16/f32-fold policy);
//   - the fused SwiGLU and the quant s-field carry a transcendental / half
//     rounding, tolerance-checked at 1e-5.
//
// Math contract verified against the fork (software/llama.cpp/autoround
// @546eca8dc), cited inline where the reference encodes a fork detail:
//   - quantize_q8_1        quantize.cu:4-48   (d=amax/127, q=roundf(xi/d), s=Sum x)
//   - vec_dot_q4_0_q8_1    vecdotq.cuh:115-134 (d4*(sumi*d8 - 8*s8), split-half)
//   - vec_dot_q4_0_ar16    vecdotq.cuh:755-780 (d4*d8*sumi, -8 folded, no s-term)
//   - block layouts        ggml-common.h:187-269
//   - mul_mat_vec_f        mmvf.cu:7-374       (f16 weights, f32 activations)
//   - SwiGLU epilogue      mmvq.cu:572-575     (up * silu(gate))
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../core/isa.cuh"
#include "../core/sync.cuh"
#include "../k0/ops/gemv.cuh"

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
// wrapper kernel (the ABI shape: range check per instruction, one op per
// launch; 384 threads per block like the real interpreter).
static constexpr int NBLK = 72;
static constexpr int NTHR = 384;

__global__ void mk_run(const mk::Instr *prog, int n_instr) {
    extern __shared__ __align__(16) char smem[];
    for (int i = 0; i < n_instr; i++) {
        const mk::Instr &ins = prog[i];
        if (blockIdx.x < ins.block_lo || blockIdx.x >= ins.block_hi) continue;
        switch (ins.kind) {
            case mk::OP_QUANT_Q8_1:      mk::op_quant_q8_1(ins, smem); break;
            case mk::OP_MMVQ_Q4_0:       mk::op_mmvq_q4_0(ins, smem); break;
            case mk::OP_MMVQ_Q4_0_FUSED: mk::op_mmvq_q4_0_fused(ins, smem); break;
            case mk::OP_MMVQ_AR16:       mk::op_mmvq_ar16(ins, smem); break;
            case mk::OP_GEMV_F16:        mk::op_gemv_f16(ins, smem); break;
            case mk::OP_HEAD_GEMV_F16:   mk::op_head_gemv_f16(ins, smem); break;
            default: break;
        }
    }
}

// Per-op probe kernels: each calls exactly one op so ptxas -v prints an
// isolated register/smem line per op (mk_run alone reports only the max).
__global__ void probe_quant_q8_1(const mk::Instr *p)      { extern __shared__ __align__(16) char s[]; mk::op_quant_q8_1(*p, s); }
__global__ void probe_mmvq_q4_0(const mk::Instr *p)       { extern __shared__ __align__(16) char s[]; mk::op_mmvq_q4_0(*p, s); }
__global__ void probe_mmvq_q4_0_fused(const mk::Instr *p) { extern __shared__ __align__(16) char s[]; mk::op_mmvq_q4_0_fused(*p, s); }
__global__ void probe_mmvq_ar16(const mk::Instr *p)       { extern __shared__ __align__(16) char s[]; mk::op_mmvq_ar16(*p, s); }
__global__ void probe_gemv_f16(const mk::Instr *p)        { extern __shared__ __align__(16) char s[]; mk::op_gemv_f16(*p, s); }
__global__ void probe_head_gemv_f16(const mk::Instr *p)   { extern __shared__ __align__(16) char s[]; mk::op_head_gemv_f16(*p, s); }

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

static mk::Instr *g_prog = nullptr;

static void run_op(const mk::Instr &ins, size_t smem_bytes) {
    CUDA_CHECK(cudaMemcpy(g_prog, &ins, sizeof(ins), cudaMemcpyHostToDevice));
    mk_run<<<NBLK, NTHR, smem_bytes>>>(g_prog, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// CPU reference helpers: mul/add pinned through volatile so the host compiler
// cannot FMA-contract (mirroring __fmul_rn/__fadd_rn); std::fmaf is the
// correctly-rounded IEEE fma == device __fmaf_rn.
static inline float fmul_rn(float a, float b) { volatile float r = a * b; return r; }
static inline float fadd_rn(float a, float b) { volatile float r = a + b; return r; }

// bit-identical to device __half2float / __float2half_rn (same intrinsics).
static inline float h2f(uint16_t h) {
    __half hh; memcpy(&hh, &h, 2); return __half2float(hh);
}
static inline uint16_t f2h(float f) {
    __half hh = __float2half_rn(f); uint16_t o; memcpy(&o, &hh, 2); return o;
}

// Exact simulation of warp_sum's __shfl_xor butterfly (offsets 16..1): every
// lane converges to the same bits; return lane 0's total.
static float butterfly32(const float *lane_val) {
    float v[32], nv[32];
    memcpy(v, lane_val, sizeof(v));
    for (int off = 16; off > 0; off >>= 1) {
        for (int l = 0; l < 32; l++) nv[l] = fadd_rn(v[l], v[l ^ off]);
        memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

static inline float silu_ref(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// reporting
static int g_fail = 0;
static double g_worst_rel = 0.0;   // worst rel error over all tol-checked ops

static void check_rel(const char *what, const float *got, const float *want,
                      size_t n, float rtol) {
    size_t bad = 0, first = (size_t)-1, badbits = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t a, b; memcpy(&a, &got[i], 4); memcpy(&b, &want[i], 4);
        if (a != b) badbits++;
        const float denom = fmaxf(fabsf(want[i]), 1e-6f);
        const float rel = fabsf(got[i] - want[i]) / denom;
        if (rel > rtol) { if (bad == 0) first = i; bad++; }
        if (rel > worst) worst = rel;
    }
    if ((double)worst > g_worst_rel) g_worst_rel = worst;
    if (bad) {
        g_fail++;
        printf("FAIL [<=%.0e ] %-26s %zu/%zu beyond tol (first %zu: got %.9g want %.9g), worst rel %.3g\n",
               (double)rtol, what, bad, n, first, got[first], want[first], (double)worst);
    } else {
        printf("ok   [<=%.0e ] %-26s %zu vals, worst rel %.3g%s\n",
               (double)rtol, what, n, (double)worst,
               badbits == 0 ? "  (bitwise-exact)" : "");
    }
}

static void check_int_exact(const char *what, const int *got, const int *want, size_t n) {
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) { if (bad == 0) first = i; bad++; }
    if (bad) {
        g_fail++;
        printf("FAIL [int  ]  %-26s %zu/%zu mismatched (first %zu: got %d want %d)\n",
               what, bad, n, first, got[first], want[first]);
    } else {
        printf("ok   [int  ]  %-26s %zu vals (exact)\n", what, n);
    }
}

// ---------------------------------------------------------------------------
// device buffer helpers
static std::mt19937 g_rng(0x67656d76u);   // 'gemv'
static void fill_rand(std::vector<float> &v, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto &x : v) x = d(g_rng);
}
static void *dmalloc(size_t bytes) { void *p; CUDA_CHECK(cudaMalloc(&p, bytes)); return p; }

// ---------------------------------------------------------------------------
// host q8_1 block builder (quantize_q8_1, quantize.cu:31-47). n<=32 real
// elements; the rest quantize as zeros (the MATRIX_ROW_PADDING behaviour).
static void host_quant_block(const float *x, int n, uint8_t *out36) {
    float xi[32];
    for (int i = 0; i < 32; i++) xi[i] = i < n ? x[i] : 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xi[i]));  // max: order-free
    const float sum = butterfly32(xi);                             // s = Sum x (butterfly)
    const float d = amax / 127.0f;
    const uint16_t dh = f2h(d), sh = f2h(sum);
    memcpy(out36 + 0, &dh, 2);
    memcpy(out36 + 2, &sh, 2);
    int8_t *q = (int8_t *)(out36 + 4);
    for (int i = 0; i < 32; i++)
        q[i] = amax == 0.0f ? 0 : (int8_t)(int)roundf(xi[i] / d);
}

// quantize a full ncols-vector into ncols/32 q8_1 blocks (all blocks full).
static std::vector<uint8_t> host_quant_vec(const std::vector<float> &x) {
    const int nblk = (int)x.size() / 32;
    std::vector<uint8_t> y((size_t)nblk * 36);
    for (int b = 0; b < nblk; b++) host_quant_block(&x[b * 32], 32, &y[(size_t)b * 36]);
    return y;
}

// ---------------------------------------------------------------------------
// host reference dots (exact declared fold orders, cf. gemv.cuh)

// Q4_0 row dot (mmvq_q40_row order): per lane ascending pair, block 2p then
// 2p+1, then tail blocks ascending; butterfly across lanes.
static float ref_mmvq_q40_row(const uint8_t *w_row, const uint8_t *q8, int ncols) {
    const int nblk = ncols / 32;
    const int npair = nblk / 2;
    const int npair_full = npair & ~31;
    auto blk = [&](int b, float acc) -> float {
        const uint8_t *wb = w_row + (size_t)b * 18;
        const uint8_t *qb = q8 + (size_t)b * 36;
        const float d4 = h2f(*(const uint16_t *)wb);
        const float d8 = h2f(*(const uint16_t *)(qb + 0));
        const float s8 = h2f(*(const uint16_t *)(qb + 2));
        const uint8_t *qs = wb + 2;
        const int8_t *qi = (const int8_t *)(qb + 4);
        int sumi = 0;
        for (int e = 0; e < 32; e++) {
            const int nib = e < 16 ? (qs[e] & 0x0F) : (qs[e - 16] >> 4);
            sumi += nib * (int)qi[e];
        }
        return std::fmaf(d4, std::fmaf((float)sumi, d8, fmul_rn(-8.0f, s8)), acc);
    };
    float lanes[32];
    for (int lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (int p = lane; p < npair_full; p += 32) { acc = blk(2 * p, acc); acc = blk(2 * p + 1, acc); }
        for (int kb = 2 * npair_full + lane; kb < nblk; kb += 32) acc = blk(kb, acc);
        lanes[lane] = acc;
    }
    return butterfly32(lanes);
}

// Q4_0_AR16 row dot (mmvq_ar16_row order): per lane ascending pair, grouping
// d8*(dA*s0 + dB*s1); tail single blocks fold (dk*d8)*sumi. -8 in the nibble,
// no s-correction (vecdotq.cuh:778-779).
static float ref_mmvq_ar16_row(const uint8_t *w_row, const uint8_t *q8, int ncols) {
    const int nblk = ncols / 16;
    const int npair = nblk / 2;
    const int npair_full = npair & ~31;
    auto ar16_blk_dot = [&](const uint8_t *wblk10, const int8_t *qi16) -> int {
        const uint8_t *qs = wblk10 + 2;   // 8 bytes = 16 element-interleaved nibbles
        int sumi = 0;
        for (int e = 0; e < 16; e++) {
            const int nib = (e & 1) ? (qs[e >> 1] >> 4) : (qs[e >> 1] & 0x0F);
            sumi += (nib - 8) * (int)qi16[e];
        }
        return sumi;
    };
    float lanes[32];
    for (int lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (int p = lane; p < npair_full; p += 32) {
            const uint8_t *qb = q8 + (size_t)p * 36;
            const float d8 = h2f(*(const uint16_t *)qb);
            const int8_t *qi = (const int8_t *)(qb + 4);
            const uint8_t *wa = w_row + (size_t)(2 * p) * 10;
            const uint8_t *wbk = w_row + (size_t)(2 * p + 1) * 10;
            const float dA = h2f(*(const uint16_t *)wa);
            const float dB = h2f(*(const uint16_t *)wbk);
            const int s0 = ar16_blk_dot(wa, qi + 0);
            const int s1 = ar16_blk_dot(wbk, qi + 16);
            const float t = std::fmaf(dB, (float)s1, fmul_rn(dA, (float)s0));
            acc = std::fmaf(d8, t, acc);
        }
        for (int kb = 2 * npair_full + lane; kb < nblk; kb += 32) {
            const uint8_t *wk = w_row + (size_t)kb * 10;
            const uint8_t *qb = q8 + (size_t)(kb >> 1) * 36;
            const float dk = h2f(*(const uint16_t *)wk);
            const float d8 = h2f(*(const uint16_t *)qb);
            const int8_t *qi = (const int8_t *)(qb + 4);
            const int sumi = ar16_blk_dot(wk, qi + (kb & 1) * 16);
            acc = std::fmaf(fmul_rn(dk, d8), (float)sumi, acc);
        }
        lanes[lane] = acc;
    }
    return butterfly32(lanes);
}

// F16 GEMV row dot (gemv_f16_row order): per lane ascending group of 8,
// elements in order, fmaf(w,x); butterfly across lanes.
static float ref_gemv_f16_row(const uint16_t *w_row, const float *x, int ncols) {
    const int ngrp = ncols / 8;
    float lanes[32];
    for (int lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (int g = lane; g < ngrp; g += 32)
            for (int j = 0; j < 8; j++) {
                const int k = 8 * g + j;
                acc = std::fmaf(h2f(w_row[k]), x[k], acc);
            }
        lanes[lane] = acc;
    }
    return butterfly32(lanes);
}

// ---------------------------------------------------------------------------
// OP_QUANT_Q8_1
static void test_quant_q8_1(uint32_t ne00, uint32_t ne0_padded) {
    std::vector<float> x(ne00);
    fill_rand(x, -3.0f, 3.0f);
    // force one block to be all-zero-magnitude to exercise the amax==0 path
    if (ne00 >= 32) for (uint32_t i = 0; i < 32; i++) x[i] = 0.0f;

    const uint32_t nblk = ne0_padded / 32u;
    float *d_x = (float *)dmalloc((size_t)ne00 * 4);
    void  *d_y = dmalloc((size_t)nblk * 36);
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)ne00 * 4, cudaMemcpyHostToDevice));

    mk::QuantQ8_1Args a = {d_x, d_y, ne00, ne0_padded};
    run_op(make_instr(mk::OP_QUANT_Q8_1, 0, NBLK, a), 16);

    std::vector<uint8_t> got((size_t)nblk * 36), want((size_t)nblk * 36);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, got.size(), cudaMemcpyDeviceToHost));
    for (uint32_t b = 0; b < nblk; b++) {
        const int n = (int)std::min<uint32_t>(32u, b * 32u < ne00 ? ne00 - b * 32u : 0u);
        host_quant_block(n > 0 ? &x[b * 32] : nullptr, n, &want[(size_t)b * 36]);
    }
    // decode d, s (half->float) and q (int8) into parallel arrays
    std::vector<float> gd(nblk), wd(nblk), gs(nblk), ws(nblk);
    std::vector<int>   gq((size_t)nblk * 32), wq((size_t)nblk * 32);
    for (uint32_t b = 0; b < nblk; b++) {
        gd[b] = h2f(*(uint16_t *)&got[b * 36 + 0]); wd[b] = h2f(*(uint16_t *)&want[b * 36 + 0]);
        gs[b] = h2f(*(uint16_t *)&got[b * 36 + 2]); ws[b] = h2f(*(uint16_t *)&want[b * 36 + 2]);
        for (int j = 0; j < 32; j++) {
            gq[b * 32 + j] = (int)(int8_t)got[b * 36 + 4 + j];
            wq[b * 32 + j] = (int)(int8_t)want[b * 36 + 4 + j];
        }
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "quant_q8_1 q (ne00=%u)", ne00);
    check_int_exact(nm, gq.data(), wq.data(), gq.size());
    snprintf(nm, sizeof(nm), "quant_q8_1 d (ne00=%u)", ne00);
    check_rel(nm, gd.data(), wd.data(), nblk, 1e-5f);
    snprintf(nm, sizeof(nm), "quant_q8_1 s (ne00=%u)", ne00);
    check_rel(nm, gs.data(), ws.data(), nblk, 1e-5f);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
}

// ---------------------------------------------------------------------------
// OP_MMVQ_Q4_0
static void test_mmvq_q4_0(int ncols, int nrows, int row_lo, int row_hi) {
    const int nblk = ncols / 32;
    std::vector<uint8_t> w((size_t)nrows * nblk * 18);
    for (auto &b : w) b = (uint8_t)(g_rng() & 0xFF);
    // give each block a sane f16 delta (small magnitude) so dst stays finite
    std::uniform_real_distribution<float> dd(0.02f, 0.12f);
    for (int r = 0; r < nrows; r++)
        for (int b = 0; b < nblk; b++) {
            const uint16_t dh = f2h(dd(g_rng) * (g_rng() & 1 ? 1.f : -1.f));
            memcpy(&w[((size_t)r * nblk + b) * 18], &dh, 2);
        }
    std::vector<float> x(ncols); fill_rand(x, -1.5f, 1.5f);
    std::vector<uint8_t> q8 = host_quant_vec(x);

    void  *d_w  = dmalloc(w.size());
    void  *d_q8 = dmalloc(q8.size());
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q8, q8.data(), q8.size(), cudaMemcpyHostToDevice));
    // sentinel so we can verify rows outside [row_lo,row_hi) stay untouched
    std::vector<float> sent(nrows, -123456.0f);
    CUDA_CHECK(cudaMemcpy(d_dst, sent.data(), (size_t)nrows * 4, cudaMemcpyHostToDevice));

    mk::MmvqQ40Args a = {d_w, d_q8, d_dst, (uint32_t)ncols, (uint32_t)row_lo, (uint32_t)row_hi};
    run_op(make_instr(mk::OP_MMVQ_Q4_0, 0, NBLK, a), (size_t)nblk * 36);

    std::vector<float> got(nrows), want(nrows, -123456.0f);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, (size_t)nrows * 4, cudaMemcpyDeviceToHost));
    for (int r = row_lo; r < row_hi; r++)
        want[r] = ref_mmvq_q40_row(&w[(size_t)r * nblk * 18], q8.data(), ncols);
    char nm[80];
    snprintf(nm, sizeof(nm), "mmvq_q4_0 K=%d [%d,%d)", ncols, row_lo, row_hi);
    check_rel(nm, got.data(), want.data(), nrows, 1e-5f);

    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
// OP_MMVQ_Q4_0_FUSED
static void test_mmvq_q4_0_fused(int ncols, int nrows) {
    const int nblk = ncols / 32;
    auto make_w = [&]() {
        std::vector<uint8_t> w((size_t)nrows * nblk * 18);
        for (auto &b : w) b = (uint8_t)(g_rng() & 0xFF);
        std::uniform_real_distribution<float> dd(0.02f, 0.12f);
        for (int r = 0; r < nrows; r++)
            for (int b = 0; b < nblk; b++) {
                const uint16_t dh = f2h(dd(g_rng) * (g_rng() & 1 ? 1.f : -1.f));
                memcpy(&w[((size_t)r * nblk + b) * 18], &dh, 2);
            }
        return w;
    };
    std::vector<uint8_t> wu = make_w(), wg = make_w();
    std::vector<float> x(ncols); fill_rand(x, -1.5f, 1.5f);
    std::vector<uint8_t> q8 = host_quant_vec(x);

    void *d_wu = dmalloc(wu.size()), *d_wg = dmalloc(wg.size()), *d_q8 = dmalloc(q8.size());
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_wu, wu.data(), wu.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wg, wg.data(), wg.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q8, q8.data(), q8.size(), cudaMemcpyHostToDevice));

    mk::MmvqQ40FusedArgs a = {d_wu, d_wg, d_q8, d_dst, (uint32_t)ncols, 0, (uint32_t)nrows};
    run_op(make_instr(mk::OP_MMVQ_Q4_0_FUSED, 0, NBLK, a), (size_t)nblk * 36);

    std::vector<float> got(nrows), want(nrows);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, (size_t)nrows * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < nrows; r++) {
        const float up   = ref_mmvq_q40_row(&wu[(size_t)r * nblk * 18], q8.data(), ncols);
        const float gate = ref_mmvq_q40_row(&wg[(size_t)r * nblk * 18], q8.data(), ncols);
        want[r] = fmul_rn(up, silu_ref(gate));   // mmvq.cu:572-575
    }
    char nm[64];
    snprintf(nm, sizeof(nm), "mmvq_q4_0_fused K=%d", ncols);
    check_rel(nm, got.data(), want.data(), nrows, 1e-5f);

    CUDA_CHECK(cudaFree(d_wu)); CUDA_CHECK(cudaFree(d_wg));
    CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
// OP_MMVQ_AR16
static void test_mmvq_ar16(int ncols, int nrows) {
    const int nblk = ncols / 16;          // AR16 blocks per row
    const int nq8  = ncols / 32;          // q8_1 blocks
    std::vector<uint8_t> w((size_t)nrows * nblk * 10);
    for (auto &b : w) b = (uint8_t)(g_rng() & 0xFF);
    std::uniform_real_distribution<float> dd(0.02f, 0.12f);
    for (int r = 0; r < nrows; r++)
        for (int b = 0; b < nblk; b++) {
            const uint16_t dh = f2h(dd(g_rng) * (g_rng() & 1 ? 1.f : -1.f));
            memcpy(&w[((size_t)r * nblk + b) * 10], &dh, 2);
        }
    std::vector<float> x(ncols); fill_rand(x, -1.5f, 1.5f);
    std::vector<uint8_t> q8 = host_quant_vec(x);
    (void)nq8;

    void *d_w = dmalloc(w.size()), *d_q8 = dmalloc(q8.size());
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q8, q8.data(), q8.size(), cudaMemcpyHostToDevice));

    mk::MmvqAr16Args a = {d_w, d_q8, d_dst, (uint32_t)ncols, 0, (uint32_t)nrows};
    run_op(make_instr(mk::OP_MMVQ_AR16, 0, NBLK, a), (size_t)nq8 * 36);

    std::vector<float> got(nrows), want(nrows);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, (size_t)nrows * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < nrows; r++)
        want[r] = ref_mmvq_ar16_row(&w[(size_t)r * nblk * 10], q8.data(), ncols);
    char nm[64];
    snprintf(nm, sizeof(nm), "mmvq_ar16 K=%d", ncols);
    check_rel(nm, got.data(), want.data(), nrows, 1e-5f);

    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
// OP_GEMV_F16 / OP_HEAD_GEMV_F16
static void test_gemv_f16(int ncols, int nrows, bool head) {
    std::vector<float> wf((size_t)nrows * ncols), x(ncols);
    fill_rand(wf, -1.0f, 1.0f);
    fill_rand(x, -1.0f, 1.0f);
    std::vector<uint16_t> w(wf.size());
    for (size_t i = 0; i < wf.size(); i++) w[i] = f2h(wf[i]);

    half  *d_w = (half *)dmalloc(w.size() * 2);
    float *d_x = (float *)dmalloc((size_t)ncols * 4);
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)ncols * 4, cudaMemcpyHostToDevice));

    mk::GemvF16Args a = {d_w, d_x, d_dst, (uint32_t)ncols, 0, (uint32_t)nrows};
    const uint16_t kind = head ? mk::OP_HEAD_GEMV_F16 : mk::OP_GEMV_F16;
    run_op(make_instr(kind, 0, NBLK, a), (size_t)ncols * 4);

    std::vector<float> got(nrows), want(nrows);
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, (size_t)nrows * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < nrows; r++)
        want[r] = ref_gemv_f16_row(&w[(size_t)r * ncols], x.data(), ncols);
    char nm[64];
    snprintf(nm, sizeof(nm), "%s K=%d", head ? "head_gemv_f16" : "gemv_f16", ncols);
    check_rel(nm, got.data(), want.data(), nrows, 1e-5f);

    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
// bandwidth (contended-indicative): weight-stream GB/s over a repeated loop.
static double g_gbps_q40 = 0.0, g_gbps_head = 0.0;

static void bench_q4_0(int ncols, int nrows, int iters) {
    const int nblk = ncols / 32;
    std::vector<uint8_t> w((size_t)nrows * nblk * 18);
    for (auto &b : w) b = (uint8_t)(g_rng() & 0xFF);
    std::vector<float> x(ncols); fill_rand(x, -1.f, 1.f);
    std::vector<uint8_t> q8 = host_quant_vec(x);
    void *d_w = dmalloc(w.size()), *d_q8 = dmalloc(q8.size());
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q8, q8.data(), q8.size(), cudaMemcpyHostToDevice));

    mk::MmvqQ40Args a = {d_w, d_q8, d_dst, (uint32_t)ncols, 0, (uint32_t)nrows};
    mk::Instr ins = make_instr(mk::OP_MMVQ_Q4_0, 0, NBLK, a);
    CUDA_CHECK(cudaMemcpy(g_prog, &ins, sizeof(ins), cudaMemcpyHostToDevice));
    const size_t smem = (size_t)nblk * 36;

    for (int i = 0; i < 5; i++) mk_run<<<NBLK, NTHR, smem>>>(g_prog, 1);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) mk_run<<<NBLK, NTHR, smem>>>(g_prog, 1);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const double bytes = (double)iters * w.size();   // weight stream (dominant)
    g_gbps_q40 = bytes / (ms * 1e-3) / 1e9;
    printf("bandwidth  mmvq_q4_0  K=%d rows=%d: %.1f GB/s (weight stream, contended-indicative; %.2f ms/%d it)\n",
           ncols, nrows, g_gbps_q40, ms, iters);
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_q8)); CUDA_CHECK(cudaFree(d_dst));
}

static void bench_head(int ncols, int nrows, int iters) {
    std::vector<uint16_t> w((size_t)nrows * ncols);
    for (auto &h : w) h = (uint16_t)(g_rng() & 0xFFFF);
    std::vector<float> x(ncols); fill_rand(x, -1.f, 1.f);
    half *d_w = (half *)dmalloc(w.size() * 2);
    float *d_x = (float *)dmalloc((size_t)ncols * 4);
    float *d_dst = (float *)dmalloc((size_t)nrows * 4);
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)ncols * 4, cudaMemcpyHostToDevice));

    mk::GemvF16Args a = {d_w, d_x, d_dst, (uint32_t)ncols, 0, (uint32_t)nrows};
    mk::Instr ins = make_instr(mk::OP_HEAD_GEMV_F16, 0, NBLK, a);
    CUDA_CHECK(cudaMemcpy(g_prog, &ins, sizeof(ins), cudaMemcpyHostToDevice));
    const size_t smem = (size_t)ncols * 4;

    for (int i = 0; i < 5; i++) mk_run<<<NBLK, NTHR, smem>>>(g_prog, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1; CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) mk_run<<<NBLK, NTHR, smem>>>(g_prog, 1);
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const double bytes = (double)iters * w.size() * 2.0;   // f16 weight stream
    g_gbps_head = bytes / (ms * 1e-3) / 1e9;
    printf("bandwidth  head_gemv  K=%d rows=%d: %.1f GB/s (f16 weight stream, contended-indicative; %.2f ms/%d it)\n",
           ncols, nrows, g_gbps_head, ms, iters);
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
int main() {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    int best = 0; size_t best_free = 0;
    for (int d = 0; d < ndev; d++) {
        CUDA_CHECK(cudaSetDevice(d));
        size_t f = 0, t = 0; CUDA_CHECK(cudaMemGetInfo(&f, &t));
        if (f > best_free) { best_free = f; best = d; }
    }
    CUDA_CHECK(cudaSetDevice(best));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, best));
    printf("device %d (%s), %.1f GiB free\n", best, prop.name, (double)best_free / (1 << 30));

    g_prog = (mk::Instr *)dmalloc(sizeof(mk::Instr));

    // --- quant: exercise the padding straddle (ne00 not a multiple of 32)
    test_quant_q8_1(5000, 5120);
    test_quant_q8_1(4096, 4096);

    // --- MMVQ Q4_0: K=5120 has a lane-balanced tail (npair=80 -> full 64 + 16
    //     tail pairs); K=2048 is tail-free (npair=32). sub-range checks the
    //     absolute-row indexing + that out-of-range dst stays untouched.
    test_mmvq_q4_0(5120, 64, 0, 64);
    test_mmvq_q4_0(5120, 64, 8, 40);
    test_mmvq_q4_0(2048, 48, 0, 48);

    // --- fused SwiGLU (up|gate over one activation)
    test_mmvq_q4_0_fused(5120, 64);

    // --- AR16: K=6144 tail-free (npair=192); K=1056 has a 2-block tail
    //     (npair=33 -> full 32 + blocks 64,65 via the kb-parity q8 halves)
    test_mmvq_ar16(6144, 64);
    test_mmvq_ar16(1056, 48);

    // --- F16 GEMV (ssm class) and the lm_head vocab stream
    test_gemv_f16(5120, 64, false);
    test_gemv_f16(5120, 128, true);

    // --- bandwidth (contended-indicative)
    bench_q4_0(5120, 8704, 300);       //  25 MB stream x 300
    bench_head(5120, 61440, 40);       // 629 MB stream x 40

    CUDA_CHECK(cudaFree(g_prog));

    if (g_fail) { printf("FAILED: %d check(s)\n", g_fail); return 1; }
    printf("all checks passed (worst rel %.3g)\n", g_worst_rel);
    return 0;
}
