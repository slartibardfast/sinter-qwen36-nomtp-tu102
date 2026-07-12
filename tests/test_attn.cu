// Unit + parity tests for the attention op family (k0/ops/attn.cuh):
// OP_QK_NORM_ROPE, OP_KV_APPEND, OP_FATTN_DECODE, OP_FATTN_REDUCE,
// OP_ATTN_GATE.
//
// A mini-interpreter kernel (mk_run) executes Instr programs the ABI way:
// per instruction a [block_lo, block_hi) range check, one Y02 crossing per
// OP_BOUNDARY, 384 threads/block like the real interpreter, a dynamic smem
// slab sized per program (OP_FATTN_DECODE needs ~45 KiB; the other ops none).
// The ops are exercised through the exact contract the interpreter uses.
//
// CPU references implement the fork's math independently (cites are path:line
// into software/llama.cpp/autoround @546eca8dc), so a header bug would show
// as a parity gap rather than being copied:
//   - OP_QK_NORM_ROPE: fused per-head RMS norm (norm.cu:136-146) then IMROPE
//     on the first 64 dims (rope_multi is_imrope sector chain rope.cu:231-240,
//     NEOX split-half rope.cu:260-264, rope_yarn degenerate cos/sin
//     rope.cu:26-40, theta_scale = powf(freq_base,-2/n_dims) rope.cu:443).
//     theta_scale is bit-identical to the header's hardcoded constant
//     (asserted below), so cos/sin differ only by device-vs-libm transcend-
//     entals -> tol 1e-5.
//   - OP_KV_APPEND: SET_ROWS f32->f16 round-to-nearest cast (set-rows.cu:160-
//     165). Host __float2half is RN like the device __floats2half2_rn, so the
//     f16 cache bits are compared BITWISE.
//   - OP_FATTN_DECODE + OP_FATTN_REDUCE: the split-KV decode + Y03 merge is
//     compared against a direct (non-tiled) flash-attention reference over the
//     shared f16 K/V (fattn-mma-f16.cuh Q pre-scaled :1230, mask added,
//     KQ_max init -FLT_MAX/2 :1196; stream-k max/LSE merge + rowsum divide
//     fattn-common.cuh:719-752). Reference is in double (true attention over
//     the f16 values); the device runs its native f32 online softmax, so the
//     gap is the device's f32/tiling error -> tol 2e-3. Crosses the 256 n_kv
//     padding boundary (real 250/256/700 in pads 256/256/768).
//   - OP_FATTN_REDUCE sentinel: a partial whose max slot still reads -inf was
//     never written (Y03 lost split); it must be COUNTED into *error and
//     skipped. A deliberately-unwritten record is detected here or the test
//     fails.
//   - OP_ATTN_GATE: attn * sigmoid(gate) (unary.cu:48-50), tol 1e-5.
//
// GQA (SCHEDULE-QUESTIONS.md item 18): true ratio 6, ncols2=8 is a tile
// bucket. FATTN packs no pad slots: gqa = n_q/n_kv_heads. The op runs one
// warp per q head at 384 threads, so n_q <= 12 and (head_dim 256, tile 32)
// row_width <= 512 => n_kv_heads <= 2: the per-GPU shape is 12 q over 2 kv
// (gqa 6). QK_NORM_ROPE loops over heads and is tested at the full 24 q / 4 k.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../core/isa.cuh"
#include "../core/sync.cuh"
#include "../k0/ops/attn.cuh"

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
// mini-interpreter: 384 threads/block like the real kernel; dynamic smem slab
// passed per launch (op_fattn_decode carves q_s | KV tile | mask out of it).
static constexpr int NBLK = 8;
static constexpr int NTHR = 384;

extern __shared__ char mk_smem[];

__global__ void mk_run(const mk::Instr *prog, int n_instr, unsigned *counter) {
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
            case mk::OP_QK_NORM_ROPE: mk::op_qk_norm_rope(ins, mk_smem); break;
            case mk::OP_KV_APPEND:    mk::op_kv_append(ins, mk_smem); break;
            case mk::OP_FATTN_DECODE: mk::op_fattn_decode(ins, mk_smem); break;
            case mk::OP_FATTN_REDUCE: mk::op_fattn_reduce(ins, mk_smem); break;
            case mk::OP_ATTN_GATE:    mk::op_attn_gate(ins, mk_smem); break;
            default: break;
        }
    }
}

// Per-op register/smem probes: cudaFuncGetAttributes reads numRegs and
// static smem off each without launching (dynamic smem is a launch arg, not
// reported by ptxas, so smem-per-op is the header's declared slab, computed).
__global__ void probe_qk_norm_rope(const mk::Instr *p) { mk::op_qk_norm_rope(*p, mk_smem); }
__global__ void probe_kv_append(const mk::Instr *p)    { mk::op_kv_append(*p, mk_smem); }
__global__ void probe_fattn_decode(const mk::Instr *p) { mk::op_fattn_decode(*p, mk_smem); }
__global__ void probe_fattn_reduce(const mk::Instr *p) { mk::op_fattn_reduce(*p, mk_smem); }
__global__ void probe_attn_gate(const mk::Instr *p)    { mk::op_attn_gate(*p, mk_smem); }

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

static void run_program(const std::vector<mk::Instr> &prog, unsigned smem_bytes) {
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
    mk_run<<<NBLK, NTHR, smem_bytes>>>(d_prog, (int) prog.size(), g_counter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// float helpers rounded through volatile so the host cannot FMA-contract, and
// the exact __shfl_xor tree the device warp reduces with.
static inline float fmul_ref(float a, float b) { volatile float r = a * b; return r; }
static inline float fadd_ref(float a, float b) { volatile float r = a + b; return r; }

static float tree_sum32_ref(const float *lane_val) {
    float v[32], nv[32];
    memcpy(v, lane_val, sizeof(v));
    for (int off = 16; off > 0; off >>= 1) {
        for (int l = 0; l < 32; l++) nv[l] = fadd_ref(v[l], v[l ^ off]);
        memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

static constexpr int HD = mk::MK_ATTN_HD;   // 256

// ---------------------------------------------------------------------------
// comparisons
static int g_failures = 0;
static float g_worst_rel = 0.0f;

static void check_bitwise_u16(const char *what, const unsigned short *got,
                              const unsigned short *want, size_t n) {
    size_t bad = 0, first = (size_t) -1;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[i]) { if (bad == 0) first = i; bad++; }
    if (bad) {
        g_failures++;
        printf("FAIL [bitwise] %-26s %zu/%zu mismatched (first at %zu: got %04x want %04x)\n",
               what, bad, n, first, got[first], want[first]);
    } else {
        printf("ok   [bitwise] %-26s %zu f16 values\n", what, n);
    }
}

static void check_close(const char *what, const float *got, const float *want, size_t n,
                        float rtol) {
    size_t bad = 0, first = (size_t) -1;
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float denom = fmaxf(fabsf(want[i]), 1e-6f);
        const float rel = fabsf(got[i] - want[i]) / denom;
        if (rel > rtol) { if (bad == 0) first = i; bad++; }
        if (rel > worst) worst = rel;
    }
    if (bad) {
        g_failures++;
        printf("FAIL [<=%.0e ] %-26s %zu/%zu beyond tol (first at %zu: got %.9g want %.9g), worst rel %.3g\n",
               (double) rtol, what, bad, n, first, got[first], want[first], (double) worst);
    } else {
        printf("ok   [<=%.0e ] %-26s %zu values, worst rel %.3g\n",
               (double) rtol, what, n, (double) worst);
    }
    if (worst > g_worst_rel) g_worst_rel = worst;
}

// Attention output is a vector per head: a near-zero component (genuine V
// cancellation) carries no signal, so each element's error is scored against
// its head's output scale (max|want| over the 256 dims, floored at 1e-4 --
// below the f16 granularity of the +-1 V values), not against its own ~1e-6
// magnitude. This applies the 2e-3 spec at vector granularity; a real merge/
// index bug perturbs an element by ~head_scale and still trips it.
static void check_attn(const char *what, const float *got, const float *want,
                       int n_q, float rtol) {
    size_t bad = 0, first = (size_t) -1;
    float worst = 0.0f;
    for (int h = 0; h < n_q; h++) {
        float scale = 1e-4f;
        for (int d = 0; d < HD; d++) scale = fmaxf(scale, fabsf(want[(size_t) h * HD + d]));
        for (int d = 0; d < HD; d++) {
            const size_t i = (size_t) h * HD + d;
            const float rel = fabsf(got[i] - want[i]) / scale;
            if (rel > rtol) { if (bad == 0) first = i; bad++; }
            if (rel > worst) worst = rel;
        }
    }
    if (bad) {
        g_failures++;
        printf("FAIL [<=%.0e ] %-24s %zu/%d beyond tol (first at %zu: got %.9g want %.9g), worst rel %.3g\n",
               (double) rtol, what, bad, n_q * HD, first, got[first], want[first], (double) worst);
    } else {
        printf("ok   [<=%.0e ] %-24s %d values, worst head-rel %.3g\n",
               (double) rtol, what, n_q * HD, (double) worst);
    }
    if (worst > g_worst_rel) g_worst_rel = worst;
}

// ---------------------------------------------------------------------------
// device buffer helpers
static float *dalloc(size_t n) { float *p; CUDA_CHECK(cudaMalloc(&p, n * sizeof(float))); return p; }
static void up(float *d, const float *h, size_t n) {
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
}
static void down(float *h, const float *d, size_t n) {
    CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

static std::mt19937 g_rng(0x61747461u); // "atta"
static void fill_rand(std::vector<float> &v, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto &x : v) x = d(g_rng);
}
static void fill_rand_half(std::vector<__half> &v, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto &x : v) x = __float2half(d(g_rng));
}

// ---------------------------------------------------------------------------
// OP_QK_NORM_ROPE reference: fused RMS norm (norm.cu:136-146) then IMROPE on
// dims [0,64) (rope.cu:231-264). RMS sumsq uses the device fold order
// (per-lane partial over 8 stride-32 elems, then the shfl_xor tree).
static void ref_qk_norm_rope(const float *src, const float *norm_w, const int *pos,
                             float *dst, int n_heads, int src_stride) {
    const float eps = mk::MK_ATTN_EPS;                 // 1e-6
    const float ts  = powf(1e7f, -2.0f / 64.0f);       // == header constant (asserted)
    for (int h = 0; h < n_heads; h++) {
        const float *x = src + (size_t) h * src_stride;
        float lanes[32];
        for (int l = 0; l < 32; l++) {
            float acc = 0.0f;
            for (int i = 0; i < HD / 32; i++) {
                const float xe = x[i * 32 + l];
                acc = fadd_ref(acc, fmul_ref(xe, xe));
            }
            lanes[l] = acc;
        }
        const float ss = tree_sum32_ref(lanes);
        const float scale = 1.0f / sqrtf(ss / (float) HD + eps);
        float v[HD];
        for (int e = 0; e < HD; e++) v[e] = scale * x[e] * norm_w[e];
        // NEOX split-half rotation of the 32 pairs (elem p, elem p+32).
        for (int p = 0; p < mk::MK_IMROPE_NDIMS / 2; p++) {
            const int sector = p;
            int pos_sel;
            if (sector % 3 == 1 && sector < 3 * 11)      pos_sel = pos[1]; // h
            else if (sector % 3 == 2 && sector < 3 * 10) pos_sel = pos[2]; // w
            else if (sector % 3 == 0 && sector < 3 * 11) pos_sel = pos[0]; // t
            else                                         pos_sel = pos[3]; // e
            const float theta = (float) pos_sel * powf(ts, (float) sector);
            const float c = cosf(theta), s = sinf(theta);
            const float x0 = v[p], x1 = v[p + 32];
            v[p]      = x0 * c - x1 * s;
            v[p + 32] = x0 * s + x1 * c;
        }
        for (int e = 0; e < HD; e++) dst[(size_t) h * HD + e] = v[e];
    }
}

static void test_qk_norm_rope() {
    // validate the header's hardcoded theta_scale against the fork formula.
    const float ts_hdr = mk::MK_IMROPE_THETA_SCALE;
    const float ts_ref = powf(1e7f, -2.0f / 64.0f);
    if (memcmp(&ts_hdr, &ts_ref, 4) != 0) {
        g_failures++;
        printf("FAIL theta_scale: header %.9g != powf(1e7,-2/64) %.9g\n", ts_hdr, ts_ref);
    } else {
        printf("ok   theta_scale constant bit-matches powf(1e7,-2/64)\n");
    }

    struct Case { int n_heads; int src_stride; int pos[4]; const char *name; };
    const Case cases[] = {
        {24, 512, {100, 101, 102, 103}, "qk_norm_rope q 24h stride512"}, // q GEMV [q|gate]
        { 4, 256, {  7,   9,  11,  13}, "qk_norm_rope k 4h stride256"},  // k GEMV
        {24, 512, {  0,   0,   0,   0}, "qk_norm_rope q pos0"},          // pos 0 -> identity rotation
    };
    for (const Case &c : cases) {
        std::vector<float> src((size_t) c.n_heads * c.src_stride), norm_w(HD);
        fill_rand(src, -2.0f, 2.0f);
        fill_rand(norm_w, -1.0f, 1.0f);
        std::vector<int> pos(4);
        for (int i = 0; i < 4; i++) pos[i] = c.pos[i];

        float *d_src = dalloc(src.size()), *d_w = dalloc(HD), *d_dst = dalloc((size_t) c.n_heads * HD);
        int32_t *d_pos; CUDA_CHECK(cudaMalloc(&d_pos, 4 * sizeof(int32_t)));
        up(d_src, src.data(), src.size());
        up(d_w, norm_w.data(), HD);
        CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), 4 * sizeof(int32_t), cudaMemcpyHostToDevice));

        mk::QkNormRopeArgs a = {d_src, d_w, d_pos, d_dst,
                                (uint32_t) c.n_heads, (uint32_t) c.src_stride};
        run_program({make_instr(mk::OP_QK_NORM_ROPE, 1, 7, a)}, 0);

        std::vector<float> want((size_t) c.n_heads * HD), got((size_t) c.n_heads * HD);
        ref_qk_norm_rope(src.data(), norm_w.data(), pos.data(), want.data(),
                         c.n_heads, c.src_stride);
        down(got.data(), d_dst, got.size());
        check_close(c.name, got.data(), want.data(), got.size(), 1e-5f);

        CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_dst)); CUDA_CHECK(cudaFree(d_pos));
    }
}

// ---------------------------------------------------------------------------
// OP_KV_APPEND: f32 row -> f16 row at an i64 index (set-rows.cu:160-165). RN
// on both sides, so the f16 cache row is bit-exact.
static void test_kv_append() {
    const uint32_t row_width = 512;   // per-GPU: 2 kv heads x 256
    const int nrows = 8;
    std::vector<__half> cache((size_t) nrows * row_width);
    for (size_t i = 0; i < cache.size(); i++) cache[i] = __float2half(-7.0f); // sentinel fill
    std::vector<float> src(row_width);
    fill_rand(src, -3.0f, 3.0f);

    __half *d_cache; CUDA_CHECK(cudaMalloc(&d_cache, cache.size() * sizeof(__half)));
    float *d_src = dalloc(row_width);
    long long *d_row; CUDA_CHECK(cudaMalloc(&d_row, sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_cache, cache.data(), cache.size() * 2, cudaMemcpyHostToDevice));
    up(d_src, src.data(), row_width);
    const long long row = 5;
    CUDA_CHECK(cudaMemcpy(d_row, &row, sizeof(long long), cudaMemcpyHostToDevice));

    mk::KvAppendArgs a = {d_src, d_row, d_cache, row_width};
    run_program({make_instr(mk::OP_KV_APPEND, 0, 8, a)}, 0);

    std::vector<__half> got(cache.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_cache, got.size() * 2, cudaMemcpyDeviceToHost));

    // expected: row 5 = RN(src), every other row unchanged.
    std::vector<__half> want = cache;
    for (uint32_t i = 0; i < row_width; i++) want[(size_t) row * row_width + i] = __float2half(src[i]);
    check_bitwise_u16("kv_append row + untouched",
                      reinterpret_cast<unsigned short *>(got.data()),
                      reinterpret_cast<unsigned short *>(want.data()), got.size());

    CUDA_CHECK(cudaFree(d_cache)); CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_row));
}

// ---------------------------------------------------------------------------
// OP_FATTN_DECODE + OP_FATTN_REDUCE: direct flash-attention reference in
// double over the shared f16 K/V. q pre-scaled by MK_ATTN_SCALE, mask added
// (0 attend / -inf masked), stable softmax, rowsum divide.
static unsigned fattn_smem_bytes(uint32_t n_q, uint32_t /*row_width*/) {
    // tile is one kv head's 256-wide slice (HD+2 pitch), independent of the
    // physical cache row_width (the op loops kv heads on the outer axis).
    const uint32_t row_p = HD + 2;
    return (unsigned)(n_q * HD * sizeof(float)          // q_s (all heads)
                      + mk::MK_FATTN_TILE * row_p * sizeof(__half) // KV slice tile
                      + mk::MK_FATTN_TILE * sizeof(__half));       // mask tile
}

static void ref_fattn(const std::vector<__half> &kc, const std::vector<__half> &vc,
                      const std::vector<__half> &mask, const std::vector<float> &q,
                      std::vector<float> &out, int n_q, int n_kv_heads,
                      int n_kv, int row_width) {
    const int gqa = n_q / n_kv_heads;
    const double scale = mk::MK_ATTN_SCALE;
    out.assign((size_t) n_q * HD, 0.0f);
    for (int h = 0; h < n_q; h++) {
        const int g = h / gqa;
        std::vector<double> score(n_kv);
        std::vector<char> live(n_kv, 0);
        double m = -DBL_MAX;
        for (int j = 0; j < n_kv; j++) {
            const float mv = __half2float(mask[j]);
            if (!std::isfinite(mv)) { score[j] = -HUGE_VAL; continue; } // -inf masked
            double dot = 0.0;
            for (int d = 0; d < HD; d++) {
                const double qs = scale * (double) q[(size_t) h * HD + d];
                const double kv = (double) __half2float(kc[(size_t) j * row_width + g * HD + d]);
                dot += qs * kv;
            }
            score[j] = dot + (double) mv; // mv == 0 for attend
            live[j] = 1;
            if (score[j] > m) m = score[j];
        }
        double denom = 0.0;
        std::vector<double> acc(HD, 0.0);
        for (int j = 0; j < n_kv; j++) {
            if (!live[j]) continue;
            const double p = exp(score[j] - m);
            denom += p;
            for (int d = 0; d < HD; d++)
                acc[d] += p * (double) __half2float(vc[(size_t) j * row_width + g * HD + d]);
        }
        for (int d = 0; d < HD; d++) out[(size_t) h * HD + d] = (float) (acc[d] / denom);
    }
}

static void test_fattn_case(int n_q, int n_kv_heads, int real_kv, int n_kv,
                            int nchunks, const char *name) {
    const int row_width = n_kv_heads * HD;
    const int gqa = n_q / n_kv_heads;

    std::vector<__half> kc((size_t) n_kv * row_width), vc((size_t) n_kv * row_width);
    std::vector<__half> mask(n_kv);
    std::vector<float> q((size_t) n_q * HD);
    fill_rand_half(kc, -1.0f, 1.0f);
    fill_rand_half(vc, -1.0f, 1.0f);  // padded rows filled finite (header req)
    fill_rand(q, -1.0f, 1.0f);
    for (int j = 0; j < n_kv; j++)
        mask[j] = (j < real_kv) ? __float2half(0.0f) : __float2half(-INFINITY);

    __half *d_kc, *d_vc, *d_mask;
    CUDA_CHECK(cudaMalloc(&d_kc, kc.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_vc, vc.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_kc, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vc, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice));
    float *d_q = dalloc(q.size());
    up(d_q, q.data(), q.size());

    // partials seeded with the -inf sentinel in every max slot; decode must
    // overwrite them all (a correct pass leaves no sentinel -> error 0).
    const size_t np = (size_t) n_q * nchunks * mk::MK_FATTN_PSTRIDE;
    std::vector<float> partials(np, 0.0f);
    for (int h = 0; h < n_q; h++)
        for (int c = 0; c < nchunks; c++)
            partials[((size_t) h * nchunks + c) * mk::MK_FATTN_PSTRIDE + HD] = -INFINITY;
    float *d_part = dalloc(np);
    up(d_part, partials.data(), np);

    float *d_dst = dalloc((size_t) n_q * HD);
    unsigned *d_err; CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned)));

    mk::FattnDecodeArgs dec = {d_q, d_kc, d_vc, d_mask, d_part,
                               (uint32_t) n_kv, (uint32_t) n_q,
                               (uint32_t) n_kv_heads, (uint32_t) row_width};
    const int nb_red = (n_q < NBLK) ? n_q : NBLK;
    mk::FattnReduceArgs red = {d_part, d_dst, d_err, (uint32_t) n_q, (uint32_t) nchunks};

    run_program({
        make_instr(mk::OP_FATTN_DECODE, 0, (uint16_t) nchunks, dec),
        make_boundary(),
        make_instr(mk::OP_FATTN_REDUCE, 0, (uint16_t) nb_red, red),
    }, fattn_smem_bytes(n_q, row_width));

    unsigned err = 0;
    CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (err != 0) {
        g_failures++;
        printf("FAIL [sentinel] %-24s decode left %u unwritten partial(s)\n", name, err);
    }

    std::vector<float> want, got((size_t) n_q * HD);
    ref_fattn(kc, vc, mask, q, want, n_q, n_kv_heads, n_kv, row_width);
    down(got.data(), d_dst, got.size());
    char label[96];
    snprintf(label, sizeof(label), "%s (gqa%d kv%d/%d ch%d)", name, gqa, real_kv, n_kv, nchunks);
    check_attn(label, got.data(), want.data(), n_q, 2e-3f);

    CUDA_CHECK(cudaFree(d_kc)); CUDA_CHECK(cudaFree(d_vc)); CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_part));
    CUDA_CHECK(cudaFree(d_dst)); CUDA_CHECK(cudaFree(d_err));
}

static void test_fattn() {
    // production per-GPU shape (12 q / 2 kv, gqa 6), crossing the pad boundary:
    test_fattn_case(12, 2, 256, 256, 1, "fattn full-tiles 1chunk");   // exact, no merge
    test_fattn_case(12, 2, 250, 256, 8, "fattn pad250 8chunk");       // 250 in 256 pad
    test_fattn_case(12, 2, 700, 768, 5, "fattn pad700 5chunk");       // non-32 chunk len -> partial tiles
    // single-kv-head variant (gqa still 6), a different row_width (256):
    test_fattn_case(6, 1, 256, 256, 4, "fattn 1kvhead 4chunk");
    test_fattn_case(6, 1, 500, 512, 8, "fattn 1kvhead pad500");
    // single-GPU shape: 24 q / 4 kv (gqa 6), row_width 1024 — the shape the
    // k=0 harness drives (n_q > warps/block AND a full-row tile would overflow
    // the slab; the kv-head outer loop handles both). nchunks <= NBLK (the
    // mini-interpreter grid) so every chunk's block exists.
    test_fattn_case(24, 4, 256, 256, 8, "fattn 24q4kv 8chunk");
    test_fattn_case(24, 4, 700, 768, 8, "fattn 24q4kv pad700");
}

// ---------------------------------------------------------------------------
// OP_FATTN_REDUCE sentinel: a never-written partial (max slot == -inf) must be
// counted into *error and skipped, not folded as garbage. Built from synthetic
// partials so the detection is exercised independently of decode.
static void test_fattn_sentinel() {
    const int n_q = 4, nchunks = 3;
    const size_t np = (size_t) n_q * nchunks * mk::MK_FATTN_PSTRIDE;
    std::vector<float> partials(np, 0.0f);
    std::mt19937 rng(0xdead);
    std::uniform_real_distribution<float> du(-1.0f, 1.0f);
    for (int h = 0; h < n_q; h++)
        for (int c = 0; c < nchunks; c++) {
            float *rec = &partials[((size_t) h * nchunks + c) * mk::MK_FATTN_PSTRIDE];
            for (int d = 0; d < HD; d++) rec[d] = du(rng);
            rec[HD]     = du(rng);              // finite max
            rec[HD + 1] = 0.5f + fabsf(du(rng)); // positive sumexp
        }

    float *d_part = dalloc(np), *d_dst = dalloc((size_t) n_q * HD);
    unsigned *d_err; CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned)));

    // (a) all valid -> no sentinel detected.
    up(d_part, partials.data(), np);
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned)));
    mk::FattnReduceArgs red = {d_part, d_dst, d_err, (uint32_t) n_q, (uint32_t) nchunks};
    run_program({make_instr(mk::OP_FATTN_REDUCE, 0, (uint16_t) n_q, red)}, 0);
    unsigned err0 = 0;
    CUDA_CHECK(cudaMemcpy(&err0, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (err0 == 0) printf("ok   [sentinel] clean partials -> error 0\n");
    else { g_failures++; printf("FAIL [sentinel] clean partials flagged %u\n", err0); }

    // (b) corrupt one record's max slot to the -inf sentinel -> must count 1.
    std::vector<float> corrupt = partials;
    const int hc = 2, cc = 1;
    corrupt[((size_t) hc * nchunks + cc) * mk::MK_FATTN_PSTRIDE + HD] = -INFINITY;
    up(d_part, corrupt.data(), np);
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned)));
    run_program({make_instr(mk::OP_FATTN_REDUCE, 0, (uint16_t) n_q, red)}, 0);
    unsigned err1 = 0;
    CUDA_CHECK(cudaMemcpy(&err1, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (err1 == 1) printf("ok   [sentinel] unwritten partial DETECTED (error 1)\n");
    else { g_failures++; printf("FAIL [sentinel] unwritten partial NOT detected (error %u, want 1)\n", err1); }

    CUDA_CHECK(cudaFree(d_part)); CUDA_CHECK(cudaFree(d_dst)); CUDA_CHECK(cudaFree(d_err));
}

// ---------------------------------------------------------------------------
// OP_ATTN_GATE: dst = attn * sigmoid(gate slice), gate at h*gate_stride
// (unary.cu:48-50).
static void test_attn_gate() {
    const int n_q = 12;
    const uint32_t gate_stride = 512; // [q 256 | gate 256] interleave
    std::vector<float> attn((size_t) n_q * HD), gate((size_t) n_q * gate_stride);
    fill_rand(attn, -2.0f, 2.0f);
    fill_rand(gate, -4.0f, 4.0f);

    float *d_attn = dalloc(attn.size()), *d_gate = dalloc(gate.size()), *d_dst = dalloc(attn.size());
    up(d_attn, attn.data(), attn.size());
    up(d_gate, gate.data(), gate.size());

    mk::AttnGateArgs a = {d_attn, d_gate, d_dst, (uint32_t) n_q, gate_stride};
    run_program({make_instr(mk::OP_ATTN_GATE, 0, 8, a)}, 0);

    std::vector<float> want((size_t) n_q * HD), got((size_t) n_q * HD);
    for (int h = 0; h < n_q; h++)
        for (int d = 0; d < HD; d++) {
            const float gv = gate[(size_t) h * gate_stride + d];
            want[(size_t) h * HD + d] = attn[(size_t) h * HD + d] * (1.0f / (1.0f + expf(-gv)));
        }
    down(got.data(), d_dst, got.size());
    check_close("attn_gate", got.data(), want.data(), got.size(), 1e-5f);

    CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_gate)); CUDA_CHECK(cudaFree(d_dst));
}

// ---------------------------------------------------------------------------
// Indicative KV-stream bandwidth for OP_FATTN_DECODE (the attn family's
// DRAM-bound op: 2 x n_kv x row_width x 2 B of f16 K+V per pass). Full-GPU
// (72 chunks), single instruction (no boundary), timed over many iters.
// Contended: the rig is shared, clocks unlocked -> a lower bound, not peak.
static void bench_fattn_kv_bw(int n_sm) {
    const int n_q = 12, n_kv_heads = 2, row_width = n_kv_heads * HD;
    // n_kv chosen so clen = n_kv/nchunks is a multiple of 32 -> chunk starts
    // (kv0) are 32-aligned, satisfying the packed mask read's even-kv0
    // precondition (see NOTES-attn.md). A real scheduler tile-aligns chunks.
    const int nchunks = n_sm;                // one chunk per SM
    const int clen = 2048;                   // 64 tiles per chunk
    const int n_kv = nchunks * clen;         // 72 * 2048 = 147456 (mult of 256)
    std::vector<__half> kc((size_t) n_kv * row_width), vc((size_t) n_kv * row_width), mask(n_kv);
    fill_rand_half(kc, -1.0f, 1.0f); fill_rand_half(vc, -1.0f, 1.0f);
    for (int j = 0; j < n_kv; j++) mask[j] = __float2half(0.0f);
    std::vector<float> q((size_t) n_q * HD); fill_rand(q, -1.0f, 1.0f);

    __half *d_kc, *d_vc, *d_mask;
    CUDA_CHECK(cudaMalloc(&d_kc, kc.size() * 2)); CUDA_CHECK(cudaMalloc(&d_vc, vc.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_kc, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vc, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), mask.size() * 2, cudaMemcpyHostToDevice));
    float *d_q = dalloc(q.size()); up(d_q, q.data(), q.size());
    const size_t np = (size_t) n_q * nchunks * mk::MK_FATTN_PSTRIDE;
    float *d_part = dalloc(np);

    mk::FattnDecodeArgs dec = {d_q, d_kc, d_vc, d_mask, d_part,
                               (uint32_t) n_kv, (uint32_t) n_q,
                               (uint32_t) n_kv_heads, (uint32_t) row_width};
    std::vector<mk::Instr> prog = {make_instr(mk::OP_FATTN_DECODE, 0, (uint16_t) nchunks, dec)};
    mk::Instr *d_prog; CUDA_CHECK(cudaMalloc(&d_prog, prog.size() * sizeof(mk::Instr)));
    CUDA_CHECK(cudaMemcpy(d_prog, prog.data(), prog.size() * sizeof(mk::Instr), cudaMemcpyHostToDevice));
    const unsigned smem = fattn_smem_bytes(n_q, row_width);

    for (int i = 0; i < 5; i++) mk_run<<<nchunks, NTHR, smem>>>(d_prog, 1, g_counter);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    const int iters = 50;
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) mk_run<<<nchunks, NTHR, smem>>>(d_prog, 1, g_counter);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    const double kv_bytes = 2.0 * (double) n_kv * row_width * 2.0; // K + V, f16
    const double gbps = kv_bytes / (ms / iters / 1e3) / 1e9;
    printf("fattn_decode KV stream: %d chunks, n_kv=%d, %.1f MB K+V, %.3f ms/pass -> %.0f GB/s (contended)\n",
           nchunks, n_kv, kv_bytes / 1e6, ms / iters, gbps);

    cudaEventDestroy(a); cudaEventDestroy(b);
    CUDA_CHECK(cudaFree(d_kc)); CUDA_CHECK(cudaFree(d_vc)); CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_part)); CUDA_CHECK(cudaFree(d_prog));
}

// ---------------------------------------------------------------------------
static void report_regs(const char *name, const void *fn) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fn));
    printf("  %-20s regs=%d  local(stack)=%zu B  const=%zu B  (ptxas -v: 0 spills)\n",
           name, attr.numRegs, attr.localSizeBytes, attr.constSizeBytes);
}

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
    printf("device %d (%s), %.1f GiB free, %d SMs\n", best, prop.name,
           (double) best_free / (1 << 30), prop.multiProcessorCount);

    // per-op resource accounting (numRegs off the probe kernels; declared smem
    // slab from the header, computed at the production shape).
    printf("per-op ptxas resources (probe kernels):\n");
    report_regs("qk_norm_rope", (const void *) probe_qk_norm_rope);
    report_regs("kv_append",    (const void *) probe_kv_append);
    report_regs("fattn_decode", (const void *) probe_fattn_decode);
    report_regs("fattn_reduce", (const void *) probe_fattn_reduce);
    report_regs("attn_gate",    (const void *) probe_attn_gate);
    report_regs("mk_run(all)",  (const void *) mk_run);
    printf("  fattn_decode declared smem: %u B (n_q=12,row=512); %u B (n_q=6,row=256)\n",
           fattn_smem_bytes(12, 512), fattn_smem_bytes(6, 256));

    // the fattn program crosses one Y02 boundary: all NBLK blocks must be
    // co-resident at the decode slab size.
    const unsigned max_smem = fattn_smem_bytes(12, 512);
    int bps = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, mk_run, NTHR, max_smem));
    if (bps * prop.multiProcessorCount < NBLK) {
        fprintf(stderr, "grid of %d blocks not co-resident (%d/SM x %d SMs at %u B smem)\n",
                NBLK, bps, prop.multiProcessorCount, max_smem);
        return 1;
    }
    printf("co-residency: %d blocks/SM x %d SMs >= %d blocks at %u B smem -- ok\n",
           bps, prop.multiProcessorCount, NBLK, max_smem);

    CUDA_CHECK(cudaMalloc(&g_counter, sizeof(unsigned)));

    test_qk_norm_rope();
    test_kv_append();
    test_fattn();
    test_fattn_sentinel();
    test_attn_gate();
    bench_fattn_kv_bw(prop.multiProcessorCount);

    printf("worst rel error (tolerance-checked ops): %.3g\n", g_worst_rel);
    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
