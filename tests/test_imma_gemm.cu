// op-a U=128 projection -- SPIKE: int8 IMMA GEMM (weight Q4_0/AR16 x q8_1 activation)
// via the Turing m8n8k16 s32.s8.s8.s32 tensor MMA -- the ONLY int8 tensor MMA on
// sm_75, which the megakernel has NEVER used. This de-risks that primitive in
// isolation BEFORE any op_gemv restructure: it validates the fragment lane-map, the
// AR16 nibble->s8 unpack into the A-fragment, and the per-block fp32 scale fold,
// against a scalar dp4a-equivalent reference that IS the decode mmvq row dot.
//
// The projection op multiplies a Q4_0/AR16 weight matrix (M rows) by a batch of U
// activation columns (here U=N up to 128), contracting the feature dim K. Decode
// runs this as the scalar vec_dot_q4_0_ar16_q8_1 (mmvq); this spike runs the SAME
// arithmetic through the int8 tensor core, one AR16 block (16 elements) per MMA.
//
// Fork math grounded in software/llama.cpp/autoround:
//   * block_q4_0_ar16 (ggml-common.h:194): 16 elems/block, ggml_half d = absmax/8,
//     uint8_t qs[8], byte j = code[2j] | code[2j+1]<<4; value = (code-8)*d.
//   * unpack_q4_0_ar16 (vecdotq.cuh:143): word (4 bytes=8 nibbles=8 elems) ->
//     lo=(q&0x0F0F0F0F), hi=((q>>4)&0x0F0F0F0F); v.x=vsubss4(byte_perm(lo,hi,0x5140),
//     0x08080808)=elems0..3, v.y=byte_perm(...,0x7362)=elems4..7. word0(qs0..3)=elems
//     0..7, word1(qs4..7)=elems8..15. Signed s8 in element order, (code-8) applied.
//   * block_q8_1 (ggml-common.h:258): 32 elems, d=amax/127, q=round(x/d) int8.
//     One q8_1 block spans TWO AR16 blocks (32 = 2*16).
//   * vec_dot_q4_0_ar16_q8_1 (vecdotq.cuh:755): sumi=sum_k (code-8)*q8 (dp4a);
//     out = w_d * a_d * sumi. SYMMETRIC: offset in the weight, NO ds8.y correction.
//
// m8n8k16 .row.col fragment map (confirmed against fork mma.cuh:920-940 + the
// tile<> get_i/get_j at :239-271). lane l: gid = l>>2 (0..7), tid = l&3 (0..3):
//   A[8x16] s8 (row-maj): a0 = 4 s8 = W[row=gid][k = tid*4+0..3], byte i -> k=tid*4+i
//   B[16x8] s8 (col-maj): b0 = 4 s8 = X[col=gid][k = tid*4+0..3], byte i -> k=tid*4+i
//   D[8x8]  s32:          d0 = C[gid][tid*2+0], d1 = C[gid][tid*2+1]
// One AR16 block == one m8n8k16 K-step (K=16). The warp cooperatively covers an
// 8-row x 8-col output tile per (n-tile) from all 32 lanes.
//
// Correctness gate (BINDING = integer path, bit-exact): the per-block s32 IMMA
// partials must EQUAL the scalar sum_k (code-8)*q8 for every (block,row,col). The
// folded fp32 output is checked to a tight tol (only FMA-order differences remain).
// The scalar reference reads the same quantized codes/quants/scales the device sees
// but never touches the AR16 interleave or the tensor fragments -- so a device
// unpack or lane-map bug cannot hide behind a matching-wrong reference.
//
// NOT RUN here (GPU lease busy): compiled cold only. The clock64 micro-bench at
// N=128 is advisory (calx-mill REFUSED-until-anchor).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                        \
    do { cudaError_t e = (call);                                               \
        if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d: %s\n", #call,\
            __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int M = 8;   // one weight row-tile == m8 (fixed by the MMA shape)

// AR16 weight block: ggml_half d + 8 interleaved nibble bytes (16 elements).
struct WB { __half d; uint8_t qs[8]; };

// ---- device: AR16 nibble -> s8 unpack, mirrors vecdotq.cuh:143 ----
__device__ __forceinline__ int rd_word(const uint8_t* p) { // byte-safe (get_int_b2)
    return (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
}
__device__ __forceinline__ int2 unpack_ar16(int q) {
    const int lo = (q >> 0) & 0x0F0F0F0F;   // even elements in bytes 0..3
    const int hi = (q >> 4) & 0x0F0F0F0F;   // odd  elements in bytes 0..3
    int2 v;
    v.x = __vsubss4(__byte_perm(lo, hi, 0x5140), 0x08080808); // elems 0..3
    v.y = __vsubss4(__byte_perm(lo, hi, 0x7362), 0x08080808); // elems 4..7
    return v;
}

// ---- device: one Turing m8n8k16 s32.s8.s8.s32 (accumulate into d0,d1) ----
__device__ __forceinline__ void imma_m8n8k16(int& d0, int& d1, unsigned a0, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(d0), "+r"(d1) : "r"(a0), "r"(b0));
}

// GEMM: C[M][N] = sum_K W_s8 * X_s8, folded per AR16 block by (w_d * a_d).
// W: [M][KB] AR16 blocks. Xq: [N][K] int8 (q8_1 quants). Xd: [N][K/32] q8_1 scales.
// partials: [KB][M][N] s32 -- the raw per-block integer dot, for the bit-exact gate.
__global__ void imma_gemm_kernel(const WB* __restrict__ W, const int8_t* __restrict__ Xq,
                                 const __half* __restrict__ Xd, int N, int K,
                                 float* __restrict__ C, int* __restrict__ partials) {
    const int lane = threadIdx.x;
    const int gid = lane >> 2, tid = lane & 3;   // gid: row/col index, tid: K-quad
    const int KB = K / 16;                        // AR16 blocks (K-steps)
    const int NQ = K / 32;                        // q8_1 blocks per column
    for (int nt = 0; nt < N / 8; nt++) {
        const int col0 = nt * 8;
        const int nA = col0 + gid;                // activation column this lane loads
        const int n0 = col0 + tid * 2 + 0;        // output col of d0
        const int n1 = col0 + tid * 2 + 1;        // output col of d1
        float acc0 = 0.f, acc1 = 0.f;
        for (int b = 0; b < KB; b++) {
            // A fragment: weight row = gid, block b. Unpack, pick this lane's 4 elems.
            const WB* bx = &W[gid * KB + b];
            const int2 v0 = unpack_ar16(rd_word(bx->qs));       // elems 0..7
            const int2 v1 = unpack_ar16(rd_word(bx->qs + 4));   // elems 8..15
            const int aelem[4] = { v0.x, v0.y, v1.x, v1.y };    // tid -> elems tid*4..+3
            const unsigned a0 = (unsigned) aelem[tid];
            // B fragment: activation col = nA, elements b*16 + tid*4 + {0..3}.
            const int base = nA * K + b * 16 + tid * 4;
            const unsigned b0 = ((unsigned) (uint8_t) Xq[base + 0])       |
                                ((unsigned) (uint8_t) Xq[base + 1] << 8)  |
                                ((unsigned) (uint8_t) Xq[base + 2] << 16) |
                                ((unsigned) (uint8_t) Xq[base + 3] << 24);
            int d0 = 0, d1 = 0;
            imma_m8n8k16(d0, d1, a0, b0);
            // raw integer partials -> bit-exact gate
            partials[(b * M + gid) * N + n0] = d0;
            partials[(b * M + gid) * N + n1] = d1;
            // per-block fp32 scale fold: partial * (w_d * a_d), summed in fp32.
            const float wd  = __half2float(bx->d);
            const int   qb  = b >> 1;                            // 2 AR16 -> 1 q8_1
            const float ad0 = __half2float(Xd[n0 * NQ + qb]);
            const float ad1 = __half2float(Xd[n1 * NQ + qb]);
            acc0 += (float) d0 * wd * ad0;
            acc1 += (float) d1 * wd * ad1;
        }
        C[gid * N + n0] = acc0;
        C[gid * N + n1] = acc1;
    }
}

// advisory clock64 micro-bench: full N x K GEMM, ITERS times, cycles recorded.
__global__ void imma_bench_kernel(const WB* __restrict__ W, const int8_t* __restrict__ Xq,
                                  const __half* __restrict__ Xd, int N, int K,
                                  int iters, float* __restrict__ sink, long long* cyc) {
    const int lane = threadIdx.x;
    const int gid = lane >> 2, tid = lane & 3;
    const int KB = K / 16, NQ = K / 32;
    float acc = 0.f;
    const long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
        for (int nt = 0; nt < N / 8; nt++) {
            const int col0 = nt * 8, nA = col0 + gid;
            const int n0 = col0 + tid * 2, n1 = n0 + 1;
            float a0f = 0.f, a1f = 0.f;
            for (int b = 0; b < KB; b++) {
                const WB* bx = &W[gid * KB + b];
                const int2 v0 = unpack_ar16(rd_word(bx->qs));
                const int2 v1 = unpack_ar16(rd_word(bx->qs + 4));
                const int aelem[4] = { v0.x, v0.y, v1.x, v1.y };
                const unsigned a0 = (unsigned) aelem[tid];
                const int base = nA * K + b * 16 + tid * 4;
                const unsigned b0 = ((unsigned) (uint8_t) Xq[base + 0])       |
                                    ((unsigned) (uint8_t) Xq[base + 1] << 8)  |
                                    ((unsigned) (uint8_t) Xq[base + 2] << 16) |
                                    ((unsigned) (uint8_t) Xq[base + 3] << 24);
                int d0 = 0, d1 = 0;
                imma_m8n8k16(d0, d1, a0, b0);
                const float wd = __half2float(bx->d);
                const int qb = b >> 1;
                a0f += (float) d0 * wd * __half2float(Xd[n0 * NQ + qb]);
                a1f += (float) d1 * wd * __half2float(Xd[n1 * NQ + qb]);
            }
            acc += a0f + a1f;
        }
    }
    const long long t1 = clock64();
    if (lane == 0) { *cyc = t1 - t0; }
    sink[lane] = acc;   // keep the work live
}

// ---- host: obviously-correct scalar quantization + reference ----
static void quantize_weight(const std::vector<float>& x, int K,     // x: [M][K]
                            std::vector<WB>& W, std::vector<int>& code) {
    const int KB = K / 16;
    W.assign(M * KB, WB{});
    code.assign(M * K, 0);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < KB; b++) {
            float absmax = 0.f;
            for (int e = 0; e < 16; e++) absmax = fmaxf(absmax, fabsf(x[m * K + b * 16 + e]));
            const float d = absmax / 8.f;
            int c[16];
            for (int e = 0; e < 16; e++) {
                int q = (d > 0.f) ? (int) lroundf(x[m * K + b * 16 + e] / d) + 8 : 8;
                q = q < 0 ? 0 : (q > 15 ? 15 : q);
                c[e] = q;
                code[m * K + b * 16 + e] = q;
            }
            WB& wb = W[m * KB + b];
            wb.d = __float2half(d);
            for (int j = 0; j < 8; j++) wb.qs[j] = (uint8_t) (c[2 * j] | (c[2 * j + 1] << 4));
        }
}

static void quantize_act(const std::vector<float>& x, int N, int K,  // x: [N][K]
                         std::vector<int8_t>& q, std::vector<__half>& d) {
    const int NQ = K / 32;
    q.assign((size_t) N * K, 0);
    d.assign((size_t) N * NQ, __float2half(0.f));
    for (int n = 0; n < N; n++)
        for (int qb = 0; qb < NQ; qb++) {
            float amax = 0.f;
            for (int e = 0; e < 32; e++) amax = fmaxf(amax, fabsf(x[(size_t) n * K + qb * 32 + e]));
            const float dd = amax / 127.f;
            d[(size_t) n * NQ + qb] = __float2half(dd);
            for (int e = 0; e < 32; e++) {
                int v = (dd > 0.f) ? (int) lroundf(x[(size_t) n * K + qb * 32 + e] / dd) : 0;
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                q[(size_t) n * K + qb * 32 + e] = (int8_t) v;
            }
        }
}

// returns 0 on pass, 1 on fail; exercises one (N,K) shape end-to-end.
static int run_shape(int N, int K, unsigned seed, const char* tag) {
    const int KB = K / 16, NQ = K / 32;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> wf(M * K), xf((size_t) N * K);
    for (float& v : wf) v = dist(rng);
    for (float& v : xf) v = dist(rng);

    std::vector<WB> W; std::vector<int> code;
    std::vector<int8_t> Xq; std::vector<__half> Xd;
    quantize_weight(wf, K, W, code);
    quantize_act(xf, N, K, Xq, Xd);

    // scalar reference: integer partials (bit-exact) + folded fp32, using the SAME
    // quantized codes/quants/scales the device reads (via __half2float on the halves).
    std::vector<int>   part_ref((size_t) KB * M * N);
    std::vector<float> C_ref((size_t) M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int b = 0; b < KB; b++) {
                int sumi = 0;
                for (int e = 0; e < 16; e++) {
                    const int w = code[m * K + b * 16 + e] - 8;          // (code-8)
                    const int a = (int) Xq[(size_t) n * K + b * 16 + e]; // q8
                    sumi += w * a;
                }
                part_ref[((size_t) b * M + m) * N + n] = sumi;
                const float wd = __half2float(W[m * KB + b].d);
                const float ad = __half2float(Xd[(size_t) n * NQ + (b >> 1)]);
                acc += (float) sumi * wd * ad;
            }
            C_ref[(size_t) m * N + n] = acc;
        }

    // device run
    WB* dW; int8_t* dXq; __half* dXd; float* dC; int* dPart;
    CUDA_CHECK(cudaMalloc(&dW,  W.size() * sizeof(WB)));
    CUDA_CHECK(cudaMalloc(&dXq, Xq.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dXd, Xd.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,  (size_t) M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPart, (size_t) KB * M * N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dW,  W.data(),  W.size() * sizeof(WB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXq, Xq.data(), Xq.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXd, Xd.data(), Xd.size() * sizeof(__half), cudaMemcpyHostToDevice));
    imma_gemm_kernel<<<1, 32>>>(dW, dXq, dXd, N, K, dC, dPart);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> C_got((size_t) M * N);
    std::vector<int>   part_got((size_t) KB * M * N);
    CUDA_CHECK(cudaMemcpy(C_got.data(), dC, C_got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(part_got.data(), dPart, part_got.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // GATE 1 (binding): every integer partial bit-exact.
    long long int_bad = 0; int wbk = -1, wbm = -1, wbn = -1, wref = 0, wgot = 0;
    for (size_t i = 0; i < part_ref.size(); i++)
        if (part_ref[i] != part_got[i]) {
            if (int_bad == 0) {
                wbk = (int) (i / ((size_t) M * N));
                wbm = (int) ((i / N) % M);
                wbn = (int) (i % N);
                wref = part_ref[i]; wgot = part_got[i];
            }
            int_bad++;
        }
    // GATE 2: folded fp32 within tight tol (FMA-order only).
    float mag = 1e-6f; for (float v : C_ref) mag = fmaxf(mag, fabsf(v));
    const float tol = 1e-3f * mag;
    int fp_bad = 0; float worst = 0.f; int wm = -1, wn = -1;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            const float diff = fabsf(C_got[(size_t) m * N + n] - C_ref[(size_t) m * N + n]);
            if (diff > worst) { worst = diff; wm = m; wn = n; }
            if (diff > tol) fp_bad++;
        }

    cudaFree(dW); cudaFree(dXq); cudaFree(dXd); cudaFree(dC); cudaFree(dPart);

    printf("[%s] M=%d N=%d K=%d  KB=%d NQ=%d  int-mag(fp) %.4g tol %.3g\n",
           tag, M, N, K, KB, NQ, mag, tol);
    if (int_bad) {
        printf("  FAIL(int): %lld/%zu partials differ; first block=%d m=%d n=%d got %d want %d\n",
               int_bad, part_ref.size(), wbk, wbm, wbn, wgot, wref);
        return 1;
    }
    printf("  ok(int): all %zu s32 partials bit-exact\n", part_ref.size());
    if (fp_bad) {
        printf("  FAIL(fp): %d/%d beyond tol; worst |diff| %.5g at m=%d n=%d (got %.6g want %.6g)\n",
               fp_bad, M * N, worst, wm, wn,
               C_got[(size_t) wm * N + wn], C_ref[(size_t) wm * N + wn]);
        return 1;
    }
    printf("  ok(fp): all %d outputs within tol; worst |diff| %.5g (%.3g of mag)\n",
           M * N, worst, worst / mag);
    return 0;
}

// advisory clock64 micro-bench at N=128, K=256.
static void bench(int N, int K) {
    const int KB = K / 16, ITERS = 1000;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> wf(M * K), xf((size_t) N * K);
    for (float& v : wf) v = dist(rng);
    for (float& v : xf) v = dist(rng);
    std::vector<WB> W; std::vector<int> code;
    std::vector<int8_t> Xq; std::vector<__half> Xd;
    quantize_weight(wf, K, W, code);
    quantize_act(xf, N, K, Xq, Xd);

    WB* dW; int8_t* dXq; __half* dXd; float* dSink; long long* dCyc;
    CUDA_CHECK(cudaMalloc(&dW,  W.size() * sizeof(WB)));
    CUDA_CHECK(cudaMalloc(&dXq, Xq.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dXd, Xd.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dSink, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCyc, sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(dW,  W.data(),  W.size() * sizeof(WB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXq, Xq.data(), Xq.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dXd, Xd.data(), Xd.size() * sizeof(__half), cudaMemcpyHostToDevice));
    imma_bench_kernel<<<1, 32>>>(dW, dXq, dXd, N, K, ITERS, dSink, dCyc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    long long cyc = 0;
    CUDA_CHECK(cudaMemcpy(&cyc, dCyc, sizeof(long long), cudaMemcpyDeviceToHost));
    cudaFree(dW); cudaFree(dXq); cudaFree(dXd); cudaFree(dSink); cudaFree(dCyc);
    const long long mmas = (long long) ITERS * (N / 8) * KB;   // one m8n8k16 per (nt,b)
    printf("[bench] N=%d K=%d iters=%d: %lld cycles, %.2f cyc/m8n8k16 IMMA, "
           "%.2f cyc per full %dx%d tile (advisory)\n",
           N, K, ITERS, cyc, (double) cyc / mmas, (double) cyc / ((long long) ITERS * (N / 8)),
           M, 8);
}

int main() {
    int fail = 0;
    fail |= run_shape(128, 256, 1234, "U=128");   // op-a batched projection shape
    fail |= run_shape(8, 32, 4321, "min");        // 1 n-tile, 2 AR16 -> 1 shared q8_1
    fail |= run_shape(64, 128, 99, "mid");        // intermediate batch
    bench(128, 256);
    if (fail) { printf("SPIKE FAIL\n"); return 1; }
    printf("SPIKE OK (compiled+parity path; not the perf claim)\n");
    return 0;
}
