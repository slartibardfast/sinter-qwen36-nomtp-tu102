// plan/0143 fused-prefill -- SPIKE (tri-solve): FP32 forward-substitution
// triangular solve T = (I - tril(kb))^-1 on a [64x64] lower-unitriangular A.
// The DeltaNet UT/WY go/no-go: is a sequential in-kernel FP32 solve tractable on
// sm_75 BEFORE any chunk-scan integration?
//
// Fork math this spike is grounded in:
//   * llama.cpp/main ggml/src/ggml-cuda/solve_tri.cu:70-76 -- the solve MUST run in
//     full FP32: cuBLAS strsm is forced to CUBLAS_DEFAULT_MATH (TF32 OFF) with the
//     comment "Yes, this is necessary, without this we get RMSE errors"; TF32's
//     10-bit mantissa truncation breaks the substitution. So: NO tensor cores, NO
//     TF32 -- pure FP32 __fmul_rn / __fadd_rn here, by construction.
//   * the fast kernel solve_tri_f32_fast (same file, ~L137-168) is the reference
//     recurrence: x[row] = (b[row] - sum_{c<row} A[row][c]*x[c]) / A[row][row].
//     The fork parallelises the dot product across warp lanes and caps k<=32; the
//     full [64x64] inverse (k=64) is exactly the case the fork punts to cuBLAS.
//     This spike answers whether a direct in-kernel solve for the full inverse is
//     feasible.
//
// Map (plan/0143 capture/prefill-lever-derivation.md:110-116 -- the intra-chunk
// UT/WY FP32 solve, "chains=8 depth=64 throughput-bound" projection getting its
// first real number):
//   * ONE warp-group per triangular system (this launch: 1 block = 1 (chunk,head)).
//     n_chunks*H independent systems would map to more blocks; the derivation's
//     C = n_chunks*H >= L/R=8 hide threshold is realised here by the columns.
//   * The solve parallelises over the COLUMNS of B (= the 64 identity columns /
//     right-hand sides): each column is an INDEPENDENT forward-substitution chain,
//     64 chains >= the FFMA hide threshold L/R=8 -> throughput-bound.
//   * SEQUENTIAL over the 64 rows (the depth-64 recurrence): row r reads x[c<r].
//   * B = I(64), so X = A^-1 = T is the full [64x64] inverse ("invert column by
//     column").
//
// Each thread owns one column of B and reads only its OWN column's running x
// (sX[*][col]); columns never cross, so no per-row __syncthreads is needed (only
// one barrier after the cooperative A load).
//
// Gate: device fp32 T vs a double-precision CPU forward-substitution reference
// over the SAME fp32 A, abs tol 1e-5 (a kernel-mapping bug produces O(SCALE) diffs
// >> 1e-5; a correct fp32 solve on this well-conditioned A tracks the double ref to
// ~1e-7). clock64 cycles for the solve are printed -- the FFMA-pipe wall in
// isolation; ADVISORY only (calx-mill REFUSED-until-anchor, no prefill anchor yet).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(e));                          \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

constexpr int N = 64;  // triangular system size (DeltaNet chunk size CS=64)
constexpr int K = 64;  // columns of B = identity -> full inverse T = A^-1

// One block = one (chunk,head) triangular system; blockDim.x = K threads, thread
// `col` solves column `col` of T = A^-1 by FP32 forward substitution.
__global__ void solve_tri_kernel(const float * __restrict__ A,
                                 float * __restrict__ T,
                                 long long * __restrict__ cycles) {
    const int col = threadIdx.x;  // 0..K-1: one identity column / independent chain
    __shared__ float sA[N * N];   // the lower-triangular matrix A (row-major)
    __shared__ float sX[N * K];   // running solution, sX[row*K + col] (per-col private)

    // cooperative coalesced load of A into shared
    for (int i = col; i < N * N; i += blockDim.x) {
        sA[i] = A[i];
    }
    __syncthreads();

    long long t0 = 0;
    if (col == 0) {
        t0 = clock64();
    }

    // Forward substitution, sequential over rows (depth-N recurrence), one column
    // per thread. Pure FP32 __fmul_rn/__fadd_rn: NO tensor cores, NO TF32
    // (solve_tri.cu:70-76 -- TF32 truncation breaks the solve).
#pragma unroll 1
    for (int r = 0; r < N; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < r; ++c) {
            sum = __fadd_rn(sum, __fmul_rn(sA[r * N + c], sX[c * K + col]));
        }
        const float b = (r == col) ? 1.0f : 0.0f;  // B = identity -> column `col`
        // unit diagonal, but divide by A[r][r] to stay a general lower-tri solver
        sX[r * K + col] = (b - sum) / sA[r * N + r];
    }

    if (col == 0) {
        *cycles = clock64() - t0;
    }

    // write T = A^-1 (upper triangle is exactly 0.0 -- inverse of lower-tri)
    for (int r = 0; r < N; ++r) {
        T[r * K + col] = sX[r * K + col];
    }
}

int main() {
    // Synthetic lower-unitriangular A = I + (strictly-lower small). SCALE keeps the
    // system well-conditioned (row L1 of the strict-lower part <= ~0.2) so a correct
    // fp32 solve tracks the double reference well within tol; a mapping bug does not.
    std::mt19937 rng(20260717);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float SCALE = 0.2f / N;

    std::vector<float> A(N * N, 0.0f);
    for (int r = 0; r < N; ++r) {
        A[r * N + r] = 1.0f;  // unit diagonal
        for (int c = 0; c < r; ++c) {
            A[r * N + c] = dist(rng) * SCALE;  // strictly-lower, small
        }
        // strictly-upper stays 0 (lower-triangular)
    }

    // Obviously-correct scalar reference: double forward substitution over the SAME
    // fp32 A, column by column. Independent of the device code path -- a kernel bug
    // cannot hide behind a matching-wrong reference.
    std::vector<double> Tref(N * K, 0.0);
    for (int col = 0; col < K; ++col) {
        for (int r = 0; r < N; ++r) {
            double sum = 0.0;
            for (int c = 0; c < r; ++c) {
                sum += (double) A[r * N + c] * Tref[c * K + col];
            }
            const double b = (r == col) ? 1.0 : 0.0;
            Tref[r * K + col] = (b - sum) / (double) A[r * N + r];
        }
    }

    float *     dA;
    float *     dT;
    long long * dCyc;
    CUDA_CHECK(cudaMalloc(&dA, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dT, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCyc, sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    solve_tri_kernel<<<1, K>>>(dA, dT, dCyc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> T(N * K);
    long long          cyc = 0;
    CUDA_CHECK(cudaMemcpy(T.data(), dT, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cyc, dCyc, sizeof(long long), cudaMemcpyDeviceToHost));

    const double tol   = 1e-5;
    int          bad   = 0;
    double       worst = 0.0;
    int          wr = -1, wc = -1;
    for (int r = 0; r < N; ++r) {
        for (int col = 0; col < K; ++col) {
            const double diff = fabs((double) T[r * K + col] - Tref[r * K + col]);
            if (diff > worst) {
                worst = diff;
                wr    = r;
                wc    = col;
            }
            if (diff > tol) {
                bad++;
            }
        }
    }

    printf("tri-solve spike: [%dx%d] FP32 forward-substitution T = A^-1, %d column-chains, "
           "depth %d, tol %.1g\n",
           N, N, K, N, tol);
    printf("solve wall: %lld cycles (thread-0 column; ADVISORY -- calx-mill "
           "REFUSED-until-anchor)\n",
           cyc);
    if (bad) {
        printf("FAIL: %d/%d beyond tol; worst |diff| %.6g at row=%d col=%d "
               "(got %.8g want %.8g)\n",
               bad, N * K, worst, wr, wc, (double) T[wr * K + wc], Tref[wr * K + wc]);
        return 1;
    }
    printf("ok: all %d entries within tol; worst |diff| %.6g at row=%d col=%d\n",
           N * K, worst, wr, wc);
    return 0;
}
