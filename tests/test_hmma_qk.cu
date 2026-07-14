// plan/0143 FATTN HMMA primary -- SPIKE 1: QK^T via m16n8k8 f32-accumulate HMMA.
//
// The FATTN deep diagnosis (capture/ledger.md 2026-07-14) showed the decode op
// is ~24 ms compute / ~16 ms load, serialized: the scalar per-lane QK dot +
// softmax + V-accumulate is the lever. This spike validates the RISKIEST piece of
// the HMMA rewrite in isolation -- the m16n8k8 fragment lane-mapping for QK^T --
// BEFORE touching op_fattn_decode (plan: careful spike-then-integrate).
//
// Shape (one kv-head group at decode): gqa=6 q-heads, head_dim 256, one 32-pos KV
// tile. QK^T = scores[q,pos] = sum_d Q[q,d]*K[pos,d], contracting d=256.
//   M = q   (6 real, padded to 16)
//   N = pos (32 = 4 tiles of 8)
//   K = d   (256 = 32 steps of 8)
// One warp computes all 6x32 scores via 4*32 = 128 HMMA instructions.
//
// Parity: the scalar reference uses the SAME f16 Q and f16 K as the HMMA path, so
// the only difference is fp32 accumulate ORDER (tensor-core tree vs sequential) --
// this isolates fragment-mapping correctness from Q's f16 quantization (a separate
// integration question). Tol 5e-3 relative to the per-tile score magnitude.
//
// m16n8k8 .row.col fragment layout (PTX ISA), lane l: gid=l>>2, tid=l&3:
//   A[16x8] f16: a0 = {A[gid][2tid], A[gid][2tid+1]}, a1 = {A[gid+8][..], ..}
//   B[8x8]  f16 (K x N, col): b0 = {B[2tid][gid], B[2tid+1][gid]}   (B[k][n])
//   D[16x8] f32: d0=D[gid][2tid] d1=D[gid][2tid+1] d2=D[gid+8][2tid] d3=D[gid+8][2tid+1]
// For QK^T: A[q][k]=Q[q][d=k], B[k][n]=K[pos=n][d=k], D[q][pos].
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(e));                          \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

constexpr int HD = 256;   // head_dim
constexpr int NQ = 6;     // gqa q-heads per kv-head
constexpr int NPOS = 32;  // KV positions in a tile

__device__ inline void hmma_f32(float& d0, float& d1, float& d2, float& d3,
                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ inline unsigned pack(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// Q: [16][HD] f16 (rows 0..NQ-1 real, rest zero-padded). K: [NPOS][HD] f16.
// scores: [16][NPOS] f32 out (only [0..NQ-1][*] meaningful).
__global__ void qk_hmma_kernel(const __half* Q, const __half* K, float* scores) {
    const int lane = threadIdx.x;          // one warp
    const int gid = lane >> 2, tid = lane & 3;
    for (int nt = 0; nt < NPOS / 8; nt++) { // 4 N-tiles of 8 positions
        const int pos_base = nt * 8;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        for (int ks = 0; ks < HD / 8; ks++) { // 32 K-steps of 8 dims
            const int d_base = ks * 8;
            // A = Q[q=gid | gid+8][d_base+2tid | +1]
            unsigned a0 = pack(Q[gid * HD + d_base + 2 * tid],
                               Q[gid * HD + d_base + 2 * tid + 1]);
            unsigned a1 = pack(Q[(gid + 8) * HD + d_base + 2 * tid],
                               Q[(gid + 8) * HD + d_base + 2 * tid + 1]);
            // B[k][n] = K[pos=n=gid][d=k=d_base+2tid | +1]
            unsigned b0 = pack(K[(pos_base + gid) * HD + d_base + 2 * tid],
                               K[(pos_base + gid) * HD + d_base + 2 * tid + 1]);
            hmma_f32(d0, d1, d2, d3, a0, a1, b0);
        }
        // D[q=gid][pos_base+2tid]=d0, [+1]=d1; q=gid+8 -> d2,d3
        scores[gid * NPOS + pos_base + 2 * tid] = d0;
        scores[gid * NPOS + pos_base + 2 * tid + 1] = d1;
        scores[(gid + 8) * NPOS + pos_base + 2 * tid] = d2;
        scores[(gid + 8) * NPOS + pos_base + 2 * tid + 1] = d3;
    }
}

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<__half> Q(16 * HD, __float2half(0.f)), K(NPOS * HD);
    for (int q = 0; q < NQ; q++)
        for (int i = 0; i < HD; i++) Q[q * HD + i] = __float2half(d(rng) * 0.1f);
    for (int p = 0; p < NPOS; p++)
        for (int i = 0; i < HD; i++) K[p * HD + i] = __float2half(d(rng) * 0.1f);

    // scalar reference over the SAME f16 values, f32 sequential accumulate.
    std::vector<float> ref(NQ * NPOS);
    for (int q = 0; q < NQ; q++)
        for (int p = 0; p < NPOS; p++) {
            float acc = 0;
            for (int i = 0; i < HD; i++)
                acc += __half2float(Q[q * HD + i]) * __half2float(K[p * HD + i]);
            ref[q * NPOS + p] = acc;
        }

    __half *dQ, *dK; float *dS;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, K.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dS, 16 * NPOS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(__half), cudaMemcpyHostToDevice));
    qk_hmma_kernel<<<1, 32>>>(dQ, dK, dS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(16 * NPOS);
    CUDA_CHECK(cudaMemcpy(got.data(), dS, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // magnitude for relative tol: max |ref|
    float mag = 1e-6f;
    for (float v : ref) mag = fmaxf(mag, fabsf(v));
    const float tol = 5e-3f * mag;
    int bad = 0; float worst = 0; int wq = -1, wp = -1;
    for (int q = 0; q < NQ; q++)
        for (int p = 0; p < NPOS; p++) {
            float diff = fabsf(got[q * NPOS + p] - ref[q * NPOS + p]);
            if (diff > worst) { worst = diff; wq = q; wp = p; }
            if (diff > tol) bad++;
        }
    printf("HMMA QK^T spike: %d q-heads x %d pos, HD %d, mag %.4g, tol %.3g\n",
           NQ, NPOS, HD, mag, tol);
    if (bad) {
        printf("FAIL: %d/%d beyond tol; worst |diff| %.5g at q=%d pos=%d "
               "(got %.6g want %.6g)\n",
               bad, NQ * NPOS, worst, wq, wp,
               got[wq * NPOS + wp], ref[wq * NPOS + wp]);
        return 1;
    }
    printf("ok: all %d scores within tol; worst |diff| %.5g (%.3g of mag) at q=%d pos=%d\n",
           NQ * NPOS, worst, worst / mag, wq, wp);
    return 0;
}
