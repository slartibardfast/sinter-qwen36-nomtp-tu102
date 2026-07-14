// plan/0143 FATTN HMMA primary -- SPIKE 2: full single-tile flash attention in
// m16n8k8 fragments (QK^T -> softmax -> P.V), parity vs a scalar flash reference.
//
// Builds on SPIKE 1 (test_hmma_qk.cu, QK^T fragment map validated). Here the whole
// decode-attention inner loop for one 32-position KV tile runs in tensor-core
// fragments, exercising the two new risk surfaces before integration:
//   (a) online softmax across the D-fragment lane layout -- each q-row's 32 scores
//       are split across the 4 lanes of its quad (lane>>2 == q), reduced by
//       __shfl width-4 (quad max / quad sum);
//   (b) the SECOND HMMA (P.V): the QK output fragment d0,d1 (=score[q][pos]) IS the
//       P.V input A-fragment after exp+f16-cast -- no reshuffle. B=V[pos][d].
//
// Layout rules (validated in SPIKE 1), lane l: gid=l>>2, tid=l&3:
//   A[M][K] f16: a0={A[gid][2tid],A[gid][2tid+1]}, a1={A[gid+8][..]}   (m=gid, k=2tid)
//   B[K][N] f16: b0={B[2tid][gid],B[2tid+1][gid]}                      (n=gid, k=2tid)
//   D[M][N] f32: d0=D[gid][2tid] d1=D[gid][2tid+1] d2=D[gid+8][2tid] d3=..+1
//   QK: A=Q[q][d], B[k=d][n=pos]=K[pos][d], D=score[q][pos]
//   PV: A=P[q][pos], B[k=pos][n=d]=V[pos][d], D=out[q][d]
//
// Parity: scalar flash reference uses the SAME f16 Q/K/V; the device casts P to f16
// for the 2nd matmul (as the fork's fattn-mma does), so out differs from an f32-P
// reference only by P's f16 rounding -> tol 5e-3 of the per-q output magnitude.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                        \
    do { cudaError_t e = (call);                                               \
        if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d: %s\n", #call,\
            __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int HD = 256;   // head_dim
constexpr int NQ = 6;     // gqa q-heads
constexpr int NPOS = 32;  // KV positions in a tile

__device__ inline void hmma_f32(float& d0, float& d1, float& d2, float& d3,
                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ inline unsigned pk(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi); return *reinterpret_cast<unsigned*>(&h);
}
__device__ inline float qmax(float v) { // quad (width-4) max
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1, 4));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2, 4));
}
__device__ inline float qsum(float v) { // quad (width-4) sum
    v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
    return v + __shfl_xor_sync(0xffffffffu, v, 2, 4);
}

// scale = 1/sqrt(HD) applied to scores (FA scale). out[NQ][HD] f32.
__global__ void attn_hmma_kernel(const __half* Q, const __half* K,
                                 const __half* V, float scale, float* out) {
    const int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    // ---- QK^T: score[gid][pos] in sc0[nt], sc1[nt] for the 4 N-tiles ----
    float sc0[4], sc1[4];
    for (int nt = 0; nt < NPOS / 8; nt++) {
        const int pb = nt * 8;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        for (int ks = 0; ks < HD / 8; ks++) {
            const int db = ks * 8;
            unsigned a0 = pk(Q[gid * HD + db + 2 * tid], Q[gid * HD + db + 2 * tid + 1]);
            unsigned a1 = pk(Q[(gid + 8) * HD + db + 2 * tid], Q[(gid + 8) * HD + db + 2 * tid + 1]);
            unsigned b0 = pk(K[(pb + gid) * HD + db + 2 * tid], K[(pb + gid) * HD + db + 2 * tid + 1]);
            hmma_f32(d0, d1, d2, d3, a0, a1, b0);
        }
        sc0[nt] = d0 * scale; sc1[nt] = d1 * scale;  // (d2,d3 = q=gid+8 padding)
    }
    // ---- online softmax over this q-row's 32 positions (quad reductions) ----
    float ml = -INFINITY;
    for (int nt = 0; nt < 4; nt++) { ml = fmaxf(ml, fmaxf(sc0[nt], sc1[nt])); }
    const float m = qmax(ml);
    float sl = 0;
    __half p0[4], p1[4];
    for (int nt = 0; nt < 4; nt++) {
        float e0 = __expf(sc0[nt] - m), e1 = __expf(sc1[nt] - m);
        sl += e0 + e1; p0[nt] = __float2half(e0); p1[nt] = __float2half(e1);
    }
    const float s = qsum(sl);
    // ---- P.V: out[gid][d] = sum_pos P[gid][pos]*V[pos][d], unnormalized ----
    // 32 d-tiles of 8; K-steps = 4 pos-tiles of 8. A=P (p0,p1 = the pos K-tile).
    for (int dt = 0; dt < HD / 8; dt++) {
        float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
        for (int ks = 0; ks < NPOS / 8; ks++) {
            const int pb = ks * 8;
            unsigned a0 = pk(p0[ks], p1[ks]);   // P[gid][pos=pb+2tid, +1]
            unsigned a1 = 0;                     // q=gid+8 padding
            unsigned b0 = pk(V[(pb + 2 * tid) * HD + dt * 8 + gid],
                             V[(pb + 2 * tid + 1) * HD + dt * 8 + gid]);
            hmma_f32(o0, o1, o2, o3, a0, a1, b0);
        }
        if (gid < NQ) {  // out[gid][dt*8 + 2tid, +1]
            out[gid * HD + dt * 8 + 2 * tid] = o0 / s;
            out[gid * HD + dt * 8 + 2 * tid + 1] = o1 / s;
        }
    }
}

int main() {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<__half> Q(16 * HD, __float2half(0.f)), K(NPOS * HD), V(NPOS * HD);
    for (int q = 0; q < NQ; q++)
        for (int i = 0; i < HD; i++) Q[q * HD + i] = __float2half(d(rng) * 0.2f);
    for (int p = 0; p < NPOS; p++)
        for (int i = 0; i < HD; i++) {
            K[p * HD + i] = __float2half(d(rng) * 0.2f);
            V[p * HD + i] = __float2half(d(rng));
        }
    const float scale = 1.0f / sqrtf((float)HD);

    // scalar flash reference (same f16 values, f32 P)
    std::vector<float> ref(NQ * HD);
    for (int q = 0; q < NQ; q++) {
        float sc[NPOS], mx = -INFINITY;
        for (int p = 0; p < NPOS; p++) {
            float acc = 0;
            for (int i = 0; i < HD; i++) acc += __half2float(Q[q*HD+i]) * __half2float(K[p*HD+i]);
            sc[p] = acc * scale; mx = fmaxf(mx, sc[p]);
        }
        float sm = 0; for (int p = 0; p < NPOS; p++) { sc[p] = expf(sc[p]-mx); sm += sc[p]; }
        for (int i = 0; i < HD; i++) {
            float o = 0; for (int p = 0; p < NPOS; p++) o += sc[p] * __half2float(V[p*HD+i]);
            ref[q*HD+i] = o / sm;
        }
    }

    __half *dQ, *dK, *dV; float *dO;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, K.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dV, V.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dO, NQ*HD*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size()*sizeof(__half), cudaMemcpyHostToDevice));
    attn_hmma_kernel<<<1, 32>>>(dQ, dK, dV, scale, dO);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(NQ*HD);
    CUDA_CHECK(cudaMemcpy(got.data(), dO, got.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float mag = 1e-6f; for (float v : ref) mag = fmaxf(mag, fabsf(v));
    const float tol = 5e-3f * mag;
    int bad = 0; float worst = 0; int wq=-1, wi=-1;
    for (int q = 0; q < NQ; q++) for (int i = 0; i < HD; i++) {
        float diff = fabsf(got[q*HD+i] - ref[q*HD+i]);
        if (diff > worst) { worst = diff; wq=q; wi=i; }
        if (diff > tol) bad++;
    }
    printf("HMMA flash spike: %d q x %d d, one %d-pos tile, mag %.4g, tol %.3g\n",
           NQ, HD, NPOS, mag, tol);
    if (bad) {
        printf("FAIL: %d/%d beyond tol; worst |diff| %.5g at q=%d d=%d (got %.6g want %.6g)\n",
               bad, NQ*HD, worst, wq, wi, got[wq*HD+wi], ref[wq*HD+wi]);
        return 1;
    }
    printf("ok: all %d outputs within tol; worst |diff| %.5g (%.3g of mag) at q=%d d=%d\n",
           NQ*HD, worst, worst/mag, wq, wi);
    return 0;
}
