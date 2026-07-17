// plan/0143 PREFILL FA-2 causal -- SPIKE (op c): batched causal flash attention in
// m16n8k8 fragments, N query tokens x gqa heads packed into the M dimension, INLINE
// causal cutoff, parity vs a scalar causal-flash reference.
//
// Extends the decode flash spike (test_hmma_attn.cu) to the two genuinely-new prefill
// surfaces (PREFILL-SEMANTICS.md §2, derivation §c):
//   (1) M-TILING: decode carries NQ=6 gqa query rows in ONE m16n8k8 (a0 only; a1=0
//       wasted). Prefill carries N_COL = N_TOK*NQ = 24 query COLUMNS across ceil(24/16)=2
//       M-subtiles, each a FULL m16n8k8 with BOTH a0 (cols s*16+gid) and a1 (cols
//       s*16+gid+8) real -- recovers the decode-wasted half.
//   (2) CAUSAL: column c=(token t, head h) at global query pos BASE+t; KV cell j masked
//       (-INF) iff j > BASE+t  (byte-for-byte the fork's mask VALUE, llama-kv-cache.cpp
//       :1566), computed INLINE from positions -- no [n_kv][N] mask tensor.
// Precision: NOT reused here (this spike is f16-operand, single-HMMA, like test_hmma_attn.cu)
// -- the 2-limb Q&P split is a separate proven layer (test_hmma_qk shows a_hi/a_lo); the
// causal M-tiling is the new surface. tol 5e-3 of the per-column output magnitude.
//
// Layout (validated in test_hmma_qk.cu), lane l: gid=l>>2, tid=l&3:
//   QK: A=Q[col][d], B[k=d][n=pos]=K[pos][d], D=score[col][pos]  (m=col in M, n=pos)
//   PV: A=P[col][pos], B[k=pos][n=d]=V[pos][d], D=out[col][d]
//   a0 -> M-row s*16+gid ; a1 -> M-row s*16+gid+8 ; D: d0,d1=a0-col@(2tid,2tid+1)
//                                                     d2,d3=a1-col@(2tid,2tid+1)
// One warp (one KV head) is enough to de-risk the M-tiling + causal primitive in
// isolation; multi-head/multi-tile/online-softmax/tail are decode-proven integration
// concerns, not new here. Single 32-position KV tile, no partial tail (decode handles it).
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

constexpr int HD    = 256;   // head_dim
constexpr int NQ    = 6;     // gqa q-heads per kv head
constexpr int N_TOK = 4;     // query tokens per tile (Turing ncols1=4, PREFILL-SEMANTICS §2)
constexpr int N_COL = N_TOK * NQ;   // 24 query columns (token-major: col = t*NQ + h)
constexpr int N_SUB = (N_COL + 15) / 16;  // 2 M-subtiles of 16
constexpr int NPOS  = 32;    // KV positions in the tile
constexpr int BASE  = 28;    // global position of query token 0 (tokens at BASE..BASE+N_TOK-1)

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

// Q[N_COL][HD] (col = token*NQ + head), K/V[NPOS][HD]. out[N_COL][HD] normalized.
__global__ void causal_hmma_kernel(const __half* Q, const __half* K,
                                   const __half* V, float scale, float* out) {
    const int lane = threadIdx.x, gid = lane >> 2, tid = lane & 3;
    constexpr int NT = NPOS / 8;   // 4 N-tiles of 8 positions

    for (int s = 0; s < N_SUB; s++) {
        const int colA = s * 16 + gid;      // a0 M-row
        const int colB = s * 16 + gid + 8;  // a1 M-row
        const int tokA = colA / NQ, tokB = colB / NQ;   // query token of each column
        const int qposA = BASE + tokA, qposB = BASE + tokB;

        // ---- QK^T: scA/scB[nt][2] = score[colA/colB][pb+2tid, +1] ----
        float scA0[NT], scA1[NT], scB0[NT], scB1[NT];
        for (int nt = 0; nt < NT; nt++) {
            const int pb = nt * 8;
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (int ks = 0; ks < HD / 8; ks++) {
                const int db = ks * 8;
                const unsigned a0 = pk(Q[colA * HD + db + 2*tid], Q[colA * HD + db + 2*tid + 1]);
                const unsigned a1 = pk(Q[colB * HD + db + 2*tid], Q[colB * HD + db + 2*tid + 1]);
                const unsigned b0 = pk(K[(pb + gid) * HD + db + 2*tid], K[(pb + gid) * HD + db + 2*tid + 1]);
                hmma_f32(d0, d1, d2, d3, a0, a1, b0);
            }
            // causal: mask position p for column X iff p > qposX (INLINE, from positions)
            const int p0 = pb + 2*tid, p1 = pb + 2*tid + 1;
            scA0[nt] = (p0 > qposA) ? -INFINITY : d0 * scale;
            scA1[nt] = (p1 > qposA) ? -INFINITY : d1 * scale;
            scB0[nt] = (p0 > qposB) ? -INFINITY : d2 * scale;
            scB1[nt] = (p1 > qposB) ? -INFINITY : d3 * scale;
        }
        // ---- per-column online softmax (quad reductions over the 4 tid lanes) ----
        float mA = -INFINITY, mB = -INFINITY;
        for (int nt = 0; nt < NT; nt++) {
            mA = fmaxf(mA, fmaxf(scA0[nt], scA1[nt]));
            mB = fmaxf(mB, fmaxf(scB0[nt], scB1[nt]));
        }
        mA = qmax(mA); mB = qmax(mB);
        float slA = 0, slB = 0;
        __half pA0[NT], pA1[NT], pB0[NT], pB1[NT];
        for (int nt = 0; nt < NT; nt++) {
            const float eA0 = (scA0[nt] > -INFINITY) ? __expf(scA0[nt] - mA) : 0.f;
            const float eA1 = (scA1[nt] > -INFINITY) ? __expf(scA1[nt] - mA) : 0.f;
            const float eB0 = (scB0[nt] > -INFINITY) ? __expf(scB0[nt] - mB) : 0.f;
            const float eB1 = (scB1[nt] > -INFINITY) ? __expf(scB1[nt] - mB) : 0.f;
            slA += eA0 + eA1; slB += eB0 + eB1;
            pA0[nt] = __float2half(eA0); pA1[nt] = __float2half(eA1);
            pB0[nt] = __float2half(eB0); pB1[nt] = __float2half(eB1);
        }
        const float sA = qsum(slA), sB = qsum(slB);
        // ---- P.V: out[colA/colB][d] = sum_pos P[col][pos]*V[pos][d], normalized ----
        for (int dt = 0; dt < HD / 8; dt++) {
            float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
            for (int ks = 0; ks < NT; ks++) {
                const int pb = ks * 8;
                const unsigned a0 = pk(pA0[ks], pA1[ks]);   // P[colA][pos=pb+2tid, +1]
                const unsigned a1 = pk(pB0[ks], pB1[ks]);   // P[colB][..]
                const unsigned b0 = pk(V[(pb + 2*tid) * HD + dt*8 + gid],
                                       V[(pb + 2*tid + 1) * HD + dt*8 + gid]);
                hmma_f32(o0, o1, o2, o3, a0, a1, b0);
            }
            if (colA < N_COL) {
                out[colA * HD + dt*8 + 2*tid]     = o0 / sA;
                out[colA * HD + dt*8 + 2*tid + 1] = o1 / sA;
            }
            if (colB < N_COL) {
                out[colB * HD + dt*8 + 2*tid]     = o2 / sB;
                out[colB * HD + dt*8 + 2*tid + 1] = o3 / sB;
            }
        }
    }
}

int main() {
    std::mt19937 rng(0xCA05A1);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<__half> Q(N_COL * HD), K(NPOS * HD), V(NPOS * HD);
    for (int c = 0; c < N_COL; c++)
        for (int i = 0; i < HD; i++) Q[c * HD + i] = __float2half(d(rng) * 0.2f);
    for (int p = 0; p < NPOS; p++)
        for (int i = 0; i < HD; i++) {
            K[p * HD + i] = __float2half(d(rng) * 0.2f);
            V[p * HD + i] = __float2half(d(rng));
        }
    const float scale = 1.0f / sqrtf((float)HD);

    // scalar CAUSAL flash reference (same f16 values, f32 P)
    std::vector<float> ref(N_COL * HD);
    for (int c = 0; c < N_COL; c++) {
        const int tok = c / NQ, qpos = BASE + tok;
        float sc[NPOS], mx = -INFINITY;
        for (int p = 0; p < NPOS; p++) {
            if (p > qpos) { sc[p] = -INFINITY; continue; }
            float acc = 0;
            for (int i = 0; i < HD; i++) acc += __half2float(Q[c*HD+i]) * __half2float(K[p*HD+i]);
            sc[p] = acc * scale; mx = fmaxf(mx, sc[p]);
        }
        float sm = 0;
        for (int p = 0; p < NPOS; p++) { float e = (sc[p] > -INFINITY) ? expf(sc[p]-mx) : 0.f; sc[p] = e; sm += e; }
        for (int i = 0; i < HD; i++) {
            float o = 0; for (int p = 0; p < NPOS; p++) o += sc[p] * __half2float(V[p*HD+i]);
            ref[c*HD+i] = o / sm;
        }
    }

    __half *dQ, *dK, *dV; float *dO;
    CUDA_CHECK(cudaMalloc(&dQ, Q.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dK, K.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dV, V.size()*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dO, N_COL*HD*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size()*sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size()*sizeof(__half), cudaMemcpyHostToDevice));
    causal_hmma_kernel<<<1, 32>>>(dQ, dK, dV, scale, dO);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(N_COL*HD);
    CUDA_CHECK(cudaMemcpy(got.data(), dO, got.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float mag = 1e-6f; for (float v : ref) mag = fmaxf(mag, fabsf(v));
    const float tol = 5e-3f * mag;
    int bad = 0; float worst = 0; int wc=-1, wi=-1;
    for (int c = 0; c < N_COL; c++) for (int i = 0; i < HD; i++) {
        float diff = fabsf(got[c*HD+i] - ref[c*HD+i]);
        if (diff > worst) { worst = diff; wc=c; wi=i; }
        if (diff > tol) bad++;
    }
    printf("FA causal spike: %d cols (%d tok x %d gqa), %d M-subtiles, %d-pos tile, BASE %d, mag %.4g, tol %.3g\n",
           N_COL, N_TOK, NQ, N_SUB, NPOS, BASE, mag, tol);
    // report the causal boundary column (token 0, most-masked) explicitly
    printf("  causal: token0 attends [0..%d] (masks %d..%d); token%d attends [0..%d]\n",
           BASE, BASE+1, NPOS-1, N_TOK-1, BASE+N_TOK-1);
    if (bad) {
        printf("FAIL: %d/%d beyond tol; worst |diff| %.5g at col=%d (tok %d) d=%d (got %.6g want %.6g)\n",
               bad, N_COL*HD, worst, wc, wc/NQ, wi, got[wc*HD+wi], ref[wc*HD+wi]);
        return 1;
    }
    printf("ok: all %d outputs within tol; worst |diff| %.5g (%.3g of mag) at col=%d (tok %d) d=%d\n",
           N_COL*HD, worst, worst/mag, wc, wc/NQ, wi);
    return 0;
}
