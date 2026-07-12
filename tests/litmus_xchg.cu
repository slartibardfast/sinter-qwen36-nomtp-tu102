// Cross-GPU Y06 exchange litmus — validates the k0/ops/xchg.cuh protocol on
// real 2-GPU NVLink hardware before the dual-GPU harness is built on it.
// Standalone (not a CMake target): build with tools/build-litmus-xchg.sh.
//
// Two cooperative kernels, one per GPU, each 72x384. Per round each GPU
// computes a partial (a known fn of round+gpu), pushes it to the PEER's
// inbox (line-filling float4 + membar.sys), crosses its LOCAL Y02 boundary,
// publishes the seqno to the peer, polls its own inbox, strong-reads the
// received payload, and folds out = p0 + p1 in fixed gpu-index order. The
// mirrored result MUST be bit-identical on both GPUs and equal the reference
// sum every round.
//
//   positive           : mirrored sums bit-identical across GPUs, 0 stale
//                        over N rounds with the consumer L1 pre-warmed.
//   -DXCHG_PLAIN_READS : payload read with a plain LDG — expect the measured
//                        xgpu stale class, CAUGHT.
//   -DXCHG_NO_MEMBAR   : drop the producer membar.sys — flag may beat the
//                        payload over NVLink, CAUGHT.
//   -DXCHG_SEQNO_EARLY : publish seqno BEFORE the local boundary — torn
//                        multi-block payload, CAUGHT.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "../core/sync.cuh"

#ifndef XCHG_ROUNDS
#define XCHG_ROUNDS 20000
#endif
#ifndef XCHG_N
#define XCHG_N 5120           // the real activation slice width
#endif

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, \
                 cudaGetErrorString(e_)); return 2; } } while (0)

__device__ __forceinline__ unsigned ld_acq_sys(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// partial for (round r, gpu g), element i: small ints so the two-GPU sum is
// an exact fp32 both sides reproduce bit-for-bit.
__device__ __forceinline__ float partial(unsigned r, int g, int i) {
    return (float)((r * 2u + (unsigned)g + (unsigned)i) & 0x3ffu)
           * (g == 0 ? 1.0f : 0.5f);
}

// Buffers for one GPU. Peer pointers are the OTHER GPU's inbox (UVA, peer
// access enabled both ways).
struct Args {
    int gpu;
    float    *local_partial;   // device-local scratch
    float    *out;             // device-local mirrored result
    float    *peer_payload;    // PEER inbox payload (write here)
    unsigned *peer_seqno;      // PEER inbox seqno  (publish here)
    const float    *my_payload;// this GPU's inbox payload (peer wrote it)
    const unsigned *my_seqno;  // this GPU's inbox seqno  (poll it)
    unsigned *peer_done;       // PEER round-done seqno (publish here)
    const unsigned *my_done;   // this GPU's round-done seqno (poll it)
    unsigned *counter;         // this GPU's Y02 grid counter (zeroed)
    unsigned *stale;           // mismatch/stale accumulator
    float    *mirror;          // last-round result, for cross-GPU compare
};

__global__ void xchg_litmus(Args a) {
    using namespace mk;
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthr = gridDim.x * blockDim.x;
    unsigned ep = 0;
    GridBoundary bar{ a.counter };

    for (unsigned r = 1; r <= XCHG_ROUNDS; r++) {
        for (int i = tid; i < XCHG_N; i += nthr)
            a.local_partial[i] = partial(r, a.gpu, i);
        bar.cross(ep);                      // partial fully written

        // pre-warm consumer L1 with our inbox's now-stale line
        if (threadIdx.x == 0) {
            volatile float w = a.my_payload[(blockIdx.x * 37) % XCHG_N];
            (void)w;
        }

        // OP_XCHG_PUSH: line-filling float4 peer store
        const int n4 = XCHG_N >> 2;
        const float4 *s4 = reinterpret_cast<const float4 *>(a.local_partial);
        float4 *d4 = reinterpret_cast<float4 *>(a.peer_payload);
        for (int i = tid; i < n4; i += nthr) d4[i] = s4[i];
#ifndef XCHG_NO_MEMBAR
        membar_sys();
#endif

#ifdef XCHG_SEQNO_EARLY
        if (tid == 0) st_release_sys(a.peer_seqno, r);   // BEFORE the boundary
#endif
        bar.cross(ep);                      // all blocks' pushes complete

        // OP_XCHG_REDUCE
#ifndef XCHG_SEQNO_EARLY
        if (tid == 0) st_release_sys(a.peer_seqno, r);
#endif
        if (threadIdx.x == 0)
            while ((int)(ld_acq_sys(a.my_seqno) - r) < 0) { }
        __syncthreads();

        for (int i = tid; i < XCHG_N; i += nthr) {
            const float mine = a.local_partial[i];
            float peer;
#ifdef XCHG_PLAIN_READS
            asm volatile("ld.global.f32 %0, [%1];" : "=f"(peer)
                         : "l"(a.my_payload + i) : "memory");
#else
            peer = ld_cg(a.my_payload + i);
#endif
            const float folded = (a.gpu == 0) ? (mine + peer) : (peer + mine);
            a.out[i] = folded;
            const float ref = partial(r, 0, i) + partial(r, 1, i);
            if (folded != ref) atomicAdd(a.stale, 1u);
            if (r == XCHG_ROUNDS) a.mirror[i] = folded;
        }
        bar.cross(ep);                      // local: all blocks done reducing

        // Cross-GPU round lockstep (harness plumbing, NOT the protocol under
        // test): neither GPU may overwrite the peer's inbox for round r+1
        // until the peer finished reading round r. Without this the reuse
        // would show as false stale. The real kernel gets this for free from
        // the per-pass host doorbell.
        if (tid == 0) st_release_sys(a.peer_done, r);
        if (threadIdx.x == 0)
            while ((int)(ld_acq_sys(a.my_done) - r) < 0) { }
        __syncthreads();
        bar.cross(ep);
    }
}

struct GpuBufs {
    float *local_partial, *out, *inbox_payload, *mirror;
    unsigned *inbox_seqno, *counter, *stale, *done_seqno;
};

static int alloc_gpu(int g, GpuBufs &b) {
    CK(cudaSetDevice(g));
    CK(cudaMalloc(&b.local_partial, XCHG_N * 4));
    CK(cudaMalloc(&b.out, XCHG_N * 4));
    CK(cudaMalloc(&b.inbox_payload, XCHG_N * 4));
    CK(cudaMalloc(&b.mirror, XCHG_N * 4));
    CK(cudaMalloc(&b.inbox_seqno, 4));
    CK(cudaMalloc(&b.counter, 4));
    CK(cudaMalloc(&b.stale, 4));
    CK(cudaMalloc(&b.done_seqno, 4));
    CK(cudaMemset(b.inbox_payload, 0, XCHG_N * 4));
    CK(cudaMemset(b.inbox_seqno, 0, 4));
    CK(cudaMemset(b.counter, 0, 4));
    CK(cudaMemset(b.stale, 0, 4));
    CK(cudaMemset(b.done_seqno, 0, 4));
    return 0;
}

int main() {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (n < 2) { std::printf("need 2 GPUs, have %d\n", n); return 2; }

    // Peer access both ways (NVLink).
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
    CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));

    GpuBufs b0{}, b1{};
    if (alloc_gpu(0, b0)) return 2;
    if (alloc_gpu(1, b1)) return 2;

    cudaStream_t s0, s1;
    CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0));
    CK(cudaSetDevice(1)); CK(cudaStreamCreate(&s1));

    Args a0{0, b0.local_partial, b0.out, /*peer=*/b1.inbox_payload,
            b1.inbox_seqno, b0.inbox_payload, b0.inbox_seqno,
            /*peer_done=*/b1.done_seqno, /*my_done=*/b0.done_seqno,
            b0.counter, b0.stale, b0.mirror};
    Args a1{1, b1.local_partial, b1.out, /*peer=*/b0.inbox_payload,
            b0.inbox_seqno, b1.inbox_payload, b1.inbox_seqno,
            /*peer_done=*/b0.done_seqno, /*my_done=*/b1.done_seqno,
            b1.counter, b1.stale, b1.mirror};

    dim3 grid(72), block(384);
    void *p0[] = {&a0}; void *p1[] = {&a1};

    CK(cudaSetDevice(0));
    CK(cudaLaunchCooperativeKernel((void *)xchg_litmus, grid, block, p0, 0, s0));
    CK(cudaSetDevice(1));
    CK(cudaLaunchCooperativeKernel((void *)xchg_litmus, grid, block, p1, 0, s1));

    // Bounded wait — never kill; report a hang and let context teardown reclaim.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    bool d0 = false, d1 = false;
    for (;;) {
        if (!d0) { cudaSetDevice(0); if (cudaStreamQuery(s0) == cudaSuccess) d0 = true; }
        if (!d1) { cudaSetDevice(1); if (cudaStreamQuery(s1) == cudaSuccess) d1 = true; }
        if (d0 && d1) break;
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "HANG: streams not complete in 120s "
                         "(d0=%d d1=%d) — reclaimed at exit, not killed\n", d0, d1);
            return 3;
        }
    }

    unsigned st0 = 0, st1 = 0;
    cudaSetDevice(0); CK(cudaMemcpy(&st0, b0.stale, 4, cudaMemcpyDeviceToHost));
    cudaSetDevice(1); CK(cudaMemcpy(&st1, b1.stale, 4, cudaMemcpyDeviceToHost));

    // Cross-GPU mirror bit-identity: both GPUs' last-round result must match.
    float m0[XCHG_N], m1[XCHG_N];
    cudaSetDevice(0); CK(cudaMemcpy(m0, b0.mirror, XCHG_N * 4, cudaMemcpyDeviceToHost));
    cudaSetDevice(1); CK(cudaMemcpy(m1, b1.mirror, XCHG_N * 4, cudaMemcpyDeviceToHost));
    unsigned mirror_mism = 0;
    for (int i = 0; i < XCHG_N; i++)
        if (std::memcmp(&m0[i], &m1[i], 4) != 0) mirror_mism++;

    const unsigned stale = st0 + st1;
#if defined(XCHG_PLAIN_READS) || defined(XCHG_NO_MEMBAR) || defined(XCHG_SEQNO_EARLY)
    std::printf("negative: stale %u (gpu0 %u gpu1 %u) mirror_mism %u -> %s\n",
                stale, st0, st1, mirror_mism,
                (stale > 0 || mirror_mism > 0) ? "CAUGHT (expected)" : "NOT CAUGHT");
    return (stale > 0 || mirror_mism > 0) ? 0 : 1;
#else
    std::printf("positive: stale %u (gpu0 %u gpu1 %u) mirror_mism %u/%d -> %s\n",
                stale, st0, st1, mirror_mism, XCHG_N,
                (stale == 0 && mirror_mism == 0) ? "PASS" : "FAIL");
    return (stale == 0 && mirror_mism == 0) ? 0 : 1;
#endif
}
