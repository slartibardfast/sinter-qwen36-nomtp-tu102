// G14 litmus for the Y02 boundary (sync_protocol Y02) and the strong-read
// rule. Cooperative launch, 72 blocks x 384 threads (the G11 shape).
//
// Positive claims, asserted over many epochs with randomized per-block work
// delays (the gate's "randomized SM completion order"):
//   P1 exactly-once counting: after E epochs the counter reads E * gridDim.
//   P2 message passing: block b writes payload[b] = epoch before arriving;
//      after the crossing, block (b+1) % grid reads it STRONG and must see
//      the current epoch, never a stale one — with the reader's L1
//      deliberately pre-warmed with last epoch's line (the persistent-kernel
//      hazard: no launch boundary invalidates L1 for us).
// Negatives (each must be CAUGHT — a negative that passes fails the gate):
//   N1 (-DLITMUS_PLAIN_READS) payload read with a plain LDG: the pre-warmed
//      L1 must yield at least one stale observation across the run.
//   N2 (-DLITMUS_NO_RELEASE) the producer publishes AFTER counting itself
//      in (the forgot-to-release protocol violation): the stale detector
//      must fire. This validates the DETECTOR — a broken protocol a green
//      test cannot see is worse than no test.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "../core/sync.cuh"

#ifndef LITMUS_EPOCHS
#define LITMUS_EPOCHS 200000
#endif

__device__ __forceinline__ void red_add_plain_gpu(unsigned *p, unsigned v) {
    asm volatile("red.global.gpu.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// READER_TID: which thread performs the pre-warm and the checked read.
// Thread 0 also executes the arrive (whose release RED carries a MEMBAR);
// a reader in another warp never executes any membar — the position all
// 383 consumer threads of the real kernel are in.
#ifndef READER_TID
#define READER_TID 0
#endif

#ifdef LITMUS_MP
// Classic intra-GPU message passing, no membar anywhere on the reader path:
// even blocks produce {payload, then release flag}, odd blocks warm the
// payload line (plain), spin on the flag (strong, separate 128B line), then
// plain-read the payload. Any stale observation here proves the per-SM L1
// stale-read hazard EXISTS locally; zero over the run means plain loads
// after a strong flag are fresh on this chip even without a membar.
__global__ void litmus(unsigned *counter, unsigned *payload, unsigned *stale,
                       unsigned *xorshift_seed) {
    mk::GridBoundary bar{counter};
    unsigned epoch = 0;
    const unsigned pair = blockIdx.x / 2;
    const bool producer = (blockIdx.x % 2) == 0;
    unsigned *pay = payload + pair * 32;        // one 128B line per pair
    unsigned *flag = payload + (gridDim.x / 2 + pair) * 32 + 16;

    for (int e = 1; e <= LITMUS_EPOCHS; e++) {
        if (threadIdx.x == 0) {
            if (producer) {
                *pay = (unsigned)e;             // plain store
                mk::membar_gl();                // producer ordering
                mk::st_release_gpu(flag, (unsigned)e);
            } else {
                unsigned warm;
                asm volatile("ld.global.u32 %0, [%1];" : "=r"(warm)
                             : "l"(pay) : "memory");
                (void)warm;
                while (mk::ld_acquire_gpu(flag) != (unsigned)e) { }
                unsigned seen;
#ifdef LITMUS_MP_CG
                seen = mk::ld_cg(pay);
#else
                asm volatile("ld.global.u32 %0, [%1];" : "=r"(seen)
                             : "l"(pay) : "memory");
#endif
                if (seen != (unsigned)e) atomicAdd(stale, 1u);
            }
        }
        bar.cross(epoch);                        // epoch framing only
        bar.cross(epoch);
    }
}
#else
__global__ void litmus(unsigned *counter, unsigned *payload, unsigned *stale,
                       unsigned *xorshift_seed) {
    mk::GridBoundary bar{counter};
    unsigned epoch = 0;
    const unsigned b = blockIdx.x;
    const unsigned peer = (b + 1) % gridDim.x;
    unsigned rng = 0x9e3779b9u ^ (b * 2654435761u) ^ *xorshift_seed;

    for (int e = 1; e <= LITMUS_EPOCHS; e++) {
        // Randomized completion order: spin a block-dependent while.
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        for (volatile int i = 0, n = rng & 0x3ff; i < n; i++) { }

        if (threadIdx.x == READER_TID) {
            // Pre-warm our L1 with the peer's CURRENT (about to be stale)
            // line via a plain load.
            unsigned warm;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(warm)
                         : "l"(payload + peer) : "memory");
            (void)warm;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
#ifndef LITMUS_NO_RELEASE
            payload[b] = (unsigned)e;            // ordered by the release RED
#endif
        }

#ifdef LITMUS_PLAIN_ARRIVE
        // Consumer-side membar isolation: arrive WITH release (producer
        // correctness kept) is replaced by plain RED + membar-free wait,
        // producer ordering supplied by an explicit membar BEFORE arriving,
        // so the only membar removed is the one on the consumer's path...
        // except thread 0 is both. With READER_TID in another warp the
        // reader path is membar-free either way.
        __syncthreads();
        epoch += 1;
        if (threadIdx.x == 0) {
            mk::membar_gl();                      // producer ordering only
            red_add_plain_gpu(counter, 1u);
            const unsigned target = epoch * gridDim.x;
            while ((int)(mk::ld_acquire_gpu(counter) - target) < 0) { }
        }
        __syncthreads();
#else
        bar.cross(epoch);
#endif

#ifdef LITMUS_NO_RELEASE
        // Protocol violation: publish only after counting ourselves in.
        if (threadIdx.x == 0) payload[b] = (unsigned)e;
#endif

        if (threadIdx.x == READER_TID) {
            unsigned seen;
#ifdef LITMUS_PLAIN_READS
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(seen)
                         : "l"(payload + peer) : "memory");
#else
            seen = mk::ld_cg(payload + peer);
#endif
            if (seen != (unsigned)e) atomicAdd(stale, 1u);
        }
        // Second crossing so nobody overwrites payload[b] for epoch e+1
        // while the peer is still checking epoch e.
        bar.cross(epoch);
    }
}
#endif // LITMUS_MP

int main(int argc, char **argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    cudaSetDevice(dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);

    unsigned *counter, *payload, *stale, *seed;
    cudaMalloc(&counter, 4);
    cudaMalloc(&payload, 4 * 32 * 2 * p.multiProcessorCount);
    cudaMalloc(&stale, 4);
    cudaMalloc(&seed, 4);
    cudaMemset(counter, 0, 4);
    cudaMemset(payload, 0, 4 * 32 * 2 * p.multiProcessorCount);
    cudaMemset(stale, 0, 4);
    unsigned hseed = (unsigned)time(nullptr);
    cudaMemcpy(seed, &hseed, 4, cudaMemcpyHostToDevice);

    dim3 grid(p.multiProcessorCount), block(384);
    void *args[] = {&counter, &payload, &stale, &seed};
    cudaError_t e = cudaLaunchCooperativeKernel((void *)litmus, grid, block, args);
    if (e != cudaSuccess) {
        fprintf(stderr, "launch: %s\n", cudaGetErrorString(e));
        return 2;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        fprintf(stderr, "sync: %s\n", cudaGetErrorString(e));
        return 2;
    }

    unsigned cnt = 0, st = 0;
    cudaMemcpy(&cnt, counter, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&st, stale, 4, cudaMemcpyDeviceToHost);
    const unsigned want = 2u * LITMUS_EPOCHS * grid.x;

    const bool count_ok = (cnt == want);
#if defined(LITMUS_MP) && !defined(LITMUS_MP_CG)
    // Probe build: report the observation either way; exit 0. The RULE is
    // decided by what this measures, not the other way round.
    printf("mp-probe: counter %u/%u stale %u -> %s\n", cnt, want, st,
           st > 0 ? "local stale L1 reads EXIST" : "no local stale observed");
    return 0;
#elif defined(LITMUS_PLAIN_READS) || defined(LITMUS_NO_RELEASE)
    // Negative build: staleness MUST have been observed.
    printf("negative: counter %u/%u stale %u -> %s\n", cnt, want, st,
           st > 0 ? "CAUGHT (expected)" : "NOT CAUGHT");
    return st > 0 ? 0 : 1;
#else
    printf("positive: counter %u/%u (exactly-once %s) stale %u -> %s\n", cnt,
           want, count_ok ? "ok" : "VIOLATED", st,
           (count_ok && st == 0) ? "PASS" : "FAIL");
    return (count_ok && st == 0) ? 0 : 1;
#endif
}
