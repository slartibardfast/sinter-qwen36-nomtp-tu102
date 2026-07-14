// Cross-GPU exchange ops (the Y06 mirrored fp32 allreduce). DUAL-GPU PREP —
// authored against core/EXCHANGE-DESIGN.md; NOT yet wired into
// k0/ops/registry.cuh or the build. It is wired + litmus-validated + packed
// when the dual-GPU harness lands (roadmap step 2); until then it is a
// standalone header that compiles but is never dispatched, so it cannot
// disturb the single-GPU bring-up.
//
// Replaces the fork's pinned-host PCIe AllReduce (the ggml_cuda_ar_kernel
// census op; call/0021's one declared divergence) with an NVLink peer
// exchange. Every ordering choice below is bound to a measured
// reference/tu102 sync_protocol row; do not weaken without re-running the
// cross-GPU litmus (tests/litmus_xchg.cu, pending).
//
// Protocol per site, symmetric on both GPUs (EXCHANGE-DESIGN.md):
//   OP_XCHG_PUSH   : store this GPU's partial slice into the PEER's inbox
//                    payload with line-filling float4 stores (the measured
//                    2.4-2.5x visibility lever), then membar.sys.
//   <OP_BOUNDARY>  : all blocks' peer stores complete before any seqno.
//   OP_XCHG_REDUCE : elected block release.sys-stores the seqno into the
//                    PEER's inbox (payload-ready signal); every block polls
//                    its OWN inbox seqno (acquire), reads the received
//                    payload STRONG (.cg — never a plain LDG, the measured
//                    xgpu stale-L1 hazard), and writes the mirrored sum in a
//                    FIXED gpu-index fold order so both GPUs produce
//                    bit-identical activations.
#pragma once
#include <cuda_runtime.h>
#include "../../core/isa.cuh"
#include "../../core/sync.cuh"

namespace mk {

// ld.acquire.sys: the seqno producer is the peer GPU (system scope), so the
// consumer poll must be a system-scoped acquire, not the .gpu form.
__device__ __forceinline__ unsigned xchg_ld_acquire_sys(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// One exchange site's mailbox pointers, resolved at pack time to UVA
// pointers (peer access enabled both ways by the dual-GPU harness).
struct XchgPushArgs {
    const float *local_partial; // this GPU's computed slice (device-local)
    float *peer_payload;        // the PEER inbox payload (peer VRAM, UVA)
    int n_elems;                // slice length (e.g. 5120)
};
static_assert(sizeof(XchgPushArgs) <= sizeof(((Instr *)0)->payload),
              "XchgPushArgs exceeds Instr payload");

struct XchgReduceArgs {
    const float *local_partial; // this GPU's slice (the same buffer PUSH sent)
    const float *my_payload;    // this GPU's inbox payload (peer wrote it)
    float *out;                 // mirrored result (device-local)
    unsigned *peer_seqno;       // PEER inbox seqno to publish into (UVA)
    const unsigned *my_seqno;   // this GPU's inbox seqno to poll (device-local)
    int n_elems;
    unsigned seqno;             // monotonic per site per pass (wrap-safe)
    int gpu_index;              // 0 or 1 — selects the fixed fold order
};
static_assert(sizeof(XchgReduceArgs) <= sizeof(((Instr *)0)->payload),
              "XchgReduceArgs exceeds Instr payload");

// Vectorized peer store: float4 in linear order (line-filling v4). The store
// pattern is load-bearing (measured 2.4-2.5x visibility vs strided), not
// stylistic. Tail handled scalar.
static __device__ MK_OPFN void op_xchg_push(const Instr &in, char *) {
    XchgPushArgs a;
    __builtin_memcpy(&a, in.payload, sizeof(a));

    const int lanes = (in.block_hi - in.block_lo);
    const int rank  = (int)blockIdx.x - (int)in.block_lo;
    const int tid   = rank * (int)blockDim.x + (int)threadIdx.x;
    const int nthr  = lanes * (int)blockDim.x;

    const int n4 = a.n_elems >> 2;
    const float4 *src4 = reinterpret_cast<const float4 *>(a.local_partial);
    float4 *dst4       = reinterpret_cast<float4 *>(a.peer_payload);
    for (int i = tid; i < n4; i += nthr)
        dst4[i] = src4[i];                      // peer store, linear order
    for (int i = (n4 << 2) + tid; i < a.n_elems; i += nthr)
        a.peer_payload[i] = a.local_partial[i]; // scalar tail

    // Each pushing block sys-orders ITS OWN peer stores: the Y02 boundary's
    // .gpu-scope release does not order sys-destined stores for a sys
    // observer (the peer GPU). The following OP_BOUNDARY then makes every
    // block's push globally complete before any seqno is published.
    membar_sys();
}

// Signal + consume + fold. Runs after the boundary that followed PUSH.
static __device__ MK_OPFN void op_xchg_reduce(const Instr &in, char *) {
    XchgReduceArgs a;
    __builtin_memcpy(&a, in.payload, sizeof(a));

    const int lanes = (in.block_hi - in.block_lo);
    const int rank  = (int)blockIdx.x - (int)in.block_lo;
    const int tid   = rank * (int)blockDim.x + (int)threadIdx.x;
    const int nthr  = lanes * (int)blockDim.x;

    // Publish "my payload is fully in your inbox" to the peer. One elected
    // thread of the whole participating grid; release.sys carries the peer
    // stores (already sys-fenced by PUSH + the boundary) ahead of the flag.
    if (rank == 0 && threadIdx.x == 0)
        st_release_sys(a.peer_seqno, a.seqno);

    // Every block waits until the peer's matching payload has landed in OUR
    // inbox (acquire on the local seqno copy), then reads it STRONG.
    if (threadIdx.x == 0)
        while ((int)(xchg_ld_acquire_sys(a.my_seqno) - a.seqno) < 0) { }
    __syncthreads();

    // Fixed fold order, identical arithmetic on both GPUs: out = p0 + p1
    // where p0 is GPU0's partial and p1 is GPU1's. This GPU owns
    // local_partial; the peer's slice is in my_payload. gpu_index picks which
    // is the left addend so the summation order matches bit-for-bit.
    for (int i = tid; i < a.n_elems; i += nthr) {
        const float mine = a.local_partial[i];
        const float peer = ld_cg(a.my_payload + i);   // strong, never plain
        a.out[i] = (a.gpu_index == 0) ? (mine + peer) : (peer + mine);
    }
}

} // namespace mk
