// Config-invariant sync primitives: the Y02 interpreter boundary and the
// load/store discipline every instantiation inherits.
//
// Measured sm_75 semantics (reference/tu102, sync_protocol.csv):
//   - ld.acquire lowers to a bare STRONG load; ALL ordering rides the
//     producer-side release MEMBAR. An acquire read is only as fresh as the
//     scope of the load itself.
//   - Per-SM L1 is NOT coherent. A plain LDG may return a stale line the SM
//     read in an earlier pass. Stock graphs never see this because kernel
//     LAUNCH boundaries invalidate L1; a persistent kernel loses that
//     implicit fence. Rule: every MUTABLE datum read across a Y02/Y06
//     boundary (activations, KV, state, flags) uses a strong or .cg load.
//     Immutable weights may use plain loads — a stale line of an unchanged
//     value is not staleness.
//   - One global counter is contention-flat on this chip (300-312 cycles,
//     2..32 contenders), which is what licenses a single grid counter for
//     72 SMs (Y02 basis row); the boundary costs ~0.4 us against grid.sync's
//     measured 825 ns.
#pragma once
#include <cuda_runtime.h>

namespace mk {

__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_gpu(unsigned *p, unsigned v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_sys(unsigned *p, unsigned v) {
    asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// Release-ordered arrive: one MEMBAR carries every prior store, the RED
// counts us in without a return trip.
__device__ __forceinline__ void red_add_release_gpu(unsigned *p, unsigned v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void membar_gl()  { asm volatile("membar.gl;" ::: "memory"); }
__device__ __forceinline__ void membar_sys() { asm volatile("membar.sys;" ::: "memory"); }

// Strong payload reads (the .cg family): bypass the incoherent L1.
__device__ __forceinline__ unsigned ld_cg(const unsigned *p) {
    unsigned v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_cg(const uint4 *p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ld_cg(const float *p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float4 ld_cg(const float4 *p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
    return v;
}

// The Y02 interpreter instruction boundary. Monotonic arrival counter,
// wrap-safe target compare, no reset (sense lives in the target): each
// crossing, every block release-arrives then a single elected thread spins
// on the counter reaching epoch * gridDim.x. Producer stores issued before
// arrive() are visible to every consumer load issued after wait() —
// provided the consumer load obeys the strong-read rule above.
struct GridBoundary {
    unsigned *counter;   // one u32 in device DRAM, zero-initialized

    // Call with the whole CTA. 'epoch' is per-thread local state that must
    // start at 0 and be carried across calls.
    __device__ __forceinline__ void cross(unsigned &epoch) {
        __syncthreads();
        epoch += 1;
        if (threadIdx.x == 0) {
            red_add_release_gpu(counter, 1u);
            const unsigned target = epoch * gridDim.x;
            while ((int)(ld_acquire_gpu(counter) - target) < 0) { }
        }
        __syncthreads();
    }
};

} // namespace mk
