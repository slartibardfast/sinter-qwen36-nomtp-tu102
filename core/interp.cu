// The persistent interpreter spine. One cooperative launch of 72x384 with a
// 60 KiB dynamic smem slab; loops over host-signalled passes; per pass,
// walks the flat instruction array, crossing the Y02 boundary at every
// OP_BOUNDARY and dispatching every other instruction whose
// [block_lo, block_hi) range contains this block. Protocol details (pass
// doorbell, halt, error discipline) are documented in core/interp.cuh.
#include "interp.cuh"
#include "sync.cuh"
#include "../k0/ops/registry.cuh"

namespace mk {

// Error cells, published for the ops at kernel start (each block writes the
// same values; the duplicate plain stores are benign).
static __device__ unsigned *g_err_dev;
static __device__ volatile unsigned *g_err_host;

__device__ void raise_error(unsigned code, unsigned aux) {
    if (g_err_host) {
        g_err_host[1] = aux;
        st_release_sys(const_cast<unsigned *>(&g_err_host[0]), code);
    }
    if (g_err_dev)
        st_release_gpu(g_err_dev, code);
}

// Doorbell poll: system-scope strong load (the producer is the host CPU).
__device__ __forceinline__ unsigned ld_acquire_sys(const volatile unsigned *p) {
    unsigned v;
    asm volatile("ld.acquire.sys.u32 %0, [%1];"
                 : "=r"(v) : "l"(const_cast<const unsigned *>(p)) : "memory");
    return v;
}

} // namespace mk

extern __shared__ char mk_smem[];

__global__ void __launch_bounds__(mk::BLOCK_THREADS)
mk_interp(const mk::Instr *program, mk::Program hdr, mk::Control ctl)
{
    using namespace mk;

    if (threadIdx.x == 0) {
        g_err_dev = ctl.err_dev;
        g_err_host = ctl.err_host;
    }
    __syncthreads();

    GridBoundary bar{hdr.y02_counter};
    unsigned epoch = 0; // per-thread, monotonic across passes, never reset
    // Slab word 0 carries the interpreter's block broadcasts (doorbell value,
    // pass-end error sample). Ops own the slab while they run; the
    // interpreter only touches it between ops.
    unsigned *slab_word = reinterpret_cast<unsigned *>(mk_smem);

    for (unsigned pass = 1;; ++pass) {
        // ---- pass doorbell ------------------------------------------------
        if (threadIdx.x == 0) {
            unsigned v;
            if (blockIdx.x == 0) {
                do {
                    v = ld_acquire_sys(ctl.doorbell_host);
                } while (v != DOORBELL_HALT && (int)(v - pass) < 0);
                st_release_gpu(ctl.doorbell_dev, v); // off the PCIe path
            } else {
                do {
                    v = ld_acquire_gpu(ctl.doorbell_dev);
                } while (v != DOORBELL_HALT && (int)(v - pass) < 0);
            }
            *slab_word = v;
        }
        __syncthreads();
        const unsigned bell = *slab_word;
        __syncthreads(); // everyone has read before ops reuse the slab
        if (bell == DOORBELL_HALT)
            return;

        long long t0 = 0;
        if (blockIdx.x == 0 && threadIdx.x == 0 && ctl.pass_cycles)
            t0 = clock64();

        // ---- the instruction loop -----------------------------------------
        for (uint32_t i = 0; i < hdr.n_instr; ++i) {
            const Instr &in = program[i]; // immutable: plain cached loads
            const uint16_t kind = in.kind;
            if (kind == OP_BOUNDARY) {
                bar.cross(epoch);
                continue;
            }
            if (kind >= OP_KIND_COUNT || !op_wired(kind)) {
                // Uniform detection: every block decodes every header, so
                // every block breaks at this same instruction and the grid
                // exits with no boundary left half-arrived.
                if (blockIdx.x == 0 && threadIdx.x == 0)
                    raise_error(kind >= OP_KIND_COUNT ? ERR_BAD_KIND
                                                      : ERR_UNWIRED_KIND, i);
                return;
            }
            if (blockIdx.x >= in.block_lo && blockIdx.x < in.block_hi)
                op_dispatch(in, mk_smem);
        }

        // ---- pass epilogue ------------------------------------------------
        // One boundary so block 0 knows every block finished (and their
        // stores are release-ordered before the done flag), and so the
        // pass-end error sample below is uniform: any raise happens before
        // the raiser's arrival here, hence is visible to all after it.
        bar.cross(epoch);
        if (threadIdx.x == 0)
            *slab_word = ld_cg(ctl.err_dev);
        __syncthreads();
        const unsigned err = *slab_word;
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (ctl.pass_cycles)
                ctl.pass_cycles[(pass - 1) % ctl.pass_cycles_cap] =
                    clock64() - t0;
            if (!err)
                st_release_sys(const_cast<unsigned *>(ctl.done), pass);
        }
        if (err)
            return; // host reads the mirrored err cell; done stays unset
    }
}

extern "C" const void *mk_interp_func() {
    return reinterpret_cast<const void *>(&mk_interp);
}
