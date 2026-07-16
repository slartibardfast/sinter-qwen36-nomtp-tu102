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
#ifdef MK_PROFILE
    if (blockIdx.x == 0 && threadIdx.x == 0)
        g_fattn_phase = ctl.op_cycles ? ctl.op_cycles + OP_KIND_COUNT : nullptr;
    // Residency census (plan/0144): each block records its SM once, so the ingest
    // can confirm 72 distinct SMs (co-residence). One store per block, at entry.
    if (threadIdx.x == 0 && ctl.smid_census) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        ctl.smid_census[blockIdx.x] = smid;
    }
#endif
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

#ifdef MK_PROFILE
        // REDLINE itemization: block-0 thread-0 attributes clock64 cycles per
        // op kind (and the boundary-wait total to op_cycles[OP_BOUNDARY]).
        // split_grid balances every antichain window by est, so block 0's
        // first-op time approximates that window's critical path.
        const bool prof = (blockIdx.x == 0 && threadIdx.x == 0 && ctl.op_cycles);
#endif

        // ---- the instruction loop -----------------------------------------
#ifndef MK_SPECIALIZED
        for (uint32_t i = 0; i < hdr.n_instr; ++i) {
            const Instr &in = program[i]; // immutable: plain cached loads
            const uint16_t kind = in.kind;
            if (kind == OP_BOUNDARY) {
#ifdef MK_PROFILE
                long long tb = prof ? clock64() : 0;
                bar.cross(epoch);
                if (prof) ctl.op_cycles[OP_BOUNDARY] += clock64() - tb;
#else
                bar.cross(epoch);
#endif
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
            if (blockIdx.x >= in.block_lo && blockIdx.x < in.block_hi) {
#ifdef MK_PROFILE
                const bool tele = prof && ctl.op_tele && i < ctl.op_tele_cap;
                long long gs = 0;
                long long to = prof ? clock64() : 0;
                // gt_start read just before the op; the clock64 span `to..c` does not
                // include the ring store below (which happens after gt_end).
                if (tele) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gs));
                op_dispatch(in, mk_smem);
                if (prof) {
                    long long c = clock64() - to;
                    ctl.op_cycles[kind] += c;
                    if (tele) {
                        long long ge;
                        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ge));
                        long long *e = ctl.op_tele + (size_t)i * 3;
                        e[0] = gs; e[1] = ge; e[2] = c;   // write AFTER ge: span-clean
                    }
                }
#else
                op_dispatch(in, mk_smem);
#endif
            }
        }
#else
        // GENERATED straight-line dispatch (k0/specialize.py, call/0023 D2):
        // the 27-way switch, the kind-validity guard, and the per-instruction
        // header decode vanish by construction; each op is a direct call and
        // block ranges are compile-time literals. Fingerprint-matched programs
        // only (miss -> the interpreter above). Spike 3: ~0.76 ms/pass reclaimed.
        (void)hdr; (void)program;
#include "../k0/mk_specialized_body.inc"
#endif

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
