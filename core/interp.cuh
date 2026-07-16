// The interpreter spine's launch envelope, control block, and error codes.
// Host-safe header: core/host.h includes it from plain C++; the device-only
// declarations are guarded behind __CUDACC__.
//
// Pass protocol (documented here, implemented in interp.cu / host.cpp):
//   - The kernel is launched once (cooperative, 72x384, 60 KiB dynamic smem)
//     and loops over passes. At the top of pass N every block waits on a
//     doorbell: the host writes N to a host-mapped u32 (doorbell_host);
//     block 0 polls it over PCIe and forwards the value to a device cell
//     (doorbell_dev) that the other 71 blocks spin on, keeping all but one
//     spinner on contention-flat device DRAM (Y02 basis row).
//   - Shutdown is the host-written stop value DOORBELL_HALT (0xFFFFFFFF) on
//     the same doorbell; every block observes it at the pass top and
//     returns. There is no OP_HALT instruction.
//   - Pass completion: after the instruction loop the kernel crosses one
//     epilogue boundary (so block 0 knows every block finished and their
//     stores are release-ordered), then block 0 st.release.sys's the pass
//     number to the host-mapped done cell.
//   - Per-pass inputs (the token-id cell) are written by the host with
//     cudaMemcpyAsync + stream sync BEFORE the doorbell store; the DMA lands
//     in device L2 before the doorbell is visible, and ops read such
//     mutable cells with .cg loads, so the value is fresh.
//
// Error protocol:
//   - A kind that is >= OP_KIND_COUNT or not wired in k0/ops/registry.cuh is
//     detected by EVERY block (each block decodes every instruction header,
//     participating or not), so all blocks break at the same instruction:
//     no boundary is left half-arrived, the grid exits, and the host sees
//     the error cell. An unknown kind is never silent.
//   - Ops raise internal errors with raise_error() and RETURN (they must not
//     hang or exit early relative to other blocks); the interpreter samples
//     the device error cell once per pass, after the epilogue boundary,
//     where release-arrive ordering makes the sample uniform across blocks.
//     The done cell is not written for an errored pass.
#pragma once
#include <cstdint>
#include "isa.cuh"

namespace mk {

// The G11 envelope (README): one block per TU102 SM.
constexpr unsigned GRID_BLOCKS   = 72;
constexpr unsigned BLOCK_THREADS = 384;
constexpr unsigned SMEM_BYTES    = 60u * 1024u;

// Host-written doorbell stop value; pass numbers count 1,2,3,...
constexpr unsigned DOORBELL_HALT = 0xFFFFFFFFu;

// Error cell [0] = code, [1] = aux (interpreter: instruction index;
// ops: their Instr::dbg_node).
enum ErrCode : unsigned {
    ERR_NONE                   = 0,
    ERR_BAD_KIND               = 1, // kind >= OP_KIND_COUNT
    ERR_UNWIRED_KIND           = 2, // valid kind, no case in the registry yet
    ERR_LOGITS_FLAG_MULTIBLOCK = 3, // OP_LOGITS_EMIT flag needs a 1-block range
};

// Everything the kernel needs beyond the program itself. All pointers are
// device-visible (the *_host cells are cudaHostAllocMapped device aliases).
struct Control {
    const volatile unsigned *doorbell_host; // host-mapped: pass seq / HALT
    unsigned *doorbell_dev;                 // device forward cell (block 0)
    volatile unsigned *done;                // host-mapped: completed pass seq
    unsigned *err_dev;                      // device error code cell
    volatile unsigned *err_host;            // host-mapped: [0] code, [1] aux
    long long *pass_cycles;                 // device ring: per-pass clock64
                                            // deltas from block 0 (G15);
                                            // nullptr disables recording
    unsigned pass_cycles_cap;               // ring capacity in entries
    // REDLINE itemization (MK_PROFILE builds only): block-0 clock64 cycles
    // accumulated per macro-op kind across passes; op_cycles[OP_BOUNDARY]
    // carries the boundary-wait total. OP_KIND_COUNT entries, nullptr or a
    // non-MK_PROFILE kernel leaves it untouched (production is unaffected).
    long long *op_cycles;
    // Primitive telemetry (MK_PROFILE builds only; plan/0144 seam). Per-op-instance
    // timing ring: 3 longs/op {gt_start_ns, gt_end_ns, cycles}, block-0, last-pass-
    // wins (op_tele_cap = n_instr entries). Residency census: block b writes its
    // %smid once (GRID_BLOCKS entries). Both nullptr disables; the %globaltimer/
    // %smid reads and the ring write live under MK_PROFILE, so production is
    // untouched. The ring write happens AFTER gt_end, so a measured span never
    // includes its own store.
    long long *op_tele;       // n_instr*3 longs, nullptr disables
    unsigned  *smid_census;   // GRID_BLOCKS entries, nullptr disables
    unsigned   op_tele_cap;   // op_tele ring capacity in ops (bounds the write)
};

} // namespace mk

#if defined(__CUDACC__)
namespace mk {
// Any thread of any block may call; duplicate raisers are benign (all codes
// are fatal, any one reaching the host suffices). Callers must return
// normally afterwards so every block still reaches the next boundary.
__device__ void raise_error(unsigned code, unsigned aux);
} // namespace mk

__global__ void __launch_bounds__(mk::BLOCK_THREADS)
mk_interp(const mk::Instr *program, mk::Program hdr, mk::Control ctl);
#endif

// The kernel's host stub address, for cudaFuncSetAttribute /
// cudaLaunchCooperativeKernel from plain C++ TUs (core/host.cpp).
extern "C" const void *mk_interp_func();
