// Host-side upload/run library for the persistent interpreter. Plain CUDA
// runtime, no ggml dependency. Lifecycle:
//
//   mk::Host h;
//   mk::host_init(h, device, pass_cycles_cap);   // allocates the cells;
//                                                // h.d_token is now valid for
//                                                // program payloads
//   ... build the Instr array (payload pointers into buffers the caller
//       allocates on h.device, plus h.d_token for OP_EMBED_LOOKUP) ...
//   mk::host_upload(h, instrs, n_instr, epoch_stride);
//   mk::host_launch(h);                          // cooperative, 72x384,
//                                                // 60 KiB dynamic smem
//   for (...) mk::host_run_pass(h, token, timeout_ms);
//   mk::host_shutdown(h, timeout_ms);            // HALT doorbell + wait
//   mk::host_destroy(h);
//
// run_pass writes the token cell (cudaMemcpyAsync + stream sync), rings the
// doorbell, then polls the host-mapped done and error cells. On timeout the
// Y02 watchdog REPORTS (doorbell/done/err state, stream status) and returns
// RUN_TIMEOUT; it never kills anything — a wedged cooperative kernel is
// reclaimed by context teardown at process exit, and destroy() skips
// cudaFree while the kernel might still touch the buffers.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "isa.cuh"
#include "interp.cuh"

namespace mk {

enum RunStatus {
    RUN_OK      = 0,
    RUN_ERR     = 1, // device error cell set; details in h.h_err[0..1]
    RUN_TIMEOUT = 2, // watchdog fired; kernel state reported to stderr
};

struct Host {
    int device = -1;
    // device side
    Instr *d_program = nullptr;
    Program hdr{};
    Control ctl{};
    unsigned *d_cells = nullptr;         // y02 counter | doorbell_dev | err_dev
    int32_t *d_token = nullptr;          // per-pass token-id input cell
    long long *d_pass_cycles = nullptr;  // G15 ring (block-0 clock64 deltas)
    long long *d_pass_ns = nullptr;      // paired [start,end] %globaltimer ns ring
    unsigned pass_cycles_cap = 0;
    long long *d_op_cycles = nullptr;    // REDLINE: per-kind cycles (OP_KIND_COUNT)
    long long *d_op_tele = nullptr;      // plan/0144: per-op-instance {gs,ge,cyc}*n_instr
    unsigned  *d_smid_census = nullptr;  // plan/0144: %smid per block (GRID_BLOCKS)
    unsigned   op_tele_cap = 0;          // n_instr (op_tele ring capacity in ops)
    // host-mapped mailbox
    unsigned *h_mail = nullptr;
    volatile unsigned *h_doorbell = nullptr;
    volatile unsigned *h_done = nullptr;
    volatile unsigned *h_err = nullptr;  // [0] code, [1] aux
    // run state
    cudaStream_t kstream = nullptr;      // the persistent kernel lives here
    cudaStream_t cstream = nullptr;      // copies/readback (never the legacy
                                         // default stream: it would sync
                                         // against the persistent kernel)
    unsigned pass = 0;                   // last pass issued
    bool launched = false;
    bool smem_optin_done = false;        // dynamic-smem opt-in already set on this
                                         // device (host_launch then skips it; lets a
                                         // caller hoist the cuFuncSetAttribute out of
                                         // an ncu profiler range, which forbids it)
};

bool host_init(Host &h, int device, unsigned pass_cycles_cap);
bool host_upload(Host &h, const Instr *prog, uint32_t n_instr,
                 uint32_t epoch_stride);
// Set the 60 KiB dynamic-smem opt-in on mk_interp for h's current device context and
// mark it done, so a later host_launch skips the cuFuncSetAttribute. Call this BEFORE
// opening an ncu range (cuKernelSetAttribute is an unsupported API inside a range).
bool host_smem_optin(Host &h);
bool host_launch(Host &h);
RunStatus host_run_pass(Host &h, int32_t token, double timeout_ms);
// HALT doorbell + bounded wait for kernel exit; false (with a report) if the
// kernel did not exit — the watchdog path, never a kill.
bool host_shutdown(Host &h, double timeout_ms);
void host_destroy(Host &h);
// Read back up to `count` device-recorded per-pass cycle deltas, oldest slot
// first (ring order; caller indexes by (pass-1) % cap).
bool host_read_pass_cycles(Host &h, long long *out, unsigned count);
// Read back the paired %globaltimer ns ring: 2*count longs, slot i at
// out[2*i]=start, out[2*i+1]=end (same ring indexing as pass_cycles).
bool host_read_pass_ns(Host &h, long long *out, unsigned count);

// Per-pass on-device stats over the newest n_samples ring slots. mean/min/max
// come from the globaltimer ns ring (authoritative wall time); ghz is the
// run's measured SM clock (clock64 cycles / ns), reported so every number
// carries its clock domain instead of assuming a conversion constant.
struct PassStats {
    double mean_ms = 0, min_ms = 0, max_ms = 0, ghz = 0;
    unsigned cnt = 0;
    bool ok = false;
};
PassStats read_pass_stats(Host &h, long long n_samples);

// REDLINE itemization: zero the per-kind cycle accumulator (call between the
// warmup and timed regions), and read back OP_KIND_COUNT accumulated totals.
// Both are no-ops unless the kernel was built with -DMK_PROFILE (the array is
// still allocated so a profile build can write it).
bool host_reset_op_cycles(Host &h);
bool host_read_op_cycles(Host &h, long long *out, unsigned count);

// plan/0144 primitive telemetry readback (MK_PROFILE builds fill it; else zero).
// op_tele: `n_op*3` longs {gt_start_ns, gt_end_ns, cycles} per op (last timed pass).
// smid_census: `count` block->SM entries. Both no-op false if the buffer is absent.
bool host_read_op_tele(Host &h, long long *out, unsigned n_op);
bool host_read_smid_census(Host &h, unsigned *out, unsigned count);

} // namespace mk
