// See host.h for the contract. Plain CUDA runtime; the kernel is reached
// through its stub address (mk_interp_func) so this TU stays pure C++.
#include "host.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace mk {

static bool ck(cudaError_t e, const char *what) {
    if (e == cudaSuccess)
        return true;
    std::fprintf(stderr, "mk-host: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

bool host_init(Host &h, int device, unsigned pass_cycles_cap) {
    h = Host{};
    h.device = device;
    if (!ck(cudaSetDevice(device), "cudaSetDevice"))
        return false;

    cudaDeviceProp p{};
    if (!ck(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties"))
        return false;
    int coop = 0;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device);
    int smem_optin = 0;
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    if ((unsigned)p.multiProcessorCount < GRID_BLOCKS || !coop ||
        (unsigned)smem_optin < SMEM_BYTES) {
        std::fprintf(stderr,
                     "mk-host: dev %d cannot honor the G11 envelope "
                     "(sms %d coop %d smem_optin %d)\n",
                     device, p.multiProcessorCount, coop, smem_optin);
        return false;
    }

    if (!ck(cudaStreamCreateWithFlags(&h.kstream, cudaStreamNonBlocking),
            "kstream") ||
        !ck(cudaStreamCreateWithFlags(&h.cstream, cudaStreamNonBlocking),
            "cstream"))
        return false;

    // Device cells, one 128 B line each: y02 counter | doorbell_dev | err_dev.
    if (!ck(cudaMalloc(&h.d_cells, 3 * 128), "d_cells") ||
        !ck(cudaMemset(h.d_cells, 0, 3 * 128), "d_cells memset") ||
        !ck(cudaMalloc(&h.d_token, 128), "d_token") ||
        !ck(cudaMemset(h.d_token, 0, 128), "d_token memset"))
        return false;
    h.hdr.y02_counter = h.d_cells;
    h.ctl.doorbell_dev = h.d_cells + 32;
    h.ctl.err_dev = h.d_cells + 64;

    // Host-mapped mailbox: doorbell | done | err[2], one 64 B line apart.
    if (!ck(cudaHostAlloc(&h.h_mail, 256, cudaHostAllocMapped), "h_mail"))
        return false;
    std::memset(h.h_mail, 0, 256);
    h.h_doorbell = h.h_mail + 0;
    h.h_done = h.h_mail + 16;
    h.h_err = h.h_mail + 32;
    unsigned *dv = nullptr;
    if (!ck(cudaHostGetDevicePointer(&dv, h.h_mail, 0), "mail devptr"))
        return false;
    h.ctl.doorbell_host = dv + 0;
    h.ctl.done = dv + 16;
    h.ctl.err_host = dv + 32;

    h.pass_cycles_cap = pass_cycles_cap;
    if (pass_cycles_cap) {
        if (!ck(cudaMalloc(&h.d_pass_cycles, (size_t)pass_cycles_cap * 8),
                "d_pass_cycles"))
            return false;
        cudaMemset(h.d_pass_cycles, 0, (size_t)pass_cycles_cap * 8);
    }
    h.ctl.pass_cycles = h.d_pass_cycles;
    h.ctl.pass_cycles_cap = pass_cycles_cap;

    // REDLINE per-kind cycle accumulator (OP_KIND_COUNT longs). Always
    // allocated (tiny); only an MK_PROFILE-built kernel writes it.
    if (!ck(cudaMalloc(&h.d_op_cycles, (size_t)OP_KIND_COUNT * 8), "d_op_cycles"))
        return false;
    cudaMemset(h.d_op_cycles, 0, (size_t)OP_KIND_COUNT * 8);
    h.ctl.op_cycles = h.d_op_cycles;
    return true;
}

bool host_upload(Host &h, const Instr *prog, uint32_t n_instr,
                 uint32_t epoch_stride) {
    if (!ck(cudaMalloc(&h.d_program, (size_t)n_instr * sizeof(Instr)),
            "d_program") ||
        !ck(cudaMemcpy(h.d_program, prog, (size_t)n_instr * sizeof(Instr),
                       cudaMemcpyHostToDevice),
            "program upload"))
        return false;
    h.hdr.n_instr = n_instr;
    h.hdr.epoch_stride = epoch_stride;
    return true;
}

bool host_launch(Host &h) {
    const void *func = mk_interp_func();
    if (!ck(cudaFuncSetAttribute(func,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM_BYTES),
            "smem opt-in"))
        return false;

    const Instr *prog = h.d_program;
    void *args[] = {(void *)&prog, (void *)&h.hdr, (void *)&h.ctl};
    if (!ck(cudaLaunchCooperativeKernel(func, dim3(GRID_BLOCKS),
                                        dim3(BLOCK_THREADS), args, SMEM_BYTES,
                                        h.kstream),
            "cooperative launch"))
        return false;
    h.launched = true;
    h.pass = 0;
    return true;
}

static void watchdog_report(Host &h, const char *when) {
    const cudaError_t q = cudaStreamQuery(h.kstream);
    std::fprintf(stderr,
                 "mk-host WATCHDOG (%s): pass %u doorbell %u done %u err %u "
                 "aux %u kstream %s\n"
                 "  the kernel may be wedged in a boundary spin; not killing "
                 "anything — context teardown at process exit reclaims it\n",
                 when, h.pass, h.h_doorbell[0], h.h_done[0], h.h_err[0],
                 h.h_err[1],
                 q == cudaSuccess ? "idle (kernel exited)"
                 : q == cudaErrorNotReady ? "running"
                                          : cudaGetErrorString(q));
}

RunStatus host_run_pass(Host &h, int32_t token, double timeout_ms) {
    if (h.h_err[0] != 0) {
        std::fprintf(stderr, "mk-host: device error %u (aux %u) already set\n",
                     h.h_err[0], h.h_err[1]);
        return RUN_ERR;
    }
    h.pass += 1;

    // Per-pass inputs land in device L2 before the doorbell becomes visible.
    if (!ck(cudaMemcpyAsync(h.d_token, &token, sizeof(token),
                            cudaMemcpyHostToDevice, h.cstream),
            "token write") ||
        !ck(cudaStreamSynchronize(h.cstream), "token sync"))
        return RUN_TIMEOUT;
    std::atomic_thread_fence(std::memory_order_release);
    *h.h_doorbell = h.pass;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double, std::milli>(timeout_ms);
    for (;;) {
        const unsigned err = h.h_err[0];
        if (err != 0) {
            std::fprintf(stderr,
                         "mk-host: device error %u (aux %u) on pass %u\n", err,
                         h.h_err[1], h.pass);
            return RUN_ERR;
        }
        if (*h.h_done == h.pass)
            return RUN_OK;
        if (std::chrono::steady_clock::now() > deadline) {
            watchdog_report(h, "run_pass");
            return RUN_TIMEOUT;
        }
    }
}

bool host_shutdown(Host &h, double timeout_ms) {
    if (!h.launched)
        return true;
    std::atomic_thread_fence(std::memory_order_release);
    *h.h_doorbell = DOORBELL_HALT;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double, std::milli>(timeout_ms);
    for (;;) {
        const cudaError_t q = cudaStreamQuery(h.kstream);
        if (q == cudaSuccess) {
            h.launched = false;
            return true;
        }
        if (q != cudaErrorNotReady)
            return ck(q, "kstream after halt");
        if (std::chrono::steady_clock::now() > deadline) {
            watchdog_report(h, "shutdown");
            return false;
        }
    }
}

void host_destroy(Host &h) {
    if (h.launched) {
        // A live (possibly wedged) kernel may still touch these buffers:
        // freeing them would fault the device. Leak; teardown reclaims.
        std::fprintf(stderr,
                     "mk-host: destroy with kernel still resident — skipping "
                     "device frees\n");
        return;
    }
    cudaFree(h.d_program);
    cudaFree(h.d_cells);
    cudaFree(h.d_token);
    cudaFree(h.d_pass_cycles);
    cudaFree(h.d_op_cycles);
    cudaFreeHost(h.h_mail);
    if (h.kstream)
        cudaStreamDestroy(h.kstream);
    if (h.cstream)
        cudaStreamDestroy(h.cstream);
    h = Host{};
    h.device = -1;
}

bool host_read_pass_cycles(Host &h, long long *out, unsigned count) {
    if (!h.d_pass_cycles || count > h.pass_cycles_cap)
        return false;
    return ck(cudaMemcpyAsync(out, h.d_pass_cycles, (size_t)count * 8,
                              cudaMemcpyDeviceToHost, h.cstream),
              "pass_cycles read") &&
           ck(cudaStreamSynchronize(h.cstream), "pass_cycles sync");
}

bool host_reset_op_cycles(Host &h) {
    if (!h.d_op_cycles)
        return false;
    return ck(cudaMemsetAsync(h.d_op_cycles, 0, (size_t)OP_KIND_COUNT * 8,
                              h.cstream),
              "op_cycles reset") &&
           ck(cudaStreamSynchronize(h.cstream), "op_cycles reset sync");
}

bool host_read_op_cycles(Host &h, long long *out, unsigned count) {
    if (!h.d_op_cycles || count > (unsigned)OP_KIND_COUNT)
        return false;
    return ck(cudaMemcpyAsync(out, h.d_op_cycles, (size_t)count * 8,
                              cudaMemcpyDeviceToHost, h.cstream),
              "op_cycles read") &&
           ck(cudaStreamSynchronize(h.cstream), "op_cycles sync");
}

} // namespace mk
