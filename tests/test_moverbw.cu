// Reduced-mover-thread NO_MATH bandwidth spike (plan/0143, de-risk for the FATTN
// smem double-buffer, call/0033). The double-buffer runs the K/V tile loads on
// the MOVER warps only (fewer than all 384 threads) while the compute warps run
// HMMA. If DRAM bandwidth collapses below ~505 GB/s at the reduced mover-thread
// count, the memory lane grows and the overlap prize (calx-mill OpComposition
// max(mem,compute)) shrinks. This microbench isolates the pure load: it replays
// mk_load_full_row's exact ld_cg uint4 pattern (attn.cuh:314) restricted to the
// first `movers` threads, NO math, DCE-guarded (XOR accumulator + a per-tile smem
// read), and sweeps movers over the warp counts a real split would use.
//
// Layout matches the deep dual-GPU FATTN op: row_width = n_kv_heads(2) * HD(256)
// = 512 f16, tile 32 rows, pitch rowp = row_width + 2, 72 chunks (1/SM), clen
// 2048 (64 tiles/chunk). Contended-indicative (shared rig, clocks unlocked): a
// lower bound. Verify by: bandwidth at 256/192 movers stays within a few % of
// the 384-thread number -> the double-buffer load will hold the bus (GO); a
// steep drop -> the mover split starves the bus (the memory lane must be
// re-costed in the composition before the attended build).

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../core/isa.cuh"
#include "../core/sync.cuh"   // mk::ld_cg(uint4)
#include "../core/interp.cuh" // SMEM_BYTES (attn.cuh's co-residency guard reads it)
#include "../k0/ops/attn.cuh" // mk::MK_ATTN_HD (head dim; layout must match the op)

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,       \
                   cudaGetErrorString(err_));                                       \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static constexpr int   NTHR       = 384;                 // resident block width
static constexpr int   TILE       = 32;                  // MK_FATTN_TILE
static constexpr uint32_t ROW_WIDTH = 2u * mk::MK_ATTN_HD; // dual-GPU: 2 kv heads * 256
static constexpr uint32_t ROWP    = ROW_WIDTH + 2u;      // op tile pitch
static constexpr uint32_t CLEN    = 2048u;               // rows/chunk (64 tiles)
static constexpr int   NCHUNKS    = 72;                  // one chunk per TU102 SM

// One block streams its chunk's K then V tile-by-tile into smem, loads issued by
// the first `movers` threads only (stride = movers, so they cover every element).
__global__ void mover_bw_kernel(const __half *kc, const __half *vc,
                                uint32_t clen, uint32_t row_width, uint32_t rowp,
                                int movers, unsigned *sink) {
    extern __shared__ char smem[];
    __half *tile = reinterpret_cast<__half *>(smem);
    const uint32_t nv4 = row_width / 8;               // uint4 (8 f16) per full row
    const size_t base = (size_t) blockIdx.x * clen * row_width;
    unsigned s = 0;
    for (uint32_t t0 = 0; t0 < clen; t0 += TILE) {
        if ((int) threadIdx.x < movers) {
            for (uint32_t i = threadIdx.x; i < (uint32_t) TILE * nv4; i += (uint32_t) movers) {
                const uint32_t t = i / nv4, c = i % nv4;
                uint4 kv = mk::ld_cg(reinterpret_cast<const uint4 *>(
                                         kc + base + (size_t)(t0 + t) * row_width) + c);
                s ^= kv.x ^ kv.y ^ kv.z ^ kv.w;
                unsigned *d = reinterpret_cast<unsigned *>(tile + (size_t) t * rowp + c * 8);
                d[0] = kv.x; d[1] = kv.y; d[2] = kv.z; d[3] = kv.w;
            }
        }
        __syncthreads();
        s ^= reinterpret_cast<unsigned *>(tile)[threadIdx.x]; // force the smem writes
        __syncthreads();
        if ((int) threadIdx.x < movers) {
            for (uint32_t i = threadIdx.x; i < (uint32_t) TILE * nv4; i += (uint32_t) movers) {
                const uint32_t t = i / nv4, c = i % nv4;
                uint4 kv = mk::ld_cg(reinterpret_cast<const uint4 *>(
                                         vc + base + (size_t)(t0 + t) * row_width) + c);
                s ^= kv.x ^ kv.y ^ kv.z ^ kv.w;
                unsigned *d = reinterpret_cast<unsigned *>(tile + (size_t) t * rowp + c * 8);
                d[0] = kv.x; d[1] = kv.y; d[2] = kv.z; d[3] = kv.w;
            }
        }
        __syncthreads();
        s ^= reinterpret_cast<unsigned *>(tile)[threadIdx.x];
        __syncthreads();
    }
    if (s == 0xFFFFFFFFu) sink[blockIdx.x] = s; // improbable: keeps s (all loads) live
}

int main() {
    const size_t elems = (size_t) NCHUNKS * CLEN * ROW_WIDTH;
    __half *d_kc, *d_vc; unsigned *d_sink;
    CUDA_CHECK(cudaMalloc(&d_kc, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vc, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_sink, NCHUNKS * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_kc, 0x3c, elems * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_vc, 0x3c, elems * sizeof(__half)));

    const unsigned smem = (unsigned) TILE * ROWP * sizeof(__half);
    const double kv_bytes = 2.0 * (double) elems * sizeof(__half); // K + V
    const int iters = 50;

    printf("mover-thread bandwidth sweep: %d chunks x %u rows x %u f16 (%.1f MB K+V/pass), "
           "smem %u B, %d resident threads/block\n",
           NCHUNKS, CLEN, ROW_WIDTH, kv_bytes / 1e6, smem, NTHR);
    printf("  %8s %6s %10s %8s\n", "movers", "warps", "ms/pass", "GB/s");

    const int sweep[] = {384, 320, 256, 192, 128, 96, 64, 32};
    double bw384 = 0.0;
    for (int movers : sweep) {
        for (int i = 0; i < 5; i++)
            mover_bw_kernel<<<NCHUNKS, NTHR, smem>>>(d_kc, d_vc, CLEN, ROW_WIDTH, ROWP, movers, d_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++)
            mover_bw_kernel<<<NCHUNKS, NTHR, smem>>>(d_kc, d_vc, CLEN, ROW_WIDTH, ROWP, movers, d_sink);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        const double gbps = kv_bytes / (ms / iters / 1e3) / 1e9;
        if (movers == 384) bw384 = gbps;
        printf("  %8d %6.1f %10.4f %8.0f  (%.0f%% of 384-thread)\n",
               movers, movers / 32.0, ms / iters, gbps, bw384 > 0 ? 100.0 * gbps / bw384 : 100.0);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }

    CUDA_CHECK(cudaFree(d_kc)); CUDA_CHECK(cudaFree(d_vc)); CUDA_CHECK(cudaFree(d_sink));
    return 0;
}
