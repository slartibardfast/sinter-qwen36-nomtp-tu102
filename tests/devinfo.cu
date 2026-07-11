// Rig precheck. The persistent kernel refuses to spin up on a substrate that
// cannot honor the G11 envelope; this probe reports the facts that decision
// reads: SM count, cooperative-launch support, smem-per-block opt-in ceiling,
// registers per SM, and peer access between the two devices.
#include <cstdio>
#include <cuda_runtime.h>

static int fail(const char *what, cudaError_t e) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return 2;
}

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return fail("cudaGetDeviceCount", e);
    std::printf("devices %d\n", n);
    for (int d = 0; d < n; d++) {
        cudaDeviceProp p{};
        if ((e = cudaGetDeviceProperties(&p, d)) != cudaSuccess)
            return fail("cudaGetDeviceProperties", e);
        int coop = 0;
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, d);
        int smem_optin = 0;
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, d);
        std::printf("dev %d: %s sm_%d%d sms %d coop %d smem_optin %d regs/sm %d\n",
                    d, p.name, p.major, p.minor, p.multiProcessorCount, coop,
                    smem_optin, p.regsPerMultiprocessor);
    }
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            if (a != b) {
                int peer = 0;
                cudaDeviceCanAccessPeer(&peer, a, b);
                std::printf("peer %d->%d %d\n", a, b, peer);
            }
    return 0;
}
