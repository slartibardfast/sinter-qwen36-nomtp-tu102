// R2 probe (plan/0139): does the raw per-GPU pointer extraction work?
//
// The one ABI-fragile piece of the meta-forwarding integration is reaching the
// per-GPU device pointers behind a meta tensor. This probe validates the
// no-fork-change path: mirror ggml_backend_meta_buffer_context far enough to
// read `simple_tensors`, pull each simple tensor's ->data, and confirm with
// cudaPointerGetAttributes that it is a real device pointer on the expected GPU
// with the expected split shape. If this holds, R3 forwarding can extract
// pointers with no fork edit; if not, we one-line-export the static accessor.
//
// Minimal footprint: two small tensors, not the model. Build/run notes at EOF.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"  // meta accessors + full ggml_backend_buffer struct
#include "ggml-alloc.h"
#include <cuda_runtime.h>

#include <map>
#include <vector>
#include <cstdio>
#include <cstring>

// --- the mirror -------------------------------------------------------------
// Only the offset of `simple_tensors` matters, and sizeof(std::map) is
// independent of its key/value types, so the first member is a placeholder map
// of the correct KIND. simple_tensors is typed exactly for the access.
namespace {
struct meta_buffer_context_mirror {
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<int, int>> split_state_cache;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>>           simple_tensors;
};

// A trivial split callback: split axis 1 (rows) 50/50 across the two devices,
// enough to force per-GPU simple tensors. (R3 uses the real Qwen3.6 geometry.)
ggml_backend_meta_split_state probe_split(const ggml_tensor * t, void * /*ud*/) {
    ggml_backend_meta_split_state s;
    memset(&s, 0, sizeof(s));
    const int64_t n = t->ne[1];
    s.axis = GGML_BACKEND_SPLIT_AXIS_1;
    s.n_segments = 1;
    s.ne[0] = n / 2;          // seg0 dev0
    s.ne[1] = n - n / 2;      // seg0 dev1
    return s;
}
} // namespace

int main() {
    ggml_backend_load_all();  // loads libggml-cuda.so from the backend dir

    ggml_backend_dev_t cuda[2] = {
        ggml_backend_dev_by_name("CUDA0"),
        ggml_backend_dev_by_name("CUDA1"),
    };
    if (!cuda[0] || !cuda[1]) { fprintf(stderr, "PROBE FAIL: CUDA0/CUDA1 not found\n"); return 2; }

    ggml_backend_dev_t meta = ggml_backend_meta_device(cuda, 2, probe_split, nullptr);
    if (!meta) { fprintf(stderr, "PROBE FAIL: meta device null\n"); return 2; }
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta);

    // Two test tensors named as the split table would classify them.
    ggml_init_params ip = { /*mem_size*/ 16 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * qkv  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 10240); // rows split
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    ggml_tensor * ffn  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 17408);
    ggml_set_name(ffn, "blk.0.ffn_up.weight");

    ggml_backend_buffer_t buf = ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx, meta_buft);
    if (!buf) { fprintf(stderr, "PROBE FAIL: meta alloc null\n"); return 2; }
    if (!ggml_backend_buffer_is_meta(buf)) { fprintf(stderr, "PROBE FAIL: buffer not meta\n"); return 2; }

    int fails = 0;
    ggml_tensor * probes[2] = { qkv, ffn };
    for (ggml_tensor * t : probes) {
        auto * mc = (meta_buffer_context_mirror *) t->buffer->context;
        auto it = mc->simple_tensors.find(t);
        if (it == mc->simple_tensors.end()) { fprintf(stderr, "PROBE FAIL: %s not in simple_tensors\n", t->name); fails++; continue; }
        const std::vector<ggml_tensor *> & simples = it->second;
        printf("== %s : ne=[%lld,%lld], %zu simple tensors ==\n",
               t->name, (long long)t->ne[0], (long long)t->ne[1], simples.size());
        for (size_t j = 0; j < simples.size(); ++j) {
            ggml_tensor * st = simples[j];
            void * p = st ? st->data : nullptr;
            cudaPointerAttributes at;
            cudaError_t e = cudaPointerGetAttributes(&at, p);
            const bool dev_ok = (e == cudaSuccess) && (at.type == cudaMemoryTypeDevice) && (at.device == (int)j);
            printf("   dev%zu: data=%p ne=[%lld,%lld] cudaDev=%d type=%d -> %s\n",
                   j, p, (long long)st->ne[0], (long long)st->ne[1],
                   (e==cudaSuccess?at.device:-1), (e==cudaSuccess?(int)at.type:-1),
                   dev_ok ? "OK" : "MISMATCH");
            if (!dev_ok) fails++;
            // shape: row-split halves the second dim
            if (st->ne[1] != (j == 0 ? t->ne[1]/2 : t->ne[1] - t->ne[1]/2)) {
                printf("   dev%zu: SPLIT-SHAPE MISMATCH (got ne[1]=%lld)\n", j, (long long)st->ne[1]);
                fails++;
            }
        }
    }
    printf("\nPROBE %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

// Build (from the sinter store, against the DL fork build):
//   FORK=/home/dconnolly/yarn-agentic/software/llama.cpp/autoround
//   BIN=/tmp/fork-dl-build/bin
//   g++ -std=c++17 -O2 integration/probe_ptr.cpp \
//       -I$FORK/ggml/include -I/opt/cuda/include \
//       -L$BIN -lggml-base -L/opt/cuda/lib64 -lcudart -Wl,-rpath,$BIN \
//       -o /tmp/mk-build/probe_ptr
// Run from $BIN (so ggml_backend_load_all finds libggml-cuda.so): cd $BIN && /tmp/mk-build/probe_ptr
