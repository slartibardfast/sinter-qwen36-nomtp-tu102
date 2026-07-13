// mk_backend.cpp — the sinter megakernel as an out-of-tree ggml dynamic backend.
//
// Registered as ONE GPU device "MK" loaded via GGML_BACKEND_PATH after CUDA.
// Architecture (call/0021, docs/INTEGRATION-FINDINGS.md): a meta-forwarding
// device whose buffer types ARE the fork's meta bufts over [CUDA0, CUDA1], so
// weights/KV/state land in the exact stock -sm tensor layout. graph_compute
// fingerprint-hits run the persistent megakernel over the raw per-GPU pointers;
// misses forward the cgraph to a genuine in-process meta backend (bit-identical
// degrade by identity). llama-server is never modified.
//
// Built in stages so each is independently testable:
//   (a) register: appear in `llama-server --list-devices`         [done]
//   (b) forward:  run the model through MK == stock -sm tensor     [this commit]
//   (c) dispatch: fingerprint hit -> megakernel                    [next]
//
// R3 (this commit) implements the forwarding / degrade path — which for R3 is
// the WHOLE path; the megakernel hit path is (c). The MK device builds ONE
// genuine meta device over CUDA0/CUDA1 at registration and forwards every
// device query to it (keeping name "MK", type GPU); the MK backend wraps a real
// in-process meta backend and forwards graph_compute to it verbatim.
//
// The fork tree is only READ (headers) and linked (libggml-base); nothing here
// is committed into llama.cpp.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

// The census-derived Qwen3.6 split geometry, defined in its own translation
// unit. For each static tensor (weights, KV cache k/v, recurrent r/s) it MUST
// return the SAME ggml_backend_meta_split_state that llama's own
// llama_meta_device_get_split_state produces under -sm tensor for Qwen3.6-27B on
// 2 equal GPUs (axis/n_segments/ne[]) — otherwise the meta layout is not
// bit-identical to stock. Declared here (external linkage), called below.
ggml_backend_meta_split_state mk_split_state(const struct ggml_tensor * t, void * ud);

// R5 megakernel dispatch (integration/mk_dispatch.cpp): on a k=0 decode-graph
// fingerprint hit it runs the persistent megakernel over llama's raw per-GPU
// pointers and returns true; otherwise false and the graph is forwarded (R3
// degrade). Layer 1 is a no-op probe (returns false) that builds+logs the
// pointer map.
bool mk_dispatch(struct ggml_cgraph * cgraph);
void mk_stock_argmax(struct ggml_cgraph * cgraph);   // diagnostic (MK_COMPARE)
void mk_zero_state(struct ggml_cgraph * cgraph);     // diagnostic (MK_ZERO_STATE)

namespace {

// ---- the two simple devices we sit on top of (looked up once) --------------
// We are loaded AFTER CUDA (reg.cpp loads GGML_BACKEND_PATH last), so CUDA0/1
// are registered by the time ggml_backend_init runs.
constexpr const char * kSimpleDevNames[] = { "CUDA0", "CUDA1" };
constexpr size_t kNumSimple = sizeof(kSimpleDevNames) / sizeof(kSimpleDevNames[0]);

static ggml_backend_dev_t g_simple[kNumSimple] = { nullptr, nullptr };

static bool lookup_simple_devices() {
    for (size_t i = 0; i < kNumSimple; ++i) {
        if (!g_simple[i]) {
            g_simple[i] = ggml_backend_dev_by_name(kSimpleDevNames[i]);
        }
        if (!g_simple[i]) {
            fprintf(stderr, "[MK] simple device %s not found (load MK after CUDA)\n",
                    kSimpleDevNames[i]);
            return false;
        }
    }
    return true;
}

// ---- device context: the one genuine meta device we forward to -------------
struct mk_device_context {
    ggml_backend_dev_t meta_dev;  // ggml_backend_meta_device(CUDA0,CUDA1, mk_split_state)
};

static mk_device_context g_mk_device_ctx = { /* .meta_dev = */ nullptr };

// Build (once) the meta device over CUDA0/CUDA1 with our split callback, and
// memoize it in the device context. Built eagerly at registration; the lazy
// path here covers the case where CUDA0/CUDA1 were not yet registered then
// (ggml_backend_meta_device memoizes and is single-threaded at load, so a single
// guarded build is safe).
static ggml_backend_dev_t mk_meta_dev(ggml_backend_dev_t dev) {
    mk_device_context * ctx = (mk_device_context *) dev->context;
    if (!ctx->meta_dev && lookup_simple_devices()) {
        ctx->meta_dev = ggml_backend_meta_device(g_simple, kNumSimple, mk_split_state, nullptr);
        if (!ctx->meta_dev) {
            fprintf(stderr, "[MK] failed to build meta device over CUDA0,CUDA1\n");
        }
    }
    return ctx->meta_dev;
}

// ---- backend (stream) context: a real in-process meta backend instance -----
struct mk_backend_context {
    ggml_backend_t meta_backend;  // ggml_backend_dev_init(meta_dev, nullptr)
};

static ggml_backend_t mk_meta_backend(ggml_backend_t backend) {
    return ((mk_backend_context *) backend->context)->meta_backend;
}

// ---- backend (stream) iface: forward everything to the meta backend --------

static const char * mk_backend_get_name(ggml_backend_t /*backend*/) { return "MK"; }

static void mk_backend_free(ggml_backend_t backend) {
    mk_backend_context * bctx = (mk_backend_context *) backend->context;
    ggml_backend_free(bctx->meta_backend);  // tears down the meta backend + its context
    delete bctx;
    delete backend;
}

static void mk_backend_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                        const void * data, size_t offset, size_t size) {
    ggml_backend_tensor_set_async(mk_meta_backend(backend), tensor, data, offset, size);
}

static void mk_backend_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor,
                                        void * data, size_t offset, size_t size) {
    ggml_backend_tensor_get_async(mk_meta_backend(backend), tensor, data, offset, size);
}

static void mk_backend_synchronize(ggml_backend_t backend) {
    ggml_backend_synchronize(mk_meta_backend(backend));
}

static enum ggml_status mk_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Fingerprint-hit k=0 decode graph -> persistent megakernel (R5). Miss /
    // prefill / batch>1 -> forward the cgraph to the meta backend (R3 degrade,
    // stock fan-out + NCCL/AR allreduce, bit-identical).
    // Stock/compare path zeroes here (no resident MK kernel); the MK path zeroes
    // inside mk_dual_step on its private stream to avoid the resident-kernel hang.
    if (getenv("MK_ZERO_STATE") && !getenv("MK_DISPATCH_RUN")) mk_zero_state(cgraph);
    if (mk_dispatch(cgraph)) return GGML_STATUS_SUCCESS;
    enum ggml_status s = ggml_backend_graph_compute(mk_meta_backend(backend), cgraph);
    if (getenv("MK_COMPARE")) mk_stock_argmax(cgraph);
    return s;
}

static ggml_guid_t mk_backend_guid() {
    // Distinct MK guid ("MK" + fixed random tail); never matches the CUDA/meta
    // guids (those are checked on the simple backends, inside the meta backend).
    static ggml_guid guid = {
        0x4d, 0x4b, 0x53, 0x1f, 0xa7, 0x62, 0x48, 0xce,
        0x9d, 0x3b, 0x0c, 0x71, 0xe4, 0x8a, 0x25, 0xb6,
    };
    return &guid;
}

static const struct ggml_backend_i mk_backend_i = {
    /* .get_name                = */ mk_backend_get_name,
    /* .free                    = */ mk_backend_free,
    /* .set_tensor_async        = */ mk_backend_set_tensor_async,
    /* .get_tensor_async        = */ mk_backend_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ mk_backend_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ mk_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

// ---- device iface ----------------------------------------------------------

static const char * mk_dev_get_name(ggml_backend_dev_t /*dev*/) { return "MK"; }

static const char * mk_dev_get_description(ggml_backend_dev_t /*dev*/) {
    return "sinter megakernel (meta-forwarding over CUDA0,CUDA1)";
}

static void mk_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Forward from the meta dev (which reports the sum of both GPUs), so -ngl
    // places the whole model on MK.
    *free = 0; *total = 0;
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    if (!meta) { return; }
    ggml_backend_dev_memory(meta, free, total);
}

static enum ggml_backend_dev_type mk_dev_get_type(ggml_backend_dev_t /*dev*/) {
    // MUST be GPU, never META — default device enumeration aborts on a META-type
    // registered device (src/llama.cpp:230-231).
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void mk_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    // Forward the meta dev's props (memory + caps + device_id) verbatim, then
    // restore our own identity: name "MK", type GPU (never META).
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    if (meta) {
        ggml_backend_dev_get_props(meta, props);
    } else {
        memset(props, 0, sizeof(*props));
    }
    props->name        = mk_dev_get_name(dev);
    props->description  = mk_dev_get_description(dev);
    props->type         = mk_dev_get_type(dev);
}

static ggml_backend_t mk_dev_init_backend(ggml_backend_dev_t dev, const char * /*params*/) {
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    if (!meta) {
        fprintf(stderr, "[MK] init_backend: meta device unavailable\n");
        return nullptr;
    }
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta, nullptr);
    if (!meta_backend) {
        fprintf(stderr, "[MK] init_backend: meta backend init failed\n");
        return nullptr;
    }
    mk_backend_context * bctx = new mk_backend_context{ /* .meta_backend = */ meta_backend };
    ggml_backend_t backend = new ggml_backend;
    backend->guid    = mk_backend_guid();
    backend->iface   = mk_backend_i;
    backend->device  = dev;
    backend->context = bctx;
    return backend;
}

static ggml_backend_buffer_type_t mk_dev_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    return meta ? ggml_backend_dev_buffer_type(meta) : nullptr;
}

static ggml_backend_buffer_type_t mk_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    return meta ? ggml_backend_dev_host_buffer_type(meta) : nullptr;
}

static bool mk_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    return meta ? ggml_backend_dev_supports_op(meta, op) : false;
}

static bool mk_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t meta = mk_meta_dev(dev);
    return meta ? ggml_backend_dev_supports_buft(meta, buft) : false;
}

static const struct ggml_backend_device_i mk_device_i = {
    /* .get_name             = */ mk_dev_get_name,
    /* .get_description      = */ mk_dev_get_description,
    /* .get_memory           = */ mk_dev_get_memory,
    /* .get_type             = */ mk_dev_get_type,
    /* .get_props            = */ mk_dev_get_props,
    /* .init_backend         = */ mk_dev_init_backend,
    /* .get_buffer_type      = */ mk_dev_get_buffer_type,
    /* .get_host_buffer_type = */ mk_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ mk_dev_supports_op,
    /* .supports_buft        = */ mk_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---- reg iface -------------------------------------------------------------

static ggml_backend_reg      g_mk_reg;     // forward-declared context below
static ggml_backend_device   g_mk_device;

static const char * mk_reg_get_name(ggml_backend_reg_t /*reg*/) { return "MK"; }
static size_t       mk_reg_dev_count(ggml_backend_reg_t /*reg*/) { return 1; }
static ggml_backend_dev_t mk_reg_dev_get(ggml_backend_reg_t /*reg*/, size_t /*i*/) { return &g_mk_device; }

static const struct ggml_backend_reg_i mk_reg_i = {
    /* .get_name         = */ mk_reg_get_name,
    /* .get_device_count = */ mk_reg_dev_count,
    /* .get_device       = */ mk_reg_dev_get,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg_t mk_backend_reg() {
    g_mk_reg.api_version = GGML_BACKEND_API_VERSION;
    g_mk_reg.iface       = mk_reg_i;
    g_mk_reg.context     = nullptr;

    g_mk_device.iface   = mk_device_i;
    g_mk_device.reg     = &g_mk_reg;
    g_mk_device.context = &g_mk_device_ctx;

    // Build the meta device ONCE, now, over CUDA0/CUDA1 (loaded before us). If
    // they are not yet registered, mk_meta_dev retries lazily on first use.
    mk_meta_dev(&g_mk_device);
    return &g_mk_reg;
}

} // namespace

GGML_BACKEND_DL_IMPL(mk_backend_reg)
