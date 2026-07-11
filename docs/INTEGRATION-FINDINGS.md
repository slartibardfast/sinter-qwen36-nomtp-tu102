# Integration seam for the persistent-decode megakernel backend

Target: unmodified `llama-server`, fork `llama.cpp` @ `546eca8dc` (branch `autoround`),
ggml 0.12.0 (`ggml/CMakeLists.txt:6-9`). All paths below are relative to
`/home/dconnolly/yarn-agentic/software/llama.cpp/autoround/` unless absolute.

Headline discovery: this fork does NOT implement `-sm tensor` with per-GPU backends in the
scheduler. It implements a **meta device/backend** (`ggml/src/ggml-backend-meta.cpp`, 2144
lines) that wraps N real devices behind a single `ggml_backend_dev_t`. The scheduler sees ONE
GPU backend; all tensor-parallel splitting, fan-out, and allreduce happen inside the meta
layer. This reshapes the buffer-ownership question — and gives us a delegation target that
makes fallback bit-identical for free.

---

## A. `-sm tensor` mechanics

**Parsing.** `common/arg.cpp:2362-2376`: `-sm/--split-mode {none,layer,row,tensor}`;
`"tensor"` sets `params.split_mode = LLAMA_SPLIT_MODE_TENSOR` (enum value 3,
`include/llama.h:198`). `-ts/--tensor-split` ratios feed the split (arg.cpp:2386-2408).

**Device construction.** `src/llama.cpp:125-186` (`llama_prepare_model_devices`): under
`LLAMA_SPLIT_MODE_TENSOR`, llama does NOT put the CUDA devices in `model->devices`.
It creates a single meta device wrapping them:
`ggml_backend_meta_device(devs, n_devs, llama_meta_device_get_split_state, &model->get_split_state_ud)`
(src/llama.cpp:143-145 with `--device`, :185-187 default). Default selection includes every
registered device whose buft != CPU buft (src/llama.cpp:163-171). The public factory is
`ggml/include/ggml-backend.h:395-399` (installed public header); the device type is
`GGML_BACKEND_DEVICE_TYPE_META` (ggml-backend.h:143-144). Arch gate:
`src/llama-model.cpp:297-301` + `llm_arch_supports_sm_tensor` (src/llama-arch.cpp:891-920;
QWEN35 is in the default=true set). `-sm tensor` also forces flash-attn on and forbids
quantized KV (src/llama-context.cpp:3384-3397).

**Split policy (per tensor name).** `llama_meta_device_get_split_state`
(src/llama-model.cpp:328-666) classifies by regex on tensor name and returns a
`ggml_backend_meta_split_state {axis, ne[16*16], n_segments}` (ggml-backend.h:360-390):

- `attn_q/k/v.weight`, fused `attn_qkv.weight`, `ffn_up/gate.weight`, `attn_gate.weight`,
  `ssm_alpha/beta/ba.weight`, `ssm_conv1d`, `output.weight` → `AXIS_1` (output-dim/rows
  split) (llama-model.cpp:407-415, 440-447, 455-460, 475-478).
- `attn_output.weight`, `ffn_down.weight`, `ssm_out.weight` → `AXIS_0` (input-dim split)
  (llama-model.cpp:425-427, 450-452, 463-465).
- norms/biases and “everything else” → `AXIS_MIRRORED` (full copy on every GPU)
  (llama-model.cpp:485-486); per-head `attn_(q|k)_norm` with ne[1]>1 → `AXIS_1`
  (llama-model.cpp:419-421).
- `cache_(k|v)_lN` → `AXIS_0` with per-KV-head granularity (llama-model.cpp:423-424 +
  granularity at :593-599) — KV is split **by heads**, all positions of a head on one GPU.
- `cache_r_lN`/`cache_s_lN` → `AXIS_0` with Qwen3.5 segment logic (llama-model.cpp:444-447;
  segments :488-527: QWEN35 segments qkv/conv1d as `{key,key,value…}`, s-cache per
  head-ratio).
- Fused tensors are segmented so each GPU gets a slice of each of Q,K,V (`n_segments`,
  qkv = `{n_embd, n_embd_gqa, n_embd_gqa}` llama-model.cpp:537-543; gate_up =
  `{n_ff_exp, n_ff_exp}` :544-548).

**Division formula.** llama-model.cpp:619-655: per segment, device j gets
`ne_s*(j+1)/n_devices` (or `-ts`-proportional), floored to a granularity (head-size LCM for
q/kv, `blck_size` for FFN), remainder to the last; `rotation = effective_layer % n_devices`
(llama-model.cpp:363-391) rotates which GPU absorbs rounding per layer. With 2 equal GPUs
this is an exact half: qkv 10240→5120 per GPU, ffn 17408→8704 — matches the measured census.

**Where each category lands.** Everything the model/context allocates on the "GPU" lands in
the ONE meta buft `Meta(CUDA0,CUDA1)`; the meta buffer holds one *simple* (plain
`ggml_backend_cuda_buffer_type`) buffer **per GPU** and one shadow *simple tensor* per GPU
per meta tensor:

| Category | buft | GPU placement |
|---|---|---|
| (i) big matmul weights | meta buft (weights buft list = meta dev's buffer type, `make_gpu_buft_list` src/llama-model.cpp:905, list built :1176-1181) | sliced: half rows (AXIS_1) or half cols (AXIS_0) on each GPU |
| (ii) small weights (norms/biases) | same meta buft (same buffer even) | MIRRORED: full copy on both GPUs. Exception: `token_embd` stays on **CPU** (`dev_input = cpu_dev`, src/llama-model.cpp:1238-1239) |
| (iii) KV `k_l/v_l` | meta buft: `ggml_backend_dev_buffer_type(model.dev_layer(il))` (src/llama-kv-cache.cpp:191-195) | AXIS_0 head-split: each GPU holds all 256K positions for half the KV heads |
| (iv) recurrent `r_l/s_l` | meta buft (src/llama-memory-recurrent.cpp:83-94) | AXIS_0 (head-aligned) split |
| (v) compute/sched buffer | sched buft for the meta backend = its default buffer type = meta buft (src/llama-context.cpp:316-334, ggml-backend.cpp:1778); CPU backend’s sched buft is swapped for the meta dev’s **host (pinned)** buft (llama-context.cpp:320-327 + meta host buft ggml-backend-meta.cpp:370-389) | meta compute buffer allocates the full requested size on EACH GPU (ggml-backend-meta.cpp:1434-1453) |

Meta-tensor `->data` is fake (base `0x1000000000000000`, ggml-backend-meta.cpp:1082-1085;
statics get constant `0x2000000000000000`, :1474). Real device pointers live only on the
per-GPU simple tensors (created in `ggml_backend_meta_buffer_init_tensor`,
ggml-backend-meta.cpp:1087-1200; data at :1157-1162).

**Census explanation.** Both GPUs run every kernel with half-size grids because the meta
backend executes the SAME subgraph on each GPU's own CUDA backend with per-GPU slice shapes
(fan-out loop ggml-backend-meta.cpp:2067-2074).

## B. Execution orchestration at decode

- **Scheduler sees 2 backends**: `Meta(CUDA0,CUDA1)` + CPU. llama-context builds one backend
  per `model.devices` entry (the single meta device), then ACCEL, then CPU
  (src/llama-context.cpp:241-268); `ggml_backend_sched_new` at :434.
- Inside the meta backend live **two full CUDA backend instances**, one per simple device
  (`ggml_backend_meta_context` ctor, ggml-backend-meta.cpp:1526-1541). This is NOT the old
  row-split buffer mechanism (`ggml_backend_cuda_split_buffer_type` ggml-cuda.cu:889ff is
  only wired for `-sm row`, src/llama-model.cpp:883-903).
- **Sched level**: decode graph forms ~2 splits — split 0 on CPU (`get_rows` over CPU-resident
  `token_embd` + input tensors, assignment rule ggml-backend.cpp:901-906, 908-929), split 1 =
  the entire transformer stack on the meta backend. One `graph_compute` call per split
  (ggml-backend.cpp:1678).
- **Meta level**: `ggml_backend_meta_graph_compute` (ggml-backend-meta.cpp:1674-2098)
  re-splits its split into subgraphs at every node whose derived split state is
  `AXIS_PARTIAL` (boundary test :1822-1828) — i.e. after `attn_output`/`ffn_down`/`ssm_out`
  matmuls (AXIS_0 weight × AXIS_0 activation → PARTIAL, `handle_mul_mat`
  ggml-backend-meta.cpp:550-553). ≈2 subgraphs (and 2 allreduces) per layer. Per subgraph it
  calls `ggml_backend_graph_compute_async` on BOTH CUDA backends (:2067-2074), then
  allreduces the boundary tensor (:2076-2095).
- **Cross-GPU exchange**: `ggml_backend_comm_init/_allreduce_tensor` proc-addresses resolved
  from the CUDA reg (ggml-backend-meta.cpp:1544-1556; typedefs ggml-backend.h:205-208). CUDA
  side `ggml_backend_cuda_comm_init` (ggml-cuda.cu:1383ff) chains: **NCCL**
  (`ncclAllReduce` on each `cuda_ctx->stream()`, ggml-cuda.cu:1184-1256) → **internal AR
  pipeline** (`ggml/src/ggml-cuda/allreduce.cu`: pinned-host staging over PCIe; small tensors
  = single kernel with in-kernel host-flag spin, large = copy-engine D2H/H2D chunks + CUDA
  events; header comment allreduce.cu:13-60) → **butterfly fallback** built from
  `ggml_backend_tensor_copy_async` + ADD one-node graphs (ggml-backend-meta.cpp:1952-2064;
  peer copies via `cudaMemcpyPeerAsync` in `ggml_backend_cuda_cpy_tensor_async`
  ggml-cuda.cu:3180-3220). Env `GGML_CUDA_ALLREDUCE` selects (ggml-cuda.cu:1400).
- **Synchronization ordering**: everything is stream-ordered per GPU (compute_async then
  allreduce on the same streams, no host sync in the loop); sched-level input copies use
  events/`ggml_backend_synchronize` (ggml-backend.cpp:1554-1674); `meta.synchronize` = sync
  every simple backend (ggml-backend-meta.cpp:1667-1672).

## C. Fallback feasibility

The whole-decode-graph-on-one-backend-instance path **is** what already happens — but that
instance is the **meta backend**, not a stock CUDA backend. A stock CUDA backend cannot
compute this graph as-allocated:

- meta tensors carry fake `->data` (ggml-backend-meta.cpp:1082-1085, :1474);
- CUDA `supports_buft` accepts only its own cuda/cuda-split bufts on its own device —
  a meta buft is rejected (`ggml_backend_cuda_device_supports_buft`, ggml-cuda.cu:5426-5430);
- meta `supports_buft` conversely accepts only bufts whose device is a meta device with the
  identical simple-device vector (ggml-backend-meta.cpp:158-175).

So our backend does NOT replicate scheduler behavior for fallback: it delegates the received
cgraph to an in-process **meta backend instance** (`ggml_backend_dev_init(meta_dev, nullptr)`
→ ggml-backend-meta.cpp:2123-2132), which runs the exact stock code path (fan-out + NCCL/AR
allreduce). If instead all tensors sat in plain CUDA0 bufts, one stock CUDA backend would
compute the whole graph — but that abandons the 2-GPU split and the 256K KV budget.

## D. Raw pointer extraction

Chain (weights, KV, and compute tensors are identical):

1. `tensor->buffer` → meta buffer. Test: `ggml_backend_buffer_is_meta()` — exported,
   declared `ggml/src/ggml-backend-impl.h:92`, verified `T` in built `libggml-base.so`.
2. `((ggml_backend_meta_buffer_context *) tensor->buffer->context)->simple_tensors[tensor][j]`
   → per-GPU `ggml_tensor *`. **`ggml_backend_meta_buffer_simple_tensor` is `static`**
   (ggml-backend-meta.cpp:440-450) and the context struct is defined only in the .cpp
   (ggml-backend-meta.cpp:395-415):
   ```c++
   struct ggml_backend_meta_buffer_context {
       std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
       std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;
       std::vector<buffer_config> buf_configs;   // buffer_config { ggml_context * ctx; ggml_backend_buffer_t buf; }
       int debug;
   };
   ```
   → we must **mirror this struct** (FLAGGED below).
3. `simple_tensor->data` IS the raw CUDA device pointer (set at ggml-backend-meta.cpp:1157-1162
   for compute buffers; via per-device `ggml_backend_alloc_ctx_tensors_from_buft` for statics,
   :1476-1479). No detour through `tensor->extra` — extras (`ggml_tensor_extra_gpu`,
   ggml-cuda.cu:974-1027) are used only by the legacy `-sm row` split buffers, not here.
4. GPU index = position `j` in the meta device's simple-dev order (== our `[CUDA0, CUDA1]`).
   If needed, the owning plain CUDA buffer's context is
   `ggml_backend_cuda_buffer_context { int device; void * dev_ptr; std::string name; }`
   (ggml-cuda.cu:622-635).
5. Split geometry: the exported public struct `ggml_backend_meta_split_state`
   (ggml-backend.h:376-390) is what our own callback returns — we know it without reading
   the (static) `ggml_backend_meta_get_split_state` cache.

**Headers / linkage.** Installed public headers: `ggml.h`, `ggml-backend.h`, `ggml-alloc.h`,
`ggml-cuda.h`, … (ggml/CMakeLists.txt:323-341). `ggml-backend-impl.h` (backend/device/reg
iface structs needed to implement a backend, plus the meta accessors' prototypes) is NOT
installed but is includable from the pinned source tree. Exported symbols verified in
`build/bin/libggml-base.so`: `ggml_backend_meta_device`, `ggml_backend_buffer_is_meta`,
`ggml_backend_buft_is_meta`, `ggml_backend_is_meta`, `ggml_backend_meta_n_backends`,
`ggml_backend_meta_simple_backend`, `ggml_backend_meta_alloc_ctx_tensors_from_buft`.
**Must mirror (pinned-fork, FLAG):** `ggml_backend_meta_buffer_context` (+
`buffer_config`) from ggml-backend-meta.cpp:395-415 — contains `std::map`/`std::vector`, so
this is same-toolchain C++-ABI mirroring; add startup sanity checks
(`buf_configs.size()==2`, simple-tensor name equality, `cudaPointerGetAttributes` device
check). Alternative that removes all mirroring: a one-line fork commit exporting the
accessor — that rebuilds `libggml-base.so` but leaves llama-server untouched; noted, not
assumed.

## E. Dynamic backend loading

- `GGML_BACKEND_DL` exists; `GGML_BACKEND_API_VERSION 2` (ggml-backend-impl.h:11); handshake:
  `ggml_backend_load` → `dlopen`, optional `ggml_backend_score()` (0 = unsupported),
  `ggml_backend_init()` must return a reg with `api_version == 2`
  (ggml-backend-reg.cpp:213-257; entry-point macros ggml-backend-impl.h:240-267).
- Discovery: `llama_backend_init` calls `ggml_backend_load_all()` **only if no backends are
  registered yet** (src/llama.cpp:89-102). `load_all_from_path` probes known names
  (`ggml_backend_load_best("cuda", …)` etc.) then loads the single path in env
  **`GGML_BACKEND_PATH`** LAST (ggml-backend-reg.cpp:555-586). ⚠️ The checked-out
  `build/` here is `GGML_BACKEND_DL:BOOL=OFF` (build/CMakeCache.txt:360) and CPU-only —
  in a non-DL build backends register statically in the registry ctor
  (ggml-backend-reg.cpp:120-166), `reg_count()!=0`, and `GGML_BACKEND_PATH` is never read.
  **The serving build must be configured `-DGGML_BACKEND_DL=ON -DGGML_CUDA=ON`** (matches
  the stated deployment: separate `libggml-cuda.so`).
- Ordering/selection: devices enumerate in registration order (ggml-backend-reg.cpp:200-211),
  so CUDA0, CUDA1, …, CPU, then our device (loaded last). `--device` (`-dev`) picks by
  name via `ggml_backend_dev_by_name` (common/arg.cpp:818-836, 2286-2292;
  `mparams.devices` common/common.cpp:1518-1519); `--list-devices` prints them
  (arg.cpp:2294-2306). Without `--device`: `-sm none` keeps only `devices[main_gpu]`
  (src/llama.cpp:248-262); layer distribution otherwise follows free-memory or `-ts` splits
  (src/llama-model.cpp:1190-1233); GPU-type devices with a `device_id` equal to an existing
  one are deduped in the non-tensor default path only (src/llama.cpp:203-223). A registered
  device must NOT report type `META` or default enumeration aborts (src/llama.cpp:230-231).
- Required device surface (ggml-backend-impl.h:160-202): name/description/memory/type/props,
  `init_backend`, `get_buffer_type` (+ `get_host_buffer_type` — llama-context uses it for the
  CPU sched buft, llama-context.cpp:320-327), `supports_op`, `supports_buft`. To (i) be
  listed: register via reg `get_device_count/get_device`; (ii) receive the whole model with
  `-ngl`: type `GPU`, report enough `memory_free` (llama-model.cpp:1192-1207), pass the
  per-weight `supports_op` probes (llama-model.cpp:1835-1870); (iii) claim the full graph:
  be the ONLY model device → backend index 0 in the sched.
- Scheduler selection: weights pin ops to the highest-priority backend with
  `supports_buft(weight buft) && supports_op` (`ggml_backend_sched_backend_from_buffer`
  ggml-backend.cpp:845-865, used at :878-933); `offload_op` only matters for host-resident
  weights (:919-926; CUDA's requires batch≥32, ggml-cuda.cu:5432-5451&nearby). Lower index =
  higher priority (ggml-backend.h:316). Sched asserts to respect: last backend must be a
  CPU-type device (ggml-backend.cpp:1736); every backend must support its sched buft
  (:1778-1779); a pre-allocated tensor whose buffer no backend can run → `GGML_ABORT`
  (:895-899).

## F. The cgraph the backend receives

Splits are formed in `ggml_backend_sched_split_graph` pass 5
(ggml-backend.cpp:1245-1376): contiguous ranges of the ORIGINAL `graph->nodes` order; the
split's executable graph is `ggml_graph_view(graph, i_start, i_end)` (:1413) — same
topological order the capture recorded. With one device claiming everything, decode =
one CPU split (token-embd `get_rows` + inputs) followed by ONE split containing the whole
transformer stack handed to our `graph_compute` (dispatch ggml-backend.cpp:1678). Two
caveats: (a) `ggml_backend_graph_optimize` (:1417) lets the split backend reorder — our
iface sets it `nullptr` (the meta backend also does, ggml-backend-meta.cpp:2116; CUDA's is
env-gated `GGML_CUDA_GRAPH_OPT` and single-GPU-only, ggml-cuda.cu:4541-4568); (b) sources
that were cross-backend inputs are rewritten to sched-created copy tensors named
`"<backend>#<name>#<c>"` (:1352-1371) — the fingerprint must tolerate that. `cb_eval`
iterates `split->graph.nodes` in this same order (:1684-1713), so capture order == received
order.

## G. cb_eval

Confirmed: `llama_context_params.cb_eval/cb_eval_user_data` (include/llama.h:361-362) is
wired to `ggml_backend_sched_set_eval_callback` on every graph (re)build in the decode path
(src/llama-context.cpp:83-84, 1264-1265) and forces per-node-range sync compute
(ggml-backend.cpp:1682-1713) — usable as the parity oracle at decode.

---

## Recommendation: meta-forwarding device, delegated fallback

**Register ONE GPU-type device ("MK") from the `.so`; its buffer types ARE the fork's meta
bufts; its fallback compute IS the fork's meta backend.** Run:

```
GGML_BACKEND_PATH=/path/mk.so llama-server --device MK -sm none -ngl 99 -fa on -ctk f16 -ctv f16 …
```

- At `ggml_backend_init` (we load after CUDA — reg.cpp:555-586), look up CUDA0/CUDA1
  (`ggml_backend_dev_by_name`) and build a genuine meta device over them via the PUBLIC
  `ggml_backend_meta_device()` (ggml-backend.h:395-399) with OUR `get_split_state` callback
  reproducing the Qwen3.6 geometry of llama-model.cpp:328-666 (hardcoded per-name table;
  validated against the census: qkv 5120/10240, ffn 8704/17408, KV head-split).
- Device iface: `get_buffer_type` → the meta buft; `get_host_buffer_type` → forwarded (CUDA
  pinned); `supports_op`/`supports_buft` → forwarded to the meta device
  (`ggml_backend_dev_supports_op/buft`); memory = sum of both GPUs (as meta does,
  ggml-backend-meta.cpp:98-109); type `GPU` (never `META`, src/llama.cpp:230-231).
- Result: weights, KV (`k_l/v_l`), recurrent state, and the sched compute buffer all land in
  genuine meta buffers over CUDA0/CUDA1 — **bit-identical layout and allocation code to
  stock `-sm tensor`** (ggml-alloc.c:1241 dispatches static allocs into the meta path).
- Our backend's `graph_compute` receives the whole decode graph as one split (F).
  Fingerprint hit → megakernel reads raw per-GPU pointers (D) and writes results into the
  same per-GPU compute-tensor slices (so llama's spliced readback,
  ggml-backend-meta.cpp:1634-1653, still works). Miss / prefill / batch>1 → forward the
  cgraph to an internal genuine meta backend instance
  (`ggml_backend_dev_init(meta_dev, nullptr)`): the exact stock subgraph fan-out +
  NCCL/internal-AR allreduce executes — requirement (1) full-speed stock prefill and
  (3) bit-identical degrade are satisfied by *identity*, not by reimplementation.
- (4) 256K KV fits: AXIS_0 head-split KV puts half the heads' full-length cache on each GPU,
  same as stock `-sm tensor`.

**Rejected alternatives.**
- *Own bufts + reimplemented fallback*: re-writes meta buffer splitting, subgraph
  scheduling, and 3-tier allreduce (~2k lines of correctness-critical TP code that drifts
  from the fork). Nothing gained — pointer extraction is no easier.
- *Per-GPU proxy devices under stock `-sm tensor`* (`--device MK0,MK1`): the meta backend
  fans out per-device HALF-graph subgraphs (~2/layer) — wrong claim granularity for a
  persistent kernel that must fuse across allreduce boundaries; and wrapping backends breaks
  the `ggml_backend_is_cuda` guid check in `ggml_backend_cuda_comm_init`
  (ggml-cuda.cu:1383-1390), degrading allreduce to the butterfly path.

**Blockers / flags.**
1. **Struct mirror** of `ggml_backend_meta_buffer_context` (ggml-backend-meta.cpp:395-415)
   to reach `simple_tensors` — the only ABI-fragile piece (std::map layout; same pinned
   toolchain). Mitigate with startup sanity checks, or retire it with a one-line fork export
   of the static accessor (rebuilds libggml-base, llama-server untouched).
2. **Serving build must be `GGML_BACKEND_DL=ON` + `GGML_CUDA=ON`** — the current
   `build/` here is DL=OFF/CPU-only (build/CMakeCache.txt:360), where `GGML_BACKEND_PATH`
   is never consulted (src/llama.cpp:99-101).
3. With the `.so` loaded, never run stock `-sm tensor` **without** `--device`: default
   selection sweeps every non-CPU-buft device into the meta TP set — including MK
   (src/llama.cpp:163-171). Always pin `--device`.
4. `-sm none` skips the `-sm tensor` forcings (src/llama-context.cpp:3384-3397): pass
   `-fa on` and f16 KV explicitly, or the built graph won't match the fingerprint capture.
   (Graph building itself has no other split-mode dependence — verified by sweep:
   the only `split_mode`/`is_meta` uses in src/ are the ones cited above.)
5. Sched coupling asserts our iface must satisfy: `supports_buft(MK, meta buft)` == true
   (ggml-backend.cpp:1779), `supports_op` covering every graph op (else `GGML_ABORT`
   ggml-backend.cpp:895-899) — both are forwarded to the meta device, whose own
   `supports_op` ANDs the CUDA devices (ggml-backend-meta.cpp:151-156).
6. Meta has no `buffer_from_host_ptr` (ggml-backend-meta.cpp:132) → no mmap-direct weights;
   loads go through the splicing `set_tensor` (ggml-backend-meta.cpp:1202-1307). Same as
   stock `-sm tensor`; slower model load only.
7. `ggml_backend_meta_device` memoizes and is not thread-safe (ggml-backend-meta.cpp:212-242);
   our callback/userdata differ from llama's so we always get a distinct meta device — fine,
   but create it once, at registration.
