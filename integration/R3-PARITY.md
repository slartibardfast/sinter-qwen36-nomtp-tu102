# R3 parity: MK meta-forwarding == stock -sm tensor (PASS)

Date: 2026-07-12. Base: F16-ssm_out (`/opt/models/Qwen3.6-27B-AR16asF16-probe.gguf`).

The MK backend (mk_backend.cpp + mk_split_state.cpp) loaded under an unmodified
llama-server/llama-completion via `GGML_BACKEND_PATH`, `--device MK -sm none`,
produces byte-identical generation to stock `--device CUDA0,CUDA1 -sm tensor
-ts 1,1`. Greedy (`--temp 0 --seed 1`), prompt "The capital of France is":

    MK    : ... assistant <think></think> Paris
    stock : ... assistant <think></think> Paris   (IDENTICAL)

Bit-identical BY CONSTRUCTION: MK builds a genuine meta device with mk_split_state
(the exact-name geometry captured from llama's own planner, 993 tensors), so its
computation IS the stock -sm tensor path; graph_compute forwards to that meta
backend verbatim. The token match confirms it.

## Build (ad-hoc; R4 will wrap this in CMake for the reproducible artifact)

    FORK=software/llama.cpp/autoround ; BIN=/opt/models/.mk-build/fork-dl/bin
    g++ -shared -fPIC -std=c++17 -O2 -DGGML_BACKEND_DL \
      -I$FORK/ggml/include -I$FORK/ggml/src \
      integration/mk_backend.cpp integration/mk_split_state.cpp \
      -L$BIN -lggml -lggml-base -Wl,-rpath,$BIN -o /opt/models/.mk-build/libggml-mk.so

## Harness note

llama-completion auto-enables conversation mode when the model has a chat
template; run it BARE (inline env vars, no timeout/env wrapper, no -st/-no-cnv,
no </dev/null, all of which either hang or abort the run), let it generate, then
kill by GPU-PID. This is a harness quirk, not an MK issue.

## llama-bench: `-fa 1` is REQUIRED (2026-07-13)

llama-bench defaults flash-attention OFF; llama-completion/-server default it ON.
With FA off the decode graph builds the non-FA **transposed-V** cache write
(`cache_v_l3 (reshaped) (view)` <- `set_rows(Vcur head-split)`), and the fork's
meta backend aborts in `handle_set_rows` (ggml-backend-meta.cpp:705,
`GGML_ASSERT(src_ss[0].axis != AXIS_1)`) because the head-split (AXIS_1) V write
is outside that handler's supported cases. The megakernel's fingerprint
(cgraph/v1, 3703 transformer-stack nodes, fattn ops) is the **FA-ON** graph, so
the correct invocation matches production:

    GGML_BACKEND_PATH=/opt/models/.mk-build/libggml-mk.so \
      llama-bench -m <F16-ssm_out.gguf> -fa 1 --device MK -sm none -p 0 -n N -r R

Result on the F16-ssm_out base, MK forwarding (R3 substrate, not yet dispatching):
`qwen35 27B F16, fa=1, tg4 = 32.27 t/s` — the stock -sm tensor number through MK,
the L5 head-to-head baseline. The dispatch probe fires here too (graph nodes=3703,
4430 meta-resolved tensors), so L1 is confirmed under llama-bench.

(The assert is over-strict for a valid head-split KV write — it returns src_ss[0]
right after — so an FA-off llama-bench path could later be unblocked by relaxing
it; not needed while the target config is FA-on.)

## Live decode geometry (llama's per-GPU pointers) MATCHES the harness layout

Captured 2026-07-13 via the MK_DISPATCH_LOG geometry probe on the FA-on decode
graph. Every per-GPU shape and stride equals the standalone harness's assumed
layout, retiring the L3 weight-slice / KV-byte-layout risks:

    cache_k/v_l3 : meta [1024,256] f16  -> per-GPU [512,256]      (kv_row=N_EMBD_GQA/2=512, pos-major)
    cache_r_l0   : meta [30720] f32     -> per-GPU [15360]        (CONV_STATE_N/2)
    cache_s_l0   : meta [786432] f32    -> per-GPU [393216]       (SSM_STATE_N/2)
    attn_qkv.w   : meta [5120,10240] Q4 -> per-GPU [5120,5120]    (AXIS_1 row-split, 5120/GPU out)
    ssm_out.w    : meta [6144,5120] F16 -> per-GPU [3072,5120]    (AXIS_0 input-split -> AllReduce; F16 base confirmed)
    output.w     : meta [5120,248320] f16 -> per-GPU [5120,124160] (AXIS_1 vocab column-split; N_VOCAB=248320)

So L3 can point the Resolver directly at llama's extracted per-GPU pointers
(weights + KV in place, `conv_state_l`->`cache_r_l`, `ssm_state_l`->`cache_s_l`);
final byte-order/interleave correctness is the L4 parity check.
