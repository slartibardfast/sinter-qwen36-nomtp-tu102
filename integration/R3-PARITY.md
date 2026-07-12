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
