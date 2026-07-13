# L6: build + run recipe (in lieu of the reproducible R4 wrap)

The k=0 sinter megakernel runs as an out-of-tree ggml backend (`libggml-mk.so`)
loaded into the unmodified DL fork via `GGML_BACKEND_PATH`. It dispatches on the
k=0 decode graph and runs the persistent dual-GPU kernel over llama's in-place
pointers; prefill and any non-decode graph forward to the in-process meta backend
(R3, byte-identical to stock `-sm tensor`).

## The model MUST match the schedule (the load-bearing gotcha)

`k0/program-split.json` uses `OP_MMVQ_AR16` for every `ssm_out` (48 DeltaNet
layers) — it dequantizes the **AR16 custom quant**. Run on the **pure AR16 model**:

    /opt/models/Qwen3.6-27B-Q4_0AR16-rust.gguf     (ssm_out type = 42, AR16)

Do NOT run the F16-ssm_out base (`AR16asF16-probe.gguf`): its `ssm_out` is F16, the
AR16 dequant reads F16 bytes as AR16, and every DeltaNet layer produces garbage
(plausible magnitude, no NaN — the failure is silent). The F16-ssm_out base is for
the MTP contest (draft acceptance, call/0020); the k=0 no-MTP megakernel is AR16.

## Build the .so (ad-hoc; R4 will CMake-wrap this)

    FORK=/home/dconnolly/yarn-agentic/software/llama.cpp/autoround
    BIN=/opt/models/.mk-build/fork-dl/bin      # the GGML_BACKEND_DL fork build
    NV="/opt/cuda/bin/nvcc -std=c++17 -arch=sm_75 --objdir-as-tempdir -DGGML_BACKEND_DL -Xcompiler -fPIC"
    # CUDA core (harness.cpp holds the dual driver + MK entry points; guard main):
    $NV -c -x cu -DMK_NO_MAIN k0/harness.cpp -o harness_mk.o     # -x cu: ops/*.cuh are device code
    $NV -c core/interp.cu -o interp.o
    $NV -c -x cu core/host.cpp -o host.o
    # backend TUs:
    for f in mk_backend mk_split_state mk_dispatch; do
      $NV -c integration/$f.cpp -I$FORK/ggml/include -I$FORK/ggml/src -o $f.o; done
    # link (-fPIC is REQUIRED: dual_run_pass has a thread_local; local-exec TLS is
    # illegal in a .so):
    /opt/cuda/bin/nvcc -shared -arch=sm_75 -Xcompiler -fPIC \
      harness_mk.o interp.o host.o mk_backend.o mk_split_state.o mk_dispatch.o \
      -L$BIN -lggml -lggml-base -Xlinker -rpath -Xlinker $BIN \
      -Xlinker -rpath -Xlinker /opt/cuda/lib64 -o /opt/models/.mk-build/libggml-mk.so

## Run (dispatch the megakernel)

    M=/opt/models/Qwen3.6-27B-Q4_0AR16-rust.gguf
    PROG=$PWD/k0/program-split.json     # absolute; the .so resolves gguf:/cache:/buf: names
    GGML_BACKEND_PATH=/opt/models/.mk-build/libggml-mk.so MK_DISPATCH_RUN=1 MK_PROGRAM=$PROG \
      <tool> -m $M --device MK -sm none -fa 1 ...

- `MK_DISPATCH_RUN=1` opts into the megakernel (else it forwards = stock baseline).
- `-fa 1` is REQUIRED for llama-bench (it defaults FA off; the FA-off transposed-V
  set_rows aborts the meta backend). completion/cli/server default FA on.
- `-sm none --device MK`: the one MK meta-device splits internally across CUDA0/1.

## Verified

- **llama-bench** (tg32, AR16): MK dispatch **30.79 t/s** vs stock (MK-forward =
  -sm tensor) **29.73 t/s** shallow; **30.71 vs 29.70** at 16K depth.
- **llama-completion / llama-cli**: greedy decode TOKEN-EXACT vs stock
  (271,248069,271,57590,248046 -> "...Paris").
- **llama-server**: the megakernel dispatches on the served decode, but the
  request currently hangs (a server thread/stream interaction with the persistent
  kernel — the decode path itself is proven correct in completion). OPEN.
- **llama-perplexity**: not built in the fork yet. OPEN.
- **Deep 256K**: dispatches at 16K; a live 256K bench is dominated by the forward
  prefill; 256K split-KV parity + the deep floor were validated separately (t#31).

## Diagnostic env knobs (all env-gated, in the .so)

MK_DISPATCH_RUN, MK_PROGRAM, MK_WCHECK=<model> (weight byte-compare vs weight_slice),
MK_ARGMAX (per-pass argmax + seed dump), MK_COMPARE (stock argmax post-forward),
MK_DBGMID (per-layer residual norm + GPU0-vs-GPU1 mirror diff), MK_ZERO_STATE
(zero the KV/state), MK_OWNEMBED=<model>+MK_TOKEN (bind token_embd, un-strip).
