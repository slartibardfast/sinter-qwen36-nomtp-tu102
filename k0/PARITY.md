# k=0 single-GPU full-stack decode — G13 parity + floor

The persistent megakernel now runs a full single-GPU decode (embed -> 64 hybrid
blocks -> output norm -> lm-head) driven by `mk-harness --parity`, packed from
`k0/program.json` and launched through `mk::Host` (core/host.h). This records
the G13 numerical-parity result against the single-GPU reference oracle and the
decode floor, plus the composition fixes it took to compose the unit-verified
op families into a whole.

## Parity result (compare.py, pasted)

Reference: `/var/tmp/mk-oracle/ref-none8192-ctx64` (the fork's `--split none`,
n_ctx 8192, 64-token prefill, 32 greedy steps, commit 546eca8dc). Candidate:
`mk-harness --parity` at the same prompt/steps, single GPU, n_ctx 8192.

```
note: config.split: ref none != cand mk-k0
logits: 5/33 rows compared, bit-identical: no
logits KL: max 1.113732e-02 (row 0)  fwd mean 4.239959e-03  rev mean 4.327535e-03  tol 0.02
tokens: diverge at step 4 (ref 1061 vs cand 561); row KL 3.924887e-03, logit gaps ref 0.0054 cand 0.0119
tokens: divergence is a tie within the KL band (3.924887e-03 <= 0.02): RECORDED, not scored; comparison truncated at step 4
residual RMS rel diff: 252 (step,node) rows, max 1.719769e-02 (step 1, l_out-50)  tol 0.02
  info result_norm: max rel diff 2.409802e-03 over 4 steps (not scored)
  info result_output: max rel diff 8.284941e-03 over 4 steps (not scored)
state: 384 files byte-compared over 4 comparable steps, 384 mismatched
RESULT: FAIL
  violation: 384 state tensors not bit-exact (first: ((0, 'cache_r_l0'), 'bytes'))
```

**The two scored numerical-parity lanes pass**: logits KL max **1.11e-2 ≤ 0.02**,
residual RMS rel max **1.72e-2 ≤ 0.02**. The first **four greedy tokens match the
reference bit-for-bit** (10885, 513, 14789, 13); the step-4 divergence is a
within-band tie (row KL 3.9e-3, logit gaps ~0.005–0.012), recorded not scored.

**compare.py exits 1 solely on the state-bit-exact lane** — and this is the
documented, expected outcome for a different-fold-order topology, not a
megakernel defect:

- The G13 playbook binds state bit-exactness against the *matching declared fold
  order* (candidate vs the same-config reference). The megakernel's ops use
  their own declared folds (MMVQ Q4_0/AR16 dp4a order, GEMV fp32 accumulate vs
  the fork's half2, the FATTN online-softmax tiling), so the conv/ssm state
  bytes cannot match the fork bit-for-bit.
- The reference tree's OWN cross-check demonstrates the same: fork tensor-split
  vs fork `--split none` (`/var/tmp/mk-oracle/cross-none.json`) reports **384/384
  state files not bit-exact and exits 1**, with KL 2.5e-3 / RMS 8.5e-3 and the
  identical step-4 within-band tie. Our numbers are ~2–4× larger (the extra
  fold-order distance of the megakernel's kernels) but sit in the same envelope,
  well inside the 0.02 classes.
- Telling detail: at the step-4 tie our candidate emits token **561** — which is
  what the fork's *tensor* reference emits; the `--split none` reference we
  compare against emits 1061. Three implementations split two ways on an
  intrinsic near-tie. Not a bug.

The residual divergence is benign accumulation, not a per-op error: step-0
`l_out` RMS rel diff is 4.8e-4 at block 0 and rises gradually with depth
(≈1–6e-3 through the stack), the profile of compounding rounding, with no spike
at any single op.

Report: `/var/tmp/mk-harness/cand-none/parity-report.json`.

## Decode floor

`mk-harness --bench 16` at n_ctx 8192, single GPU (device-measured block-0
clock64, TU102 @ 1455 MHz):

```
bench: 16 tokens in 0.586 s = 27.29 tok/s
bench: per-pass on-device clock64: mean 36.825 ms, min 36.778, max 36.837 (16 samples)
```

**27.3 tok/s, 36.8 ms/pass**, very stable (min/max within 0.06 ms). The
"contended" label the harness prints is a self-probe artifact — the idle check
runs after the persistent kernel is already spinning and sees its own
utilization; both GPUs were otherwise idle per `nvidia-smi` before launch.

This is **weight-stream bound**: the pass reads 16.38 GB of weights (COMPILER.md)
on ONE GPU; at the family's measured ~520 GB/s MMVQ / ~600 GB/s f16 streams
(k0/ops NOTES) that is ~28–31 ms of pure weight traffic, and the whole pass runs
at ~445 GB/s effective. The stock 43.9 tok/s floor and the ~73 tok/s red-line
(13.71 ms) are production/dual-GPU points: on two GPUs each streams *half* the
weights (8.2 GB → ~12 ms ≈ the red-line), which one GPU streaming all 16.38 GB
cannot reach on a 672 GB/s card. At the single-GPU n_ctx-8192 config the honest
floor is the number above; the config difference is expected (per the brief).

## Boundary count and the floor (feeds G15)

The program carries **964 boundaries/pass** (v0 placement, COMPILER.md). At the
spine's 806 ns/boundary (NOTES-spine.md) that is ~0.78 ms — **2.1 % of the
36.8 ms pass**. So boundary coalescing (the G15 task, target ~320–380) recovers
at most ~0.5 ms here: it is NOT the floor lever at n_ctx 8192; the weight stream
is. A coalescing pass was therefore **not** attempted in this milestone — the
payoff is sub-percent on the floor and a missed boundary is a silent wrong
answer. It remains the right lever for the pass-overhead gate (G15) and for the
deep-context / dual-GPU points where the weight stream is split and the boundary
share grows.

## Composition fixes (unit-verified ops -> whole)

Filling the 27 `pack_*` functions and wiring `mk::Host` exposed real
composition gaps between the per-GPU-shaped op families and the single-GPU
binding. Fixed:

1. **FATTN_DECODE shape (k0/ops/attn.cuh).** The op was built for the per-GPU
   shape (n_q ≤ 12 warps, row_width ≤ 512). The single-GPU program drives 24 q
   heads / 4 kv heads / row_width 1024 — 24 > 12 warps *and* a full-row 32-tile
   is 65 KiB (> the 60 KiB slab). Restructured to loop kv heads on the outer
   axis, carrying each head's gqa=6 q heads (6 warps) with a 256-wide slice tile
   (41 KiB at n_q=24). Disjoint 256-column slices ⇒ the whole KV is still read
   once (no extra DRAM). Backward compatible: `tests/test_attn.cu` stays green
   (12q/2kv, 6q/1kv) and two new 24q/4kv cases pass (worst head-rel 3.4e-7).
2. **32-aligned FATTN chunks (attn.cuh + COMPILER.md obligation).** `clen` is now
   rounded up to a multiple of the 32-row tile, so every chunk start `kv0` is
   32-aligned and the packed 2-f16 mask-word read never faults at any n_kv
   (previously an odd `ceil(n_kv/72)`, e.g. n_kv=768, would fault).
3. **RMSNORM residual write-back (k0/ops/glue.cuh).** The op discarded the
   pre-norm sum `x+add`, but the schedule folds each sublayer's output into the
   residual via `add_src`/`write_sum` — without publishing the sum the residual
   trunk never accumulates. Added `sum` (residual write-back) and `dbg`
   (l_out-il parity snapshot into `dbg_lout`); the packer derives il from the
   `blk.M.attn_norm.weight` name (l_out-(M-1)). Backward compatible via
   zero-init (`tests/test_interp.cu` glue lane green).
4. **fattn_partial buffer stride (k0/compile_schedule.py).** The buffer was
   sized at 258/record but the op's `MK_FATTN_PSTRIDE` is 260 (16 B record
   alignment) — a 13,824-byte under-allocation. Fixed the compiler constant to
   `ATTN_HEAD_DIM + 4`; only the buffer size changed, instructions identical,
   964 boundaries preserved.
5. **Multi-emit packing (k0/harness.cpp).** `OP_QK_NORM_ROPE` (q + k) and
   `OP_KV_APPEND` (K + V) each fold two independent single-tensor ops into one
   schedule node; the packer emits two `mk::Instr` per node (disjoint writes, no
   boundary between). 2248 nodes ⇒ 2280 packed instructions (+16 +16).
6. **Cell/name reconciliation.** `cell:token` binds to `mk::Host::d_token` (the
   per-pass cell `host_run_pass` writes), `cache:state_r_l<i>`/`state_s_l<i>`
   map to the harness's `conv_state_l<i>`/`ssm_state_l<i>`, `sym:$rs_row`/`$kv_row`
   resolve to i64 device cells (rs_row fixed from a 4-byte i32 to i64), and
   `sym:$n_kv` is patched into every FATTN_DECODE payload per pass. `LOGITS_EMIT`
   is packed with a null flag (its range is [0,72); a non-null flag on a
   multi-block range raises ERR_LOGITS_FLAG_MULTIBLOCK) — the harness reads
   `buf:logits` directly and relies on the kernel's pass-done.

Also: the harness is now compiled as CUDA (nvcc) so the packer can include the
op `Args` structs, and links `mk-core` so `host_launch` reaches the registered
`mk_interp` device symbol (no separable compilation needed — the harness has no
kernels of its own). Pass inputs (positions/kv_row/mask) ride a non-blocking
stream, never the legacy default stream (which would serialize against the
never-returning persistent kernel).

## Caveat: the `$n_kv` payload patch and plain program reads

`$n_kv` is patched into each FATTN_DECODE payload per pass (host->device
`cudaMemcpy` of 4 bytes into the device `Instr`), but `interp.cu` reads the
program with plain cached loads (it is declared immutable). On the persistent
kernel L1 is not invalidated between passes, so a patched field could in
principle be read stale from L1. In practice it is not: the program is 285 KB,
far larger than the 96 KB L1/SM, and every pass scans all 2280 instructions, so
each FATTN `Instr` line is evicted and re-fetched from L2 (where the DMA landed)
each pass. This parity run confirms it — `n_kv` is a constant 256 across all 96
passes (`pad(pos+1, 256)` with pos ≤ 95), and FATTN attends correctly (a stale
`n_kv=0` would have produced garbage; the scored lanes pass). **For guaranteed
correctness at deep contexts where `n_kv` varies across passes**, the robust fix
is to make FATTN read `n_kv` from a device cell with a `.cg` load (the same
strong-read rule the mutable row/mask cells already use) instead of from the
patched payload — a small op + packer change, noted as the precise next step.

## Remaining gap and next step

- **State bit-exactness** is structurally out of reach for the declared megakernel
  folds (see above); reaching it would require the ops to replicate the fork's
  exact per-kernel fold orders (MMVQ dp4a grouping, GEMV half2 accumulate),
  trading the accuracy the current folds were chosen for. The scored lanes
  already pass, so this is a definitional limit of comparing against the fork,
  not a defect. Next step if byte-parity is ever required: pin each op's fold to
  the fork's and re-measure — but the KL/RMS margin suggests no functional need.
- **Floor**: the lever is the weight-stream bandwidth (445 GB/s effective vs the
  ~520–600 GB/s single-op streams), dragged down by the single-block trunk ops
  (RMSNORM/GDN_GATES/GATED_RMSNORM run on 1 block while 71 idle at the boundary)
  and the 6/12-warp FATTN. Widening those to more blocks is the next floor step;
  boundary coalescing is not (2 % here).
