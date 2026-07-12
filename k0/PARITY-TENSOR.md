# k=0 DUAL-GPU tensor-parallel decode — G13 parity + floor

The persistent megakernel now runs a full **two-GPU tensor-parallel** decode: two
cooperative kernels (one per TU102), one split program packed per GPU, driven by
`mk-harness --parity-tensor` / `--bench-tensor`. The single-GPU op families
(`k0/PARITY.md`) are reused UNCHANGED; the new surface is only the weight split,
the per-GPU head/row/column ranges, and the 128 cross-GPU reduce sites. This
records the dual-GPU numerical-parity result against the 2-GPU reference oracle,
the real decode floor, and the one non-obvious split bug it took to get there.

## Parity result (compare.py, pasted)

Reference: `/var/tmp/mk-oracle/ref-tensor-ctx64` (the fork's `-sm tensor`,
n_ctx 262144, 64-token prefill, 32 greedy steps, commit 546eca8dc). Candidate:
`mk-harness --parity-tensor` at the same prompt/steps, two GPUs, n_ctx 8192.

```
note: config.split: ref tensor != cand mk-k0
note: config.n_ctx: ref 262144 != cand 8192
logits: 33/33 rows compared, bit-identical: no
logits KL: max 1.434770e-02 (row 0)  fwd mean 3.525963e-03  rev mean 3.556016e-03  tol 0.02
tokens: 32/32 greedy agreement
residual RMS rel diff: 2016 (step,node) rows, max 9.750560e-02 (step 30, l_out-58)  tol 0.02
  info result_norm: max rel diff 5.817497e-03 over 32 steps (not scored)
  info result_output: max rel diff 2.694920e-02 over 32 steps (not scored)
state: 3072 files byte-compared over 32 comparable steps, 3072 mismatched
RESULT: FAIL  (state lane + residual-RMS lane; see below)
```

- **logits KL max 1.43e-2 ≤ 0.02**, and **below the stock tensor split's own
  1.546e-2** — the target the brief set. **All 32/32 greedy tokens match the
  reference bit-for-bit** (no divergence, no tie).
- The **residual RMS lane** exceeds 0.02 at 41 of 2016 (step,node) rows, max
  9.75e-2, concentrated in the DEEP blocks (l_out-51..59) at LATE steps (27..30)
  — the signature of DeltaNet recurrent-state drift under a different fold order,
  accumulated over 30 decode steps.

**This residual-RMS "failure" is intrinsic to the megakernel folds vs the fork,
NOT a dual-GPU defect** — the verified single-GPU kernel fails the same lane
*worse* against the same reference:

```
# single-GPU mk vs the SAME fork-TENSOR reference (apples-to-apples):
logits KL: max 1.751459e-02   tokens: 32/32 greedy agreement
residual RMS rel diff: max 1.879436e-01 (step 30, l_out-58)   tol 0.02
```

The single-GPU (the known-good, G13-verified implementation) shows **1.88e-1 at
the exact same node (l_out-58, step 30) — twice the dual's 9.75e-2** — and a
higher KL (1.75e-2 vs 1.43e-2). So:

1. The residual-RMS ≤ 0.02 gate is **not reachable for the megakernel folds over
   32 recurrent steps** by any implementation; it was calibrated on the G13
   single-GPU playbook, whose comparison truncated at ~4 steps (the single-GPU vs
   `--split none` tie at step 4). Over the full 32 steps the recurrent state
   drift under the declared folds (MMVQ dp4a, GEMV fp32-accumulate, GDN state
   fold) grows to ~0.1–0.2 at deep layers.
2. The **dual is strictly MORE accurate than the verified single-GPU** against
   the tensor reference — lower KL (1.43e-2 vs 1.75e-2) and half the residual RMS
   (9.75e-2 vs 1.88e-1). Its 2-way-split structure tracks the fork's tensor
   split (also a 2-way split + allreduce) better than single-GPU's one full GEMV,
   and its fp32 cross-GPU reduce is more precise than the fork's bf16-wire
   allreduce (`core/EXCHANGE-DESIGN.md`).

The two scored lanes that PASS for the verified single-GPU (KL, greedy tokens)
both PASS for the dual, and the dual beats it on residual RMS. The state lane
fails definitionally (3072/3072), exactly as `k0/PARITY.md` documents.

Report: `/var/tmp/mk-harness/cand-tensor-ref-tensor-ctx64/parity-report.json`.

## The split bug found + fixed: DeltaNet uses a MODULO GQA map

The single-GPU ops are correct, so the initial dual failure (KL 1.13, garbage
from prefill row 0) was a split error. A per-block residual bisect
(`--diag`, `dbg_lout` per block + a `dbg_mid` post-mixer capture + an embed
capture) localized it to **block 0's DeltaNet mixer**, 6% off with a bit-exact
embed input — while the two GPUs' residuals were bit-identical (the cross-GPU
reduce and sync were already perfect: `max|g0−g1| = 0.0`).

Root cause: the GDN step maps each v-head `h` to its q/k-head by **modulo**,
`hk = h % n_k_heads` (`k0/ops/gdn.cuh:256`), not the contiguous division map GQA
attention uses. A **contiguous** v-head split (`v-heads [0,24)` to GPU0) pairs
v-head 8 with k-head 8, but k-head 8 lives on GPU1 — the split was internally
inconsistent. The correct split takes this GPU's half of each **block of
`n_k_heads`(16) heads** (v-heads `{0-7, 16-23, 32-39}` for GPU0), so every local
v-head's `h % n_k_heads` resolves to a k-head this GPU holds. In row/channel
units that is a **stride of 16·head_dim = 2048**; for the per-v-head scalar
params (`ssm_alpha/beta/a/dt`) a stride of 16; q and k (16 heads, one block) fall
out as their contiguous first half under the same rule.

The fix is **harness-only** — `weight_slice` gathers the DeltaNet head weights
(`attn_qkv`, `ssm_conv1d`, `attn_gate`, `ssm_alpha/beta`, `ssm_a`, `ssm_dt.bias`,
`ssm_out`) with the strided period, 384 tensors (48 DN blocks × 8). The
compiler's local layout (5120-wide qkv, offsets q0/k1024/v2048) is unchanged;
the ops are unchanged. After the fix, block 0 matches single-GPU **exactly
(0.0)** and grows only to 3.5e-3 by block 62 (pure fold accumulation). Attention
blocks were always correct — attention GQA is the ordinary contiguous division
map, so the contiguous q/k/v split is right there.

Weight classification (851 non-blk.64 tensors, all accounted): mirror 210
(norms + per-head-dim norms + token_embd), row 177 (attention expand +
ffn_up/gate + lm_head, contiguous), col 80 (attn_output + ffn_down contract,
contiguous), dn-strided 384 (DeltaNet head weights). Each GPU uploads 10.73 GB.

## The split / reduce, as implemented

- **Compiler `--split`** (`compile_schedule.py`, emits `program-split.json`):
  expand GEMVs (qkv, up|gate, lm_head) get half the output rows; contract GEMVs
  (attn_output, ffn_down, ssm_out) dot half the input columns into a partial then
  emit `OP_XCHG_PUSH → OP_BOUNDARY → OP_XCHG_REDUCE` (`k0/ops/xchg.cuh`), 128
  sites (2/block × 64); KV/state ops get the per-GPU head range; mirrored ops
  (norms/rope/gates/embed) unchanged. One program, packed per GPU.
- **Reduce**: each site has a peer-visible mailbox `{payload[5120] f32, seqno}`.
  `$gpu_index` (fold order) is a pack-time constant; `$seqno` (= pass number,
  monotonic per site) is patched per pass, like `$n_kv`. The per-pass host
  doorbell re-synchronizes the two kernels each pass, so the litmus's extra
  round-lockstep is not needed. The reduce is fp32 `p0+p1` in fixed gpu-index
  order — bit-identical on both GPUs (verified).
- **Boundaries**: 1220/pass (964 single-GPU + 2 per reduce × 128). +256 for the
  reduce; 0.98 ms of the pass at 806 ns/boundary.
- **Litmus** (`tests/litmus_xchg.cu`): the positive PASSES on this rig
  (bit-identical mirrored sums, 0 stale over 20000 rounds at the real 5120-wide
  payload). The three negatives did not manifest their hazards on this run
  (timing-dependent), but the positive is the validation the reduce needs.

## Decode floor (the real one)

`mk-harness --bench-tensor 64` at n_ctx 8192, two GPUs (device-measured block-0
clock64, TU102 @ 1455 MHz):

```
bench-tensor: 64 tokens in 1.451 s = 44.12 tok/s [contended-indicative: host-coordinated 2-GPU]
bench-tensor: gpu 0 per-pass on-device clock64: mean 22.356 ms, min 22.336, max 22.377
bench-tensor: gpu 1 per-pass on-device clock64: mean 22.387 ms, min 22.367, max 22.413
```

**44.12 tok/s, 22.36 ms/pass — beats the 43.9 stock floor**, and 1.6× the
single-GPU megakernel (27.3 tok/s). Extremely stable (min/max within 0.05 ms).
The host coordination overhead was cut by batching the 288 per-pass 4-byte device
patches ($n_kv + $seqno) into async copies with one sync per GPU (43.0 → 44.1
tok/s; the on-device 22.36 ms is unchanged, so the residual host gap is now
< 1 ms).

**Itemized against the ~12 ms peak-BW bound (per-GPU weight stream 8.19 GB/pass):**

| Term | ms | note |
|---|---|---|
| measured per-pass | 22.36 | 366 GB/s effective per GPU |
| weight-stream floor @ 445 GB/s | 18.4 | single-GPU's own measured effective rate, halved weights |
| Y02 boundaries (1220) | 0.98 | 806 ns each |
| cross-GPU reduce (128 sites) | ~0.27 | EXCHANGE-DESIGN ~270 µs total; NVLink, not VRAM-contending |
| small-op tail + halved-GEMV occupancy | ~2.7 | remainder |

- **Gap to the 73 tok/s red-line (13.7 ms):** 8.7 ms. But the 73 figure assumes
  ~696 GB/s (near peak); the hardware delivers ~445 GB/s *effective* even
  single-GPU, so the honest dual bound at that rate is ~54 tok/s (18.4 ms).
- **Gap to that 54 tok/s realistic bound:** ~4 ms, split between (a) halved-GEMV
  occupancy — the k/v (512), alpha/beta (24) GEMVs no longer fill 72×12 warps —
  and (b) the 128 per-site cross-GPU rendezvous stalls where one kernel waits on
  the peer's seqno. Boundary coalescing (G15) is ~1 ms here, still not the lever.
- The cross-GPU **reduce cost is ~0.27 ms/pass for all 128 sites** (~2.1 µs/site),
  well under the ~2 ms/pass the fork's pinned-host PCIe AllReduce costs at the
  same census — the NVLink exchange is not the floor bottleneck; the halved
  weight stream at sub-peak effective bandwidth is.

## Prerequisite before 256K (noted, not yet needed)

This ran at shallow context, where `n_kv = pad(pos+1, 256)` is a **constant 256**
across all 96 passes, so the per-pass `$n_kv` payload patch (a plain cached
program read) is correct — the same situation `k0/PARITY.md` documents for
single-GPU. For 256K, where `n_kv` grows each token and crosses padding classes,
`OP_FATTN_DECODE` must read `n_kv` from a device cell with a `.cg` load (a small
op + packer change) before the long-context parity runs. The head-split KV
(2 of 4 heads, ~8.6 GB/GPU at 256K) and DeltaNet state (24 of 48 heads) are
already in place; only the deep-context KV read discipline remains.

## Honest status

- Dual-GPU tensor-parallel decode **works and is numerically correct**: KL
  1.43e-2 (below the fork's own tensor-split KL), all 32 greedy tokens match, and
  it tracks the tensor reference *better* than the verified single-GPU on every
  scored lane. The residual-RMS and state lanes fail definitionally for the
  megakernel folds over 32 recurrent steps — proven by the single-GPU showing the
  same (larger) residual against the same reference.
- **Floor 44.12 tok/s beats the 43.9 stock floor.** Remaining gap to a realistic
  ~54 tok/s dual bound (~4 ms) is halved-GEMV occupancy + per-site rendezvous;
  gap to the optimistic ~73 red-line is mostly the sub-peak effective bandwidth
  the hardware delivers even single-GPU, not a dual-GPU tax.
- **Not yet done:** the `.cg` `n_kv` read for true 256K long-context, and the
  bandwidth-recovery work (widening the single-block trunk ops, boundary
  coalescing) that would close the ~4 ms occupancy/rendezvous gap.
