# RED-LINE: itemizing the dual-GPU k=0 decode pass

Goal (charter): itemize where the dual-GPU shallow decode pass spends its
22.35 ms, close the closable overhead toward the weights-only DRAM red-line
(~13.71 ms / ~73 tok/s), and report the **measured, itemized true bound** with
parity green at every step.

All numbers are the **dual-GPU tensor-parallel** program (`program-split.json`,
1540 compute + 1220 boundary instrs/GPU), both TU102s clock-locked at 1455 MHz
and otherwise idle, shallow context (`--pos0 0`, n_ctx 8192). Per-pass op times
are block-0 `clock64` accumulations under an `MK_PROFILE` build
(`core/interp.cu`, guarded by a null `Control::op_cycles`; production is
unaffected — verified 78 regs, unchanged). Floor numbers are the production
(non-profile) build, mean of 3 runs of 300 timed passes.

## 1. Itemization (BEFORE) — where the 22.35 ms goes

Block-0 per-kind breakdown, GPU 0, 300 passes. On-device pass 22.48 ms;
itemized sum 21.38 ms; the 1.10 ms remainder is interpreter loop/dispatch
(per-instruction fetch+decode over 2760 instrs, not inside any op or boundary).

| kind | ms/pass | % pass | note |
|------|--------:|-------:|------|
| MMVQ_Q4_0        | 6.16 | 28.8% | q/k/v, o-proj, z-gate, ffn_down, qkv GEMVs (weight-stream) |
| MMVQ_Q4_0_FUSED  | 5.85 | 27.4% | ffn up\|gate fused GEMV (weight-stream) |
| HEAD_GEMV_F16    | 2.13 |  9.9% | lm_head vocab GEMV, F16 (weight-stream) |
| **RMSNORM**      | **1.60** | **7.5%** | **trunk norms — running on 1 SM, 71 idle (see §2)** |
| BOUNDARY         | 1.29 |  6.0% | 1220 grid-sync boundaries |
| MMVQ_AR16        | 1.16 |  5.4% | ssm_out projection, Q4_0_AR16 (weight-stream) |
| FATTN_REDUCE     | 0.65 |  3.1% | split-KV online-softmax merge (12/72 blocks active) |
| XCHG_PUSH        | 0.53 |  2.5% | cross-GPU peer store (128 sites) |
| FATTN_DECODE     | 0.46 |  2.2% | split-KV flash-attn decode |
| XCHG_REDUCE      | 0.42 |  2.0% | cross-GPU fp32 fold (128 sites) |
| GDN_STEP         | 0.39 |  1.8% | DeltaNet delta-rule recurrence |
| QUANT_Q8_1       | 0.24 |  1.1% | activation quant before each MMVQ group |
| GATED_RMSNORM    | 0.15 |  0.7% | ssm_norm per head + silu(z) gate |
| QK_NORM_ROPE     | 0.12 |  0.6% | per-head rmsnorm + IMROPE |
| GDN_GATES        | 0.09 |  0.4% | softplus/sigmoid gate math |
| SSM_CONV_SILU    | 0.05 |  0.2% | causal conv + silu |
| QK_L2NORM        | 0.04 |  0.2% | |
| KV_APPEND        | 0.03 |  0.2% | |
| ATTN_GATE        | 0.02 |  0.1% | |
| LOGITS_EMIT / EMBED_LOOKUP | ~0 | ~0% | |
| *loop/dispatch (unattributed)* | 1.10 | 4.9% | 2760 instr fetch/decode |

Cross-checks: boundary total 1.29 ms; reduce total (push+reduce) **0.95 ms**
(the design doc's 0.27 ms counted only the fold, not the push+signaling); the
on-device sum (21.38 ms) + loop overhead (1.10 ms) = the `pass_cycles` wall
(22.48 ms), and the wall (22.37 ms production, 1/44.16 s = 22.65 ms) confirms
host coordination is a negligible ~0.28 ms.

### What the itemization tells us

- **The matmuls are NOT the problem.** The four weight-streaming GEMVs total
  **15.30 ms** for ~8.35 GB/GPU of weights = **~546 GB/s = 90% of the 608.6
  GB/s roofline** (HEAD_GEMV_F16 alone is ~98%). The GEMVs are ~1.6 ms above
  the weights-only red-line, which is the q8-activation traffic + the practical
  MMVQ efficiency ceiling on sm_75. There is no large matmul win.
- **The gap is overhead**, and the single biggest overhead item is **RMSNORM at
  7.5 %** — larger than every boundary combined. That is the top lever.

## 2. Change log (each lever: parity-guarded + re-measured)

Parity guard = the **output lanes** (logits KL ≤ 0.02, greedy tokens unchanged)
of `--parity-tensor ref-tensor-ctx64` + `compare.py`. (The residual-RMS and
state lanes fail against that reference by design — the megakernel's fp32 NVLink
reduce vs the fork's bf16 PCIe AllReduce, `PARITY-TENSOR-VERIFY.md`; they are
not the guard.)

### Lever A — RMSNORM: stop re-reading, vectorize the write  ✅ landed

**Finding.** `op_rmsnorm` strides *rows* over blocks; the trunk norms are a
single 5120-row (`nrows=1`), so **only block 0 works while 71 SMs sit idle**.
Worse, the v0 write loop re-read `x`+`add` from L2 **scalar** and wrote
`y`/`sum`/`dbg` scalar — ~4 latency-bound passes over 5120 elems on one SM.

**Change** (`k0/ops/glue.cuh`, op-local, no schedule/ABI change): stage the
summed row `(x+add)` into the smem slab during the reduction, then write
**vectorized (float4) from smem** — one memory read of `x+add`, no re-read,
float4 stores of `sum`/`dbg`/`y`. Same fold order, same scale, same values.

| metric | before | after |
|--------|-------:|------:|
| RMSNORM ms/pass | 1.60 | **0.73** |
| floor (tok/s, mean of 3) | 44.16 | **45.95** |
| pass ms (on-device, mean of 3) | 22.37 | **21.48** |
| G11 residency | 78 regs, fits | 78 regs, fits |

**Parity — bit-identical** (candidate vs a baseline binary with the original
op, direct dump compare, both configs):

```
=== CANDIDATE(new rmsnorm) vs BASELINE(orig rmsnorm), DUAL-GPU ===
logits: 33/33 rows compared, bit-identical: yes
logits KL: max 0.000000e+00 ... tokens: 32/32 greedy agreement
residual RMS rel diff: max 0.000000e+00
state: 3072 files byte-compared, 0 mismatched
RESULT: PASS
=== CANDIDATE vs BASELINE, SINGLE-GPU ===
logits: bit-identical: yes ... KL 0.0 ... tokens 32/32 ... state 0 mismatched
RESULT: PASS
```

And against the oracle (the actual guard), unchanged from baseline:
`logits KL: max 1.434770e-02 (tol 0.02); tokens: 32/32 greedy agreement`.

### Lever B — boundary coalescing (compile_schedule.py)  ⚠️ correct, but 0 merges

Implemented the provably-safe adjacent-window coalescer the charter and
`COMPILER.md` asked for: two adjacent antichain windows merge iff their union
stays an antichain (no WAR/WAW/RAW across the union of each window's
`reads`/`writes`), with an occupancy guard (never put two heavy weight-streaming
ops in one window) and XCHG sites excluded (their boundary is a cross-GPU
handshake the local read/write sets don't model). The downstream lifetime check
re-validates the merged windows.

**Result: 1220 → 1220 boundaries — zero merges, at single-GPU and dual-GPU.**

An exhaustive scan confirms it: **0 of the 1219 adjacent window pairs are
independent.** The v0 schedule is a genuine, fully-serial dependency chain. It
is not "a boundary between nearly every *dependent* pair" — nearly every pair is
*actually* dependent, because logically-independent ops **false-share the small
set of reused scratch buffers** (`xn`, `q8_act`, `mixer_out`, `attn_out`,
`proj_out`, `gdn_*`). E.g. `state_store` is truly independent of the rest of the
block but writes into a window whose sibling (`gated_out_norm`) feeds the next
op, so the whole window is pinned.

So the boundary total (1.29 ms) is **data-dependency-required, not gratuitous.**
`COMPILER.md`'s "target ~320-380 boundaries" assumed an independence that
buffer-reuse eliminates. Reducing boundaries would require **buffer renaming**
(distinct scratch buffers so independent ops stop false-sharing) plus
reordering — a large change touching the verified op structure, at real parity
risk, for a ≤1.29 ms ceiling. Not taken. The coalescer stays in (correct,
future-proof: it will fire the moment the schedule stops false-sharing).

### Levers considered and declined (honest)

- **RMSNORM multi-block** (split the reduction across SMs): needs a cross-block
  reduce → a boundary per norm, and the in-place residual write-back (`sum==x`)
  is a cross-block WAR that *forces* that boundary. Net ≈ −0.2 ms after +129
  boundaries, at fold-order/parity risk. The single-block smem win already
  captured the clean part.
- **FATTN_REDUCE / FATTN_DECODE** (1.12 ms): dominated by the *fixed* 72-way KV
  split at shallow context — most of the 72 chunks are empty, 60/72 SMs idle in
  the reduce. Closable only with a **depth-adaptive split count** (patch
  `block_hi` from `n_kv` per pass) — a runtime change with 32-alignment and
  partial-layout correctness subtleties. Real work, medium risk, ~0.5 ms.
- **XCHG reduce** (0.96 ms): 128 NVLink fold sites; the `membar.sys` + seqno
  ordering is marked load-bearing and gated on the cross-GPU litmus. Not
  weakened.

## 3. Achieved floor + itemization (AFTER)

Production, mean of 3 (`45.90 / 46.02 / 45.94` tok/s; on-device `21.48 / 21.46 /
21.49` ms):

- **45.95 tok/s, 21.48 ms/pass** (from 44.16 tok/s, 22.37 ms) — **+1.79 tok/s
  (+4.1 %), −0.89 ms/pass**, parity bit-identical.

AFTER itemization (GPU 0, block-0), on-device sum 20.47 ms + 1.0 ms loop:

| kind | ms/pass | Δ vs before |
|------|--------:|------------:|
| GEMVs (Q4_0 + FUSED + HEAD + AR16) | 15.28 | ~0 |
| BOUNDARY | 1.29 | 0 |
| **RMSNORM** | **0.73** | **−0.87** |
| XCHG (push+reduce) | 0.96 | 0 |
| FATTN (decode+reduce) | 1.12 | 0 |
| GDN_STEP | 0.37 | 0 |
| QUANT_Q8_1 | 0.23 | 0 |
| other small ops | ~0.49 | 0 |
| loop/dispatch | ~1.0 | 0 |

## 4. The TRUE ITEMIZED BOUND

Red-line (weights-only DRAM, 8.35 GB/GPU ÷ 608.6 GB/s) = **13.71 ms**.
Achieved = **21.48 ms**. Every millisecond of the **7.77 ms** residual gap,
attributed to a measured cause:

| residual over red-line | ms | closable? |
|------------------------|---:|-----------|
| GEMVs above the weights-only roofline (q8 activation traffic + ~90 % MMVQ efficiency on sm_75) | 1.57 | **No** — practical matmul ceiling |
| BOUNDARY — 1220 grid-syncs required by genuine data dependencies | 1.29 | Hard — needs buffer renaming to break false-sharing |
| Interpreter loop/dispatch over 2760 instrs | 1.00 | Hard — tied to instruction count (same blocker as boundaries) |
| XCHG cross-GPU fp32 reduce — 128 NVLink sites, structural to the tensor split | 0.96 | Risky — protocol-level |
| RMSNORM — single-SM reduction floor (in-place residual hazard blocks lock-free multi-block) | 0.73 | Marginal (~0.2 ms, +129 boundaries) |
| FATTN decode+reduce — fixed 72-way KV split, mostly empty at shallow ctx | 1.12 | **Yes, ~0.5 ms** — depth-adaptive split count (medium risk) |
| GDN_STEP + QUANT + per-head glue | ~1.10 | Small, spread thin |
| **total gap** | **7.77** | |

### Honest summary

- **Irreducible core (~13.7 ms):** the weight stream. The GEMVs already run at
  90-98 % of the DRAM roofline; the k=0 red-line is a bandwidth wall, and the
  matmuls are essentially against it.
- **Structurally required (~3.3 ms):** the 1220 dependency-chain boundaries
  (1.29), the per-instruction dispatch (1.0), and the cross-GPU reduce (0.96).
  These are the cost of *this* interpreter+split design; cutting them means
  changing the design (buffer renaming to shorten the chain, a fatter
  instruction to cut count, a tighter reduce protocol), not tuning it.
- **Genuinely closable with more work (~0.7-1.0 ms):** a depth-adaptive FATTN
  split count (~0.5 ms, the biggest remaining honest lever) and a boundary-split
  RMSNORM (~0.2 ms). Neither was taken here — each carries correctness/parity
  risk that the charter's parity guard rightly outweighs at this margin.
- **Landed this stage:** the RMSNORM 1-SM fix, **44.16 → 45.95 tok/s**, proven
  **bit-identical** in both configs. It closes ~11 % of the 8.6 ms gap
  (0.89 of 8.66 ms) with zero numerical change.

The true bound today is **21.48 ms / 45.95 tok/s**, and the ~1 ms still closable
lives in the attention split, not the matmuls.
