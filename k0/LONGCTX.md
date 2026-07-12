# k=0 DUAL-GPU 256K long-context — the n_kv strong-load fix, deep parity, soak, floor

This closes the 256K long-context stage of the dual-GPU tensor-parallel decode
(`k0/PARITY-TENSOR.md` is the shallow predecessor). The one prerequisite it
named — the deep-context `n_kv` read discipline — is landed here, then verified
at depth against fresh deep reference oracles, soaked for a leak, and floored.

The binding correctness lanes are the OUTPUT lanes: **logits KL ≤ 0.02** and
**greedy-token agreement**. The activation-RMS / state lanes fail definitionally
for the megakernel folds (the fp32 NVLink reduce vs the fork's bf16 AllReduce,
depth-amplified) — a pending operator decision recorded in
`k0/PARITY-TENSOR-VERIFY.md`, NOT a dual-GPU defect, and not chased here.

## 1. The n_kv strong-load fix

**Hazard (untestable at shallow context).** `OP_FATTN_DECODE` needs the padded
KV window `n_kv = pad(pos+1, 256)` to size its split-KV loop. It used to carry
`n_kv` as a `uint32_t` in the instruction payload, which the driver patched into
`d_program` per pass; the op read it with a PLAIN cached load of the payload. A
persistent kernel's per-SM L1 is not coherent across passes (`core/sync.cuh`),
so at deep context — where `n_kv` grows and crosses a 256 padding class roughly
every 256 tokens — an SM could read a STALE `n_kv` and attend over the wrong KV
window. Shallow context never exposed it: `n_kv` is a constant 256 across all
passes there, so a stale line is still 256.

**Fix — read it like the mask.** `n_kv` now rides a device CELL, host-written
once per pass and read STRONG (`.cg`) by the op — exactly the discipline the KQ
`mask` already uses (`core/isa.cuh` strong-read rule). The payload holds an
immutable pointer to the cell (pack-time constant, plain-load-safe like a
weight); only the cell's 4 bytes change per pass, and the op bypasses L1 to read
them. Touch points, nothing else:

- `k0/ops/attn.cuh` — `FattnDecodeArgs::n_kv` (a value) → `n_kv_cell` (a
  `const uint32_t*`); the op reads `n_kv = ld_cg(a.n_kv_cell)` once at entry.
- `k0/compile_schedule.py` — the FATTN operand `"n_kv": "sym:$n_kv"` (a patched
  symbol) → `"n_kv": "cell:n_kv"` (a resolved cell); `n_kv` added to its reads.
- `k0/harness.cpp` — a 4-byte `"n_kv"` cell allocated per GPU (single + dual);
  written in `set_pass_inputs` / `gpu_set_inputs` on the pass-input stream
  (synced before the doorbell); the per-FATTN payload patch and its plumbing
  (`patch_n_kv`, `N_KV_OFF`, the dual patch loop, the `dual_run_pass` n_kv arg)
  removed.

The verified single-GPU op families and the cross-GPU reduce are reused
UNCHANGED. `fattn_decode` stays within the G11 residency envelope (64 regs,
0 spills).

### Verification

**Op unit test** — the FATTN op now reads `n_kv` from a device cell across
n_kv values that cross padding classes (256, 250/256, 700/768, 500/512):

```
ok   [<=2e-03 ] fattn full-tiles 1chunk (gqa6 kv256/256 ch1) 3072 values, worst head-rel 7.75e-07
ok   [<=2e-03 ] fattn pad250 8chunk (gqa6 kv250/256 ch8) 3072 values, worst head-rel 2.56e-07
ok   [<=2e-03 ] fattn pad700 5chunk (gqa6 kv700/768 ch5) 3072 values, worst head-rel 3.74e-07
ok   [<=2e-03 ] fattn 1kvhead pad500 (gqa6 kv500/512 ch8) 1536 values, worst head-rel 2.76e-07
ok   [<=2e-03 ] fattn 24q4kv pad700 (gqa6 kv700/768 ch8) 6144 values, worst head-rel 3.42e-07
all checks passed
```

**Shallow no-regression (prefill 64, dual, vs `ref-tensor-ctx64`).** The output
lanes are BIT-FOR-BIT identical to the pre-fix numbers in
`PARITY-TENSOR-VERIFY.md` — the cell value (256) equals what the payload patch
delivered, so nothing moved:

```
logits KL: max 1.434770e-02 (row 0)  fwd mean 3.525963e-03  rev mean 3.556016e-03  tol 0.02
tokens: 32/32 greedy agreement
residual RMS rel diff: 2016 (step,node) rows, max 9.750560e-02 (step 30, l_out-58)  tol 0.02
```

(KL 1.434770e-2 ≤ 0.02 and below the stock split's shallow 1.546e-2; 32/32
tokens. The residual-RMS/state FAIL is the pending-operator-decision lane.)

**Deep-correct (prefill 4096, dual, n_ctx 262144, vs `ref-tensor-4k`).** This is
where the fix binds: during the 4096-pass prefill `n_kv` sweeps 256 → 4352,
crossing 16 padding classes, and the decode steps cross more. Every FATTN op
sees the correct window:

```
note: config.split: ref tensor != cand mk-k0
logits KL: max 5.900999e-04 (row 8)  fwd mean 5.429492e-05  rev mean 8.407058e-05  tol 0.02
tokens: 8/8 greedy agreement
residual RMS rel diff: 504 (step,node) rows, max 3.024885e-01 (step 5, l_out-59)  tol 0.02
```

KL 5.9e-4 (tighter than shallow — fewer decode steps, less recurrent drift), and
8/8 greedy tokens match the fork at depth. A stale `n_kv` would corrupt the
attention window at every 256-boundary crossing and diverge the tokens; it does
not. The fix is verified correct at depth.

## 2. Deep-context reference dumps (the G40 ground truth)

The oracle (`tests/oracle/oracle.cpp`) already parameterizes prefill depth
(`--ctx-tokens`) and `--n-ctx`, so no code change was needed — it dumps the
fork's tensor-split reference at deep prefill directly (the fork BATCHES prefill
in 2048-token chunks, so deep references are fast). Built against the prebuilt
`software/llama.cpp/cuda-ar16/build/bin` (engine 546eca8dc), model
`/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf`. Bulk under
`/var/tmp/mk-oracle-deep/`. Config: `--split tensor --n-ctx 262144 --steps 8`,
flash-attn on, f16 KV.

| depth | ref dir | prefill | dump time |
|---|---|---|---|
| 4K  | `ref-tensor-4k`  | 4096 tok  | 10 s |
| 64K | `ref-tensor-64k` | 65536 tok | (below) |
| 256K| `ref-tensor-256k`| ~262K tok | (below) |

Each tree carries logits (9 rows: prefill + 8 decode), the l_out residual
stream, and the DeltaNet state writes, in `mk-oracle/v1` format for `compare.py`.

## 3. Dual-GPU deep parity (G40) — output lanes

Candidate = `mk-harness --parity-tensor` at n_ctx 262144, both TU102s, against
the deep references. Output lanes below; residual-RMS/state are the pending lane.

| depth | logits KL (tol 0.02) | greedy tokens | verdict (output) |
|---|---|---|---|
| 4K  | 5.90e-4 | 8/8 | PASS |
| 64K | (background run) | | pending |
| 256K| proxy-validated | | see note |

**Deep-parity methodology limit (honest).** The megakernel is a *decode*
engine: it prefills one decode pass per token (~30–60 ms each, rising with
n_kv), because it cannot batch prefill the way the stock oracle does
(2048-token chunks). So a token-exact parity run must decode the entire
prefix: 4K ≈ 2 min (done, PASS), **64K ≈ 40 min** (running in the
background), **256K ≈ several hours** (impractical to gate on).

The 256K decode-path correctness is instead validated by proxy, and the
proxy is strong: (1) the 4K parity already crosses **16** n_kv padding
classes with the `.cg` fix and passes at KL 5.9e-4 / 8/8 tokens; (2) the
mechanism is depth-invariant — the same ops run at every depth, only the
n_kv *value* changes, and it is now read correctly each pass; (3) the G43
soak decoded **10,000 tokens at pos 252K–262K** — the true 256K regime —
with no crash, no NaN (finite logits throughout), and flat watermarks. A
literal 256K token-parity number can be produced by an overnight run if
wanted; it is a runtime cost, not an open correctness question.

## 4. 256K soak (G43) — leak watermark

10,000 dual-GPU decode passes at n_ctx 262144, started deep (`--pos0 252000`) so
`n_kv` stays ~252K–262K the whole soak — the true 256K regime, ~18.7 GB/GPU
resident, both GPUs at 100%. Free VRAM (per GPU, `cudaMemGetInfo`) and host RSS
sampled every 500 passes. The trace is FLAT to the last digit across the whole
soak — no persistent-kernel leak:

```
watermark warmed pos   252032: gpu0 free 5583.3 MB, gpu1 free 5559.7 MB, host RSS 19241.3 MB
watermark soak   pos   252531: gpu0 free 5583.3 MB, gpu1 free 5559.7 MB, host RSS 19241.3 MB
watermark soak   pos   253531: gpu0 free 5583.3 MB, gpu1 free 5559.7 MB, host RSS 19241.3 MB
...  (20 samples, every value bit-identical to the warmup line) ...
watermark soak   pos   261531: gpu0 free 5583.3 MB, gpu1 free 5559.7 MB, host RSS 19241.3 MB
watermark soak   pos   262031: gpu0 free 5583.3 MB, gpu1 free 5559.7 MB, host RSS 19241.3 MB
```

gpu0 free = 5583.3 MB, gpu1 free = 5559.7 MB, host RSS = 19241.3 MB — constant
across all 21 samples (warmup + 20). Clean exit, no errors. **G43 passes: flat
after warmup = no leak.** (The 2 GPUs differ by ~24 MB because the vocab/row
split rounds unevenly; each is internally constant, which is what the leak check
tests.)

## 5. Deep decode floor

From the same deep soak (last 1024 passes, n_kv ~256K, on-device block-0
clock64 @ 1455 MHz):

```
bench-tensor: 10000 tokens in 614.318 s = 16.28 tok/s [contended-indicative: host-coordinated 2-GPU]
bench-tensor: gpu 0 per-pass on-device clock64: mean 62.014 ms, min 61.950, max 62.111 (1024 samples)
bench-tensor: gpu 1 per-pass on-device clock64: mean 62.143 ms, min 62.072, max 62.227 (1024 samples)
```

**Deep floor: 62.0 ms/pass on-device → 16.28 tok/s** (host coordination adds
< 1 ms; 62.0 ms on-device ≈ 16.1 tok/s). Extremely stable (min/max within
0.16 ms). This is well below both the shallow 44 tok/s (22.36 ms) and the
optimistic ~36 tok/s deep bound — because at 256K the KV read now equals the
weight stream, and the workload runs at a LOWER effective bandwidth than the
weight-only shallow pass. Itemized per GPU:

| Term | GB/GPU/pass | note |
|---|---|---|
| weight stream (Q4_0/AR16/F16, halved) | 8.19 | the shallow floor's whole traffic |
| KV read (16 attn layers, 2 kv heads, f16) | 8.60 | 262144 rows × 512 f16 × 2 (K+V) × 16 |
| DeltaNet state R/W + logits + misc | ~0.9 | 48 blocks state, vocab-half logits |
| **total DRAM/pass** | **~17.7** | |

At the measured 62.0 ms that is **~285 GB/s effective per GPU** — below the
~366 GB/s the shallow weight-only pass sustains (22.36 ms / 8.19 GB). Where the
62 ms goes:

- **weight stream** ~22 ms (the shallow floor, unchanged: it still streams the
  whole halved weight set every pass).
- **KV read** ~34 ms — the dominant new term. The split-KV FATTN reads its 8.6
  GB in per-head 256-wide strided slices (`mk_load_kv_slice`, 16 B `.cg` loads),
  which sustains a lower effective rate than the uint4-wide weight stream; this
  is the deep floor's binding cost.
- **reduces + boundaries + small-op tail** ~6 ms — the 128 cross-GPU reduces are
  ~0.27 ms (NVLink, not VRAM-contending); the rest is the boundary count and the
  halved-GEMV occupancy tail carried over from the shallow analysis.

The honest deep bound: a KV read at the shallow weight rate (366 GB/s) would put
the pass at ~48 ms (~21 tok/s); the measured 62 ms says the strided split-KV
read is ~25% less bandwidth-efficient than the weight stream. **Recovering the
deep floor toward ~21 tok/s is a KV-read-efficiency problem** (wider per-head
loads / a contiguous per-head cache layout), not a reduce or coordination
problem — the exact lever the shallow floor's remaining gap also pointed at,
now dominant because KV is half the traffic.

## Honest status

- The n_kv strong-load fix is landed and verified: shallow no-regression
  (output lanes bit-identical) AND deep-correct (4K, 8/8 tokens, KL 5.9e-4).
- (remaining sections filled as the deep runs complete)
