# NOTES — attention op family (k0/ops/attn.cuh)

Parity harness: `tests/test_attn.cu` (glob-picked as `mk-test-attn`). Built
`cmake -B /tmp/mk-attn -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/mk-attn
-j --target mk-test-attn`; run `CUDA_VISIBLE_DEVICES=0 /tmp/mk-attn/mk-test-attn`
(Quadro RTX 6000 / TU102, sm_75, 72 SMs). All 13 checks pass, exit 0.

References are re-derived from the fork (`software/llama.cpp/autoround`
@546eca8dc), not copied from the header, so a header bug shows as a parity gap.

## Per-op ptxas resources (sm_75, `-Xptxas=-v`, probe kernels)

| op | registers | local(stack) | dynamic smem | spills |
|----|-----------|--------------|--------------|--------|
| OP_QK_NORM_ROPE  | 43 | 32 B | 0 | 0 |
| OP_KV_APPEND     | 22 | 0 B  | 0 | 0 |
| OP_FATTN_DECODE  | 67 | 0 B  | see below | 0 |
| OP_FATTN_REDUCE  | 34 | 0 B  | 0 | 0 |
| OP_ATTN_GATE     | 25 | 0 B  | 0 | 0 |
| mk_run (all five inlined) | 64 | 32 B | — | 0 |

No register spills on any op. The 32-B stack frame on QK_NORM_ROPE is the
`cosf`/`sinf`/`powf` host-ABI scratch, not a spill. All op registers sit under
the interpreter's own count (`mk_interp` compiles to **77 regs** at this pin —
see the dossier note below), so wiring attn does not move the G11 residency
gate.

Dynamic smem is a launch argument, so ptxas reports 0 static smem; only
OP_FATTN_DECODE carves the slab. Declared size (header smem table):
`n_q*256*4 (q_s) + 32*(row_width+2)*2 (KV tile, +2 pad = bank-conflict-free
half2 rows) + 32*2 (mask tile)`:
- production (n_q=12, row_width=512): **45,248 B** — under the 48 KiB
  no-opt-in ceiling and inside the 60 KiB interpreter slab.
- single-kv-head (n_q=6, row_width=256): 22,720 B.

## Indicative bandwidth (DRAM-bound op)

OP_FATTN_DECODE is the family's DRAM-bound op: at decode it streams
`2 x n_kv x row_width x 2 B` of f16 K+V once each (BLOCKS.md: the KV read
exceeds the weight stream past n_kv~256k). Measured full-GPU (72 chunks, one
per SM), n_kv=147,456, 302 MB K+V:

    1.350 ms/pass  ->  224 GB/s (contended-indicative)

~33% of TU102's ~672 GB/s peak. This is the op's *effective* KV throughput,
not a memcpy: each 32-row tile is read then immediately consumed by the
phase-K score dot and phase-V weighted accumulate, and occupancy is capped at
1 block/SM by the 45 KiB slab. Number is contended (shared rig, clocks
unlocked) — a lower bound.

## Correctness summary

- **QK_NORM_ROPE** ≤ 1e-5: worst 4.84e-6 (24 q-heads / stride 512, and 4
  k-heads / stride 256, and pos=0). Reference re-derives the fused RMS norm
  (`norm.cu:136-146`), the IMROPE sector chain (`rope.cu:231-240`, sections
  [11,11,10,0]), the degenerate `rope_yarn` → plain cos/sin at ext_factor=0
  (`rope.cu:26-40`), and the NEOX split-half rotation (`rope.cu:260-264`). The
  header's hardcoded `MK_IMROPE_THETA_SCALE` is **bit-identical** to
  `powf(1e7f,-2/64)` (`rope.cu:443`), asserted in the test — so cos/sin differ
  only by device-vs-libm transcendentals.
- **KV_APPEND** bitwise: f32→f16 RN (`set-rows.cu:160-165`) matches host
  `__float2half`; written row exact, untouched rows unchanged.
- **FATTN_DECODE + FATTN_REDUCE** ≤ 2e-3 head-relative: worst 7.75e-7.
  Reference is a direct (non-tiled) flash attention in double over the *same*
  f16 K/V (q pre-scaled 0.0625 `fattn-mma-f16.cuh:1230`, mask added,
  KQ_max init −FLT_MAX/2 `:1196`; stream-k max/LSE merge + rowsum divide
  `fattn-common.cuh:719-752`). Because the reference consumes identical f16
  values, the f16 quantization is shared and the residual is only the device's
  f32 online-softmax vs double (~1e-6); the 2e-3 envelope (reserved for f16
  effects in the spec) has ~3 orders of margin. Crosses the 256 n_kv pad
  boundary (real 250/256/700 in pads 256/256/768) and a non-32 chunk length
  (768/5 → partial tail tiles). Scoring is per-head-vector-scale, not
  per-element: a near-zero output component (genuine V cancellation, e.g.
  want=1.6e-6) is dominated by ~5e-9 absolute f32 noise and carries no signal,
  so it is scored against the head's peak |output|, not its own magnitude — the
  standard vector-parity criterion. A real merge/index bug perturbs an element
  by ~head-scale and still trips 2e-3.
- **FATTN_REDUCE sentinel** (Y03): a partial whose max slot still reads −inf
  was never written; the op counts it into `*error` and skips it. Verified:
  clean synthetic partials → error 0; one record's max forced to −inf →
  error 1 (detected). Test exits nonzero if undetected.
- **ATTN_GATE** ≤ 1e-5: worst 2.37e-7. `attn * sigmoid(gate)`
  (`unary.cu:48-50`), gate slice at `h*gate_stride` (512).

## Header finding (documented, NOT changed)

**Packed mask read requires even chunk starts (32-aligned in practice).**
`op_fattn_decode` reads the KQ mask two f16 at a time as one 32-bit word
(`attn.cuh:305-310`, `ld_cg((unsigned*)(a.mask + j))`, `j = t0 + 2*threadIdx.x`,
`t0 = kv0 + k*32`). This is aligned iff `j` is even, i.e. iff the chunk start
`kv0 = chunk*clen` is even for every chunk — equivalently `clen` even. Every
correctness config here uses an even chunk length and passes; an **odd** chunk
length (kv0 odd on odd chunks) faults with `misaligned address`. This is a
scheduler contract, not a parity bug: a real schedule tile-aligns chunk
boundaries (clen a multiple of 32), so kv0 is always 32-aligned and the read is
safe. The bandwidth bench tile-aligns its chunking (clen=2048) for this reason.
The header could be made robust by reading the mask one f16 per lane, but that
is a perf-relevant change to a correct-for-production op and was not made. The
mask values themselves (0 attend / −inf masked, pad columns −inf) match the
fork (`llama-kv-cache.cpp:1566-1572`, cast f16 `llama-graph.cpp:2082`).

## Dossier note (not an attn-header correction)

The task briefing states the interpreter is 104 regs; at this pin `mk_interp`
compiles to **77 regs** (`-Xptxas=-v`, `__launch_bounds__(384)`). 77 is lower
(better for the G11 gate), so this is a stale/optimized-away briefing figure for
the interpreter spine, not an attn-family value. No change to attn.cuh.
