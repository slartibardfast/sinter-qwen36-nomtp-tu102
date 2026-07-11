# Interpreter spine — measured notes

Reconstructed 2026-07-12 from the recovered build (the original agent wrote
the spine but credit-out cut it before verification; numbers below are
re-measured on the rig, both GPUs idle-ish, clocks locked 1455 MHz).

## G11 residency — CLOSES

The full interpreter (`mk_interp`, dispatching all 27 op kinds via
`k0/ops/registry.cuh`) compiles at:

```
_Z9mk_interp... : 104 regs, 0 B static smem, 1 barriers, spills 0/0
```

The 60 KiB slab is DYNAMIC shared memory (opt-in via
`cudaFuncAttributeMaxDynamicSharedMemorySize`), so it does not appear in the
static ptxas smem figure. calx-mill residency at the recorded flags
(`--block-threads 384 --block-smem 61440 --grid-blocks 72`):

```
104 regs, occupancy 19/32 warps (59%) at 384 threads/block,
1 blocks/instance, cooperative grid 72 on 72 instances: fits    (exit 0)
```

The G11 derivation projected ≤168 regs at the 12-warp floor; the real
kernel clears it with headroom (104 regs admits 19 warps). This is the
measured half of the projected-vs-measured pair — G11 is closed for the
k=0 interpreter. (Re-check when the schedule packer inlines heavier ops
into the dispatch; the register count is a property of the compiled switch,
and adding ops or growing an op's live set can move it — the gate re-runs
every build.)

## G15 interpreter overhead — OPEN (finding)

`tests/test_interp.cu`, synthetic program, 256 passes:

```
boundaries only : 403.8 us/pass (dev) -> 806 ns/boundary
nop + boundary  : 582.5 us/pass (dev) -> 1163 ns/boundary (fetch+dispatch +357 ns/instr)
vs Y02 estimate 400 ns, grid.sync comparator 825 ns
```

Two findings:
1. The Y02 counter boundary measures **806 ns**, essentially grid.sync-class
   (825 ns), NOT the ~2× cheaper (~400 ns) the g11-derivation / sync_protocol
   Y02 row projected. A grid barrier is fundamentally two L2 round-trips
   (all-arrive, then all-observe); the 400 ns estimate counted one
   direction. The boundary is the correct primitive; the cost estimate was
   optimistic.
2. The v0 schedule (`k0/compile_schedule.py`) emits **964 boundaries/pass**.
   At 806 ns that is 777 µs = **5.7 % of the 13.71 ms budget** — over the
   G15 2 % gate. Even the plan's ~500-boundary budget lands at 4.25 %.

Path to green (single-GPU-parity milestone work, not a spine defect):
antichain coalescing in the compiler. The trunk is 64 blocks; a well-fused
block needs ~5 boundaries (norm, mixer GEMV, core op, out-proj, residual),
so ~320–380 boundaries is the target. At 806 ns that is 2.3–2.8 % — still
tight, so the boundary itself may also want tuning (elect one arriving
warp; avoid the second `__syncthreads`; overlap the spin with useful work).
Tracked in `plan/0135 mtp-off-build.md`.

## Correctness

- Glue ops (embed → rmsnorm → residual_add → logits_emit) run green through
  the persistent kernel: pass-0 maxrel 0, pass-1 maxrel 2.38e-7 vs the host
  reference; the release-flagged completion (`0xC0FFEE`) is observed each
  pass.
- Error path: a program with an UNWIRED kind (and one with an out-of-range
  kind) sets the device error cell and EXITS the grid — status 1, clean
  exit, no hang. The "unknown kind is never a silent skip" contract holds.
