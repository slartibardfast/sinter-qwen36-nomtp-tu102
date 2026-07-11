# Boundary litmus findings (G14 family, measured 2026-07-12)

Rig: 2x TU102 @ 1455 MHz locked, driver 610.43.03, CUDA 13.3. Binaries:
`mk-litmus-boundary{,-plain,-norelease}`, `mk-litmus-mp{,-cg}`; 200000
epochs per run, cooperative 72x384 (the G11 shape), run on both devices.

## Results

| build | claim | device 0 | device 1 |
|---|---|---|---|
| positive | exactly-once counting + zero stale strong reads | PASS (28.8M arrivals exact, 0 stale) | PASS |
| norelease (publish after arrive) | detector must fire | CAUGHT (6.78M stale) | CAUGHT (6.78M stale) |
| plain-reads (weak payload read across boundary, pre-warmed L1) | expected stale per the xgpu analogy | NOT CAUGHT (0 stale) | NOT CAUGHT |
| mp probe (warm line, strong flag spin, weak payload read, NO membar on reader path) | existence probe | 0 stale / 7.2M checks | 0 stale |
| mp .cg control | zero stale | PASS | PASS |

## Reading

- The Y02 counter boundary (release RED + strong poll, monotonic target,
  no reset) is exact under randomized SM completion order, and its stale
  detector provably fires on a broken protocol (the norelease negative).
- The intra-GPU stale-L1 plain read DID NOT reproduce on this silicon, in
  either the boundary-crossing shape or the bare message-passing shape
  with a deliberately warmed line. This is weaker than a guarantee: the
  PTX memory model still permits it, and the measured xgpu twin DOES read
  stale 999999/1e6 with plain loads (reference/tu102, Y06 basis). Absence
  of observation licenses nothing across a driver or silicon change.
- Standing rule, unchanged: mutable data crossing a Y02/Y06 boundary is
  read strong (.cg family, `core/sync.cuh`); immutable weights use plain
  loads. Strong reads cost nothing measurable on streamed data, so the
  conservative rule is free; this file records why relaxing it locally
  would PROBABLY work and why we do not.

## SASS notes

- `red.release.gpu.global.add.u32` lowers to `MEMBAR.ALL.GPU` +
  `RED.E.ADD.STRONG.GPU` — the release membar is explicit in SASS and
  sits on the arriving thread's path.
- On sm_75 cuobjdump, ordinary weak cached loads render as `LDG.E.SYS`;
  do not misread the `.SYS` as a scope qualifier (strong forms render
  `.STRONG.<scope>`). The G10 check_sass discipline keys on the STRONG
  forms.
