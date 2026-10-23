# G11 derivation: the interpreter register envelope (calx-mill)

The projection WORKED.md's adopter step names, derived 2026-07-11 through
calx-mill @ `46704b7` — the bump that added the cooperative-launch residency
check this derivation needed. An occupancy percentage alone cannot see the
cliff: registers allocate per warp but placement is per block, so a kernel
can read a healthy warps/SM figure while zero whole CTAs fit and the
cooperative launch deadlocks. `blocks_per_instance`/`cooperative_fits` are
Kani-verified (capacity respect, monotonicity, and the persistent-kernel
lemma: grid == instances iff one block fits per instance).

## The envelope

TU102 SM: 65536 registers at 256/warp allocation, 64 KiB shared memory,
32-warp ceiling; the megakernel launch shape is a cooperative grid of 72
blocks, one per SM, with a 60 KiB smem slab per block. A W-warp CTA places
iff W x roundup(32R, 256) <= 65536. Max regs/thread by warps per SM:

| warps/SM | threads/CTA | max regs/thread | one allocation step over    |
|----------|-------------|-----------------|-----------------------------|
| 8        | 256         | 255 (tool cap)  | n/a                         |
| 12       | 384         | **168**         | 176 -> 0 blocks, DEADLOCK   |
| 16       | 512         | 128             | 136 -> 0 blocks             |
| 24       | 768         | 80              | 88 -> 0 blocks              |
| 32       | 1024        | 64              | 72 -> 0 blocks              |

G11's binding (cooperative launch, <= 168 regs/thread, >= 12 warps/SM) sits
exactly on the 12-warp cliff. The boundary is pinned upstream as a unit test
(`tu102_cooperative_residency_is_a_register_cliff`) and reproduces from any
`ptxas -v` output:

```
calx-mill ptxas <ptxas-v-file> --block-threads 384 --block-smem 61440 --grid-blocks 72
  interp_168: ... occupancy 12/32 warps (37%) ... 1 blocks/instance, cooperative grid 72 on 72 instances: fits
  interp_176: ... occupancy 11/32 warps (34%) ... 0 blocks/instance, cooperative grid 72 on 72 instances: DEADLOCK
```

(nonzero exit on any DEADLOCK, so the check gates a build.)

## Measured anchors (reference/tu102 @ 789ff42)

- The runtime agrees with the model where they meet: the grid.sync bench's
  occupancy gate accepted every fitting cooperative config and the driver
  refused the oversubscribed twin (`cudaErrorCooperativeLaunchTooLarge`),
  matching `cooperative_fits` at the 256- and 1024-thread points.
- The register budget is the binding axis of the weight streamer, not
  bandwidth: the smem-page sweep found NO depth-vs-size knee (611-623 GB/s,
  >= 98% of P2 peak, at every config) while register cost spans 70-255 by
  (depth, size). The shallow d2s16KB shape needs 92 registers and already
  sustains 612.6 GB/s — comfortably inside the 168-register envelope, where
  the deep d5s8KB shape (255 registers) cannot place at 12 warps at all.
  The no-knee finding means choosing the cheap shape costs <= 1.7% of
  bandwidth: the envelope and the stream are compatible by measurement,
  not by hope.
- One grid.sync at this shape costs 825-897 ns measured (the Y02 comparator;
  the hand-rolled counter boundary stays ~2x cheaper).

G11 itself closes only when the real kernel's `ptxas -v` runs through this
same check at the recorded flags — this derivation is the projected half of
the predicted-vs-measured pair, not the gate.
