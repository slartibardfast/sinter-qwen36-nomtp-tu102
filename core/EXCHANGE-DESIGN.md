# The Y06 cross-GPU exchange (design of record, to implement for dual-GPU)

The k=0 kernel's one cross-GPU primitive: a mirrored fp32 allreduce of a
partial activation vector (5120 f32 = 20 KB), 2 sites per block x 64 blocks
= 128 per pass, replacing the fork's pinned-host PCIe AllReduce (call/0021's
declared divergence). Every ordering choice below is bound to a measured
sync_protocol row; do not weaken any of it without re-running the litmus.

## Protocol per site (symmetric on both GPUs)

1. OP_XCHG_PUSH (participating blocks): store the locally computed partial
   slice into the PEER's mailbox payload region with line-filling float4
   stores in linear order (the measured 2.4-2.5x visibility lever:
   sync.sentinel.pattern.lat lin4 10.6us vs stride1 27.8us at 10KB — the
   store pattern is load-bearing, not stylistic). Then EACH pushing block
   executes membar.sys (its own peer stores must be sys-ordered; the Y02
   boundary's .gpu-scope release does NOT order sys-destined stores for a
   sys observer). Then cross one Y02 boundary.
2. After the boundary, one elected block does st.release.sys of the site
   seqno into the peer's mailbox. Seqno is monotonic per site per pass
   (wrap-safe compare), never reset.
3. OP_XCHG_REDUCE (on each GPU): poll the LOCAL L2 seqno copy
   (ld.acquire), then read the received payload with .cg loads (Y06: poll
   local seqno then LDG.cg payload; NEVER a plain LDG — the xgpu stale-L1
   read is the measured 999999/1e6 hazard), then write the mirrored result
   out[i] = p0[i] + p1[i] in GPU-INDEX order on BOTH GPUs (the declared
   fold order: local/remote roles differ per GPU, the arithmetic order does
   not). Bit-identical mirrored activations, unlike stock's bf16 wire.

## Mailboxes

Per site per direction: {payload[5120] f32, pad to 128B line, seqno u32 in
its own line}. Allocated at init on each GPU; peer access enabled both ways
(cudaDeviceEnablePeerAccess); UVA pointers exchanged host-side into the
Instr args at pack time.

## Costs (measured bases)

Push 20 KB one-way ~1.7 us (x.nvlink.relacq.oneway 10240b=1.68us class);
boundary ~0.4 us; poll ~161.5 ns/L2 hit. 128 sites/pass ~= 270 us against
the ~2 ms/pass the stock PCIe path costs at the same census counts.
Never overlap a push with the local weight stream more than the schedule
already does (x.nvlink.contention row: -16.4% to 509 GB/s while streaming).

## Litmus obligations (extend tests/litmus_boundary.cu patterns)

- Positive: MP across GPUs at the real payload size with warmed consumer
  L1; zero stale over a long run; mirrored-sum bit-identity check between
  the two GPUs' results.
- Negatives (must be CAUGHT): (a) plain payload reads — expect the
  measured ~1e6/1e6 stale class; (b) membar.sys dropped from the pushing
  blocks — flag-before-payload over NVLink; (c) seqno published before the
  boundary (a torn multi-block payload).
- G30 binds at this size: re-run the composed-visibility gate methodology
  at 20 KB.
