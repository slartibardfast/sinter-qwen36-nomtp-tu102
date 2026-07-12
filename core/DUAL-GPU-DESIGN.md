# Dual-GPU tensor-parallel decode (the 256K target)

The design of record for the production config: the k=0 kernel across two
TU102s, so 256K context fits (weights 18.9 GB + KV 17.2 GB > one 24 GB card;
split, each GPU holds ~18 GB). This is standard tensor parallelism, matched
to the fork's `-sm tensor` layout (`docs/INTEGRATION-FINDINGS.md`) so the
kernel reproduces the stock split's decode. The single-GPU kernel
(`k0/PARITY.md`) is reused whole; this adds the split and the cross-GPU
reduce.

## The split (per weight, from the fork's meta split callback)

- **Residual stream: mirrored.** Both GPUs hold the full 5120-wide `x`. Every
  norm, elementwise, gate, rope and the token embedding runs mirrored (both
  compute the same thing on the same mirrored input — no split, no reduce).
- **Expand projections: row-split** (`qkv`, `up|gate` fused, `lm_head`). Each
  GPU computes half the output rows: GPU0 rows `[0, n/2)`, GPU1 `[n/2, n)`.
  No reduce — the output is legitimately split (head-split q/k/v, ffn-split
  intermediate, vocab-split logits). Attention then runs on each GPU's own
  heads; the FFN silu·up on each GPU's own half.
- **Contract projections: column-split then cross-GPU reduce**
  (`attn_output`, `ffn_down`, `ssm_out`). Each GPU dots its half of the input
  columns → a *partial* 5120-wide sum. The two partials are summed across the
  GPUs (`OP_XCHG_PUSH` → boundary → `OP_XCHG_REDUCE`, `k0/ops/xchg.cuh`,
  hardware-validated) back to the mirrored residual. **Two reduce sites per
  block** (attention-out and ffn-down for attention blocks; ssm-out and
  ffn-down for DeltaNet blocks) × 64 = **128 reduces/pass** — exactly the
  census AllReduce count, confirming the layout.
- **KV cache + DeltaNet state: head-split.** Each GPU owns 2 of 4 KV heads
  and 24 of 48 DeltaNet heads — its own heads' cache and state, read/written
  locally, never exchanged. 256K KV = ~8.6 GB/GPU; fits with the weight half.

## Two kernels, host-coordinated per pass

Two cooperative launches, one per GPU (each 72×384, 60 KiB slab). Peer access
both ways. Per decode pass the host writes the **same token** to both
doorbell cells (the residual is mirrored, both GPUs decode the same token),
rings both doorbells, and waits on both done flags. Within a pass the two
kernels run their (split) schedules and rendezvous only at the 128 reduce
sites via the peer mailboxes — no host round-trip per reduce. The Y02
boundary count per GPU is unchanged from single-GPU plus the 2 boundaries
each reduce needs.

## The dual-GPU program

`compile_schedule.py` gains a `--split` mode emitting a program parameterized
by `gpu_index`: expand GEMVs get the per-GPU row range, contract GEMVs write a
local partial then an `OP_XCHG_PUSH`/boundary/`OP_XCHG_REDUCE` triple, KV/state
ops address the per-GPU head range. Per-weight axis (row / column / mirror)
comes from the fork's split classification. The harness patches `gpu_index`
and the peer mailbox pointers at pack time. One reduce mailbox pool per GPU
(payload + seqno per site), monotonic seqno per site per pass.

## Parity + floor

- **Correctness:** match the `ref-tensor` oracle at n_ctx 262144, prefill 64
  (the 2-GPU reference tree, `/var/tmp/mk-oracle/ref-tensor-ctx64`), same
  scored lanes (logits KL ≤ 0.02, residual RMS ≤ 0.02, greedy tokens). The
  cross-GPU reduce fold order is declared (`xchg.cuh`: `p0 + p1` in GPU-index
  order, bit-identical on both GPUs).
- **Floor:** now the real one — each GPU streams ~8.35 GB/pass, so the weight
  bound is ~12 ms → ~73 tok/s at shallow context; beat the 43.9 stock floor,
  itemize the residual ms toward the bound. At 256K the KV read (~8.6 GB/GPU)
  adds, sliding toward the ~36 tok/s deep bound.

## Prerequisite before deep context

The per-pass `n_kv` symbol is currently read with a plain cached load
(correct at fixed n_kv). For 256K, where n_kv grows each token and crosses
padding classes, the fattn ops must read n_kv from a device cell with a
strong (`.cg`) load. Small op+packer change; do it before the long-context
parity runs.
