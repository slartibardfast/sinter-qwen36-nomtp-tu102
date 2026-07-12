# R5 L3 build spec: dual re-host under mk_dispatch

Reuse the harness dual core (k0/harness.cpp) verbatim; only the weight/KV source
changes (llama's extracted pointers, not a GGUF load). The Resolver is the seam.

## What to reuse (refactor to expose; do NOT rewrite)
From k0/harness.cpp, expose (external linkage, e.g. k0/dual_core.h + a shared TU
the .so links; guard `main` and the parity/bench drivers behind `#ifndef MK_NO_MAIN`):
- `Resolver`, `GpuCtx`, `PackedProgram`, `pack_program`, `with_payload`/`emit`, all
  `pack_<kind>` op fns (the ~25 macro-op packers), `weight_slice`/`weight_axis`/
  `dn_stride` (only if MK needs to validate layout), `gpu_alloc_buffers`,
  `gpu_alloc_mailboxes`, `gpu_wire_mailboxes`, `gpu_set_inputs`, `dual_run_pass`,
  `dual_shutdown`, the constants (N_EMBD, N_LAYER, N_EMBD_GQA, CONV_STATE_N,
  SSM_STATE_N, N_VOCAB, is_attn_layer, MBOX_*).

## mk_dispatch, on first fingerprint hit (setup)
Replaces `dual_setup` + `gpu_upload_weights`:
1. Peer access both ways (as dual_setup).
2. Build the pointer map (L1, done): name -> {per-GPU dev ptr} via the mirror.
3. For each GPU g, GpuCtx c: `c.ln.init(g)`, `c.R.add("token", c.ln.d_token(),128)`.
4. WEIGHTS: for every weight the program binds (`gguf:NAME`), `c.R.add(NAME,
   ptrmap[NAME].p[g], bytes)`. llama's slice already matches the schedule's layout
   (gpu_upload_weights was built to reproduce it); if L4 parity fails, that
   assumption is the first suspect.
5. KV/STATE in place: point the Resolver at llama's cache buffers, name-mapped:
   `cache_k_l<il>`/`cache_v_l<il>` -> llama `cache_k_l/v_l`; `conv_state_l<il>` ->
   llama `cache_r_l<il>`; `ssm_state_l<il>` -> llama `cache_s_l<il>`. (Confirm the
   harness's KV byte layout == llama's head-split cache layout; if not, this is the
   second parity suspect.)
6. SCRATCH: MK-alloc the non-KV buffers gpu_alloc_buffers makes (positions, kv_row,
   rs_row, n_kv, mask_f16, result_output=this GPU's vocab half, done/error, the
   program `buffers` table). Do NOT MK-alloc the caches (step 5).
7. mailboxes: gpu_alloc_mailboxes + gpu_wire_mailboxes (count OP_XCHG_REDUCE).
8. pack_program(program, c.R, &c.mtab, g); upload_program (host_upload + launch).

## Per decode graph_compute (the hot path)   [seed-from-residual, no token]
0. POSITION (risk #4, RESOLVED 2026-07-13): MK's graph exposes NO position/idx
   named tensor and only ONE `MK#` boundary input (the residual). The megakernel
   computes RoPE/KV internally from a scalar `pos`, so read only that scalar from
   the set_rows KV-write index: find the first OP_SET_ROWS node (op=42), take its
   src[1] (k_idxs, a mirrored int32 = [n_past]); cudaMemcpy 4 B from its per-GPU
   data -> pos. (Robust across warmup/resets/deep, unlike a call counter.)
1. Seed the residual: cudaMemcpy MK#model.input_embed#0[gpu] (5120 f32, mirrored)
   -> buf:residual[gpu] on each GPU. Strip the leading OP_EMBED_LOOKUP from the
   program at load (its gguf:token_embd.weight isn't in MK's graph -> pack would
   fail; and buf:residual is now seeded, not embedded).
2. gpu_set_inputs(c, pos) both GPUs: positions/kv_row/n_kv/mask (token unused).
3. dual_run_pass(g2, /*token=*/0, seqno, timeout)  (token arg dead once embed
   is stripped; kept for signature).
4. Gather logits: each GPU's `result_output` is its vocab half (lm_head column-
   split); cudaMemcpy each half into the graph OUTPUT tensor's per-GPU simple
   tensor data. llama's meta get_tensor then gathers the two halves.

## Embed  (RESOLVED 2026-07-13: seed from the CPU residual)
CONFIRMED via probe: token_embd.weight is NOT in MK's graph, and node[0] is
`norm-0` (RMS_NORM) reading `src0 = MK#model.input_embed#0`. So llama does the CPU
embed (split-0) and MK's split-1 graph starts from that residual. The megakernel
must NOT run OP_EMBED_LOOKUP; instead, each pass copy `MK#model.input_embed#0`'s
per-GPU data into the megakernel's residual buffer (the embed op's output buffer),
and make the schedule's leading embed a no-op (or overwrite its output post-embed).
`MK#model.input_embed#0` is a meta tensor -> extract its 2 per-GPU pointers like
any other; it is the per-pass INPUT (replaces `token`). Position still needed for
RoPE/KV-write; read llama's decode position (inp_pos or the KV head cell).

Implication for mk_dual_step: signature becomes (residual_seed_ptrs[2], pos,
out0, out1) — no token. Copy seed -> residual buffer on each GPU, run, gather.

## Lifecycle
Setup once on first hit (guarded static / per-backend-context). Kernels resident
across passes. dual_shutdown on backend free or fingerprint change.

## Gates
- L4 correctness: greedy tokens match stock + llama-perplexity within the
  output-preserving bound (not bit-exact). A layout mismatch (step 4/5) surfaces
  here as garbage/NaN logits.
- L5: llama-bench tg tok/s vs stock, shallow + deep; served-token check under
  llama-server; perplexity pass.

## Risks (ranked)
1. Weight slice layout: RETIRED 2026-07-13. The live-graph geometry probe (see
   R3-PARITY "Live decode geometry") shows llama's per-GPU weight shapes+strides
   equal the harness's (qkv 5120/GPU AXIS_1, ssm_out 3072/GPU AXIS_0 F16, output
   124160/GPU AXIS_1). Segment interleave for qkv still verified at L4.
2. KV byte layout: RETIRED 2026-07-13. cache_k/v [512,256] pos-major, cache_r
   [15360], cache_s [393216] all equal the harness (kv_row=512, CONV/2, SSM/2).
3. Embed seed (input residual vs schedule embed).
4. Position/n_kv sourcing: read llama's actual decode position, not a synthetic pos.
5. Resident-kernel lifecycle vs llama rebuilding the graph (fingerprint re-check).
6. Tool config: llama-bench MUST pass `-fa 1` (FA-on) to build the fingerprint
   graph; FA-off aborts in the meta backend's set_rows handler (see R3-PARITY).
