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

## Per decode graph_compute (the hot path)
1. gpu_set_inputs(c, pos) both GPUs: positions/kv_row/n_kv/mask from the graph's
   position input (read llama's inp_pos / the KV head-cell), token from the graph.
2. dual_run_pass(g2, token, seqno, timeout).
3. Gather logits: each GPU's `result_output` is its vocab half (lm_head column-
   split); gather the two halves into the graph's OUTPUT tensor (the logits llama
   samples). The output tensor is a meta tensor -> write each half to its per-GPU
   simple tensor's data, or gather to the single output buffer llama reads.

## Embed
MK's graph is split-1 (transformer stack); llama does the CPU embed (split-0) and
hands the residual as MK's graph input. So SKIP the schedule's OP_EMBED_LOOKUP and
seed the megakernel residual from the graph's input tensor (the CPU-computed
embed), OR (if token_embd is reachable per-GPU) let the megakernel do embed from
d_token. Decide during L3; the input-seed path matches llama's split.

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
1. Weight slice layout: llama's meta slice vs the schedule's expected. Premise says
   they match; verify via L4.
2. KV byte layout: harness cache vs llama head-split cache. Verify via L4.
3. Embed seed (input residual vs schedule embed).
4. Position/n_kv sourcing: read llama's actual decode position, not a synthetic pos.
5. Resident-kernel lifecycle vs llama rebuilding the graph (fingerprint re-check).
