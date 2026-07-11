# Decode-graph block structure, Qwen3.6-27B hybrid (k=0 megakernel capture)

Source of record: `capture/lg0-fingerprint/out262k/ctx33000-step0.txt` (deep
prefill, n_kv padding class 33024), cross-checked against `ctx64-step0.txt`
(n_kv 256). 3704 nodes + 990 leaves, structure bit-stable across depths; the
only cross-depth variation is in `ne` extents (67 record lines: per attention
layer the K/V window VIEW ne2 and PERMUTE ne1, plus the shared mask leaf pair
and its conversion CPY), exactly the `template.json` mask. All shapes below are
from the ctx33000 file; `n_kv` marks the extents that scale with context depth.
Node numbers cited are from that file; the per-block *pattern* is what binds.

Engine: llama.cpp autoround @546eca8dc, model builder `src/models/qwen35.cpp`
(split ssm_alpha/ssm_beta variant, NOT `qwen3next.cpp`'s fused ssm_ba).
op_params decoded per `ggml/src/ggml.c` conventions; all values below are the
decoded ones and are identical across all 32 recorded step files.

## Layer layout

64 transformer blocks: **48 gated-DeltaNet + 16 full attention**. Attention
blocks are every 4th block, indices `i % 4 == 3`:

    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63

Graph block i corresponds to `blk.i` in the GGUF. The GGUF additionally holds a
65th block (`blk.64.*`, the nextn/MTP head: attention + FFN + eh_proj) which is
**absent from this decode graph** (MTP off).

Constant parameters, everywhere they appear:

- RMS_NORM and L2_NORM eps = **1e-06** (par `bd378635` LE f32), all 209 + 96.
- GLU = **swiglu** (op 2), swapped=0, alpha=0, limit=0, all 64.
- UNARY sub-ops used: silu(10), sigmoid(7), softplus(15).
- ROPE (all 32): n_dims=**64** (of head_dim 256, 25% partial rotary),
  mode=**IMROPE(40)** (interleaved M-RoPE), sections=[11,11,10,0]
  (11+11+10=32 = n_dims/2 rotary pairs), n_ctx_orig=262144, freq_base=**1e7**,
  freq_scale=1, ext_factor=0, attn_factor=1, beta_fast=32, beta_slow=1.
  Positions input is i32[4]: 4 M-RoPE position ids per token.
- FLASH_ATTN_EXT (all 16): scale=**0.0625** (=1/sqrt(256)), max_bias=0,
  logit_softcap=0, prec=**GGML_PREC_F32** (op_params[3]=10).
- MUL_MAT par is all-zero everywhere (no prec override).

## Prologue (block_idx = pre)

| node | op | out shape | params | src | note |
|---|---|---|---|---|---|
| n0 | GET_ROWS | 5120,1 f32 | | l0 (token_embd.weight f16 5120x248320), l1 (i32[1] token id) | embedding row lookup |
| n6 | VIEW | 1 i32 | offset=0 | l5 | rs copy-index main view; output never consumed |
| n8 | VIEW | 0 i32 | offset=4 | l5 | rs clear-index view (0 elements); never consumed |

n0's output re-enters the graph as leaf **l2** (f32 5120), a scheduler-split
copy: the residual trunk root. The trunk crosses a split boundary only here;
from l2 onward the whole 64-block chain is node-to-node (n59 -> n66 -> n123 ->
... -> n3699). n6/n8 mirror l6/l7 (presumed same host data, see QUESTIONS).

## DeltaNet block archetype (48 blocks; 64 nodes each)

Shown with layer-relative names; node numbers from block 1 (n67..n130).
`x` = block input (previous ffn residual add; l2 for block 0).

| # | op | out shape | params | src wiring | macro |
|---|---|---|---|---|---|
| +0 | RMS_NORM | 5120,1 | eps=1e-06 | x | rmsnorm (attn_norm) |
| +1 | MUL | 5120,1 | | ^, attn_norm.weight f32[5120] | rmsnorm |
| +2 | RESHAPE | 30720,1 | | conv_state leaf f32[30720] (=10240x3) | conv_state_shift |
| +3 | VIEW | 0,1 | offset=0 | ^ | conv_state_shift |
| +4 | SCALE | 0,1 | scale=0 bias=0 | ^ | conv_state_shift (0-elem clear no-op) |
| +5 | GET_ROWS | 30720,1 | | +2, l6 (i32[1] rs row) | conv_state_shift (fetch seq's window) |
| +6 | GET_ROWS | 30720,0 | | +2, l7 (i32[0]) | conv_state_shift (clear fetch, 0 rows) |
| +7 | VIEW | 30720,0 | offset=122880 | conv_state leaf | conv_state_shift |
| +8 | CPY | 30720,0 | | +6 -> +7 | conv_state_shift (clear write, no-op) |
| +9 | RESHAPE | 3,10240 | | +5 | conv_state_shift (window as d_conv-1 x channels) |
| +10 | MUL_MAT | 10240,1 | | attn_qkv.weight q4_0 5120x10240, +1 | qkv_mmvq |
| +11 | RESHAPE | 10240,1 | | ^ | qkv_mmvq |
| +12 | TRANSPOSE | 1,10240 | | ^ | qkv_mmvq |
| +13 | CONCAT | 4,10240 | dim=0 | +9, +12 | conv_concat_store (window+new token) |
| +14 | VIEW | 3,10240 | offset=4 | ^ | conv_concat_store (shift by 1 elem) |
| +15 | VIEW | 30720,1 | offset=0 | conv_state leaf | conv_concat_store |
| +16 | CPY | 30720,1 | | +14 -> +15 | conv_concat_store (conv-state write-back) |
| +17 | RESHAPE | 786432,1 | | ssm_state leaf f32[786432] (=128x128x48) | ssm_state_fetch |
| +18 | VIEW | 0,1 | offset=0 | ^ | ssm_state_fetch |
| +19 | SCALE | 0,1 | scale=0 bias=0 | ^ | ssm_state_fetch (no-op) |
| +20 | GET_ROWS | 786432,1 | | +17, l6 | ssm_state_fetch |
| +21 | GET_ROWS | 786432,0 | | +17, l7 | ssm_state_fetch (no-op) |
| +22 | VIEW | 786432,0 | offset=3145728 | ssm_state leaf | ssm_state_fetch |
| +23 | CPY | 786432,0 | | +21 -> +22 | ssm_state_fetch (no-op) |
| +24 | SSM_CONV | 10240,1 | | +13 (4,10240 window), ssm_conv1d.weight f32 4x10240 | ssm_conv_silu |
| +25 | UNARY | 10240,1 | silu | ^ | ssm_conv_silu |
| +26 | VIEW | 128,16 | offset=0 | ^ | qk_l2norm (q: 16 heads x 128) |
| +27 | L2_NORM | 128,16 | eps=1e-06 | ^ | qk_l2norm |
| +28 | VIEW | 128,16 | offset=8192 | +25 | qk_l2norm (k at elem 2048) |
| +29 | L2_NORM | 128,16 | eps=1e-06 | ^ | qk_l2norm |
| +30 | VIEW | 128,48 | offset=16384 | +25 | qk_l2norm (v at elem 4096: 48 heads x 128) |
| +31 | MUL_MAT | 48,1 | | ssm_alpha.weight f16 5120x48, +1 | gdn_gates |
| +32 | RESHAPE | 48,1 | | ^ | gdn_gates |
| +33 | ADD | 48,1 | | ^, ssm_dt.bias f32[48] | gdn_gates |
| +34 | UNARY | 48,1 | softplus | ^ | gdn_gates |
| +35 | MUL | 48,1 | | ^, ssm_a f32[48] (-exp(A_log) factor) | gdn_gates |
| +36 | RESHAPE | 1,48 | | ^ | gdn_gates (g, decay gate) |
| +37 | MUL_MAT | 48,1 | | ssm_beta.weight f16 5120x48, +1 | gdn_gates |
| +38 | RESHAPE | 1,48 | | ^ | gdn_gates |
| +39 | UNARY | 1,48 | sigmoid | ^ | gdn_gates (beta, mixing coeff) |
| +40 | RESHAPE | 128,128,48 | | +20 | ssm_state_fetch |
| +41 | RESHAPE | 786432,1 | | ^ | ssm_state_fetch |
| +42 | GATED_DELTA_NET | 6144,129 | | q=+27, k=+29, v=+30, g=+36, beta=+39, state=+41 | deltanet_step |
| +43 | VIEW | 128,128,48 | offset=24576 | ^ | ssm_state_store (rows 1..128 = new state) |
| +44 | VIEW | 786432,1 | offset=0 | ssm_state leaf | ssm_state_store |
| +45 | CPY | 786432,1 | | +43 -> +44 | ssm_state_store (state write-back) |
| +46 | VIEW | 128,48 | offset=0 | +42 | gated_out_norm (row 0 = token output) |
| +47 | RMS_NORM | 128,48 | eps=1e-06 | ^ | gated_out_norm (per-head) |
| +48 | MUL | 128,48 | | ^, ssm_norm.weight f32[128] | gated_out_norm |
| +49 | MUL_MAT | 6144,1 | | attn_gate.weight q4_0 5120x6144, +1 | gated_out_norm (z gate) |
| +50 | RESHAPE | 128,48 | | ^ | gated_out_norm |
| +51 | UNARY | 128,48 | silu | ^ | gated_out_norm |
| +52 | MUL | 128,48 | | +48, +51 | gated_out_norm |
| +53 | RESHAPE | 6144,1 | | ^ | gated_out_norm |
| +54 | MUL_MAT | 5120,1 | | ssm_out.weight **q4_0_ar16** 6144x5120, +53 | out_proj_ar16 |
| +55 | RESHAPE | 5120,1 | | ^ | out_proj_ar16 |
| +56 | ADD | 5120,1 | | ^, x | residual_add (attn) |
| +57 | RMS_NORM | 5120,1 | eps=1e-06 | ^ | rmsnorm (ffn_norm = post_attention_norm.weight) |
| +58 | MUL | 5120,1 | | ^, post_attention_norm.weight f32[5120] | rmsnorm |
| +59 | MUL_MAT | 17408,1 | | ffn_gate.weight q4_0 5120x17408, +58 | ffn_fused |
| +60 | MUL_MAT | 17408,1 | | ffn_up.weight q4_0 5120x17408, +58 | ffn_fused |
| +61 | GLU | 17408,1 | swiglu | +59 (gate), +60 (up) | ffn_fused |
| +62 | MUL_MAT | 5120,1 | | ffn_down.weight q4_0 17408x5120, +61 | ffn_fused |
| +63 | ADD | 5120,1 | | ^, +56 | residual_add (ffn) -> next block input |

The qkv GEMV output layout after SSM_CONV+silu is `[q 16x128 | k 16x128 |
v 48x128]` (VIEW offsets 0 / 8192 / 16384 bytes). GATED_DELTA_NET output is
6144 x 129: row 0 the token output, rows 1..128 the state snapshot (VIEW
offset 24576 B), copied back over the ssm_state leaf. The conv window
write-back is the CONCAT shifted one element (VIEW offset 4 B). The
state-clear sub-chains (0-element SCALE / GET_ROWS / CPY at +3/+4, +6/+8,
+18/+19, +21/+23) are structural no-ops at batch-1 decode.

## Attention block archetype (16 blocks; 39 nodes, +1 in block 3)

Node numbers from block 3 (n195..n234). GQA: **24 q-heads** x 4 kv-heads,
head_dim 256 (see QUESTIONS: the briefing said 32 q-heads; measured 24).

| # | op | out shape | params | src wiring | macro |
|---|---|---|---|---|---|
| +0 | RMS_NORM | 5120,1 | eps=1e-06 | x | rmsnorm (attn_norm) |
| +1 | MUL | 5120,1 | | ^, attn_norm.weight f32[5120] | rmsnorm |
| +2 | MUL_MAT | 12288,1 | | attn_q.weight q4_0 5120x12288, +1 | q_gemv_norm_rope (q+gate interleaved per head: [q 256 \| gate 256] x 24) |
| +3 | VIEW | 256,24 | offset=0 | ^ | q_gemv_norm_rope (q slice; per-head stride 512 f32, from builder) |
| +4 | RMS_NORM | 256,24 | eps=1e-06 | ^ | q_gemv_norm_rope |
| +5 | MUL | 256,24 | | ^, attn_q_norm.weight f32[256] | q_gemv_norm_rope |
| +6 | ROPE | 256,24 | IMROPE n_dims=64 (see header) | ^, l57 (i32[4] positions) | q_gemv_norm_rope |
| +7 | MUL_MAT | 1024,1 | | attn_v.weight q4_0 5120x1024, +1 | v_gemv |
| +8 | RESHAPE | 256,4 | | ^ | v_gemv (V: no norm, no rope) |
| +9 | MUL_MAT | 1024,1 | | attn_k.weight q4_0 5120x1024, +1 | k_gemv_norm_rope |
| +10 | RESHAPE | 256,4 | | ^ | k_gemv_norm_rope |
| +11 | RMS_NORM | 256,4 | eps=1e-06 | ^ | k_gemv_norm_rope |
| +12 | MUL | 256,4 | | ^, attn_k_norm.weight f32[256] | k_gemv_norm_rope |
| +13 | ROPE | 256,4 | same as +6 | ^, l57 | k_gemv_norm_rope |
| +14 | VIEW | 1024,1 | offset=0 | ^ | kv_append (roped K flattened) |
| +15 | SET_ROWS | 1024,262144 f16 | | values=+14, idx=l61 (i64[1] k_idxs), dest=cache_k | kv_append (row index is tensor DATA, not a view offset) |
| +16 | VIEW | 1024,1 | offset=0 | +8 | kv_append |
| +17 | SET_ROWS | 1024,262144 f16 | | values=+16, idx=l63 (i64[1] v_idxs), dest=cache_v | kv_append |
| +18 | VIEW | 256,24 | offset=0 | +6 | fattn_prep |
| +19 | PERMUTE | 256,1,24 | axes=[0 2 1 3] | ^ | fattn_prep (q: head_dim, n_tok, heads) |
| +20 | VIEW | 256,4,**n_kv** f16 | offset=0 | cache_k | fattn_prep (n_kv=33024 here; 256 in ctx64 file) |
| +21 | PERMUTE | 256,**n_kv**,4 | axes=[0 2 1 3] | ^ | fattn_prep |
| +22 | VIEW | 256,4,**n_kv** f16 | offset=0 | cache_v | fattn_prep |
| +23 | PERMUTE | 256,**n_kv**,4 | axes=[0 2 1 3] | ^ | fattn_prep |
| (+23a) | CPY | **n_kv**,1 f16 | | l65 (mask f32) -> l66 (mask f16) | mask_convert -- block 3 ONLY; n219 |
| +24 | FLASH_ATTN_EXT | 256,24 | scale=0.0625 max_bias=0 softcap=0 prec=F32 | q=+19, K=+21, V=+23, mask=n219 | fattn_decode |
| +25 | RESHAPE | 6144,1 | | ^ | attn_gate_out |
| +26 | VIEW | 256,24 | offset=1024 | +2 | attn_gate_out (gate slice of q GEMV) |
| +27 | CONT | 6144,1 | | ^ | attn_gate_out |
| +28 | UNARY | 6144,1 | sigmoid | ^ | attn_gate_out |
| +29 | MUL | 6144,1 | | +25, +28 | attn_gate_out |
| +30 | MUL_MAT | 5120,1 | | attn_output.weight q4_0 6144x5120, +29 | o_gemv |
| +31 | ADD | 5120,1 | | ^, x | residual_add (attn) |
| +32..+38 | | | | RMS_NORM+MUL, 3x MUL_MAT + GLU, ADD -- identical to DeltaNet +57..+63 | rmsnorm / ffn_fused / residual_add |

The mask CPY (n219) executes once, in block 3; all 16 FLASH_ATTN_EXT nodes
reference it as src3 (verified). l57/l61/l63 are shared by all attention
blocks; l6/l7 by all 48 DeltaNet blocks. KV write position never appears in a
view offset: SET_ROWS carries the row index as tensor data (l61/l63), which is
why op_params are depth-invariant (the LG0 no-par-on-VIEW finding).

## Epilogue (block_idx = post)

| node | op | out shape | params | src | macro |
|---|---|---|---|---|---|
| n3700 | GET_ROWS | 5120,1 | | n3699 (block 63 ffn add), l987 (i32[1] out-row id) | logits_row_select |
| n3701 | RMS_NORM | 5120,1 | eps=1e-06 | ^ | rmsnorm (output_norm) |
| n3702 | MUL | 5120,1 | | ^, output_norm.weight f32[5120] | rmsnorm |
| n3703 | MUL_MAT | 248320,1 | | output.weight **f16** 5120x248320, ^ | head_gemv |

output.weight is a separate tensor from token_embd.weight (distinct leaves
l989 / l0 and distinct GGUF entries) -- untied head, a 2.54 GB f16 GEMV read.

## Macro-op vocabulary (982 instances covering 3704/3704 nodes)

| macro_kind | instances | nodes | DRAM-heavy? |
|---|---|---|---|
| rmsnorm | 129 | 258 | no (weights f32[5120] etc.) |
| residual_add | 128 | 128 | no |
| ffn_fused | 64 | 256 | yes: 3 q4_0 GEMVs, 150.4 MB/block |
| conv_state_shift | 48 | 384 | minor (conv state 123 KB R) |
| qkv_mmvq | 48 | 144 | yes: q4_0 29.5 MB |
| conv_concat_store | 48 | 192 | minor (123 KB W) |
| ssm_state_fetch | 48 | 432 | yes: 3.1 MB R |
| ssm_conv_silu | 48 | 96 | no |
| qk_l2norm | 48 | 240 | no |
| gdn_gates | 48 | 432 | minor (2 f16 GEMVs, 0.98 MB) |
| deltanet_step | 48 | 48 | state held in the op |
| ssm_state_store | 48 | 144 | yes: 3.1 MB W |
| gated_out_norm | 48 | 384 | yes: attn_gate q4_0 17.7 MB |
| out_proj_ar16 | 48 | 96 | yes: AR16 19.7 MB |
| q_gemv_norm_rope | 16 | 80 | yes: q4_0 35.4 MB |
| v_gemv | 16 | 32 | yes: q4_0 2.9 MB |
| k_gemv_norm_rope | 16 | 80 | yes: q4_0 2.9 MB |
| kv_append | 16 | 64 | write 2x2 KB |
| fattn_prep | 16 | 96 | no (views/permutes) |
| fattn_decode | 16 | 16 | yes: KV read 2 x n_kv x 2 KB |
| attn_gate_out | 16 | 80 | no |
| o_gemv | 16 | 16 | yes: q4_0 17.7 MB |
| embed_lookup | 1 | 1 | 10 KB row read |
| rs_index_setup | 1 | 2 | no (dead views) |
| mask_convert | 1 | 1 | n_kv x 6 B |
| logits_row_select | 1 | 1 | no |
| head_gemv | 1 | 1 | yes: f16 2.54 GB |

## DRAM budget per decode step (decimal GB)

- Weight stream: **16.37 GB** = 48 DeltaNet blocks x 218.2 MB (10.48 GB)
  + 16 attention blocks x 209.4 MB (3.35 GB) + head 2.54 GB. 497 weight GEMVs:
  352 q4_0 + 48 q4_0_ar16 + 96 f16(5120x48) + 1 f16 head. (NOTES records
  16.69 GB; 0.3 GB unexplained, see QUESTIONS.)
- KV read (fattn_decode): 16 blocks x 2 x n_kv x 1024 x 2 B =
  **17.18 GB at n_kv=262144** (exceeds the weight stream), 2.16 GB at 33024.
- GDN ssm state: 151 MB read + 151 MB write; conv windows 11.8 MB R+W.
- KV append: 64 KB write/step across the 16 blocks.
