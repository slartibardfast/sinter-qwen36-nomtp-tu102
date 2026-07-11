# Open questions and ambiguities from the extraction

Nothing below blocked the extraction; every node and leaf is assigned. These
are the points where the record alone was insufficient and an external source
or a stated assumption filled the gap, plus genuine discrepancies.

## Discrepancies against the given model context

1. **q-heads: measured 24, briefed 32.** The attention q GEMV is 5120x12288 =
   24 heads x (256 q + 256 gate) interleaved; FLASH_ATTN_EXT output is 256x24,
   its q input 256x1x24. The "GQA q-heads 32" in the task briefing does not
   match this graph. kv-heads 4 and head_dim 256 confirmed.
2. **65 GGUF blocks, 64 in the graph.** `blk.64.*` (nextn/MTP: attn_*, ffn_*,
   eh_proj, enorm/hnorm/shared_head_norm) is present in the GGUF but absent
   from the decode graph (MTP off). Any weight-residency plan must decide
   whether to keep blk.64 resident for a later MTP-on kernel.
3. **Weight-stream total: 16.37 GB computed vs 16.69 GB in NOTES.md.** My sum
   over the 851 weight leaves actually read per step (excluding token_embd,
   whose read is one 10 KB row) is 16.368e9 B. The NOTES figure is 0.32 GB
   higher; possibly it includes token_embd or blk.64. Not resolved here.

## Capture-format limitations (not decodable from the record)

4. **VIEW strides are not recorded.** The record carries only the VIEW byte
   offset (op_params) and the result ne; nb[] is absent. Affected places where
   strides are load-bearing and were taken from the builder source
   (`src/models/qwen35.cpp`), not the dump:
   - the q / gate interleaved views of the 12288-wide q GEMV (offsets 0 and
     1024 B, per-head stride 512 f32);
   - the K/V cache read views (256,4,n_kv over a 1024 x 262144 f16 buffer)
     and their [0,2,1,3] permutes;
   - the 1024,1 views flattening roped-K / V for SET_ROWS.
   A megakernel interpreter regenerating buffers from this schedule must
   re-derive nb[] from the builder conventions; the CSV cannot supply them.
5. **Pointer identity across scheduler splits.** The dump records no pointers,
   so these identities are inferred, not proven:
   - l2 (f32 5120) is taken to be the cross-split copy of n0's embedding row
     (shape, type, position, and singleton use all fit; nothing else produces
     a 5120 f32 before n1).
   - n6/n8 (views of l5, outputs never consumed by any node) are taken to be
     the compute-side source of l6/l7 (the rs main/clear row-index leaves used
     by all 96 recurrent-state GET_ROWS). Functionally l6=i32[1] main row,
     l7=i32[0] clear rows regardless.

## Name assignments that relied on wiring, not shape (shape+type ties)

All resolved with high confidence from wiring plus the qwen35.cpp builder;
listed because shape+type alone could not distinguish them:

6. **ssm_alpha vs ssm_beta** (both f16 5120x48): alpha = the GEMV feeding
   ADD(ssm_dt.bias) -> softplus -> MUL(ssm_a); beta = the GEMV feeding
   sigmoid. Matches qwen35.cpp lines 364-380 exactly.
7. **ffn_gate vs ffn_up** (both q4_0 5120x17408): GLU src0 = gate, src1 = up
   (ggml_glu_split argument order).
8. **attn_k vs attn_v** (both q4_0 5120x1024): k's GEMV goes through
   k_norm RMS_NORM + ROPE before SET_ROWS; v's goes straight to SET_ROWS.
9. **attn_q_norm vs attn_k_norm** (both f32[256]): by head count of the MUL
   output (24 vs 4).
10. **attn_norm vs post_attention_norm** (both f32[5120] per block): by trunk
    position (pre-mixer vs pre-FFN).
11. **ssm_dt.bias vs ssm_a** (both f32[48]): ADD operand vs MUL operand.
12. **token_embd.weight vs output.weight** (both f16 5120x248320): GET_ROWS
    source vs final MUL_MAT weight. They are distinct tensors (untied head),
    two GGUF entries and two distinct leaves.
13. **cache_k vs cache_v** (both f16 1024x262144): the SET_ROWS whose value
    chain includes ROPE writes cache_k.

## Semantics flagged for the megakernel design

14. **Zero-extent no-op chains.** Each DeltaNet block carries two
    clear-unused-state sub-chains per state (VIEW ne=0 -> SCALE(0,0),
    GET_ROWS ne1=0 -> CPY ne1=0) that move zero bytes at batch-1 decode
    (l7 is i32[0]). They exist for multi-sequence state management. Assumed
    droppable in a k=0 single-sequence kernel, but that is a design decision,
    not something the capture proves safe.
15. **rs_index_setup (n6/n8) is dead in-graph** yet presented by the
    scheduler each step. Same caveat as 14.
16. **MUL_MAT op_params all zero**: GGML_PREC_DEFAULT is indistinguishable
    from "never set". FLASH_ATTN_EXT, by contrast, explicitly pins
    GGML_PREC_F32 (op_params[3]=10); the megakernel fattn must honour f32
    accumulation.
17. **NOTES.md "16 of 63" attention layers**: the graph measures 16 of 64
    blocks (48 DeltaNet + 16 attention). The 63 in NOTES appears to be a typo;
    recorded here so the discrepancy is not silently propagated.

## RESOLVED

18. **RESOLVED (attn op family): `ncols2=8` in `flash_attn_ext_f16<256,256,1,8,0,0>`
    is a tile-shape bucket, not the GQA ratio.** This closes the FA half of
    discrepancy 1 (measured 24 q-heads vs the briefed 32). The dossier
    (SEMANTICS.md §8) glossed `ncols2=8` as "the GQA ratio (32/4)"; the graph
    has 24 q-heads, ratio 24/4 = **6**. Fork source (all cites into
    `software/llama.cpp/autoround`): the dispatch selects `ncols2` by *bucket*,
    not equality — `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2` picks 8 for
    any `use_gqa_opt && gqa_ratio > 4` (`ggml/src/ggml-cuda/fattn.cu:92-95`;
    `gqa_ratio = Q->ne[2]/K->ne[2]`, `fattn.cu:64-65`). `ncols2` is the number
    of Q-head *slots* packed per MMA tile so the KV stream is read once per KV
    head; the runtime `gqa_ratio` is passed into the kernel separately
    (`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1772`), which runs
    `iter_z_gqa = ceil(gqa_ratio/ncols2)` = ceil(6/8) = 1 tile per KV head
    (`fattn-mma-f16.cuh:1783`) and pads the dead slots: Q loads for slots with
    `zt_gqa*ncols2 + c >= gqa_ratio` are zero-filled
    (`fattn-mma-f16.cuh:1224,1232-1239`), their outputs suppressed at store
    (`fattn-mma-f16.cuh:1657`), and the stream-k fixup skips them
    (`fattn-common.cuh:715`). So the census template runs one 8-wide tile per
    KV head with 6 live + 2 zero-padded Q slots (25% padded MMA work; KV read
    once either way). **Math for our kernel:** the true ratio is 6; per GPU
    (KV split by heads) 12 q-heads over 2 kv-heads, 6 q per kv group; no 8
    appears anywhere. `k0/ops/attn.cuh` derives `gqa = n_q / n_kv_heads` from
    the instruction args and packs no pad slots.
