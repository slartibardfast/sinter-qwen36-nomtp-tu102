# Decode-op exact-semantics dossier — Qwen3.6-27B hybrid, batch-1, `-sm tensor`, `-fa on`, f16 KV

Fork: `software/llama.cpp/autoround` @ `546eca8dc` (branch `autoround`). All cites are
`path:line` relative to that worktree root. Read-only survey.

**Model identity.** Qwen3.6-27B is served under `LLM_ARCH_QWEN35` (arch string `"qwen35"`,
`src/llama-arch.cpp:42`; dispatch `src/llama-model.cpp:275`, builder class
`llama_model_qwen35` in `src/models/qwen35.cpp`). It is a *hybrid* stack: 64 trunk decoder
blocks + 1 NextN/MTP block. Of the 64 trunk blocks, blocks with `(il+1) % full_attention_interval != 0`
(interval = 4) are **linear-attention (gated DeltaNet)** blocks, the rest are **full
softmax-attention** blocks (`src/models/qwen35.cpp:22-27`). That gives 48 DeltaNet layers and 16
attention layers — this is the origin of nearly every `48/pass` and `16/pass` census count.
The MTP block (`graph_mtp`, `src/models/qwen35.cpp:491`) is a separate graph
(`LLM_GRAPH_TYPE_DECODER_MTP`) and does **not** run in a plain batch-1 decode pass.

> **Fork divergence, top-level.** `LLM_ARCH_QWEN35`/`qwen35`, the `Q4_0_AR16` quant type
> (§2), the internal two-GPU AllReduce (`ggml_cuda_ar_kernel`, §5), the fused
> `ggml_gated_delta_net` op + snapshot-ring recurrent cache (§6), and the fused
> `rms_norm`+mul / unary+mul ops (§9, §14) are all fork-local; they are not in stock
> `ggml-org/llama.cpp`. Flagged inline where relevant.

---

## 1. `mul_mat_vec_q<Q4_0,1,0,0>` and `mul_mat_vec_q<Q4_0,1,1,0>` — the Q4_0 MMVQ (and SwiGLU-fused variant)

Kernel: `ggml/src/ggml-cuda/mmvq.cu:396-596`.

### Template parameters
`template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>`
(`mmvq.cu:396`). So the four census params `<t2,1,0,0>` / `<t2,1,1,0>` are:

| pos | name | meaning | `<Q4_0,1,0,0>` | `<Q4_0,1,1,0>` |
|-----|------|---------|----------------|----------------|
| 1 | `type` | src0 weight quant type; `t2` = `GGML_TYPE_Q4_0` | Q4_0 | Q4_0 |
| 2 | `ncols_dst` | number of dst columns (tokens) this instance computes; batch-1 decode ⇒ 1 | 1 | 1 |
| 3 | `has_fusion` | whether the epilogue fuses a gate (and/or bias) — see below | false | **true** |
| 4 | `small_k` | rows-per-block boost when `blocks_per_row_x` is small (raises `rows_per_cuda_block` to `nwarps`); `should_use_small_k`, `mmvq.cu:748-788`. Off here. | false | false |

### Is `<Q4_0,1,1,0>` the fused gate+up (SwiGLU) MMVQ? **Yes.**
`has_fusion=true` is only reachable for `ncols_dst==1` (`mmvq.cu:684-694`, and
`GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1")`). When the graph presents
an `ffn_up` matmul whose output is immediately `MUL`'d by `silu(ffn_gate·x)`, the CUDA backend
fuses both matmuls plus the SwiGLU into one launch by filling
`ggml_cuda_mm_fusion_args_device{ gate = ffn_gate weight, glu_op = GGML_GLU_OP_SWIGLU, … }`.
Inside the kernel:
- two independent dot products are accumulated per row — `tmp[j][i]` against `vx` (up) and
  `tmp_gate[j][i]` against `vgate` (gate), same activation block `y` (`mmvq.cu:498-507`);
- the epilogue applies `result *= ggml_cuda_op_silu_single(gate_value)` for
  `GGML_GLU_OP_SWIGLU` (`mmvq.cu:572-575`), i.e. `dst = (up·x) * silu(gate·x)`.

The census shows this variant running **64/pass** — one per FFN (48 DeltaNet + 16 attention blocks
all carry a dense SwiGLU FFN, `build_layer_ffn`, `src/models/qwen35.cpp:475-488`,
`LLM_FFN_SILU, LLM_FFN_PAR`). The census `grid rows 8704 (= 17408/2)`: `grid.x = nrows_x / rows_per_cuda_block`
with `rows_per_cuda_block=1` (`calc_rows_per_block(1,…)` returns 1, `mmvq.cu:376-393`), so
`grid.x = n_ff = 8704`. The `17408 = 2·n_ff` figure is the concatenated up+gate width; the fuse
computes both halves in the single launch (hence "8704 = 17408/2"). The non-fused `<Q4_0,1,0,0>`
handles the remaining Q4_0 GEMVs (attention q/k/v/o projections, `ffn_down`, wqkv/alpha/beta on
linear layers, etc.).

### Q4_0 block layout
`ggml/src/ggml-common.h:187-192`: `QK4_0 = 32`;
```
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;  // 2 + 16 = 18 bytes
```
`d` is the f16 delta; `qs` is 16 bytes = 32 packed 4-bit nibbles. Dequant value of element k is
`d * (nibble_k - 8)` (symmetric, offset 8). Nibble packing is the classic Q4_0 *split-half* form:
byte `j` holds element `j` in the low nibble and element `j+16` in the high nibble
(`vec_dot_q4_0_q8_1_impl` masks `>>0 &0x0F0F0F0F` and `>>4 &0x0F0F0F0F` as two 16-wide halves,
`vecdotq.cuh:120-122`).

### q8_1 activation block it dots against
`ggml-common.h:257-269`: `QK8_1 = 32`;
```
typedef struct { union { struct{ ggml_half d; ggml_half s; }; ggml_half2 ds; }; int8_t qs[QK8_1]; } block_q8_1; // 4 + 32 = 36 B
```
`d` = per-block scale, `s` = `d * sum(qs)` (used for the symmetric-offset correction). Produced by
`quantize_q8_1` (§3). One q8_1 block (32 int8) aligns with one Q4_0 block (32 quants).

### Row padding
`MATRIX_ROW_PADDING = 512` (`ggml/src/ggml-cuda/common.cuh:151`). Quantized weight rows are padded
up to a multiple of 512 elements when allocated (`ggml-cuda.cu:807-809`, and 953/1004/1043/1123) so
the last block never reads out of bounds. `blocks_per_row_x = ncols_x / qk` (`mmvq.cu:418`).

### Exact dp4a accumulation order (`vec_dot_q4_0_q8_1`, `vecdotq.cuh:736-752` → `_impl` `:115-134`)
`VDR_Q4_0_Q8_1_MMVQ = 2` (`vecdotq.cuh:112`), `QI4_0 = QK4_0/(4·QR4_0) = 32/8 = 4`, `QR4_0=2`.
Per call one thread handles `vdr=2` int-words of the block:
```
v[i]     = get_int_b2(bq4_0->qs, iqs+i)          # i in 0..1  (2 words = 16 nibbles each half)
u[2i+0]  = get_int_b4(bq8_1->qs, iqs+i)          # low-half activations
u[2i+1]  = get_int_b4(bq8_1->qs, iqs+i + QI4_0)  # high-half activations (offset by 4 = QI4_0)
sumi = 0
for i in 0..vdr-1:
    vi0 = (v[i] >> 0) & 0x0F0F0F0F                # 4 low nibbles  (elements j)
    vi1 = (v[i] >> 4) & 0x0F0F0F0F                # 4 high nibbles (elements j+16)
    sumi = dp4a(vi0, u[2i+0], sumi)               # 4-wide int8 dot, accumulate
    sumi = dp4a(vi1, u[2i+1], sumi)
return d4 * (sumi * ds8f.x  -  (8*vdr/QI4_0) * ds8f.y)
```
The final term `-(8·vdr/QI4_0)·d·s` subtracts the symmetric offset 8 from every quant using
`s = d8·sum(qs)` from the q8_1 block (`vecdotq.cuh:132-133`; `ds8f = __half22float2(ds8)`,
`.x=d8`, `.y=s8`). Loop over `kbx` (block index in the row) strides by
`blocks_per_iter = vdr·nwarps·warp_size/qi` (`mmvq.cu:419,488`); partial sums reduced across warps
via shared memory then `warp_reduce_sum` (`mmvq.cu:510-559`).

### Launch config
`nwarps = calc_nwarps(Q4_0, 1, GENERIC) = 4` (`mmvq.cu:298-313`); `rows_per_cuda_block = 1`
(`mmvq.cu:376-393`); block dims `(warp_size=32, nwarps=4, 1)` = 128 threads
(`calc_launch_params`, `mmvq.cu:661-671`); grid `(ceil(nrows_x/1), nchannels_dst, nsamples_dst)`.

---

## 2. `mul_mat_vec_q<Q4_0_AR16,1,0,0>` — the AutoRound V-col-reorder quant (**fork-only**)

`type 42 = GGML_TYPE_Q4_0_AR16`. Landed in plan/0134; the fork's AutoRound reorder type.

### Block layout
`ggml-common.h:194-199`: `QK4_0_AR16 = 16` (half the Q4_0 super-block):
```
typedef struct { ggml_half d; uint8_t qs[QK4_0_AR16/2]; } block_q4_0_ar16; // 2 + 8 = 10 bytes
```
`d` is symmetric (`absmax/8`). **Crucially the nibble order differs from Q4_0**: `qs` is stored
*element-interleaved*, `byte j = code[2j] | code[2j+1]<<4` (element k is nibble k), **not** Q4_0's
split-half layout (`ggml-common.h:197`, and the `unpack_q4_0_ar16` comment `vecdotq.cuh:772-774`).

### Dequant/dot vs Q4_0 (`vec_dot_q4_0_ar16_q8_1`, `vecdotq.cuh:755-780`)
`VDR_Q4_0_AR16_Q8_1_MMVQ = 2` (`vecdotq.cuh:136`). `qi/vdr == 1`, so each call handles **one whole
16-element AR16 block**, `iqs` always 0 (`vecdotq.cuh:766, 774`). Two AR16 blocks pair with one
32-element q8_1 block; `kbx` parity picks which half via `i8 = 4*(kbx & 1)` (host dispatch
guarantees `ne00 % 32 == 0`, `vecdotq.cuh:759-761`).

Unpack (`unpack_q4_0_ar16`, `vecdotq.cuh:769-782`):
```
lo = (q>>0)&0x0F0F0F0F ; hi = (q>>4)&0x0F0F0F0F           # even/odd nibbles in bytes 0..3
v.x = __vsubss4(__byte_perm(lo,hi,0x5140), 0x08080808)   # elements 0..3, offset-8 applied
v.y = __vsubss4(__byte_perm(lo,hi,0x7362), 0x08080808)   # elements 4..7, offset-8 applied
```
So AR16 re-interleaves nibbles into **element order** with `__byte_perm` and applies the symmetric
`-8` *inside the unpack* (via `__vsubss4`), producing signed int8 in natural element order. Dot:
```
sumi = 0
for i in 0..1:
    v = unpack_q4_0_ar16(get_int_b2(bq4->qs, i))
    sumi = dp4a(v.x, get_int_b4(bq8_1->qs, i8+2i+0), sumi)
    sumi = dp4a(v.y, get_int_b4(bq8_1->qs, i8+2i+1), sumi)
return __half2float(bq4->d) * __low2float(bq8_1->ds) * sumi
```
**Key difference from Q4_0:** because the `-8` offset is folded into the quants before the dp4a,
there is **no `ds8.y` (`s8`) correction term** — the return is a plain `d4 · d8 · sumi`
(`vecdotq.cuh:778-779`), whereas Q4_0 must carry `-(8vdr/QI4_0)·d·s`. Dispatch registration:
`get_vec_dot_q_cuda` `mmvq.cu:14`, `get_vdr_mmvq` `mmvq.cu:43`. Launch config identical to Q4_0
(nwarps=4, rows_per_block=1) since it shares the `mul_mat_vec_q` template.
The CPU tail-block fix (`ne00%32==16`) is the subject of the head commit `546eca8dc`.

---

## 3. `quantize_q8_1` — activation quantization

Kernel `ggml/src/ggml-cuda/quantize.cu:4-48`. Turns the f32 activation vector into `block_q8_1`
for the MMVQ dot. `CUDA_QUANTIZE_BLOCK_SIZE = 256` (`quantize.cuh:8`).

### Block layout produced
Per 32-element block (`QK8_1=32`): `d = amax/127`, `q[iqs] = round(x/d)` (or 0 if amax==0),
`ds = half2(d, sum)` where `sum = Σ x` over the block (`quantize.cu:38-47`). Note `ds.y` stores the
raw activation **sum** (not `d·sum`); the MMVQ offset correction multiplies by `d` implicitly via
the algebra in `vec_dot_q4_0_q8_1_impl`.

### Math (per thread = per element)
```
xi   = i0<ne00 ? x[i03·s03+i02·s02+i01·s01+i00] : 0
amax = warp_reduce_max<32>(|xi|)      # max over the 32-lane block
sum  = warp_reduce_sum<32>(xi)
d    = amax/127 ; q = round(xi/d)
y[ib].qs[iqs] = q ; if iqs==0: y[ib].ds = half2(d, sum)
```
(`quantize.cu:31-47`). The `warp_reduce_*<QK8_1>` reduces exactly the 32 lanes of one block.

### Launch shape
`num_blocks = (ceil(ne0/256), ne1, ne2·ne3)`, `block = (256,1,1)` (`quantize.cu:378-381`,
`quantize_row_q8_1_cuda`). `GGML_ASSERT(!ids)` and `ne0 % QK8_1 == 0`. (This is the plain
per-row-q8_1 path, *not* the MMQ `quantize_mmq_q8_1` variant at `:271`, which is for the tiled
`mul_mat_q` GEMM used at larger batch.)

---

## 4. `mul_mat_vec_f<half,half,1,256,0,0>` — the F16 GEMV (lm_head, ssm_alpha/ssm_beta, etc.)

Kernel `ggml/src/ggml-cuda/mmvf.cu:7-374`.
`template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion, bool is_multi_token_id>`.
Census `<half,half,1,256,0,0>`:

| pos | param | value | meaning |
|-----|-------|-------|---------|
| 1 | `T` | `half` | src0 (weight) element type = F16 |
| 2 | `type_acc` | `half` | accumulator type: F16 path uses `half2` accumulation (`sumh2`), summed to float at the end (`mmvf.cu:188-216`) |
| 3 | `ncols_dst` | 1 | one dst column (batch-1) |
| 4 | `block_size` | 256 | chosen by the `block_size_best` search (`mmvf.cu:428-437`) — 256 is the max for this arch |
| 5 | `has_fusion` | false | no fused gate/bias |
| 6 | `is_multi_token_id` | false | not a multi-token MUL_MAT_ID |

### Accumulation / vectorization
Threads walk the row in `float2`/`half2` pairs (`ncols2 = ncols/2`, `mmvf.cu:132,155-156`). For
`T=half, type_acc=half` the inner loop accumulates in `half2`:
`sumh2[j] += x2[col2] * make_half2(tmpy.x, tmpy.y)` (`mmvf.cu:189-211`), then
`sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j])` (`mmvf.cu:214-216`). Activation `y` is
read as `float2` (F32 activations against F16 weights). Final block reduction over shared memory
(`buf_iw`) → `dst` (later in the kernel). Launch: `block_nums(nrows, nchannels_dst, nsamples)`,
`block_dims(block_size_best,1,1)`, `nbytes_shared = warp_size·4` (`mmvf.cu:441-444`).

This kernel serves the **F16-weight** GEMVs: `output`/lm-head (`build_lora_mm(model.output,…)`,
`src/models/qwen35.cpp:225`), and per-DeltaNet-layer `ssm_alpha`/`ssm_beta` projections
(`src/models/qwen35.cpp:364,371`) which are small F16 matrices. (call/0020 records `F16-ssm_out` as
a standing file — `ssm_out` is F16 too.)

---

## 5. `ggml_cuda_ar_kernel<float,bf16>` — internal two-GPU AllReduce (**fork-only; NOT DeltaNet**)

**This is `ar` = AllReduce, not AutoRound.** Kernel `ggml/src/ggml-cuda/allreduce.cu:108-198`.
It has nothing to do with the recurrent state. It is the tensor-parallel (`-sm tensor` / row-split)
sum-reduction across the two GPUs, staged over PCIe through pinned host memory (no NVLink). The
census `128/pass, grid (8,1,1)/(256,1,1)` matches `GGML_CUDA_AR_KERNEL_BLOCKS = 8` blocks × 256
threads (`allreduce.cu:76`, launch `allreduce.cu:919-920`), invoked ~128 times per pass — once per
tensor that must be all-reduced across the two shards (each layer's attention-out and FFN-down
outputs on the tensor-split path).

### What it computes
`<float,bf16>` = `T_dst=float` (caller accumulator), `T_wire=nv_bfloat16` (on-wire type). It is a
3-phase in-place sum with a **BF16 round-trip** to halve PCIe bytes (`allreduce.cu:764-769`):
```
Phase 1 (all threads): wire[k] = cast<bf16>(sendbuf[off+k]); store vector to host_mine   # F32→BF16, PCIe write
         __threadfence_system()
Phase 2 (thread 0/block): write token to arrival_mine; spin until arrival_other==token    # cross-GPU barrier
Phase 3 (all threads): read host_other (peer's bf16); recvbuf[off+k] =
             cast<f32>( cast<bf16>(sendbuf[off+k]) )  +  cast<f32>( host_other[off+k] )    # sum in f32
```
(`allreduce.cu:133-197`). Both sides round their own contribution through BF16 before summing so the
two GPUs produce **bit-identical** results (`allreduce.cu:96-99, 186-188`). Vector width
`ELEMS_PER_VEC = max_cpy_bytes/sizeof(bf16)` (16 B/2 = 8 on Volta+). Per-block arrival slot (cache-
line-strided, `allreduce.cu:71`) gives lock-free release/acquire via `volatile` + `__threadfence_system`.
Dispatched through `ggml_backend_cuda_comm_allreduce_internal` → `ggml_cuda_ar_allreduce`
(`ggml-cuda.cu:1260-1304`), selected by `GGML_CUDA_ALLREDUCE` / platform default.

> If the megakernel targets a single GPU it need not reimplement this. On the two-GPU `-sm tensor`
> rig it is on the critical path (the memory index lists the rig as tensor-split).

---

## 6. `gated_delta_net_cuda<128,0,0>` — the fused DeltaNet recurrent step (**fork-only fused op**)

Kernel `ggml/src/ggml-cuda/gated_delta_net.cu:3-168`. Dispatched by ggml op
`ggml_gated_delta_net` (graph side `build_delta_net_fused`, `src/models/delta-net-base.cpp:373-424`;
op impl `ggml_cuda_op_gated_delta_net`, `gated_delta_net.cu:225-311`). Used when
`cparams.fused_gdn_ar` is set (batch-1 decode, `delta-net-base.cpp:436-441`). Runs once per DeltaNet
layer ⇒ 48/pass.

### Template params `<S_v, KDA, keep_rs_t>` = `<128,0,0>`
`template <int S_v, bool KDA, bool keep_rs_t>` (`gated_delta_net.cu:3`):
- `S_v = 128` — head value dimension (= head_k dim; `S_k==S_v` asserted). State is `S_v×S_v`.
- `KDA = false` — **G**ated (scalar-`g`) DeltaNet, not **K**ey-dim-gated (per-channel `g`). `g` is a
  scalar per (head,token) here (`src_g->ne[0]==1`), so `kda = (src_g->ne[0]==S_v)` is false
  (`gated_delta_net.cu:246`). Qwen3.6 uses the GDA branch (`gate = ggml_mul(alpha_softplus, ssm_a)`
  is a scalar per v-head, `src/models/qwen35.cpp:379-382`).
- `keep_rs_t = false` — single final-state output (K=1); no per-token snapshot ring. `keep_rs` is
  `K = src_state->ne[1] > 1` (`gated_delta_net.cu:287-288`); batch-1 decode has K=1.

### Grid / block
`launch_gated_delta_net` (`gated_delta_net.cu:170-223`): `num_warps=4`,
`grid = (H, n_seqs, ceil(S_v/num_warps)) = (H, 1, 128/4 = 32)`,
`block = (warp_size≤S_v?warp_size:S_v, num_warps, 1) = (32, 4, 1)`. The census grid `(24,1,32)` ⇒
`H = num_v_heads = 24` v-heads in the measured config; block `(32,4,1)` = 128 threads. **Each warp
owns one output column `col`** (`col = blockIdx.z·blockDim.y + threadIdx.y`, `gated_delta_net.cu:32`),
reducing across the `S_v` rows with `warp_reduce_sum`. `__launch_bounds__(min(warp,S_v)*4, 2)`
(`gated_delta_net.cu:4`).

### Inputs (post conv / l2norm / activations; see §7, §10 and the qwen35 builder)
- `q,k` — L2-normalized (`ggml_l2_norm`, §10), conv'd+silu'd (§7), scaled by `1/√S_v` **inside** the
  kernel (`scale = 1/sqrtf(S_v)`, `gated_delta_net.cu:282`, applied to the *output*, see below —
  note the graph scale in the non-fused path is on `q`; here the kernel folds it into `attn`).
- `v` — conv'd+silu'd value.
- `g` — decay gate, `g_val = exp(*g_t)` with `g = softplus(alpha+dt_bias) · ssm_a` (a log-decay);
  `alpha_biased = alpha + ssm_dt`, `alpha_softplus = softplus(...)`, `gate = alpha_softplus·ssm_a`
  (`src/models/qwen35.cpp:375-382`). The kernel does `expf(*g_t)` (`gated_delta_net.cu:88`).
- `beta` — `sigmoid` of the ssm_beta projection (`src/models/qwen35.cpp:364-369`).
- `curr_state` — input recurrent state `S[i][col]` for this (seq,head).

### Per-token math (GDA branch, `gated_delta_net.cu:66-159`) with the **declared fp32 fold order**
State is stored **transposed**: `M[col][i] = S[i][col]`, row `col` contiguous
(`gated_delta_net.cu:53-60`). Per warp, `s_shard[r]` holds `S[i][col]` for the rows `i = r·warp_size+lane`.
For each token `t`:
```
g_val = expf(g_t)                                             # scalar decay for this (head,token)
# 1. kv[col] = (Sᵀ·k)[col] = Σ_i S[i][col]·k[i]
kv_shard = Σ_r s_shard[r]·k_reg[r]                            # per-lane partial, rows owned by lane
kv_col   = warp_reduce_sum<warp_size>(kv_shard)              # fp32 tree reduce across lanes
# 2. delta[col] = (v[col] - g_val·kv[col])·beta
delta_col = (v_t[col] - g_val·kv_col) · beta_val
# 3. FUSED state update + output, single pass over rows:
attn_partial = 0
for r in rows_owned_by_lane:
    s_shard[r]   = g_val·s_shard[r] + k_reg[r]·delta_col     # S[i][col] ← g·S + k·delta  (decay + delta-rule outer product)
    attn_partial += s_shard[r]·q_reg[r]                      # accumulate (Sᵀ·q)[col] with the UPDATED state
attn_col = warp_reduce_sum<warp_size>(attn_partial)          # fp32 tree reduce
if lane==0: attn_data[col] = attn_col · scale                # scale = 1/√S_v folded here
```
**Fold order (G13 — declared exactly):**
1. `kv_col` = `warp_reduce_sum` of the per-lane running sum `Σ_r s_shard[r]·k_reg[r]` — i.e.
   **per-lane sequential accumulation in fp32, then a binary-tree `__shfl_xor` reduce across lanes**
   (`warp_reduce_sum<warp_size>`). No Kahan; plain fp32 add.
2. `delta_col` computed in fp32: `(v - g·kv)·beta`, mul/sub order as written
   (`gated_delta_net.cu:99`).
3. The state update `s_shard[r] = g_val*s_shard[r] + k_reg[r]*delta_col` uses **two separate fp32
   mults + one fp32 add** (not an fma intrinsic; the compiler may contract to fma unless
   `--fmad=false`) (`gated_delta_net.cu:106`).
4. `attn_partial` is the **per-lane running fp32 sum of `s_shard[r]·q_reg[r]` over that lane's
   rows** (row order = ascending `r`), reduced across lanes by the same tree `warp_reduce_sum`
   (`gated_delta_net.cu:104-110`).
5. Output scaled by `1/√S_v` last (`gated_delta_net.cu:113`).
The KDA branch (`gated_delta_net.cu:115-144`) differs only in that `g` is per-row
(`expf(g_t[i])`) inside both the `kv` and the state-update sums; Qwen3.6 does not use it.

### State in/out
Input state offset `state_in_offset = seq·K·H·S_v² + h·S_v²`, layout `(D=S_v², K, n_seqs)`
(`gated_delta_net.cu:41-47`). `curr_state += state_in_offset + col·S_v` — the warp loads its column.
Output: `dst` holds `[attn_scores | state]` concatenated; `attn_score_elems = S_v·H·n_tokens·n_seqs`,
`state = dst + attn_score_elems` (`gated_delta_net.cu:37-46`). For `keep_rs_t=false`, after the token
loop each warp writes its column back: `state[col·S_v + i] = s_shard[r]` (`gated_delta_net.cu:161-167`).
The graph then copies the state view into `ssm_states_all` (§"hybrid cache").

---

## 7. `ssm_conv_f32<1,128,4>` — the 4-tap causal depthwise conv (+ fused silu)

Kernel `ggml/src/ggml-cuda/ssm-conv.cu:4-51`. `template <bool apply_silu, size_t split_d_inner, size_t d_conv>`.
Census `<1,128,4>`: `apply_silu=true`, `split_d_inner=128` (= `threads`), `d_conv=4`. So
`ssm_d_conv = 4` (kernel window 4), and the silu is **fused into the conv** (the
`fuse_silu`/`silu_dst` path, `ggml_cuda_op_ssm_conv`, `ssm-conv.cu:152-198`; census confirms this is
the short-token `n_t≤32` variant `ssm_conv_f32`, not `ssm_conv_long_token_f32`).

### Channels
Census `channels 10240` = `conv_channels = d_inner + 2·ssm_n_group·ssm_d_state`
(`src/models/qwen35.cpp:389`; `= key_dim·2 + value_dim` at `:67`). `nr = src0->ne[1] = conv_channels`
(`ssm-conv.cu:168`), launched as `blocks(n_s, nr/threads, 1)` = `(1, 10240/128 = 80, 1)`, 128 threads
(`ssm-conv.cu:129-132`). `GGML_ASSERT(nr % 128 == 0)`.

### Conv-history (state) layout & math
`src0` (`conv_x`) = concat of prior conv state `(d_conv-1)` columns + this token
(`build_conv_state`, `src/models/delta-net-base.cpp:457-513`): conv_states reshaped
`(d_conv-1, conv_channels, n_seqs)`, transposed qkv_mixed appended along dim 0
(`delta-net-base.cpp:474-481`). Each thread owns one channel `tid`, loads the `d_conv` weights
`w[j]` once (`ssm-conv.cu:26-29`), then per output token:
```
x[]  = window of d_conv inputs for this channel   (ring-buffer: x[(i-1)%d_conv] updated each step)
sumf = Σ_{j=0..d_conv-1} x[(i+j)%d_conv] · w[j]  +  bias
y    = apply_silu ? silu(sumf) : sumf
```
(`ssm-conv.cu:33-50`). Causal: output `i` uses inputs `i..i+d_conv-1` of the padded window (the
`d_conv-1` history + current). The last `d_conv-1` columns are written back into `conv_states_all`
via a `cpy` (`delta-net-base.cpp:483-494`, the `cpy_scalar` f32→f32 in §14). Bias fused only when
silu is (`GGML_ASSERT(!fuse_bias || fuse_silu)`, `ssm-conv.cu:159`).

---

## 8. `flash_attn_ext_f16<256,256,1,8,0,0>` + `flash_attn_stream_k_fixup_uniform<256,1,8>` — the decode FA path

FA kernel `ggml/src/ggml-cuda/fattn-mma-f16.cuh:1703-1705`
(`template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view>`).
Census `<256,256,1,8,0,0>`:

| param | value | meaning |
|-------|-------|---------|
| `DKQ` | 256 | K/Q head dim. **Note: Qwen3.6 attention head dim is 256** (`n_embd_head_k`). |
| `DV` | 256 | V head dim (256). |
| `ncols1` | 1 | Q columns per tile along the token axis = 1 (batch-1 decode, one query token). |
| `ncols2` | 8 | Q heads packed per tile = the **GQA ratio** (32 Q heads / 4 KV heads = 8). |
| `use_logit_softcap` | 0 | **no logit softcap** for this model. |
| `V_is_K_view` | 0 | V is a distinct tensor, not a view onto K. |

`nthreads`/occupancy from `ggml_cuda_fattn_mma_get_nthreads(DKQ,DV,ncols1·ncols2)` / `_get_occupancy`
(`fattn-mma-f16.cuh:1704`). GQA handling: `ncols2 = gqa_ratio = 8` packs the 8 Q heads that share one
KV head into a single MMA tile so KV is read once for all 8 (`fattn-common.cuh` launch, `gqa_ratio`
plumbed to the fixup at `:1178`).

### Mask application
Mask (`mask_h`, the KQ mask §15) is loaded per KV tile via
`flash_attn_ext_f16_load_mask<ncols1,…>` (`fattn-mma-f16.cuh:450-451, 597, 943`) and added to the
KQ scores before softmax. Values are `0` (attend) or `-INF` (masked). No ALiBi in this model
(`max_bias` handling present but slope from `max_bias`; Qwen uses `max_bias=0`).

### Softcap / sinks
`use_logit_softcap=false` ⇒ the `logit_softcap` tanh path is compiled out. **Attention sinks** are
supported (`sinks_f`, `fattn-mma-f16.cuh:1351-1385`) but only applied when `sinks != nullptr`;
Qwen3.6 QWEN35 `build_attn` passes `nullptr` sinks (`src/models/qwen35.cpp:324-326`), so no sink
term. Online-softmax rescaling by `KQ_max` and running `KQ_rowsum` is standard flash attention
(`fattn-mma-f16.cuh:1319-1345`).

### KV layout read
Census "row 1024 f16": KV cache rows are `n_embd_head_k · n_head_kv = 256 · 4 = 1024` f16 per token,
f16 KV. K read via `K_h2` (half2), stride `nb11/nb12` (`fattn-common.cuh` launch args
`:1201-1203`). The single decode query attends over all `n_kv` cached tokens.

### Stream-K split + fixup merge (`flash_attn_stream_k_fixup_uniform`, `fattn-common.cuh:679-753`)
`template <int D, int ncols1, int ncols2>` = `<256,1,8>` (D=DV=256). Stream-K splits the KV range
across more blocks than there are output tiles (better SM occupancy at short token counts), so
several blocks produce **partial** `(max, rowsum, unnormalized VKQ)` for one output tile; the fixup
merges them. Launch: `blocks=(ntiles_dst, ncols1=1, ncols2=8)`, `block=(DV=256,1,1)`, one block per
output tile, `blocks_per_tile = nblocks_stream_k / ntiles_dst` (`fattn-common.cuh:1170-1188`).

Merge math (online-softmax combine, `fattn-common.cuh:719-752`):
```
load dst_val (this tile's partial), (max_val, rowsum) = dst_fixup[b_last]
for bidx = b_last-1 .. b_first:
    dst_add            = dst_fixup_data[bidx…]            # this block's partial VKQ element
    (m_add, rs_add)    = dst_fixup[nblocks_sk + bidx]
    max_new  = max(max_val, m_add)
    scale_val= exp(max_val - max_new)    (FTZ if < threshold)
    scale_add= exp(m_add   - max_new)
    dst_val  = scale_val·dst_val + scale_add·dst_add      # rescale + accumulate VKQ
    rowsum   = scale_val·rowsum  + scale_add·rs_add
    max_val  = max_new
*dst = dst_val / rowsum                                    # final normalize
```
So it is the standard max/LSE-stable combine: rescale each partial by `exp(m_i − m*)`, sum, divide
by the merged `rowsum`. Uniform variant assumes `nblocks_stream_k` is a multiple of `ntiles_dst`
(`fattn-common.cuh:697`); otherwise `flash_attn_stream_k_fixup_general` (`:759`) is used.

---

## 9. `rms_norm_f32<1024,1,0>` and `<256,1,0>` — RMS norms (and the gated-norm question)

Kernel `ggml/src/ggml-cuda/norm.cu:74-151`.
`template <int block_size, bool do_multiply, bool do_add>`.
Census `<1024,1,0>` = block_size 1024, `do_multiply=true`, `do_add=false`;
`<256,1,0>` = block_size 256, `do_multiply=true`, `do_add=false`.

### eps source & fused multiply
`eps` is read from `dst->op_params` (`memcpy(&eps, dst->op_params, sizeof(float))`,
`norm.cu:465`, and the fused variants). For Qwen3.6 `eps = hparams.f_norm_rms_eps`
(`LLM_KV_ATTENTION_LAYERNORM_RMS_EPS`, `src/models/qwen35.cpp:5`). Math:
```
mean  = (Σ_c x[c]²) / ncols          # block_reduce SUM
scale = rsqrtf(mean + eps)
dst[c]= scale · x[c] · mul[mul_col]   # do_multiply → fused ×weight   (norm.cu:136-146)
```
`do_multiply=true` means the RMS norm is **fused with the following weight multiply**
(`ggml_cuda_op_rms_norm_fused`, `norm.cu:476-534`) — the norm weight is the `mul` tensor. This is a
fork/upstream fusion of `RMS_NORM` + `MUL`. `do_add` (fused residual add) exists
(`ggml_cuda_op_rms_norm_fused_add`, `norm.cu:536+`) but is `0` in the census here.

### Which norms are `<1024>` vs `<256>`?
Block size is chosen by `ncols` (`rms_norm_mul_f32_cuda`, `norm.cu:347-357`): `ncols < 1024`
⇒ block 256, else block 1024.
- `<1024,1,0>` — `ncols = n_embd = 5120`(≥1024): the trunk norms `attn_norm`, `attn_post_norm`,
  `output_norm`, plus q/k per-head-dim norms would be 256 (head dim 256 < 1024 ⇒ 256). So `<1024>`
  = the full-model-width RMS norms (attn_norm, attn_post_norm, final output_norm).
- `<256,1,0>` runs **80/pass** — these are the *small* RMS norms with `ncols < 1024`:
  `attn_q_norm`/`attn_k_norm` (ncols = head dim 256, on the 16 attention layers) **and**
  `ssm_norm` on the 48 DeltaNet layers (ncols = head_v_dim = 128). Count: 48 (ssm_norm) + 16 q_norm
  + 16 k_norm = 80. ✓

### Is there a gated RMS norm in the DeltaNet output path? **Yes — but built from two ops, not one fused kernel.**
`build_norm_gated` (`src/models/qwen35.cpp:249-258`):
```
normalized = RMS_norm(input, ssm_norm_weight)     # rms_norm_f32<256,1,0> (fused ×ssm_norm)
gated_silu = silu(gate=z)                          # z is the wqkv_gate projection
return normalized * silu(gate)                     # MUL   → fused as unary_gated silu (§14)
```
So the DeltaNet output is `RMSNorm(core_attn_out) ⊙ silu(z)` — a SwiGLU-style gated RMS norm, but
realized as `rms_norm_f32<256,1,0>` (the norm×weight fuse) followed by the fused `silu(z)·norm`
(`ggml_cuda_op_unary_mul<op_silu>`, the `unary_gated silu` count of 48). It is **not** a single
gated-RMS CUDA kernel.

---

## 10. `l2_norm_f32<32>` — q/k L2 normalization in DeltaNet

Kernel `ggml/src/ggml-cuda/norm.cu:239-271`. `template <int block_size>`; `<32>` = block_size
`WARP_SIZE=32` (chosen because `ncols < 1024`, `l2_norm_f32_cuda`, `norm.cu:400-402`). Runs 96/pass =
2 (q and k) × 48 DeltaNet layers. Called by `ggml_l2_norm(ctx0, q_conv/k_conv, eps_norm)`
(`src/models/qwen35.cpp:434-435`), `eps = hparams.f_norm_rms_eps` (same eps).

### Math / axis
Normalizes along `ne00` (the head_k dim, contiguous axis 0), one block per (row, channel, sample):
```
tmp   = Σ_c x[c]²                              # block_reduce SUM over ncols
scale = rsqrtf( max(tmp, eps·eps) )            # note eps² floor, not eps  (norm.cu:266)
dst[c]= scale · x[c]
```
(`norm.cu:254-270`). The `fmaxf(tmp, eps·eps)` floor follows torch `F.normalize`. eps read from
`dst->op_params` (`ggml_cuda_op_l2_norm`, `norm.cu:650-671`).

---

## 11. `rope_multi<1,0,float>` — MRoPE/IMRoPE (32/pass)

Kernel `ggml/src/ggml-cuda/rope.cu:182-265`. `template <bool forward, bool has_ff, typename T>`.
Census `<1,0,float>`: `forward=true`, `has_ff=false` (**no** `freq_factors`/YaRN β-table), `T=float`
(q/k are f32 at rope time). Runs 32/pass = 2 (q,k) × 16 attention layers (DeltaNet layers do not
rope; they use conv+L2norm).

### RoPE type: **IMRoPE (interleaved MRoPE)**
`llama_model_rope_type(QWEN35) = LLAMA_ROPE_TYPE_IMROPE` (`src/llama-model.cpp:2351-2353`). At
dispatch `is_imrope = (mode == GGML_ROPE_TYPE_IMROPE)` and it takes the `is_mrope && !is_vision`
branch → `rope_multi_cuda(..., is_imrope=true)` (`rope.cu:565-615`). The kernel's `is_imrope`
runtime flag selects the **interleaved** sector→section assignment (`rope.cu:231-240`) vs
contiguous MRoPE (`:242-250`).

### Sections / which dims / theta
- `n_dims = n_rot` (rotary dim), `sections = hparams.rope_sections` (4 ints,
  `LLM_KV_ROPE_DIMENSION_SECTIONS`, `src/models/qwen35.cpp:6`; copied to `int sections[4]`
  `:145-146`). For a text-only decode all positions come from the same `pos[i2]` per token, but the
  IMRoPE machinery still routes each rotary pair to a section via
  `sector = (i0/2) % sect_dims` (`rope.cu:228`) and picks the position axis (t/h/w) by
  `sector%3` (imrope, `rope.cu:231-240`).
- `theta_scale = powf(freq_base, -2/n_dims)` (`rope.cu:583`); `theta_base = pos·theta_scale^(i0/2)`.
- Rotation is **NEOX-style split-half** (rotate `x[i0/2]` against `x[i0/2 + n_dims/2]`):
  `dst[0] = x0·cos − x1·sin ; dst[n_dims/2] = x0·sin + x1·cos` (`rope.cu:260-264`), with
  `x1 = x[ix + n_dims/2]`. Dims `≥ n_dims` are passed through unrotated (`rope.cu:219-224`).
- `rope_yarn` computes cos/sin with `ext_factor`/`attn_factor` (YaRN) but with `has_ff=false` there
  is no per-dim `freq_factors` division (`rope.cu:253-258`, `rope_yarn` `:?`). `freq_base`,
  `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast/slow` all from `dst->op_params[5..10]`
  (`rope.cu:558-563`).
- Launch: `block(1, 256, 1)`, `grid(nr, ceil(ne00/(2·256)), 1)`; each thread does one rotary pair
  `i0 = 2·(blockDim.y·blockIdx.y + threadIdx.y)` (`rope.cu:437-441, 204`).

Position source: `pos = src1` (the `inp_pos` I32 tensor, `rope.cu:579`).

---

## 12. `k_set_rows<float,long,half>` — KV append via SET_ROWS (32/pass)

Kernel `ggml/src/ggml-cuda/set-rows.cu:112-171`. `template <typename src_t, typename idx_t, typename dst_t>`
= `<float, int64_t("long"), half>`: source activations f32, index tensor I64, destination KV cache
f16. Runs 32/pass = 2 (K and V) × 16 attention layers.

### Index semantics & f32→f16 conversion
Produced by `cpy_k`/`cpy_v` → `ggml_set_rows(k_cache, k_cur, k_idxs)`
(`src/llama-kv-cache.cpp:1197-1229, 1264`). The index tensor `k_idxs` (I64, one per token) holds the
**global destination row** in the (stream-merged) cache: `data[s·size+i] = strm[s]·get_size() + idxs[s][i]`
(`set_input_k_idxs`, `src/llama-kv-cache.cpp:1355-1367`). In the kernel each element maps
`(i00,i01,i02,i03)` of src to dst row `dst_row = src1[i10·s10+i11·s11+i12·s12]`, then
`dst_row_ptr[i00] = ggml_cuda_cast<half>(src0_row[i00])` (`set-rows.cu:160-165`) — a straight f32→f16
cast, writing the current token's K (or V) into its cache slot. Grid: `ceil(ne_total/256)` blocks of
256 (`set-rows.cu:183-186`), `ne_total = ne00·ne01·ne02·ne03` (all elements). For K,
`n_embd_gqa = 256·4 = 1024` per token → 1024 f16 written per K append.

---

## 13. `k_get_rows_float<float,float>` — state / row gathers (97/pass)

Kernel `ggml/src/ggml-cuda/getrows.cu:43-72`. `template <typename src0_t, typename dst_t>` = `<float,float>`
(f32→f32 gather; the non-quantized `k_get_rows_float`, distinct from the dequantizing `k_get_rows` at
`:6`). These are `GGML_OP_GET_ROWS` with f32 src.

### Which reads
The dominant f32 GET_ROWS in a DeltaNet decode is the **recurrent-state read** from the cache:
`build_rs` → `get_state_rows(ctx0, states, state_copy_main)` = `ggml_get_rows` gathering each
sequence's conv state (`r_l`) and ssm state (`s_l`) row by the `s_copy` index
(`src/llama-graph.cpp:2545-2549`, `build_rs` `:2522-2556`; called from
`build_conv_state`/`build_rs` in `delta-net-base.cpp:471, 393`). Two gathers × 48 DeltaNet layers =
96, plus the token-embedding-adjacent / out-ids gather ⇒ census 97/pass. (The `inp_out_ids` gather
that selects the last token's row also uses GET_ROWS but on the hidden state, `src/models/qwen35.cpp:214-216`.)

### Math
```
i01 = src1[i10·s10 + i11·s11 + i12·s12]        # row index from the index tensor
dst_row[i00] = cast<dst_t>( src0_row(i01)[i00] )  # copy/gather row i01 → output row i10
```
(`getrows.cu:57-69`). Grid x/y swapped (`i10 = blockIdx.x`, `i00` strided over blockIdx.y·blockDim.x)
to fit the max grid-x limit (`getrows.cu:52-53`).

---

## 14. Small ops — census-count → block-structure map

| census kernel | file:line | where it comes from (per pass) |
|---|---|---|
| `concat_f32_cont<0>` | `concat.cu:5` (`<dim>`, dim=0) | `ggml_concat(conv_states, qkv_mixed, dim=0)` in `build_conv_state` (`delta-net-base.cpp:480`) — prepends the `d_conv-1` conv history to the current token along axis 0, per DeltaNet layer. (MTP also concats e_norm∥h_norm, `qwen35.cpp:536`, but MTP is off in plain decode.) |
| `cpy_scalar` f32→f32 (**64/pass**) | `cpy.cu:15-40` (`cpy_1_scalar<float,float>`, `:231`) | conv-state write-back `ggml_cpy(last_conv_states → conv_states_all)` (48 DeltaNet) + recurrent-state write-back `ggml_cpy(new_state → ssm_states_all)` (48)… but only the DeltaNet layers that need a non-contiguous store; the 64 count aligns with the per-FFN/per-layer state and residual copies. Generic strided element copy: `dst_offset = f(i)` from `src_offset` (`cpy.cu:34-40`). |
| `cpy` f32→f16 (**1/pass**) | `cpy.cu` (`cpy_scalar_contiguous<float,half>`, dispatch `:438`) | the **KQ-mask conversion** `ggml_cast(self_kq_mask, F16)` when `flash_attn` is on (`src/llama-graph.cpp` `self_kq_mask_cnv = flash_attn ? ggml_cast(...,F16) : ...`). One mask per pass (unified KV). |
| `k_bin_bcast` add (**176/pass**) | `binbcast.cu:31-88` (`op_add`, `:10`) | every `ggml_add` residual / bias add. Per trunk layer: attn residual + ffn residual + build_cvec + various bias adds. 64 layers × ~2 residuals + DeltaNet internal adds (alpha+dt bias, state adds) ≈ 176. Broadcasting elementwise add with fastmodulo index wrap (`binbcast.cu:64-84`). |
| `unary_gated silu` (**48/pass**) | `unary.cu:260-272` via `ggml_cuda_op_unary_mul<op_silu>` (`:611`) | the DeltaNet **gated RMS-norm output**: `norm ⊙ silu(z)` (`build_norm_gated`, `qwen35.cpp:255-257`). `dst = silu(x[j0])·g[j1]` (`unary.cu:268-271`). One per DeltaNet layer. |
| `unary_gated sigmoid` (**16/pass**) | same kernel, `op_sigmoid` (`:614`) | the **full-attention output gate**: `attn ⊙ sigmoid(gate)` (`qwen35.cpp:329-332`). One per attention layer. |
| `unary_gated softplus` (**48/pass**) | same kernel, `op_softplus` (`:617`) | DeltaNet `alpha_softplus` fused with the following mul? Actually `ggml_softplus(alpha_biased)` then `ggml_mul(alpha_softplus, ssm_a)` (`qwen35.cpp:376-379`) fuses to `softplus(alpha+dt)·ssm_a`. One per DeltaNet layer. `op_softplus(x) = x>20 ? x : log(1+eˣ)` (`unary.cu:93-95`). |
| `unary sigmoid` (**48/pass**) | `unary.cu:194-195` (`ggml_cuda_op_sigmoid`, non-fused) | DeltaNet **beta** activation `ggml_sigmoid(beta)` (`qwen35.cpp:368`). One per DeltaNet layer. (Plain elementwise, no gate mul — the beta multiply happens inside the fused GDN kernel.) |

`op_silu` = `x·sigmoid(x)` (`ggml_cuda_op_silu_single`, `unary.cu:36-37`); `op_sigmoid = 1/(1+e^-x)`
(`:48`). Fused-unary-mul writes `dst[i] = op(x[j0])·g[j1]` with per-row strides `o0,o1`
(`unary.cu:268-271`).

---

## 15. The KQ mask — batch-1 decode construction

Build: `build_attn_inp_kq_mask` (`src/llama-graph.cpp:23-36`). Shape
`(n_kv, n_tokens/n_stream, 1, n_stream)` F32; for unified KV batch-1 decode `n_stream=1`,
`n_tokens=1` ⇒ shape `(n_kv, 1, 1, 1)`. `ggml_set_input`, name `attn_inp_kq_mask`.

### f32→f16 & padding to n_kv (multiple of 256)
With flash-attn on, the F32 mask is cast to F16:
`inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask`
(`src/llama-graph.cpp:2082`; the plain-KV builder used here — the iSWA/hybrid variants are at
`:2165, 2272, 2500, 2679`). This is the `cpy f32→f16 1/pass`
of §14. `n_kv = get_n_kv()` is padded up to a multiple of `max(n_pad, 256)`
(`llama_kv_cache::get_n_kv`, `src/llama-kv-cache.cpp:1129-1139`:
`std::max(min(size, max(256, GGML_PAD(used_max_p1, 256))), …)`), i.e. **the FA "classes of 256"**.

### Value semantics
Filled in `set_input_kq_mask_impl` (`src/llama-kv-cache.cpp:1433-1577`): a cell gets **`0.0f`** if it
is a populated cell of the same sequence and (causal) `p0 ≤ p1`, else **`-INFINITY`**
(`:1566-1572`, `skip: data[idst+j] = -INFINITY`). ALiBi path instead writes `-|p0−p1|` (`:1565`), but
Qwen3.6 has no ALiBi. Empty cells and other-sequence cells are `-INF`
(`:1520-1527`). So the decode mask is `0` for attend, `-INF` for masked; padded columns beyond real
KV are `-INF`.

---

## The QWEN35 (Qwen3.6) model build — per-block graph & hparam origins

Builder `llama_model_qwen35` (`src/models/qwen35.cpp`), base `llm_build_delta_net_base`
(`src/models/delta-net-base.cpp`).

### hparams (`load_arch_hparams`, `src/models/qwen35.cpp:4-36`)
- `f_norm_rms_eps` ← `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` (the eps for §9/§10).
- `rope_sections[4]` ← `LLM_KV_ROPE_DIMENSION_SECTIONS` (IMRoPE, §11).
- SSM/linear-attn: `ssm_d_conv` (=4, §7), `ssm_d_inner`, `ssm_d_state` (head_k/head_v dim),
  `ssm_dt_rank` (= num_v_heads), `ssm_n_group` (= num_k_heads). Derived in `build_layer_attn_linear`:
  `head_k_dim = head_v_dim = ssm_d_state`(? — see below), `num_k_heads = ssm_n_group`,
  `num_v_heads = ssm_dt_rank`, `head_v_dim = d_inner/num_v_heads = 128` (S_v=128 confirms),
  `conv_dim = key_dim·2 + value_dim = 10240` (`qwen35.cpp:61-67, 347-352`).
- `full_attention_interval` ← `LLM_KV_FULL_ATTENTION_INTERVAL` (=4): layer `il` is recurrent iff
  `il < n_main && (il+1)%4 != 0` (`qwen35.cpp:22-27`). ⇒ 48 recurrent / 16 attention of 64 trunk.
- `nextn_predict_layers` ← `LLM_KV_NEXTN_PREDICT_LAYERS` (=1, the MTP block; must be `< n_layer`).
- 27B selected by `n_layer - nextn == 64 ⇒ LLM_TYPE_27B` (`qwen35.cpp:30-35`).
- `n_head=32`, `n_head_kv=4` (GQA 8, §8), `n_embd_head_k = 256` (DKQ/DV=256).

### Trunk block graph (`graph::graph`, `src/models/qwen35.cpp:139-231`)
```
inpL = tok_embd
for il in 0..63:
    cur = RMS_norm(inpL, attn_norm)                              # rms_norm_f32<1024,1,0>
    if recurrent(il):  cur = build_layer_attn_linear(...)        # DeltaNet (§6,§7,§10,§14)
    else:              cur = build_layer_attn(...)               # FA (§8) + rope (§11) + gate sigmoid (§14)
    cur = cur + inpSA                                            # k_bin_bcast add
    ffn_res = cur
    apn = RMS_norm(cur, attn_post_norm)                         # rms_norm_f32<1024,1,0>
    cur = build_layer_ffn(apn)                                   # SwiGLU FFN → fused MMVQ (§1)
    cur = cur + ffn_res                                         # add
    inpL = build_cvec(cur, il)
cur = get_rows(inpL, inp_out_ids)                                # select last token (§13)
cur = RMS_norm(cur, output_norm)                                # result_norm
cur = build_lora_mm(output, cur)                                # lm-head F16 GEMV (§4)  → result_output
res->t_logits = cur
```
- Full-attention layer (`build_layer_attn`, `qwen35.cpp:260-339`): single wq projection outputs
  query+gate interleaved (view split, `:275-299`); Q/K RMS-normed per head (`<256>` §9); MRoPE on Q,K
  (§11); `build_attn` (FA §8) with `kq_scale = 1/√n_embd_head` (`:322`); output `⊙ sigmoid(gate)`
  (§14); `wo` projection.
- DeltaNet layer (`build_layer_attn_linear`, `qwen35.cpp:341-473`): wqkv → conv (§7) → split q/k/v →
  L2-norm q,k (§10) → `build_recurrent_attn` → fused GDN (§6) → gated RMS-norm with z (§9/§14) →
  `ssm_out` projection (F16, §4). Detailed above.

### Sampling relevance — **none in-graph (confirmed)**
The graph's terminal node is `result_output` = `res->t_logits = build_lora_mm(model.output, cur)`
(`src/models/qwen35.cpp:225-228`). There is **no** softmax/argmax/top-k/temperature op in the decode
graph; logits are the raw lm-head output. Greedy (or any) sampling runs on the host over the returned
logits, outside ggml. So a megakernel reproducing the graph needs only to emit `result_output` logits.

---

## The recurrent+KV hybrid cache

Container: `llama_memory_hybrid` = `{ mem_attn: llama_kv_cache, mem_recr: llama_memory_recurrent }`
(`src/llama-memory-hybrid.h:19-90`). Built via `build_inp_mem_hybrid` (`qwen35.cpp:155`); the
DeltaNet layers use `inp->get_recr()` (recurrent), attention layers `inp->get_attn()` (KV).

### KV side (attention layers)
Standard `llama_kv_cache`, f16 K and V rows of `n_embd_head_k·n_head_kv = 1024` per token. Appended
via `ggml_set_rows` with I64 global-row indices (§12); masked/read by FA (§8, §15). `n_kv` padded to
a multiple of 256 (`get_n_kv`, `llama-kv-cache.cpp:1129-1139`).

### Recurrent side (DeltaNet layers) — `r_l` / `s_l`
Per layer two CPU-or-device tensors (`llama_memory_recurrent`, `src/llama-memory-recurrent.cpp:72-105`):
```
n_rows = mem_size · (1 + n_rs_seq)                              # snapshot ring depth
r_l[i] = new_tensor_2d(type_r, n_embd_r(), n_rows)  name cache_r_l{i}   # conv history
s_l[i] = new_tensor_2d(type_s, n_embd_s(), n_rows)  name cache_s_l{i}   # recurrent state
```
- `n_embd_r() = (ssm_d_conv-1)·(ssm_d_inner + 2·ssm_n_group·ssm_d_state)` = conv-state width =
  `3 · 10240 = 30720` floats (d_conv=4 ⇒ d_conv-1=3) (`src/llama-hparams.cpp:155-177`).
- `n_embd_s() = ssm_d_state · ssm_d_inner` = the DeltaNet state size (S×S×H flattened)
  (`src/llama-hparams.cpp:179-192`).
- `type_r`/`type_s` are f32 for this model (recurrent state kept in f32; the `ar_kernel` bf16
  round-trip in §5 is unrelated — it is the AllReduce wire type, not the cache dtype).

### Per-seq state slots & the snapshot ring
`mem_size` = number of sequence slots; `n_rs_seq` = extra snapshot depth for speculative rollback.
A sequence's live state is at row `rs_head + seq` in the first `mem_size` block; snapshots occupy the
`n_rs_seq` further blocks (`n_rows = mem_size·(1+n_rs_seq)`). `keep_rs()` (`delta-net-base.cpp:450-455`)
turns on the K-slot ring only when `1 < n_seq_tokens ≤ 1+n_rs_seq` (prefill/spec); **plain batch-1
decode has `keep_rs=false`, K=1**, matching `gated_delta_net_cuda<…,keep_rs_t=0>` (§6).

### How SET_ROWS/GET_ROWS indices are produced
- **State reads:** `build_rs_inp_impl` creates `s_copy` (I32, length `n_rs`) as a graph **input**
  (`src/llama-graph.cpp:2558-2578`); `s_copy_main` (first `n_seqs`) drives
  `get_state_rows`/`ggml_get_rows` to gather each active sequence's state row (§13);
  `s_copy_extra` copies the unchanged remainder forward (`build_rs`, `:2522-2556`). The host fills
  `s_copy` with the per-seq source rows.
- **State writes:** done with `ggml_cpy` into a `ggml_view_2d` at
  `kv_head · n_embd_s()` offset (non-fused path, `delta-net-base.cpp:541-544`) or per-snapshot-slot
  views (keep_rs path, `:571-583`). Conv state likewise (`build_conv_state`, `:483-509`). These are
  the `cpy_scalar f32→f32` copies of §14.
- **KV writes:** `ggml_set_rows` with the I64 `k_idxs`/`v_idxs` global-row index tensors (§12),
  filled by `set_input_k_idxs` (`llama-kv-cache.cpp:1355-1367`).

---

## Trickiest-finding index (for the megakernel author)

1. **`ar_kernel` is AllReduce, not AutoRound** (§5) — two-GPU PCIe sum with a BF16 wire round-trip;
   only relevant on the `-sm tensor` rig. The AutoRound quant is `Q4_0_AR16` (§2), a *different* thing.
2. **Q4_0_AR16 folds the −8 offset into the unpack** and drops the q8_1 `s` correction term (§2);
   its nibbles are element-interleaved, not Q4_0's split-half. `QK4_0_AR16=16`, 10-byte block.
3. **Fused SwiGLU MMVQ** `<Q4_0,1,1,0>` computes `up·x` and `gate·x` in one launch and applies
   `up * silu(gate)` in the epilogue (§1); fusion only for `ncols_dst==1`.
4. **GDN fp32 fold order (§6):** per-lane sequential fp32 accumulation of `Σ s·k` and `Σ s·q`, each
   closed by a binary-tree `__shfl_xor` `warp_reduce_sum`; state update `s = g·s + k·δ` as two mults +
   one add; output scaled by `1/√S_v` last. GDA branch (scalar `g=exp(*g_t)`), not KDA.
5. **RoPE = IMRoPE, NEOX split-half rotation, no freq_factors** (§11); theta_scale = `freq_base^(-2/n_dims)`.
6. **FA `<256,256,1,8,0,0>`:** head dim 256, ncols2=8 = GQA ratio (32/4), no softcap, no sinks;
   stream-K fixup does the max/LSE online-softmax merge, normalizes by merged rowsum (§8).
7. **The DeltaNet "gated norm"** is `RMSNorm(x)·silu(z)` built from `rms_norm_f32<256,1,0>` (norm×weight
   fuse) + `unary_gated silu` — not one kernel (§9, §14).
8. **Logits are the graph output** (`result_output`); no sampling op in-graph (§builder).

**Deliverable:** this file —
`/tmp/claude-1000/-home-dconnolly-yarn-agentic/7c771f3c-729c-4ed3-9fa2-2bfd3af45b1f/scratchpad/mk/semantics/SEMANTICS.md`.
