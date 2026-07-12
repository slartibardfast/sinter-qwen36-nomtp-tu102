# Prefill-op exact-semantics dossier — Qwen3.6-27B hybrid, batch N>1, `-sm tensor`, `-fa on`, f16 KV

Fork: `software/llama.cpp/autoround` @ `546eca8dc` (branch `autoround`). All cites are
`path:line` relative to that worktree root. Read-only survey. Sibling of the decode dossier
`docs/SEMANTICS.md` (read it first — this file assumes its op-set, its census, its §-numbers, and
its house style: explicit indexed pseudocode, fold orders declared, `file:line` throughout).

**Scope.** This dossier extracts *how the fork prefills* the served model (batch N>1: tensor-core
GEMM, batched causal FA, the DeltaNet prefill scan, batched KV/state write), so plan/0137 can
reimplement it inside the persistent kernel. The model identity, the recurrent+KV hybrid cache, the
per-block graph, and the hparam origins are all in `SEMANTICS.md`'s closing sections and are not
repeated; only the **N>1 deltas** are documented here.

**Model constants used throughout** (origins in `SEMANTICS.md`): 64 trunk blocks = 48 gated-DeltaNet
(`(il+1)%4 != 0`) + 16 full-attention (`src/models/qwen35.cpp:22-27`); `S_v = S_k = head_v_dim =
head_k_dim = ssm_d_state = 128`; `H = num_v_heads = ssm_dt_rank = 24`; `num_k_heads = ssm_n_group`;
`d_inner = ssm_d_inner = 3072`; `conv_channels = 10240`; attention head dim `n_embd_head_k = 256`,
`n_head = 32`, `n_head_kv = 4` (GQA 8). Recurrent state is **f32** (`type_r=type_s=GGML_TYPE_F32`,
`src/llama-model.cpp:2021-2022`).

---

## 0. Headline — the fork's DeltaNet prefill is a token-loop, not a chunk-parallel scan

The crux of 0137 is the chunked gated-DeltaNet prefill. The single most important finding of this
survey:

> **The fork's *production* DeltaNet prefill path is the same single-step recurrent CUDA kernel as
> decode, run as a sequential `for (t = 0; t < N; t++)` token loop inside one launch. It is NOT a
> chunk-parallel scan.** The kernel author says so in a `TODO` beside the launcher:
> `//TODO: Add chunked kernel for even faster pre-fill` (`ggml/src/ggml-cuda/gated_delta_net.cu:181`).

A genuinely chunk-parallel algorithm (intra-chunk matmuls, decay masks, a triangular solve,
per-chunk state carry) **does exist in the fork** — `build_delta_net_chunking`
(`src/models/delta-net-base.cpp:16-287`) — but it is the **disabled fallback**, reached only when the
fused op is turned off (device-placement mismatch on the tensor-split rig). Both paths are documented
below (§1 crux). This distinction changes 0137's difficulty completely and is the subject of the
TRACTABILITY VERDICT (§6).

Dispatch selecting which runs — `build_delta_net` (`src/models/delta-net-base.cpp:426-448`):
```
n_seq_tokens = q->ne[2]                                     # = tokens in this ubatch (:434)
if n_seq_tokens == 1:                                       # DECODE
    return fused_gdn_ar ? build_delta_net_fused            # → GGML_OP_GATED_DELTA_NET, name __fgdn_ar__
                        : build_delta_net_autoregressive    # → ggml-op single step (no fused op)
# n_seq_tokens > 1:                                         # PREFILL
if fused_gdn_ch:  return build_delta_net_fused              # → GGML_OP_GATED_DELTA_NET, name __fgdn_ch__  ← PRODUCTION
return build_delta_net_chunking                             # → the chunk-parallel ggml subgraph  ← FALLBACK
```
Both `fused_gdn_ar` and `fused_gdn_ch` default **true** (`src/llama-context.cpp:176-177`); the
`auto_fgdn` resolver at context construction reserves a check graph and only disables a flag on a
GDN-vs-layer **device mismatch** (`src/llama-context.cpp:491-571`). `n_rs_seq` defaults **0**
(`src/llama-context.cpp:3326`), so `keep_rs()` is always false in the served config
(`src/models/delta-net-base.cpp:450-455`) ⇒ prefill runs the fused op with the snapshot slot count
`K = 1` (final state only).

---

## 1. THE CRUX — chunked gated-DeltaNet prefill

### 1.0 Two paths, one op-boundary

Both prefill paths consume the identical post-conv/L2norm/activation inputs that the DeltaNet layer
builds (`build_layer_attn_linear`, `src/models/qwen35.cpp:341-473`, unchanged from decode except that
`n_seq_tokens = N`; see `SEMANTICS.md` §6/§7/§10 for conv, L2norm, and the `g = softplus(α+dt)·ssm_a`,
`β = sigmoid(...)` activations) and both produce the same two outputs — the per-token attention output
`[S_v, H_v, N, n_seqs]` and the final recurrent state `[S_v, S_v, H_v, n_seqs]`. They differ only in
*how* the scan between those boundaries is computed.

Inputs at the boundary (all f32): `q,k` = `[S_k=128, H_k, N, n_seqs]` (L2-normed, conv+silu'd),
`v` = `[S_v=128, H_v=24, N, n_seqs]`, `g` = `[1, H_v, N, n_seqs]` (scalar log-decay per (head,token)),
`β` = `[1, H_v, N, n_seqs]`, and input state `s` = `[S_v, S_v, H_v, n_seqs]`.

### 1.1 PRODUCTION path — the fused single-step kernel over N tokens (`build_delta_net_fused`)

`build_delta_net_fused` (`src/models/delta-net-base.cpp:373-424`) reshapes `s` to the 3-D
`(S_v·S_v·H_v, K=1, n_seqs)` form and emits **one** `ggml_gated_delta_net` op
(`:402-403`), named `__fgdn_ch__` because `n_tokens>1` (`:406-408`). The op node's result tensor is
`[S_v·H_v, N·n_seqs + K·S_v·n_seqs, 1, 1]` f32 — attention scores concatenated with the state slots
(`ggml/src/ggml.c:6211-6218`); the builder views the two regions back out at `:410-421`.

The op is dispatched to `ggml_cuda_op_gated_delta_net` → `gated_delta_net_cuda<S_v=128, KDA=false,
keep_rs_t=false>` (`ggml/src/ggml-cuda/gated_delta_net.cu:225-311`, launch `:170-223`). **This is the
exact same kernel and template instantiation decode uses** (`SEMANTICS.md` §6); `n_tokens` is read
from `v->ne[2]` (`gated_delta_net.cu:243`) and is the only thing that differs from decode. Grid/block
are unchanged: `grid=(H=24, n_seqs, ceil(S_v/4)=32)`, `block=(32,4,1)`, each warp owns one output
column `col`, `s_shard[rows_per_lane]` holds that column of the transposed state in registers
(`gated_delta_net.cu:50-60`).

**The scan is a strictly sequential token loop (`gated_delta_net.cu:66-159`), GDA branch
(`!KDA`, `:87-114`).** The state `s_shard` is *register-resident across the whole loop* — there is no
per-chunk re-materialization; it is one continuous fp32 recurrence:

```
# per (head h_idx, sequence, column col, lane); s_shard[r] = S[i=r*32+lane][col] in registers
for t in 0 .. N-1:                                              # ASCENDING token order — the outer fold axis
    load k_reg[r], q_reg[r]  = k_t[i], q_t[i]                   # rows i owned by this lane
    g_val    = expf(g_t)                                        # scalar decay this (head,token)   :88
    # (1) kv[col] = (Sᵀ·k)[col] = Σ_i S[i][col]·k[i]
    kv_shard = Σ_r  s_shard[r] * k_reg[r]                       # per-lane sequential fp32 sum     :92-95
    kv_col   = warp_reduce_sum<32>(kv_shard)                    # binary-tree __shfl_xor, fp32     :96
    # (2) delta[col] = (v[col] - g_val·kv[col])·beta
    delta_col = (v_t[col] - g_val * kv_col) * beta_val          # fp32, order as written           :99
    # (3) FUSED state update + output, single ascending-r pass:
    attn_partial = 0
    for r in 0 .. rows_per_lane-1:                              # ascending r
        s_shard[r]   = g_val * s_shard[r] + k_reg[r] * delta_col   # 2 mults + 1 add (may contract to fma) :106
        attn_partial += s_shard[r] * q_reg[r]                      # accumulate with UPDATED state        :107
    attn_col = warp_reduce_sum<32>(attn_partial)               # binary-tree fp32 reduce           :110
    if lane == 0:  attn_data[col] = attn_col * scale           # scale = 1/√S_v folded LAST        :113,282
# after the loop (keep_rs_t=false): write the final state column back
for r: state[col*S_v + i] = s_shard[r]                         # final state → dst state region   :161-167
```

**Declared fold order (G13) — production prefill:** identical to decode's declared fold order
(`SEMANTICS.md` §6, the five-step list), *applied per token with the token axis folded strictly
sequentially ascending `t = 0 → N-1`*. The inter-token carry is the register-resident `s_shard`
(no summation, no snapshot); the two warp reductions (`kv_col`, `attn_col`) are binary-tree
`warp_reduce_sum<32>`; the state update is two fp32 mults + one fp32 add; the output is scaled by
`1/√S_v` last. Because it is literally decode's kernel iterated, **production prefill is bit-parity
reproducible against decode's recurrence** — a megakernel that reuses this recurrence gets exact
parity for free.

**Multi-ubatch carry (the real "inter-chunk" seam in production).** A prompt longer than one ubatch
is split into ubatches of ≤ `n_ubatch` tokens (`n_ubatch` defaults to `n_batch`, typically 2048,
capped by `n_ctx`; `src/llama-context.cpp:181-183`, split at `memory->init_batch(..., n_ubatch, ...)`,
`:1702`). Each ubatch: **reads** the layer's state from the cache (`build_rs`, `src/models/qwen35.cpp:393`
→ `SEMANTICS.md` §13), runs the fused kernel over that ubatch's N tokens, then **writes** the single
final state back to `ssm_states_all` (one `ggml_cpy`, `src/models/delta-net-base.cpp:541-544`). So in
production the "chunk size" is the ubatch (≤2048) and the inter-chunk state carry is a **cache
round-trip**, not an in-register hand-off. Within a ubatch the carry is register-resident across all N
tokens as above.

### 1.2 The reduction-to-decode seam (where prefill hands the final state to decode's bank 0)

Trivial in production, by construction. Prefill's last ubatch runs the same recurrence and writes its
final `s_shard` to `ssm_states_all` at the `kv_head` slot (`src/models/delta-net-base.cpp:541-544`).
The first decode step then **reads that same cache row** via `build_rs`/`get_state_rows`
(`src/models/qwen35.cpp:393`; `SEMANTICS.md` §13) as its bank-0 initial state and runs
`gated_delta_net_cuda<128,0,0>` for one token. The state tensor layout, dtype (f32), and per-(seq,head)
offset (`state_in_offset = seq·K·H·S_v² + h·S_v²`, `gated_delta_net.cu:41-47`) are **identical** for
the last prefill token and the first decode token — the same kernel, `K=1`. There is no conversion at
the seam; prefill's final state *is* decode's bank-0 state. For 0137's single-engine goal this is the
clean part: KV and DeltaNet state live in one layout from first prompt token to last decode token.

At chunk-size 1 the production scan already *is* the decode single-step recurrence (`N=1` re-enters
the same loop body once). The fallback chunk algorithm (§1.3) reduces to it only in the limit
`CS→1`; at its real `CS=64` it is a different arithmetic (see §6).

### 1.3 FALLBACK path — the chunk-parallel scan (`build_delta_net_chunking`)

Reached only when `fused_gdn_ch` was auto-disabled. This is the **algorithmic template** 0137 wants
for a tensor-core prefill (the chunk-parallel gated delta rule / "UT transform"), realized as a graph
of ~30 ggml ops over f32 tensors. Chunk size **`CS = 64`** for the GDA (scalar-`g`) case this model
uses (`kda=false`; `CS = kda ? 16 : 64`, `src/models/delta-net-base.cpp:61`).

**Tiling** (`src/models/delta-net-base.cpp:55-85`): after permuting q,k,v,g,β to
`[S, N, H, n_seqs]`, the token axis is padded to a multiple of `CS`
(`pad = (CS - N%CS)%CS`, `n_chunks = (N+pad)/CS`, `:63-64`; `ggml_pad` zero-fills, `:66-70`) and
reshaped so `[S, CS, n_chunks, H·n_seqs]` — token `t = chunk·CS + local`, `local∈[0,CS)` the inner
`ne1`, `chunk` the `ne2` (`:78-85`). Padded tokens carry `k=v=β=0` and `g=0` (⇒ decay factor
`exp(0)=1`, no state change), so the final chunk's state equals the state after the last real token.

**Intra-chunk contribution (all `ggml_mul_mat`, i.e. cuBLAS on f32).** The cumulative log-decay is a
prefix sum `g_cs = cumsum(g)` (`:89`, `ggml/src/ggml-cuda/cumsum.cu` — CUB `BlockScan` inclusive
sum, fp32, per row). The intra-chunk **decay mask** is `exp(tril(g_cs[j] − g_cs[i]))`
(`:126-136`): `decay[i][j] = exp(g_cs_j − g_cs_i)` for `j ≥ i`, zero above the diagonal
(`ggml_tri(..., GGML_TRI_TYPE_LOWER_DIAG)`, `:134`; `tri` zeros outside the kept triangle,
`ggml/src/ggml-cuda/tri.cu:26-40`). Two `[CS,CS]` score matrices per chunk are formed and masked:
```
kb[i][j] = ( Σ_s k[s][i]·k_b[s][j] ) · decay[i][j]        # k·k_b, decayed   :139-140
kq[i][j] = ( Σ_s k[s][i]·q[s][j]  ) · decay[i][j]         # k·q,   decayed   :143-144
kq       = tri(kq, LOWER_DIAG)                            # strictly causal within chunk :147
```
where `k_b = k·β`, `v_b = v·β` (`:72-73`). These `ggml_mul_mat` calls are `[CS×S]·[S×CS] = [64×128]·
[128×64]` f32 GEMMs → cuBLAS with the default **`CUBLAS_TF32_TENSOR_OP_MATH`** math mode
(`ggml/src/ggml-cuda/common.cuh:1459`), i.e. **TF32 tensor cores**.

**The delta-rule inverse (the hard sequential kernel).** The chunked delta rule needs
`T = (I − tril(k·k_b·decay))⁻¹` (the WY/UT representation). The fork builds it by an explicit
triangular solve (`:152-167`):
```
attn     = tri(kb, LOWER)            # strict lower (exclude diagonal)   :152
identity = diag(fill(view,1))        # I_[CS×CS]                          :155-158
lhs      = attn + identity           # I + strict_lower(kb)              :160
attn     = -attn                                                         :163
lin_solve= solve_tri(lhs, -strict_lower(kb), left=true, lower=true, uni=false)   # solves lhs·X = -attn  :166
attn     = lin_solve + identity      # = (I + tril)^{-1}                 :167
```
`ggml_solve_tri` is **O(n³) forward/back-substitution**, not a matmul (`ggml/include/ggml.h:2523-2542`).
Here `A` is `[CS=64, CS]` and `B` is `[CS=64, CS]`, so `n=64, k=64`; since `k=64 > MAX_K_FAST=32` the
CUDA op takes the **`cublasStrsmBatched`** branch (batched triangular solve,
`ggml/src/ggml-cuda/solve_tri.cu:264-274, 30-79`), which **explicitly forces `CUBLAS_DEFAULT_MATH`
(no TF32 tensor cores) — "without this we get RMSE errors"** (`solve_tri.cu:70-76`) — full FP32. This
is the piece that does not fit an HMMA tile and is precision-sensitive (see §6).

The intra-chunk output and the state read-in are then assembled with more `mul_mat`:
`v_new = v_b·T − k_cd·s` (state contribution subtracted per chunk), `v_attn = kq·v_new`,
`attn_inter = q_g_exp·s` (`:171-256`), where `k_cd` (`k_cumdecay`), `q_g_exp`, and the key-gdiff
tensors fold the per-position `exp(g_cs)` / `exp(g_last − g_cs)` decays (`:174-227`).

**Inter-chunk state carry — the declared fold order for the fallback.** The state is carried by a
sequential loop over chunks (`src/models/delta-net-base.cpp:235-274`), ascending `chunk = 0 →
n_chunks-1`:
```
s = reshape(state_in, [S_v, S_v, 1, H_v·n_seqs])                         # :229
for chunk in 0 .. n_chunks-1:                                            # ASCENDING — the inter-chunk fold axis
    v_prime    = ch_k_cd · s                       # [CS,S_v]  state read into this chunk    :243
    v_t_new    = ch_v_t − v_prime                  # subtract state contribution             :247
    v_attn     = v_t_new · ch_kq                   # intra-chunk (causal, decayed)           :251
    attn_inter = s · ch_q_g_exp                    # inter-chunk (from carried state)         :255
    o_ch       = attn_inter + v_attn               # this chunk's output                     :259
    v[:, chunk] = o_ch                                                                       :262
    kgv        = ch_kg_t · v_t_new                 # [S_k,S_v] key-gdiff ⊗ new value         :266
    s          = s · ch_g_last_exp_t               # decay carried state by exp(g_last)      :271
    s          = s + kgv                           # add this chunk's contribution           :272
```
i.e. `s_{chunk+1} = s_chunk · exp(g_last_chunk) + Kgdiff_chunkᵀ · V_new_chunk`. Each `·` is a
`ggml_mul_mat` (cuBLAS TF32). **Declared fold order (G13) — fallback prefill:** inter-chunk carry
sequential ascending over chunks; each chunk's intra-chunk terms are cuBLAS-GEMM reductions
(TF32 tensor-core accumulation order, not a hand-declared tree) except the `T`-matrix solve which is
a batched STRSM in **full FP32**. This fold order is **not** bit-identical to the §1.1 sequential
recurrence (matmul/TF32 associativity differs from the per-token fp32 running sum) — it agrees only to
tolerance. Final state reshaped to `[S_v, S_v, H_v, n_seqs]` and truncated of padding (`:277-284`).

### 1.4 State tensor shapes/dtypes — prefill vs decode

| tensor | prefill (N>1) | decode (N=1) | dtype | cite |
|---|---|---|---|---|
| DeltaNet input state `s` (per layer, per seq) | `[S_v,S_v,H_v,n_seqs]` = `[128,128,24,1]`, reshaped `(393216, K=1, n_seqs)` | same | f32 | `qwen35.cpp:393-394`, `delta-net-base.cpp:402` |
| DeltaNet output state | `[S_v,S_v,H_v,n_seqs]`, **one final** state written | same | f32 | `delta-net-base.cpp:541-544` |
| GDN op result | `[S_v·H_v, N·n_seqs + S_v·n_seqs, 1,1]` (scores ⧺ state) | `[S_v·H_v, 1 + S_v, 1,1]` | f32 | `ggml.c:6211-6218` |
| ssm-state cache row `n_embd_s` | `128·3072 = 393216` per (layer,seq) | same | f32 | `llama-hparams.cpp:193` |
| conv-state cache row `n_embd_r` | `3·10240 = 30720` per (layer,seq) | same | f32 | `llama-hparams.cpp:176` |
| q,k / v / g / β into the scan | `[128,H,N,ns]` / `[128,24,N,ns]` / `[1,24,N,ns]` / `[1,24,N,ns]` | N=1 | f32 | `delta-net-base.cpp:381-399` |

The cache tensors and their per-(seq,head) layout are **identical** across prefill and decode — the
only difference is `N` on the transient scan tensors. With `keep_rs()=false` (served config), prefill
writes exactly **one** final conv-state and **one** final ssm-state per (layer,seq) regardless of N
(`delta-net-base.cpp:483-494` conv, `:534-546` ssm); the snapshot-ring `keep_rs` path (K = n_rs_seq+1
per-token snapshots, `:495-509`, `:549-585`) is dormant because `n_rs_seq=0`.

---

## 2. Batched causal attention prefill (the full-attention layers)

Same MMA/HMMA flash-attention kernel family as decode, retiled for N query tokens; **no separate
prefill kernel**. Dispatch: `ggml_cuda_flash_attn_ext` → `ggml_cuda_get_best_fattn_kernel`
(`ggml/src/ggml-cuda/fattn.cu:340,540`); for head-dim 256 with GQA opt applicable and Turing MMA
available it returns `BEST_FATTN_KERNEL_MMA_F16` for both N=1 and N>1 (`fattn.cu:457-478`).

**Template instantiation (Turing sm_75).** `switch_ncols2<256,256>` fixes `ncols2 = gqa_ratio = 8`
(the 8 Q heads sharing one KV head, `fattn.cu:37-92`). `switch_ncols1<256,256,8>` then picks `ncols1`
(query tokens per tile) from `Q->ne[1] = N` (`fattn.cu:9-34`):
- N=1 → `ncols1=1` → decode's `flash_attn_ext_f16<256,256,1,8,0,0>` (`:14-18`);
- N>2 on Turing → **clamped to `ncols1=4`** by the arch guard `highest_compiled_arch(cc)==TURING`
  (`:28-32`), giving the **prefill instantiation `flash_attn_ext_f16<256,256,4,8,0,0>`** — 4 query
  tokens × 8 GQA heads = **32 columns per CTA tile** (Ampere+ would use `ncols1=8`, 64 cols).

**Tensor cores.** Yes — the same kernel uses Turing HMMA `mma.sync`. Turing lacks `m16n8k16`, so both
K·Qᵀ and P·V emit 2×/4× `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`
(`ggml/src/ggml-cuda/mma.cuh:982-987, 1010-1021`, guarded `TURING_MMA_AVAILABLE`,
`common.cuh:252-254`), staged with `ldmatrix.sync` (`mma.cuh:791,835,892`).

**Causal mask.** The KV-cache KQ mask is `ggml_new_tensor_4d(F32, n_kv, n_tokens/n_stream, 1,
n_stream)` (`src/llama-graph.cpp:32`), cast to F16 for FA (`:2082`). Decode `(n_kv,1,1,1)`; **prefill
`(n_kv, N, 1, 1)`** — one mask row per query token. The triangular structure is filled per (query i,
KV cell j) by comparing positions: `p1 = pos[i]`, `p0 = cells.pos_get(j)`, and **`if (p0 > p1) →
-INFINITY else 0.0f`** (`src/llama-kv-cache.cpp:1542-1573`, `idst = n_kv·i` at `:1477`). N distinct
`p1` cutoffs give the batched lower-triangular (causal) mask; the additive `0/−INF` mask is added to
the scores pre-softmax (loaded per KV tile by `flash_attn_ext_f16_load_mask`,
`fattn-mma-f16.cuh:451,597`). No ALiBi, no SWA, no sinks (`qwen35.cpp:324-326`).

**How it differs from decode's single-query FA:** `ncols1` 1→4 (different `ncols=8`→`32` resource
config, `fattn-mma-f16.cuh:71/93/159`); mask `(n_kv,1,1,1)`→`(n_kv,N,1,1)`; a long-prefill mask
pre-scan `flash_attn_mask_to_KV_max` that skips fully-masked 256-wide KV blocks activates only when
`N ≥ 1024` (`fattn-common.cuh:1035`). **Stream-K:** the MMA path always launches with `stream_k=true`
(`fattn-mma-f16.cuh:1953-1954`) but the *KV-splitting* is a runtime decision `use_stream_k = cc≥Ada ||
tiles_efficiency% < 75` (`fattn-common.cuh:1066`): on Turing decode (~4 output tiles) it is **on**
(splits KV + `flash_attn_stream_k_fixup_*` combine, `SEMANTICS.md` §8); at large-N prefill the tiles
saturate the SMs so it is effectively **off** (each block owns a whole output tile, no fixup). So the
stream-K fixup merge is a decode optimization, not a prefill one.

---

## 3. The prefill GEMMs — MMQ int8 tensor cores, not dequant→cuBLAS

At N>1 the trunk weight matmuls (qkv/gate/up/down/ssm_out, attention q/k/v/o) stop being memory-bound
GEMV and route through **MMQ** — the quantized int8 tiled GEMM — on Turing. Dispatch in
`ggml_cuda_mul_mat` (`ggml/src/ggml-cuda/ggml-cuda.cu:2538`), predicates at `:2547-2556`, ladder at
`:2599-2621`:
- `use_mul_mat_vec_q` requires `src1->ne[1] ≤ MMVQ_MAX_BATCH_SIZE = 8` (`ggml-cuda.cu:2553`,
  `mmvq.cuh:3`) — the **decode** path (also the AR16 tail guard `ne00%32==0` lives here, `:2554`).
- For **N > 8** MMVQ is out; `use_mul_mat_q` is gated by `ggml_cuda_should_use_mmq` (`:2579`), which on
  Turing returns **true unconditionally** (`turing_mma_available(cc)` ⇒ `mmq.cu:311-313`). So prefill
  **always uses MMQ** on sm_75, for all N>8; there is no N-threshold to cuBLAS.

**Both Q4_0 and Q4_0_AR16 have first-class MMQ kernels** (the fork added AR16 to every MMQ dispatch
point): `should_use_mmq` support list (`mmq.cu:279-280`), type switch
(`mmq.cu:11-16`), MMQ traits `mmq_type_traits<...,GGML_TYPE_Q4_0_AR16>` with `load_tiles_q4_0_ar16`
and the `q8_0_16` vec-dots (`mmq.cuh:511-559, 3338-3343`), and a compiled instance
`template-instances/mmq-instance-q4_0_ar16.cu:5`. **AR16 does not dequantize to f16 for cuBLAS at
prefill.** Activations are quantized to q8_1 (`quantize_mmq_q8_1_cuda`); the int8 tensor-core MMA
accumulates in **int32** (`mma.sync...s32.s8.s8.s32`, `mma.cuh:922-967`, C-tile `tile<16,8,int>`,
`mmq.cuh:1437-1439`), then per-block int32 partials are scaled by the f32 weight/activation scales and
summed into an **fp32** accumulator (`mmq.cuh:1499-1503, 1531-1538`). So the fork's prefill GEMM is
**int8 IMMA, fp32-accumulate**.

> **Divergence from 0137's README plan (worth an explicit call/).** The README proposes an **f16 HMMA**
> GEMM (Q4_0 dequantized to f16 in the operand load, fp32 accumulate). The fork does **not** do that —
> it keeps weights quantized and uses **int8 IMMA (MMQ)**. Both accumulate in fp32, so the call/0022
> precision ledger holds either way, but they are different kernels with different numerics. The only
> way the fork reaches an f16-HMMA cuBLAS GEMM for these quant types is if MMQ is force-disabled
> (`GGML_CUDA_FORCE_CUBLAS`) — and even then a plain Turing GeForce would default to
> `CUBLAS_COMPUTE_16F` (fp16 accumulate) unless the fp32-force override is set
> (`ggml-cuda.cu:1711-1741`). The DeltaNet chunk-path f32 matmuls (§1.3), by contrast, are cuBLAS
> **TF32** (`common.cuh:1459`).

---

## 4. Batched KV write (the full-attention layers)

The N>1 batched equivalent of decode's single `k_set_rows<float,int64_t,half>` (`SEMANTICS.md` §12)
is the **exact same op and kernel** — attention supplies an N-column f32 source and an N-entry I64
index tensor, and the CUDA grid grows to cover all N tokens in **one launch**:
- Graph: `cpy_k` builds `ggml_set_rows(k_cache, k_cur[n_embd_gqa, N], k_idxs[N])`
  (`src/llama-kv-cache.cpp:1214-1229`); `cpy_v` likewise for the FA (non-transposed) V path (`:1252-1264`).
  Index tensors are I64 1-D of length N (`build_input_k_idxs`, `:1288-1296`).
- Index fill: each of the N tokens gets a distinct global destination row
  `data[s·size+i] = strm[s]·get_size() + idxs[s][i]` (`set_input_k_idxs`, `:1362-1368`) — N contiguous
  rows `head..head+N-1` for a fresh single-stream slot.
- Kernel: `set_rows_cuda` computes `ne_total = ne00·ne01·ne02·ne03 = n_embd_gqa·N`
  (`ggml/src/ggml-cuda/set-rows.cu:183`) and does **one** `k_set_rows` launch of `ceil(ne_total/256)`
  blocks (`:184-206`); each token's `n_embd_gqa` elements scatter to `dst_row = k_idxs[token]`
  (`:158-165`). Decode is the same code with `ne01=1`. The f32→f16 cast is per-element and identical
  (`ggml_cuda_cast<half>`, `:165`, F16 dst branch `:232-234`).

DeltaNet state writes do **not** scale with N under `keep_rs()=false` (§1.4): the whole prefill window
collapses to one final conv-state copy (`delta-net-base.cpp:483-494`) and one final ssm-state copy
(`:534-546`) per (layer,seq).

---

## 5. Prefill graph shape / fingerprint dispatch

Prefill and decode use the **same builder and the same graph type** — there is no prefill/decode
`llm_graph_type` (`enum` = DEFAULT/ENCODER/DECODER/DECODER_MTP, `src/llama-graph.h:31-35`);
`build_arch_graph` branches only on MTP (`src/models/qwen35.cpp:132-137`) and never inspects batch
size; both enter via `process_ubatch(..., LLM_GRAPH_TYPE_DEFAULT, ...)` (`src/llama-context.cpp:1774,
25-31`). The token count enters as `ubatch.n_seq_tokens` and rides on tensor dim `ne[2]`
(`src/models/qwen35.cpp:348-357`; the DeltaNet dispatcher reads `q->ne[2]`,
`src/models/delta-net-base.cpp:434`). The context reserves **two** worst-case graphs — PP (prefill,
`graph_reserve(n_tokens,...)`, `src/llama-context.cpp:582`) and TG (decode, bs=1, `:602`) — recording
`n_nodes_pp`/`n_nodes_tg` separately (`:596-608, 636-645`), confirming two distinct instantiations.

**Is the prefill cgraph structurally different?** Depends on the fused flags:
- **Fused (served fast path): same op multiset, larger tensors** (`n_nodes_pp == n_nodes_tg`). One
  `GGML_OP_GATED_DELTA_NET` per DeltaNet layer, one `GGML_OP_FLASH_ATTN_EXT` per attention layer, the
  `inp_out_ids` GET_ROWS present in both (topology held constant by design,
  `src/llama-graph.cpp:1825-1831`). The distinguishing signals are **not** node count but: (1) the GDN
  op **node-name suffix** — `__fgdn_ch__` (prefill, `delta-net-base.cpp:407`) vs `__fgdn_ar__` (decode,
  `:405`); (2) the **`ne[2] = n_seq_tokens`** batch dim on the GDN and FA ops and `kq_mask->ne[1]`;
  (3) the `inp_out_ids` operand size `n_outputs`. The fork *already* fingerprints exactly this way in
  `auto_fgdn`/`auto_fa`: it walks `ggml_graph_node`, matches op type, and asserts the name prefix
  `__fattn__` / `__fgdn_ar__` / `__fgdn_ch__` (`src/llama-context.cpp:461-467, 504-550`; name defines
  `src/llama-impl.h:73-75`).
- **Non-fused fallback: structurally different** — the chunk subgraph (`cumsum`, `tri`×3, `solve_tri`,
  `diag`, `fill`, `pad`, extra `mul_mat`, plus a per-chunk unrolled loop whose node count scales with
  `n_chunks = ceil(N/CS)`, `delta-net-base.cpp:235-274`) has a different op multiset and `n_nodes_pp
  ≫ n_nodes_tg` — fingerprintable by raw node count.

**For 0137's fingerprint dispatch:** a batch-N prefill graph must fingerprint-miss the batch-1 decode
instantiation. In the fused config the miss is keyed by the GDN node-name suffix and the `ne[2]` batch
dim (node count alone is insufficient — it matches decode). Exclude the MTP head (`graph_mtp`,
`qwen35.cpp:491-631`, no GDN op) from the discrimination.

---

## 6. TRACTABILITY VERDICT

**The DeltaNet prefill crux is category (b) as the fork ships it, with category (a) available as a
harder reimplementation — and the honest recommendation is to ship (b) first.**

- **What the fork actually does at prefill (production, `fused_gdn_ch=true`, `n_rs_seq=0`) is (b): a
  loop of the single-step kernel.** `gated_delta_net_cuda<128,false,false>` iterates the exact decode
  recurrence over the ubatch's N tokens in one launch (`gated_delta_net.cu:66-159`), the author's own
  `TODO` conceding it is not chunk-parallel (`:181`). **This is cheap to reuse and gives bit-parity
  with decode for free** (same kernel, same fp32 fold order, §1.1) — but it is **slow**: sequential in
  N, S²=16384 fp32 work per token per head, no tensor cores, across all **48** DeltaNet layers. It is
  resident (one launch per layer per ubatch), so it is far better than the megakernel's current
  per-token decode-prefill crawl, but it leaves the DeltaNet layers as the prefill throughput floor.

- **A clean chunk-parallel algorithm (a) exists as the fork's disabled fallback**
  (`build_delta_net_chunking`, §1.3): intra-chunk contributions as decayed `[64×64]` matmuls (cuBLAS
  **TF32 tensor cores**), inter-chunk state carried by a sequential per-chunk matmul recurrence. 0137
  can reimplement this with HMMA; it is the right target for prefill parity-with-stock throughput. But
  it is **not bit-identical** to the sequential recurrence, so — consistent with README step 4 — 0137
  must declare its **own** chunk-parallel fold order for G13 and bind only **tolerance** parity to the
  fork, never bit-parity.

- **The single biggest implementation risk: the intra-chunk delta-rule inverse (the UT/WY
  `T = (I − tril(k·k_b·decay))⁻¹` transform).** In the fork it is a batched **triangular solve**
  (`ggml_solve_tri` → `cublasStrsmBatched`, `solve_tri.cu:264-274`), an O(CS³) forward/back
  substitution that (i) is **sequential within the chunk** (does not map onto an HMMA tile), and (ii)
  the fork runs in **full FP32 with tensor cores explicitly disabled** because TF32 injects unacceptable
  RMSE ("without this we get RMSE errors", `solve_tri.cu:70-76`). Reimplementing a per-chunk, per-head
  batched triangular solve *inside a persistent cooperative kernel*, at FP32, while every other GEMM in
  the block wants f16/int8 tensor cores, is the genuine research-grade hazard — it is the reason "just
  use tensor cores for the chunk scan" is not straightforward. Secondary risks: the chunk graph also
  needs `cumsum` (prefix-sum), `tri`/`diag`/`fill` masking, and `exp` decay tables inside the kernel;
  and the fork's prefill trunk GEMM is **int8 IMMA (MMQ)**, not the f16 HMMA the README assumes (§3) —
  0137 must decide whether to match MMQ or substitute its own f16 path (a call/0022-adjacent decision).

**Gate recommendation for whether 0137's full build is worth committing to:** yes, but stage it. Step
4 of the README (the chunked scan) should first land the **(b) token-loop recurrence** as the
correctness-and-parity baseline (trivial to make bit-exact vs decode, unblocks the unified single-engine
seam immediately, §1.2), then pursue the **(a) chunk-parallel** form as the throughput upgrade with the
triangular-solve as its isolated, separately-gated sub-problem (its own calx-mill/G13 derivation). Do
not commit to (a) as a prerequisite for unification — the seam, the KV/state ownership, and the
decode-parity re-pass (README step 6) are all achievable on (b) alone, and (b) is the escape hatch if
the in-kernel FP32 triangular solve proves intractable on sm_75.

---

**Deliverable:** this file —
`software/megakernel/main/docs/PREFILL-SEMANTICS.md`.
