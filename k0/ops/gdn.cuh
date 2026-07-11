// The gated-DeltaNet op family: the eight ops that realize one DeltaNet
// block's recurrent path (48 blocks per pass) under the k0 schedule.
//
//   OP_STATE_LOAD         get_rows fetch of a state-cache row (conv or ssm)
//   OP_CONV_SHIFT_CONCAT  conv window build + shifted-history commit
//   OP_SSM_CONV_SILU      4-tap causal conv + silu, per channel
//   OP_QK_L2NORM          per-head l2 norm of q and k
//   OP_GDN_GATES          softplus/sigmoid/exp gate math -> per-head g, beta
//   OP_GDN_STEP           the delta-rule state update + output partial (G13)
//   OP_GATED_RMSNORM      per-head rmsnorm x weight, then x silu(z)
//   OP_STATE_STORE        set_rows commit of a state-cache row
//
// Semantics are the fork's, ggml/src/ggml-cuda/gated_delta_net.cu:66-159
// (GDA branch, keep_rs=false), ssm-conv.cu:33-50, norm.cu:239-271 (l2) and
// :74-151 (rms, do_multiply), unary.cu op_softplus/op_sigmoid and
// unary.cuh ggml_cuda_op_silu_single, at autoround @546eca8dc. Batch-1
// decode only (n_tokens = 1), matching the k0 capture.
//
// fp32 fold order (G13, declared): inside OP_GDN_STEP every partial sum is
// a per-lane sequential fp32 accumulation over that lane's rows (ascending
// r, rows i = r*32 + lane), closed by the binary-tree __shfl_xor warp
// reduction; delta = (v - g*kv) * beta; the state update is two separate
// fp32 mults + one fp32 add; the output partial accumulates against the
// just-updated state in the same row loop; the 1/sqrt(S_v) scale is applied
// to the output LAST. Every mul-feeding-add in the declared folds is
// written with __fmul_rn/__fadd_rn/__fsub_rn, which ptxas never contracts
// into FMA, so the op's numerics are compiler-flag-invariant and match a
// plain-float CPU reference bit for bit (tests/test_gdn.cu).
//
// Loads: state caches, working buffers and activations are mutable data
// read across Y02 boundaries -> strong .cg loads (core/sync.cuh rule).
// Weights (conv kernel, dt bias, ssm_a, ssm_norm.weight) are immutable ->
// plain loads. Shared memory: this family uses NONE of the 60 KiB slab;
// state tiles live in registers (12 f32 per lane in OP_GDN_STEP).
#pragma once
#include <cstdint>

#include "../../core/isa.cuh"
#include "../../core/sync.cuh"

namespace mk {

// One setting in this schedule, so hardcoded: DeltaNet head dim
// (S_k == S_v == 128, ggml ssm_d_state) and the conv tap count
// (ssm_d_conv == 4).
constexpr int GDN_SV    = 128;
constexpr int GDN_DCONV = 4;

// Strong i64 load: state-row indices are host-written between passes.
__device__ __forceinline__ int64_t ld_cg_s64(const int64_t *p) {
    long long v;
    asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return (int64_t) v;
}

// The binary-tree __shfl_xor warp reduction closing every per-lane partial
// in this family (fork: warp_reduce_sum<32>, common.cuh:421-428).
__device__ __forceinline__ float gdn_warp_tree_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, offset, 32));
    }
    return x;
}

__device__ __forceinline__ float gdn_silu(float x) {
    return x / (1.0f + expf(-x));      // ggml_cuda_op_silu_single, unary.cuh:96-98
}

// ---------------------------------------------------------------------------
// OP_STATE_LOAD: dst[r * n_elems + i] = src[rows[r] * row_stride + i]
// (GET_ROWS over the f32 snapshot-ring cache; getrows.cu:43-72 semantics,
// index type widened to i64 per the k0 cache design.)
struct StateLoadArgs {
    const float   *src;         // state cache base
    const int64_t *rows;        // i64 row indices, n_rows of them
    float         *dst;         // working buffer, n_rows x n_elems contiguous
    int64_t        row_stride;  // f32 elements between cache rows
    int32_t        n_elems;     // row length (30720 conv, 786432 ssm)
    int32_t        n_rows;      // 1 at batch-1 decode
};
static_assert(sizeof(StateLoadArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_state_load(const Instr &ins, char *smem) {
    (void) smem;
    const StateLoadArgs &a = *reinterpret_cast<const StateLoadArgs *>(ins.payload);
    const int nthreads = (ins.block_hi - ins.block_lo) * blockDim.x;
    const int tid      = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (int32_t r = 0; r < a.n_rows; r++) {
        const float *src = a.src + ld_cg_s64(a.rows + r) * a.row_stride;
        float       *dst = a.dst + (int64_t) r * a.n_elems;
        if ((a.n_elems & 3) == 0 && ((((uintptr_t) src) | ((uintptr_t) dst)) & 15) == 0) {
            const int n4 = a.n_elems >> 2;
            for (int i = tid; i < n4; i += nthreads) {
                reinterpret_cast<float4 *>(dst)[i] = ld_cg(reinterpret_cast<const float4 *>(src) + i);
            }
        } else {
            for (int i = tid; i < a.n_elems; i += nthreads) {
                dst[i] = ld_cg(src + i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OP_CONV_SHIFT_CONCAT: build the (GDN_DCONV, channels) conv input window
// from the fetched history plus this token's qkv vector, and commit the
// shifted history (window elements 1..3 per channel) to the conv cache row.
// Matches build_conv_state, delta-net-base.cpp:457-513: conv_input =
// concat(conv_states(3,C), qkv^T(1,C), dim 0); last_conv_states = view of
// conv_input at +1 element -> cpy back. Layout is tap-contiguous per
// channel: hist[c*3 + j], win[c*4 + j].
struct ConvShiftConcatArgs {
    const float   *hist;        // (GDN_DCONV-1, channels) from OP_STATE_LOAD
    const float   *xnew;        // (channels) this token's qkv GEMV output
    float         *win;         // out: (GDN_DCONV, channels) conv window
    float         *state;       // conv state cache base (commit target)
    const int64_t *row;         // i64[1] destination cache row
    int64_t        row_stride;  // f32 elements between conv cache rows
    int32_t        channels;    // 10240 full model / 5120 per GPU half
};
static_assert(sizeof(ConvShiftConcatArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_conv_shift_concat(const Instr &ins, char *smem) {
    (void) smem;
    const ConvShiftConcatArgs &a = *reinterpret_cast<const ConvShiftConcatArgs *>(ins.payload);
    const int nthreads = (ins.block_hi - ins.block_lo) * blockDim.x;
    const int tid      = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    float *commit = a.state + ld_cg_s64(a.row) * a.row_stride;
    for (int c = tid; c < a.channels; c += nthreads) {
        const float h0 = ld_cg(a.hist + c * (GDN_DCONV - 1) + 0);
        const float h1 = ld_cg(a.hist + c * (GDN_DCONV - 1) + 1);
        const float h2 = ld_cg(a.hist + c * (GDN_DCONV - 1) + 2);
        const float x  = ld_cg(a.xnew + c);
        float *w = a.win + c * GDN_DCONV;
        w[0] = h0; w[1] = h1; w[2] = h2; w[3] = x;
        float *s = commit + c * (GDN_DCONV - 1);
        s[0] = h1; s[1] = h2; s[2] = x;
    }
}

// ---------------------------------------------------------------------------
// OP_SSM_CONV_SILU: y[c] = silu(sum_{j=0..3} win[c][j] * w[c][j]), the
// sequential tap order of ssm_conv_f32<1,128,4> (ssm-conv.cu:33-50) at
// n_t = 1. No bias (qwen35's conv has none). Window and weight are both
// tap-contiguous (4, channels) f32, one float4 per channel; the schedule
// guarantees 16-byte alignment for both.
struct SsmConvSiluArgs {
    const float *win;      // (GDN_DCONV, channels), mutable -> .cg
    const float *weight;   // (GDN_DCONV, channels) ssm_conv1d.weight, immutable
    float       *dst;      // (channels)
    int32_t      channels;
};
static_assert(sizeof(SsmConvSiluArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_ssm_conv_silu(const Instr &ins, char *smem) {
    (void) smem;
    const SsmConvSiluArgs &a = *reinterpret_cast<const SsmConvSiluArgs *>(ins.payload);
    const int nthreads = (ins.block_hi - ins.block_lo) * blockDim.x;
    const int tid      = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (int c = tid; c < a.channels; c += nthreads) {
        const float4 x = ld_cg(reinterpret_cast<const float4 *>(a.win) + c);
        const float4 w = reinterpret_cast<const float4 *>(a.weight)[c];
        float sum = 0.0f;
        sum = __fadd_rn(sum, __fmul_rn(x.x, w.x));
        sum = __fadd_rn(sum, __fmul_rn(x.y, w.y));
        sum = __fadd_rn(sum, __fmul_rn(x.z, w.z));
        sum = __fadd_rn(sum, __fmul_rn(x.w, w.w));
        a.dst[c] = gdn_silu(sum);
    }
}

// ---------------------------------------------------------------------------
// OP_QK_L2NORM: per head, dst = src * rsqrtf(max(sum(src^2), eps^2))
// (l2_norm_f32<32>, norm.cu:239-271, including the eps^2 floor). One warp
// per head; the lane sum order is the fork's block_size=32 stride (cols
// lane, lane+32, ..). In-place (dst == src) is safe: each head's values
// are register-cached by its owning warp before any write.
struct QkL2NormArgs {
    const float *src;      // (GDN_SV, n_heads)
    float       *dst;
    int32_t      n_heads;
    float        eps;      // 1e-6
};
static_assert(sizeof(QkL2NormArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_qk_l2norm(const Instr &ins, char *smem) {
    (void) smem;
    const QkL2NormArgs &a = *reinterpret_cast<const QkL2NormArgs *>(ins.payload);
    constexpr int rows_per_lane = GDN_SV / 32;
    const int warps_per_blk = blockDim.x >> 5;
    const int nwarps = (ins.block_hi - ins.block_lo) * warps_per_blk;
    const int wid    = (blockIdx.x - ins.block_lo) * warps_per_blk + (threadIdx.x >> 5);
    const int lane   = threadIdx.x & 31;
    for (int h = wid; h < a.n_heads; h += nwarps) {
        const float *x = a.src + (int64_t) h * GDN_SV;
        float       *y = a.dst + (int64_t) h * GDN_SV;
        float xr[rows_per_lane];
        float ss = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            xr[r] = ld_cg(x + r * 32 + lane);
            ss    = __fadd_rn(ss, __fmul_rn(xr[r], xr[r]));
        }
        ss = gdn_warp_tree_sum(ss);
        const float scale = rsqrtf(fmaxf(ss, a.eps * a.eps));
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            y[r * 32 + lane] = scale * xr[r];
        }
    }
}

// ---------------------------------------------------------------------------
// OP_GDN_GATES: per v-head h,
//   g[h]    = expf(softplus(alpha_raw[h] + dt_bias[h]) * a[h])
//   beta[h] = sigmoid(beta_raw[h])
// The graph chain ADD -> softplus -> MUL(ssm_a) (qwen35.cpp:375-382) plus
// the expf the fork's GDN kernel applies (gated_delta_net.cu:88), hoisted
// here so OP_GDN_STEP is transcendental-free (same bits: one expf of the
// same input either way). softplus per unary.cu:93-95 (x > 20 ? x :
// logf(1 + expf(x))); sigmoid per unary.cu:47-49.
struct GdnGatesArgs {
    const float *alpha_raw;  // (n_heads) ssm_alpha GEMV output
    const float *beta_raw;   // (n_heads) ssm_beta GEMV output
    const float *dt_bias;    // (n_heads) ssm_dt.bias, immutable
    const float *a;          // (n_heads) ssm_a (-exp(A_log)), immutable
    float       *g;          // out: per-head decay, already exp'd
    float       *beta;       // out: per-head mixing coeff
    int32_t      n_heads;
};
static_assert(sizeof(GdnGatesArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_gdn_gates(const Instr &ins, char *smem) {
    (void) smem;
    const GdnGatesArgs &a = *reinterpret_cast<const GdnGatesArgs *>(ins.payload);
    const int nthreads = (ins.block_hi - ins.block_lo) * blockDim.x;
    const int tid      = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (int h = tid; h < a.n_heads; h += nthreads) {
        const float ab = ld_cg(a.alpha_raw + h) + a.dt_bias[h];
        const float sp = (ab > 20.0f) ? ab : logf(1.0f + expf(ab));
        a.g[h]    = expf(sp * a.a[h]);
        a.beta[h] = 1.0f / (1.0f + expf(-ld_cg(a.beta_raw + h)));
    }
}

// ---------------------------------------------------------------------------
// OP_GDN_STEP: the delta-rule recurrence, gated_delta_net_cuda<128,0,0>
// (GDA, keep_rs=false), gated_delta_net.cu:66-167, n_tokens = 1. One warp
// per (head, column) task; state transposed, M[col][i] = S[i][col], row
// col contiguous. Per column:
//   kv    = sum_i S[i][col] * k[i]          (lane partials + tree reduce)
//   delta = (v[col] - g * kv) * beta
//   S[i][col] = g * S[i][col] + k[i] * delta        (two mults + one add)
//   attn[col] = (sum_i S'[i][col] * q[i]) * scale   (same loop; scale last)
// The v-head -> q/k-head map is MODULO: hk = h % n_k_heads (fork fastmodulo,
// gated_delta_net.cu:34, and ggml repeat semantics on the non-fused path).
// g is consumed pre-exp'd from OP_GDN_GATES. state_out may alias state_in
// (each column is register-held by its owning warp before write-back).
struct GdnStepArgs {
    const float *q;          // (GDN_SV, n_k_heads) l2-normalized
    const float *k;          // (GDN_SV, n_k_heads) l2-normalized
    const float *v;          // (GDN_SV, n_heads)
    const float *g;          // (n_heads) decay, exp'd
    const float *beta;       // (n_heads)
    const float *state_in;   // (GDN_SV*GDN_SV, n_heads) transposed
    float       *state_out;  // same layout; may alias state_in
    float       *attn_out;   // (GDN_SV, n_heads)
    int32_t      n_heads;    // H_v: 48 full model / 24 per GPU half
    int32_t      n_k_heads;  // H_k: 16 full model /  8 per GPU half
    float        scale;      // 1/sqrtf(GDN_SV), output scale, applied LAST
};
static_assert(sizeof(GdnStepArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_gdn_step(const Instr &ins, char *smem) {
    (void) smem;
    const GdnStepArgs &a = *reinterpret_cast<const GdnStepArgs *>(ins.payload);
    constexpr int rows_per_lane = GDN_SV / 32;
    const int warps_per_blk = blockDim.x >> 5;
    const int nwarps  = (ins.block_hi - ins.block_lo) * warps_per_blk;
    const int wid     = (blockIdx.x - ins.block_lo) * warps_per_blk + (threadIdx.x >> 5);
    const int lane    = threadIdx.x & 31;
    const int n_tasks = a.n_heads * GDN_SV;
    for (int task = wid; task < n_tasks; task += nwarps) {
        const int h   = task / GDN_SV;
        const int col = task - h * GDN_SV;
        const int hk  = h % a.n_k_heads;

        const float *q_t = a.q + (int64_t) hk * GDN_SV;
        const float *k_t = a.k + (int64_t) hk * GDN_SV;
        const float g_val    = ld_cg(a.g + h);
        const float beta_val = ld_cg(a.beta + h);
        const float v_col    = ld_cg(a.v + (int64_t) h * GDN_SV + col);
        const int64_t s_off  = ((int64_t) h * GDN_SV + col) * GDN_SV;

        float s_shard[rows_per_lane];
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * 32 + lane;
            s_shard[r]  = ld_cg(a.state_in + s_off + i);
            k_reg[r]    = ld_cg(k_t + i);
            q_reg[r]    = ld_cg(q_t + i);
        }

        // 1. kv[col] = sum_i S[i][col] * k[i]: per-lane sequential fp32
        //    partial over ascending r, then the tree reduce.
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard = __fadd_rn(kv_shard, __fmul_rn(s_shard[r], k_reg[r]));
        }
        const float kv_col = gdn_warp_tree_sum(kv_shard);

        // 2. delta[col] = (v[col] - g * kv[col]) * beta
        const float delta_col = __fmul_rn(__fsub_rn(v_col, __fmul_rn(g_val, kv_col)), beta_val);

        // 3+4. fused state update + output partial, one pass over the rows.
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r]   = __fadd_rn(__fmul_rn(g_val, s_shard[r]), __fmul_rn(k_reg[r], delta_col));
            attn_partial = __fadd_rn(attn_partial, __fmul_rn(s_shard[r], q_reg[r]));
        }
        const float attn_col = gdn_warp_tree_sum(attn_partial);

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            a.state_out[s_off + r * 32 + lane] = s_shard[r];
        }
        if (lane == 0) {
            // 5. output scaled by 1/sqrt(S_v) LAST.
            a.attn_out[(int64_t) h * GDN_SV + col] = __fmul_rn(attn_col, a.scale);
        }
    }
}

// ---------------------------------------------------------------------------
// OP_GATED_RMSNORM: per head, dst = (rsqrtf(sum(x^2)/128 + eps) * x) * w
// * silu(z) -- the fork's two-op sequence rms_norm_f32<256,1,0> (norm.cu:
// 74-151, do_multiply) followed by the fused unary_gated silu (unary.cu:
// 260-272), build_norm_gated (qwen35.cpp:248-258). One warp per head (the
// fork reduces with block_size 256; the cross-warp sum order differs, which
// is why this op is tolerance-checked, not bit-compared).
struct GatedRmsNormArgs {
    const float *x;      // (GDN_SV, n_heads) GDN attn output
    const float *w;      // (GDN_SV) ssm_norm.weight, immutable
    const float *z;      // (GDN_SV, n_heads) attn_gate GEMV output
    float       *dst;    // (GDN_SV, n_heads)
    int32_t      n_heads;
    float        eps;    // 1e-6
};
static_assert(sizeof(GatedRmsNormArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_gated_rmsnorm(const Instr &ins, char *smem) {
    (void) smem;
    const GatedRmsNormArgs &a = *reinterpret_cast<const GatedRmsNormArgs *>(ins.payload);
    constexpr int rows_per_lane = GDN_SV / 32;
    const int warps_per_blk = blockDim.x >> 5;
    const int nwarps = (ins.block_hi - ins.block_lo) * warps_per_blk;
    const int wid    = (blockIdx.x - ins.block_lo) * warps_per_blk + (threadIdx.x >> 5);
    const int lane   = threadIdx.x & 31;
    for (int h = wid; h < a.n_heads; h += nwarps) {
        const float *x = a.x + (int64_t) h * GDN_SV;
        const float *z = a.z + (int64_t) h * GDN_SV;
        float       *y = a.dst + (int64_t) h * GDN_SV;
        float xr[rows_per_lane];
        float ss = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            xr[r] = ld_cg(x + r * 32 + lane);
            ss    = __fadd_rn(ss, __fmul_rn(xr[r], xr[r]));
        }
        ss = gdn_warp_tree_sum(ss);
        const float scale = rsqrtf(ss / (float) GDN_SV + a.eps);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int col = r * 32 + lane;
            y[col] = scale * xr[r] * a.w[col] * gdn_silu(ld_cg(z + col));
        }
    }
}

// ---------------------------------------------------------------------------
// OP_STATE_STORE: dst[rows[r] * row_stride + i] = src[r * n_elems + i]
// (SET_ROWS by i64 row index, set-rows.cu:112-171 semantics, f32 -> f32).
struct StateStoreArgs {
    const float   *src;         // working buffer, n_rows x n_elems contiguous
    float         *dst;         // state cache base
    const int64_t *rows;        // i64 row indices, n_rows of them
    int64_t        row_stride;  // f32 elements between cache rows
    int32_t        n_elems;
    int32_t        n_rows;      // 1 at batch-1 decode
};
static_assert(sizeof(StateStoreArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_state_store(const Instr &ins, char *smem) {
    (void) smem;
    const StateStoreArgs &a = *reinterpret_cast<const StateStoreArgs *>(ins.payload);
    const int nthreads = (ins.block_hi - ins.block_lo) * blockDim.x;
    const int tid      = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (int32_t r = 0; r < a.n_rows; r++) {
        const float *src = a.src + (int64_t) r * a.n_elems;
        float       *dst = a.dst + ld_cg_s64(a.rows + r) * a.row_stride;
        if ((a.n_elems & 3) == 0 && ((((uintptr_t) src) | ((uintptr_t) dst)) & 15) == 0) {
            const int n4 = a.n_elems >> 2;
            for (int i = tid; i < n4; i += nthreads) {
                reinterpret_cast<float4 *>(dst)[i] = ld_cg(reinterpret_cast<const float4 *>(src) + i);
            }
        } else {
            for (int i = tid; i < a.n_elems; i += nthreads) {
                dst[i] = ld_cg(src + i);
            }
        }
    }
}

} // namespace mk
