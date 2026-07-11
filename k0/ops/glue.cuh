// Glue ops: OP_RMSNORM, OP_RESIDUAL_ADD, OP_EMBED_LOOKUP, OP_LOGITS_EMIT.
//
// Every op follows the ISA conventions (core/isa.cuh): signature
// op_<kind>(const Instr&, char *smem), work derived from
// (blockIdx.x - block_lo), (block_hi - block_lo), threadIdx.x; payload cast
// to the op's Args struct; mutable data read strong (.cg), immutable weights
// read plain. Ops own the whole smem slab while they run; the interpreter
// only touches it between ops.
//
// Alignment: the vector paths engage when the pointers are 16-byte aligned
// and the width divides; otherwise the op falls back to a scalar loop, so a
// schedule with offset views stays correct (just slower).
#pragma once
#include <cuda_fp16.h>
#include "../../core/isa.cuh"
#include "../../core/sync.cuh"
#include "../../core/interp.cuh"

namespace mk {

__device__ __forceinline__ bool aligned16(const void *p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// OP_RMSNORM: y = x * rsqrt(mean(x^2) + eps) * w, eps 1e-6, fused weight
// multiply, matching the fork's rms_norm_f32 semantics (SEMANTICS.md #9:
// mean = sum(x^2)/ncols in fp32, scale = rsqrtf(mean + eps), then the fused
// mul). Fold order here: each thread accumulates its strided elements
// sequentially in fp32, then a __shfl_xor tree per warp, then a tree over
// the 12 warp partials. Optional add-src: the normed input is (x + add);
// the sum itself is not published (schedule a RESIDUAL_ADD when the sum is
// needed downstream). Rows are strided over the participating blocks; the
// trunk norms (attn_norm / post_attention_norm / output_norm) are 5120x1,
// nrows = 1, one block.
// ---------------------------------------------------------------------------
constexpr float RMSNORM_EPS = 1e-6f;

struct RmsnormArgs {
    const float *x;    // input row(s), mutable -> strong reads
    const float *w;    // norm weight [ncols], immutable -> plain reads
    const float *add;  // optional pre-norm residual add, nullptr if none
    float *y;          // output row(s); y == x (in place) is allowed
    uint32_t ncols;
    uint32_t nrows;
};
static_assert(sizeof(RmsnormArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_rmsnorm(const Instr &in, char *smem) {
    const RmsnormArgs &a = *reinterpret_cast<const RmsnormArgs *>(in.payload);
    const unsigned nblk = in.block_hi - in.block_lo;
    float *red = reinterpret_cast<float *>(smem); // one partial per warp

    for (uint32_t row = blockIdx.x - in.block_lo; row < a.nrows; row += nblk) {
        const float *x = a.x + (size_t)row * a.ncols;
        const float *add = a.add ? a.add + (size_t)row * a.ncols : nullptr;
        float *y = a.y + (size_t)row * a.ncols;

        float acc = 0.0f;
        const bool vec = (a.ncols % 4u == 0) && aligned16(x) &&
                         (!add || aligned16(add));
        if (vec) {
            const float4 *x4 = reinterpret_cast<const float4 *>(x);
            const float4 *a4 = reinterpret_cast<const float4 *>(add);
            for (uint32_t i = threadIdx.x; i < a.ncols / 4; i += blockDim.x) {
                float4 v = ld_cg(x4 + i);
                if (add) {
                    const float4 r = ld_cg(a4 + i);
                    v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
                }
                acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (uint32_t i = threadIdx.x; i < a.ncols; i += blockDim.x) {
                float v = ld_cg(x + i);
                if (add) v += ld_cg(add + i);
                acc += v * v;
            }
        }

        acc = warp_sum(acc);
        const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = lane < (blockDim.x >> 5) ? red[lane] : 0.0f;
            v = warp_sum(v);
            if (lane == 0)
                red[0] = rsqrtf(v / (float)a.ncols + RMSNORM_EPS);
        }
        __syncthreads();
        const float scale = red[0];
        __syncthreads(); // red[] is reused by the next row

        for (uint32_t i = threadIdx.x; i < a.ncols; i += blockDim.x) {
            float v = ld_cg(x + i);
            if (add) v += ld_cg(add + i);
            y[i] = scale * v * a.w[i];
        }
    }
}

// ---------------------------------------------------------------------------
// OP_RESIDUAL_ADD: y = a + b, f32, n elements interleave-strided over the
// participating blocks.
// ---------------------------------------------------------------------------
struct ResidualAddArgs {
    const float *a; // mutable -> strong reads
    const float *b; // mutable -> strong reads
    float *y;
    uint32_t n;
};
static_assert(sizeof(ResidualAddArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_residual_add(const Instr &in, char *) {
    const ResidualAddArgs &a =
        *reinterpret_cast<const ResidualAddArgs *>(in.payload);
    const unsigned nblk = in.block_hi - in.block_lo;
    const unsigned tid = (blockIdx.x - in.block_lo) * blockDim.x + threadIdx.x;
    const unsigned stride = nblk * blockDim.x;

    if ((a.n % 4u == 0) && aligned16(a.a) && aligned16(a.b) && aligned16(a.y)) {
        const float4 *a4 = reinterpret_cast<const float4 *>(a.a);
        const float4 *b4 = reinterpret_cast<const float4 *>(a.b);
        float4 *y4 = reinterpret_cast<float4 *>(a.y);
        for (uint32_t i = tid; i < a.n / 4; i += stride) {
            const float4 u = ld_cg(a4 + i), v = ld_cg(b4 + i);
            y4[i] = make_float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
        }
    } else {
        for (uint32_t i = tid; i < a.n; i += stride)
            a.y[i] = ld_cg(a.a + i) + ld_cg(a.b + i);
    }
}

// ---------------------------------------------------------------------------
// OP_EMBED_LOOKUP: f16 embedding row -> f32 residual root. The row index is
// an i32 token cell written by the host before the pass doorbell (mutable
// -> one strong read, block-broadcast through the slab). The embedding
// matrix itself is an immutable weight -> plain loads.
// ---------------------------------------------------------------------------
struct EmbedLookupArgs {
    const __half *emb;    // token_embd.weight base
    const int32_t *token; // i32 token-id cell (device, host-written per pass)
    float *y;             // f32 out row [ncols]
    uint32_t ncols;       // 5120
    uint32_t row_stride;  // elements between consecutive rows (5120)
};
static_assert(sizeof(EmbedLookupArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_embed_lookup(const Instr &in, char *smem) {
    const EmbedLookupArgs &a =
        *reinterpret_cast<const EmbedLookupArgs *>(in.payload);
    int *bc = reinterpret_cast<int *>(smem);
    if (threadIdx.x == 0)
        *bc = (int)ld_cg(reinterpret_cast<const unsigned *>(a.token));
    __syncthreads();
    const int tok = *bc;
    __syncthreads();

    const __half *row = a.emb + (size_t)tok * a.row_stride;
    const unsigned nblk = in.block_hi - in.block_lo;
    const unsigned tid = (blockIdx.x - in.block_lo) * blockDim.x + threadIdx.x;
    const unsigned stride = nblk * blockDim.x;

    if ((a.ncols % 2u == 0) && ((reinterpret_cast<uintptr_t>(row) & 3u) == 0) &&
        ((reinterpret_cast<uintptr_t>(a.y) & 7u) == 0)) {
        const __half2 *row2 = reinterpret_cast<const __half2 *>(row);
        float2 *y2 = reinterpret_cast<float2 *>(a.y);
        for (uint32_t i = tid; i < a.ncols / 2; i += stride)
            y2[i] = __half22float2(row2[i]);
    } else {
        for (uint32_t i = tid; i < a.ncols; i += stride)
            a.y[i] = __half2float(row[i]);
    }
}

// ---------------------------------------------------------------------------
// OP_LOGITS_EMIT: copy this GPU's logits half to the output buffer, then
// st.release the completion flag the host polls. The flag store must order
// after EVERY participating block's copies, and ops have no cross-block
// barrier of their own, so a non-null flag requires a single-block range
// (block_hi - block_lo == 1); a wider range raises
// ERR_LOGITS_FLAG_MULTIBLOCK (copy still done, flag skipped). Schedules
// that want a multi-block copy put the flag emit in its own 1-block
// instruction after a BOUNDARY, or lean on the interpreter's pass-level
// done cell, which is always released after the epilogue boundary.
// ---------------------------------------------------------------------------
struct LogitsEmitArgs {
    const float *src;    // logits produced this pass, mutable -> strong reads
    float *dst;          // result_output destination (device or host-mapped)
    uint32_t n;
    uint32_t flag_value; // value released to *flag
    unsigned *flag;      // host-polled completion flag; nullptr = none
};
static_assert(sizeof(LogitsEmitArgs) <= sizeof(Instr::payload), "payload");

__device__ inline void op_logits_emit(const Instr &in, char *) {
    const LogitsEmitArgs &a =
        *reinterpret_cast<const LogitsEmitArgs *>(in.payload);
    const unsigned nblk = in.block_hi - in.block_lo;
    const unsigned tid = (blockIdx.x - in.block_lo) * blockDim.x + threadIdx.x;
    const unsigned stride = nblk * blockDim.x;

    if ((a.n % 4u == 0) && aligned16(a.src) && aligned16(a.dst)) {
        const float4 *s4 = reinterpret_cast<const float4 *>(a.src);
        float4 *d4 = reinterpret_cast<float4 *>(a.dst);
        for (uint32_t i = tid; i < a.n / 4; i += stride)
            d4[i] = ld_cg(s4 + i);
    } else {
        for (uint32_t i = tid; i < a.n; i += stride)
            a.dst[i] = ld_cg(a.src + i);
    }

    if (a.flag != nullptr) {
        if (nblk != 1u) {
            if (threadIdx.x == 0)
                raise_error(ERR_LOGITS_FLAG_MULTIBLOCK, in.dbg_node);
            return;
        }
        __syncthreads(); // whole block's copies issued before the release
        if (threadIdx.x == 0)
            st_release_sys(a.flag, a.flag_value);
    }
}

} // namespace mk
