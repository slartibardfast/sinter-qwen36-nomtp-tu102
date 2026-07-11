// The GEMV op family for the k=0 persistent megakernel: activation
// quantization (q8_1), the Q4_0 / Q4_0_AR16 MMVQ dot kernels (plain and
// SwiGLU-fused), and the F16 GEMVs (ssm_alpha/ssm_beta class and the
// lm_head vocab stream).
//
// Math contract: docs/SEMANTICS.md sections 1-4, verified against the fork
// (software/llama.cpp/autoround @ 546eca8dc):
//   - block_q4_0 (18 B, QK 32, split-half nibbles) ggml-common.h:187-192;
//     dot vec_dot_q4_0_q8_1 vecdotq.cuh:736-752 -> _impl :115-134
//     (return d4 * (sumi*d8 - (8*vdr/QI4_0)*s8), i.e. -8*s8 per 32-block).
//   - block_q4_0_ar16 (10 B, QK 16, element-interleaved nibbles, offset -8
//     folded in the unpack, NO s-correction term) ggml-common.h:194-199,
//     unpack_q4_0_ar16 + vec_dot_q4_0_ar16_q8_1 vecdotq.cuh:755-780.
//   - block_q8_1 (36 B: half2 ds = {d = amax/127, s = raw sum(x) over the
//     32 elements}, then 32 int8) ggml-common.h:257-269; producer
//     quantize_q8_1 quantize.cu:4-48 (q = amax==0 ? 0 : roundf(xi/d); the
//     activation buffer is sized/zero-padded to MATRIX_ROW_PADDING=512
//     elements, common.cuh:151, ggml-cuda.cu:1847).
//   - F16 GEMV mul_mat_vec_f mmvf.cu:7-374 (weights half, activations f32).
//   - fused SwiGLU epilogue: dst = (up.x) * silu(gate.x), silu(x) =
//     x / (1 + exp(-x)), mmvq.cu:572-575 + unary.cuh:96-98. Qwen3.6 FFNs
//     carry no bias (BLOCKS.md +59..+61), so the fusion bias paths
//     (mmvq.cu:442-479) are not implemented here.
//
// Load discipline (the G12 lesson, reference/tu102 bench/proj/dequant_gemv.cu):
//   - Weight payloads are IMMUTABLE, read with plain vectorized loads. The
//     quantized streams walk block PAIRS as u32 words (a lone 18 B / 10 B
//     block is only 2-aligned; the pair is 4-aligned) exactly like the bench
//     kernel that measured 97-99.6% of DRAM peak; 2-byte weight loads plus
//     scalar activation loads are the measured 70-75%-of-peak trap. The F16
//     streams use uint4 (8-half) loads.
//   - Activations (q8_1 blocks, f32 x) are MUTABLE data produced before the
//     previous Y02 boundary, so the strong-read rule applies (core/sync.cuh):
//     each CTA stages them ONCE into the smem slab with ld.cg word loads,
//     then the row loops read smem. This also keeps the .cg traffic from
//     amplifying L2 reads by 2-3x of the DRAM stream.
//
// Work partition (all six ops): warp-per-output-row (or warp-per-q8-block
// for the quantizer), flattened over the instruction's CTA range:
//   nwarps = (block_hi - block_lo) * (blockDim.x/32)
//   gw     = (blockIdx.x - block_lo) * (blockDim.x/32) + threadIdx.x/32
//   rows r = row_lo + gw, r += nwarps  (adjacent warps stream adjacent rows)
//
// smem slab budget (60 KiB shared with the interpreter; slab must be
// 16-byte aligned). Per-invocation staging use:
//   op_quant_q8_1        0 B
//   op_mmvq_q4_0         (K/32)*36 B   (K=5120: 5760 B; K=17408: 19584 B)
//   op_mmvq_q4_0_fused   (K/32)*36 B   (one shared y; K=5120: 5760 B)
//   op_mmvq_ar16         (K/32)*36 B   (K=6144: 6912 B)
//   op_gemv_f16          K*4 B         (K=5120: 20480 B)
//   op_head_gemv_f16     K*4 B         (K=5120: 20480 B)
// Every slab-using op closes with __syncthreads() so the next instruction
// may safely overwrite the slab.
//
// fp32 fold orders: G13 declares fold orders only for the GDN state fold and
// the cross-GPU exchange (core/isa.cuh header); the GEMV folds are therefore
// fixed by THIS file and documented per op. All scale folds are pinned with
// __fmul_rn/__fmaf_rn so the CPU reference in tests/test_gemv.cu reproduces
// them bit-exactly (no silent FMA contraction by the compiler).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

#include "../../core/isa.cuh"
#include "../../core/sync.cuh"

namespace mk {

// ---------------------------------------------------------------------------
// Args structs (payload <= 112 B, core/isa.cuh)
// ---------------------------------------------------------------------------

// OP_QUANT_Q8_1: f32 vector -> block_q8_1 stream. y holds ne0_padded/32
// blocks of 36 B; blocks past ne00 quantize zeros (d=0, s=0, q=0), exactly
// the fork's MATRIX_ROW_PADDING behaviour (quantize.cu:31, i0<ne00 guard).
struct QuantQ8_1Args {
    const float *x;       // f32 activation vector (mutable: read .cg)
    void        *y;       // block_q8_1 out, 4-byte aligned
    uint32_t ne00;        // real element count
    uint32_t ne0_padded;  // ne00 rounded up to MATRIX_ROW_PADDING (512)
};
static_assert(sizeof(QuantQ8_1Args) <= 112, "payload overflow");

// OP_MMVQ_Q4_0: dst[r] = sum_k dequant(w[r,k]) * x[k] via dp4a against q8_1.
// Rows r in [row_lo, row_hi) are this GPU's half under the meta split; w and
// dst are indexed by ABSOLUTE row (the uploader resolves the pointers so that
// absolute-row indexing lands in local buffers). Requires ncols % 64 == 0
// (block pairs; every scheduled K is a multiple of 512).
struct MmvqQ40Args {
    const void *w;        // block_q4_0 rows, row r at byte r*(ncols/32)*18
    const void *y;        // block_q8_1 activation blocks (mutable: staged .cg)
    float      *dst;      // f32 out, dst[r]
    uint32_t ncols;       // K, row length in elements
    uint32_t row_lo;
    uint32_t row_hi;
};
static_assert(sizeof(MmvqQ40Args) <= 112, "payload overflow");

// OP_MMVQ_Q4_0_FUSED: the up|gate SwiGLU fusion; two independent Q4_0 row
// streams over one shared activation, dst[r] = (up_r.x) * silu(gate_r.x)
// (mmvq.cu:498-507 twin accumulation, :572-575 epilogue).
struct MmvqQ40FusedArgs {
    const void *w_up;
    const void *w_gate;
    const void *y;
    float      *dst;
    uint32_t ncols;
    uint32_t row_lo;
    uint32_t row_hi;
};
static_assert(sizeof(MmvqQ40FusedArgs) <= 112, "payload overflow");

// OP_MMVQ_AR16: Q4_0_AR16 rows (QK 16, 10 B blocks, element-interleaved
// nibbles, offset folded, no s-term). Requires ncols % 32 == 0 (the fork's
// host dispatch guarantee, vecdotq.cuh:759-761).
struct MmvqAr16Args {
    const void *w;        // block_q4_0_ar16 rows, row r at byte r*(ncols/16)*10
    const void *y;
    float      *dst;
    uint32_t ncols;
    uint32_t row_lo;
    uint32_t row_hi;
};
static_assert(sizeof(MmvqAr16Args) <= 112, "payload overflow");

// OP_GEMV_F16: f16 rows x f32 vector, f32 accumulate. Requires ncols % 8 == 0
// (uint4 weight loads) and ncols*4 <= slab bytes.
struct GemvF16Args {
    const half  *w;       // f16 rows, row r at w + (size_t)r*ncols
    const float *x;       // f32 activation (mutable: staged .cg)
    float       *dst;     // f32 out, dst[r]
    uint32_t ncols;
    uint32_t row_lo;
    uint32_t row_hi;
};
static_assert(sizeof(GemvF16Args) <= 112, "payload overflow");

// OP_HEAD_GEMV_F16: the lm_head stream over this GPU's vocab rows
// (124160 x 5120 half). Same math and inner loop as OP_GEMV_F16 (the
// warp-per-row uint4 stream already runs at DRAM roofline on the bench's
// f16.k5120n248320 twin); a distinct kind so the schedule and G12 accounting
// see the vocab stream as its own op.
using HeadGemvF16Args = GemvF16Args;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int mk_dp4a(int a, int b, int c) {
    asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(c) : "r"(a), "r"(b));
    return c;
}

static __device__ __forceinline__ float half_lo(uint32_t v) {
    return __half2float(__ushort_as_half((unsigned short)(v & 0xFFFFu)));
}
static __device__ __forceinline__ float half_hi(uint32_t v) {
    return __half2float(__ushort_as_half((unsigned short)(v >> 16)));
}

// unpack_q4_0_ar16 (vecdotq.cuh:764-775): one u32 of interleaved nibbles ->
// two u32 of signed int8 in element order, offset -8 applied.
static __device__ __forceinline__ int2 mk_unpack_ar16(int q) {
    const int lo = (q >> 0) & 0x0F0F0F0F;
    const int hi = (q >> 4) & 0x0F0F0F0F;
    int2 v;
    v.x = __vsubss4(__byte_perm(lo, hi, 0x5140), 0x08080808);
    v.y = __vsubss4(__byte_perm(lo, hi, 0x7362), 0x08080808);
    return v;
}

// ggml_cuda_op_silu_single (unary.cuh:96-98).
static __device__ __forceinline__ float mk_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Stage nwords of a mutable global buffer into the smem slab with .cg word
// loads (strong-read rule). Whole CTA participates; closes with a barrier.
static __device__ __forceinline__ void stage_words_cg(
        uint32_t *smem_dst, const uint32_t *src, uint32_t nwords) {
    for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x)
        smem_dst[i] = ld_cg(src + i);
    __syncthreads();
}

// Butterfly reduce, the fork's warp_reduce_sum shape (common.cuh:421-428):
// offsets 16,8,4,2,1. All lanes end with the full sum.
static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off; off >>= 1)
        x += __shfl_xor_sync(0xFFFFFFFFu, x, off, 32);
    return x;
}

struct WarpSlice {
    int gw;      // global warp index within the instruction's CTA range
    int nwarps;  // total warps across the range
    int lane;
};
static __device__ __forceinline__ WarpSlice warp_slice(const Instr &I) {
    WarpSlice s;
    const int wpb = (int)(blockDim.x >> 5);
    s.nwarps = (int)(I.block_hi - I.block_lo) * wpb;
    s.gw     = (int)(blockIdx.x - I.block_lo) * wpb + (int)(threadIdx.x >> 5);
    s.lane   = (int)(threadIdx.x & 31);
    return s;
}

// ---------------------------------------------------------------------------
// OP_QUANT_Q8_1
// ---------------------------------------------------------------------------
// One warp per 32-element q8_1 block, lane j <-> element 32*b + j; blocks
// warp-strided over the CTA range. Replicates quantize_q8_1
// (quantize.cu:31-47) exactly: butterfly max/sum (offsets 16,8,4,2,1),
// d = amax/127 (fp32 divide), q = amax==0 ? 0 : roundf(xi/d),
// ds = half2(d, raw sum(x)). Blocks past ne00 quantize zeros (padding).
//
// Fold order: amax and sum are butterfly __shfl_xor reduces over the 32
// lanes, identical to the fork's warp_reduce_max/sum<32>.
// smem: 0 bytes.
static __device__ void op_quant_q8_1(const Instr &I, char *smem) {
    (void)smem;
    const QuantQ8_1Args &a = *reinterpret_cast<const QuantQ8_1Args *>(I.payload);
    const WarpSlice ws = warp_slice(I);
    const uint32_t nblk = a.ne0_padded / 32u;

    for (uint32_t b = (uint32_t)ws.gw; b < nblk; b += (uint32_t)ws.nwarps) {
        const uint32_t i = b * 32u + (uint32_t)ws.lane;
        const float xi = i < a.ne00 ? ld_cg(a.x + i) : 0.0f;
        float amax = fabsf(xi);
        float sum  = xi;
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 32));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off, 32);
        }
        const float d = amax / 127.0f;
        const int   q = amax == 0.0f ? 0 : (int)roundf(xi / d);

        int8_t *blk = reinterpret_cast<int8_t *>(a.y) + (size_t)b * 36u;
        blk[4 + ws.lane] = (int8_t)q;
        if (ws.lane == 0)
            *reinterpret_cast<half2 *>(blk) = __floats2half2_rn(d, sum);
    }
}

// ---------------------------------------------------------------------------
// Q4_0 row dot (shared by OP_MMVQ_Q4_0 and the fused variant)
// ---------------------------------------------------------------------------
// Per-lane partial over one row. Lane l consumes block PAIRS p = l, l+32, ...
// < npair_full (npair & ~31, so all 32 lanes stay balanced), each pair as 9
// plain u32 weight loads (36 B, 4-aligned since ncols % 64 == 0); the
// remaining blocks are walked one 2-byte-aligned block per lane (the bench's
// remainder scheme). q8_1 words come from the staged smem copy: block kb is
// words [9*kb, 9*kb+9), word 0 = ds, words 1..8 = qs.
//
// Fold order (fixed here; G13 is silent on GEMV folds): per lane, ascending
// pair index, within a pair block 2p then 2p+1, then the tail blocks
// ascending; each block folds as
//     acc = fmaf(d4, fmaf((float)sumi, d8, -8*s8), acc)
// with -8*s8 = __fmul_rn(-8, s8) (the whole-block form of vecdotq.cuh:133's
// -(8*vdr/QI4_0)*s8 correction); the 32 lane partials close with the
// butterfly warp_sum. Integer dp4a order inside a block is value-exact.
static __device__ __forceinline__ float mmvq_q40_row(
        const uint32_t *__restrict__ w_row, const uint32_t *ys,
        int npair, int npair_full, int nblk, int lane) {
    float acc = 0.0f;

    for (int p = lane; p < npair_full; p += 32) {
        const uint32_t *pw = w_row + (size_t)p * 9u;
        uint32_t wv[9];
#pragma unroll
        for (int i = 0; i < 9; i++) wv[i] = pw[i];
        const uint32_t *q0 = ys + (size_t)(2 * p) * 9u;
        const uint32_t *q1 = q0 + 9;

        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int v0 = (int)__funnelshift_r(wv[i], wv[i + 1], 16);
            s0 = mk_dp4a(v0 & 0x0F0F0F0F, (int)q0[1 + i], s0);
            s0 = mk_dp4a((v0 >> 4) & 0x0F0F0F0F, (int)q0[5 + i], s0);
            const int v1 = (int)wv[5 + i];
            s1 = mk_dp4a(v1 & 0x0F0F0F0F, (int)q1[1 + i], s1);
            s1 = mk_dp4a((v1 >> 4) & 0x0F0F0F0F, (int)q1[5 + i], s1);
        }
        acc = __fmaf_rn(half_lo(wv[0]),
                        __fmaf_rn((float)s0, half_lo(q0[0]),
                                  __fmul_rn(-8.0f, half_hi(q0[0]))), acc);
        acc = __fmaf_rn(half_hi(wv[4]),
                        __fmaf_rn((float)s1, half_lo(q1[0]),
                                  __fmul_rn(-8.0f, half_hi(q1[0]))), acc);
    }

    for (int kb = 2 * npair_full + lane; kb < nblk; kb += 32) {
        const uint16_t *q16 = reinterpret_cast<const uint16_t *>(w_row) + (size_t)kb * 9u;
        const uint32_t *qb = ys + (size_t)kb * 9u;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int v = (int)((uint32_t)q16[1 + 2 * i] | ((uint32_t)q16[2 + 2 * i] << 16));
            sumi = mk_dp4a(v & 0x0F0F0F0F, (int)qb[1 + i], sumi);
            sumi = mk_dp4a((v >> 4) & 0x0F0F0F0F, (int)qb[5 + i], sumi);
        }
        acc = __fmaf_rn(__half2float(__ushort_as_half(q16[0])),
                        __fmaf_rn((float)sumi, half_lo(qb[0]),
                                  __fmul_rn(-8.0f, half_hi(qb[0]))), acc);
    }
    return acc;
}

// OP_MMVQ_Q4_0. smem: (ncols/32)*36 B staged q8_1.
static __device__ void op_mmvq_q4_0(const Instr &I, char *smem) {
    const MmvqQ40Args &a = *reinterpret_cast<const MmvqQ40Args *>(I.payload);
    const int nblk = (int)(a.ncols / 32u);
    const int npair = nblk / 2;
    const int npair_full = npair & ~31;
    uint32_t *ys = reinterpret_cast<uint32_t *>(smem);
    stage_words_cg(ys, reinterpret_cast<const uint32_t *>(a.y), (uint32_t)nblk * 9u);

    const WarpSlice ws = warp_slice(I);
    const size_t row_words = (size_t)npair * 9u;   // 18 B/block as u32
    for (uint32_t r = a.row_lo + (uint32_t)ws.gw; r < a.row_hi; r += (uint32_t)ws.nwarps) {
        const uint32_t *w_row = reinterpret_cast<const uint32_t *>(a.w) + (size_t)r * row_words;
        const float acc = warp_sum(mmvq_q40_row(w_row, ys, npair, npair_full, nblk, ws.lane));
        if (ws.lane == 0) a.dst[r] = acc;
    }
    __syncthreads();   // slab handoff: no warp may still read ys after return
}

// OP_MMVQ_Q4_0_FUSED. Two sequential row dots (up then gate) over the one
// staged y, then the SwiGLU epilogue dst[r] = up * silu(gate)
// (mmvq.cu:572-575). Fold order per stream identical to OP_MMVQ_Q4_0.
// smem: (ncols/32)*36 B.
static __device__ void op_mmvq_q4_0_fused(const Instr &I, char *smem) {
    const MmvqQ40FusedArgs &a = *reinterpret_cast<const MmvqQ40FusedArgs *>(I.payload);
    const int nblk = (int)(a.ncols / 32u);
    const int npair = nblk / 2;
    const int npair_full = npair & ~31;
    uint32_t *ys = reinterpret_cast<uint32_t *>(smem);
    stage_words_cg(ys, reinterpret_cast<const uint32_t *>(a.y), (uint32_t)nblk * 9u);

    const WarpSlice ws = warp_slice(I);
    const size_t row_words = (size_t)npair * 9u;
    for (uint32_t r = a.row_lo + (uint32_t)ws.gw; r < a.row_hi; r += (uint32_t)ws.nwarps) {
        const uint32_t *up_row =
            reinterpret_cast<const uint32_t *>(a.w_up) + (size_t)r * row_words;
        const uint32_t *gate_row =
            reinterpret_cast<const uint32_t *>(a.w_gate) + (size_t)r * row_words;
        const float up   = warp_sum(mmvq_q40_row(up_row,   ys, npair, npair_full, nblk, ws.lane));
        const float gate = warp_sum(mmvq_q40_row(gate_row, ys, npair, npair_full, nblk, ws.lane));
        if (ws.lane == 0) a.dst[r] = __fmul_rn(up, mk_silu(gate));
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// OP_MMVQ_AR16
// ---------------------------------------------------------------------------
// Lane l consumes AR16 block PAIRS p (2 x 16 elements = one q8_1 block, 20 B
// = 5 plain u32 weight loads, 4-aligned since ncols % 32 == 0), pairs
// balanced to npair_full = npair & ~31, remainder one 10 B block per lane
// with 2-byte loads and the q8 half picked by kb parity (vecdotq.cuh:766).
//
// Fold order (fixed here): per lane, ascending pair index; per pair
//     acc = fmaf(d8, fmaf(d1,(float)s1, __fmul_rn(d0,(float)s0)), acc)
// (the bench grouping d8*(d0*s0 + d1*s1); same math as the fork's per-block
// d4*d8*sumi, different rounding grouping -- G13 is silent on GEMV folds);
// tail blocks fold as acc = fmaf(__fmul_rn(d4,d8), (float)sumi, acc); lanes
// close with the butterfly warp_sum. No s-correction term: the -8 offset is
// applied inside the unpack (vecdotq.cuh:778-779).
static __device__ __forceinline__ float mmvq_ar16_row(
        const uint32_t *__restrict__ w_row, const uint32_t *ys,
        int npair, int npair_full, int nblk, int lane) {
    float acc = 0.0f;

    for (int p = lane; p < npair_full; p += 32) {
        const uint32_t *pw = w_row + (size_t)p * 5u;
        uint32_t wv[5];
#pragma unroll
        for (int i = 0; i < 5; i++) wv[i] = pw[i];
        const uint32_t *q8 = ys + (size_t)p * 9u;

        int s0 = 0, s1 = 0;
        int2 v = mk_unpack_ar16((int)__funnelshift_r(wv[0], wv[1], 16));
        s0 = mk_dp4a(v.x, (int)q8[1], s0);
        s0 = mk_dp4a(v.y, (int)q8[2], s0);
        v = mk_unpack_ar16((int)__funnelshift_r(wv[1], wv[2], 16));
        s0 = mk_dp4a(v.x, (int)q8[3], s0);
        s0 = mk_dp4a(v.y, (int)q8[4], s0);
        v = mk_unpack_ar16((int)wv[3]);
        s1 = mk_dp4a(v.x, (int)q8[5], s1);
        s1 = mk_dp4a(v.y, (int)q8[6], s1);
        v = mk_unpack_ar16((int)wv[4]);
        s1 = mk_dp4a(v.x, (int)q8[7], s1);
        s1 = mk_dp4a(v.y, (int)q8[8], s1);

        const float t = __fmaf_rn(half_hi(wv[2]), (float)s1,
                                  __fmul_rn(half_lo(wv[0]), (float)s0));
        acc = __fmaf_rn(half_lo(q8[0]), t, acc);
    }

    for (int kb = 2 * npair_full + lane; kb < nblk; kb += 32) {
        const uint16_t *q16 = reinterpret_cast<const uint16_t *>(w_row) + (size_t)kb * 5u;
        const uint32_t *qb = ys + (size_t)(kb >> 1) * 9u;
        const int i8 = 1 + 4 * (kb & 1);
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int q = (int)((uint32_t)q16[1 + 2 * i] | ((uint32_t)q16[2 + 2 * i] << 16));
            const int2 v = mk_unpack_ar16(q);
            sumi = mk_dp4a(v.x, (int)qb[i8 + 2 * i + 0], sumi);
            sumi = mk_dp4a(v.y, (int)qb[i8 + 2 * i + 1], sumi);
        }
        acc = __fmaf_rn(__fmul_rn(__half2float(__ushort_as_half(q16[0])), half_lo(qb[0])),
                        (float)sumi, acc);
    }
    return acc;
}

// smem: (ncols/32)*36 B staged q8_1.
static __device__ void op_mmvq_ar16(const Instr &I, char *smem) {
    const MmvqAr16Args &a = *reinterpret_cast<const MmvqAr16Args *>(I.payload);
    const int nblk = (int)(a.ncols / 16u);
    const int npair = nblk / 2;
    const int npair_full = npair & ~31;
    uint32_t *ys = reinterpret_cast<uint32_t *>(smem);
    stage_words_cg(ys, reinterpret_cast<const uint32_t *>(a.y), (uint32_t)(a.ncols / 32u) * 9u);

    const WarpSlice ws = warp_slice(I);
    const size_t row_words = (size_t)npair * 5u;   // 10 B/block as u32
    for (uint32_t r = a.row_lo + (uint32_t)ws.gw; r < a.row_hi; r += (uint32_t)ws.nwarps) {
        const uint32_t *w_row = reinterpret_cast<const uint32_t *>(a.w) + (size_t)r * row_words;
        const float acc = warp_sum(mmvq_ar16_row(w_row, ys, npair, npair_full, nblk, ws.lane));
        if (ws.lane == 0) a.dst[r] = acc;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// OP_GEMV_F16 / OP_HEAD_GEMV_F16
// ---------------------------------------------------------------------------
// Lane l consumes uint4 groups g = l, l+32, ... (8 halfs each, plain
// vectorized weight loads); x is staged f32 in smem, read as float4 pairs.
//
// Fold order (fixed here): per lane, ascending group index, elements in
// order within the group, acc = fmaf(w_k, x_k, acc); lanes close with the
// butterfly warp_sum. This deviates from the fork's half2 accumulator
// (mmvf.cu:188-216, type_acc=half): fp32 accumulation is strictly more
// accurate and G13 does not bind on this op's fold.
static __device__ __forceinline__ float gemv_f16_row(
        const uint4 *__restrict__ w_row, const float4 *xs, int ngrp, int lane) {
    float acc = 0.0f;
    for (int g = lane; g < ngrp; g += 32) {
        const uint4 pk = w_row[g];
        const half2 *h2 = reinterpret_cast<const half2 *>(&pk);
        const float4 xa = xs[2 * g + 0];
        const float4 xb = xs[2 * g + 1];
        float2 f;
        f = __half22float2(h2[0]);
        acc = __fmaf_rn(f.x, xa.x, acc); acc = __fmaf_rn(f.y, xa.y, acc);
        f = __half22float2(h2[1]);
        acc = __fmaf_rn(f.x, xa.z, acc); acc = __fmaf_rn(f.y, xa.w, acc);
        f = __half22float2(h2[2]);
        acc = __fmaf_rn(f.x, xb.x, acc); acc = __fmaf_rn(f.y, xb.y, acc);
        f = __half22float2(h2[3]);
        acc = __fmaf_rn(f.x, xb.z, acc); acc = __fmaf_rn(f.y, xb.w, acc);
    }
    return acc;
}

static __device__ __forceinline__ void gemv_f16_common(const Instr &I, char *smem) {
    const GemvF16Args &a = *reinterpret_cast<const GemvF16Args *>(I.payload);
    float *xs = reinterpret_cast<float *>(smem);
    stage_words_cg(reinterpret_cast<uint32_t *>(xs),
                   reinterpret_cast<const uint32_t *>(a.x), a.ncols);

    const WarpSlice ws = warp_slice(I);
    const int ngrp = (int)(a.ncols / 8u);
    for (uint32_t r = a.row_lo + (uint32_t)ws.gw; r < a.row_hi; r += (uint32_t)ws.nwarps) {
        const uint4 *w_row =
            reinterpret_cast<const uint4 *>(a.w + (size_t)r * a.ncols);
        const float acc = warp_sum(
            gemv_f16_row(w_row, reinterpret_cast<const float4 *>(xs), ngrp, ws.lane));
        if (ws.lane == 0) a.dst[r] = acc;
    }
    __syncthreads();
}

// smem: ncols*4 B staged x.
static __device__ void op_gemv_f16(const Instr &I, char *smem) {
    gemv_f16_common(I, smem);
}

// smem: ncols*4 B staged x. Same inner loop as op_gemv_f16 (see the
// HeadGemvF16Args note); the big-row stream needs no extra tuning: the
// warp-per-row uint4 pattern is the bench's 97-99.6%-of-peak f16 twin.
static __device__ void op_head_gemv_f16(const Instr &I, char *smem) {
    gemv_f16_common(I, smem);
}

} // namespace mk
