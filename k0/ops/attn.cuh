// The attention op family for the k=0 persistent megakernel: the 16
// attention blocks (graph indices i%4==3) of the Qwen3.6-27B hybrid stack.
//
// Semantics are replicated from the fork (software/llama.cpp/autoround
// @546eca8dc), cites are path:line into that tree:
//   - per-head RMS norm:      ggml/src/ggml-cuda/norm.cu:74-151 (fused xweight)
//   - IMROPE:                 ggml/src/ggml-cuda/rope.cu:182-265 (rope_multi,
//                             is_imrope sector chain :231-240, NEOX split-half
//                             :260-264; rope_yarn :22-41 degenerates to plain
//                             cos/sin at ext_factor=0, freq_scale=1,
//                             attn_factor=1, no freq_factors)
//   - KV append (SET_ROWS):   ggml/src/ggml-cuda/set-rows.cu:112-171
//                             (f32 -> f16 RN cast at an i64 row index)
//   - flash attention:        ggml/src/ggml-cuda/fattn-mma-f16.cuh (Q
//                             pre-scaled :1230, mask added to KQ, f32 online
//                             softmax, KQ_max init -FLT_MAX/2 :1196)
//   - split-KV merge (Y03):   ggml/src/ggml-cuda/fattn-common.cuh:719-752
//                             (max/LSE-stable combine, SOFTMAX_FTZ_THRESHOLD
//                             -20 :11, final divide by merged rowsum)
//   - output gate:            attn * sigmoid(gate), op_sigmoid
//                             ggml/src/ggml-cuda/unary.cu:48-50
//
// GQA note (SCHEDULE-QUESTIONS.md item 18): the census template
// flash_attn_ext_f16<256,256,1,8,0,0> carries ncols2=8, a power-of-two
// tile-width BUCKET chosen for any gqa_ratio > 4 (fattn.cu:92-95); the true
// measured ratio is 24/4 = 6 (per GPU: 12 q-heads / 2 kv-heads). This
// implementation packs no pad slots: gqa = n_q / n_kv_heads.
//
// Model constants with exactly one setting are hardcoded (head_dim 256,
// eps 1e-6, FA scale 0.0625 = 1/sqrt(256), the IMROPE parameters); per-GPU
// extents (head counts, KV row width, n_kv) come from the instruction args.
//
// Declared fp32 fold orders (the G13 discipline):
//   - RMS sumsq: per-lane sequential over 8 stride-32 elements, then a
//     binary-tree shfl_xor warp reduce.
//   - FATTN_DECODE: per (head, chunk), tiles of 32 KV positions ascending;
//     within a tile, scores reduced by shfl_xor tree; V accumulated per lane
//     over t = 0..31 ascending; online rescale once per tile.
//   - FATTN_REDUCE: chunks merged ascending c = 0..n_chunks-1 with the
//     fixup's max/LSE math (the fork's fixup walks descending from the last
//     partial, fattn-common.cuh:732; both orders are max/LSE-stable, ours is
//     the declared order).
#pragma once
#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>
#include "../../core/isa.cuh"
#include "../../core/sync.cuh"

namespace mk {

// ---------------------------------------------------------------- constants

// head_dim for q/k/v (the graph's 256; FA scale is 1/sqrt of this).
constexpr int MK_ATTN_HD = 256;
// FA scale, graph constant scale=0.0625 (BLOCKS.md, all 16 FLASH_ATTN_EXT).
constexpr float MK_ATTN_SCALE = 0.0625f;
// RMS_NORM eps (graph constant 1e-06 on all 209 RMS_NORMs).
constexpr float MK_ATTN_EPS = 1e-6f;
// IMROPE: n_dims=64 rotated of 256, sections [11,11,10,0], freq_base 1e7.
// theta_scale = powf(freq_base, -2/n_dims) (rope.cu:443) = 1e7^(-1/32),
// correctly rounded to f32:
constexpr float MK_IMROPE_THETA_SCALE = 0x1.356656p-1f; // 0.60429638624191284
constexpr int MK_IMROPE_NDIMS = 64;
// Split-KV partial record: [0..255] f32 V accumulator, [256] running max,
// [257] running sumexp, [258..259] pad (16B record alignment).
constexpr int MK_FATTN_PSTRIDE = 260;
// KV positions per smem tile in OP_FATTN_DECODE.
constexpr int MK_FATTN_TILE = 32;
// The Y03 sentinel: partial buffers are seeded with -inf by the uploader
// before the pass. OP_FATTN_DECODE can never produce max == -inf (the online
// max starts at -FLT_MAX/2 and only grows, matching the fork's KQ_max init,
// fattn-mma-f16.cuh:1196; a fully-masked chunk stores -FLT_MAX/2), so a
// record whose max reads -inf was never written and OP_FATTN_REDUCE counts
// it into the error flag instead of folding pre-store garbage into the LSE
// (sync_protocol.csv row Y03 failure mode).
// MK_FATTN_SENTINEL == -INFINITY, spelled as a constant expression:
#define MK_FATTN_SENTINEL (-__builtin_huge_valf())

// ------------------------------------------------------------- local helpers

__device__ __forceinline__ float warp_sum_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_max_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Strong 64-bit read (the SET_ROWS row index is mutable host-uploaded data
// read across a boundary; same rule as the .cg family in core/sync.cuh).
__device__ __forceinline__ long long ld_cg_i64(const long long *p) {
    long long v;
    asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

#ifdef MK_FATTN_HMMA
// plan/0143 FATTN primary: tensor-core flash decode (spikes ae74152/a90f67f).
// m16n8k8 f32-accumulate HMMA + quad (width-4) softmax reductions. Fragment map
// (lane l: gid=l>>2, tid=l&3) A[m=gid][k=2tid], B[n=gid][k=2tid], D[m=gid][n=2tid].
__device__ __forceinline__ void mk_hmma_f32(float &d0, float &d1, float &d2,
                                            float &d3, unsigned a0, unsigned a1,
                                            unsigned b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ unsigned mk_pk(half lo, half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<unsigned *>(&h);
}
__device__ __forceinline__ float mk_qmax(float v) { // quad max, width 4
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1, 4));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2, 4));
}
__device__ __forceinline__ float mk_qsum(float v) { // quad sum, width 4
    v += __shfl_xor_sync(0xffffffffu, v, 1, 4);
    return v + __shfl_xor_sync(0xffffffffu, v, 2, 4);
}
#endif

// ------------------------------------------------------------ OP_QK_NORM_ROPE
//
// Per-head RMS norm (fused norm-weight multiply) then IMROPE on the first 64
// dims. One instruction per tensor: q (n_heads local q heads, src_stride 512
// f32, the [q 256 | gate 256] interleave of the attn_q GEMV) and k (n_heads
// local kv heads, src_stride 256). dst is contiguous [n_heads][256] f32.

struct QkNormRopeArgs {
    const float   *src;        // head h at src + h*src_stride (strong reads)
    const float   *norm_w;     // f32[256] per-head norm weight (immutable)
    const int32_t *pos;        // i32[4]: the 4 M-RoPE position ids (strong)
    float         *dst;        // contiguous [n_heads][MK_ATTN_HD]
    uint32_t       n_heads;
    uint32_t       src_stride; // f32 elems between heads: 512 (q) / 256 (k)
};
static_assert(sizeof(QkNormRopeArgs) <= 112, "payload overflow");

__device__ MK_OPFN void op_qk_norm_rope(const Instr &ins, char *) {
    const QkNormRopeArgs a = *reinterpret_cast<const QkNormRopeArgs *>(ins.payload);
    const int warps_per_block = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int gw   = (blockIdx.x - ins.block_lo) * warps_per_block + threadIdx.x / 32;
    const int gw_n = (ins.block_hi - ins.block_lo) * warps_per_block;

    // Positions are per-token, shared by every head: compute the lane's
    // cos/sin once. Lane l owns elements {l, l+32, l+64, ..., l+224}; the
    // NEOX split-half pair for rotary pair p = l is (elem l, elem l+32) =
    // (v[0], v[1]), so no cross-lane exchange is needed.
    const int p0 = (int) ld_cg(reinterpret_cast<const unsigned *>(a.pos) + 0);
    const int p1 = (int) ld_cg(reinterpret_cast<const unsigned *>(a.pos) + 1);
    const int p2 = (int) ld_cg(reinterpret_cast<const unsigned *>(a.pos) + 2);
    const int p3 = (int) ld_cg(reinterpret_cast<const unsigned *>(a.pos) + 3);

    // is_imrope sector chain, rope.cu:231-240, evaluated in source order
    // (h, w, t, else e). sect_dims = 11+11+10+0 = 32 = n_dims/2, so
    // sector == pair index == lane. 3*sections = {33, 33, 30}.
    const int sector = lane;
    int pos_sel;
    if (sector % 3 == 1 && sector < 3 * 11) {
        pos_sel = p1; // h
    } else if (sector % 3 == 2 && sector < 3 * 10) {
        pos_sel = p2; // w
    } else if (sector % 3 == 0 && sector < 3 * 11) {
        pos_sel = p0; // t
    } else {
        pos_sel = p3; // e (never taken with sections [11,11,10,0])
    }
    // theta_base = pos * theta_scale^p (rope.cu:233-239); rope_yarn with
    // ext_factor=0, freq_scale=1, attn_factor=1 is plain cos/sin (:22-41).
    const float theta = (float) pos_sel * powf(MK_IMROPE_THETA_SCALE, (float) sector);
    const float cos_t = cosf(theta);
    const float sin_t = sinf(theta);

    for (uint32_t h = gw; h < a.n_heads; h += gw_n) {
        const float *x = a.src + (size_t) h * a.src_stride;
        float v[MK_ATTN_HD / 32];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < MK_ATTN_HD / 32; i++) {
            v[i] = ld_cg(x + i * 32 + lane);
            ss += v[i] * v[i];
        }
        ss = warp_sum_f32(ss);
        // norm.cu:136-146: scale = rsqrtf(sum/ncols + eps); dst = scale*x*w.
        const float scale = rsqrtf(ss / (float) MK_ATTN_HD + MK_ATTN_EPS);
#pragma unroll
        for (int i = 0; i < MK_ATTN_HD / 32; i++) {
            v[i] = scale * v[i] * a.norm_w[i * 32 + lane];
        }
        // NEOX split-half rotation on dims [0, 64): rope.cu:260-264.
        const float x0 = v[0], x1 = v[1];
        v[0] = x0 * cos_t - x1 * sin_t;
        v[1] = x0 * sin_t + x1 * cos_t;
        // dims >= n_dims pass through (rope.cu:219-224).
        float *d = a.dst + (size_t) h * MK_ATTN_HD;
#pragma unroll
        for (int i = 0; i < MK_ATTN_HD / 32; i++) {
            d[i * 32 + lane] = v[i];
        }
    }
}

// --------------------------------------------------------------- OP_KV_APPEND
//
// SET_ROWS: one f32 row (this GPU's kv heads' halves, contiguous) -> f16
// into the KV cache at the i64 row index carried as tensor DATA
// (set-rows.cu:160-165; straight f32 -> f16 RN cast). One instruction per
// cache (K from the roped-k dst, V from the v GEMV output).

struct KvAppendArgs {
    const float     *src;      // f32[row_width] (strong reads)
    const long long *row_idx;  // i64[1] destination row (strong read)
    half            *cache;    // rows of row_width f16, dense
    uint32_t         row_width;// f16 elems per row (local: 512)
};
static_assert(sizeof(KvAppendArgs) <= 112, "payload overflow");

__device__ MK_OPFN void op_kv_append(const Instr &ins, char *) {
    const KvAppendArgs a = *reinterpret_cast<const KvAppendArgs *>(ins.payload);
    const long long row = ld_cg_i64(a.row_idx);
    half *d = a.cache + (size_t) row * a.row_width;
    const uint32_t nthr = (ins.block_hi - ins.block_lo) * blockDim.x;
    const uint32_t t0   = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (uint32_t i4 = t0; i4 < a.row_width / 4; i4 += nthr) {
        const float4 v = ld_cg(reinterpret_cast<const float4 *>(a.src) + i4);
        half2 *d2 = reinterpret_cast<half2 *>(d) + 2 * i4;
        d2[0] = __floats2half2_rn(v.x, v.y);
        d2[1] = __floats2half2_rn(v.z, v.w);
    }
}

// ------------------------------------------------------------ OP_FATTN_DECODE
//
// Batch-1 split-KV decode attention over this GPU's kv heads. Each
// participating block owns one contiguous KV chunk (chunk index =
// blockIdx.x - block_lo, n_chunks = block_hi - block_lo) and processes it
// for ALL local q heads, producing one partial record per (head, chunk).
//
// The kv heads are the OUTER loop: for each kv head, its gqa = n_q/n_kv_heads
// q heads are carried one-warp-per-head (gqa <= blockDim.x/32) while the smem
// tile holds only that kv head's 256-wide slice of the cache row. This bounds
// the slab independent of the physical row_width, so the single-GPU shape
// (24 q / 4 kv, row_width 1024) fits where a full-row tile would be 65 KiB;
// each slice is a disjoint 256 columns of the row, so the four passes read the
// whole KV once (no extra DRAM). KV positions run in smem tiles of 32 rows:
// phase K loads the K slice (16B strong loads, the G12 discipline) and scores
// 32 positions (lane = position); phase V reuses the smem for the V slice and
// accumulates.
//
// smem (within the 60 KiB slab): q_s n_q*256 f32 (all heads staged once), then
// a 32-row tile of (256 + 2) f16 (the +2 pad makes lane-per-row half2 reads
// bank-conflict-free), then 32 f16 of mask. Single-GPU (n_q=24): 41,152 B;
// per-GPU (n_q=12): 28,864 B. Independent of row_width.
//
// The mask is the padded f32->f16 KQ mask: 0 attend / -inf masked, columns
// [real n_kv, padded n_kv) are -inf. Padded-class cache rows are never
// weighted (p = 0) but ARE loaded; like the fork's FA, this assumes they
// hold finite f16 (an all-NaN uninitialized row would poison the 0*x
// product only through the score path, which the -inf mask kills before the
// max; the V path multiplies by p = 0 only when p != 0 is false, i.e. it is
// skipped entirely).

struct FattnDecodeArgs {
    const float    *q;         // [n_q][256] roped q, unscaled (strong reads)
    const half     *k_cache;   // dense rows of row_width f16 (strong reads)
    const half     *v_cache;   // dense rows of row_width f16 (strong reads)
    const half     *mask;      // [n_kv] f16, 0 / -inf (strong reads)
    float          *partials;  // [n_q][n_chunks][MK_FATTN_PSTRIDE]
    const uint32_t *n_kv_cell;  // device cell: padded KV length (multiple of 256),
                                // host-written once per pass, read STRONG (.cg).
                                // The payload pointer is immutable (pack-time);
                                // only the CELL varies per pass — like `mask`.
    uint32_t        n_q;       // local q heads (12)
    uint32_t        n_kv_heads;// local kv heads (2); gqa = n_q / n_kv_heads
    uint32_t        row_width; // f16 elems per cache row (512)
};
static_assert(sizeof(FattnDecodeArgs) <= 112, "payload overflow");

// Cooperative tile load: the MK_ATTN_HD-wide slice of one kv head (base offset
// `head_off` inside each cache row, stride `row_width`) for rows [t0, t0+32)
// into the padded smem tile; rows >= limit are zero-filled (finite, so a
// skipped V contribution can never read NaN). The tile pitch `row_p` is
// MK_ATTN_HD+2 f16 (bank-conflict-free half2 rows), independent of the cache's
// full row_width — one kv head's 256 dims fit the slab even when the physical
// cache row packs all kv heads (single-GPU: row_width=1024, 4 heads).
__device__ inline void mk_load_kv_slice(const half *cache, half *tile, uint32_t t0,
                                        uint32_t limit, uint32_t row_width,
                                        uint32_t head_off, uint32_t row_p) {
    const uint32_t nv4 = MK_ATTN_HD / 8; // uint4 (8 f16) per 256-wide slice = 32
    for (uint32_t i = threadIdx.x; i < MK_FATTN_TILE * nv4; i += blockDim.x) {
        const uint32_t t = i / nv4, c = i % nv4;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (t0 + t < limit) {
            v = ld_cg(reinterpret_cast<const uint4 *>(
                          cache + (size_t)(t0 + t) * row_width + head_off) + c);
        }
        unsigned *d = reinterpret_cast<unsigned *>(tile + t * row_p + c * 8);
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
}

// Contiguous full-row cooperative load: the WHOLE cache row (all local kv heads,
// `row_width` f16) for rows [t0,t0+32) into the tile at pitch `rowp`. Coalesced
// (no per-head stride) -- the plan/0143 FATTN lever: the per-head slice load
// reads KV at ~33% of roofline (strided by row_width); reading the full row once
// and slicing in smem is the fix.
__device__ inline void mk_load_full_row(const half *cache, half *tile, uint32_t t0,
                                        uint32_t limit, uint32_t row_width, uint32_t rowp) {
    const uint32_t nv4 = row_width / 8;   // uint4 per full row
    for (uint32_t i = threadIdx.x; i < MK_FATTN_TILE * nv4; i += blockDim.x) {
        const uint32_t t = i / nv4, c = i % nv4;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (t0 + t < limit)
            v = ld_cg(reinterpret_cast<const uint4 *>(cache + (size_t)(t0 + t) * row_width) + c);
        unsigned *d = reinterpret_cast<unsigned *>(tile + (size_t)t * rowp + c * 8);
        d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
    }
}

#ifdef MK_FATTN_HMMA
// The HMMA flash body as a __noinline__ callee so its fragment/accumulator
// registers stay in ITS frame and do NOT inflate the shared interpreter (or the
// specialized kernel) register count -- the G11 72-block co-residency gate.
// Recomputes the dual-GPU pointers from (smem, a); the caller's guard has already
// established the contiguous full-row tile fits the 60 KiB slab.
__device__ __noinline__ inline void mk_fattn_hmma_dual(const FattnDecodeArgs &a, char *smem,
        uint32_t n_kv, int nchunks, int chunk) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t gqa = a.n_q / a.n_kv_heads;
    const uint32_t rw = a.row_width, rowp = rw + 2;
    // No q_s smem copy: q is read STRONG (.cg) from global in the QK pack below --
    // the roped q is tiny (n_q*HD f32), L2-hot, and re-reading it per tile keeps the
    // slab small enough to co-reside (full-row tile + out_acc alone are ~45 KiB;
    // adding q_s pushed 57.6 KiB, which the co-residency gate rejects).
    half  *tile  = reinterpret_cast<half *>(smem);
    half  *mask_t = tile + (size_t) MK_FATTN_TILE * rowp;
    uint32_t clen = (n_kv + nchunks - 1) / nchunks;
    clen = (clen + MK_FATTN_TILE - 1) / MK_FATTN_TILE * MK_FATTN_TILE;
    const uint32_t kv0 = min((uint32_t)(chunk * clen), n_kv);
    const uint32_t kv1 = min(kv0 + clen, n_kv);

    const int q_gid = lane >> 2, q_tid = lane & 3;
    const uint32_t kvh = (uint32_t) warp;
    const bool wcompute = kvh < a.n_kv_heads;
    float *out_acc = reinterpret_cast<float *>(mask_t + MK_FATTN_TILE);  // [n_q][HD]
    float *ms      = out_acc + (size_t) a.n_q * MK_ATTN_HD;              // [n_q][2]
    // FATTN sub-phase profiler (MK_PROFILE only): block-0 thread-0 accrues
    // clock64 deltas per phase into g_fattn_phase[]. thread-0 (warp 0) is always
    // a compute thread (kvh 0 < n_kv_heads), so it traverses every phase. Only
    // one thread reads the counter -> negligible perturbation of the other 383.
#ifdef MK_PROFILE
    const bool mkp = (blockIdx.x == 0 && threadIdx.x == 0 && g_fattn_phase);
    long long _pt = mkp ? clock64() : 0;
    #define MKP_LAP(ph) do { if (mkp) { long long _n = clock64(); \
        g_fattn_phase[ph] += _n - _pt; _pt = _n; } } while (0)
#else
    #define MKP_LAP(ph) do {} while (0)
#endif
    for (uint32_t i = threadIdx.x; i < a.n_q * MK_ATTN_HD; i += blockDim.x) out_acc[i] = 0.0f;
    for (uint32_t i = threadIdx.x; i < a.n_q * 2; i += blockDim.x)
        ms[i] = (i & 1) ? 0.0f : -FLT_MAX / 2.0f;
    __syncthreads();
    MKP_LAP(FATTN_PH_SETUP);

    constexpr int NT = MK_FATTN_TILE / 8;
    for (uint32_t t0 = kv0; t0 < kv1; t0 += MK_FATTN_TILE) {
        mk_load_full_row(a.k_cache, tile, t0, kv1, rw, rowp);
        if (threadIdx.x < MK_FATTN_TILE / 2) {
            const uint32_t j = t0 + 2 * threadIdx.x;
            unsigned w = 0;
            if (j < kv1) w = ld_cg(reinterpret_cast<const unsigned *>(a.mask + j));
            reinterpret_cast<unsigned *>(mask_t)[threadIdx.x] = w;
        }
        __syncthreads();
        MKP_LAP(FATTN_PH_KLOAD);
        half  pv_p0[NT], pv_p1[NT], pv_p0lo[NT], pv_p1lo[NT];  // 2-limb split of f32 P
        float pv_f = 1.0f;
        // ALL 32 lanes of a computing warp run the mma + quad-shfl (both are
        // warp-synchronous -- guarding by q_gid<gqa would diverge and deadlock).
        // Padding rows (q_gid >= gqa) use a safe dummy head index and never store.
        if (wcompute) {
            const bool real = q_gid < (int) gqa;
            const uint32_t qg = real ? (kvh * gqa + q_gid) : 0u;   // safe idx; padding discarded
            const uint32_t kvoff = kvh * MK_ATTN_HD;
            const float *qrow = a.q + (size_t) qg * MK_ATTN_HD;    // global, .cg
            // Independent accumulator fragments (TU102 paper tab:tensor: dependent
            // HMMA latency 14 cyc vs 2 cyc/inst throughput -> 7x). The NT position-
            // tiles are independent chains; the ks-outer loop interleaves their HMMAs
            // so the 14-cyc latency of one hides under the others -> throughput-bound.
            // q is shared across nt per d-slice, so it is read + 2-limb-split ONCE per
            // ks (was NT-times redundant): cuts the .cg q traffic and the split math 4x.
            float dd[NT][4] = {};
            for (int ks = 0; ks < MK_ATTN_HD / 8; ks++) {
                const int db = ks * 8;
                const float qs0 = MK_ATTN_SCALE * ld_cg(qrow + db + 2 * q_tid);
                const float qs1 = MK_ATTN_SCALE * ld_cg(qrow + db + 2 * q_tid + 1);
                const half qh0 = __float2half(qs0), qh1 = __float2half(qs1);
                const unsigned a_hi = mk_pk(qh0, qh1);              // f32-q recovery:
                const unsigned a_lo = mk_pk(__float2half(qs0 - __half2float(qh0)),
                                            __float2half(qs1 - __half2float(qh1)));
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const half *krow = tile + (size_t)(nt * 8 + q_gid) * rowp + kvoff;
                    const unsigned b0 = mk_pk(krow[db + 2 * q_tid], krow[db + 2 * q_tid + 1]);
                    mk_hmma_f32(dd[nt][0], dd[nt][1], dd[nt][2], dd[nt][3], a_hi, 0u, b0);
                    mk_hmma_f32(dd[nt][0], dd[nt][1], dd[nt][2], dd[nt][3], a_lo, 0u, b0);
                }
            }
            MKP_LAP(FATTN_PH_QK);
            float sc0[NT], sc1[NT];
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const int pb = nt * 8;
                const bool ok0 = (t0 + pb + 2 * q_tid) < kv1;
                const bool ok1 = (t0 + pb + 2 * q_tid + 1) < kv1;
                sc0[nt] = ok0 ? dd[nt][0] + __half2float(mask_t[pb + 2 * q_tid]) : -INFINITY;
                sc1[nt] = ok1 ? dd[nt][1] + __half2float(mask_t[pb + 2 * q_tid + 1]) : -INFINITY;
            }
            float ml = -INFINITY;
            for (int nt = 0; nt < NT; nt++) ml = fmaxf(ml, fmaxf(sc0[nt], sc1[nt]));
            ml = mk_qmax(ml);
            const float m_old = ms[qg * 2];
            const float m_new = fmaxf(m_old, ml);
            pv_f = expf(m_old - m_new);
            float sl = 0.0f;
            for (int nt = 0; nt < NT; nt++) {
                const float e0 = sc0[nt] > -INFINITY ? expf(sc0[nt] - m_new) : 0.0f;
                const float e1 = sc1[nt] > -INFINITY ? expf(sc1[nt] - m_new) : 0.0f;
                sl += e0 + e1;
                pv_p0[nt] = __float2half(e0); pv_p1[nt] = __float2half(e1);
                pv_p0lo[nt] = __float2half(e0 - __half2float(pv_p0[nt]));  // f32-P recovery
                pv_p1lo[nt] = __float2half(e1 - __half2float(pv_p1[nt]));
            }
            const float s_new = pv_f * ms[qg * 2 + 1] + mk_qsum(sl);
            if (real && q_tid == 0) { ms[qg * 2] = m_new; ms[qg * 2 + 1] = s_new; }
            MKP_LAP(FATTN_PH_SOFTMAX);
        }
        __syncthreads();
        mk_load_full_row(a.v_cache, tile, t0, kv1, rw, rowp);
        __syncthreads();
        MKP_LAP(FATTN_PH_VLOAD);
        if (wcompute) {   // all 32 lanes run the P.V mma; only real rows store
            const bool real = q_gid < (int) gqa;
            const uint32_t qg = real ? (kvh * gqa + q_gid) : 0u;
            const uint32_t kvoff = kvh * MK_ATTN_HD;
            float *oa = out_acc + (size_t) qg * MK_ATTN_HD;
            // Same independent-accumulator technique for P.V: the output d-tiles are
            // independent chains, interleaved G at a time to hide the 14-cyc HMMA
            // latency; the P fragment (p_hi,p_lo) is shared across d-tiles per ks.
            constexpr int G = 4;
            for (int dg = 0; dg < MK_ATTN_HD / 8; dg += G) {
                float oo[G][4] = {};
                for (int ks = 0; ks < NT; ks++) {
                    const int pb = ks * 8;
                    const unsigned p_hi = mk_pk(pv_p0[ks], pv_p1[ks]);
                    const unsigned p_lo = mk_pk(pv_p0lo[ks], pv_p1lo[ks]);
                    const half *vrow0 = tile + (size_t)(pb + 2 * q_tid) * rowp + kvoff;
                    const half *vrow1 = tile + (size_t)(pb + 2 * q_tid + 1) * rowp + kvoff;
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const int dcol = (dg + g) * 8 + q_gid;
                        const unsigned vb = mk_pk(vrow0[dcol], vrow1[dcol]);  // V single f16
                        mk_hmma_f32(oo[g][0], oo[g][1], oo[g][2], oo[g][3], p_hi, 0u, vb);
                        mk_hmma_f32(oo[g][0], oo[g][1], oo[g][2], oo[g][3], p_lo, 0u, vb);
                    }
                }
                if (real) {
#pragma unroll
                    for (int g = 0; g < G; g++) {
                        const int dt = dg + g;
                        oa[dt * 8 + 2 * q_tid]     = pv_f * oa[dt * 8 + 2 * q_tid]     + oo[g][0];
                        oa[dt * 8 + 2 * q_tid + 1] = pv_f * oa[dt * 8 + 2 * q_tid + 1] + oo[g][1];
                    }
                }
            }
        }
        __syncthreads();
        MKP_LAP(FATTN_PH_PV);
    }
    if (wcompute && q_gid < (int) gqa) {
        const uint32_t qg = kvh * gqa + q_gid;
        float *rec = a.partials + ((size_t) qg * nchunks + chunk) * MK_FATTN_PSTRIDE;
        const float *oa = out_acc + (size_t) qg * MK_ATTN_HD;
        for (int dt = 0; dt < MK_ATTN_HD / 8; dt++) {
            rec[dt * 8 + 2 * q_tid]     = oa[dt * 8 + 2 * q_tid];
            rec[dt * 8 + 2 * q_tid + 1] = oa[dt * 8 + 2 * q_tid + 1];
        }
        if (q_tid == 0) { rec[MK_ATTN_HD] = ms[qg * 2]; rec[MK_ATTN_HD + 1] = ms[qg * 2 + 1]; }
    }
    MKP_LAP(FATTN_PH_SETUP);   // fold the tail partials store into setup
#undef MKP_LAP
}
#endif

__device__ MK_OPFN void op_fattn_decode(const Instr &ins, char *smem) {
    const FattnDecodeArgs a = *reinterpret_cast<const FattnDecodeArgs *>(ins.payload);
    // Dual-GPU (row_width<=512) fits the full-row tile in the 60 KiB slab -> the
    // contiguous-load path (plan/0143: FATTN reads coalesced; measured ~7% deep,
    // bit-identical). Single-GPU (row_width 1024) overflows the slab -> per-head.
    if ((size_t) MK_FATTN_TILE * (a.row_width + 2) * 2 + (size_t) a.n_q * MK_ATTN_HD * 4 + 64
            <= (size_t) SMEM_BYTES) {
    // ---- contiguous-load variant (plan/0143 primary lever) --------------------
    // One warp = one q head (12 warps = 12 q heads dual-GPU). Full row loaded once
    // per tile (coalesced); each warp slices its kv head's 256-wide window in smem.
    // Math is IDENTICAL to the per-head path below (t0 ascending, same fold) ->
    // bit-identical output; only the DRAM read pattern changes.
    const uint32_t n_kv = ld_cg(reinterpret_cast<const unsigned *>(a.n_kv_cell));
    const int nchunks = ins.block_hi - ins.block_lo;
    const int chunk   = blockIdx.x - ins.block_lo;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t gqa = a.n_q / a.n_kv_heads;
    const uint32_t rw  = a.row_width;
    const uint32_t rowp = rw + 2;               // full-row pitch (bank pad)
    float *q_s   = reinterpret_cast<float *>(smem);
    half  *tile  = reinterpret_cast<half *>(q_s + a.n_q * MK_ATTN_HD);
    half  *mask_t = tile + (size_t)MK_FATTN_TILE * rowp;

    uint32_t clen = (n_kv + nchunks - 1) / nchunks;
    clen = (clen + MK_FATTN_TILE - 1) / MK_FATTN_TILE * MK_FATTN_TILE;
    const uint32_t kv0 = min((uint32_t)(chunk * clen), n_kv);
    const uint32_t kv1 = min(kv0 + clen, n_kv);

    for (uint32_t i = threadIdx.x; i < a.n_q * MK_ATTN_HD; i += blockDim.x)
        q_s[i] = MK_ATTN_SCALE * ld_cg(a.q + i);
    __syncthreads();

    const uint32_t qh = (uint32_t) warp;        // this warp's global q head
    const bool active = qh < a.n_q;
    const uint32_t head_off = (qh / gqa) * MK_ATTN_HD;   // its kv head's slice
#ifndef MK_FATTN_HMMA
    float m = -FLT_MAX / 2.0f, s = 0.0f;
    float vacc[MK_ATTN_HD / 32] = {0.0f};

    for (uint32_t t0 = kv0; t0 < kv1; t0 += MK_FATTN_TILE) {
        mk_load_full_row(a.k_cache, tile, t0, kv1, rw, rowp);
        if (threadIdx.x < MK_FATTN_TILE / 2) {
            const uint32_t j = t0 + 2 * threadIdx.x;
            unsigned w = 0;
            if (j < kv1) w = ld_cg(reinterpret_cast<const unsigned *>(a.mask + j));
            reinterpret_cast<unsigned *>(mask_t)[threadIdx.x] = w;
        }
        __syncthreads();
        float p = 0.0f, f = 1.0f;
        if (active) {
            const uint32_t j = t0 + lane;
            float score = -INFINITY;
            if (j < kv1) {
                const half  *kr = tile + (size_t)lane * rowp + head_off;
                const float *qh_p = q_s + qh * MK_ATTN_HD;
                float dot = 0.0f;
#pragma unroll
                for (int c = 0; c < MK_ATTN_HD; c += 2) {
                    const half2 kk = *reinterpret_cast<const half2 *>(kr + c);
                    dot += qh_p[c] * __low2float(kk) + qh_p[c + 1] * __high2float(kk);
                }
                score = dot + __half2float(mask_t[lane]);
            }
            const float m_new = fmaxf(m, warp_max_f32(score));
            f = expf(m - m_new);
            p = (j < kv1) ? expf(score - m_new) : 0.0f;
            s = s * f + warp_sum_f32(p);
            m = m_new;
        }
        __syncthreads();
        mk_load_full_row(a.v_cache, tile, t0, kv1, rw, rowp);
        __syncthreads();
        if (active) {
#pragma unroll
            for (int i = 0; i < MK_ATTN_HD / 32; i++) vacc[i] *= f;
            const half *vb = tile + head_off + lane * 8;
            for (int t = 0; t < MK_FATTN_TILE; t++) {
                const float pt = __shfl_sync(0xffffffffu, p, t);
                if (pt != 0.0f) {
                    const half *vr = vb + (size_t)t * rowp;
#pragma unroll
                    for (int i = 0; i < MK_ATTN_HD / 32; i += 2) {
                        const half2 vv = *reinterpret_cast<const half2 *>(vr + i);
                        vacc[i]     += pt * __low2float(vv);
                        vacc[i + 1] += pt * __high2float(vv);
                    }
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        float *rec = a.partials + ((size_t) qh * nchunks + chunk) * MK_FATTN_PSTRIDE;
#pragma unroll
        for (int i = 0; i < MK_ATTN_HD / 32; i++) rec[lane * 8 + i] = vacc[i];
        if (lane == 0) { rec[MK_ATTN_HD] = m; rec[MK_ATTN_HD + 1] = s; }
    }
    return;
#else
    // ---- HMMA flash decode (plan/0143 primary; spikes ae74152/a90f67f) --------
    // The heavy body is a __noinline__ callee (mk_fattn_hmma_dual) so its fragment
    // and accumulator registers do not inflate the shared interpreter frame (G11
    // 72-block co-residency). Out-accumulator + per-q (max,sum) live in the slab.
    (void) qh; (void) active; (void) head_off; (void) gqa;
    mk_fattn_hmma_dual(a, smem, n_kv, nchunks, chunk);
    return;
#endif
    }
    // ---- per-head slice path (single-GPU, or any row_width that overflows) ----
    // n_kv changes every 256 tokens at deep context; the persistent kernel's L1
    // is incoherent across passes, so a plain read of a per-pass-patched payload
    // could return a stale window. Read it STRONG from the host-written cell.
    const uint32_t n_kv = ld_cg(reinterpret_cast<const unsigned *>(a.n_kv_cell));
    const int nchunks = ins.block_hi - ins.block_lo;
    const int chunk   = blockIdx.x - ins.block_lo;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

    // The smem tile holds ONE kv head's 256-wide slice (not the full row):
    // this keeps the slab bounded at the single-GPU shape (24 q / 4 kv,
    // row_width 1024) where a full-row tile would be 65 KiB. The op loops the
    // kv heads on the outer axis; each carries its gqa q heads (one warp per q
    // head, gqa <= blockDim/32).
    const uint32_t row_p = MK_ATTN_HD + 2; // slice pitch in f16
    float *q_s    = reinterpret_cast<float *>(smem);
    half  *tile   = reinterpret_cast<half *>(q_s + a.n_q * MK_ATTN_HD);
    half  *mask_t = tile + MK_FATTN_TILE * row_p;

    // 32-aligned chunk boundaries (NOTES-attn.md / COMPILER.md): clen rounded
    // up to a multiple of the tile so kv0 = chunk*clen is 32-aligned and the
    // packed 2-f16 mask word read never faults. Trailing chunks fall empty.
    uint32_t clen = (n_kv + nchunks - 1) / nchunks;
    clen = (clen + MK_FATTN_TILE - 1) / MK_FATTN_TILE * MK_FATTN_TILE;
    const uint32_t kv0  = min((uint32_t)(chunk * clen), n_kv);
    const uint32_t kv1  = min(kv0 + clen, n_kv);

    // Q pre-scaled exactly like the fork (fattn-mma-f16.cuh:1230), kept f32
    // (the graph pins GGML_PREC_F32 on FLASH_ATTN_EXT). All n_q heads staged.
    for (uint32_t i = threadIdx.x; i < a.n_q * MK_ATTN_HD; i += blockDim.x) {
        q_s[i] = MK_ATTN_SCALE * ld_cg(a.q + i);
    }
    __syncthreads();

    const uint32_t gqa = a.n_q / a.n_kv_heads;

    for (uint32_t kvh = 0; kvh < a.n_kv_heads; kvh++) {
        const uint32_t head_off = kvh * MK_ATTN_HD;  // slice base in each row
        const uint32_t qh = kvh * gqa + (uint32_t) warp; // this warp's global q head
        const bool active = (uint32_t) warp < gqa;

        float m = -FLT_MAX / 2.0f; // fork KQ_max init, fattn-mma-f16.cuh:1196
        float s = 0.0f;
        float vacc[MK_ATTN_HD / 32] = {0.0f};

        for (uint32_t t0 = kv0; t0 < kv1; t0 += MK_FATTN_TILE) {
            // ---- phase K: scores for 32 positions, this kv head's slice
            mk_load_kv_slice(a.k_cache, tile, t0, kv1, a.row_width, head_off, row_p);
            if (threadIdx.x < MK_FATTN_TILE / 2) {
                // mask tile as 16 words (n_kv even: padded to 256; kv0 32-aligned).
                const uint32_t j = t0 + 2 * threadIdx.x;
                unsigned w = 0;
                if (j < kv1) w = ld_cg(reinterpret_cast<const unsigned *>(a.mask + j));
                reinterpret_cast<unsigned *>(mask_t)[threadIdx.x] = w;
            }
            __syncthreads();

            float p = 0.0f, f = 1.0f;
            if (active) {
                const uint32_t j = t0 + lane; // this lane's KV position
                float score = -INFINITY;      // chunk-tail positions stay -inf
                if (j < kv1) {
                    const half  *kr = tile + lane * row_p; // slice already selected
                    const float *qh_p = q_s + qh * MK_ATTN_HD;
                    float dot = 0.0f;
#pragma unroll
                    for (int c = 0; c < MK_ATTN_HD; c += 2) {
                        const half2 kk = *reinterpret_cast<const half2 *>(kr + c);
                        dot += qh_p[c] * __low2float(kk) + qh_p[c + 1] * __high2float(kk);
                    }
                    score = dot + __half2float(mask_t[lane]); // q pre-scaled; mask 0/-inf
                }
                const float m_new = fmaxf(m, warp_max_f32(score));
                f = expf(m - m_new);                      // m > -inf always: no NaN
                p = (j < kv1) ? expf(score - m_new) : 0.0f;
                s = s * f + warp_sum_f32(p);
                m = m_new;
            }
            __syncthreads();

            // ---- phase V: same smem, weighted accumulation
            mk_load_kv_slice(a.v_cache, tile, t0, kv1, a.row_width, head_off, row_p);
            __syncthreads();
            if (active) {
#pragma unroll
                for (int i = 0; i < MK_ATTN_HD / 32; i++) vacc[i] *= f;
                const half *vb = tile + lane * 8; // lane owns dims [8*lane, 8*lane+8)
                for (int t = 0; t < MK_FATTN_TILE; t++) {
                    const float pt = __shfl_sync(0xffffffffu, p, t); // warp-uniform
                    if (pt != 0.0f) {
                        const half *vr = vb + t * row_p;
#pragma unroll
                        for (int i = 0; i < MK_ATTN_HD / 32; i += 2) {
                            const half2 vv = *reinterpret_cast<const half2 *>(vr + i);
                            vacc[i]     += pt * __low2float(vv);
                            vacc[i + 1] += pt * __high2float(vv);
                        }
                    }
                }
            }
            __syncthreads(); // tile is reloaded next iteration
        }

        // One partial record per (head, chunk); written even for an empty or
        // fully-masked chunk (max = -FLT_MAX/2, sumexp = 0) so that only a
        // never-executed write leaves the -inf sentinel in place.
        if (active) {
            float *rec = a.partials + ((size_t) qh * nchunks + chunk) * MK_FATTN_PSTRIDE;
#pragma unroll
            for (int i = 0; i < MK_ATTN_HD / 32; i++) rec[lane * 8 + i] = vacc[i];
            if (lane == 0) {
                rec[MK_ATTN_HD]     = m;
                rec[MK_ATTN_HD + 1] = s;
            }
        }
        __syncthreads(); // tile handoff before the next kv head reloads it
    }
}

// ------------------------------------------------------------ OP_FATTN_REDUCE
//
// The Y03 merge: fold the per-chunk partials of each head with the fork's
// max/LSE-stable combine (fattn-common.cuh:732-749, SOFTMAX_FTZ_THRESHOLD
// -20), then divide by the merged rowsum. Heads are strided across the
// participating blocks; threads [0,256) each own one output dim.
//
// Sentinel: a partial whose max slot still reads -inf was never written
// (see MK_FATTN_SENTINEL above); it is counted into *error and skipped
// instead of being folded, so a lost split turns into a detectable count,
// not silently wrong attention output.

struct FattnReduceArgs {
    const float *partials;  // [n_q][n_chunks][MK_FATTN_PSTRIDE] (strong reads)
    float       *dst;       // [n_q][256]
    unsigned    *error;     // sentinel detections (host-checked; zeroed per pass)
    uint32_t     n_q;
    uint32_t     n_chunks;
};
static_assert(sizeof(FattnReduceArgs) <= 112, "payload overflow");

__device__ MK_OPFN void op_fattn_reduce(const Instr &ins, char *) {
    const FattnReduceArgs a = *reinterpret_cast<const FattnReduceArgs *>(ins.payload);
    const int nb  = ins.block_hi - ins.block_lo;
    const int rel = blockIdx.x - ins.block_lo;
    const int d   = threadIdx.x;
    if (d >= MK_ATTN_HD) return;

    for (uint32_t h = rel; h < a.n_q; h += nb) {
        float max_val = -INFINITY, rowsum = 0.0f, acc = 0.0f;
        for (uint32_t c = 0; c < a.n_chunks; c++) {
            const float *rec = a.partials + ((size_t) h * a.n_chunks + c) * MK_FATTN_PSTRIDE;
            const float m_c = ld_cg(rec + MK_ATTN_HD);
            if (m_c == MK_FATTN_SENTINEL) { // never-written partial (Y03)
                if (d == 0) atomicAdd(a.error, 1u);
                continue;
            }
            const float s_c = ld_cg(rec + MK_ATTN_HD + 1);
            const float v_c = ld_cg(rec + d);
            // fattn-common.cuh:737-748 with ascending chunk order; the FTZ
            // guard also makes the -inf running start exact (first live
            // chunk contributes with weight 1, the start with weight 0).
            const float max_new   = fmaxf(max_val, m_c);
            const float diff_val  = max_val - max_new;
            const float diff_add  = m_c - max_new;
            const float scale_val = diff_val >= -20.0f ? expf(diff_val) : 0.0f;
            const float scale_add = diff_add >= -20.0f ? expf(diff_add) : 0.0f;
            acc    = scale_val * acc    + scale_add * v_c;
            rowsum = scale_val * rowsum + scale_add * s_c;
            max_val = max_new;
        }
        // Final normalize (fattn-common.cuh:752). rowsum > 0 at decode: the
        // current token's own position is always unmasked.
        a.dst[(size_t) h * MK_ATTN_HD + d] = acc / rowsum;
    }
}

// --------------------------------------------------------------- OP_ATTN_GATE
//
// dst = attn * sigmoid(gate) (graph MUL(attn, sigmoid(gate slice)), the
// gate slice being the odd 256-halves of the attn_q GEMV's [q|gate]
// interleave: head h's gate at qgemv + h*512 + 256; the offline compiler
// resolves `gate` to that +256 base).

struct AttnGateArgs {
    const float *attn;        // [n_q*256] from OP_FATTN_REDUCE (strong reads)
    const float *gate;        // head h at gate + h*gate_stride (strong reads)
    float       *dst;         // [n_q*256] contiguous
    uint32_t     n_q;
    uint32_t     gate_stride; // f32 elems between heads (512)
};
static_assert(sizeof(AttnGateArgs) <= 112, "payload overflow");

__device__ MK_OPFN void op_attn_gate(const Instr &ins, char *) {
    const AttnGateArgs a = *reinterpret_cast<const AttnGateArgs *>(ins.payload);
    const uint32_t total = a.n_q * MK_ATTN_HD;
    const uint32_t nthr  = (ins.block_hi - ins.block_lo) * blockDim.x;
    const uint32_t t0    = (blockIdx.x - ins.block_lo) * blockDim.x + threadIdx.x;
    for (uint32_t i = t0; i < total; i += nthr) {
        const uint32_t h = i / MK_ATTN_HD, d = i % MK_ATTN_HD;
        const float gv = ld_cg(a.gate + (size_t) h * a.gate_stride + d);
        const float av = ld_cg(a.attn + i);
        // op_sigmoid, unary.cu:48-50; MUL order attn * sigmoid per the graph.
        a.dst[i] = av * (1.0f / (1.0f + expf(-gv)));
    }
}

} // namespace mk
