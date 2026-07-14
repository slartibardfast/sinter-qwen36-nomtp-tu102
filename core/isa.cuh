// The interpreter ISA: the contract between the offline schedule compiler
// (k0/compile_schedule.py), the host uploader, and every op implementation.
//
// A program is a flat array of fixed-size instructions executed in order by
// the persistent cooperative kernel (72 blocks x 384 threads, 60 KiB smem
// slab). Instructions between two boundaries form an antichain: they carry
// disjoint [block_lo, block_hi) ranges and no data dependencies, so the
// interpreter runs every instruction whose range contains blockIdx.x, then
// crosses one Y02 boundary per BOUNDARY instruction (core/sync.cuh).
//
// Conventions every op obeys:
//   - Signature: __device__ void op_<kind>(const Instr&, char *smem);
//     the op derives its share of work from (blockIdx.x - block_lo),
//     (block_hi - block_lo) and threadIdx.x. Nothing else is implicit.
//   - All pointers in Instr are raw device pointers, fully resolved at
//     upload time (this GPU's half under the meta split). No indirection
//     tables in the hot loop.
//   - Activations are f32; KV cache f16; recurrent state f32; weights
//     Q4_0 / Q4_0_AR16 / F16 exactly as the GGUF stores them (shared
//     buffers, never repacked).
//   - Mutable data read across a boundary is read STRONG (.cg helpers in
//     core/sync.cuh); weights use plain vectorized loads (uint4-wide, the
//     G12 discipline: scalar weight loads are the 70-75%-of-peak trap).
//   - fp32 fold orders that G13 declares (the GDN state fold, the
//     cross-GPU exchange p0+p1) are fixed by the op implementation and
//     documented where they are implemented.
#pragma once
#include <cstdint>

// Op-function inline attribute (call/0023 D2). The interpreter folds ops into
// op_dispatch's switch (__forceinline__); the SPECIALIZED straight-line kernel
// must NOT inline (1540 inlined op bodies blow the Turing L1I -- the naive
// specialized build measured 229,704 SASS instrs, ~28x the ~8K-instr L1I).
// __noinline__ makes the body 1540 CALLs to 24 op bodies emitted once.
// `inline` gives the (header-defined) op bodies COMDAT/weak linkage so both
// TUs that include this header -- interp.cu (the kernel) and harness.cpp (the
// packer) -- may each emit a copy and the linker folds them to one; without it
// __noinline__'s external symbols collide (ODR) at link. `inline` is a linkage
// attribute, orthogonal to __noinline__'s no-inline-at-callsite codegen.
#if defined(MK_SPECIALIZED)
#define MK_OPFN __noinline__ inline
#else
#define MK_OPFN __forceinline__
#endif

namespace mk {

enum MacroKind : uint16_t {
    OP_NOP = 0,
    OP_BOUNDARY,        // cross the Y02 boundary (no other payload)
    // prologue / epilogue
    OP_EMBED_LOOKUP,    // token embedding row -> residual x (f16 -> f32)
    OP_RMSNORM,         // eps 1e-6, fused weight multiply, optional add-src
    OP_QUANT_Q8_1,      // f32 activation -> q8_1 blocks for MMVQ
    OP_HEAD_GEMV_F16,   // lm_head rows (this GPU's vocab half) -> logits
    OP_LOGITS_EMIT,     // logits half -> result_output layout + host flag
    // GEMV family
    OP_MMVQ_Q4_0,       // row-range GEMV, q8_1 activations
    OP_MMVQ_Q4_0_FUSED, // up|gate fused rows + silu(gate)*up epilogue
    OP_MMVQ_AR16,       // Q4_0_AR16 rows (interleaved nibbles, no s-term)
    OP_GEMV_F16,        // f16 rows (ssm_alpha/ssm_beta and friends)
    // DeltaNet block
    OP_CONV_SHIFT_CONCAT, // conv state shift + qkv concat store
    OP_SSM_CONV_SILU,     // 4-tap causal conv + silu, per channel
    OP_QK_L2NORM,         // per-head l2 norm of q,k
    OP_GDN_GATES,         // softplus/sigmoid/a/dt gate math
    OP_GDN_STEP,          // the delta-rule state update + output partial
    OP_GATED_RMSNORM,     // ssm_norm per head + silu(z) gating
    // attention block
    OP_QK_NORM_ROPE,    // per-head rmsnorm + IMROPE (sections, 64 dims)
    OP_KV_APPEND,       // f32 -> f16 SET_ROWS by i64 row index
    OP_FATTN_DECODE,    // split-KV online-softmax over this GPU's kv heads
    OP_FATTN_REDUCE,    // Y03 max/LSE merge of split partials
    OP_ATTN_GATE,       // sigmoid(gate) * attn out (24 gated q heads)
    // elementwise / glue
    OP_RESIDUAL_ADD,
    OP_STATE_LOAD,      // recurrent state fetch (get_rows semantics)
    OP_STATE_STORE,     // recurrent state commit (set_rows semantics)
    // cross-GPU (Y06 discipline; core/exchange.cuh)
    OP_XCHG_PUSH,       // peer-store local partial (line-filling v4) + seqno
    OP_XCHG_REDUCE,     // poll seqno, p0 + p1 in fixed order -> mirrored out
    OP_KIND_COUNT
};

// One instruction, 128 bytes, kind-specific payload. The offline compiler
// packs the payload struct for the kind; ops cast payload to their own
// args type (static_assert(sizeof(Args) <= sizeof(Instr::payload))).
struct alignas(16) Instr {
    uint16_t kind;       // MacroKind
    uint16_t block_lo;   // first participating block
    uint16_t block_hi;   // one past last participating block
    uint16_t flags;      // op-local
    uint32_t dbg_node;   // first source cgraph node index (debug schedule)
    uint32_t _pad;
    unsigned char payload[112];
};
static_assert(sizeof(Instr) == 128, "instruction size is part of the ABI");

// Program header uploaded once per schedule; instructions follow.
struct Program {
    uint32_t n_instr;
    uint32_t epoch_stride;   // boundaries per pass (G15 accounting)
    unsigned *y02_counter;   // the grid boundary counter for this GPU
};

} // namespace mk
