// The op registry: the ONE place an op family integrates with the
// interpreter. To wire a new family:
//   1. add its #include below;
//   2. add one MK_OP(OP_KIND, op_function) line to MK_WIRED_OPS.
// Both op_wired() and op_dispatch() are generated from the same list, so a
// kind cannot be dispatchable without being declared wired (or vice versa).
// A kind that is NOT in this list makes the interpreter raise
// ERR_UNWIRED_KIND and exit the grid — never a silent skip (interp.cu).
#pragma once
#include "glue.cuh"
#include "gdn.cuh"
// TODO(gemv family): #include "gemv.cuh"    + MK_OP lines for OP_QUANT_Q8_1,
//     OP_MMVQ_Q4_0, OP_MMVQ_Q4_0_FUSED, OP_MMVQ_AR16, OP_GEMV_F16,
//     OP_HEAD_GEMV_F16
// TODO(attn family): #include "attn.cuh"    + MK_OP lines for
//     OP_QK_NORM_ROPE, OP_KV_APPEND, OP_FATTN_DECODE, OP_FATTN_REDUCE,
//     OP_ATTN_GATE
// TODO(xchg family): #include "xchg.cuh"    + MK_OP lines for OP_XCHG_PUSH,
//     OP_XCHG_REDUCE

namespace mk {

// One MK_OP(kind, device_function) line per wired op.
#define MK_WIRED_OPS(MK_OP)                            \
    MK_OP(OP_RMSNORM, op_rmsnorm)                      \
    MK_OP(OP_RESIDUAL_ADD, op_residual_add)            \
    MK_OP(OP_EMBED_LOOKUP, op_embed_lookup)            \
    MK_OP(OP_LOGITS_EMIT, op_logits_emit)              \
    MK_OP(OP_STATE_LOAD, op_state_load)                \
    MK_OP(OP_CONV_SHIFT_CONCAT, op_conv_shift_concat)  \
    MK_OP(OP_SSM_CONV_SILU, op_ssm_conv_silu)          \
    MK_OP(OP_QK_L2NORM, op_qk_l2norm)                  \
    MK_OP(OP_GDN_GATES, op_gdn_gates)                  \
    MK_OP(OP_GDN_STEP, op_gdn_step)                    \
    MK_OP(OP_GATED_RMSNORM, op_gated_rmsnorm)          \
    MK_OP(OP_STATE_STORE, op_state_store)

__device__ __forceinline__ bool op_wired(uint16_t kind) {
    switch (kind) {
    case OP_NOP:
    case OP_BOUNDARY:
#define MK_OP(k, fn) case k:
    MK_WIRED_OPS(MK_OP)
#undef MK_OP
        return true;
    default:
        return false;
    }
}

__device__ __forceinline__ void op_dispatch(const Instr &in, char *smem) {
    switch (in.kind) {
#define MK_OP(k, fn) case k: fn(in, smem); break;
    MK_WIRED_OPS(MK_OP)
#undef MK_OP
    default: // OP_NOP does nothing; unwired kinds never reach dispatch
        break;
    }
}

} // namespace mk
