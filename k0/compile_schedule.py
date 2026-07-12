#!/usr/bin/env python3
"""compile_schedule.py - offline schedule compiler for the k=0 persistent megakernel.

Input : k0/schedule.csv + k0/leaves.csv (the extracted 3704-node decode graph,
        format per k0/parse.py and docs/BLOCKS.md).
Output: k0/program.json - a SYMBOLIC program over the core/isa.cuh ISA.
        Tensor operands are names (GGUF weight names, cache role+layer names,
        scratch-buffer names, host input cells, $runtime symbols); a C++ packer
        resolves names to device pointers and packs the 128-byte Instrs later.

Bindings compiled in (design decisions, recorded in the JSON header):
  - single GPU: every GEMV row range is FULL (the dual-GPU compile later
    parameterizes rows by the meta split);
  - single sequence (k=0): the zero-extent clear chains and the dead
    rs_index_setup views are dropped (SCHEDULE-QUESTIONS items 14/15);
  - mask_convert is dropped: the harness writes the f16 mask directly.

Run: python3 k0/compile_schedule.py   (no arguments; paths relative to script)
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --split: emit the dual-GPU tensor-parallel program (core/DUAL-GPU-DESIGN.md).
# NGPU=2. Row-split expand GEMVs (qkv/up|gate/lm_head) get half the output
# rows; col-split contract GEMVs (attn_output/ffn_down/ssm_out) dot half the
# input columns into a local partial, then an OP_XCHG_PUSH -> OP_BOUNDARY ->
# OP_XCHG_REDUCE triple folds p0+p1 to the mirrored residual. KV/state ops get
# the per-GPU head range (halved counts/offsets). Mirrored ops (norms,
# elementwise, rope, gates, embed) are unchanged: both GPUs run them on the
# mirrored residual. The op implementations are reused UNCHANGED; the split is
# entirely in the halved ranges/counts and the inserted reduce. One program
# serves both GPUs: the harness packs it per GPU, patching $gpu_index (fold
# order) and the peer mailbox pointers.
SPLIT = "--split" in sys.argv
NGPU = 2 if SPLIT else 1


def sp(n):
    """This GPU's local share of a split dimension (exact half under --split)."""
    assert n % NGPU == 0, "split dim %d not divisible by %d" % (n, NGPU)
    return n // NGPU


N_NODES = 3704
GRID = 72              # persistent blocks per GPU
EPS = 1e-6
N_EMBD = 5120
N_FF = 17408
N_VOCAB = 248320
GDN_HEADS = 48
GDN_HEAD_DIM = 128
ATTN_Q_HEADS = 24      # measured (QUESTIONS item 1), not the briefed 32
ATTN_KV_HEADS = 4
ATTN_HEAD_DIM = 256
CONV_CHANNELS = 10240
D_CONV = 4

BPE = {"f32": 4.0, "f16": 2.0, "i32": 4.0, "i64": 8.0,
       "q4_0": 18.0 / 32.0, "q4_0_ar16": 10.0 / 16.0}


def fail(msg):
    print("FAIL: " + msg, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- load csvs
def load_nodes():
    nodes = {}
    with open(os.path.join(HERE, "schedule.csv")) as f:
        for row in csv.DictReader(f):
            i = int(row["node_idx"])
            nodes[i] = {
                "op": row["op"], "type": row["type"],
                "ne": [int(x) for x in row["ne"].split("x")],
                "params": row["decoded_params"],
                "src": row["src_refs"].split(";") if row["src_refs"] else [],
                "block": row["block_idx"],
                "mid": int(row["macro_op_id"]), "mkind": row["macro_kind"],
            }
    return nodes


def load_leaves():
    leaves = {}
    with open(os.path.join(HERE, "leaves.csv")) as f:
        for row in csv.DictReader(f):
            leaves[int(row["leaf_idx"])] = {
                "type": row["type"],
                "ne": [int(x) for x in row["ne"].split("x")],
                "cls": row["classification"],
            }
    return leaves


NODES = load_nodes()
LEAVES = load_leaves()
if len(NODES) != N_NODES:
    fail("expected %d nodes, got %d" % (N_NODES, len(NODES)))


def leaf(ref):
    assert ref.startswith("l"), ref
    return LEAVES[int(ref[1:])]


def weight_name(ref):
    cls = leaf(ref)["cls"]
    assert cls.startswith("weight:"), (ref, cls)
    return cls.split(":", 1)[1]


def weight_bytes(name_or_ref, ref=True):
    lf = leaf(name_or_ref)
    return math.prod(lf["ne"]) * BPE[lf["type"]]


def cache_name(ref):
    """cache_k(blk.3) -> cache_k_l3 ; state:conv_state(blk.0) -> state_r_l0."""
    cls = leaf(ref)["cls"]
    m = re.match(r"cache_([kv])\(blk\.(\d+)\)$", cls)
    if m:
        return "cache_%s_l%s" % (m.group(1), m.group(2))
    m = re.match(r"state:(conv|ssm)_state\(blk\.(\d+)\)$", cls)
    if m:
        return "state_%s_l%s" % ("r" if m.group(1) == "conv" else "s", m.group(2))
    fail("not a cache/state leaf: %s -> %s" % (ref, cls))


def resolve_leaf(ref):
    """Follow view-like nodes back to the underlying leaf ref."""
    while ref.startswith("n"):
        n = NODES[int(ref[1:])]
        assert n["op"] in ("VIEW", "PERMUTE", "RESHAPE", "TRANSPOSE", "CONT"), \
            (ref, n["op"])
        ref = n["src"][0]
    return ref


# ---------------------------------------------------------------- instances
INST_NODES = {}
INST_KIND = {}
for i in sorted(NODES):
    n = NODES[i]
    INST_NODES.setdefault(n["mid"], []).append(i)
    INST_KIND[n["mid"]] = n["mkind"]

BLOCK_INSTS = {}   # block -> kind -> [mid], mids in first-node order
for mid in sorted(INST_NODES, key=lambda m: INST_NODES[m][0]):
    first = INST_NODES[mid][0]
    blk = NODES[first]["block"]
    BLOCK_INSTS.setdefault(blk, {}).setdefault(INST_KIND[mid], []).append(mid)


def insts(blk, kind, want=None):
    ms = BLOCK_INSTS.get(blk, {}).get(kind, [])
    if want is not None and len(ms) != want:
        fail("block %s: expected %d %s instances, got %d" % (blk, want, kind, len(ms)))
    return [INST_NODES[m] for m in ms]


# ---------------------------------------------------------------- scratch buffers
# Under --split every buffer downstream of a row-split expand (or feeding a
# col-split contract) holds this GPU's LOCAL half at offset 0; the mirrored
# trunk (residual, xn) and the contract OUTPUT (proj_out, full 5120 partial
# then reduced to mirrored) stay full width.
Q8_ELEMS = sp(N_FF)                   # largest LOCAL quantized activation vector
Q8_BYTES = Q8_ELEMS // 32 * 36        # q8_1: 36 B per 32-element block
# pstride 260 = 256 vkq + max + sumexp + 2 pad (16 B record alignment,
# MK_FATTN_PSTRIDE in k0/ops/attn.cuh); the buffer must match the op's stride.
FATTN_PARTIAL_ELEMS = GRID * sp(ATTN_Q_HEADS) * (ATTN_HEAD_DIM + 4)

BUFFERS = [
    ("residual", "f32", N_EMBD, "the residual trunk x (mirrored)"),
    ("xn", "f32", N_EMBD, "rmsnorm output (mirrored, feeds row-split expands)"),
    ("q8_act", "q8_1", Q8_ELEMS,
     "quantized activations for MMVQ; sized for the largest LOCAL quantized "
     "vector (the 8704-wide half FFN GLU output); rewritten several times/block"),
    ("mixer_out", "f32", sp(12288),
     "mixer GEMV output: attn q|gate (local 12x512=6144) or DN qkv (local "
     "5120); on DN blocks reused for post-conv qkv (OP_SSM_CONV_SILU dst)"),
    ("conv_ws", "f32", D_CONV * sp(CONV_CHANNELS),
     "conv concat window [d_conv=4 x local channels]"),
    ("gdn_alpha", "f32", sp(GDN_HEADS),
     "ssm_alpha GEMV out; becomes g = softplus(alpha+dt_bias)*a in place"),
    ("gdn_beta", "f32", sp(GDN_HEADS),
     "ssm_beta GEMV out; becomes sigmoid(beta) in place"),
    ("gdn_state", "f32", sp(786432), "GDN state working copy (local heads)"),
    ("attn_out", "f32", sp(6144),
     "mixer output vector: GDN token out / FATTN merged out; gated in place"),
    ("gate_out", "f32", sp(6144), "DN z-gate GEMV out (attn_gate.weight)"),
    ("k_stage", "f32", sp(1024), "current-token K row; normed+roped in place"),
    ("v_stage", "f32", sp(1024), "current-token V row"),
    ("fattn_q", "f32", sp(ATTN_Q_HEADS) * ATTN_HEAD_DIM,
     "compact roped q (local heads x 256), deinterleaved from mixer_out"),
    ("fattn_partial", "f32", FATTN_PARTIAL_ELEMS,
     "split-KV partials: 72 splits x local heads x (256 vkq + max + sumexp)"),
    ("ffn_ws", "f32", sp(N_FF), "FFN intermediate silu(gate)*up (local half)"),
    ("proj_out", "f32", N_EMBD,
     "block output projection (ssm_out / attn_output / ffn_down) partial, "
     "reduced in place to the mirrored residual fold (next OP_RMSNORM add_src)"),
    ("logits", "f32", sp(N_VOCAB), "lm head output (this GPU's vocab half)"),
]


def buf_bytes(name):
    for nm, ty, elems, _ in BUFFERS:
        if nm == name:
            return Q8_BYTES if ty == "q8_1" else elems * 4
    fail("unknown buffer " + name)


# ---------------------------------------------------------------- emission
PROG = []      # instruction dicts, program order (boundaries inserted later)
WINDOWS = []   # list of [PROG index, ...] antichain windows
COVERED = {}   # node idx -> PROG idx
DROPPED = {}   # node idx -> reason
WEIGHTS_USED = []


def I(kind, nodes, args, est=4096, reads=(), writes=()):
    for x in nodes:
        if x in COVERED or x in DROPPED:
            fail("node n%d assigned twice" % x)
        COVERED[x] = len(PROG)
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("gguf:"):
            WEIGHTS_USED.append(v[5:])
    ins = {"kind": kind, "dbg_node": nodes[0] if nodes else None,
           "nodes": nodes, "args": args,
           "_est": est, "_r": set(reads), "_w": set(writes)}
    PROG.append(ins)
    return len(PROG) - 1


def W(*idxs):
    WINDOWS.append(list(idxs))


def drop(nodes, reason):
    for x in nodes:
        if x in COVERED or x in DROPPED:
            fail("node n%d assigned twice (drop)" % x)
        DROPPED[x] = reason


def split_live(nodes):
    """Separate zero-extent clear-chain nodes (any ne dim == 0) from live ones."""
    live = [x for x in nodes if 0 not in NODES[x]["ne"]]
    dead = [x for x in nodes if 0 in NODES[x]["ne"]]
    return live, dead


def mulmats(nodes):
    return [x for x in nodes if NODES[x]["op"] == "MUL_MAT"]


def rmsnorm_weight(nodes):
    muls = [x for x in nodes if NODES[x]["op"] == "MUL" and NODES[x]["src"][1].startswith("l")]
    assert len(muls) == 1, nodes
    return weight_name(NODES[muls[0]]["src"][1])


def emit_rmsnorm(nodes, weight, src, dst, add_src, extra_nodes=(), extra=None):
    args = {"src": "buf:" + src, "weight": "gguf:" + weight, "dst": "buf:" + dst,
            "width": N_EMBD, "eps": EPS}
    reads, writes = {src}, {dst}
    if add_src:
        # ISA "optional add-src": sum = src + add_src is written back to src
        # (the residual) and the norm of the sum to dst.
        args["add_src"] = "buf:" + add_src
        args["write_sum"] = "buf:" + src
        reads.add(add_src)
        writes.add(src)
    if extra:
        args.update(extra)
    return I("OP_RMSNORM", list(nodes) + list(extra_nodes), args,
             reads=reads, writes=writes)


def emit_quant(src, elems, anchor):
    ins = I("OP_QUANT_Q8_1", [],
            {"src": "buf:" + src, "elems": elems, "dst": "buf:q8_act"},
            reads={src}, writes={"q8_act"})
    PROG[ins]["dbg_node"] = anchor
    PROG[ins]["inserted"] = True
    return ins


def emit_mmvq(kind, nodes, weight_ref, src_elems, dst, dst_off=0, axis="row"):
    """axis='row' (expand): this GPU computes M/NGPU output rows over the FULL
    (mirrored) K-column input. axis='col' (contract): this GPU dots K/NGPU
    input columns (its local activation half) over ALL M output rows into a
    partial; the caller follows with emit_col_reduce. Both compact to local
    buffers at offset 0 (the uploader slices each weight to this GPU's half)."""
    name = weight_name(weight_ref)
    lf = leaf(weight_ref)
    K, M = lf["ne"][0], lf["ne"][1]
    assert K == src_elems, (name, lf["ne"], src_elems)
    if axis == "row":
        row_hi, ncols = sp(M), K            # half rows, full mirrored input
        est = weight_bytes(weight_ref) / NGPU
    else:                                   # col: full rows, half input columns
        row_hi, ncols = M, sp(K)
        est = weight_bytes(weight_ref) / NGPU
    return I(kind, nodes,
             {"weight": "gguf:" + name, "src": "buf:q8_act",
              "src_elems": ncols, "dst": "buf:" + dst, "dst_off": dst_off,
              "row_lo": 0, "row_hi": row_hi, "split_axis": axis},
             est=est, reads={"q8_act"}, writes={dst})


# One reduce mailbox site per col-split contract projection (2/block x 64 =
# 128). The push and the reduce are each a single-instruction window (whole
# grid), so each adds one Y02 boundary: +2 boundaries per site.
XCHG_SITE = [0]


def emit_col_reduce(anchor):
    """The cross-GPU fold for a col-split contract at buf:proj_out: push the
    local partial to the peer's mailbox, boundary, then p0+p1 -> mirrored
    proj_out (in place). No-op single-GPU."""
    if not SPLIT:
        return
    site = XCHG_SITE[0]
    XCHG_SITE[0] += 1
    ipush = I("OP_XCHG_PUSH", [],
              {"local_partial": "buf:proj_out",
               "peer_payload": "mbox:peer_payload:%d" % site,
               "n_elems": N_EMBD, "site": site},
              est=N_EMBD * 4, reads={"proj_out"}, writes=set())
    PROG[ipush]["dbg_node"] = anchor
    PROG[ipush]["inserted"] = True
    W(ipush)
    ired = I("OP_XCHG_REDUCE", [],
             {"local_partial": "buf:proj_out",
              "my_payload": "mbox:my_payload:%d" % site,
              "out": "buf:proj_out",
              "peer_seqno": "mbox:peer_seqno:%d" % site,
              "my_seqno": "mbox:my_seqno:%d" % site,
              "n_elems": N_EMBD, "seqno": "sym:$seqno",
              "gpu_index": "sym:$gpu_index", "site": site},
             est=N_EMBD * 4, reads={"proj_out"}, writes={"proj_out"})
    PROG[ired]["dbg_node"] = anchor
    PROG[ired]["inserted"] = True
    W(ired)


# ---- per-block emitters ---------------------------------------------------

def emit_dn_block(bi, pending_add):
    rms = insts(bi, "rmsnorm", 2)
    radd = insts(bi, "residual_add", 2)
    css, = insts(bi, "conv_state_shift", 1)
    qkv, = insts(bi, "qkv_mmvq", 1)
    ccs, = insts(bi, "conv_concat_store", 1)
    ssf, = insts(bi, "ssm_state_fetch", 1)
    scs, = insts(bi, "ssm_conv_silu", 1)
    l2n, = insts(bi, "qk_l2norm", 1)
    gg, = insts(bi, "gdn_gates", 1)
    dstep, = insts(bi, "deltanet_step", 1)
    sst, = insts(bi, "ssm_state_store", 1)
    gon, = insts(bi, "gated_out_norm", 1)
    opr, = insts(bi, "out_proj_ar16", 1)
    ffn, = insts(bi, "ffn_fused", 1)

    # -- attn_norm (folds the previous block's ffn residual add)
    W(emit_rmsnorm(rms[0], rmsnorm_weight(rms[0]), "residual", "xn",
                   "proj_out" if pending_add else None, extra_nodes=pending_add))
    W(emit_quant("xn", N_EMBD, qkv[0]))

    # -- the wide antichain: qkv + z GEMVs, alpha/beta GEMVs, state prefetch
    qkv_w = NODES[mulmats(qkv)[0]]["src"][0]
    i_qkv = emit_mmvq("OP_MMVQ_Q4_0", qkv, qkv_w, N_EMBD, "mixer_out")

    z_mm = mulmats(gon)
    assert len(z_mm) == 1
    z_nodes = [z_mm[0], z_mm[0] + 1]
    assert NODES[z_mm[0] + 1]["op"] == "RESHAPE"
    i_z = emit_mmvq("OP_MMVQ_Q4_0", z_nodes, NODES[z_mm[0]]["src"][0],
                    N_EMBD, "gate_out")

    gg_mm = mulmats(gg)
    assert len(gg_mm) == 2
    gemv_nodes, gemv_i = {}, {}
    for mm in gg_mm:
        assert NODES[mm + 1]["op"] == "RESHAPE"
        wref = NODES[mm]["src"][0]
        name = weight_name(wref)
        which = "alpha" if name.endswith("ssm_alpha.weight") else "beta"
        assert which == "beta" or not name.endswith("ssm_beta.weight")
        gemv_nodes[which] = [mm, mm + 1]
        gemv_i[which] = I("OP_GEMV_F16", [mm, mm + 1],
                          {"weight": "gguf:" + name, "src": "buf:xn",
                           "src_elems": N_EMBD, "row_lo": 0,
                           "row_hi": sp(GDN_HEADS),   # row-split: this GPU's heads
                           "dst": "buf:gdn_" + which},
                          est=weight_bytes(wref) / NGPU,
                          reads={"xn"}, writes={"gdn_" + which})
    assert set(gemv_i) == {"alpha", "beta"}

    ssf_live, ssf_dead = split_live(ssf)
    drop(ssf_dead, "zero-extent ssm-state clear chain (single-sequence k=0)")
    s_get = [x for x in ssf_live if NODES[x]["op"] == "GET_ROWS"]
    assert len(s_get) == 1 and len(ssf_live) == 4
    s_leaf = resolve_leaf(NODES[s_get[0]]["src"][0])
    s_name = cache_name(s_leaf)
    assert leaf(NODES[s_get[0]]["src"][1])["cls"] == "other:rs_main_index"
    i_load = I("OP_STATE_LOAD", ssf_live,
               {"src": "cache:" + s_name, "row": "sym:$rs_row",
                "dst": "buf:gdn_state", "elems": sp(786432)},  # head-split state
               est=sp(786432) * 4, reads={s_name}, writes={"gdn_state"})
    W(i_qkv, i_z, gemv_i["alpha"], gemv_i["beta"], i_load)

    # -- gate math + conv shift/concat (mutually independent)
    gg_rest = [x for x in gg if x not in gemv_nodes["alpha"] + gemv_nodes["beta"]]
    assert [NODES[x]["op"] for x in gg_rest] == ["ADD", "UNARY", "MUL", "RESHAPE", "UNARY"]
    dt_ref = NODES[gg_rest[0]]["src"][1]
    a_ref = NODES[gg_rest[2]]["src"][1]
    i_gates = I("OP_GDN_GATES", gg_rest,
                {"alpha": "buf:gdn_alpha", "beta": "buf:gdn_beta",
                 "dt_bias": "gguf:" + weight_name(dt_ref),
                 "a": "gguf:" + weight_name(a_ref),
                 "g_dst": "buf:gdn_alpha", "beta_dst": "buf:gdn_beta",
                 "heads": sp(GDN_HEADS)},   # dt_bias/a sliced to this GPU's heads
                est=512, reads={"gdn_alpha", "gdn_beta"},
                writes={"gdn_alpha", "gdn_beta"})

    css_live, css_dead = split_live(css)
    drop(css_dead, "zero-extent conv-state clear chain (single-sequence k=0)")
    c_get = [x for x in css_live if NODES[x]["op"] == "GET_ROWS"]
    assert len(c_get) == 1 and len(css_live) == 3
    r_name = cache_name(resolve_leaf(NODES[c_get[0]]["src"][0]))
    i_conv = I("OP_CONV_SHIFT_CONCAT", css_live + ccs,
               {"conv_state": "cache:" + r_name, "row": "sym:$rs_row",
                "token_col": "buf:mixer_out", "channels": sp(CONV_CHANNELS),
                "d_conv": D_CONV, "window_dst": "buf:conv_ws",
                "state_writeback": "cache:" + r_name},
               est=2 * (D_CONV - 1) * sp(CONV_CHANNELS) * 4,
               reads={r_name, "mixer_out"}, writes={"conv_ws", r_name})
    W(i_gates, i_conv)

    # -- conv+silu -> qkv activations (reuse mixer_out), l2norm in place
    conv_w = NODES[[x for x in scs if NODES[x]["op"] == "SSM_CONV"][0]]["src"][1]
    W(I("OP_SSM_CONV_SILU", scs,
        {"window": "buf:conv_ws", "kernel": "gguf:" + weight_name(conv_w),
         "channels": sp(CONV_CHANNELS), "d_conv": D_CONV, "silu": True,
         "dst": "buf:mixer_out"},
        reads={"conv_ws"}, writes={"mixer_out"}))
    # local qkv layout halves each segment: q [0,1024) k [1024,2048) v [2048,5120)
    W(I("OP_QK_L2NORM", l2n,
        {"buf": "buf:mixer_out", "q_off": 0, "k_off": sp(2048),
         "heads": sp(16), "head_dim": GDN_HEAD_DIM, "eps": EPS},
        reads={"mixer_out"}, writes={"mixer_out"}))

    # -- the delta-rule step
    W(I("OP_GDN_STEP", dstep,
        {"qkv": "buf:mixer_out", "q_off": 0, "k_off": sp(2048), "v_off": sp(4096),
         "g": "buf:gdn_alpha", "beta": "buf:gdn_beta", "state": "buf:gdn_state",
         "dst": "buf:attn_out", "v_heads": sp(GDN_HEADS), "k_heads": sp(16),
         "head_dim": GDN_HEAD_DIM, "scale": 1.0 / math.sqrt(GDN_HEAD_DIM)},
        est=sp(786432) * 4,
        reads={"mixer_out", "gdn_alpha", "gdn_beta", "gdn_state"},
        writes={"gdn_state", "attn_out"}))

    # -- state commit + gated output norm (mutually independent)
    st_view = [x for x in sst if NODES[x]["op"] == "VIEW"
               and NODES[x]["src"][0].startswith("l")]
    assert len(st_view) == 1
    assert cache_name(NODES[st_view[0]]["src"][0]) == s_name
    i_store = I("OP_STATE_STORE", sst,
                {"src": "buf:gdn_state", "dst": "cache:" + s_name,
                 "row": "sym:$rs_row", "elems": sp(786432)},
                est=sp(786432) * 4, reads={"gdn_state"}, writes={s_name})
    gon_rest = [x for x in gon if x not in z_nodes]
    assert len(gon_rest) == 6
    i_gated = I("OP_GATED_RMSNORM", gon_rest,
                {"src": "buf:attn_out", "gate": "buf:gate_out",
                 "weight": "gguf:" + rmsnorm_weight(gon_rest),
                 "dst": "buf:attn_out", "heads": sp(GDN_HEADS),
                 "head_dim": GDN_HEAD_DIM, "eps": EPS},
                est=49152, reads={"attn_out", "gate_out"}, writes={"attn_out"})
    W(i_gated, i_store)

    # -- output projection (AR16, col-split -> cross-GPU reduce) and FFN
    W(emit_quant("attn_out", sp(6144), opr[0]))
    i_op = emit_mmvq("OP_MMVQ_AR16", opr, NODES[mulmats(opr)[0]]["src"][0],
                     6144, "proj_out", axis="col")
    W(i_op)
    emit_col_reduce(PROG[i_op]["dbg_node"])
    W(emit_rmsnorm(rms[1], rmsnorm_weight(rms[1]), "residual", "xn",
                   "proj_out", extra_nodes=radd[0]))
    emit_ffn(ffn)
    return radd[1]   # the ffn residual add folds into the next norm


def emit_ffn(ffn):
    glu = [x for x in ffn if NODES[x]["op"] == "GLU"]
    assert len(glu) == 1
    glu = glu[0]
    gate_mm = int(NODES[glu]["src"][0][1:])
    up_mm = int(NODES[glu]["src"][1][1:])
    down_mm = [x for x in mulmats(ffn) if NODES[x]["src"][1] == "n%d" % glu]
    assert len(down_mm) == 1
    gate_w = NODES[gate_mm]["src"][0]
    up_w = NODES[up_mm]["src"][0]
    down_w = NODES[down_mm[0]]["src"][0]
    assert weight_name(gate_w).endswith("ffn_gate.weight")
    assert weight_name(up_w).endswith("ffn_up.weight")
    assert weight_name(down_w).endswith("ffn_down.weight")

    W(emit_quant("xn", N_EMBD, gate_mm))
    # up|gate: row-split (this GPU's half of the 17408 intermediate)
    W(I("OP_MMVQ_Q4_0_FUSED", [gate_mm, up_mm, glu],
        {"weight_gate": "gguf:" + weight_name(gate_w),
         "weight_up": "gguf:" + weight_name(up_w), "glu": "swiglu",
         "src": "buf:q8_act", "src_elems": N_EMBD, "dst": "buf:ffn_ws",
         "row_lo": 0, "row_hi": sp(N_FF), "split_axis": "row"},
        est=(weight_bytes(gate_w) + weight_bytes(up_w)) / NGPU,
        reads={"q8_act"}, writes={"ffn_ws"}))
    # ffn_down: col-split (this GPU's half of the input) -> cross-GPU reduce
    W(emit_quant("ffn_ws", sp(N_FF), down_mm[0]))
    i_dn = emit_mmvq("OP_MMVQ_Q4_0", down_mm, down_w, N_FF, "proj_out", axis="col")
    W(i_dn)
    emit_col_reduce(PROG[i_dn]["dbg_node"])


ROPE_PARAMS = {"n_dims": 64, "mode": "IMROPE", "sections": [11, 11, 10, 0],
               "freq_base": 1e7, "freq_scale": 1.0, "ext_factor": 0.0,
               "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
               "n_ctx_orig": 262144}


def emit_attn_block(bi, pending_add):
    rms = insts(bi, "rmsnorm", 2)
    radd = insts(bi, "residual_add", 2)
    qg, = insts(bi, "q_gemv_norm_rope", 1)
    vg, = insts(bi, "v_gemv", 1)
    kg, = insts(bi, "k_gemv_norm_rope", 1)
    kva, = insts(bi, "kv_append", 1)
    fprep, = insts(bi, "fattn_prep", 1)
    fdec, = insts(bi, "fattn_decode", 1)
    ago, = insts(bi, "attn_gate_out", 1)
    og, = insts(bi, "o_gemv", 1)
    ffn, = insts(bi, "ffn_fused", 1)

    W(emit_rmsnorm(rms[0], rmsnorm_weight(rms[0]), "residual", "xn",
                   "proj_out" if pending_add else None, extra_nodes=pending_add))
    W(emit_quant("xn", N_EMBD, qg[0]))

    # -- the q|gate / k / v GEMV antichain
    q_mm = mulmats(qg)
    assert len(q_mm) == 1
    i_q = emit_mmvq("OP_MMVQ_Q4_0", q_mm, NODES[q_mm[0]]["src"][0],
                    N_EMBD, "mixer_out")
    k_mm = mulmats(kg)
    assert len(k_mm) == 1
    k_nodes = [k_mm[0], k_mm[0] + 1]
    assert NODES[k_mm[0] + 1]["op"] == "RESHAPE"
    i_k = emit_mmvq("OP_MMVQ_Q4_0", k_nodes, NODES[k_mm[0]]["src"][0],
                    N_EMBD, "k_stage")
    i_v = emit_mmvq("OP_MMVQ_Q4_0", vg, NODES[mulmats(vg)[0]]["src"][0],
                    N_EMBD, "v_stage")
    W(i_q, i_k, i_v)

    # -- fused q,k per-head rmsnorm + IMROPE
    q_rest = [x for x in qg if x not in q_mm]
    k_rest = [x for x in kg if x not in k_nodes]
    assert [NODES[x]["op"] for x in q_rest] == ["VIEW", "RMS_NORM", "MUL", "ROPE"]
    assert [NODES[x]["op"] for x in k_rest] == ["RMS_NORM", "MUL", "ROPE"]
    for x in q_rest + k_rest:
        if NODES[x]["op"] == "ROPE":
            assert leaf(NODES[x]["src"][1])["cls"] == "positions"
    W(I("OP_QK_NORM_ROPE", q_rest + k_rest,
        {"q_src": "buf:mixer_out", "q_off": 0, "q_head_stride": 512,
         "q_heads": sp(ATTN_Q_HEADS),
         "q_norm_weight": "gguf:" + rmsnorm_weight(q_rest),
         "q_dst": "buf:fattn_q",
         "k_src": "buf:k_stage", "k_heads": sp(ATTN_KV_HEADS),
         "k_norm_weight": "gguf:" + rmsnorm_weight(k_rest),
         "k_dst": "buf:k_stage",
         "head_dim": ATTN_HEAD_DIM, "eps": EPS,
         "positions": "cell:positions", "rope": ROPE_PARAMS},
        reads={"mixer_out", "k_stage"}, writes={"fattn_q", "k_stage"}))

    # -- KV append
    setrows = [x for x in kva if NODES[x]["op"] == "SET_ROWS"]
    assert len(setrows) == 2
    caches = {}
    for x in setrows:
        idx_cls = leaf(NODES[x]["src"][1])["cls"]
        which = "k" if idx_cls == "other:k_idxs" else "v"
        assert idx_cls == "other:%s_idxs" % which
        caches[which] = cache_name(NODES[x]["src"][2])
    assert set(caches) == {"k", "v"}
    W(I("OP_KV_APPEND", kva,
        {"k_src": "buf:k_stage", "v_src": "buf:v_stage",
         "cache_k": "cache:" + caches["k"], "cache_v": "cache:" + caches["v"],
         "row": "sym:$kv_row",
         "row_width": sp(ATTN_KV_HEADS) * ATTN_HEAD_DIM},  # head-split KV row
        reads={"k_stage", "v_stage"}, writes={caches["k"], caches["v"]}))

    # -- flash attention: prep views fold into the decode args
    fa = fdec[0]
    assert NODES[fa]["op"] == "FLASH_ATTN_EXT"
    assert cache_name(resolve_leaf(NODES[fa]["src"][1])) == caches["k"]
    assert cache_name(resolve_leaf(NODES[fa]["src"][2])) == caches["v"]
    W(I("OP_FATTN_DECODE", fprep + fdec,
        {"q": "buf:fattn_q", "cache_k": "cache:" + caches["k"],
         "cache_v": "cache:" + caches["v"], "mask": "cell:mask_f16",
         "n_kv": "cell:n_kv", "q_heads": sp(ATTN_Q_HEADS),
         "kv_heads": sp(ATTN_KV_HEADS),
         "gqa": sp(ATTN_Q_HEADS) // sp(ATTN_KV_HEADS),
         "head_dim": ATTN_HEAD_DIM, "scale": 0.0625, "prec": "f32",
         "partials": "buf:fattn_partial",
         "partial_layout": "[split][q_head][256 vkq | max | sumexp]"},
        est=64 << 20,
        reads={"fattn_q", caches["k"], caches["v"], "mask_f16", "n_kv"},
        writes={"fattn_partial"}))
    i_red = I("OP_FATTN_REDUCE", [],
              {"partials": "buf:fattn_partial", "n_splits": GRID,
               "q_heads": sp(ATTN_Q_HEADS), "head_dim": ATTN_HEAD_DIM,
               "dst": "buf:attn_out"},
              reads={"fattn_partial"}, writes={"attn_out"})
    PROG[i_red]["dbg_node"] = fa
    PROG[i_red]["inserted"] = True
    W(i_red)

    # -- sigmoid(gate) * attn, o-projection (col-split -> reduce), FFN
    W(I("OP_ATTN_GATE", ago,
        {"attn": "buf:attn_out", "gate_src": "buf:mixer_out", "gate_off": 256,
         "gate_head_stride": 512, "heads": sp(ATTN_Q_HEADS),
         "head_dim": ATTN_HEAD_DIM, "dst": "buf:attn_out"},
        reads={"attn_out", "mixer_out"}, writes={"attn_out"}))
    W(emit_quant("attn_out", sp(6144), og[0]))
    i_o = emit_mmvq("OP_MMVQ_Q4_0", og, NODES[mulmats(og)[0]]["src"][0],
                    6144, "proj_out", axis="col")
    W(i_o)
    emit_col_reduce(PROG[i_o]["dbg_node"])
    W(emit_rmsnorm(rms[1], rmsnorm_weight(rms[1]), "residual", "xn",
                   "proj_out", extra_nodes=radd[0]))
    emit_ffn(ffn)
    return radd[1]


# ---------------------------------------------------------------- main build
# prologue
emb, = insts("pre", "embed_lookup", 1)
tok_w = NODES[emb[0]]["src"][0]
W(I("OP_EMBED_LOOKUP", emb,
    {"weight": "gguf:" + weight_name(tok_w), "token": "cell:token",
     "dst": "buf:residual", "n_embd": N_EMBD},
    reads={"token"}, writes={"residual"}))

rsi, = insts("pre", "rs_index_setup", 1)
drop(rsi, "dead rs_index_setup views; outputs unreferenced in-graph "
          "(QUESTIONS item 15; single-sequence k=0 binding)")
mc, = insts("3", "mask_convert", 1)
drop(mc, "mask f32->f16 conversion moved host-side: the harness/backend "
         "writes the f16 mask (cell:mask_f16) directly")

# trunk
pending = []
for bi in range(64):
    kinds = BLOCK_INSTS.get(str(bi), {})
    if "fattn_decode" in kinds:
        pending = emit_attn_block(str(bi), pending)
    else:
        pending = emit_dn_block(str(bi), pending)

# epilogue: the final ffn residual add and the batch-1-identity out-row select
# both fold into the final OP_RMSNORM.
lrs, = insts("post", "logits_row_select", 1)
assert leaf(NODES[lrs[0]]["src"][1])["cls"] == "other:out_row_index"
frms, = insts("post", "rmsnorm", 1)
W(emit_rmsnorm(frms, rmsnorm_weight(frms), "residual", "xn", "proj_out",
               extra_nodes=pending + lrs,
               extra={"row_select": "sym:$out_row",
                      "note": "row_select folds the epilogue GET_ROWS "
                              "(batch-1: always row 0 of a 1-row trunk)"}))
hg, = insts("post", "head_gemv", 1)
head_w = NODES[hg[0]]["src"][0]
# lm_head is row-split (vocab-split): each GPU produces its half of the 248320
# logits; the harness concatenates the two halves for the full vector.
W(I("OP_HEAD_GEMV_F16", hg,
    {"weight": "gguf:" + weight_name(head_w), "src": "buf:xn",
     "row_lo": 0, "row_hi": sp(N_VOCAB), "split_axis": "row",
     "dst": "buf:logits"},
    est=weight_bytes(head_w) / NGPU, reads={"xn"}, writes={"logits"}))
i_emit = I("OP_LOGITS_EMIT", [],
           {"src": "buf:logits", "elems": sp(N_VOCAB),
            "dst": "cell:result_output", "flag": "cell:done_flag"},
           reads={"logits"}, writes={"result_output", "done_flag"})
PROG[i_emit]["dbg_node"] = hg[0]
PROG[i_emit]["inserted"] = True
W(i_emit)

# ---------------------------------------------------------------- coverage
emitted_nodes = len(COVERED)
dropped_nodes = len(DROPPED)
if emitted_nodes + dropped_nodes != N_NODES:
    missing = sorted(set(range(N_NODES)) - set(COVERED) - set(DROPPED))
    fail("coverage: %d emitted + %d dropped != %d; unaccounted: %s"
         % (emitted_nodes, dropped_nodes, N_NODES, missing[:20]))

# every weight leaf must be claimed exactly once by an instruction operand
all_weights = sorted(leaf("l%d" % k)["cls"].split(":", 1)[1]
                     for k in LEAVES if LEAVES[k]["cls"].startswith("weight:"))
if sorted(WEIGHTS_USED) != all_weights:
    used = {}
    for w in WEIGHTS_USED:
        used[w] = used.get(w, 0) + 1
    dup = [w for w, c in used.items() if c > 1]
    miss = [w for w in all_weights if w not in used]
    fail("weight operand mismatch; dup=%s missing=%s" % (dup[:5], miss[:5]))

# ---------------------------------------------------------------- boundary coalescing
# v0 emitted one boundary after every window; most windows are single ops in a
# strictly dependent chain, but a fraction of ADJACENT windows are provably
# independent and can share one boundary window. Two adjacent windows may merge
# iff their union stays a valid antichain: no instruction in one reads or writes
# a buffer the other writes (the same WAR/WAW/RAW test the lifetime check uses,
# lifted to the union of each window's reads/writes). Every block runs a merged
# window's instructions concurrently on disjoint block ranges, so a missed
# hazard is a silent wrong answer -- the union test below is the guard, and the
# lifetime check re-runs on the merged windows.
#
# Occupancy guard: split_grid divides the 72 blocks across a window's ops by
# est, so merging a heavy (weight-streaming) op with light glue keeps the heavy
# op on nearly all blocks, but merging two heavy ops would halve each one's SMs.
# A merge is refused when it would place two heavy ops (est >= HEAVY_EST) in one
# window. Pre-existing multi-heavy windows are never split -- coalescing only
# removes boundaries, never adds them.
#
# Cross-GPU exchange sites (OP_XCHG_PUSH / OP_XCHG_REDUCE) are excluded from
# coalescing entirely: their boundary is a cross-GPU handshake the local
# reads/writes sets do not model, so it must never be dissolved.
HEAVY_EST = 512 * 1024
COALESCE = os.environ.get("MK_NO_COALESCE") != "1"
XCHG_KINDS = {"OP_XCHG_PUSH", "OP_XCHG_REDUCE"}


def _win_rw(w):
    r, wr = set(), set()
    for i in w:
        r |= PROG[i]["_r"]
        wr |= PROG[i]["_w"]
    return r, wr


def _independent(wa, wb):
    ra, wra = _win_rw(wa)
    rb, wrb = _win_rw(wb)
    # hazard if either window's writes touch the other's reads or writes
    return not (wra & (rb | wrb)) and not (wrb & ra)


def _n_heavy(w):
    return sum(1 for i in w if PROG[i]["_est"] >= HEAVY_EST)


def _has_xchg(w):
    return any(PROG[i]["kind"] in XCHG_KINDS for i in w)


if COALESCE:
    n_before = len(WINDOWS)
    merged = []
    for w in WINDOWS:
        if (merged and not _has_xchg(merged[-1]) and not _has_xchg(w) and
                _independent(merged[-1], w) and
                _n_heavy(merged[-1]) + _n_heavy(w) <= 1):
            merged[-1] = merged[-1] + w
        else:
            merged.append(w)
    WINDOWS = merged
    print("coalesce: %d windows -> %d (removed %d boundaries)"
          % (n_before, len(WINDOWS), n_before - len(WINDOWS)), file=sys.stderr)


# ---------------------------------------------------------------- block ranges
def split_grid(ests):
    n = len(ests)
    assert n <= GRID
    total = float(sum(ests))
    sh = [max(1, int(GRID * b / total)) for b in ests]
    order = sorted(range(n), key=lambda i: -ests[i])
    i = 0
    while sum(sh) != GRID:
        j = order[i % n]
        if sum(sh) < GRID:
            sh[j] += 1
        elif sh[j] > 1:
            sh[j] -= 1
        i += 1
    return sh


for w in WINDOWS:
    if len(w) == 1:
        PROG[w[0]]["block_lo"], PROG[w[0]]["block_hi"] = 0, GRID
        continue
    lo = 0
    for idx, share in zip(w, split_grid([PROG[i]["_est"] for i in w])):
        PROG[idx]["block_lo"], PROG[idx]["block_hi"] = lo, lo + share
        lo += share
    assert lo == GRID

# ---------------------------------------------------------------- lifetime check
hazards = 0
for wi, w in enumerate(WINDOWS):
    for a in range(len(w)):
        for b in range(len(w)):
            if a == b:
                continue
            ia, ib = PROG[w[a]], PROG[w[b]]
            clash = ia["_w"] & (ib["_r"] | ib["_w"])
            if clash:
                hazards += 1
                print("HAZARD window %d: %s writes %s used by %s"
                      % (wi, ia["kind"], sorted(clash), ib["kind"]))
if hazards:
    fail("%d intra-window lifetime hazards" % hazards)

# ---------------------------------------------------------------- accounting
n_boundaries = len(WINDOWS)
n_compute = len(PROG)
n_instr = n_compute + n_boundaries

weight_stream = sum(weight_bytes("l%d" % k) for k in LEAVES
                    if LEAVES[k]["cls"].startswith("weight:")
                    and LEAVES[k]["cls"] != "weight:token_embd.weight")
embd_row = N_EMBD * 2                       # one f16 embedding row
state_rw = 48 * (786432 * 4 * 2) + 48 * (2 * (D_CONV - 1) * CONV_CHANNELS * 4)
kv_append_w = 16 * 2 * 1024 * 2
logits_rw = N_VOCAB * 4 * 3                 # head write + emit copy R+W
fixed = weight_stream + embd_row + state_rw + kv_append_w + logits_rw
kv_coef = 16 * 2 * 1024 * 2 + 16 * 2       # split-KV read + mask read, per n_kv


def dram(n_kv):
    return fixed + kv_coef * n_kv


# ---------------------------------------------------------------- serialize
def serialize(ins):
    out = {"kind": ins["kind"], "block_lo": ins["block_lo"],
           "block_hi": ins["block_hi"], "dbg_node": ins["dbg_node"],
           "nodes": ins["nodes"], "args": ins["args"]}
    if ins.get("inserted"):
        out["inserted"] = True
    return out


instructions = []
for w in WINDOWS:
    instructions.extend(serialize(PROG[i]) for i in w)
    instructions.append({"kind": "OP_BOUNDARY", "block_lo": 0, "block_hi": GRID,
                         "dbg_node": None, "nodes": [], "args": {}})
assert len(instructions) == n_instr

buffers = [{"name": nm, "type": ty, "elems": elems,
            "bytes": Q8_BYTES if ty == "q8_1" else elems * 4, "purpose": why}
           for nm, ty, elems, why in BUFFERS]

drop_summary = {}
for node, why in DROPPED.items():
    drop_summary.setdefault(why, []).append(node)

program = {
    "meta": {
        "source": "k0/schedule.csv + k0/leaves.csv (3704 nodes, 990 leaves); "
                  "compiled by k0/compile_schedule.py",
        "isa": "core/isa.cuh MacroKind; grid 72 blocks",
        "split": ("dual-GPU tensor-parallel: NGPU=2, one program packed per "
                  "GPU. Expand GEMVs (qkv/up|gate/lm_head) row-split (half "
                  "output rows); contract GEMVs (attn_output/ffn_down/ssm_out) "
                  "col-split (half input columns) + OP_XCHG_PUSH/BOUNDARY/"
                  "OP_XCHG_REDUCE fold (%d sites); KV/state head-split; "
                  "residual/norms/rope/gates/embed mirrored"
                  % XCHG_SITE[0]) if SPLIT else "single GPU",
        "binding": {
            "gpu": ("dual GPU: this GPU computes its half of every split "
                    "dimension; the harness slices each weight to this GPU's "
                    "row/column/head range and patches $gpu_index + peer "
                    "mailbox pointers at pack time") if SPLIT else
                   ("single GPU: all GEMV row ranges are FULL [0, nrows); the "
                    "dual-GPU compile parameterizes rows by the meta split"),
            "sequence": "single sequence (k=0): zero-extent clear chains and "
                        "the dead rs_index_setup views are dropped "
                        "(SCHEDULE-QUESTIONS items 14/15, a design decision "
                        "the capture alone does not prove safe)",
            "mask": "the f32->f16 mask conversion (n219) is host-side: the "
                    "harness writes cell:mask_f16 (0 / -inf, width n_kv) "
                    "directly; the f32 mask leaf l65 is not materialized",
            "q_heads": "24 q-heads x 4 kv-heads, head_dim 256 (measured, "
                       "QUESTIONS item 1; the briefed 32 is wrong)",
        },
        "mapping": {
            "embed_lookup": "OP_EMBED_LOOKUP",
            "rmsnorm": "OP_RMSNORM (fused weight mul; residual_add folds in "
                       "as the optional add-src, sum written back to residual)",
            "residual_add": "folded into the following OP_RMSNORM (add_src); "
                            "no OP_RESIDUAL_ADD instructions are emitted",
            "conv_state_shift + conv_concat_store": "OP_CONV_SHIFT_CONCAT "
                            "(live nodes; zero-extent clears dropped)",
            "qkv_mmvq": "OP_MMVQ_Q4_0",
            "ssm_state_fetch": "OP_STATE_LOAD (live nodes)",
            "ssm_conv_silu": "OP_SSM_CONV_SILU",
            "qk_l2norm": "OP_QK_L2NORM",
            "gdn_gates": "2x OP_GEMV_F16 (ssm_alpha, ssm_beta) + OP_GDN_GATES",
            "deltanet_step": "OP_GDN_STEP",
            "ssm_state_store": "OP_STATE_STORE",
            "gated_out_norm": "OP_MMVQ_Q4_0 (attn_gate z GEMV) + OP_GATED_RMSNORM",
            "out_proj_ar16": "OP_MMVQ_AR16",
            "ffn_fused": "OP_MMVQ_Q4_0_FUSED (gate|up + swiglu) + OP_MMVQ_Q4_0 "
                         "(ffn_down)",
            "q_gemv_norm_rope": "OP_MMVQ_Q4_0 (q|gate GEMV) + OP_QK_NORM_ROPE "
                                "(q part)",
            "k_gemv_norm_rope": "OP_MMVQ_Q4_0 (k GEMV) + OP_QK_NORM_ROPE (k part)",
            "v_gemv": "OP_MMVQ_Q4_0",
            "kv_append": "OP_KV_APPEND (K and V SET_ROWS in one instruction)",
            "fattn_prep": "folded into OP_FATTN_DECODE args (views/permutes "
                          "become strides)",
            "fattn_decode": "OP_FATTN_DECODE (+ inserted OP_FATTN_REDUCE for "
                            "the split-KV merge)",
            "attn_gate_out": "OP_ATTN_GATE",
            "o_gemv": "OP_MMVQ_Q4_0",
            "logits_row_select": "folded into the epilogue OP_RMSNORM source "
                                 "select (batch-1 identity), NOT the head GEMV: "
                                 "the row select precedes the final norm",
            "head_gemv": "OP_HEAD_GEMV_F16 + inserted OP_LOGITS_EMIT",
            "mask_convert": "dropped (host-side, see binding.mask)",
            "rs_index_setup": "dropped (dead in-graph)",
            "inserted": "OP_QUANT_Q8_1 (before every quantized-weight GEMV "
                        "group), OP_FATTN_REDUCE, OP_LOGITS_EMIT carry no "
                        "source nodes; dbg_node points at the anchor node",
        },
        "namespaces": {
            "gguf:": "weight tensor by GGUF name",
            "cache:": "per-layer cache tensor: cache_k_l<i>, cache_v_l<i> "
                      "(f16 1024 x 262144), state_r_l<i> (f32 30720), "
                      "state_s_l<i> (f32 786432)",
            "buf:": "scratch buffer from the buffers table",
            "cell:": "host-written input / host-read output cell",
            "sym:$": "runtime scalar the host patches into the instruction "
                     "payload per pass",
        },
        "runtime_symbols": {
            "$kv_row": "destination cache row for this token's KV append "
                       "(i64; one value, shared by all 16 attention layers)",
            "$rs_row": "recurrent-state row for the active sequence (i32; "
                       "constant under the single-sequence binding but kept "
                       "symbolic)",
            "$out_row": "epilogue row select; always 0 at batch-1",
            "$seqno": "(--split) monotonic per-site exchange seqno = pass "
                      "number; host patches each OP_XCHG_REDUCE payload/pass",
            "$gpu_index": "(--split) 0 or 1; selects the fixed p0+p1 fold "
                          "order; pack-time constant per GPU",
        },
        "input_cells": {
            "token": "i32[1] token id",
            "positions": "i32[4] M-RoPE position ids for this token",
            "n_kv": "u32[1] current KV window length, padded to a multiple of "
                    "256; host-written once per pass, read STRONG (.cg) by "
                    "OP_FATTN_DECODE so a deep-context change is never a stale "
                    "cached-payload read",
            "mask_f16": "f16[n_kv] KQ mask, 0 attend / -inf masked, "
                        "host-written",
            "result_output": "f32[248320] logits out (host-visible)",
            "done_flag": "host completion flag set by OP_LOGITS_EMIT",
        },
        "coverage": {"nodes": N_NODES, "emitted_nodes": emitted_nodes,
                     "dropped_nodes": dropped_nodes},
        "drops": [{"reason": why, "count": len(nodes), "nodes": sorted(nodes)}
                  for why, nodes in sorted(drop_summary.items())],
        "per_pass": {
            "instructions_total": n_instr,
            "instructions_compute": n_compute,
            "boundaries_per_pass": n_boundaries,
            "epoch_stride": n_boundaries,
            "g15_budget": 500,
            "dram_bytes": {
                "formula": "%d + %d * n_kv" % (fixed, kv_coef),
                "weight_stream": weight_stream + embd_row,
                "state_rw": state_rw,
                "kv_append_write": kv_append_w,
                "logits_rw": logits_rw,
                "per_n_kv": kv_coef,
                "at_n_kv_33024": dram(33024),
                "at_n_kv_262144": dram(262144),
                "note": "scratch/activation traffic excluded (L2-resident, "
                        "< 10 MB/pass); weights counted once per pass at "
                        "logical (unpadded) size",
            },
        },
    },
    "buffers": buffers,
    "instructions": instructions,
}

out_path = os.path.join(HERE, "program-split.json" if SPLIT else "program.json")
with open(out_path, "w") as f:
    json.dump(program, f, indent=1)

# ---------------------------------------------------------------- report
kind_counts = {}
for ins in PROG:
    kind_counts[ins["kind"]] = kind_counts.get(ins["kind"], 0) + 1

print("coverage: emitted %d + dropped %d = %d (OK)"
      % (emitted_nodes, dropped_nodes, N_NODES))
print("instructions: %d compute + %d boundaries = %d total (%.0f KiB program)"
      % (n_compute, n_boundaries, n_instr, n_instr * 128 / 1024))
print("boundaries per pass: %d (G15 budget ~500)" % n_boundaries)
print("lifetime check: no intra-window hazards across %d windows" % len(WINDOWS))
print()
print("%-22s %6s" % ("kind", "count"))
for k in sorted(kind_counts, key=lambda k: -kind_counts[k]):
    print("%-22s %6d" % (k, kind_counts[k]))
print()
print("%-14s %6s %10s  %s" % ("buffer", "type", "bytes", "purpose"))
total_b = 0
for b in buffers:
    total_b += b["bytes"]
    print("%-14s %6s %10d  %s" % (b["name"], b["type"], b["bytes"],
                                  b["purpose"].split(";")[0]))
print("%-14s %6s %10d" % ("TOTAL", "", total_b))
print()
print("weight stream/pass: %.3f GB (+ one %d-B embedding row)"
      % (weight_stream / 1e9, embd_row))
print("DRAM/pass: %.3f GB at n_kv=33024, %.3f GB at n_kv=262144"
      % (dram(33024) / 1e9, dram(262144) / 1e9))
print("wrote %s" % out_path)
