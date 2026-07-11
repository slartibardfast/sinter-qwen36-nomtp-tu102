#!/usr/bin/env python3
"""parse.py - decode the LG0 cgraph fingerprint dump into schedule.csv + leaves.csv.

Input : a step file from plan/0135 capture/lg0-fingerprint/out262k/ (record format
        defined by fingerprint.cpp: n<i>|op=..|type=..|ne=..|par=<hex>|src=..,
        leaves l<k>|type=..|ne=.. emitted inline before first use).
Output: schedule.csv (one row per node, all 3704) and leaves.csv (one row per
        leaf, all 990) in the directory of this script.

op_params decoding follows ggml (llama.cpp autoround @546eca8dc, ggml/src/ggml.c):
  VIEW            u64 byte offset at params[0]
  SCALE           f32 scale, f32 bias
  RMS_NORM/L2_NORM f32 eps
  CONCAT          i32 dim
  UNARY           i32 ggml_unary_op
  GLU             i32 ggml_glu_op, i32 swapped, f32 alpha, f32 limit
  PERMUTE         i32 axes[4]
  ROPE            i32 n_past,n_dims,mode,n_ctx,n_ctx_orig; f32 freq_base,
                  freq_scale,ext_factor,attn_factor,beta_fast,beta_slow;
                  i32 sections[4]
  FLASH_ATTN_EXT  f32 scale,max_bias,logit_softcap; i32 prec at [3]
  others          no params (all-zero par asserted)
"""

import csv
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE = "/home/dconnolly/yarn-agentic/plan/0135-persistent-decode-megakernel/capture"
DEFAULT_IN = os.path.join(CAPTURE, "lg0-fingerprint/out262k/ctx33000-step0.txt")
GGUF_CSV = os.path.join(CAPTURE, "gguf-tensors.csv")

UNARY_NAMES = ["abs","sgn","neg","step","tanh","elu","relu","sigmoid","gelu",
               "gelu_quick","silu","hardswish","hardsigmoid","exp","expm1",
               "softplus","gelu_erf","xielu","floor","ceil","round","trunc"]
GLU_NAMES = ["reglu","geglu","swiglu","swiglu_oai","geglu_erf","geglu_quick"]
ROPE_MODES = {0: "NORMAL", 2: "NEOX", 8: "MROPE", 24: "VISION", 40: "IMROPE"}


def f32(b):
    v = struct.unpack("<f", b)[0]
    return format(v, ".10g")


def i32(b):
    return struct.unpack("<i", b)[0]


def decode_params(op, parhex):
    p = bytes.fromhex(parhex)
    if op == "VIEW":
        return "offset=%d" % struct.unpack("<Q", p[:8])[0]
    if op == "SCALE":
        return "scale=%s bias=%s" % (f32(p[0:4]), f32(p[4:8]))
    if op in ("RMS_NORM", "L2_NORM"):
        return "eps=%s" % f32(p[0:4])
    if op == "CONCAT":
        return "dim=%d" % i32(p[0:4])
    if op == "UNARY":
        k = i32(p[0:4])
        return "unary=%s(%d)" % (UNARY_NAMES[k] if k < len(UNARY_NAMES) else "?", k)
    if op == "GLU":
        k = i32(p[0:4])
        return "glu=%s(%d) swapped=%d alpha=%s limit=%s" % (
            GLU_NAMES[k] if k < len(GLU_NAMES) else "?", k,
            i32(p[4:8]), f32(p[8:12]), f32(p[12:16]))
    if op == "PERMUTE":
        return "axes=[%d %d %d %d]" % tuple(i32(p[4*i:4*i+4]) for i in range(4))
    if op == "ROPE":
        ints = [i32(p[4*i:4*i+4]) for i in range(5)]
        flts = [f32(p[4*i:4*i+4]) for i in range(5, 11)]
        secs = [i32(p[4*i:4*i+4]) for i in range(11, 15)]
        mode = ints[2]
        return ("n_dims=%d mode=%s(%d) n_ctx_orig=%d freq_base=%s freq_scale=%s "
                "ext_factor=%s attn_factor=%s beta_fast=%s beta_slow=%s "
                "sections=[%d %d %d %d]") % (
                    ints[1], ROPE_MODES.get(mode, "?"), mode, ints[4],
                    flts[0], flts[1], flts[2], flts[3], flts[4], flts[5], *secs)
    if op == "FLASH_ATTN_EXT":
        return "scale=%s max_bias=%s logit_softcap=%s prec=%d" % (
            f32(p[0:4]), f32(p[4:8]), f32(p[8:12]), i32(p[12:16]))
    # every other op in this graph carries no parameters
    assert p == bytes(64), "unexpected op_params on %s: %s" % (op, parhex)
    return ""


NODE_RE = re.compile(r"n(\d+)\|op=([A-Z_0-9]+)\|type=(\w+)\|ne=([\d,-]+)\|par=([0-9a-f]{128})\|src=(.*)$")
LEAF_RE = re.compile(r"l(\d+)\|type=(\w+)\|ne=([\d,-]+)$")


def load(path):
    nodes, leaves = {}, {}
    for line in open(path):
        line = line.rstrip("\n")
        m = NODE_RE.match(line)
        if m:
            i = int(m.group(1))
            nodes[i] = {
                "op": m.group(2), "type": m.group(3),
                "ne": [int(x) for x in m.group(4).split(",")],
                "par": m.group(5),
                "src": m.group(6).split(",") if m.group(6) else [],
            }
            continue
        m = LEAF_RE.match(line)
        assert m, "unparsed line: " + line
        leaves[int(m.group(1))] = {
            "type": m.group(2), "ne": [int(x) for x in m.group(3).split(",")]}
    return nodes, leaves


# ---------------------------------------------------------------- archetypes
# (op, macro_kind, instance_tag): consecutive rows with the same
# (kind, tag) inside one block belong to one macro instance, even when the
# rows are not adjacent (ssm_state_fetch has a late RESHAPE,RESHAPE pair).
DN_ARCH = [
    ("RMS_NORM", "rmsnorm", "attn_norm"), ("MUL", "rmsnorm", "attn_norm"),
    ("RESHAPE", "conv_state_shift", ""), ("VIEW", "conv_state_shift", ""),
    ("SCALE", "conv_state_shift", ""), ("GET_ROWS", "conv_state_shift", ""),
    ("GET_ROWS", "conv_state_shift", ""), ("VIEW", "conv_state_shift", ""),
    ("CPY", "conv_state_shift", ""), ("RESHAPE", "conv_state_shift", ""),
    ("MUL_MAT", "qkv_mmvq", ""), ("RESHAPE", "qkv_mmvq", ""),
    ("TRANSPOSE", "qkv_mmvq", ""),
    ("CONCAT", "conv_concat_store", ""), ("VIEW", "conv_concat_store", ""),
    ("VIEW", "conv_concat_store", ""), ("CPY", "conv_concat_store", ""),
    ("RESHAPE", "ssm_state_fetch", ""), ("VIEW", "ssm_state_fetch", ""),
    ("SCALE", "ssm_state_fetch", ""), ("GET_ROWS", "ssm_state_fetch", ""),
    ("GET_ROWS", "ssm_state_fetch", ""), ("VIEW", "ssm_state_fetch", ""),
    ("CPY", "ssm_state_fetch", ""),
    ("SSM_CONV", "ssm_conv_silu", ""), ("UNARY", "ssm_conv_silu", ""),
    ("VIEW", "qk_l2norm", ""), ("L2_NORM", "qk_l2norm", ""),
    ("VIEW", "qk_l2norm", ""), ("L2_NORM", "qk_l2norm", ""),
    ("VIEW", "qk_l2norm", ""),
    ("MUL_MAT", "gdn_gates", ""), ("RESHAPE", "gdn_gates", ""),
    ("ADD", "gdn_gates", ""), ("UNARY", "gdn_gates", ""),
    ("MUL", "gdn_gates", ""), ("RESHAPE", "gdn_gates", ""),
    ("MUL_MAT", "gdn_gates", ""), ("RESHAPE", "gdn_gates", ""),
    ("UNARY", "gdn_gates", ""),
    ("RESHAPE", "ssm_state_fetch", ""), ("RESHAPE", "ssm_state_fetch", ""),
    ("GATED_DELTA_NET", "deltanet_step", ""),
    ("VIEW", "ssm_state_store", ""), ("VIEW", "ssm_state_store", ""),
    ("CPY", "ssm_state_store", ""),
    ("VIEW", "gated_out_norm", ""), ("RMS_NORM", "gated_out_norm", ""),
    ("MUL", "gated_out_norm", ""), ("MUL_MAT", "gated_out_norm", ""),
    ("RESHAPE", "gated_out_norm", ""), ("UNARY", "gated_out_norm", ""),
    ("MUL", "gated_out_norm", ""), ("RESHAPE", "gated_out_norm", ""),
    ("MUL_MAT", "out_proj_ar16", ""), ("RESHAPE", "out_proj_ar16", ""),
    ("ADD", "residual_add", "attn"),
    ("RMS_NORM", "rmsnorm", "ffn_norm"), ("MUL", "rmsnorm", "ffn_norm"),
    ("MUL_MAT", "ffn_fused", ""), ("MUL_MAT", "ffn_fused", ""),
    ("GLU", "ffn_fused", ""), ("MUL_MAT", "ffn_fused", ""),
    ("ADD", "residual_add", "ffn"),
]

ATTN_ARCH = [
    ("RMS_NORM", "rmsnorm", "attn_norm"), ("MUL", "rmsnorm", "attn_norm"),
    ("MUL_MAT", "q_gemv_norm_rope", ""), ("VIEW", "q_gemv_norm_rope", ""),
    ("RMS_NORM", "q_gemv_norm_rope", ""), ("MUL", "q_gemv_norm_rope", ""),
    ("ROPE", "q_gemv_norm_rope", ""),
    ("MUL_MAT", "v_gemv", ""), ("RESHAPE", "v_gemv", ""),
    ("MUL_MAT", "k_gemv_norm_rope", ""), ("RESHAPE", "k_gemv_norm_rope", ""),
    ("RMS_NORM", "k_gemv_norm_rope", ""), ("MUL", "k_gemv_norm_rope", ""),
    ("ROPE", "k_gemv_norm_rope", ""),
    ("VIEW", "kv_append", ""), ("SET_ROWS", "kv_append", ""),
    ("VIEW", "kv_append", ""), ("SET_ROWS", "kv_append", ""),
    ("VIEW", "fattn_prep", ""), ("PERMUTE", "fattn_prep", ""),
    ("VIEW", "fattn_prep", ""), ("PERMUTE", "fattn_prep", ""),
    ("VIEW", "fattn_prep", ""), ("PERMUTE", "fattn_prep", ""),
    # optional CPY (mask f32->f16) inserted here in the first attention block
    ("FLASH_ATTN_EXT", "fattn_decode", ""),
    ("RESHAPE", "attn_gate_out", ""), ("VIEW", "attn_gate_out", ""),
    ("CONT", "attn_gate_out", ""), ("UNARY", "attn_gate_out", ""),
    ("MUL", "attn_gate_out", ""),
    ("MUL_MAT", "o_gemv", ""),
    ("ADD", "residual_add", "attn"),
    ("RMS_NORM", "rmsnorm", "ffn_norm"), ("MUL", "rmsnorm", "ffn_norm"),
    ("MUL_MAT", "ffn_fused", ""), ("MUL_MAT", "ffn_fused", ""),
    ("GLU", "ffn_fused", ""), ("MUL_MAT", "ffn_fused", ""),
    ("ADD", "residual_add", "ffn"),
]


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
    nodes, leaves = load(in_path)
    assert len(nodes) == 3704 and len(leaves) == 990, (len(nodes), len(leaves))

    consumers = {}
    for i in sorted(nodes):
        for s in nodes[i]["src"]:
            consumers.setdefault(s, []).append(i)

    # ---- trunk walk: the residual stream. The embedding GET_ROWS output (n0)
    # crosses a scheduler split and re-enters as leaf l2.
    trunk = "l2"
    layer_bounds = []  # (attn_norm, attn_add, ffn_norm, ffn_add)
    for _ in range(64):
        c = consumers.get(trunk, [])
        norms = [i for i in c if nodes[i]["op"] == "RMS_NORM"]
        adds = [i for i in c if nodes[i]["op"] == "ADD"]
        assert len(norms) == 1 and len(adds) == 1, (trunk, c)
        an, aa = norms[0], adds[0]
        c2 = consumers.get("n%d" % aa, [])
        norms2 = [i for i in c2 if nodes[i]["op"] == "RMS_NORM"]
        adds2 = [i for i in c2 if nodes[i]["op"] == "ADD"]
        assert len(norms2) == 1 and len(adds2) == 1, (aa, c2)
        fn, fa = norms2[0], adds2[0]
        layer_bounds.append((an, aa, fn, fa))
        trunk = "n%d" % fa
    ends = [b[3] for b in layer_bounds]
    assert all(ends[i] < ends[i+1] for i in range(63))
    # epilogue: GET_ROWS(out-row select) then RMS_NORM+MUL then MUL_MAT head
    tail = consumers.get(trunk, [])
    assert [nodes[i]["op"] for i in tail] == ["GET_ROWS"]

    # ---- block index per node
    block_of = {}
    block_of[0] = "pre"           # n0 embedding GET_ROWS
    block_of[6] = "pre"           # n6/n8: views of the rs copy-index buffer,
    block_of[8] = "pre"           # outputs unreferenced in-graph (see QUESTIONS)
    prev = 0
    ranges = []
    for L, (an, aa, fn, fa) in enumerate(layer_bounds):
        lo = prev + 1
        ranges.append((lo, fa))
        for i in range(lo, fa + 1):
            if i not in block_of:
                block_of[i] = L
        prev = fa
    for i in range(ends[-1] + 1, len(nodes)):
        block_of[i] = "post"
    assert len(block_of) == len(nodes)

    # ---- layer kind
    kind_of_layer = []
    for L, (lo, hi) in enumerate(ranges):
        ops = [nodes[i]["op"] for i in range(lo, hi + 1)]
        kind_of_layer.append("attn" if "FLASH_ATTN_EXT" in ops else "delta")
    attn_layers = [L for L, k in enumerate(kind_of_layer) if k == "attn"]
    assert len(attn_layers) == 16 and kind_of_layer.count("delta") == 48

    # ---- macro assignment by archetype matching
    macro_kind, macro_id = {}, {}
    next_id = [0]

    def new_inst():
        next_id[0] += 1
        return next_id[0] - 1

    macro_kind[0], macro_id[0] = "embed_lookup", new_inst()
    idx_setup = new_inst()
    for i in (6, 8):
        macro_kind[i], macro_id[i] = "rs_index_setup", idx_setup

    mask_cpy_node = None
    for L, (lo, hi) in enumerate(ranges):
        seq = [i for i in range(lo, hi + 1) if i not in macro_kind]
        arch = DN_ARCH if kind_of_layer[L] == "delta" else ATTN_ARCH
        inst = {}
        ai = 0
        for i in seq:
            op = nodes[i]["op"]
            if (kind_of_layer[L] == "attn" and op == "CPY"
                    and arch[ai][0] == "FLASH_ATTN_EXT"):
                # the shared mask f32->f16 conversion, first attention block only
                assert mask_cpy_node is None
                mask_cpy_node = i
                macro_kind[i], macro_id[i] = "mask_convert", new_inst()
                continue
            aop, kind, tag = arch[ai]
            assert op == aop, ("block %d node n%d: %s != %s" % (L, i, op, aop))
            key = (kind, tag)
            if key not in inst:
                inst[key] = new_inst()
            macro_kind[i], macro_id[i] = kind, inst[key]
            ai += 1
        assert ai == len(arch), (L, ai, len(arch))

    post = sorted(i for i, b in block_of.items() if b == "post")
    assert [nodes[i]["op"] for i in post] == ["GET_ROWS", "RMS_NORM", "MUL", "MUL_MAT"]
    macro_kind[post[0]], macro_id[post[0]] = "logits_row_select", new_inst()
    fin = new_inst()
    for i in post[1:3]:
        macro_kind[i], macro_id[i] = "rmsnorm", fin
    macro_kind[post[3]], macro_id[post[3]] = "head_gemv", new_inst()
    assert len(macro_kind) == len(nodes)

    # every fattn reads the one converted mask
    for L in attn_layers:
        fat = [i for i in range(ranges[L][0], ranges[L][1] + 1)
               if nodes[i]["op"] == "FLASH_ATTN_EXT"][0]
        assert nodes[fat]["src"][3] == "n%d" % mask_cpy_node

    # n6/n8 outputs are never consumed
    assert "n6" not in consumers and "n8" not in consumers

    # ---- leaf classification
    first_ref = {}
    for i in sorted(nodes):
        for s in nodes[i]["src"]:
            if s.startswith("l"):
                first_ref.setdefault(int(s[1:]), i)
    assert sorted(first_ref) == sorted(leaves)

    def producer(ref):
        return nodes[int(ref[1:])] if ref.startswith("n") else None

    leaf_rows = {}

    def classify(k):
        lf = leaves[k]
        t, ne = lf["type"], lf["ne"]
        n0 = first_ref[k]
        cons = consumers["l%d" % k]
        blk = block_of[n0]
        fn = nodes[n0]
        slot = fn["src"].index("l%d" % k)

        def w(name, why):
            return ("weight:%s" % name, why)

        if t == "f16" and ne[:2] == [5120, 248320]:
            if fn["op"] == "GET_ROWS":
                return w("token_embd.weight", "f16 5120x248320 read by the embedding GET_ROWS (n0)")
            return w("output.weight", "f16 5120x248320 as MUL_MAT weight of the final logits GEMV")
        if t == "q4_0" and ne[:2] == [5120, 10240]:
            return w("blk.%d.attn_qkv.weight" % blk, "unique q4_0 5120x10240 in block")
        if t == "q4_0" and ne[:2] == [5120, 6144]:
            return w("blk.%d.attn_gate.weight" % blk, "unique q4_0 5120x6144 in block (z-gate GEMV)")
        if t == "q4_0_ar16":
            return w("blk.%d.ssm_out.weight" % blk, "unique Q4_0_AR16 6144x5120 in block")
        if t == "q4_0" and ne[:2] == [5120, 17408]:
            glu = [j for j in consumers["n%d" % cons[0]] if nodes[j]["op"] == "GLU"]
            assert len(glu) == 1
            g = nodes[glu[0]]
            which = "ffn_gate" if g["src"][0] == "n%d" % cons[0] else "ffn_up"
            return w("blk.%d.%s.weight" % (blk, which),
                     "q4_0 5120x17408; its GEMV is GLU src%d (src0=gate, src1=up)"
                     % g["src"].index("n%d" % cons[0]))
        if t == "q4_0" and ne[:2] == [17408, 5120]:
            return w("blk.%d.ffn_down.weight" % blk, "unique q4_0 17408x5120 in block")
        if t == "q4_0" and ne[:2] == [5120, 12288]:
            return w("blk.%d.attn_q.weight" % blk, "unique q4_0 5120x12288 in block (q+gate interleaved)")
        if t == "q4_0" and ne[:2] == [5120, 1024]:
            mm = cons[0]
            down = consumers["n%d" % mm]
            chain = consumers["n%d" % down[0]]
            ops = {nodes[j]["op"] for j in chain}
            if "RMS_NORM" in ops:
                return w("blk.%d.attn_k.weight" % blk, "q4_0 5120x1024 whose GEMV feeds k_norm RMS_NORM + ROPE")
            return w("blk.%d.attn_v.weight" % blk, "q4_0 5120x1024 whose GEMV goes straight to the V-cache SET_ROWS")
        if t == "q4_0" and ne[:2] == [6144, 5120]:
            return w("blk.%d.attn_output.weight" % blk, "unique q4_0 6144x5120 in block")
        if t == "f16" and ne[:2] == [5120, 48]:
            mm = cons[0]
            reshape = consumers["n%d" % mm][0]
            nxt = {nodes[j]["op"] for j in consumers["n%d" % reshape]}
            if "ADD" in nxt:
                return w("blk.%d.ssm_alpha.weight" % blk,
                         "f16 5120x48 GEMV feeding ADD(ssm_dt.bias)->softplus->MUL(ssm_a) (decay path)")
            return w("blk.%d.ssm_beta.weight" % blk, "f16 5120x48 GEMV feeding sigmoid (mixing path)")
        if t == "f32" and ne[:2] == [48, 1]:
            if fn["op"] == "ADD":
                return w("blk.%d.ssm_dt.bias" % blk, "f32[48] ADD operand on the alpha path")
            assert fn["op"] == "MUL"
            return w("blk.%d.ssm_a" % blk, "f32[48] MUL operand after softplus (-exp(A_log) factor)")
        if t == "f32" and ne[:2] == [4, 10240]:
            return w("blk.%d.ssm_conv1d.weight" % blk, "f32 4x10240 SSM_CONV kernel")
        if t == "f32" and ne[:2] == [128, 1]:
            return w("blk.%d.ssm_norm.weight" % blk, "f32[128] MUL after the per-head RMS_NORM of the GDN output")
        if t == "f32" and ne[:2] == [256, 1]:
            mul = cons[0]
            heads = nodes[mul]["ne"][1]
            which = "attn_q_norm" if heads == 24 else "attn_k_norm"
            return w("blk.%d.%s.weight" % (blk, which),
                     "f32[256] norm weight; MUL output has %d heads (24=q, 4=k)" % heads)
        if t == "f32" and ne[:2] == [5120, 1]:
            if blk == "post":
                return w("output_norm.weight", "f32[5120] MUL in the final norm")
            if k == 2:
                return ("other:embd_trunk_input",
                        "f32[5120] copy of the embedding GET_ROWS output (n0) re-entering "
                        "across a scheduler split; the residual trunk root")
            if macro_kind[cons[0]] == "rmsnorm":
                tag = "attn_norm" if nodes[cons[0]]["src"][0] == "n%d" % layer_bounds[blk][0] else "post_attention_norm"
                # first rmsnorm of the block is attn_norm, second is post_attention_norm
                an = layer_bounds[blk][0]
                fnorm = layer_bounds[blk][2]
                mulc = cons[0]
                tag = "attn_norm" if nodes[mulc]["src"][0] == "n%d" % an else "post_attention_norm"
                return w("blk.%d.%s.weight" % (blk, tag),
                         "f32[5120] MUL paired with %s RMS_NORM"
                         % ("pre-mixer" if tag == "attn_norm" else "pre-FFN"))
        if t == "f16" and ne[:2] == [1024, 262144]:
            sr = [j for j in cons if nodes[j]["op"] == "SET_ROWS"][0]
            val = producer(nodes[sr]["src"][0])   # VIEW of the appended row
            src_chain = producer(val["src"][0])
            which = "cache_k" if src_chain["op"] == "ROPE" else "cache_v"
            return ("%s(blk.%d)" % (which, blk),
                    "f16 1024x262144 KV line; SET_ROWS value %s ROPE (K is roped, V is not)"
                    % ("comes from" if which == "cache_k" else "does not come from"))
        if t == "f32" and ne[0] == 30720:
            return ("state:conv_state(blk.%d)" % blk,
                    "f32[30720]=10240x3 rolling conv window (d_conv-1 tokens x conv channels)")
        if t == "f32" and ne[0] == 786432:
            return ("state:ssm_state(blk.%d)" % blk,
                    "f32[786432]=128x128x48 GDN recurrent state")
        if t == "i32" and k == 1:
            return ("input_tokens", "i32[1] row index of the embedding GET_ROWS (n0)")
        if t == "i32" and k == 5:
            return ("other:rs_copy_indices_buf",
                    "i32 buffer viewed by n6 (main,1 elem) and n8 (clear,0 elems at +4B); "
                    "outputs unreferenced, presumed source of l6/l7 across splits")
        if t == "i32" and k == 6:
            return ("other:rs_main_index",
                    "i32[1] row id selecting the sequence's conv/ssm state (GET_ROWS src1, all 48 DN blocks)")
        if t == "i32" and k == 7:
            return ("other:rs_clear_index",
                    "i32[0] empty clear-row list; makes the paired GET_ROWS/CPY no-ops at decode")
        if t == "i32" and ne[0] == 4:
            return ("positions", "i32[4] = 4 M-RoPE position ids per token (ROPE src1, all 16 attn blocks)")
        if t == "i64":
            sr = cons[0]
            val = producer(nodes[sr]["src"][0])
            which = "k_idxs" if producer(val["src"][0])["op"] == "ROPE" else "v_idxs"
            return ("other:%s" % which,
                    "i64[1] destination row index of the %s-cache SET_ROWS (shared by all 16 attn blocks)"
                    % which[0])
        if mask_cpy_node in cons:
            which = "src" if slot == 0 else "dst"
            return ("mask", "%s[n_kv=%d] KQ mask, CPY %s (%s)" % (
                t, ne[0], which,
                "input" if which == "src" else "converted buffer, FLASH_ATTN_EXT src3"))
        if t == "i32" and blk == "post":
            return ("other:out_row_index", "i32[1] logits row selector (epilogue GET_ROWS src1)")
        raise AssertionError("unclassified leaf l%d %s %s first-ref n%d" % (k, t, ne, n0))

    gguf = {}
    with open(GGUF_CSV) as f:
        for row in csv.DictReader(f):
            gguf[row["name"]] = (row["type"].lower(), row["shape"])
    for k in sorted(leaves):
        cls, why = classify(k)
        if cls.startswith("weight:"):
            name = cls.split(":", 1)[1]
            assert name in gguf, name
            gt, gs = gguf[name]
            ne = leaves[k]["ne"]
            shape = "x".join(str(x) for x in ne[:2]) if ne[1] > 1 else str(ne[0])
            assert gs == shape, (name, gs, shape)
            assert gt == leaves[k]["type"].replace("q4_0_ar16", "q4_0_ar16"), (name, gt)
        leaf_rows[k] = (cls, why)

    # gguf coverage: every non-blk.64, non-nextn weight must be claimed
    claimed = {r[0].split(":", 1)[1] for r in leaf_rows.values() if r[0].startswith("weight:")}
    expected = {n for n in gguf if not n.startswith("blk.64.")}
    missing = expected - claimed
    extra = claimed - expected
    assert not extra, extra
    assert missing == set(), missing

    # ---- write schedule.csv
    with open(os.path.join(HERE, "schedule.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["node_idx", "op", "type", "ne", "decoded_params",
                     "src_refs", "block_idx", "macro_op_id", "macro_kind"])
        for i in sorted(nodes):
            n = nodes[i]
            wr.writerow([i, n["op"], n["type"],
                         "x".join(str(x) for x in n["ne"]),
                         decode_params(n["op"], n["par"]),
                         ";".join(n["src"]), block_of[i],
                         macro_id[i], macro_kind[i]])

    with open(os.path.join(HERE, "leaves.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["leaf_idx", "type", "ne", "classification",
                     "first_referencing_node", "reasoning"])
        for k in sorted(leaves):
            lf = leaves[k]
            wr.writerow([k, lf["type"], "x".join(str(x) for x in lf["ne"]),
                         leaf_rows[k][0], "n%d" % first_ref[k], leaf_rows[k][1]])

    # ---- report
    import collections
    kinds = collections.Counter(macro_kind.values())
    insts = collections.Counter()
    for i in macro_kind:
        insts[(macro_kind[i], macro_id[i])] = 1
    inst_per_kind = collections.Counter(k for k, _ in insts)
    print("nodes assigned: %d/%d" % (len(macro_kind), len(nodes)))
    print("leaves classified: %d/%d" % (len(leaf_rows), len(leaves)))
    print("attention layers:", attn_layers)
    print("%-20s %9s %6s" % ("macro_kind", "instances", "nodes"))
    for k in sorted(inst_per_kind, key=lambda k: -inst_per_kind[k]):
        print("%-20s %9d %6d" % (k, inst_per_kind[k], kinds[k]))
    print("total macro instances:", sum(inst_per_kind.values()))


if __name__ == "__main__":
    main()
