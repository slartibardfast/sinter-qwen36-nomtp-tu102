#!/usr/bin/env python3
"""compare.py - G13 numerical-parity oracle comparator (plan/0135)

Compares two mk-oracle/v1 dump trees (reference vs candidate) against the
calibrated G13 tolerance classes (spec/playbook.allium, capture/g13-envelope):

  1. logits: per-position KL (float64 log-softmax, both directions) over
     comparable positions <= --kl-tol (default 0.02 nats).
  2. residual stream: per-node RMS relative difference |rms_c - rms_r|/rms_r
     for l_out-* nodes <= --rms-tol (default 0.02); result_norm/result_output
     reported informationally.
  3. DeltaNet state: every state/s<S>/*.bin (cache_s_l* and cache_r_l* writes)
     byte-identical (state_bit_exact_tol = 0.0).
  4. greedy tokens: identical sequences, with tie handling per the playbook:
     a divergence at a position whose logits are within the KL band in both
     directions is a tie-within-band, RECORDED but not scored; comparison is
     truncated at the first divergence (contexts differ afterwards).

Comparable positions: logits row r depends on prompt + tokens[0..r-1], so
rows 0..d are comparable when tokens first diverge at index d (all rows if no
divergence). Node/state records of decode step s consume token s, so steps
0..d-1 are comparable.

Usage:
    compare.py REF_DIR CAND_DIR [--kl-tol 0.02] [--rms-tol 0.02]
               [--report OUT.json] [--allow-config-mismatch]

Exit codes: 0 = pass (ties allowed), 1 = tolerance violation, 2 = structural
mismatch (shape/inventory/config).
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

violations = []
notes = []


def fail_structural(msg):
    print(f"STRUCTURAL: {msg}")
    sys.exit(2)


def load_index(d):
    p = os.path.join(d, "index.json")
    if not os.path.exists(p):
        fail_structural(f"{p} missing")
    with open(p) as f:
        idx = json.load(f)
    if idx.get("format") != "mk-oracle/v1":
        fail_structural(f"{p}: format {idx.get('format')!r} != mk-oracle/v1")
    return idx


def load_logits(d, idx):
    raw = np.fromfile(os.path.join(d, "logits.bin"), dtype=np.float32)
    rows, cols = idx["logits"]["rows"], idx["logits"]["cols"]
    if raw.size != rows * cols:
        fail_structural(f"{d}/logits.bin: size {raw.size} != {rows}x{cols}")
    return raw.reshape(rows, cols)


def load_nodes_rms(d):
    """(step, name) -> rms for summarized nodes."""
    out = {}
    with open(os.path.join(d, "nodes.csv")) as f:
        for row in csv.DictReader(f):
            if row["summary"] == "ok":
                out[(int(row["step"]), row["name"])] = float(row["rms"])
    return out


def load_files(d, kind):
    """(step, name) -> (path, nbytes) for files.csv rows of the given kind."""
    out = {}
    with open(os.path.join(d, "files.csv")) as f:
        for row in csv.DictReader(f):
            if row["kind"] == kind:
                key = (int(row["step"]), row["name"])
                if key in out:
                    fail_structural(f"{d}/files.csv: duplicate {kind} entry {key}")
                out[key] = (row["path"], int(row["nbytes"]))
    return out


def log_softmax(x):
    x = x.astype(np.float64)
    m = np.max(x, axis=1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))


def kl_nats(a, b):
    """KL(p_a || p_b) per row, softmax in float64."""
    la, lb = log_softmax(a), log_softmax(b)
    return np.sum(np.exp(la) * (la - lb), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--kl-tol", type=float, default=0.02)
    ap.add_argument("--rms-tol", type=float, default=0.02)
    ap.add_argument("--report", default=None)
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="tolerate split/n_ctx differences (cross-check trees); "
                         "prompt, steps and n_vocab must still match")
    args = ap.parse_args()

    idx_r = load_index(args.ref)
    idx_c = load_index(args.cand)
    cfg_r, cfg_c = idx_r["config"], idx_c["config"]

    # hard structural requirements: same inputs, same shapes
    for k in ("n_vocab", "steps", "ctx_tokens"):
        if cfg_r[k] != cfg_c[k]:
            fail_structural(f"config.{k}: ref {cfg_r[k]} != cand {cfg_c[k]}")
    if idx_r["prompt_tokens"] != idx_c["prompt_tokens"]:
        fail_structural("prompt_tokens differ")
    # soft: serving config may legitimately differ for the fp32-order cross-check
    for k in ("split", "n_ctx"):
        if cfg_r[k] != cfg_c[k]:
            msg = f"config.{k}: ref {cfg_r[k]} != cand {cfg_c[k]}"
            if not args.allow_config_mismatch:
                fail_structural(msg + " (use --allow-config-mismatch for cross-check trees)")
            notes.append(msg)
            print(f"note: {msg}")

    steps = cfg_r["steps"]
    tok_r = idx_r["emitted_tokens"]
    tok_c = idx_c["emitted_tokens"]

    # --- greedy token agreement / divergence ---------------------------------
    div = next((i for i in range(steps) if tok_r[i] != tok_c[i]), None)
    n_rows_cmp = steps + 1 if div is None else div + 1  # logits rows 0..div
    n_steps_cmp = steps if div is None else div         # node/state steps 0..div-1

    # --- logits KL ------------------------------------------------------------
    log_r = load_logits(args.ref, idx_r)
    log_c = load_logits(args.cand, idx_c)
    lr, lc = log_r[:n_rows_cmp], log_c[:n_rows_cmp]
    bit_identical = bool(np.array_equal(lr, lc))
    kl_fwd = kl_nats(lr, lc)
    kl_rev = kl_nats(lc, lr)
    kl_max = float(max(kl_fwd.max(), kl_rev.max()))
    kl_argmax = int(kl_fwd.argmax() if kl_fwd.max() >= kl_rev.max() else kl_rev.argmax())
    print(f"logits: {n_rows_cmp}/{steps + 1} rows compared, bit-identical: {'yes' if bit_identical else 'no'}")
    print(f"logits KL: max {kl_max:.6e} (row {kl_argmax})  fwd mean {kl_fwd.mean():.6e}  rev mean {kl_rev.mean():.6e}  tol {args.kl_tol}")
    if kl_max > args.kl_tol:
        violations.append(f"logits KL max {kl_max:.6e} > {args.kl_tol} at row {kl_argmax}")

    tie = None
    if div is None:
        print(f"tokens: {steps}/{steps} greedy agreement")
    else:
        # tie handling: within-band divergence is recorded, not scored
        row_kl = max(float(kl_fwd[div]), float(kl_rev[div]))
        gap_r = float(log_r[div][tok_r[div]] - log_r[div][tok_c[div]])
        gap_c = float(log_c[div][tok_c[div]] - log_c[div][tok_r[div]])
        tie = {
            "position": div, "ref_token": tok_r[div], "cand_token": tok_c[div],
            "row_kl": row_kl, "ref_logit_gap": gap_r, "cand_logit_gap": gap_c,
            "within_band": row_kl <= args.kl_tol,
        }
        print(f"tokens: diverge at step {div} (ref {tok_r[div]} vs cand {tok_c[div]}); "
              f"row KL {row_kl:.6e}, logit gaps ref {gap_r:.4f} cand {gap_c:.4f}")
        if tie["within_band"]:
            print(f"tokens: divergence is a tie within the KL band ({row_kl:.6e} <= {args.kl_tol}): RECORDED, not scored; comparison truncated at step {div}")
        else:
            violations.append(f"greedy divergence at step {div} outside the band (row KL {row_kl:.6e} > {args.kl_tol})")

    # --- residual stream RMS rel diff ------------------------------------------
    rms_r = load_nodes_rms(args.ref)
    rms_c = load_nodes_rms(args.cand)
    resid = re.compile(r"^l_out-\d+$")
    keys_r = {k for k in rms_r if k[0] < n_steps_cmp and resid.match(k[1])}
    keys_c = {k for k in rms_c if k[0] < n_steps_cmp and resid.match(k[1])}
    if keys_r != keys_c:
        fail_structural(f"residual-stream node sets differ over comparable steps "
                        f"(ref only: {sorted(keys_r - keys_c)[:5]}, cand only: {sorted(keys_c - keys_r)[:5]})")
    rms_max, rms_arg = -1.0, None
    for k in sorted(keys_r):
        a, b = rms_c[k], rms_r[k]
        rel = 0.0 if a == b else (abs(a - b) / abs(b) if b != 0 else float("inf"))
        if rel > rms_max:
            rms_max, rms_arg = rel, k
    print(f"residual RMS rel diff: {len(keys_r)} (step,node) rows, max {rms_max:.6e}"
          + (f" (step {rms_arg[0]}, {rms_arg[1]})" if rms_arg else "") + f"  tol {args.rms_tol}")
    if rms_max > args.rms_tol:
        violations.append(f"residual RMS rel diff {rms_max:.6e} > {args.rms_tol} at {rms_arg}")

    for name in ("result_norm", "result_output"):
        vals = []
        for k, v in rms_r.items():
            if k[0] < n_steps_cmp and k[1] == name and k in rms_c:
                a, b = rms_c[k], v
                vals.append(0.0 if a == b else (abs(a - b) / abs(b) if b != 0 else float("inf")))
        if vals:
            print(f"  info {name}: max rel diff {max(vals):.6e} over {len(vals)} steps (not scored)")

    # --- DeltaNet state bit-exactness -------------------------------------------
    st_r = load_files(args.ref, "state")
    st_c = load_files(args.cand, "state")
    keys_r = {k for k in st_r if k[0] < n_steps_cmp}
    keys_c = {k for k in st_c if k[0] < n_steps_cmp}
    if keys_r != keys_c:
        fail_structural(f"state inventories differ over comparable steps "
                        f"(ref only: {sorted(keys_r - keys_c)[:5]}, cand only: {sorted(keys_c - keys_r)[:5]})")
    n_cmp, mismatched = 0, []
    for k in sorted(keys_r):
        pr, nr = st_r[k]
        pc, nc = st_c[k]
        if nr != nc:
            mismatched.append((k, "size"))
            continue
        with open(os.path.join(args.ref, pr), "rb") as f:
            br = f.read()
        with open(os.path.join(args.cand, pc), "rb") as f:
            bc = f.read()
        n_cmp += 1
        if br != bc:
            mismatched.append((k, "bytes"))
    print(f"state: {n_cmp} files byte-compared over {n_steps_cmp} comparable steps, "
          f"{len(mismatched)} mismatched")
    if mismatched:
        for k, why in mismatched[:10]:
            print(f"  state mismatch ({why}): step {k[0]} {k[1]}")
        violations.append(f"{len(mismatched)} state tensors not bit-exact (first: {mismatched[0]})")

    # --- verdict -----------------------------------------------------------------
    if args.report:
        with open(args.report, "w") as f:
            json.dump({
                "ref": args.ref, "cand": args.cand,
                "kl_tol": args.kl_tol, "rms_tol": args.rms_tol,
                "rows_compared": n_rows_cmp, "steps_compared": n_steps_cmp,
                "logits_bit_identical": bit_identical,
                "kl_max": kl_max, "kl_fwd_mean": float(kl_fwd.mean()), "kl_rev_mean": float(kl_rev.mean()),
                "residual_rms_rel_max": rms_max,
                "state_files_compared": n_cmp, "state_mismatched": len(mismatched),
                "token_divergence": tie, "notes": notes, "violations": violations,
            }, f, indent=2)
        print(f"wrote {args.report}")

    if violations:
        print("RESULT: FAIL")
        for v in violations:
            print(f"  violation: {v}")
        sys.exit(1)
    print("RESULT: PASS" + (" (with recorded tie)" if tie and tie["within_band"] else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
