#!/usr/bin/env bash
# G13 oracle reference-run matrix (plan/0135). Bulk data -> /var/tmp/mk-oracle.
# Serving config: tensor split, fa on, f16 KV, n_ctx 262144; prefill depths 64
# and 2100, 32 greedy decode steps, each run twice across process restarts for
# the run-to-run check. Plus the fp32-order cross-check: split none, n_ctx 8192,
# prefill 64.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf
OUT=/var/tmp/mk-oracle
TAG=$(git -C /home/dconnolly/yarn-agentic/software/llama.cpp/autoround rev-parse --short HEAD)

mkdir -p "$OUT"

run() { # name split n_ctx ctx_tokens
    local name=$1 split=$2 n_ctx=$3 depth=$4
    echo "=== $name (split=$split n_ctx=$n_ctx depth=$depth) ==="
    ./oracle --model "$MODEL" --split "$split" --n-ctx "$n_ctx" \
             --ctx-tokens "$depth" --steps 32 --tag "$TAG" \
             --out "$OUT/$name" > "$OUT/$name.log" 2>&1
}

run ref-tensor-ctx64        tensor 262144 64
run ref-tensor-ctx64-rerun  tensor 262144 64
run ref-tensor-ctx2100       tensor 262144 2100
run ref-tensor-ctx2100-rerun tensor 262144 2100
run ref-none8192-ctx64      none   8192   64

echo "=== self-parity (tree vs itself) ==="
python3 compare.py "$OUT/ref-tensor-ctx64" "$OUT/ref-tensor-ctx64" --report "$OUT/self-parity.json"

echo "=== run-to-run (fresh process, same config) ==="
python3 compare.py "$OUT/ref-tensor-ctx64"  "$OUT/ref-tensor-ctx64-rerun"  --report "$OUT/r2r-ctx64.json"
python3 compare.py "$OUT/ref-tensor-ctx2100" "$OUT/ref-tensor-ctx2100-rerun" --report "$OUT/r2r-ctx2100.json"
echo "--- byte-level tree diff (strongest form) ---"
diff -r "$OUT/ref-tensor-ctx64"  "$OUT/ref-tensor-ctx64-rerun"  && echo "ctx64: trees byte-identical"
diff -r "$OUT/ref-tensor-ctx2100" "$OUT/ref-tensor-ctx2100-rerun" && echo "ctx2100: trees byte-identical"

echo "=== cross-check: production tensor split vs fp32-order reference (informational) ==="
python3 compare.py "$OUT/ref-tensor-ctx64" "$OUT/ref-none8192-ctx64" \
        --allow-config-mismatch --report "$OUT/cross-none.json" || true

echo "done"
