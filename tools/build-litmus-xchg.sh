#!/usr/bin/env bash
# Build the cross-GPU exchange litmus (standalone, not a CMake target — it
# needs both GPUs and its own cooperative launch, kept out of the main build).
# Emits four binaries into /tmp: the positive and the three negatives.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp}"
NVCC="nvcc -arch=sm_75 -std=c++17 -O2"

$NVCC tests/litmus_xchg.cu -o "$OUT/mk-litmus-xchg"
$NVCC -DXCHG_PLAIN_READS tests/litmus_xchg.cu -o "$OUT/mk-litmus-xchg-plain"
$NVCC -DXCHG_NO_MEMBAR   tests/litmus_xchg.cu -o "$OUT/mk-litmus-xchg-nomembar"
$NVCC -DXCHG_SEQNO_EARLY tests/litmus_xchg.cu -o "$OUT/mk-litmus-xchg-seqnoearly"
echo "built: $OUT/mk-litmus-xchg{,-plain,-nomembar,-seqnoearly}"
