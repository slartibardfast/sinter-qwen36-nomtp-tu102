#!/usr/bin/env bash
# Build the G13 oracle dumper against the pre-built engine (do not rebuild it).
# Headers: the autoround worktree (engine pin 546eca8dc).
# Libs:    the cuda-ar16 worktree's CUDA build, same commit; the autoround
#          worktree's own build/ is CPU-only (GGML_CUDA=OFF).
set -euo pipefail
cd "$(dirname "$0")"

ENGINE=/home/dconnolly/yarn-agentic/software/llama.cpp/autoround
LIBDIR=/home/dconnolly/yarn-agentic/software/llama.cpp/cuda-ar16/build/bin

g++ -O2 -std=c++17 oracle.cpp -o oracle \
    -I"$ENGINE/include" -I"$ENGINE/ggml/include" \
    -L"$LIBDIR" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$LIBDIR"

echo "built: $(pwd)/oracle"
