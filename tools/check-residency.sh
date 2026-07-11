#!/usr/bin/env bash
# G11 residency gate: run a ptxas -v log through calx-mill's cooperative
# residency check at the recorded envelope (72-block cooperative grid, 384
# threads, 60 KiB dynamic smem slab). Nonzero exit on any DEADLOCK — a build
# whose interpreter cannot place one block per SM must not ship.
# Derivation: plan/0135 spec/worked-example-mtp-off/g11-derivation.md.
set -euo pipefail
LOG="${1:?usage: check-residency.sh <ptxas-v-log> [calx-mill-bin]}"
CALX="${2:-calx-mill}"

exec "$CALX" ptxas "$LOG" --block-threads 384 --block-smem 61440 --grid-blocks 72
