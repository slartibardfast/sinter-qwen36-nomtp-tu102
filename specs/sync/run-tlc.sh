#!/usr/bin/env bash
# Run TLC on the megakernel sync specs (co-located with the software, plan/0012).
#
# Usage:
#   ./run-tlc.sh WarpSpecHandoff WarpSpecHandoff_overlap
#   ./run-tlc.sh MegakernelSync  MegakernelSync_k0
#   ./run-tlc.sh SignalEdge      SignalEdge_y06_plain_reads   # a MUST-FAIL negative
#
# Args: <module> <config-basename> [extra TLC flags...]
#   config file = <config-basename>.cfg; extra flags pass through to tlc2.TLC
#   (e.g. -deadlock to disable deadlock checking for a safety-only MUST-FAIL cfg).
#
# Resolve tla2tools.jar in this order:
#   1. $TLA_TOOLS (explicit override)
#   2. <this-repo-root>/.tla-tools/tla2tools.jar
#   3. the host project's .tla-tools/tla2tools.jar (yarn-agentic checkout)
set -euo pipefail

MODULE="${1:?module name, e.g. WarpSpecHandoff}"
CONFIG="${2:?config basename, e.g. WarpSpecHandoff_overlap}"
shift 2
EXTRA=("$@")
SPEC_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_FILE="${SPEC_DIR}/${CONFIG}.cfg"
TLA_FILE="${SPEC_DIR}/${MODULE}.tla"

if [[ -z "${TLA_TOOLS:-}" ]]; then
    REPO_ROOT="$(git -C "${SPEC_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    for candidate in \
        "${REPO_ROOT:+${REPO_ROOT}/.tla-tools/tla2tools.jar}" \
        "${HOME}/yarn-agentic/.tla-tools/tla2tools.jar"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            TLA_TOOLS="${candidate}"
            break
        fi
    done
fi

if [[ -z "${TLA_TOOLS:-}" || ! -f "${TLA_TOOLS}" ]]; then
    echo "tla2tools.jar not found. Set \$TLA_TOOLS or place it at <repo>/.tla-tools/tla2tools.jar" >&2
    exit 1
fi
[[ -f "${CFG_FILE}" ]] || { echo "config not found: ${CFG_FILE}" >&2; exit 1; }
[[ -f "${TLA_FILE}" ]] || { echo "module not found: ${TLA_FILE}" >&2; exit 1; }

cd "${SPEC_DIR}"
exec java -XX:+UseParallelGC -cp "${TLA_TOOLS}" tlc2.TLC \
    -nowarning "${EXTRA[@]}" -config "${CFG_FILE}" "${TLA_FILE}"
