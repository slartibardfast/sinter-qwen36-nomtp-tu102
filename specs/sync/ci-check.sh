#!/usr/bin/env bash
# The specula lane gate for the co-located megakernel sync suite (plan/0012 #12).
# Every POSITIVE config must model-check clean ("No error has been found"); every
# MUST-FAIL negative must produce its counterexample (an invariant/temporal
# violation). A negative that passes clean means the model lost its hazard -- a
# spec regression, failed here. Run locally or from CI (specula.yml).
#
#   $TLA_TOOLS -> tla2tools.jar   (CI downloads it; locally, run-tlc.sh resolves it)
set -uo pipefail
cd "$(dirname "$0")"

# module config expectation  (pass = clean; fail = must produce a counterexample)
# negatives run with -deadlock so a stall cannot mask the intended safety violation.
CASES=(
    "MegakernelSync   MegakernelSync_k0            pass"
    "MegakernelSync   MegakernelSync_k3            pass"
    "SignalEdge       SignalEdge_y06_fenced_strong pass"
    "SignalEdge       SignalEdge_sentinel_trio     pass"
    "SignalEdge       SignalEdge_y06_plain_reads   fail"
    "SignalEdge       SignalEdge_sentinel_unordered_stores fail"
    "SignalEdge       SignalEdge_sentinel_collision fail"
    "WaitCount        WaitCount_init_ordered       pass"
    "WaitCount        WaitCount_init_unordered     fail"
    "WarpSpecHandoff  WarpSpecHandoff_overlap      pass"
    "WarpSpecHandoff  WarpSpecHandoff_no_fence     fail"
    "WarpSpecHandoff  WarpSpecHandoff_no_warguard  fail"
)

rc=0
for case in "${CASES[@]}"; do
    read -r mod cfg expect <<< "$case"
    if [[ "$expect" == fail ]]; then flags="-deadlock"; else flags=""; fi
    out="$(./run-tlc.sh "$mod" "$cfg" $flags 2>&1)"
    clean=0; grep -q "No error has been found" <<< "$out" && clean=1
    counterexample=0
    grep -qE "is violated|Temporal properties were violated|Deadlock reached" <<< "$out" && counterexample=1
    if [[ "$expect" == pass && "$clean" == 1 ]]; then
        echo "PASS  $cfg  (clean)"
    elif [[ "$expect" == fail && "$counterexample" == 1 && "$clean" == 0 ]]; then
        echo "PASS  $cfg  (counterexample as intended)"
    else
        echo "FAIL  $cfg  (expected $expect; clean=$clean counterexample=$counterexample)"
        rc=1
    fi
done
echo "-----"
[[ "$rc" == 0 ]] && echo "specula lane: ALL ${#CASES[@]} configs behaved as specified" \
                 || echo "specula lane: FAILURES above"
exit $rc
