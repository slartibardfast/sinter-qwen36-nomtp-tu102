# Megakernel sync specs — co-located with the software (plan/0012 #12)

These TLA+ specs are **co-located with the software they model**, never under
`plan/*/spec/` (which evades the mandatory verification lanes — `host-lifecycle
software --check` flags that as a HAZARD). They moved here from
`plan/0135-persistent-decode-megakernel/spec/` on 2026-07-15.

The specula lane (TLC) checks the megakernel's cooperative sync protocol. Run any
config with `./run-tlc.sh <Module> <config-basename> [extra TLC flags]`. Resolve
`tla2tools.jar` via `$TLA_TOOLS` or `<repo>/.tla-tools/tla2tools.jar`.

| module | models | positive | must-fail negatives |
|---|---|---|---|
| `MegakernelSync` | the pass loop over speculative banks (parametric in k) | `MegakernelSync_k0`, `MegakernelSync_k3` | (invariants degenerate at k=0) |
| `SignalEdge` | the cross-GPU Y06 NVLink producer→consumer handoff, L1 staleness | `SignalEdge_y06_fenced_strong` | `SignalEdge_y06_plain_reads`, `SignalEdge_sentinel_*` |
| `WaitCount` | the runtime-written-count wait obligation | `WaitCount_init_ordered` | `WaitCount_init_unordered` |
| `WarpSpecHandoff` | the block-scope MOVER/COMPUTE double-buffer handoff (plan/0143) | `WarpSpecHandoff_overlap` | `WarpSpecHandoff_no_fence`, `WarpSpecHandoff_no_warguard` (run `-deadlock`) |

`WarpSpecHandoff` is the spec-first concurrency contract for the FATTN f16-overlap
(LEAD) + pipe-overlap levers (plan/0143 task #45): the warp-specialized double-buffer
that replaces the per-tile `__syncthreads`. Its two must-fail negatives pin the two
ordering edges the barrier used to give for free — E1 fill→consume visibility
(`ProducerFence`) and E2 consume→refill WAR (`WARGuard`). Both must hold in the kernel.

A must-fail negative that passes cleanly means the model lost the hazard — treat it
as a spec bug, not a green check.

Still owed here (tracked): the megakernel **behaviour** specs (`playbook.allium`,
`worked-example-mtp-off/`) also co-locate under `specs/`, after their allium-lane
re-verification and plan/0135 doc-reference updates.
