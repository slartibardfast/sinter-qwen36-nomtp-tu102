# G13 numerical-parity oracle, reference dump manifest (plan/0135)

Produced 2026-07-11 on the dedicated rig (2x Quadro RTX 6000 TU102, clocks
locked 1455 MHz, driver 610.43.03, GPUs otherwise idle).

## Provenance

- Engine: llama.cpp autoround worktree, commit
  `546eca8dc9cac8ecfdd944f0a62326390432da5c` (`546eca8dc`), prebuilt CUDA libs
  from `/home/dconnolly/yarn-agentic/software/llama.cpp/cuda-ar16/build/bin`
  (arch 75). Engine NOT rebuilt; the dumper links against it (see build.sh).
- Model: `/opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf`, 19246993696 bytes,
  mtime 2026-07-10 17:56 (read-only; sha not taken per rig rules). n_vocab
  248320; blocks 0..63 = 48 DeltaNet linear + 16 full attention (3,7,...,63).
- Dumper: `oracle.cpp` (this dir), built by `./build.sh` with the host g++,
  headers from the autoround worktree. Extends capture/g13-envelope/probe.cpp;
  observation is cb_eval-only (ask==true -> observe, data fetched at
  ask==false via ggml_backend_tensor_get); graph structure untouched.
- Fixed input: the probes' BASE_TEXT (identical string in probe.cpp /
  fingerprint.cpp / oracle.cpp), tokenized without specials, BOS once, tiled
  cyclically and truncated to exactly the prefill depth.

## Runs (bulk data under /var/tmp/mk-oracle/)

All runs: greedy fp32 first-max argmax, no sampler, no RNG; -ngl 999, flash
attention enabled, f16 KV; 32 decode steps. Exact commands (run by ./run.sh):

```
./oracle --model /opt/models/Qwen3.6-27B-Q4_0AR16-b9222.gguf --split tensor \
    --n-ctx 262144 --ctx-tokens 64   --steps 32 --tag 546eca8dc --out /var/tmp/mk-oracle/ref-tensor-ctx64
./oracle ... --split tensor --n-ctx 262144 --ctx-tokens 64   ... --out /var/tmp/mk-oracle/ref-tensor-ctx64-rerun
./oracle ... --split tensor --n-ctx 262144 --ctx-tokens 2100 ... --out /var/tmp/mk-oracle/ref-tensor-ctx2100
./oracle ... --split tensor --n-ctx 262144 --ctx-tokens 2100 ... --out /var/tmp/mk-oracle/ref-tensor-ctx2100-rerun
./oracle ... --split none   --n-ctx 8192   --ctx-tokens 64   ... --out /var/tmp/mk-oracle/ref-none8192-ctx64
```

- `ref-tensor-ctx64`, `ref-tensor-ctx2100`: THE references at the serving
  config (-sm tensor, fa on, f16 KV, n_ctx 262144), prefill depths 64 / 2100.
- `*-rerun`: same config, fresh process, for the run-to-run check.
- `ref-none8192-ctx64`: the fp32-order cross-check tree (--split none, single
  GPU, n_ctx 8192, prefill 64).

## Dump-tree format (mk-oracle/v1)

Per run dir: `index.json` (deterministic run record: config, prompt +
emitted token ids, layouts; no timestamps, trees byte-comparable),
`logits.bin` (33 rows x 248320 f32; row 0 = prefill output, row s+1 = output
of decode step s; token s chosen greedily from row s), `tokens.txt` (32 ids),
`nodes.csv` (per decode step, every scheduler-presented node in callback
order: step,idx,name,op,type,ne0..3,summary,rms,mean,min,max,v0..v7; fp64
summaries; summary vocabulary: ok = data fetched; big = >8 MiB, the 16
attention layers' whole-KV SET_ROWS nodes, mostly never-written cache; skip =
non-float; empty = zero-extent bookkeeping CPYs; hostview = views of
host-buffer inputs, which the tensor-split meta backend skips in subgraph
planning — observing one trips GGML_ASSERT(i_start == cgraph->n_nodes) at
ggml-backend-meta.cpp:1857, so they are recorded at ask time and never
observed; meta = tensor-split runs only: the meta backend cannot service
get_tensor on arbitrary nodes, GGML_ASSERT(ggml_is_contiguous) at :1311 and
GGML_ABORT on SPLIT_AXIS_PARTIAL, and the split state is not queryable, so at
tensor split only the needed families are fetched: l_out-*, result_norm,
result_output, state writes — the same set the g13-envelope acts probe
fetched at tensor split. Split-none runs carry full per-node summaries:
capture=full vs targeted is recorded in index.json config),
`full/s<S>/l_out-<il>.bin` (residual stream, f32, 5120 x 1, blocks 0..62;
block 63's residual has no l_out node — covered by result_norm/result_output
summaries, same coverage as the g13-envelope calibration),
`state/s<S>/cache_{s,r}_l<il>.bin` (every DeltaNet state written per step:
48 x cache_s = ssm state, f32 786432 = 128x128x48, and 48 x cache_r = conv
state, f32 30720; raw bytes for bit-exact comparison), `files.csv` (inventory
of the .bin files). Node identification detail: the conv-state write CPY is
cb()-renamed to `state_update_target-<il> (copy of last_conv_states-<il>)`;
the dump maps it back to its physical bank name `cache_r_l<il>`.

## Comparison

`compare.py REF_DIR CAND_DIR [--kl-tol 0.02] [--rms-tol 0.02] [--report r.json]
[--allow-config-mismatch]` — tolerances per spec/playbook.allium (calibrated
in capture/g13-envelope/NOTES.md): per-position logits KL <= 0.02 nats (fp64
log-softmax, both directions), residual-stream per-node RMS rel diff <= 0.02,
DeltaNet state bit-exact, greedy agreement with within-band ties recorded not
scored, comparison truncated at the first token divergence. Exit 0 pass /
1 tolerance violation / 2 structural mismatch.

## RESULTS (matrix of record, 2026-07-11)

Tree sizes: 4.8 GB per run (state/ dominates: 32 steps x 96 files, 48 x 3 MiB
cache_s + 48 x 120 KiB cache_r), 24 GB total under /var/tmp/mk-oracle/.
Per tree: logits.bin 32.8 MB (33 x 248320 f32), 2016 l_out files, 3072 state
files, nodes.csv 118,528 rows. 3704 nodes per decode step at every config and
depth (matches the LG0 fingerprint count).

- Self-parity (ref-tensor-ctx64 vs itself): PASS, exact zeros — KL 0, RMS rel
  0, 3072/3072 state files identical. `summaries/self-parity.json`.
- Run-to-run (fresh process, same config, both depths): PASS, exact zeros,
  and `diff -r` shows the whole trees byte-identical (r2r-ctx64.json,
  r2r-ctx2100.json). Reproduces the g13-envelope finding: the run-to-run
  envelope is exactly zero.
- Cross-check (production tensor split vs the fp32-order --split none tree,
  --allow-config-mismatch, informational): logits KL max 2.504e-3 and
  residual RMS rel max 8.53e-3, both well inside the 0.02 class; greedy
  diverged at step 4 as a tie within the band (ref gap 0.064, cand gap
  0.0054) — recorded, not scored, comparison truncated there. The state lane
  reports all 384 comparable files NOT bit-exact and exits 1: expected across
  different split topologies (the playbook binds state bit-exactness against
  the instantiation's DECLARED fold order, i.e. candidate vs the
  matching-config reference), and it demonstrates the state lane detects
  differences. `summaries/cross-none.json`.

Node-order/naming notes for the candidate side: callback order is scheduler
split-execution order (topological within splits) and was bit-stable across
runs; the ssm write appears as `cache_s_l<il> (view) (copy of new_state-<il>)`
but the conv write is cb()-renamed to `state_update_target-<il> (copy of
last_conv_states-<il>)` (mapped back to cache_r_l<il> in the dump); the
zero-extent rs bookkeeping CPYs (`... (copy of )`, ne1=0) are recorded but
carry no data; l_out exists for blocks 0..62 only — block 63 (attention) has
no l_out node, its residual is covered by result_norm/result_output.
