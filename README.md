# megakernel

The persistent decode megakernel for `plan/0135` of
[yarn-agentic](https://github.com/slartibardfast/yarn-agentic): batch-1
decode of Qwen3.6-27B (hybrid gated-DeltaNet + full attention) on a pair of
NVLink-linked TU102 GPUs (sm_75), run underneath an unmodified `llama-server`
as an out-of-tree dynamic ggml backend.

This repository is the software; the thought lives in the host project. Read
there first:

- `plan/0135-persistent-decode-megakernel/` — the playbook, the worked
  MTP-off example, the specs (allium contracts, parametric TLA+ sync model),
  the measured decode anatomy (`capture/`), and the build plan
  (`mtp-off-build.md`).
- `reference/megakernel/` — the design package: pipeline/memory/sync tables,
  the gate manifest (the release discipline), latency budgets.
- `reference/tu102/` — the measured substrate table every timing here is
  grounded in.

## Layout

| dir | holds | binds to |
|---|---|---|
| `core/` | the config-invariant core: persistent interpreter spine, sync protocol (Y-rows), weight streaming, buffer plumbing | every instantiation |
| `k0/` | the MTP-off instantiation: its fused macro-op schedule and binding (`spec_depth 0`, tensor split, fingerprint `cgraph/v1:dd907a1b…`) | the worked example |
| `tests/` | parity harnesses (G13 oracle compare), litmus tests (G14, with negatives), bench entry points | gate manifest rows |
| `tools/` | build gates: the calx-mill residency check every build must pass (G11 envelope) | G10/G11 |
| `docs/` | the extracted anatomy and semantics dossiers; build lore | knowledge transfer |

The playbook expects other serving configurations (the production MTP
depth-3 kernel among them) to be **different instantiations beside `k0/`,
sharing `core/`** — never one kernel accreting config flags.

## The G11 envelope (binds every build)

Cooperative grid of 72 blocks, 384 threads/block (12 warps/SM), ≤168
registers/thread, 60 KiB shared-memory slab per block. The 12-warp register
cliff is one allocation step wide: 176 registers places zero blocks and the
cooperative launch deadlocks. Every build runs `ptxas -v` output through
calx-mill's residency check (`tools/`); a DEADLOCK is a build failure, not a
warning.

## Building

CUDA 13.3 (`/opt/cuda`), sm_75. Build out of tree in `/tmp` (the rig's
source tree sits on ntfs; building there is a recorded hazard):

```
cmake -B /tmp/mk-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/mk-build -j
```

Reproducible-build discipline (greenfield: mandatory, no exemption): the
recorded recipe in the host's `.host-software` must rebuild the artifact
byte-identically; paths are neutralized with `-ffile-prefix-map`, nothing
embeds timestamps. `cmake --build /tmp/mk-build --target repro-check` builds
twice and compares hashes.

## License

Unlicense (public domain), matching the sibling tools.
