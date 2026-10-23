# The worked example: MTP-off, tensor split, batch-1 decode

The playbook (`plan/0135` README), run end to end for the simplest real
configuration, with every number measured on this rig. This is the example an
adopter follows to instantiate the playbook for any other configuration; the
production variant (MTP depth 3) is the face-off's deliverable and is *not*
worked here. The example deliberately stops at the kernel boundary: the
playbook's final step — the persistent kernel itself — is the adopter's, and
the judges never implement (`call/0019`).

## The playbook's five steps, applied

**1. Capture the configuration's decode anatomy.** Done in
`../../capture/`: `tensor-plain-census.csv` + `tensor-plain-anatomy.txt`
(nsys, 124 decode passes, engine at the recorded pin `546eca8dc`). Headline
anatomy: 3,849 kernel launches per pass (1,924 per GPU), CUDA graphs
inactive, GPUs 87–88% busy with 2.9–3.2 ms/pass of gap, ~150 MB/pass of
recurrent-state DtoD churn.

**2. Derive the fusion set.** The census IS the fusion set: ~30 distinct
kernels per GPU per pass, dominated by `mul_mat_vec_q<Q4_0,ncols1>` (two
shapes, 56% of busy), the fp16 head GEMV, `ggml_cuda_ar_kernel` (128/pass
— identified 2026-07-12 as the fork's cross-GPU AllReduce over pinned-host
PCIe with a bf16 wire, not state banking; the megakernel replaces it with
NVLink peer exchanges, an itemized win), the AR16 in_proj MMVQ (48/pass =
the linear-attention layers), FA over the 16 full-attention layers, and an
elementwise tail of ≤2 µs kernels whose entire cost is scheduling. Per-kernel DRAM rates and
occupancies for the top set are in `../../capture/ncu-*.csv`.

**3. Anchor the DRAM budget.** From `gguf-tensors.csv`: per-pass weight
stream 16.69 GB (8.35 GB/GPU excluding the embedding row-lookup, including
the vocab-split fp16 head half) → **13.71 ms/pass floor** at 608.6 GB/s per
GPU. Measured: 22.8 ms/pass untraced (43.9 tok/s) = **60% of roofline**; the
gap decomposes into launch/sync gaps (~3 ms), the MMVQ stream running at
397–408 GB/s (65–67% of roofline), the small-kernel tail, and the state
churn. That decomposition is the kernel's work-list, in priority order.

**4. Instantiate the gate manifest.** The prerequisite measurement rows are
judge-owned and shared. The single-GPU gates bind as written (G10 SASS
parity, G11 cooperative launch at ≤168 regs/thread ≥12 warps/SM, G12 ≥70% of
608.6 GB/s sustained, G13 parity per the playbook tolerances, G14 litmus
set, G15 interpreter overhead ≤2%). The speculation gates are **vacuous at
k = 0** — they bind the production instantiation, not this one; the
playbook's degeneracy tripwires (DegenerateAtZero in the TLA model,
BankedStateEquivalence at accepted = 0) stand in for them. The dual-GPU,
long-context, agentic and integration gates bind as written.

**5. Bind the specs.** `mtp-off.allium` beside this file: one
`DecodeConfiguration` default (spec_depth 0, no drafter, tensor split,
fingerprint recorded at first shim hash) importing the playbook contracts
untouched. Concurrency: the SAME `../MegakernelSync.tla` at the
`MegakernelSync_k0.cfg` instance — TLC-checked complete (27 states):
`NoStaleRead`, `BankIsolation`, `BaseIntegrity`, `DegenerateAtZero`, and
the `Progress` liveness all hold with drafting and rollback unreachable.

## What an adopter builds from here

The persistent kernel for this binding: one cooperative launch per GPU
serving the fused set from step 2 inside the budget from step 3, passing the
gates from step 4, verified against obligations generated from the specs in
step 5 (`tools/allium`; drive the real dispatch, not a test entry point).
G11's register/occupancy plan becomes fact at the first `ptxas -v` run;
calx-mill projects it beforehand (the consumer-crate pattern from
plan/0136; the projection is `g11-derivation.md` beside this file). Everything above this line is data and contract — nothing in
this example constrains HOW the kernel is structured, which is the point:
two independent implementations of the production sibling are about to
compete on exactly that freedom.
