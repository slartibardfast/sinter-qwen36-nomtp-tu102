# Dual-GPU parity — integrator's independent verification

Re-ran from a clean rebuild (not trusting the build agent's numbers). All
reproduced; one framing correction below.

## Confirmed

- **Floor beaten: 44.08 tok/s, 22.35/22.39 ms/pass** (both GPUs, on-device
  clock64), independently measured. 1.6× the single-GPU 27.3. The stock
  floor is 43.9.
- **Output parity holds**: logits KL max **1.434e-2 ≤ 0.02** (below the stock
  tensor split's own shallow 1.546e-2), **32/32 greedy tokens match**
  `ref-tensor-ctx64`. Single and dual megakernel emit **identical tokens**
  (config-invariant output).
- **Cross-GPU reduce ≈ 0.27 ms/pass** for all 128 sites (~2.1 µs/site over
  NVLink) vs the fork's ~2 ms PCIe AllReduce.
- Op unit tests stay green (headers unchanged); the split bug the agent
  fixed (DeltaNet v-head→k-head is a `% n_k_heads` map, needing a strided
  period-16 head split) is real and its block-0 residual is bit-identical
  to single-GPU (7e-8).

## Correction to the agent's framing: the activation-RMS lane

The agent called the residual-RMS exceedance "definitional, more accurate
than single-GPU." That is directionally right but understates it. Measured:

- Our dual residual RMS vs `ref-tensor` reaches **9.75e-2** (1430/2016 nodes
  over 0.02), and it is already **0.147 at early steps** — NOT purely
  deep-late.
- The **stock tensor split's own** residual RMS vs stock single-GPU is
  **8.6e-3** (under the bound). So our discrepancy is ~10–17× the legitimate
  stock reordering.
- Single-vs-dual megakernel residual grows **monotonically from 7e-8 at
  block 0 to ~6% at block 62** (46/62 steps non-decreasing) — a clean
  depth-amplification signature.

Mechanism (settled): the megakernel's **fp32 NVLink reduce differs from the
fork's bf16 PCIe AllReduce**; each of the 2 reduces/block introduces a small
difference vs the reference, and the model's non-contractive blocks
(DeltaNet recurrence, attention softmax) amplify it geometrically through 63
blocks. It is **output-preserving** (tokens identical, KL in band) and fully
explained — not silent wrongness, not a bug. But it means:

**The G13 activation-RMS lane (0.02), as literally specified, does NOT pass
on the cross-GPU config.** That tolerance was calibrated on same-reduce
legitimate reordering at shallow depth (`capture/g13-envelope`); a
higher-precision-but-different reduce amplified through depth is outside what
it anticipated. This needs an operator/gate decision, recorded honestly
rather than papered over:

- The binding correctness lanes for the cross-GPU config are the **output
  lanes** — logits KL and greedy-token agreement — both of which pass, and
  pass at/below the stock split's own numbers.
- The activation-RMS lane should either be **recalibrated** to admit
  legitimate reduce-variant divergence (comparing intermediate activations
  bit-for-bit across two different reduces is not a fair equality), or
  scoped to the same-reduce (single-GPU) config where it binds cleanly.
- A stronger check, if wanted: a reference dumped with OUR fp32 reduce (the
  fork doesn't produce one), against which the residual RMS would collapse.

None of this touches the output: the kernel decodes the correct tokens and
beats the floor. It is a calibration question about an intermediate
diagnostic, surfaced not hidden.
