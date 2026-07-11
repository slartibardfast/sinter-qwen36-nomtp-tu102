# Gated-DeltaNet op family — measured notes

Verified 2026-07-12 (`tests/test_gdn.cu`, freer GPU, clocks locked). The
original agent wrote `gdn.cuh` and the test and reported "state bit-exact";
credit-out cut it before the register dump — the run below is the
re-verification.

## Headline: recurrent state is BITWISE-EXACT across a multi-token chain

The single most parity-sensitive op (G21/G13: "the recurrence must not know
it was speculated"; at k=0 it must equal plain sequential decode bit for
bit). Chained over 3+ synthetic tokens, at both the full 48-head config and
the per-GPU 24-head split:

```
[bitwise] tN ssm state (chained)   393216 / 786432 values   exact
[bitwise] tN conv state row         15360 / 30720 values    exact
[bitwise] tN gdn output              3072 / 6144 values      exact
```

The declared fp32 fold order (`docs/SEMANTICS.md`: per-lane sequential
accumulation of Σ S[i][col]·k[i], binary-tree `__shfl_xor` reduce for
`kv_col`, `delta = (v − g·kv)·beta`, fused `s = g·s + k·delta`, output
accumulated against the just-updated state, `1/√S_v` scaling last, one warp
per output column) is reproduced exactly by the device op, so the CPU
reference and the kernel agree at the bit. The pre-step gate/norm/conv ops
that legitimately reduce in a different order than a serial CPU loop are
held to ≤1e-5 rel and measured far tighter:

```
[<=1e-5] gated_rmsnorm   worst rel 3.7e-7
[<=1e-5] conv+silu       worst rel 2.2e-7
[<=1e-5] l2norm q / k    worst rel 2.3e-7
[<=1e-5] gates g / beta  worst rel 5.3e-7 / 1.6e-7
```

## Registers

`mk-test-gdn` op wrappers compile at **64 registers**, 0 spills — inside the
168 envelope with wide headroom, consistent with the 104-reg full
interpreter (the GDN ops are not the register-binding family).

## Note

Bit-exactness here is against the DECLARED fold order, which is what G13/G21
require — not against an arbitrary reordering. The instantiation spec
records this fold order; any future kernel that changes it re-derives its
golden.
