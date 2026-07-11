# GEMV op family — verification notes

Companion to `k0/ops/gemv.cuh`; parity test `tests/test_gemv.cu`. Built with
`cmake --build /tmp/mk-gemv --target mk-test-gemv` (CUDA 13.3, sm_75), run on a
Quadro RTX 6000 (TU102). All six ops exercised from a 72×384 wrapper kernel,
one Instr per launch, dynamic smem slab sized per op.

## Per-op ptxas resource accounting (`-Xptxas=-v`, sm_75)

| op | registers | barriers | spills | static smem | dynamic smem slab |
|----|-----------|----------|--------|-------------|-------------------|
| `op_quant_q8_1`       | 22 | 0 | 0 | 0 | 0 B |
| `op_mmvq_q4_0`        | 55 | 1 | 0 | 0 | `(K/32)·36` B |
| `op_mmvq_q4_0_fused`  | 60 | 1 | 0 | 0 | `(K/32)·36` B |
| `op_mmvq_ar16`        | 49 | 1 | 0 | 0 | `(K/32)·36` B |
| `op_gemv_f16`         | 47 | 1 | 0 | 0 | `K·4` B |
| `op_head_gemv_f16`    | 47 | 1 | 0 | 0 | `K·4` B |

Combined `mk_run` switch (all six inlined): **60 regs** (= the fused op, the max),
0 spills. Isolated via per-op `probe_*` entry kernels so ptxas prints one line
each. No register spills anywhere; the slab is entirely dynamic (staged
activations), so ptxas static smem is 0 and the real budget is the per-launch
dynamic amount above. Largest slab in this suite: `op_gemv_f16` at K=5120 →
20480 B, well under the interpreter's 60 KiB / the 48 KiB default-opt-in cap.

## Parity result (worst relative error)

Every check ran green. dp4a-integer and f16/f32-fold paths come out
**bitwise-exact** against a CPU reference that reproduces each op's declared
fold order (per-lane sequential `std::fmaf` == `__fmaf_rn`, scale muls pinned
through `volatile` == `__fmul_rn`, the 32-lane `__shfl_xor` butterfly
simulated, and f16↔f32 through the same `__half2float`/`__float2half_rn`):

| op | worst rel | note |
|----|-----------|------|
| `quant_q8_1` q (int8)   | exact | integer, bit-for-bit |
| `quant_q8_1` d, s (f16) | 0 | bitwise-exact (butterfly sum simulated) |
| `mmvq_q4_0` (K=5120 tail, K=2048 tail-free, sub-range [8,40)) | 0 | bitwise-exact |
| `mmvq_q4_0_fused` (K=5120) | **1.91e-07** | silu transcendental (`expf`), only non-exact op |
| `mmvq_ar16` (K=6144 tail-free, K=1056 2-block tail) | 0 | bitwise-exact |
| `gemv_f16` (K=5120) | 0 | bitwise-exact |
| `head_gemv_f16` (K=5120) | 0 | bitwise-exact |

Suite worst rel = **1.91e-07** (the fused silu). The dp4a integer sub-sums are
exact by construction (integer add is order-free); their being *bitwise*-exact
end-to-end confirms both the integer path and the surrounding scale fold.

Coverage notes baked into the sizes:
- Q4_0 K=5120 has a lane-balanced tail (`npair=80` → 64 full + 16 tail pairs);
  K=2048 is tail-free (`npair=32`). The sub-range `[8,40)` verifies absolute-row
  `w`/`dst` indexing and that out-of-range `dst` rows stay untouched (sentinel).
- AR16 K=1056 forces `npair=33` → 32 full + AR16 blocks 64,65 through the tail
  loop, exercising the `kb`-parity q8-half pick (`i8 = 1 + 4·(kb&1)`) for both
  the even and odd tail block.
- quant ne00=5000 straddles a block (block 156 partially past ne00) and forces
  block 0 all-zero to hit the `amax==0 → q=0, d=0, s=0` padding path.

## Bandwidth (contended-indicative, TU102)

Repeated-pass weight-stream rate (dominant DRAM traffic; the staged q8/x is a
one-time per-block smem fill, negligible against the row stream):

| op | shape | stream | measured |
|----|-------|--------|----------|
| `mmvq_q4_0`      | 8704 rows × K=5120 | 25.1 MB × 300 | **519.8 GB/s** |
| `op_head_gemv_f16` | 61440 rows × K=5120 | 629 MB × 40 | **598.7 GB/s** |

Quadro RTX 6000 practical DRAM peak is ~600–620 GB/s (672 GB/s theoretical,
384-bit GDDR6 @ 14 Gbps). The head GEMV at 598.7 GB/s is ~96% of practical peak
— the vectorized `uint4` (8-half) weight load clears the 70–75%-of-peak scalar-
load trap (the G12 lesson, reference/tu102 `dequant_gemv`). The Q4_0 figure is
lower because its 25 MB stream partly lives in the 6 MB L2 across the 300 tiny
back-to-back launches (per-launch overhead also dilutes it); it is a lower bound
on the stream rate, still far above the scalar trap. Both are contended-
indicative (shared rig).

## Dossier corrections

**None.** The op header's fork cites were re-verified against
`software/llama.cpp/autoround @ 546eca8dc` and all hold:
- `quantize_q8_1` `quantize.cu:31-47`: `xi = i0<ne00 ? x : 0`, `amax =
  warp_reduce_max<32>`, `sum = warp_reduce_sum<32>(xi)` (raw Σx, **not** d·Σqs),
  `d = amax/127`, `q = amax==0 ? 0 : roundf(xi/d)`, `ds = make_half2(d, sum)`.
  The header's `__floats2half2_rn(d, sum)` matches `make_half2` (both round-to-
  nearest). Confirmed.
- `vec_dot_q4_0_q8_1_impl` `vecdotq.cuh:115-134`: split-half nibbles (`>>0`/`>>4`
  masked `0x0F0F0F0F`), return `d4·(sumi·d8 − (8·vdr/QI4_0)·s8)` with
  `8·vdr/QI4_0 = 8·2/4 = 4` per half-block call. The header's whole-block
  `−8·s8` is the correct aggregate of the fork's two half-block `−4·s8` terms
  (both `.y = s8 = Σx`, so `w·x = d4·(d8·sumi − 8·Σx)`). Confirmed bitwise.
- `vec_dot_q4_0_ar16_q8_1` `vecdotq.cuh:755-780`: element-interleaved nibbles
  (`byte j = code[2j] | code[2j+1]<<4`), `−8` folded in `unpack_q4_0_ar16` via
  `__vsubss4` (never saturates: nibble−8 ∈ [−8,7]), **no** s-correction, per-
  block `d4·d8·sumi`, `i8 = 4·(kbx&1)` half-pick. The header's pair grouping
  `d8·(dA·s0 + dB·s1)` is a different (legal) rounding grouping of the same math;
  it reproduces the device bitwise. Confirmed.
- Block layouts `ggml-common.h:187-269`: q4_0 18 B, q4_0_ar16 10 B (QK 16),
  q8_1 36 B (`ds` half2 = {d, s}). Confirmed.
- F16 GEMV `mmvf.cu`: f16 weights, f32 activations. The header deliberately
  uses fp32 accumulation instead of the fork's `half2` accumulator (strictly
  more accurate; G13 does not bind this fold). The test's reference matches the
  header's fp32 fold, not the fork's half2 path — bitwise-exact against the op.

The CPU reference encodes these fork details directly; its bitwise agreement
with the device op is independent confirmation that both the header and the
dossier are right for this family. No header edit was required.
