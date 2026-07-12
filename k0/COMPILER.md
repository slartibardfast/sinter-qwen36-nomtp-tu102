# The k=0 schedule compiler

`compile_schedule.py` reads the extracted decode graph (`schedule.csv`,
`leaves.csv`) and emits `program.json` — the symbolic interpreter program a
C++ packer later resolves to device pointers and packs into 128-byte
`mk::Instr` structs (`core/isa.cuh`).

## Output of record (2026-07-12)

- **Coverage: 3704/3704 nodes** accounted (emitted or documented-dropped).
- **2248 instructions**, of which **964 are OP_BOUNDARY** (v0 placement —
  see below).
- **Weight stream 16.379 GB/pass** (confirms the extraction's 16.37 GB;
  the design package's 16.69 figure is 0.32 GB high — it counted something
  the decode graph does not read per step, likely blk.64 or the embedding
  matrix).
- **DRAM/pass 18.861 GB at n_kv 33024, 33.884 GB at n_kv 262144** (the deep
  KV read overtaking the weight stream is the deep-context red-line row).
- **Scratch buffers 6.37 MB** total (table in the run output / program.json
  `buffers`), reused across blocks.

## Documented drops (single-sequence k=0 binding)

- the per-block zero-extent state-clear chains and dead `rs_index_setup`
  views (QUESTIONS 14/15) — they move zero bytes at batch-1 single-seq;
- `mask_convert` — the harness/backend writes the f16 KQ mask directly
  (`binding.mask` in program.json), so the f32→f16 CPY is host-side.

## RESOLVED: boundary placement is v0, and it is (almost) forced

The antichain-coalescing pass now exists (`compile_schedule.py`, the
`# boundary coalescing` block): two adjacent windows merge iff their union
stays an antichain (no WAR/WAW/RAW across the union of each window's
`reads=`/`writes=`), with an occupancy guard (no two heavy weight-streaming
ops per window) and XCHG sites excluded.

It removes **zero** boundaries: an exhaustive scan finds **0 of the 1219
adjacent window pairs independent**, at single-GPU and dual-GPU. The v0
schedule is a genuine, fully-serial dependency chain — logically-independent
ops false-share the reused scratch buffers (`xn`, `q8_act`, `mixer_out`,
`attn_out`, `proj_out`, `gdn_*`), so nearly every adjacent pair has a real
hazard. The "~320-380 boundaries" target assumed independence that
buffer-reuse eliminates; reducing boundaries needs **buffer renaming** to
break the false-sharing, not just coalescing. The boundary total is ~1.29 ms
(6 % of the dual-GPU pass) and is data-dependency-required. Full itemization
and the true bound: `k0/REDLINE.md`.

## Constraint: FATTN KV chunks must be 32-aligned

The `OP_FATTN_DECODE` mask read (`attn.cuh`) loads two f16 mask entries as
one 32-bit word, so a chunk's KV start must be even (32-aligned in
production). The parity test confirmed an odd chunk start faults with
`misaligned address`. When the compiler splits KV across the participating
blocks for `OP_FATTN_DECODE`, every chunk boundary MUST be a multiple of 32
— this is a hard scheduling obligation, not a soft preference. (Attn NOTES,
`k0/ops/NOTES-attn.md`.)

## Runtime symbols

`n_kv` (padded mask width and KV window) and the per-pass row indices are
marked runtime symbols the host patches per pass, not baked constants
(program.json `meta.runtime_symbols`), so one compiled program serves every
context depth up to the bound n_ctx.

## Reproduce

`python3 k0/compile_schedule.py` (pure Python, deterministic; writes
`k0/program.json`).
