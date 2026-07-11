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

## OPEN: boundary placement is v0 (drives G15)

The 964 boundaries are the conservative "boundary between adjacent
dependent instructions" placement; it fails G15 (2 % gate) at the measured
806 ns/boundary (5.7 % of budget — `k0/ops/NOTES-spine.md`). The compiler
needs an antichain-coalescing pass: two instructions may share one boundary
window iff neither reads a buffer the other writes (the buffer read/write
sets are already tracked per instruction — `reads=`/`writes=`). Target
~320–380 boundaries. This is the next single-GPU-milestone task, tracked in
`plan/0135 mtp-off-build.md`.

## Runtime symbols

`n_kv` (padded mask width and KV window) and the per-pass row indices are
marked runtime symbols the host patches per pass, not baked constants
(program.json `meta.runtime_symbols`), so one compiled program serves every
context depth up to the bound n_ctx.

## Reproduce

`python3 k0/compile_schedule.py` (pure Python, deterministic; writes
`k0/program.json`).
