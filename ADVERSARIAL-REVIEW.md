# Adversarial review — the k=0 megakernel against its two goals

Reviewer stance: refute, not summarize. Goals judged:
- **(a)** a *reproducible methodology* a different session could follow to write a *different* megakernel;
- **(b)** *100% of the TU102 hardware* (RTX 6000 sm_75, 2× GPU, ~608 GB/s DRAM, ~114 TFLOP/s fp16 tensor).

Everything below was checked against the tree at megakernel HEAD `9376813`, calx-mill `c9c7077`
(matches its `.host-software` pin), on the live rig. Empirical results are pasted with the command
that produced them. Solid parts are conceded at the end of each finding where they exist, so the
damage lands where it is real.

---

## Verdicts (one honest sentence each)

- **(b) 100% TU102 performance:** NOT met and not close — the full decode pass runs at **63.8% of
  the k=0 DRAM red-line** (21.48 ms vs 13.71 ms), the delivered 45.95 tok/s edges the plain stock
  floor by ~5% but is **20–35% *below* the production incumbent** it must eventually beat (57.6 tok/s
  `-sm tensor`+MTP-d3; ~71 tok/s mainline MTP), because the single biggest lever the whole plan is
  named after — MTP speculation — is entirely unbuilt.
- **(a) reproducible megakernel methodology:** partially real but oversold — the *build* artifacts
  reproduce and the spec/gate scaffolding exists, but the deliverable is absent from the
  reproducibility ledger it claims to obey, the integration seam is unimplemented, and the "playbook"
  has been executed exactly once, on the degenerate config, so its generality is asserted rather than
  demonstrated.

---

## Findings, most-damaging first

### 1. The whole point — MTP — is unbuilt, and the delivered kernel is SLOWER than production. Defeats (b).

- **Claim as framed:** commit `ea4cfe7` "dual-GPU tensor-parallel decode runs, beats the floor (44.08 tok/s)";
  REDLINE.md "45.95 tok/s … The true bound today." The plan is titled *"Persistent decode megakernel
  (Qwen3.6-27B + native MTP, 2× TU102)"* and its production target is **tensor-split + MTP depth 3**.
- **What I found:** k=0 has **zero speculation** — it is, by the docs' own words, "the degenerate
  no-speculation case" (`mtp-off-build.md`: G20–G24 "vacuous at k=0 by degeneracy"). The delivered
  number is therefore being compared to the *plain* stock floor (43.9), the one comparator that also
  has no speculation. Against the configs that matter:
  - k=0 megakernel: **45.95 tok/s** (dual-GPU, no MTP)
  - stock `-sm tensor` + MTP d3: **57.6 tok/s** (FACEOFF.md rubric tie-break) → k=0 is **−20%**
  - mainline MTP incumbent: **71.1 tok/s** (user MEMORY, `plan-0135-faceoff-pivot`) → k=0 is **−35%**
- **Evidence:** `grep GGML_BACKEND_DL/ggml_backend_reg/graph_compute` over `core/` `k0/` → **no hits**;
  S-rows (the MTP pipeline schedule, S00–S28) and A-rows (agentic, A00–A15) have **zero references in
  the kernel source**; the entire speculation stage G20–G24 is `pending` with empty evidence in
  `gate_results.csv`. The 45.95 figure is REDLINE.md §3; 57.6/71 are FACEOFF.md line 91 and MEMORY.
- **Severity: defeats (b).** Presenting a no-speculation kernel that *ties the plain floor* as
  "beats the floor" is the central self-deception. On any production metric the delivered artifact is
  a regression, not a 100%-of-hardware result. The k=0 kernel is a legitimate *worked example*; it is
  not a performance result, and the celebratory commit framing invites the misread.

### 2. "100% of the hardware" is ~64% of the DRAM red-line; the framing leans on the "hardware-limited" excuse CLAUDE.md forbids. Dents/defeats (b).

- **Claim:** REDLINE.md "the k=0 red-line is a bandwidth wall, and the matmuls are essentially against
  it … The true bound today is 21.48 ms / 45.95 tok/s."
- **What I found (arithmetic re-derived):**
  ```
  GEMV GB/s: 545.8  => % of 608.6: 89.7      # 8.35 GB / 15.30 ms  (they say 90% — OK)
  red-line ms: 13.72                          # 8.35 GB / 608.6 GB/s
  tok/s at 13.71ms: 72.94
  45.95/73 = 62.9 % of k0 red-line
  21.48ms achieved vs 13.71 red-line = 63.8 % roofline efficiency
  ```
  So only the **isolated weight-streaming GEMVs** approach the roofline (90–98%). The **full pass is
  63.8% of the red-line.** The other 36% is 7.77 ms of gap, and REDLINE.md's own itemization attributes
  ~3.3 ms of it to **design-imposed** overhead: interpreter loop/dispatch (1.0 ms, "tied to instruction
  count"), 1220 grid-sync boundaries (1.29 ms, needs buffer renaming), cross-GPU fp32 reduce (0.96 ms).
  The doc concedes these are "the cost of *this* interpreter+split design," i.e. **not the hardware** —
  a different megakernel design would not pay them. A further ~1 ms is called "genuinely closable" —
  FATTN depth-adaptive split (~0.5 ms, "the biggest remaining honest lever") and RMSNORM multi-block
  (~0.2 ms) — and then **declined** "at this margin" for parity risk.
- **Severity: dents (b), and partly defeats it on honesty.** REDLINE is far more honest than most of
  this project — it itemizes, and it does not claim red-line — so it is not a *pure* hardware-limited
  dodge. But "bandwidth wall … matmuls essentially against it" is the top-line takeaway, and it papers
  over that (i) a third of the miss is the project's own interpreter/split design and (ii) the single
  biggest declared-closable lever was left unclosed. For a goal literally defined as "red-line the
  hardware," declining the 0.5 ms FATTN win because it is "risky at this margin" is exactly the reflex
  CLAUDE.md names ("the moment you reach for 'hardware-limited' … is when to instrument and verify").

### 3. The megakernel deliverable is NOT in `.host-software`, yet claims greenfield-mandatory reproducibility against it. Dents (a).

- **Claim:** README.md "Reproducible-build discipline (greenfield: mandatory, no exemption): the
  recorded recipe in the host's `.host-software` must rebuild the artifact byte-identically";
  CMakeLists.txt line 9 repeats it.
- **What I found:** `grep -c megakernel .host-software` → **0**. There is no `[software "megakernel"]`
  stanza, no `pin`, no `artifact` hash, no `deploy`, no `--check`/`--verify-build` coverage. The
  headline deliverable of the whole effort is outside the audit ledger the CLAUDE.md production-anchor
  section makes mandatory. `mtp-off-build.md` even lists the missing stanza as an open box ("[ ] Remote
  + `.host-software` stanza"). So the reproducibility *claim* is asserted against a ledger entry that
  does not exist.
- **Empirical mitigation (conceded):** the *property* does hold locally — I ran the double-build:
  ```
  $ bash tools/repro-check.sh <src>
  repro-check: byte-identical
  60fbebac885ec1c084d6ba69129723491597476c6c39eb937a8efe089a74c8d6  ./mk-harness
  … (11 artifacts, two from-scratch CUDA builds, identical)   REPRO-CHECK EXIT: 0
  ```
  Two clean nvcc builds are byte-identical. But byte-identity between two builds *on this host now* is
  not an anchor: there is no recorded hash to catch source drift, and no CI gate binds it.
- **Severity: dents (a).** The reproducibility discipline is real as a *script* and passes; it is
  *not* real as an *auditable anchor*, and the README/CMakeLists overstate it as satisfied.

### 4. The integration stage is entirely unimplemented — the "worked example" does not meet its own "fully working" bar. Dents (a) and (b).

- **Claim:** README/FACEOFF/`mtp-off-build.md` bar: *"every k=0-instantiated gate green **through the
  integration stage** under an unmodified `llama-server`."* README: "run underneath an unmodified
  `llama-server` as an out-of-tree dynamic ggml backend … `libggml-megakernel.so` (`GGML_BACKEND_DL`)."
- **What I found:** there is no backend. No `GGML_BACKEND_DL`, no `ggml_backend_reg`, no
  `graph_compute`, no `libggml-megakernel` target anywhere in `core/`, `k0/`, or `CMakeLists.txt`.
  `mk-harness` is a **standalone** driver that mmaps the GGUF directly (`k0/gguf.h`) and launches the
  persistent kernel itself. The integration box in `mtp-off-build.md` is `[ ]` open; LG0–LG5 appear in
  no results ledger. So the kernel has **never run as a backend under llama-server**, never been
  perplexity-parity'd (LG2), never soaked at the server level (LG5).
- **Severity: dents (a) and (b).** The face-off's "iteration zero fully working" precondition is not
  met — the deliverable is a bench/parity harness, not the serving artifact the plan describes. Every
  end-to-end claim is a synthetic-harness claim, not a served-token claim.

### 5. Gate discipline is ~18% realized (7 of 38); every release-critical stage is 0% run. Dents (a).

- **Claim:** README "the release discipline (the gate manifest, six stages)"; `gate_manifest.csv`
  advertises G00–G54 + LG0–LG5 as "the release discipline."
- **What I found (counted from `gate_results.csv` + `gate_manifest.csv` + plan boxes):**
  - **Green with binding verification: 7** — G05, G11, G13, G43 (in `gate_results.csv`) + G00, G01, G02
    (prereq, closed in the tu102 substrate work, per git log; **not recorded in `gate_results.csv`** —
    the ledger is split across two homes).
  - **Partial: 2** — G12 (73% single-GPU, not the dual per-GPU-half number), G40 (4K/64K pass, 256K proxy).
  - **Pending / empty-evidence / unbuilt: 29** — including the **entire** speculation stage
    (G20–G24), **entire** agentic stage (G50–G54), most of dual-GPU (G30–G34), long-ctx acceptance
    (G41, G42), and **all six** integration gates (LG0–LG5, which aren't even in the manifest CSV —
    they're "proposed").
  - `mtp-off-build.md` reclassifies ~15 of those as "vacuous at k=0" or "inherited by construction."
    That is defensible for a k=0 example, but it means the manifest as a *release mechanism* is
    exercised on ~7 gates, declared-N/A on ~15, and pending on ~7.
- **Evidence:** `gate_results.csv` has 20 rows literally `pending` with empty `evidence`. The
  `evidence` column is hand-authored prose, not mechanical output; `check_gates.py` only auto-verifies
  stage-0 from the ops table.
- **Severity: dents (a).** "Gate discipline" is real for the handful of k=0 substrate gates and
  aspirational for the ~80% that would validate the production value (MTP uplift, agentic, server
  integration). Calling the manifest "the release discipline" driving the work overstates it.

### 6. q8 KV — the single biggest deep-context lever — is unbuilt and unmeasured for accuracy. Dents (b).

- **Claim:** `precision_ledger.md` KV-cache-dtype row: "q8 halves the deep KV read → **~+45% at 256K
  (16.28→~24 tok/s)**."
- **What I found:** same row, accuracy column: "**UNMEASURED — THE GAP**"; decision "f16 (pending the
  number)." The perf win is known (user MEMORY: q8 shallow 61.8 t/s, frees 5 GB) but it is **neither
  integrated into the megakernel nor accuracy-validated**. The ledger itself flags this as "the
  concrete next measurement." So the largest deep-context performance the effort could claim is left
  on the table with a note.
- **Severity: dents (b).** For a 100%-of-hardware goal at the 256K config this milestone spent real
  effort on (G43 soak), the dominant lever is deferred.

### 7. The deep-context result has no baseline, and 256K parity is proxy-only. Dents (b).

- **Claim:** G43 `pass` — "256K FITS dual-GPU; deep floor **16.28 tok/s** / 62.0 ms per pass"; G40
  "256K proxy-validated."
- **What I found:** there is **no stock/reference 256K decode tok/s anywhere** to compare 16.28
  against (grep over `reference/megakernel`, `plan/0135*`, `software/megakernel` for a stock deep
  number returns nothing). The G43 soak proves the kernel *fits and does not leak* at 256K — a real
  and useful result — but 16.28 tok/s is a floor with nothing to beat. And literal 256K token-parity
  was **never run**: G40's own note says full-prefill 256K is "runtime-bound (… 256K ~hours)," so the
  close is the 4K padding-class sweep + the soak + "the depth-invariant mechanism," i.e. a proxy.
- **Severity: dents (b).** A deep-context "win" with no opponent and a proxied correctness close.

### 8. G13 "pass" glosses that `compare.py` returns FAIL/exit 1, and the dual-GPU RMS lane misses tolerance 5×. Dents (a) honesty.

- **Claim:** `gate_results.csv` G13 `pass`; `PARITY.md` "The two scored numerical-parity lanes pass."
- **What I found:** `PARITY.md` pastes the actual tool verdict — `RESULT: FAIL … compare.py exits 1`
  — because the state lane is 384/384 not-bit-exact and tokens diverge at step 4. On the dual-GPU
  config the **activation-RMS lane exceeds tolerance by ~5×** (max 9.75e-2 vs 0.02 — `mtp-off-build.md`
  line 185, `PARITY-TENSOR-VERIFY.md`). The gate is recorded green by *reclassifying* the two failing
  lanes as "not the guard."
- **Conceded:** the reclassification is genuinely supported. I checked the reference's own cross-check:
  ```
  $ cat /var/tmp/mk-oracle/cross-none.json
  "state_files_compared": 384, "state_mismatched": 384,
  "kl_max": 0.00250…, "residual_rms_rel_max": 0.00853…,
  "token_divergence": { "position": 4, … "within_band": true }
  ```
  The fork's *own* tensor-vs-none comparison gives the identical 384/384 mismatch and the same
  within-band step-4 tie — so "state bit-exactness fails definitionally across fold orders" is a real
  argument, not invented. (Caveat: that cross-check also differs in `n_ctx` 262144 vs 8192, so it is
  not a perfectly clean fold-order isolation.)
- **Severity: dents (a) honesty.** The *result* (greedy tokens match, logits KL in band) is real and
  arguably more reference-faithful than stock. But "G13 pass" for a run whose tool prints `RESULT: FAIL`
  and whose dual-GPU RMS lane blows through tolerance 5× is the reactive-honesty pattern: the failure
  is disclosed only in the body, the ledger says pass.

### 9. The ~120-code identifier system is largely decoration relative to what shipped. Dents (a).

- **Claim:** the coded spec system (G/LG gates, Y sync rows, M memory rows, S pipeline rows, A agentic
  rows) presented as a reproducible spec substrate.
- **What I found (grep for each family in `core/`+`k0/`):** S-rows (29 codes) → **0** source refs;
  A-rows (16) → **0**; M-rows (29) → **0**; Y-rows (13) → only **Y02, Y03, Y06** load-bearing (31/9/7
  refs), the other 10 are documentation. So of ~120 codes, the load-bearing set in the shipped kernel
  is a handful of Y-rows plus the ~7 run gates. The S (MTP pipeline) and A (agentic) families formally
  specify features that do not exist yet.
- **Severity: dents (a), mostly cosmetic.** The load-bearing codes are real and useful; the bulk is
  formalization of the unbuilt production/agentic kernels — over-formalization a human likely would not
  reproduce, and easy to mistake for realized structure.

### 10. Bookkeeping drift — the plan's "honest checkbox state" lags its own numbers. Cosmetic, dents (a).

- **What I found:** `mtp-off-build.md` still records the headline as "**44.08 tok/s / 22.35 ms**
  … Contended-indicative" with "[ ] Triplicate clocks-locked scored run" **open**, while the newer
  REDLINE.md (commit `9376813`) reports a **clock-locked mean-of-3 45.95 tok/s / 21.48 ms** after the
  RMSNORM fix. The plan box was not updated. The FACEOFF rubric explicitly prizes "honest checkbox
  state"; here the checkbox trails the measurement by a full lever.
- **Severity: cosmetic**, but it is the exact drift the audited-plan discipline exists to prevent.

### 11. The floor comparator is a standing number, not a same-session head-to-head. Cosmetic, dents (b).

- **What I found:** FACEOFF.md: "standing value 43.9 tok/s … re-confirmed 2026-07-12: mainline with
  the ported AR16 reads 43.47 on the same quant." REDLINE.md's floor rows are the *megakernel's* own
  triplicate; the stock 43.9/43.47 is **not** re-measured in the same session on the identical config.
  So "+4.7%" compares a freshly-measured mk against a standing stock number that itself floats between
  43.47 and 43.9.
- **Severity: cosmetic**, but it softens the already-small win.

---

## What is solid (conceded — credibility requires it)

Empirically verified this session, all passing:

- **program.json / program-split.json are byte-reproducible from `compile_schedule.py`:**
  ```
  $ python3 k0/compile_schedule.py && python3 k0/compile_schedule.py --split
  SINGLE: byte-identical    SPLIT: byte-identical
  ```
- **megakernel double-build is byte-identical** (finding 3 — the property holds even though the anchor
  is missing).
- **calx-mill container artifact reproduces its recorded hash:**
  ```
  $ podman --storage-driver vfs run --rm -v $PWD:/src:ro … rust@sha256:44637ff2… \
      cargo build --release --locked && sha256sum …/calx-mill
  68b0191ded86d338bfe6a91bb5bff2077e07328329d61bc7fda16c99b2394e26  calx-mill   ✓ matches
  ```
  **But** `host-lifecycle software --verify-build .` — the methodology's *own* gate — returns
  **exit 1** on this host: it invokes podman without `--storage-driver vfs` and dies with
  `graph driver "overlay" overwritten by … "vfs"`. So the automated reproducibility gate is red
  as-shipped; the hash only reproduces via a manual workaround. (`gguf-recast` reproduces cleanly
  through the same tool: `ok gguf-recast rebuild reproduces … @ 78129b99…`.)
- **calx-mill Kani proofs pass:** `Complete - 14 successfully verified harnesses, 0 failures, 14 total.
  KANI EXIT: 0`.
- **The GEMVs genuinely run at 90–98% of the DRAM roofline** (arithmetic re-derived above: 545.8 GB/s
  = 89.7%). The weight-stream half of the decode really is near the hardware bound.
- **The RMSNORM single-SM fix is a real, bit-identical +4.1% win** (1.60→0.73 ms/pass), and REDLINE's
  itemization is the most honest artifact in the set — it names the design-imposed overheads as such.
- **The G43 256K soak** (10K passes, watermarks byte-flat, no leak, clean exit) is a real result.
- **The state-lane "definitional" defense is supported** by the reference's own cross-check (finding 8).

## One-line bottom line

The effort produced a **reproducible-*building*, spec-scaffolded, numerically-plausible k=0 worked
example whose GEMVs approach roofline** — genuine engineering — but it is **not** a demonstrated
reproducible *methodology* (executed once, on the degenerate config, outside its own audit ledger,
with the integration seam and 80% of the gates unbuilt), and it is **nowhere near 100% of the
hardware** (64% of the k=0 red-line, and a 20–35% regression against the MTP production incumbent the
plan exists to beat, with the MTP and q8-KV levers untouched).
