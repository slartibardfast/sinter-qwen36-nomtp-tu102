------------------------------ MODULE MegakernelSync ------------------------------
(* The megakernel playbook's parametric sync-protocol model.                      *)
(*                                                                                *)
(* ONE model for every instantiation: speculation depth and drafter presence are  *)
(* CONSTANTS, so the same invariants are checked at k = 0 (the MTP-off worked     *)
(* example) and k = 3 (the production instantiation). A protocol that encodes MTP *)
(* assumptions structurally fails the k = 0 instance — the anti-overfit tripwire  *)
(* at the concurrency level, mirroring playbook.allium at the behavioural level.  *)
(*                                                                                *)
(* Abstraction: workers are the coarse agents of the design (one per GPU's SM     *)
(* fleet region participating in a pass), not 72 modelled SMs. Banks are the      *)
(* recurrent-state banks of the banked-state design (gate G21/A02). The model     *)
(* covers the pass loop: command -> [draft] -> verify -> decide -> rollback or    *)
(* commit -> completion. Gates bound: G31 (raced rollback: RollbackSafe,          *)
(* NoStaleRead), G21's concurrency half (BankIsolation), Y02-style progress.      *)
(*                                                                                *)
(* DRAFT status: the Y-row litmus mapping (sync_protocol.csv) refines the fence   *)
(* granularity; instantiations bind their own worker counts. Checked instances:   *)
(* MegakernelSync_k0.cfg, MegakernelSync_k3.cfg. The memory-operation refinement  *)
(* of this model's edges lives beside it: SignalEdge.tla (fenced-flag vs          *)
(* sentinel signaling under the measured sm_75 semantics) and WaitCount.tla       *)
(* (the runtime-written-count obligation), each with safe and must-fail           *)
(* instances.                                                                     *)

EXTENDS Naturals, FiniteSets

CONSTANTS
    SpecDepth,      \* drafted tokens per pass: 0 for the MTP-off instantiation
    HasDrafter,     \* BOOLEAN: a drafting agent participates in the pass
    Workers         \* the set of worker agents (per-GPU SM fleets)

ASSUME SpecDepth \in 0..3
ASSUME HasDrafter = (SpecDepth > 0)
ASSUME Workers # {}

Banks == 0..SpecDepth              \* bank 0 is the base (committed) lineage
Positions == 0..SpecDepth          \* verified positions in one pass

VARIABLES
    phase,          \* the pass phase: "idle", "drafting", "verifying", "deciding",
                    \*                 "rollback", "commit"
    accepted,       \* decided acceptance count for the current pass (0..SpecDepth)
    bankState,      \* [Banks -> {"committed", "speculative", "rolling_back", "free"}]
    readers,        \* [Workers -> Banks \cup {NoBank}] bank each worker currently reads
    passes          \* committed pass counter (progress metric, bounded in cfg)

NoBank == SpecDepth + 1

TypeOK ==
    /\ phase \in {"idle", "drafting", "verifying", "deciding", "rollback", "commit"}
    /\ accepted \in 0..SpecDepth
    /\ bankState \in [Banks -> {"committed", "speculative", "rolling_back", "free"}]
    /\ readers \in [Workers -> Banks \cup {NoBank}]
    /\ passes \in Nat

Init ==
    /\ phase = "idle"
    /\ accepted = 0
    /\ bankState = [b \in Banks |-> IF b = 0 THEN "committed" ELSE "free"]
    /\ readers = [w \in Workers |-> NoBank]
    /\ passes = 0

(* A decode command arrives: the pass begins. With a drafter the draft   *)
(* phase populates speculative banks; without one the pass goes straight *)
(* to verification of the single position.                               *)
BeginPass ==
    /\ phase = "idle"
    /\ phase' = IF HasDrafter THEN "drafting" ELSE "verifying"
    /\ UNCHANGED <<accepted, bankState, readers, passes>>

Draft ==
    /\ phase = "drafting"
    /\ bankState' = [b \in Banks |-> IF b > 0 THEN "speculative" ELSE bankState[b]]
    /\ phase' = "verifying"
    /\ UNCHANGED <<accepted, readers, passes>>

(* Workers attach to banks during verification. A worker may read the    *)
(* committed base or a speculative bank of the CURRENT pass; it must     *)
(* never attach to a bank being rolled back (NoStaleRead).               *)
AttachReader(w, b) ==
    /\ phase = "verifying"
    /\ bankState[b] \in {"committed", "speculative"}
    /\ readers' = [readers EXCEPT ![w] = b]
    /\ UNCHANGED <<phase, accepted, bankState, passes>>

DetachReader(w) ==
    /\ readers[w] # NoBank
    /\ readers' = [readers EXCEPT ![w] = NoBank]
    /\ UNCHANGED <<phase, accepted, bankState, passes>>

(* Verification completes; the acceptance outcome is decided. Every      *)
(* outcome 0..SpecDepth is reachable (G22's fuzzer domain).              *)
Decide(a) ==
    /\ phase = "verifying"
    /\ a \in 0..SpecDepth
    /\ accepted' = a
    /\ phase' = IF a < SpecDepth THEN "rollback" ELSE "commit"
    /\ UNCHANGED <<bankState, readers, passes>>

(* Rejected banks roll back. The GUARD is the protocol's load-bearing    *)
(* obligation: rollback may not begin while any worker still reads a     *)
(* bank that will be invalidated (G31's raced rollback).                 *)
Rollback ==
    /\ phase = "rollback"
    /\ \A w \in Workers: readers[w] = NoBank \/ readers[w] <= accepted
    /\ bankState' = [b \in Banks |->
                        IF b > accepted /\ bankState[b] = "speculative"
                        THEN "rolling_back" ELSE bankState[b]]
    /\ phase' = "commit"
    /\ UNCHANGED <<accepted, readers, passes>>

(* Accepted speculative banks fold into the committed lineage; rolled-   *)
(* back banks return to the pool. The pass completes.                    *)
Commit ==
    /\ phase = "commit"
    /\ \A w \in Workers: readers[w] = NoBank
    /\ bankState' = [b \in Banks |->
                        IF b = 0 THEN "committed"
                        ELSE IF b <= accepted /\ bankState[b] = "speculative"
                             THEN "free"    \* folded into base, slot recycled
                        ELSE IF bankState[b] = "rolling_back" THEN "free"
                        ELSE bankState[b]]
    /\ phase' = "idle"
    /\ passes' = passes + 1
    /\ UNCHANGED <<accepted, readers>>

vars == <<phase, accepted, bankState, readers, passes>>

ControlNext ==
    \/ BeginPass
    \/ Draft
    \/ \E a \in 0..SpecDepth: Decide(a)
    \/ Rollback
    \/ Commit

DetachAny == \E w \in Workers: DetachReader(w)

Next ==
    \/ ControlNext
    \/ \E w \in Workers, b \in Banks: AttachReader(w, b)
    \/ DetachAny

(* Fairness: control actions and detaches are weakly fair; attaches are   *)
(* not (a worker may re-attach forever DURING verification, but Decide    *)
(* stays continuously enabled there, so WF(ControlNext) still fires it;   *)
(* after Decide no new attach is possible, so WF(DetachAny) drains the    *)
(* readers and the pass completes).                                       *)
Spec == Init /\ [][Next]_vars /\ WF_vars(ControlNext) /\ WF_vars(DetachAny)

(* TLC state bound: the pass loop repeats identically, two passes suffice. *)
PassBound == passes <= 2

(* ----------------------------- Invariants ------------------------------ *)

(* G31: no worker ever holds a read on a bank in rollback.                  *)
NoStaleRead ==
    \A w \in Workers: readers[w] = NoBank \/ bankState[readers[w]] # "rolling_back"

(* G21 concurrency half: outside a pass, only the committed lineage lives.  *)
BankIsolation ==
    phase = "idle" => \A b \in Banks: bankState[b] \in {"committed", "free"}

(* The base lineage is never speculative and never rolled back.             *)
BaseIntegrity ==
    bankState[0] = "committed"

(* At k = 0 the speculative machinery is structurally absent: no bank       *)
(* other than the base ever exists, and rollback is unreachable. This is    *)
(* the degeneracy obligation the playbook demands of every instantiation.   *)
DegenerateAtZero ==
    SpecDepth = 0 => (phase # "rollback" /\ phase # "drafting")

(* ------------------------------ Liveness ------------------------------- *)

(* Every begun pass eventually commits (bounded by the cfg's pass ceiling). *)
Progress == []((phase # "idle") ~> (phase = "idle"))

===================================================================================
