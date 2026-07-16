------------------------- MODULE HandoverCarry -------------------------
(* The prefill -> decode state carry-over, no teardown (plan/0143 capstone:    *)
(* "Prefill->decode handover, no teardown (the capstone, TLA+)"). ONE resident *)
(* cooperative kernel runs prefill (fills the model state) then decode         *)
(* (consumes it) with NO relaunch between the phases -- so there is no kernel   *)
(* launch boundary to give the grid-wide happens-before for free. The state    *)
(* the decode phase reads (the DeltaNet recurrent banks, the KV rows, n_past)  *)
(* is EXACTLY what the prefill phase left: no teardown, no reinit. The only     *)
(* ordering that carries the state across the phase seam is an explicit         *)
(* grid-wide barrier (grid.sync / cooperative groups this_grid().sync()).       *)
(*                                                                            *)
(* The hazard this pins: in a cooperative kernel the blocks are independent     *)
(* agents. Prefill is data-parallel over blocks -- each block writes part of    *)
(* the state (its KV rows / its DeltaNet chunk). Decode is data-DEPENDENT on    *)
(* the WHOLE state: a decode block's attention reads KV rows written by EVERY   *)
(* prefill block. WITHOUT the handover barrier, a block that finishes its own   *)
(* prefill write may race ahead into decode and read state a SLOWER block has   *)
(* not written yet -- a premature consume of uninitialized / mid-write state.   *)
(* The grid.sync closes it: every block completes its prefill writes before     *)
(* ANY block begins a decode read (arrive-before-any-leave).                    *)
(*                                                                            *)
(* Knob: HandoverBarrier. With it the model is safe (NoPrematureConsume) and    *)
(* live; without it TLC MUST exhibit a decode block reading an unwritten slot.  *)
(* Grid scope, sibling to MegakernelSync.tla (the pass loop) -- this module is  *)
(* the phase seam between the two halves of the fused megakernel, not the       *)
(* speculative pass. Blocks is a CONSTANT set of cooperating blocks.            *)

EXTENDS Naturals

CONSTANTS
    Blocks,          \* the cooperating resident blocks (per-SM agents)
    HandoverBarrier  \* the grid.sync at the prefill -> decode phase seam

ASSUME Blocks # {}
ASSUME HandoverBarrier \in BOOLEAN

VARIABLES
    pfDone,      \* [Blocks -> BOOLEAN] block finished writing its prefill state slot
    atBarrier,   \* [Blocks -> BOOLEAN] block arrived at the handover grid.sync
    inDecode,    \* [Blocks -> BOOLEAN] block entered the decode phase
    read,        \* [Blocks -> BOOLEAN] block performed its decode state read
    sawUnwritten \* [Blocks -> BOOLEAN] the decode read observed a not-yet-written slot

vars == <<pfDone, atBarrier, inDecode, read, sawUnwritten>>

Init ==
    /\ pfDone = [b \in Blocks |-> FALSE]
    /\ atBarrier = [b \in Blocks |-> FALSE]
    /\ inDecode = [b \in Blocks |-> FALSE]
    /\ read = [b \in Blocks |-> FALSE]
    /\ sawUnwritten = [b \in Blocks |-> FALSE]

(* Prefill: a block writes its part of the model state (its KV rows / DeltaNet *)
(* chunk). No teardown -- this slot is what decode will later read.            *)
PrefillWrite(b) ==
    /\ ~pfDone[b]
    /\ pfDone' = [pfDone EXCEPT ![b] = TRUE]
    /\ UNCHANGED <<atBarrier, inDecode, read, sawUnwritten>>

(* A block arrives at the handover barrier once its own prefill write is done. *)
Arrive(b) ==
    /\ pfDone[b]
    /\ ~atBarrier[b]
    /\ atBarrier' = [atBarrier EXCEPT ![b] = TRUE]
    /\ UNCHANGED <<pfDone, inDecode, read, sawUnwritten>>

(* Enter decode. The grid.sync is the ONLY carry: WITH the barrier a block may *)
(* cross the seam only after EVERY block has arrived (all prefill writes are    *)
(* globally complete -- arrive-before-any-leave). WITHOUT it a block crosses    *)
(* as soon as its OWN prefill write is done, racing slower blocks.             *)
EnterDecode(b) ==
    /\ pfDone[b]
    /\ ~inDecode[b]
    /\ (HandoverBarrier => \A c \in Blocks: atBarrier[c])
    /\ inDecode' = [inDecode EXCEPT ![b] = TRUE]
    /\ UNCHANGED <<pfDone, atBarrier, read, sawUnwritten>>

(* Decode: the block reads the whole carried state. It is a premature consume  *)
(* if ANY block's prefill slot is not yet written when this read happens.      *)
DecodeRead(b) ==
    /\ inDecode[b]
    /\ ~read[b]
    /\ read' = [read EXCEPT ![b] = TRUE]
    /\ sawUnwritten' = [sawUnwritten EXCEPT ![b] = (\E c \in Blocks: ~pfDone[c])]
    /\ UNCHANGED <<pfDone, atBarrier, inDecode>>

(* The fused kernel has drained -- every block finished its decode read. An     *)
(* explicit stuttering step at the terminal state so completion is not         *)
(* reported as a spurious deadlock; a genuine mid-handover stall still          *)
(* surfaces as one.                                                            *)
Done ==
    /\ \A b \in Blocks: read[b]
    /\ UNCHANGED vars

Next ==
    \/ \E b \in Blocks: PrefillWrite(b)
    \/ \E b \in Blocks: Arrive(b)
    \/ \E b \in Blocks: EnterDecode(b)
    \/ \E b \in Blocks: DecodeRead(b)
    \/ Done

(* Every block's actions are weakly fair, so prefill completes, the barrier    *)
(* releases, and decode drains -- the invariants are non-vacuous and the       *)
(* positive instance reaches Progress.                                         *)
Fairness ==
    /\ \A b \in Blocks: WF_vars(PrefillWrite(b))
    /\ \A b \in Blocks: WF_vars(Arrive(b))
    /\ \A b \in Blocks: WF_vars(EnterDecode(b))
    /\ \A b \in Blocks: WF_vars(DecodeRead(b))

Spec == Init /\ [][Next]_vars /\ Fairness

TypeOK ==
    /\ pfDone \in [Blocks -> BOOLEAN]
    /\ atBarrier \in [Blocks -> BOOLEAN]
    /\ inDecode \in [Blocks -> BOOLEAN]
    /\ read \in [Blocks -> BOOLEAN]
    /\ sawUnwritten \in [Blocks -> BOOLEAN]

(* The capstone safety obligation: no decode block ever consumes state a       *)
(* prefill block has not finished writing. The bit-exact "fused == separate"   *)
(* handover gate rests on this -- a premature read makes the fused kernel       *)
(* diverge from the separate prefill-then-decode reference.                    *)
NoPrematureConsume == \A b \in Blocks: ~sawUnwritten[b]

(* The fused kernel completes: every block finishes its decode read. *)
Progress == <>(\A b \in Blocks: read[b])

=============================================================================
