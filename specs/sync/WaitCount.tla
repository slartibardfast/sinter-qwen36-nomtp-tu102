------------------------------- MODULE WaitCount -------------------------------
(* Decrement-to-zero fan-in (the ETC idiom: the last notifier observes     *)
(* zero and acts) and the init-before-first-notify obligation the ETC      *)
(* paper never acknowledges: a runtime-written wait count must be ordered  *)
(* before every notify. A recycled count slot holds its prior value; a     *)
(* notify that races the init decrements that stale value, and the pass    *)
(* either proceeds early (stale 1: consumer runs before the producers      *)
(* finish - corruption) or loses a decrement (stale 0: the count never     *)
(* reaches zero - deadlock). Bound onto any runtime-written count that     *)
(* ever enters our protocol (none does today; the obligation is recorded   *)
(* ahead of need, from the Event Tensor read of 2026-07-11).               *)

EXTENDS Naturals, FiniteSets

CONSTANTS
    Producers,     \* the notifying producers
    InitOrdered    \* the init write happens-before every notify

ASSUME Producers # {}
ASSUME InitOrdered \in BOOLEAN

VARIABLES
    stale,       \* the recycled slot's prior value (0 or 1, chosen at Init)
    count,       \* the count word
    initDone,    \* the init write has landed
    notified,    \* producers that have notified
    proceeded    \* a notifier observed zero and released the consumer

vars == <<stale, count, initDone, notified, proceeded>>

Init ==
    /\ stale \in {0, 1}
    /\ count = stale
    /\ initDone = FALSE /\ notified = {} /\ proceeded = FALSE

WriteInit ==
    /\ ~initDone
    /\ count' = Cardinality(Producers) /\ initDone' = TRUE
    /\ UNCHANGED <<stale, notified, proceeded>>

(* The decrement returns the post-value; the notifier that takes the count *)
(* to zero releases the consumer (count = 1 in the pre-state).             *)
Notify(p) ==
    /\ p \notin notified
    /\ InitOrdered => initDone
    /\ notified' = notified \cup {p}
    /\ count' = IF count > 0 THEN count - 1 ELSE 0
    /\ proceeded' = (proceeded \/ count = 1)
    /\ UNCHANGED <<stale, initDone>>

NotifyAny == \E p \in Producers : Notify(p)

(* The fan-in has drained: init landed and every producer notified. An        *)
(* explicit stuttering step at that terminal state so protocol completion is   *)
(* not reported as a spurious deadlock (keeps deadlock-checking meaningful for  *)
(* a genuine stall, e.g. the lost-decrement negative where this never holds).  *)
Done == /\ initDone
        /\ notified = Producers
        /\ UNCHANGED vars

Next == WriteInit \/ NotifyAny \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(WriteInit) /\ WF_vars(NotifyAny)

TypeOK ==
    /\ stale \in {0, 1}
    /\ count \in 0..Cardinality(Producers)
    /\ initDone \in BOOLEAN /\ proceeded \in BOOLEAN
    /\ notified \subseteq Producers

(* The consumer runs only after every producer finished.                   *)
ProceedSound == proceeded => notified = Producers

(* The pass completes (fails on the lost-decrement trace).                 *)
Progress == <>proceeded

=============================================================================
