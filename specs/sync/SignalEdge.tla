---------------------------- MODULE SignalEdge ----------------------------
(* The signaling-edge refinement of the sync protocol: ONE producer-to-    *)
(* consumer handoff (Y06's shape) at the memory-operation level, where the *)
(* measured sm_75 facts live (reference/tu102 @ 789ff42, the sysatom and   *)
(* sentinel families, 2026-07-11). MegakernelSync.tla models the pass loop *)
(* over banks; this module models why that loop's edges are fenced and     *)
(* strongly read:                                                          *)
(*   - ld.acquire lowers to a bare strong load on sm_75; ALL ordering      *)
(*     rides the producer's release MEMBAR, and a consumer whose payload   *)
(*     reads are plain re-reads its own stale L1 (measured 999999/1e6).    *)
(*   - sentinel (in-band tail) signaling needs THREE extras at once:       *)
(*     strong polls, ordered stores (observed on this rig, guaranteed      *)
(*     nowhere), and a domain-separated sentinel value. Each negative cfg  *)
(*     drops one and TLC MUST produce the counterexample; a clean pass on  *)
(*     a negative means the model lost the hazard.                         *)
(* Freeze verdict this module records: the fenced flag protocol stays the  *)
(* default (Kog's 9x is absent on this rig; the fenced flag WINS locally   *)
(* at >= 1 KB payloads); sentinel is adoptable only on edges whose failure *)
(* degrades to waste (Y07) and only with the trio intact.                  *)
(*                                                                         *)
(* Cache model: cBody/cTail are the consumer's cached lines from before    *)
(* the pass; Refresh* copies global to cache at any time (eviction and    *)
(* refill are the environment). A plain read returns the cache; a strong   *)
(* (.cg) read returns global. The poll eventually observes (WF on          *)
(* RefreshTail); the one-shot body read at poll-success is where the       *)
(* staleness bites - exactly the measured asymmetry.                       *)

EXTENDS Naturals

CONSTANTS
    Scheme,            \* "flag" (dedicated tail word) or "sentinel" (in-band)
    ProducerFence,     \* membar between the body store and the tail store
    StrongReads,       \* consumer poll + body reads are .cg/strong scope
    StoresOrdered,     \* unfenced stores still arrive body-first (observed)
    SentinelCollision  \* the data domain contains the sentinel value

ASSUME Scheme \in {"flag", "sentinel"}
ASSUME ProducerFence \in BOOLEAN
ASSUME StrongReads \in BOOLEAN
ASSUME StoresOrdered \in BOOLEAN
ASSUME SentinelCollision \in BOOLEAN
ASSUME SentinelCollision => Scheme = "sentinel"

SENT == 0                                    \* the sentinel / reset fill
OLD  == 1                                    \* the previous pass's datum
Fresh  == IF SentinelCollision THEN SENT ELSE 2
TAG  == 2                                    \* the flag scheme's pass seqno

VARIABLES
    gBody, gTail,             \* global (L2) contents
    cBody, cTail,             \* the consumer's cached copies
    bodyStored, tailStored,   \* producer progress
    done, taken               \* consumer outcome

vars == <<gBody, gTail, cBody, cTail, bodyStored, tailStored, done, taken>>

(* flag: the tag word was cleared for this pass, no buffer reset needed.   *)
(* sentinel: the buffer was reset to SENT; the caches still hold the       *)
(* pre-reset lines - the ring-reuse reality.                               *)
Init ==
    /\ bodyStored = FALSE /\ tailStored = FALSE
    /\ done = FALSE /\ taken = SENT
    /\ IF Scheme = "flag"
       THEN gBody = OLD  /\ gTail = SENT /\ cBody = OLD /\ cTail = SENT
       ELSE gBody = SENT /\ gTail = SENT /\ cBody = OLD /\ cTail = OLD

StoreBody ==
    /\ ~bodyStored
    /\ gBody' = Fresh /\ bodyStored' = TRUE
    /\ UNCHANGED <<gTail, cBody, cTail, tailStored, done, taken>>

TailValue == IF Scheme = "flag" THEN TAG ELSE Fresh

(* The tail store: with a fence (or observed ordering) it cannot pass the  *)
(* body store; without both, it may.                                       *)
StoreTail ==
    /\ ~tailStored
    /\ bodyStored \/ (~ProducerFence /\ ~StoresOrdered)
    /\ gTail' = TailValue /\ tailStored' = TRUE
    /\ UNCHANGED <<gBody, cBody, cTail, bodyStored, done, taken>>

RefreshBody ==
    /\ cBody' = gBody
    /\ UNCHANGED <<gBody, gTail, cTail, bodyStored, tailStored, done, taken>>

RefreshTail ==
    /\ cTail' = gTail
    /\ UNCHANGED <<gBody, gTail, cBody, bodyStored, tailStored, done, taken>>

ReadTail == IF StrongReads THEN gTail ELSE cTail
ReadBody == IF StrongReads THEN gBody ELSE cBody
PollHit  == IF Scheme = "flag" THEN ReadTail = TAG ELSE ReadTail # SENT

Consume ==
    /\ ~done /\ PollHit
    /\ taken' = ReadBody /\ done' = TRUE
    /\ UNCHANGED <<gBody, gTail, cBody, cTail, bodyStored, tailStored>>

Next ==
    \/ StoreBody \/ StoreTail
    \/ RefreshBody \/ RefreshTail
    \/ Consume

(* RefreshBody is deliberately UNFAIR: a stale body line may persist       *)
(* forever - the measured hazard. Everything else eventually happens.      *)
Spec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(StoreBody) /\ WF_vars(StoreTail)
    /\ WF_vars(RefreshTail) /\ WF_vars(Consume)

TypeOK ==
    /\ gBody \in 0..2 /\ gTail \in 0..2 /\ cBody \in 0..2 /\ cTail \in 0..2
    /\ bodyStored \in BOOLEAN /\ tailStored \in BOOLEAN
    /\ done \in BOOLEAN /\ taken \in 0..2

(* The consumer, when it consumes, took THIS pass's datum.                 *)
NoStaleConsume == done => taken = Fresh

(* The consumer eventually consumes (fails under sentinel collision:       *)
(* the poll can never distinguish data from reset fill).                   *)
Progress == <>done

=============================================================================
