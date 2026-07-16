-------------------------- MODULE PipeOverlap --------------------------
(* The pipe-overlap warp-spec handoff, CONSUMER-side ordering (plan/0143      *)
(* pipe-overlap lever, gate-manifest row "Pipe-overlap", predicted -2.8 ms).  *)
(*                                                                            *)
(* Pipe-overlap runs MOVER warps that prefetch the NEXT op's operands into a  *)
(* smem staging slot while COMPUTE warps finish the CURRENT op, then the      *)
(* compute warps consume the staged operand. WarpSpecHandoff.tla pins the     *)
(* PRODUCER-side edges of that handoff (E1 store->flag visibility via         *)
(* ProducerFence, E2 buffer-reuse WAR via WARGuard). This sibling pins the    *)
(* edge WarpSpecHandoff leaves to the barrier: the CONSUMER's payload read    *)
(* must be ordered AFTER it observes the ready-flag.                          *)
(*                                                                            *)
(* On sm_75 the flag poll and the payload read are two independent smem       *)
(* loads. Absent an acquire (a read-read fence between the flag load and the  *)
(* payload load, i.e. __threadfence_block or a .relaxed->.acquire on the      *)
(* flag), ptxas is free to hoist the payload load ABOVE the spin on the flag  *)
(* -- the classic "spin on a plain flag, use a plain payload" bug -- so the   *)
(* consumer latches the slot's PRE-store (stale) contents even though it      *)
(* later sees the flag raised. This is the hazard the barrier hid for free    *)
(* and the warp-spec pipe must re-close with a consumer acquire.              *)
(*                                                                            *)
(* Cache model mirrors SignalEdge.tla: cPayload is the consumer's latched     *)
(* register copy of the smem slot; LatchPayload (deliberately UNFAIR) is the  *)
(* hoisted/early load that may never re-read. A plain read returns the latch; *)
(* an acquire read returns the live slot (ordered after the poll). Distinct   *)
(* from SignalEdge (cross-GPU L1 line, .cg strong load) in scope and fix:     *)
(* here the slot is intra-block smem and the fix is a consumer read-read      *)
(* fence, not a strong global load.                                           *)
(*                                                                            *)
(* Instances: PipeOverlap_acquire.cfg (both edges -> safe + live),            *)
(* PipeOverlap_plain_reads.cfg (ConsumerAcquire dropped -> NoStaleConsume     *)
(* MUST fail; the brief's named negative), PipeOverlap_no_release.cfg         *)
(* (ProducerRelease dropped -> NoStaleConsume MUST fail, the producer edge).  *)

EXTENDS Naturals

CONSTANTS
    ProducerRelease, \* __threadfence_block between the payload store and the ready-flag store
    ConsumerAcquire  \* read-read fence: the payload read is ordered AFTER the flag poll

ASSUME ProducerRelease \in BOOLEAN
ASSUME ConsumerAcquire \in BOOLEAN

STALE == 0                       \* the slot's previous-stage contents
FRESH == 1                       \* this stage's operand
DOWN  == 0                       \* ready-flag unset
UP    == 1                       \* ready-flag raised

VARIABLES
    gPayload,      \* the smem staging slot (producer truth): STALE or FRESH
    gFlag,         \* the ready-flag: DOWN or UP
    cPayload,      \* the consumer's latched register copy of the slot
    payloadStored, \* the mover has written the operand into the slot
    done,          \* the compute warp has consumed
    taken          \* what the consumer actually took

vars == <<gPayload, gFlag, cPayload, payloadStored, done, taken>>

Init ==
    /\ gPayload = STALE
    /\ gFlag = DOWN
    /\ cPayload = STALE            \* the pre-store latch (a hoisted early load)
    /\ payloadStored = FALSE
    /\ done = FALSE
    /\ taken = STALE

(* The mover stores the next operand into the staging slot. *)
MoverStore ==
    /\ ~payloadStored
    /\ gPayload' = FRESH
    /\ payloadStored' = TRUE
    /\ UNCHANGED <<gFlag, cPayload, done, taken>>

(* The mover raises the ready-flag. With the release fence the operand store  *)
(* must be visible first; without it the flag store may pass the operand      *)
(* store (producer store-store reorder).                                      *)
MoverRaiseFlag ==
    /\ gFlag = DOWN
    /\ (payloadStored \/ ~ProducerRelease)
    /\ gFlag' = UP
    /\ UNCHANGED <<gPayload, cPayload, payloadStored, done, taken>>

(* The consumer's payload load completes at some (possibly early) time,       *)
(* latching whatever the slot holds then. Deliberately UNFAIR: a hoisted      *)
(* stale latch may persist and never re-read -- the measured reorder hazard.  *)
LatchPayload ==
    /\ cPayload' = gPayload
    /\ UNCHANGED <<gPayload, gFlag, payloadStored, done, taken>>

ReadPayload == IF ConsumerAcquire THEN gPayload ELSE cPayload

(* The compute warp polls the flag and, on a hit, consumes the operand.       *)
(* An acquire read fetches the live slot (ordered after this poll); a plain   *)
(* read takes the possibly-stale latch.                                       *)
Consume ==
    /\ ~done
    /\ gFlag = UP
    /\ taken' = ReadPayload
    /\ done' = TRUE
    /\ UNCHANGED <<gPayload, gFlag, cPayload, payloadStored>>

Next ==
    \/ MoverStore \/ MoverRaiseFlag
    \/ LatchPayload
    \/ Consume

(* LatchPayload is UNFAIR (a stale hoisted load may persist); the mover and   *)
(* the one-shot consume are fair so the pipeline drains and Progress is       *)
(* non-vacuous.                                                               *)
Spec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(MoverStore) /\ WF_vars(MoverRaiseFlag) /\ WF_vars(Consume)

TypeOK ==
    /\ gPayload \in {STALE, FRESH} /\ cPayload \in {STALE, FRESH}
    /\ gFlag \in {DOWN, UP}
    /\ payloadStored \in BOOLEAN /\ done \in BOOLEAN
    /\ taken \in {STALE, FRESH}

(* The load-bearing safety obligation: when the compute warp consumes, it     *)
(* took THIS stage's operand, never the previous stage's stale contents.      *)
NoStaleConsume == done => taken = FRESH

(* The compute warp eventually consumes: the pipe drains. *)
Progress == <>done

=============================================================================
