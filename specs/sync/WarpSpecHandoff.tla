-------------------------- MODULE WarpSpecHandoff --------------------------
(* The warp-specialized MOVER/COMPUTE double-buffer handoff at BLOCK scope,     *)
(* sm_75. The refinement the FATTN f16-overlap (LEAD) and pipe-overlap levers   *)
(* of plan/0143 introduce: the per-tile `__syncthreads` that used to separate   *)
(* the KV load from the HMMA compute is replaced by a warp split -- MOVER warps *)
(* issue `ld_cg` + `st.shared` into a smem tile buffer while COMPUTE warps run  *)
(* HMMA over the previous tile. Two buffers let the mover fill tile t+1 while   *)
(* compute consumes tile t, so a full barrier per tile (which would serialize   *)
(* the two roles and re-add the 16.2 ms load leg the lever exists to hide) is   *)
(* replaced by per-buffer flags. This module checks the two ordering edges the  *)
(* barrier used to give for free.                                               *)
(*                                                                              *)
(*   E1  fill -> consume (visibility): the mover's tile stores must be VISIBLE  *)
(*       before compute reads them. On sm_75 that is a `__threadfence_block`    *)
(*       between the `st.shared` and the ready-flag store; WITHOUT it the flag  *)
(*       store can pass the data stores (store-store reordering) and compute    *)
(*       reads a half-filled tile. Knob: ProducerFence.                         *)
(*   E2  consume -> refill (WAR): the mover may reuse a buffer only after the   *)
(*       compute warps have DRAINED it. On sm_75 that is a consumed-flag the    *)
(*       mover polls; WITHOUT it the mover's `st.shared` overwrites a tile that *)
(*       compute is still reading (write-after-read). Knob: WARGuard.           *)
(*                                                                              *)
(* Sibling to SignalEdge.tla (which models the cross-GPU Y06 NVLink edge and    *)
(* its L1-staleness) and WaitCount.tla. Here the scope is intra-block smem, so  *)
(* the hazard is producer-side store reordering + buffer reuse, not an L1 line  *)
(* going stale; there is no consumer cache. ONE invariant, NoStaleTile, catches *)
(* BOTH edges, and each knob independently closes one violation path -- so the  *)
(* two MUST-FAIL negatives (drop one knob each) pin the two edges separately,   *)
(* mirroring the suite's must-fail discipline.                                  *)
(*                                                                              *)
(* Instances: WarpSpecHandoff_overlap.cfg (both knobs -> safe + live),          *)
(* WarpSpecHandoff_no_fence.cfg (E1 dropped -> NoStaleTile MUST fail),          *)
(* WarpSpecHandoff_no_warguard.cfg (E2 dropped -> NoStaleTile MUST fail).       *)

EXTENDS Naturals

CONSTANTS
    ProducerFence,   \* __threadfence_block between the tile store and the ready-flag (E1)
    WARGuard         \* mover waits for the consumed-flag before refilling a buffer (E2)

ASSUME ProducerFence \in BOOLEAN
ASSUME WARGuard \in BOOLEAN

NRounds == 3                       \* tiles 0,1,2: buffer (r % 2) is reused at r=2 (exercises WAR)
Rounds  == 0 .. NRounds - 1
Buffers == {0, 1}
STALE   == 8                       \* a buffer datum belonging to no current round
NONE    == 9                       \* the ready-flag's unset value
Buf(r)  == r % 2                   \* the double-buffer index for round r

VARIABLES
    gData,        \* [Buffers -> Rounds \cup {STALE}]  datum physically in each smem buffer
    gReady,       \* [Buffers -> Rounds \cup {NONE}]   the ready-flag the mover raises
    bufDrained,   \* [Buffers -> BOOLEAN]  compute has read this buffer since its current fill
    mRound,       \* the round the mover is producing (0 .. NRounds)
    mData,        \* the mover has written gData for mRound (before raising the flag)
    cRound,       \* the round the consumer is taking (0 .. NRounds)
    cPending,     \* the consumer latched a poll-hit for cRound and will read next
    cGot          \* [Rounds -> Rounds \cup {STALE}]  what the consumer took for each round

vars == <<gData, gReady, bufDrained, mRound, mData, cRound, cPending, cGot>>

TypeOK ==
    /\ gData \in [Buffers -> Rounds \cup {STALE}]
    /\ gReady \in [Buffers -> Rounds \cup {NONE}]
    /\ bufDrained \in [Buffers -> BOOLEAN]
    /\ mRound \in 0 .. NRounds
    /\ mData \in BOOLEAN
    /\ cRound \in 0 .. NRounds
    /\ cPending \in BOOLEAN
    /\ cGot \in [Rounds -> Rounds \cup {STALE}]

Init ==
    /\ gData = [b \in Buffers |-> STALE]
    /\ gReady = [b \in Buffers |-> NONE]
    /\ bufDrained = [b \in Buffers |-> TRUE]   \* empty buffers are safe to overwrite
    /\ mRound = 0
    /\ mData = FALSE
    /\ cRound = 0
    /\ cPending = FALSE
    /\ cGot = [r \in Rounds |-> STALE]

(* The mover writes the tile into its buffer. With the WAR guard it may not     *)
(* overwrite a buffer the consumer has not drained since the last fill.         *)
MoverStoreData ==
    /\ mRound < NRounds
    /\ ~mData
    /\ (~WARGuard \/ bufDrained[Buf(mRound)])
    /\ gData' = [gData EXCEPT ![Buf(mRound)] = mRound]
    /\ bufDrained' = [bufDrained EXCEPT ![Buf(mRound)] = FALSE]
    /\ mData' = TRUE
    /\ UNCHANGED <<gReady, mRound, cRound, cPending, cGot>>

(* The mover raises the ready-flag. With the fence the tile store must be done  *)
(* first (mData); without it the flag store may pass the data store, so the     *)
(* flag can be raised while gData still holds the previous datum -- the E1      *)
(* store-store reorder.                                                         *)
MoverRaiseReady ==
    /\ mRound < NRounds
    /\ (mData \/ ~ProducerFence)
    /\ gReady' = [gReady EXCEPT ![Buf(mRound)] = mRound]
    /\ mRound' = mRound + 1
    /\ mData' = FALSE
    /\ UNCHANGED <<gData, bufDrained, cRound, cPending, cGot>>

(* The consumer polls its buffer's ready-flag for the current round and, on a   *)
(* hit, latches -- modeling the poll and the tile read as separate steps so a   *)
(* racing mover overwrite (E2) can interleave between them.                     *)
ConsumerPoll ==
    /\ cRound < NRounds
    /\ ~cPending
    /\ gReady[Buf(cRound)] = cRound
    /\ cPending' = TRUE
    /\ UNCHANGED <<gData, gReady, bufDrained, mRound, mData, cRound, cGot>>

(* The consumer reads the tile it latched, records what it actually got, and    *)
(* marks the buffer drained so the mover may reuse it.                          *)
ConsumerRead ==
    /\ cPending
    /\ cGot' = [cGot EXCEPT ![cRound] = gData[Buf(cRound)]]
    /\ bufDrained' = [bufDrained EXCEPT ![Buf(cRound)] = TRUE]
    /\ cRound' = cRound + 1
    /\ cPending' = FALSE
    /\ UNCHANGED <<gData, gReady, mRound, mData>>

MoverNext == MoverStoreData \/ MoverRaiseReady
ConsumerNext == ConsumerPoll \/ ConsumerRead

(* The pipeline has drained -- both roles finished. An explicit stuttering step *)
(* at the terminal state so it is not reported as a spurious deadlock (the only *)
(* reachable terminal; a mid-pipeline stall would still surface as a deadlock). *)
Done == /\ mRound = NRounds
        /\ cRound = NRounds
        /\ ~cPending
        /\ UNCHANGED vars

Next == MoverNext \/ ConsumerNext \/ Done

(* Both roles are weakly fair -- the mover keeps producing, the consumer keeps  *)
(* draining -- so the pipeline drains and the invariants are non-vacuous.       *)
Spec == Init /\ [][Next]_vars /\ WF_vars(MoverNext) /\ WF_vars(ConsumerNext)

(* ------------------------------ Invariants ------------------------------ *)

(* The load-bearing safety obligation: whenever the consumer has finished a    *)
(* round, it took THAT round's datum -- never a half-filled tile (E1) and      *)
(* never a tile the mover overwrote out from under it (E2).                    *)
NoStaleTile ==
    \A r \in Rounds: (cRound > r) => (cGot[r] = r)

(* ------------------------------ Liveness ------------------------------- *)

(* Every round is eventually consumed: the pipeline drains to completion.      *)
Progress == <>(cRound = NRounds)

===========================================================================
