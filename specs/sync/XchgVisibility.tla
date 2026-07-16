------------------------- MODULE XchgVisibility -------------------------
(* The cross-GPU XCHG push-in-epilogue ordering (plan/0143 XCHG-overlap       *)
(* lever, gate-manifest row "XCHG overlap", predicted -0.7 ms). Models the    *)
(* op_xchg_push -> OP_BOUNDARY -> op_xchg_reduce protocol of k0/ops/xchg.cuh  *)
(* at system scope, the sm_75 sync_protocol facts reference/tu102 measured.   *)
(*                                                                            *)
(* The lever fuses the PUSH into the PRIOR op's epilogue: the mover warps      *)
(* issue the peer float4 payload stores (xchg.cuh op_xchg_push, the measured   *)
(* 2.4-2.5x line-filling visibility store) while the previous op's compute      *)
(* still runs, so the push is moved EARLIER. For that reorder to stay correct  *)
(* two edges must hold, and this module pins each as a must-fail negative:      *)
(*                                                                            *)
(*   E1  push visibility: the peer payload stores must be membar.sys-fenced    *)
(*       (xchg.cuh:88 membar_sys(), then the OP_BOUNDARY that globally         *)
(*       completes every block's push) BEFORE the elected release.sys of the   *)
(*       seqno (xchg.cuh:105 st_release_sys). WITHOUT the sys fence the seqno  *)
(*       release can pass the payload stores: the peer polls the seqno         *)
(*       (ld.acquire.sys, xchg.cuh:110), sees it raised, and folds a payload   *)
(*       still in flight. Knob: PushFenced.                                    *)
(*   E2  strong consume: the peer must read the received payload STRONG        *)
(*       (ld_cg / .cg, xchg.cuh:119 "strong, never plain"). A plain LDG        *)
(*       re-reads the consumer's own stale L1 line (the measured xgpu          *)
(*       999999/1e6 staleness), defeating a correctly ordered producer.        *)
(*       Knob: StrongConsume.                                                  *)
(*                                                                            *)
(* Sibling to SignalEdge.tla (the Y06 NVLink flag, whose axis is the flag-vs-  *)
(* sentinel signaling scheme). This module is the XCHG-specific edge pair the  *)
(* epilogue-overlap must preserve, bound line-for-line to xchg.cuh; the        *)
(* seqno poll itself is acquire.sys (given), so the hazards modelled are the   *)
(* push ordering (E1) and the payload L1 staleness (E2), one negative each.    *)
(* One pushing block is modelled; the OP_BOUNDARY generalizes E1 to all        *)
(* blocks' pushes completing before any seqno (that fan-in is WaitCount.tla).  *)

EXTENDS Naturals

CONSTANTS
    PushFenced,    \* membar.sys after the peer payload stores, before the seqno release (E1)
    StrongConsume  \* the peer reads the received payload with .cg / ld_cg, not a plain LDG (E2)

ASSUME PushFenced \in BOOLEAN
ASSUME StrongConsume \in BOOLEAN

STALE == 0                       \* the peer inbox's previous-pass payload
FRESH == 1                       \* this pass's pushed slice
NOSEQ == 0                       \* the seqno word before this pass
SEQ   == 1                       \* this pass's published seqno

VARIABLES
    gPayload,     \* the peer inbox payload (producer truth, peer VRAM): STALE or FRESH
    gSeqno,       \* the peer inbox seqno word: NOSEQ or SEQ
    cPayload,     \* the consumer's cached (L1) copy of the payload line
    pushStored,   \* the mover finished the peer float4 payload stores
    done,         \* the peer folded (op_xchg_reduce)
    taken         \* the payload the peer actually folded

vars == <<gPayload, gSeqno, cPayload, pushStored, done, taken>>

Init ==
    /\ gPayload = STALE
    /\ gSeqno = NOSEQ
    /\ cPayload = STALE           \* the peer's L1 holds the previous pass's line
    /\ pushStored = FALSE
    /\ done = FALSE
    /\ taken = STALE

(* op_xchg_push: the mover float4-stores this GPU's slice into the peer inbox. *)
PushStore ==
    /\ ~pushStored
    /\ gPayload' = FRESH
    /\ pushStored' = TRUE
    /\ UNCHANGED <<gSeqno, cPayload, done, taken>>

(* op_xchg_reduce elected release: st_release_sys the seqno into the peer      *)
(* inbox. With the sys fence (+ boundary) the payload stores must be complete  *)
(* first; without it the seqno release may pass the payload stores.            *)
SeqnoRelease ==
    /\ gSeqno = NOSEQ
    /\ (pushStored \/ ~PushFenced)
    /\ gSeqno' = SEQ
    /\ UNCHANGED <<gPayload, cPayload, pushStored, done, taken>>

(* The peer's L1 refills the payload line from VRAM at some time. Deliberately *)
(* UNFAIR: a stale L1 line may persist forever -- the measured xgpu hazard.    *)
RefreshPayload ==
    /\ cPayload' = gPayload
    /\ UNCHANGED <<gPayload, gSeqno, pushStored, done, taken>>

ReadPayload == IF StrongConsume THEN gPayload ELSE cPayload

(* op_xchg_reduce consume: poll the local seqno (ld.acquire.sys) and, on a     *)
(* hit, read the received payload and fold. A .cg read fetches VRAM; a plain   *)
(* LDG takes the possibly-stale L1 copy.                                       *)
Consume ==
    /\ ~done
    /\ gSeqno = SEQ
    /\ taken' = ReadPayload
    /\ done' = TRUE
    /\ UNCHANGED <<gPayload, gSeqno, cPayload, pushStored>>

Next ==
    \/ PushStore \/ SeqnoRelease
    \/ RefreshPayload
    \/ Consume

(* RefreshPayload is UNFAIR (a stale L1 line may persist); the push, the seqno *)
(* release and the one-shot consume are fair so the exchange drains.          *)
Spec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(PushStore) /\ WF_vars(SeqnoRelease) /\ WF_vars(Consume)

TypeOK ==
    /\ gPayload \in {STALE, FRESH} /\ cPayload \in {STALE, FRESH}
    /\ gSeqno \in {NOSEQ, SEQ}
    /\ pushStored \in BOOLEAN /\ done \in BOOLEAN
    /\ taken \in {STALE, FRESH}

(* The load-bearing safety obligation: when the peer folds, it read THIS       *)
(* pass's pushed slice -- never a payload still in flight (E1) and never a      *)
(* stale L1 line (E2). Folding a stale slice corrupts the mirrored allreduce.  *)
NoStaleConsume == done => taken = FRESH

(* The peer eventually folds: the exchange completes. *)
Progress == <>done

=============================================================================
