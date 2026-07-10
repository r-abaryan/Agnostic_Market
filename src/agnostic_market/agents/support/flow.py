"""The support gated flow (Tier 3, AGENTS §A4/§A9) — refund-to-new-instrument + the
step-up verification loop (T3), with the A10a replay invariants in the node structure.

The T3 shape (proven against real langgraph before this code — see the plan's task-zero
spike). Refund to a NEW instrument requires L2 regardless of amount (§A4b); an L1 caller is
raised mid-flow by a committed OTP, THEN the refund resumes behind a readback interrupt:

    assemble ─▶ guardrail ─(L2 already)──────────▶ confirm[INT#2] ─▶ place
                   │                                    ▲
                   └─(L1, new instr.)─▶ risk_check ─▶ dispatch ─▶ collect[INT#1: verify+raise]

Why the decomposition (A10a): LangGraph re-runs an interrupted node from the TOP on resume,
and does NOT commit a node's update until it RETURNS. So the two interrupts sit in SEPARATE
nodes; `collect` verifies the OTP AFTER its own (only) interrupt and raises the store level
on its return (committed before the readback interrupt); and every side effect is either
post-interrupt (`place`) or idempotent-pre-interrupt (`dispatch` keyed by attempt — else a
replay re-sends the OTP, S3).

Authority discipline: `verification_level` lives ONLY in the VerificationStore (read live at
the guardrail, written only by `collect` via a committed OTP) — never a checkpointed channel,
so a replayed checkpoint can't re-grant a level (§A388). The refund effect is idempotent by
a per-INTENT key (the OrderStore is the arbiter). The readback is GRAPH-authored (one-author
rule), covers the fields ISSUE_REFUND_POLICY declares, and consent over a barged-over
readback is not consent (§4a) — the confirm node re-confirms once.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel

from agnostic_market.agents._consent import classify_consent
from agnostic_market.agents._toolcalls import ack_extra_tool_calls
from agnostic_market.agents.support.prompt import compose_support_prompt
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.orders import CANCELLED_STATUS, CancelError, OrderStore, RefundError
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.confirmation import (
    ISSUE_REFUND_POLICY,
    RefundDestination,
    ToolConfirmationPolicy,
    refund_required_level,
)
from agnostic_market.dtos.state import (
    HandoffRequest,
    PendingCancel,
    PendingRefund,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.support")

# Max OTP collect attempts before the flow gives up and offers a human (§A9). Bounded +
# deterministic (A10a rule 3): the collect node counts committed misses, not a loop.
_MAX_OTP_ATTEMPTS = 2

# A masked instrument reference is all voice ever handles — never a raw PAN (PCI, SECURITY
# §6). The fixture "new card on file" for the build phase; a real stored-instrument ref later.
_NEW_INSTRUMENT_REF = "card ending 4471"


def _last_user_text(state: ReasoningState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _readback_line(pending: PendingRefund, policy: ToolConfirmationPolicy) -> str:
    """The GRAPH-authored refund readback — the `interrupt()` payload.

    Renders every field ISSUE_REFUND_POLICY declares; a declared field with no rendered
    value fails loudly (readback can't be silently forgotten, §7a). Speech-native: no card
    number is ever spoken beyond the masked tail.
    """
    rendered: dict[str, str] = {
        "total_amount": f"${pending.amount_usd:.2f}",
        "new_payment_instrument_ref": pending.instrument_ref,
    }
    missing = policy.confirm_fields - rendered.keys()
    if missing:
        raise ValueError(f"readback cannot render declared confirm_fields: {sorted(missing)}")
    return (
        f"Just to confirm: a {rendered['total_amount']} refund to your "
        f"{pending.instrument_ref}. Shall I go ahead?"
    )


def _cancel_readback_line(pending: PendingCancel) -> str:
    """The GRAPH-authored cancel readback — the `interrupt()` payload. Names the order so a
    mis-heard reference can't void the wrong one; states irreversibility (§7a discipline)."""
    return (
        f"Just to confirm - cancel your order for {pending.summary} ({pending.order_id})? "
        "This can't be undone."
    )


class _ProposeRefund(BaseModel):
    order_key: str
    amount_usd: float
    destination: RefundDestination


class _ProposeCancel(BaseModel):
    order_key: str


@dataclass(frozen=True)
class SupportNodes:
    """The support flow's node callables + its caller-facing (speakable) node names.

    Wiring is the graph builder's job (frontline/graph.py); this module owns only behavior.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    risk_check: Callable[[ReasoningState], dict[str, object]]
    dispatch: Callable[[ReasoningState], dict[str, object]]
    collect: Callable[[ReasoningState], dict[str, object]]
    confirm: Callable[[ReasoningState], dict[str, object]]
    place: Callable[[ReasoningState], dict[str, object]]
    cancel_guardrail: Callable[[ReasoningState], dict[str, object]]
    cancel_confirm: Callable[[ReasoningState], dict[str, object]]
    cancel_void: Callable[[ReasoningState], dict[str, object]]
    abort: Callable[[ReasoningState], dict[str, object]]
    escape_human: Callable[[ReasoningState], dict[str, object]]
    # After assemble: which action did the model propose? "refund" | "cancel" | "leave" | "clarify".
    route_after_assemble: Callable[[ReasoningState], str]
    # Branch after the refund guardrail depends on the LIVE verification level (in the store,
    # which graph.py can't see) — so the flow owns this router, closed over the store.
    # "confirm" (level sufficient) | "stepup" (needs the OTP loop) | "handover" (cleared).
    route_after_guardrail: Callable[[ReasoningState], str]
    # Branch after collect: the OTP verify may have raised to L2 (-> confirm), exhausted
    # attempts / flagged risk (-> handover), or asked for a re-collect (-> dispatch again).
    route_after_collect: Callable[[ReasoningState], str]
    # Branch after the cancel guardrail depends on the LIVE order status + risk (store-owned).
    # "confirm" (processing, no risk) | "handover" (shipped/delivered or risk-flagged).
    route_after_cancel_guardrail: Callable[[ReasoningState], str]
    speakable_nodes: frozenset[str]


def build_support_nodes(
    reasoning_model: BaseChatModel,
    order_store: OrderStore,
    verification_store: VerificationStore,
    otp: OtpProvider,
    risk: RiskProvider,
    policy: PolicyContext,
    *,
    display_name: str,
) -> SupportNodes:
    """Build the support flow's nodes, closed over the session's stores + providers + policy
    (§A5: tenant/policy/verification bound in code at build time — never carried in state)."""

    @tool
    def propose_refund(order_key: str, amount_usd: float, destination: str) -> str:
        """Propose a refund: which order (option number), how much, and where it goes."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def propose_cancel(order_key: str) -> str:
        """Propose cancelling an order the caller no longer wants (option number)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_support() -> str:
        """Leave the support flow (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    model = reasoning_model.bind_tools([propose_refund, propose_cancel, leave_support])

    def _leave(new_messages: list, call_id: str) -> dict[str, object]:
        new_messages.append(ToolMessage("left support", tool_call_id=call_id))
        write_event({"event": "support_left", "reason": "left_flow"})
        return {
            "messages": new_messages,
            "active_flow": "left_support",
            "pending_refund": None,
            "pending_cancel": None,
        }

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE support: propose a REFUND (order+amount+destination) or a CANCEL
        (order), or clarify, or leave. Mints PendingRefund/PendingCancel (per-intent key) on a
        valid proposal — A10a rule 2: the idempotency key exists in state before any effect."""
        orders = order_store.actionable_orders()
        by_key = {o.key: o for o in orders}
        prompt = SystemMessage(compose_support_prompt(display_name, orders))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal gets ONE corrective re-prompt
            response = model.invoke(messages)
            new_messages.append(response)
            if not response.tool_calls:
                return {"messages": new_messages}  # clarifying question (streamed) — stay
            ack_extra_tool_calls(response, new_messages)
            call = response.tool_calls[0]
            if call["name"] == "leave_support":
                return _leave(new_messages, call["id"])

            if call["name"] == "propose_cancel":
                try:
                    cancel = _ProposeCancel.model_validate(call["args"])
                except ValueError:
                    cancel = None
                chosen = by_key.get(cancel.order_key) if cancel else None
                if chosen is None:
                    feedback = f"Invalid order. Valid order numbers: {', '.join(sorted(by_key))}."
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                new_messages.append(
                    ToolMessage(f"proposed cancel on order {chosen.key}", tool_call_id=call["id"])
                )
                pending_cancel = PendingCancel(
                    order_id=chosen.order_id,
                    summary=chosen.summary,
                    idempotency_key=uuid.uuid4().hex,
                    created_at=time.time(),
                )
                return {"messages": new_messages, "pending_cancel": pending_cancel}

            # else: propose_refund
            try:
                proposal = _ProposeRefund.model_validate(call["args"])
            except ValueError:
                proposal = None
            chosen = by_key.get(proposal.order_key) if proposal else None
            if chosen is None or proposal.amount_usd <= 0:
                feedback = (
                    f"Invalid proposal. Valid order numbers: {', '.join(sorted(by_key))}; "
                    "amount must be > 0."
                )
                new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                messages = [prompt, *state.messages, *new_messages]
                continue
            new_messages.append(
                ToolMessage(
                    f"proposed refund on order {chosen.key} ${proposal.amount_usd:.2f} "
                    f"to {proposal.destination}",
                    tool_call_id=call["id"],
                )
            )
            pending = PendingRefund(
                order_id=chosen.order_id,
                amount_usd=round(proposal.amount_usd, 2),
                destination=proposal.destination,
                instrument_ref=(
                    _NEW_INSTRUMENT_REF
                    if proposal.destination != "original"
                    else "original payment method"
                ),
                idempotency_key=uuid.uuid4().hex,
                attempt_key=uuid.uuid4().hex,
                created_at=time.time(),
            )
            return {"messages": new_messages, "pending_refund": pending}
        logger.warning("support assemble: two invalid proposals; leaving flow")
        write_event({"event": "support_left", "reason": "invalid_proposals"})
        return {
            "messages": new_messages,
            "active_flow": "left_support",
            "pending_refund": None,
            "pending_cancel": None,
        }

    def guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced refund policy (never the model), evaluated in tiers:

        1. CANCELLED order: the void already reversed the charge — an honest "nothing to
           refund" decline (the store would refuse anyway; this owns the caller-facing line).
        2. REMEDY steer: full money-back to the ORIGINAL instrument on a still-cancellable
           order IS a cancellation (industry: unshipped = cancel, not refund) — convert to a
           PendingCancel and route to the cancel path. Deliberately BEFORE the amount gate:
           a void is self-serve at any amount, so it must not be sent to a human by a
           threshold that exists for post-fulfillment refund abuse (the 2026-07-10 live
           incoherence: "$258 needs a person" ... then cancel voided the same $258).
        3. AMOUNT gate (merchant policy, within platform bounds): a refund above
           `refund_require_human_above_usd` goes to a PERSON regardless of level. It authors
           its own decline (clear-before-speak) and ENDs — there is NO handover routing yet
           (no warm-transfer target exists until Phase 4); the human intent is recorded in
           telemetry so the transfer can be wired there without re-finding this spot.
        4. LEVEL gate (§A4b fraud floor): destination -> required level vs the LIVE store
           level, decided by `route_after_guardrail`; no side effect here."""
        pending = state.pending_refund
        assert pending is not None  # only reached with a proposal minted
        if order_store.order_status(pending.order_id) == CANCELLED_STATUS:
            write_event({"event": "refund_denied", "reason": "order_cancelled"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That order was already cancelled, so the charge is reversed - "
                        "there's nothing left to refund on it."
                    )
                ],
            }
        captured = order_store.captured_total(pending.order_id)
        if (
            pending.destination == "original"
            and captured is not None
            and round(pending.amount_usd, 2) >= round(captured, 2)
            and order_store.refunded_so_far(pending.order_id) == 0
            and order_store.is_cancellable(pending.order_id)
        ):
            write_event({"event": "refund_steered_to_cancel", "order_id": pending.order_id})
            return {
                "pending_refund": None,
                "pending_cancel": PendingCancel(
                    order_id=pending.order_id,
                    summary=order_store.order_item_summary(pending.order_id),
                    idempotency_key=uuid.uuid4().hex,
                    created_at=time.time(),
                ),
                "messages": [
                    AIMessage(
                        "That order hasn't shipped yet, so I can cancel it instead - the "
                        f"full ${captured:.2f} goes back to your original payment method."
                    )
                ],
            }
        if pending.amount_usd > policy.refund_require_human_above_usd:
            # Over the merchant's human-review line: speak the specific reason and END (no
            # generic deferral over it), same one-voice pattern as the shipped-cancel decline.
            write_event(
                {
                    "event": "refund_needs_human",
                    "reason": "over_amount_threshold",
                    "handover_intent": "human",
                }
            )
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"A ${pending.amount_usd:.2f} refund is above what I can process on "
                        "this call - our support team can handle a refund that size for you."
                    )
                ],
            }
        required = refund_required_level(pending.amount_usd, pending.destination)
        if verification_store.current_level() < required:
            write_event({"event": "refund_stepup_required", "required_level": required})
        return {}

    def risk_check_node(state: ReasoningState) -> dict[str, object]:
        """SIM-swap / port-out check on the number-on-file (§A4a). Flagged -> do NOT trust an
        OTP; escalate to a person. ANI is never the authenticator."""
        if risk.check_sim_swap():
            write_event({"event": "refund_stepup_failed", "reason": "sim_swap_risk"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        return {}

    def dispatch_node(state: ReasoningState) -> dict[str, object]:
        """Dispatch the OTP to the number-on-file — IDEMPOTENT per step-up attempt (S3: a
        pre-interrupt effect re-runs on replay; the attempt key makes the re-send a no-op)."""
        pending = state.pending_refund
        assert pending is not None
        otp.dispatch(pending.attempt_key)
        return {}

    def collect_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt #1: collect the committed OTP, then verify (CODE seam).

        The verify sits AFTER this node's only interrupt — a replay re-runs the node from the
        top, re-hits the interrupt, and re-verifies the freshly-collected code (harmless,
        idempotent: verify_otp just re-checks). It does NOT precede a LATER interrupt, so it
        is not the S3 hazard (that was a dispatch above an interrupt). On match: raise the
        store to L2 and RETURN (commits before the readback interrupt in the next node). On
        miss: re-collect ONCE (new attempt key -> legit re-dispatch), then human (§A9)."""
        pending = state.pending_refund
        assert pending is not None
        answer = interrupt("For security, please read me the 6-digit code we just sent you.")
        if verification_store.verify_otp(str(answer.get("text", ""))):
            write_event({"event": "refund_stepup_ok", "raised_to": 2})
            return {}  # level now L2 in the store; router -> confirm
        tries = pending.otp_tries + 1
        if tries >= _MAX_OTP_ATTEMPTS:
            write_event({"event": "refund_stepup_failed", "reason": "otp_exhausted"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        # Re-collect: bump tries + a NEW attempt key so the re-dispatch is a legitimate send.
        return {
            "pending_refund": pending.model_copy(
                update={"otp_tries": tries, "attempt_key": uuid.uuid4().hex}
            )
        }

    def confirm_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt #2: the refund readback + deterministic consent. NO side effects;
        §4a barge -> re-confirm once (a truncated 'yes' is not consent). Clock-A TTL checked
        FIRST (clear-before-speak) so a resume arriving after expiry cancels the stale refund
        without consuming the answer (§A10 rule 6; same discipline as checkout)."""
        pending = state.pending_refund
        assert pending is not None
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "refund_expired", "reason": "pending_ttl"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That refund confirmation sat for a while, so I haven't processed "
                        "anything. If you'd still like it, just tell me again."
                    )
                ],
            }
        answer = interrupt(_readback_line(pending, ISSUE_REFUND_POLICY))
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(
                f"Sorry - just to be clear: a ${pending.amount_usd:.2f} refund to your "
                f"{pending.instrument_ref}. Yes or no?"
            )
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "refund_cancelled", "reason": "human_requested"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="refund", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "refund_cancelled", "reason": "declined"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [AIMessage("Okay, I won't refund anything - nothing has changed.")],
            }
        return {}  # yes: pending survives; router -> place

    def place_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Idempotent by the store's
        per-intent key dedup + cumulative-cap enforcement; re-validates the LIVE level (§A4c
        server-side re-validation - the money must not move if the level lapsed)."""
        pending = state.pending_refund
        assert pending is not None
        required = refund_required_level(pending.amount_usd, pending.destination)
        if verification_store.current_level() < required:
            # Belt-and-suspenders: the level was lost between apply and place (revoked). Do
            # NOT move money; drop back to a human. (Not reachable in the happy path.)
            write_event({"event": "refund_stepup_failed", "reason": "level_lapsed_at_place"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        try:
            record = order_store.issue_refund(
                pending.idempotency_key,
                order_id=pending.order_id,
                amount_usd=pending.amount_usd,
                destination=pending.destination,
            )
        except RefundError as exc:
            logger.warning("refund refused by store: %s", exc)
            write_event({"event": "refund_denied", "reason": "store_refused"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [
                    # Outcome only — the handover node's deferral speaks the transfer
                    # sentence; adding one here would say it twice.
                    AIMessage(
                        "I wasn't able to process that refund against the order - "
                        "nothing has changed."
                    )
                ],
                "handover": HandoffRequest(
                    destination="human", reason_code="refund", source="gate"
                ),
            }
        write_event(
            {
                "event": "refund_confirmed",
                "return_id": record.return_id,
                "amount": record.amount_usd,
                "verification": [g.get("method") for g in verification_store.grants],
            }
        )
        return {
            "pending_refund": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - your ${record.amount_usd:.2f} refund is on its way to your "
                    f"{pending.instrument_ref}. Your reference is {record.return_id}."
                )
            ],
        }

    # --- cancel-order sub-path (single interrupt: guardrail -> confirm -> void) -----------

    def cancel_guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced eligibility (never the model): read the LIVE order status + risk, and
        on an INELIGIBLE order author the honest decline. On an eligible one it takes no side
        effect (router -> confirm). Cancel is L1 (no step-up) — voiding an unshipped order
        isn't money-movement to a new destination (§A4b's fraud vector doesn't apply; §A4a
        doesn't tier cancel to L2), so no OTP loop. Decline reasons, each with its own honest
        line (ONE voice, clear-before-speak):
          - already cancelled: nothing to do, say so;
          - not cancellable (shipped/delivered): industry-correct — direct cancel ends at
            shipment; the honest path is a return once it arrives (returns aren't built yet);
          - refunds already issued: a void reverses the FULL charge, which on top of a prior
            partial refund returns money twice — the mixed case belongs to a person;
          - risk-flagged: a risk signal on the session doesn't get a silent void (§A4a) —
            hand to a person (no spoken line; the handover deferral is the single voice)."""
        pending = state.pending_cancel
        assert pending is not None
        status = order_store.order_status(pending.order_id)
        if status == CANCELLED_STATUS:
            write_event({"event": "cancel_declined", "reason": "already_cancelled"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"That order for {pending.summary} is already cancelled and the "
                        "charge reversed - there's nothing more you need to do."
                    )
                ],
            }
        if not order_store.is_cancellable(pending.order_id):
            write_event({"event": "cancel_declined", "reason": "not_cancellable"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"That order for {pending.summary} has already shipped, so I can't "
                        "cancel it here - the way to handle it is a return once it arrives, "
                        "which our support team can set up for you."
                    )
                ],
            }
        if order_store.refunded_so_far(pending.order_id) > 0:
            write_event({"event": "cancel_declined", "reason": "has_refunds"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "There's already a refund on that order, so I can't cancel it on "
                        "this call - our support team can sort out the rest for you."
                    )
                ],
            }
        if risk.check_sim_swap():
            # A risk signal on the session doesn't get a silent void (§A4a) — hand to a
            # person. The escape pattern: NO spoken line here; the handover node's deferral
            # is the single voice (mirrors checkout/refund escape_human).
            write_event({"event": "cancel_stepup_to_human", "reason": "risk_flagged"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        return {}  # eligible: router -> cancel_confirm

    def cancel_confirm_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt: the cancel readback + deterministic consent. NO side effects; §4a
        barge -> re-confirm once (a truncated 'yes' does not authorize an irreversible void).
        Clock-A TTL checked FIRST (clear-before-speak) — a stale 'yes' can't void an order."""
        pending = state.pending_cancel
        assert pending is not None
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "cancel_expired", "reason": "pending_ttl"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That sat for a while, so I haven't changed anything. If you still "
                        "want to cancel it, just tell me again."
                    )
                ],
            }
        answer = interrupt(_cancel_readback_line(pending))
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(
                f"Sorry - just to be clear: cancel your order for {pending.summary}? Yes or no?"
            )
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "cancel_declined", "reason": "human_requested"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="cancel_order", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "cancel_declined", "reason": "declined"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [AIMessage("Okay, I'll leave that order as it is - nothing changed.")],
            }
        return {}  # yes: pending survives; router -> void

    def cancel_void_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Idempotent by the store's
        per-intent key; the store RE-VALIDATES status (§A4c) so a proposal that went stale
        (order shipped since assemble) can't void."""
        pending = state.pending_cancel
        assert pending is not None
        try:
            record = order_store.cancel_order(pending.idempotency_key, order_id=pending.order_id)
        except CancelError as exc:
            logger.warning("cancel refused by store: %s", exc)
            write_event({"event": "cancel_denied", "reason": "store_refused"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    # Outcome only — the handover node's deferral speaks the transfer
                    # sentence; adding one here would say it twice. No guessed reason: the
                    # store may have refused for shipment, a prior refund, or anything a
                    # real SoR adds later.
                    AIMessage("I wasn't able to cancel that order - nothing has changed.")
                ],
                "handover": HandoffRequest(
                    destination="human", reason_code="cancel_order", source="gate"
                ),
            }
        write_event({"event": "cancel_confirmed", "order_id": record.order_id})
        return {
            "pending_cancel": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - I've cancelled your order for {record.summary}. The "
                    f"${record.total_usd:.2f} charge goes back to your original "
                    "payment method."
                )
            ],
        }

    def abort_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: explicit abort while support was in flight."""
        write_event({"event": "support_cancelled", "reason": "aborted"})
        return {
            "pending_refund": None,
            "pending_cancel": None,
            "active_flow": None,
            "messages": [AIMessage("No problem - I've dropped that. Nothing has changed.")],
        }

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-support (§A9 no-trap)."""
        write_event({"event": "support_cancelled", "reason": "human_requested"})
        return {
            "pending_refund": None,
            "pending_cancel": None,
            "active_flow": None,
            "handover": HandoffRequest(destination="human", reason_code="other", source="gate"),
        }

    def route_after_guardrail(state: ReasoningState) -> str:
        if state.pending_cancel is not None:
            return "cancel"  # remedy steer: full money-back on an unshipped order is a void
        pending = state.pending_refund
        if pending is None:
            return "declined"  # over-amount/cancelled: the node spoke its own line + ends
        required = refund_required_level(pending.amount_usd, pending.destination)
        return "confirm" if verification_store.current_level() >= required else "stepup"

    def route_after_collect(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # attempts exhausted -> human
        pending = state.pending_refund
        assert pending is not None  # collect clears only via handover
        required = refund_required_level(pending.amount_usd, pending.destination)
        # Level raised to the requirement -> proceed; otherwise a re-collect was requested.
        return "confirm" if verification_store.current_level() >= required else "dispatch"

    def route_after_assemble(state: ReasoningState) -> str:
        # Which effect did assemble mint? (a valid proposal sets exactly one pending.)
        if state.active_flow == "left_support":
            return "leave"  # model left / two invalid proposals -> normal pipeline answers
        if state.pending_cancel is not None:
            return "cancel"
        if state.pending_refund is not None:
            return "refund"
        return "clarify"  # a clarifying question was streamed; end the turn in-flow

    def route_after_cancel_guardrail(state: ReasoningState) -> str:
        # Three outcomes the guardrail already resolved:
        #  - risk-flagged: cleared pending + set handover -> the human sink;
        #  - shipped/ineligible: cleared pending, spoke its own line, NO handover -> END;
        #  - eligible: pending survives, nothing spoken -> the readback interrupt.
        if state.handover is not None:
            return "handover"
        if state.pending_cancel is None:
            return "declined"  # ineligible: the node spoke + ended
        return "confirm"

    return SupportNodes(
        assemble=assemble_node,
        guardrail=guardrail_node,
        risk_check=risk_check_node,
        dispatch=dispatch_node,
        collect=collect_node,
        confirm=confirm_node,
        place=place_node,
        cancel_guardrail=cancel_guardrail_node,
        cancel_confirm=cancel_confirm_node,
        cancel_void=cancel_void_node,
        abort=abort_node,
        escape_human=escape_human_node,
        route_after_assemble=route_after_assemble,
        route_after_guardrail=route_after_guardrail,
        route_after_collect=route_after_collect,
        route_after_cancel_guardrail=route_after_cancel_guardrail,
        speakable_nodes=frozenset(
            {
                "support_guardrail",  # authors the over-amount-threshold decline line
                "support_risk_check",
                "support_collect",
                "support_confirm",
                "support_place",
                "support_cancel_guardrail",  # authors the shipped/ineligible decline line
                "support_cancel_confirm",
                "support_cancel_void",
                "support_abort",
            }
        ),
    )
