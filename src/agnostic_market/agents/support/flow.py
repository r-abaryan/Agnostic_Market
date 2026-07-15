"""The support gated flow (Tier 3, AGENTS §A4/§A9) — refunds + cancellations (T3/Group A),
returns and profile changes (Group C), with the A10a replay invariants in every sub-path.

The T3 refund shape (proven against real langgraph before this code — see the plan's
task-zero spike). Refund to a NEW instrument requires L2 regardless of amount (§A4b); an L1
caller is raised mid-flow by a committed OTP, THEN the refund resumes behind a readback:

    assemble ─▶ guardrail ─(L2 already)──────────▶ confirm[INT#2] ─▶ place
                   │                                    ▲
                   └─(L1, new instr.)─▶ risk_check ─▶ dispatch ─▶ collect[INT#1: verify+raise]

Group C sub-paths (same disciplines, built on the same seams):
    RETURNS  assemble/refund-tier-4 ─▶ return_guardrail ─▶ return_confirm[INT] ─▶ return_place
             (single interrupt, no step-up — no money moves at creation; the recorded refund
             releases at the Phase-4 SoR. The refund guardrail's return-first tier STEERS
             into this path, eligibility-checked BEFORE the steer line is spoken.)
    PROFILE  assemble ─▶ profile_guardrail ─▶ [the step-up chain] ─▶ profile_confirm[INT]
             ─▶ profile_place  (address/contact = L2 platform floor; the OTP goes to the
             number-on-file — changing the factor requires the OLD factor, §A4a. The chain
             is the SAME code as the refund's, via the _stepup.py factory.)

Why the decomposition (A10a): LangGraph re-runs an interrupted node from the TOP on resume,
and does NOT commit a node's update until it RETURNS. So the two interrupts sit in SEPARATE
nodes; `collect` verifies the OTP AFTER its own (only) interrupt and raises the store level
on its return (committed before the readback interrupt); and every side effect is either
post-interrupt (`place`/`return_place`/`profile_place`) or idempotent-pre-interrupt
(`dispatch` keyed by attempt — else a replay re-sends the OTP, S3).

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

from agnostic_market.agents._consent import classify_cancel_consent, classify_consent
from agnostic_market.agents._toolcalls import ack_extra_tool_calls
from agnostic_market.agents.support._stepup import build_stepup_nodes
from agnostic_market.agents.support.prompt import compose_support_prompt
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.orders import (
    CANCELLED_STATUS,
    FULFILLED_STATUSES,
    CancelError,
    LastOrderPointer,
    OrderStore,
    RefundError,
    ReturnError,
)
from agnostic_market.commerce.profile import ProfileError, ProfileStore
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.confirmation import (
    CREATE_RETURN_POLICY,
    ISSUE_REFUND_POLICY,
    PROFILE_CHANGE_POLICIES,
    ProfileField,
    RefundDestination,
    ToolConfirmationPolicy,
    profile_change_required_level,
    refund_required_level,
)
from agnostic_market.dtos.state import (
    HandoffRequest,
    PendingCancel,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.support")

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


def _return_readback_line(pending: PendingReturn, policy: ToolConfirmationPolicy) -> str:
    """The GRAPH-authored return readback — the `interrupt()` payload (Group C).

    Same loud-fail contract as `_readback_line`: the rendered dict MUST carry every field
    CREATE_RETURN_POLICY declares, even though the sentence phrases them naturally. States
    the honest money outcome — the refund follows the PROCESSED return, to the ORIGINAL
    payment method (a v1 constant: a steered new-instrument refund must not carry an
    unverified payout destination onto the return, dtos/state.py PendingReturn)."""
    rendered: dict[str, str] = {
        "order_id": pending.order_id,
        "total_amount": f"${pending.refund_due_usd:.2f}",
    }
    missing = policy.confirm_fields - rendered.keys()
    if missing:
        raise ValueError(f"readback cannot render declared confirm_fields: {sorted(missing)}")
    return (
        f"Just to confirm - set up a return for your {pending.summary} "
        f"({rendered['order_id']})? Once the return is processed, the "
        f"{rendered['total_amount']} refund goes back to your original payment method. "
        "Shall I go ahead?"
    )


def _profile_readback_line(pending: PendingProfileChange) -> str:
    """The GRAPH-authored profile-change readback — the `interrupt()` payload (Group C).

    The new VALUE is a declared critical field (one STT error = goods to the wrong street /
    an OTP factor the caller doesn't hold), so the loud-fail contract guarantees it is
    literally spoken. Spoken to the caller only — never logged (PII discipline)."""
    policy = PROFILE_CHANGE_POLICIES[pending.field]
    key = "new_address" if pending.field == "address" else "new_contact"
    rendered: dict[str, str] = {key: pending.new_value}
    missing = policy.confirm_fields - rendered.keys()
    if missing:
        raise ValueError(f"readback cannot render declared confirm_fields: {sorted(missing)}")
    noun = "delivery address" if pending.field == "address" else "contact number"
    return (
        f"Just to confirm - update the {noun} on your account to {rendered[key]}. "
        "Shall I go ahead?"
    )


class _ProposeRefund(BaseModel):
    order_key: str
    amount_usd: float
    destination: RefundDestination


class _ProposeCancel(BaseModel):
    order_key: str


class _ProposeReturn(BaseModel):
    order_key: str


class _ProposeProfileChange(BaseModel):
    field: ProfileField
    new_value: str


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
    # "confirm" (level sufficient) | "stepup" (OTP loop) | "cancel" (remedy steer) |
    # "return" (return-first steer, Group C) | "declined" (node spoke + ends).
    route_after_guardrail: Callable[[ReasoningState], str]
    # Branch after collect: the OTP verify may have raised to L2 (-> confirm), exhausted
    # attempts / flagged risk (-> handover), or asked for a re-collect (-> dispatch again).
    route_after_collect: Callable[[ReasoningState], str]
    # Branch after the cancel guardrail depends on the LIVE order status + risk (store-owned).
    # "confirm" (processing, no risk) | "handover" (shipped/delivered or risk-flagged).
    route_after_cancel_guardrail: Callable[[ReasoningState], str]
    # Returns sub-path (Group C): guardrail -> confirm[interrupt] -> place[effect].
    return_guardrail: Callable[[ReasoningState], dict[str, object]]
    return_confirm: Callable[[ReasoningState], dict[str, object]]
    return_place: Callable[[ReasoningState], dict[str, object]]
    # "confirm" (eligible) | "cancel" (unshipped steer) | "declined" (guardrail spoke + ends).
    route_after_return_guardrail: Callable[[ReasoningState], str]
    # Profile-change sub-path (Group C): guardrail -> [step-up chain] -> confirm -> place.
    # The step-up nodes are the FACTORY's profile instances — same bodies as the refund
    # chain, own graph node names + confirm target (R5).
    profile_guardrail: Callable[[ReasoningState], dict[str, object]]
    profile_risk_check: Callable[[ReasoningState], dict[str, object]]
    profile_dispatch: Callable[[ReasoningState], dict[str, object]]
    profile_collect: Callable[[ReasoningState], dict[str, object]]
    profile_confirm: Callable[[ReasoningState], dict[str, object]]
    profile_place: Callable[[ReasoningState], dict[str, object]]
    # "confirm" (level sufficient) | "stepup" (needs the OTP loop on the OLD factor).
    route_after_profile_guardrail: Callable[[ReasoningState], str]
    # "confirm" | "dispatch" (re-collect) | "handover" — the factory's decision router.
    route_after_profile_collect: Callable[[ReasoningState], str]
    speakable_nodes: frozenset[str]


def build_support_nodes(
    reasoning_model: BaseChatModel,
    order_store: OrderStore,
    verification_store: VerificationStore,
    otp: OtpProvider,
    risk: RiskProvider,
    policy: PolicyContext,
    profile_store: ProfileStore,
    pointer: LastOrderPointer,
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
    def propose_return(order_key: str) -> str:
        """Propose sending an order back (option number) - the refund follows the return."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def propose_profile_change(field: str, new_value: str) -> str:
        """Propose updating the account's delivery address or contact number.
        field is 'address' or 'contact'; new_value is the caller's stated new value."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_support() -> str:
        """Leave the support flow (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    model = reasoning_model.bind_tools(
        [propose_refund, propose_cancel, propose_return, propose_profile_change, leave_support]
    )

    def _leave(new_messages: list, call_id: str) -> dict[str, object]:
        new_messages.append(ToolMessage("left support", tool_call_id=call_id))
        write_event({"event": "support_left", "reason": "left_flow"})
        return {
            "messages": new_messages,
            "active_flow": "left_support",
            "pending_refund": None,
            "pending_cancel": None,
            "pending_return": None,
            "pending_profile_change": None,
        }

    def _return_eligibility(order_id: str) -> str | None:
        """ONE return-eligibility check, shared by the return guardrail AND the refund
        tier-4 steer — the steer must never promise a return this would then decline (the
        ask-then-decline class, live call #9 P2, prevented by construction). Returns a
        closed reason slug or None (eligible). Ordering mirrors the guardrail tiers:
        status truth first, then dedup, then the window."""
        status = order_store.order_status(order_id)
        if status == CANCELLED_STATUS:
            return "order_cancelled"
        if status not in FULFILLED_STATUSES:
            return "not_shipped"
        if order_store.return_for_order(order_id) is not None:
            return "already_open"
        delivered = order_store.delivered_at_epoch(order_id)
        if delivered is not None and time.time() - delivered > policy.return_window_days * 86400:
            return "out_of_window"
        return None

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE support: propose a REFUND (order+amount+destination) or a CANCEL
        (order), or clarify, or leave. Mints PendingRefund/PendingCancel (per-intent key) on a
        valid proposal — A10a rule 2: the idempotency key exists in state before any effect."""
        orders = order_store.actionable_orders()
        by_key = {o.key: o for o in orders}
        prompt = SystemMessage(
            compose_support_prompt(display_name, orders, policy, pointer.get())
        )
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

            if call["name"] == "propose_return":
                try:
                    proposed = _ProposeReturn.model_validate(call["args"])
                except ValueError:
                    proposed = None
                chosen = by_key.get(proposed.order_key) if proposed else None
                if chosen is None:
                    feedback = f"Invalid order. Valid order numbers: {', '.join(sorted(by_key))}."
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                new_messages.append(
                    ToolMessage(f"proposed return on order {chosen.key}", tool_call_id=call["id"])
                )
                # refund_due is CODE-computed: what's still refundable on the order (captured
                # minus refunds paid minus prior return promises) — never model arithmetic.
                refund_due = max(
                    0.0,
                    round(
                        chosen.total_usd
                        - order_store.refunded_so_far(chosen.order_id)
                        - order_store.return_refund_due(chosen.order_id),
                        2,
                    ),
                )
                pending_return = PendingReturn(
                    order_id=chosen.order_id,
                    summary=chosen.summary,
                    refund_due_usd=refund_due,
                    idempotency_key=uuid.uuid4().hex,
                    created_at=time.time(),
                )
                return {"messages": new_messages, "pending_return": pending_return}

            if call["name"] == "propose_profile_change":
                try:
                    change = _ProposeProfileChange.model_validate(call["args"])
                except ValueError:
                    change = None
                if change is None or not change.new_value.strip():
                    feedback = (
                        "Invalid proposal. field must be 'address' or 'contact'; "
                        "new_value must be the caller's stated new value."
                    )
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                # NO value echo in the tool result (PII: thread history is model-visible
                # context, but the persisted line must not carry the raw value a second
                # time beyond the pending itself).
                new_messages.append(
                    ToolMessage(
                        f"proposed profile change: {change.field}", tool_call_id=call["id"]
                    )
                )
                pending_change = PendingProfileChange(
                    field=change.field,
                    new_value=change.new_value.strip(),
                    # The MASKED on-file contact the OTP goes to — for a contact change
                    # this is the OLD factor (change-the-factor needs the factor, §A4a).
                    factor_ref=profile_store.contact_on_file(),
                    idempotency_key=uuid.uuid4().hex,
                    attempt_key=uuid.uuid4().hex,
                    created_at=time.time(),
                )
                return {"messages": new_messages, "pending_profile_change": pending_change}

            if call["name"] == "propose_refund":
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

            # Explicit terminal guard (not fall-through): every bound tool has a branch above,
            # so an unhandled name means a NEW tool was bound without a handler — fail loud
            # rather than silently misroute it into the last branch (a silent wrong-mutation).
            raise ValueError(f"support assemble: unhandled tool {call['name']!r}")
        logger.warning("support assemble: two invalid proposals; leaving flow")
        write_event({"event": "support_left", "reason": "invalid_proposals"})
        return {
            "messages": new_messages,
            "active_flow": "left_support",
            "pending_refund": None,
            "pending_cancel": None,
            "pending_return": None,
            "pending_profile_change": None,
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
           `refund_require_human_above_usd` goes to a PERSON regardless of level. This is the
           stronger gate (an authorization/fraud ceiling), so it precedes return-first: a
           person handling an oversized refund also arranges the return, whereas leading with
           "just set up a return" would hide that the amount needs review (live 2026-07-10:
           a $250 shipped refund spoke only the return line, masking the $200 human line).
           It authors its own decline (clear-before-speak) and ENDs — there is NO handover
           routing yet (no warm-transfer target until Phase 4); the human intent is recorded
           in telemetry so the transfer can be wired there without re-finding this spot.
        1b. OPEN RETURN (Group C, between the cancelled and steer tiers): the refund path
           for an order with an open return IS that return — point at its RMA and end;
           every later gate is moot (the promise was vetted at return creation).
        4. RETURN-FIRST STEER (industry standard, live 2026-07-10: a $179.98 refund paid
           out on shipped shoes with no return created — money back AND goods kept): a
           refund on a SHIPPED/DELIVERED order above the merchant's `returnless_under_usd`
           (and within the human line) waits for the return — Group C converts it INTO the
           returns sub-path (eligibility-checked BEFORE the steer line is spoken, via the
           shared helper; out-of-window declines honestly instead). At/below the threshold
           = a deliberate returnless refund, proceeds.
        5. LEVEL gate (§A4b fraud floor): destination -> required level vs the LIVE store
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
        open_return = order_store.return_for_order(pending.order_id)
        if open_return is not None:
            # An OPEN RETURN makes every later gate moot (Group C): the refund path for this
            # order IS the return — its promise was vetted when the return was created, so
            # re-gating the amount here would speak a contradictory second line. Placed
            # before the steer/amount tiers; disjoint from the cancel steer by construction
            # (open returns exist only on fulfilled orders; the steer needs cancellable).
            write_event({"event": "refund_denied", "reason": "return_already_open"})
            return {
                "pending_refund": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"A return is already set up for that order ({open_return.rma_id}) - "
                        f"the ${open_return.refund_due_usd:.2f} refund is issued once it's "
                        "processed."
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
            # Over the merchant's human-review line: the stronger gate, checked BEFORE
            # return-first (a person handling the amount also arranges the return). Speak the
            # specific reason and END (no generic deferral over it), same one-voice pattern
            # as the shipped-cancel decline.
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
        if (
            order_store.order_status(pending.order_id) in FULFILLED_STATUSES
            and pending.amount_usd > policy.refund_returnless_under_usd
        ):
            # Return-first STEER (Group C, within the human line): the goods are out; over
            # the returnless threshold the refund waits for the return — and the returns
            # sub-path can now ARRANGE it. Eligibility is checked BEFORE speaking (the
            # shared helper): the steer must never promise a return the return guardrail
            # would then decline (ask-then-decline, live call #9 P2). The only reachable
            # ineligible reason here is out_of_window (cancelled/unshipped can't be
            # FULFILLED; already_open was caught by the earlier tier).
            if _return_eligibility(pending.order_id) == "out_of_window":
                write_event({"event": "refund_needs_return", "reason": "out_of_window"})
                return {
                    "pending_refund": None,
                    "active_flow": None,
                    "messages": [
                        AIMessage(
                            "Since that order has already shipped, the "
                            f"${pending.amount_usd:.2f} refund is issued once the return is "
                            "set up - but that delivery is outside the "
                            f"{policy.return_window_days}-day return window, so our support "
                            "team will need to review it."
                        )
                    ],
                }
            # Eligible: convert the refund into a return (the tier-2 mint-and-route
            # pattern). Amount/summary carried; destination deliberately NOT carried —
            # v1 returns refund to the ORIGINAL payment method (PendingReturn docstring).
            write_event({"event": "refund_steered_to_return", "order_id": pending.order_id})
            return {
                "pending_refund": None,
                "pending_return": PendingReturn(
                    order_id=pending.order_id,
                    summary=order_store.order_item_summary(pending.order_id)
                    or pending.order_id,
                    refund_due_usd=pending.amount_usd,
                    idempotency_key=uuid.uuid4().hex,
                    created_at=time.time(),
                ),
                "messages": [
                    AIMessage(
                        f"Since that order has already shipped, the ${pending.amount_usd:.2f} "
                        "refund is issued once the return is set up - I can arrange that "
                        "return for you now."
                    )
                ],
            }
        required = refund_required_level(pending.amount_usd, pending.destination)
        if verification_store.current_level() < required:
            write_event({"event": "refund_stepup_required", "required_level": required})
        return {}

    # The T3 step-up chain (risk_check -> dispatch -> collect) — extracted VERBATIM to the
    # family-parametrized factory (_stepup.py, Group C) so the profile flow runs the SAME
    # code. Refund behavior + event names are byte-identical to the pre-factory nodes; the
    # five T3 security tests are the regression gate for this extraction.
    refund_stepup = build_stepup_nodes(
        verification_store,
        otp,
        risk,
        pending_field="pending_refund",
        required_level=lambda p: refund_required_level(p.amount_usd, p.destination),
        event_prefix="refund",
    )

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
        pointer.set(pending.order_id)  # the order most recently discussed (Group C L4)
        write_event(
            {
                "event": "refund_confirmed",
                "refund_id": record.refund_id,
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
                    f"{pending.instrument_ref}. Your reference is {record.refund_id}."
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
        # Cancel-polarity consent: the question IS "shall I cancel?", so "yeah cancel it"
        # must read as yes (plain classify_consent reads the 'cancel' as a no — live
        # 2026-07-10 the phrase never even reached here, but this seals the readback too).
        answer = interrupt(_cancel_readback_line(pending))
        verdict = classify_cancel_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            # The retry names the order too — the caller may have reached this branch by
            # questioning WHICH order it is (live: "isn't my most recent ORD-1001?").
            retry = interrupt(
                f"Sorry - just to be clear: cancel your order for {pending.summary} "
                f"({pending.order_id})? Yes or no?"
            )
            verdict = classify_cancel_consent(str(retry.get("text", "")))
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
        pointer.set(record.order_id)  # the order most recently discussed (Group C L4)
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

    # --- returns sub-path (Group C; single interrupt: guardrail -> confirm -> place) ------

    def return_guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced return eligibility (never the model), via the SHARED
        `_return_eligibility` helper — the same check the refund tier-4 steer runs BEFORE
        promising a return, so this node's declines are unreachable for steered pendings
        (§A4c defense only) and load-bearing for DIRECT propose_return.

        Tier order: status truth (cancelled / unshipped-steers-to-cancel), dedup
        (already-open names the RMA), the WINDOW, then the amount. Window-before-amount is
        a deliberate inversion of the refund flow's amount-before-return-first, same
        masking lesson (the decisive gate speaks): out-of-window is an absolute eligibility
        fact — no self-serve return exists at ANY amount, and the person reviewing the
        window exception handles the amount too."""
        pending = state.pending_return
        assert pending is not None
        reason = _return_eligibility(pending.order_id)
        if reason == "order_cancelled":
            write_event({"event": "return_denied", "reason": "order_cancelled"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"That order for {pending.summary} was already cancelled and the "
                        "charge reversed - there's nothing to return."
                    )
                ],
            }
        if reason == "not_shipped":
            # Remedy steer (the refund tier-2 class): a "return" on unshipped goods IS a
            # cancel — nothing has been sent, so nothing can go back. Mint the cancel and
            # route into its path; its guardrail re-validates eligibility.
            captured = order_store.captured_total(pending.order_id)
            write_event({"event": "return_steered_to_cancel", "order_id": pending.order_id})
            return {
                "pending_return": None,
                "pending_cancel": PendingCancel(
                    order_id=pending.order_id,
                    summary=pending.summary,
                    idempotency_key=uuid.uuid4().hex,
                    created_at=time.time(),
                ),
                "messages": [
                    AIMessage(
                        "That order hasn't shipped yet, so there's nothing to send back - "
                        f"I can cancel it instead; the full ${captured:.2f} goes back to "
                        "your original payment method."
                    )
                ],
            }
        if reason == "already_open":
            existing = order_store.return_for_order(pending.order_id)
            assert existing is not None  # the eligibility helper just saw it
            write_event({"event": "return_denied", "reason": "already_open"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"A return for that order is already set up - your reference is "
                        f"{existing.rma_id}. The ${existing.refund_due_usd:.2f} refund is "
                        "issued once it's processed."
                    )
                ],
            }
        if reason == "out_of_window":
            write_event({"event": "return_denied", "reason": "out_of_window"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"That order was delivered more than {policy.return_window_days} "
                        "days ago, which is outside the return window - our support team "
                        "can review an exception for you."
                    )
                ],
            }
        if pending.refund_due_usd > policy.refund_require_human_above_usd:
            # DIRECT proposals only (a steered return already passed the refund amount
            # gate): an oversized refund promise needs a person, same authorization ceiling
            # as the refund flow's tier 3.
            write_event(
                {
                    "event": "return_needs_human",
                    "reason": "over_amount_threshold",
                    "handover_intent": "human",
                }
            )
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"A ${pending.refund_due_usd:.2f} refund is above what I can set up "
                        "on this call - our support team can arrange that return and refund "
                        "for you."
                    )
                ],
            }
        return {}  # eligible: router -> return_confirm

    def return_confirm_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt: the return readback + deterministic consent. NO side effects;
        §4a barge -> re-confirm once; Clock-A TTL FIRST (clear-before-speak). PLAIN
        classify_consent — the question is "set up the return?", so a "cancel" correctly
        reads as no (cancel-polarity applies only where cancelling IS the question)."""
        pending = state.pending_return
        assert pending is not None
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "return_expired", "reason": "pending_ttl"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That return confirmation sat for a while, so I haven't set "
                        "anything up. If you'd still like it, just tell me again."
                    )
                ],
            }
        answer = interrupt(_return_readback_line(pending, CREATE_RETURN_POLICY))
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(
                f"Sorry - just to be clear: set up a return for {pending.summary} "
                f"({pending.order_id})? Yes or no?"
            )
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "return_cancelled", "reason": "human_requested"})
            return {
                "pending_return": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="refund", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "return_cancelled", "reason": "declined"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    AIMessage("Okay, I won't set up a return - nothing has changed.")
                ],
            }
        return {}  # yes: pending survives; router -> place

    def return_place_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Idempotent by the
        store's per-intent key; the store RE-VALIDATES eligibility (§A4c) so a proposal
        that went stale can't create a ghost return. No live level re-check: creating a
        return is L1 (no money moves here — the recorded refund releases at the Phase-4
        SoR, which re-runs the destination->level check at release)."""
        pending = state.pending_return
        assert pending is not None
        try:
            record = order_store.create_return(
                pending.idempotency_key,
                order_id=pending.order_id,
                refund_due_usd=pending.refund_due_usd,
                destination="original",  # v1 constant — see PendingReturn's docstring
            )
        except ReturnError as exc:
            logger.warning("return refused by store: %s", exc)
            write_event({"event": "return_denied", "reason": "store_refused"})
            return {
                "pending_return": None,
                "active_flow": None,
                "messages": [
                    # Outcome only — the handover node's deferral speaks the transfer line.
                    AIMessage("I wasn't able to set up that return - nothing has changed.")
                ],
                "handover": HandoffRequest(
                    destination="human", reason_code="refund", source="gate"
                ),
            }
        pointer.set(record.order_id)  # the order most recently discussed (Group C L4)
        write_event(
            {
                "event": "return_confirmed",
                "rma_id": record.rma_id,
                "order_id": record.order_id,
                "refund_due": record.refund_due_usd,
            }
        )
        return {
            "pending_return": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - your return for {pending.summary} is set up; your reference "
                    f"is {record.rma_id}. The ${record.refund_due_usd:.2f} refund goes "
                    "back to your original payment method once the return is processed."
                )
            ],
        }

    # --- profile-change sub-path (Group C; the refund T3 shape minus money:
    # ---  guardrail -> [risk_check -> dispatch -> collect(INT)] -> confirm(INT) -> place) --

    profile_stepup = build_stepup_nodes(
        verification_store,
        otp,
        risk,
        pending_field="pending_profile_change",
        required_level=lambda p: profile_change_required_level(p.field),
        event_prefix="profile",
    )

    def profile_guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-gate for a profile change: no merchant tiers (address/contact have no
        merchant knobs — the L2 requirement is the §A4a platform floor), so this node only
        records whether a step-up is needed; the router (closed over the LIVE store) decides
        confirm vs stepup. The OTP goes to the number-on-file — for a contact change that is
        the OLD factor: changing the factor requires the factor (the ladder constraint)."""
        pending = state.pending_profile_change
        assert pending is not None
        if verification_store.current_level() < profile_change_required_level(pending.field):
            write_event(
                {"event": "profile_stepup_required", "required_level": 2, "field": pending.field}
            )
        return {}

    def route_after_profile_guardrail(state: ReasoningState) -> str:
        pending = state.pending_profile_change
        assert pending is not None
        required = profile_change_required_level(pending.field)
        return "confirm" if verification_store.current_level() >= required else "stepup"

    def profile_confirm_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt: the profile-change readback + deterministic consent. NO side
        effects; §4a barge -> re-confirm once; Clock-A TTL FIRST (clear-before-speak).
        PLAIN classify_consent (nothing here reads 'cancel' as the affirmative)."""
        pending = state.pending_profile_change
        assert pending is not None
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "profile_expired", "reason": "pending_ttl"})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That change sat for a while, so I haven't updated anything. If "
                        "you'd still like it, just tell me again."
                    )
                ],
            }
        answer = interrupt(_profile_readback_line(pending))
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            noun = "delivery address" if pending.field == "address" else "contact number"
            retry = interrupt(
                f"Sorry - just to be clear: update the {noun} on your account to "
                f"{pending.new_value}? Yes or no?"
            )
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "profile_change_cancelled", "reason": "human_requested"})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code=(
                        "address_change" if pending.field == "address" else "contact_change"
                    ),
                    source="model",
                ),
            }
        if verdict == "no":
            write_event({"event": "profile_change_cancelled", "reason": "declined"})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "messages": [
                    AIMessage("Okay, I'll leave your details as they are - nothing has changed.")
                ],
            }
        return {}  # yes: pending survives; router -> place

    def profile_place_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Re-validates the LIVE
        level FIRST (§A4c — the change must not apply if the L2 grant lapsed between collect
        and here), then applies via the store's per-intent key (idempotent). Telemetry
        carries the FIELD SLUG + grant methods only — NEVER the value (PII discipline; the
        value is spoken to the caller, not persisted to observability)."""
        pending = state.pending_profile_change
        assert pending is not None
        required = profile_change_required_level(pending.field)
        if verification_store.current_level() < required:
            write_event({"event": "profile_stepup_failed", "reason": "level_lapsed_at_place"})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        try:
            record = profile_store.update_profile(
                pending.idempotency_key, field=pending.field, new_value=pending.new_value
            )
        except ProfileError as exc:
            logger.warning("profile change refused by store: %s", type(exc).__name__)
            write_event({"event": "profile_change_denied", "reason": "store_refused"})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "messages": [
                    # Outcome only — the handover node's deferral speaks the transfer line.
                    AIMessage("I wasn't able to update that - nothing has changed.")
                ],
                "handover": HandoffRequest(
                    destination="human",
                    reason_code=(
                        "address_change" if pending.field == "address" else "contact_change"
                    ),
                    source="gate",
                ),
            }
        write_event(
            {
                "event": "profile_change_confirmed",
                "field": record.field,
                "verification": [g.get("method") for g in verification_store.grants],
            }
        )
        noun = "delivery address" if record.field == "address" else "contact number"
        return {
            "pending_profile_change": None,
            "active_flow": None,
            "messages": [
                AIMessage(f"Done - the {noun} on your account is updated to {record.new_value}.")
            ],
        }

    def abort_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: explicit abort while support was in flight. The copy names
        what was dropped (the REQUEST) and what wasn't touched (orders) — "I've dropped
        that" alone let a caller believe an order was cancelled (live 2026-07-10)."""
        write_event({"event": "support_cancelled", "reason": "aborted"})
        return {
            "pending_refund": None,
            "pending_cancel": None,
            "pending_return": None,
            "pending_profile_change": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    "No problem - I've dropped that request. Your orders are unchanged."
                )
            ],
        }

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-support (§A9 no-trap)."""
        write_event({"event": "support_cancelled", "reason": "human_requested"})
        return {
            "pending_refund": None,
            "pending_cancel": None,
            "pending_return": None,
            "pending_profile_change": None,
            "active_flow": None,
            "handover": HandoffRequest(destination="human", reason_code="other", source="gate"),
        }

    def route_after_guardrail(state: ReasoningState) -> str:
        if state.pending_cancel is not None:
            return "cancel"  # remedy steer: full money-back on an unshipped order is a void
        if state.pending_return is not None:
            return "return"  # return-first steer: the refund converted into a return
        pending = state.pending_refund
        if pending is None:
            return "declined"  # over-amount/cancelled: the node spoke its own line + ends
        required = refund_required_level(pending.amount_usd, pending.destination)
        return "confirm" if verification_store.current_level() >= required else "stepup"

    def route_after_assemble(state: ReasoningState) -> str:
        # Which effect did assemble mint? (a valid proposal sets exactly one pending.)
        if state.active_flow == "left_support":
            return "leave"  # model left / two invalid proposals -> normal pipeline answers
        if state.pending_cancel is not None:
            return "cancel"
        if state.pending_return is not None:
            return "return"
        if state.pending_profile_change is not None:
            return "profile"
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

    def route_after_return_guardrail(state: ReasoningState) -> str:
        # "cancel" (unshipped steer minted a PendingCancel) | "declined" (guardrail spoke
        # its own line + ends) | "confirm" (eligible: pending survives -> the readback).
        if state.pending_cancel is not None:
            return "cancel"
        if state.pending_return is None:
            return "declined"
        return "confirm"

    return SupportNodes(
        assemble=assemble_node,
        guardrail=guardrail_node,
        risk_check=refund_stepup.risk_check,
        dispatch=refund_stepup.dispatch,
        collect=refund_stepup.collect,
        confirm=confirm_node,
        place=place_node,
        cancel_guardrail=cancel_guardrail_node,
        cancel_confirm=cancel_confirm_node,
        cancel_void=cancel_void_node,
        abort=abort_node,
        escape_human=escape_human_node,
        route_after_assemble=route_after_assemble,
        route_after_guardrail=route_after_guardrail,
        route_after_collect=refund_stepup.route_after_collect,
        route_after_cancel_guardrail=route_after_cancel_guardrail,
        return_guardrail=return_guardrail_node,
        return_confirm=return_confirm_node,
        return_place=return_place_node,
        route_after_return_guardrail=route_after_return_guardrail,
        profile_guardrail=profile_guardrail_node,
        profile_risk_check=profile_stepup.risk_check,
        profile_dispatch=profile_stepup.dispatch,
        profile_collect=profile_stepup.collect,
        profile_confirm=profile_confirm_node,
        profile_place=profile_place_node,
        route_after_profile_guardrail=route_after_profile_guardrail,
        route_after_profile_collect=profile_stepup.route_after_collect,
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
                "support_return_guardrail",  # authors the ineligible-return decline lines
                "support_return_confirm",
                "support_return_place",
                "support_profile_risk_check",
                "support_profile_collect",
                "support_profile_confirm",
                "support_profile_place",
                "support_abort",
            }
        ),
    )
