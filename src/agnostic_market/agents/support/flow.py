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
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, model_validator

from agnostic_market.agents._consent import classify_cancel_consent, classify_consent
from agnostic_market.agents._toolcalls import ack_extra_tool_calls, unknown_tool_result
from agnostic_market.agents.clarification import (
    advance_clarification,
    with_clarification_lifecycle,
)
from agnostic_market.agents.support._stepup import build_stepup_nodes
from agnostic_market.agents.support.prompt import compose_support_prompt
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    order_mutation_allowed,
    order_read_allowed,
)
from agnostic_market.commerce.orders import (
    CANCELLED_STATUS,
    FULFILLED_STATUSES,
    CancelError,
    OrderCandidate,
    OrderStore,
    RecentOrderContext,
    RefundError,
    ReturnError,
    render_batch_cancel_outcome,
    render_order_list_line,
)
from agnostic_market.commerce.payment_instruments import PaymentInstrumentDirectory
from agnostic_market.commerce.profile import ProfileError, ProfileStore
from agnostic_market.commerce.spoken import (
    caller_stated_order_id,
    caller_stated_phone,
)
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
    validate_confirmation_rendering,
)
from agnostic_market.dtos.orchestration import (
    CancellableOrderScope,
    CancelOrders,
    CancelScope,
    ChangeProfile,
    ExplicitOrderSet,
    ExplicitOrderTarget,
    IntentRequest,
    ListOrders,
    RefundOrder,
    ReturnOrder,
)
from agnostic_market.dtos.state import (
    BatchCancelOutcome,
    CancelTarget,
    HandoffRequest,
    PendingCancelBatch,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    PolicyContext,
    ReasoningState,
    SupportAuthorizationDetail,
    SupportClarification,
    SupportQuestionDetail,
)

logger = logging.getLogger("agnostic_market.agents.support")

# Sentinel: the mutation authorization needs the caller to verify their identity (OTP) first.
# An UNBOUND caller's mutation can never authorize this turn (a rung-1 contact grant is
# READ-only, Fix 2) — the assemble branch routes into the identity flow instead of granting or
# revealing anything. Distinct from a corrective LINE (which is spoken); this one is CONTROL flow.
_NEEDS_IDENTITY: Literal["<needs-identity>"] = "<needs-identity>"

# Platform-authored Support clarification copy. Authorization failures deliberately share one
# existence-oracle-safe line; the threshold variant adds only truthful store-contact guidance.
_SUPPORT_COMBINED_NOT_FOUND = (
    "I couldn't find an order matching those details. Please double-check the order number "
    "and the email or phone on the account. I haven't changed anything."
)
_SUPPORT_NOT_FOUND_OFFER_HUMAN = (
    "I couldn't find an order matching those details. Please contact the store directly for "
    "help verifying the order. I haven't changed anything."
)
_SUPPORT_CLARIFICATION_LINES: dict[SupportQuestionDetail | SupportAuthorizationDetail, str] = {
    "action": "What would you like help with: a cancellation, return, refund, or profile update?",
    "order": "What is the order number, for example ORD-1234?",
    "amount": "What amount would you like refunded?",
    "refund_destination": "Should the refund go back to the original payment method?",
    "profile_field": "Would you like to update your delivery address or contact number?",
    "profile_value": "What new delivery address or contact number would you like to use?",
    "order_match": _SUPPORT_COMBINED_NOT_FOUND,
    "order_match_human_help": _SUPPORT_NOT_FOUND_OFFER_HUMAN,
}
_ORDER_SELECTION_ACK = "order selection checked"
_ORDER_REFERENCE_REQUIRED_ACK = "caller-stated order number required"
_PROFILE_VALUE_REQUIRED_ACK = "caller-stated profile value required"
_CONTINUATION_NOT_FOUND = (
    "I couldn't find an order matching those details, so I haven't changed anything."
)
# The denial counter's bucket for references that never resolved to a store order — one
# bounded key, never the caller's free text (an attacker probing random ids must not grow
# an unbounded dict of their own strings).
_UNRESOLVED_ORDER = "<unresolved>"


def _last_user_text(state: ReasoningState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _caller_stated_profile_value(utterance: str, field: ProfileField, value: str) -> bool:
    stated = " ".join(utterance.casefold().split())
    proposed = " ".join(value.casefold().split())
    if field != "contact":
        return bool(proposed) and proposed in stated
    return caller_stated_phone(utterance, value)


def _caller_stated_refund_amount(utterance: str, amount_usd: float) -> float | None:
    variants = {f"{amount_usd:g}", f"{amount_usd:.2f}"}
    amount = "|".join(re.escape(value) for value in sorted(variants, key=len, reverse=True))
    pattern = rf"(?:\$\s*(?:{amount})\b|\b(?:{amount})\s*(?:dollars?|usd)\b)"
    return round(amount_usd, 2) if re.search(pattern, utterance, re.IGNORECASE) else None


def _caller_stated_refund_destination(
    utterance: str, destination: RefundDestination
) -> RefundDestination | None:
    if destination == "original":
        return destination
    lowered = utterance.casefold()
    if destination == "new_instrument" and re.search(
        r"\b(?:new|different|another)\b[^.?!]*\b(?:card|instrument)\b", lowered
    ):
        return destination
    if destination == "new_address" and re.search(
        r"\b(?:new|different|another)\b[^.?!]*\baddress\b", lowered
    ):
        return destination
    return None


def _refund_confirmation_phrase(pending: PendingRefund, policy: ToolConfirmationPolicy) -> str:
    rendered: dict[str, str] = {
        "order_id": pending.order_id,
        "total_amount": f"${pending.amount_usd:.2f}",
        "new_payment_instrument_ref": pending.instrument_ref,
    }
    phrase = (
        f"a {rendered['total_amount']} refund on your order for {pending.summary} "
        f"({rendered['order_id']}) to your {rendered['new_payment_instrument_ref']}"
    )
    return validate_confirmation_rendering(policy, rendered, phrase)


def _mint_cancel_batch(targets: list[tuple[str, str]]) -> PendingCancelBatch:
    """The ONE place a PendingCancelBatch is minted (direct multi-cancel AND the single-order
    remedy steers, which pass one target). Each `(order_id, summary)` gets its own per-INTENT
    idempotency key (A10a rule 2: the key exists in state before any effect; a batch of N
    yields exactly N effects). A single cancel is a one-target batch."""
    return PendingCancelBatch(
        targets=tuple(
            CancelTarget(order_id=oid, summary=summary, idempotency_key=uuid.uuid4().hex)
            for oid, summary in targets
        ),
        created_at=time.time(),
    )


def _cancel_target_phrase(target: CancelTarget) -> str:
    return f"your order for {target.summary} ({target.order_id})"


def _ineligible_clause(o: BatchCancelOutcome) -> str:
    """The pre-consent honest note about a target the preflight can't cancel — amount-free,
    stated so the caller hears the whole truth before the readback question."""
    if o.outcome == "already_cancelled":
        return f"your order for {o.summary} ({o.order_id}) is already cancelled"
    if o.outcome == "has_refunds":
        return (
            f"your order for {o.summary} ({o.order_id}) already has a refund, so that one needs "
            "our support team"
        )
    # not_cancellable (shipped/delivered)
    return (
        f"your order for {o.summary} ({o.order_id}) has already shipped, so that one needs a "
        "return instead"
    )


def _cancel_readback_line(pending: PendingCancelBatch) -> str:
    """The GRAPH-authored cancel readback — the `interrupt()` payload. Names the EXACT eligible
    target(s) so a mis-heard reference can't void the wrong one, STATES any ineligible ones
    (whole truth before consent), states irreversibility (§7a). Amount-free: single-cancel
    speech stays byte-identical; amounts appear only in the final store-backed result. A
    one-target batch reads exactly like the old single-cancel readback."""
    targets = pending.targets
    if len(targets) == 1:
        eligible = f"cancel {_cancel_target_phrase(targets[0])}"
    else:
        phrases = [_cancel_target_phrase(t) for t in targets]
        eligible = "cancel " + "; ".join(phrases[:-1]) + f"; and {phrases[-1]}"
    prefix = ""
    if pending.ineligible:
        notes = [_ineligible_clause(o) for o in pending.ineligible]
        joined = "; ".join(notes[:-1]) + (f"; and {notes[-1]}" if len(notes) > 1 else notes[0])
        prefix = joined[0].upper() + joined[1:] + ". "
    return f"{prefix}Just to confirm - {eligible}? This can't be undone."


def _return_confirmation_phrase(pending: PendingReturn, policy: ToolConfirmationPolicy) -> str:
    rendered: dict[str, str] = {
        "order_id": pending.order_id,
        "total_amount": f"${pending.refund_due_usd:.2f}",
    }
    phrase = (
        f"set up a return for your {pending.summary} "
        f"({rendered['order_id']})? You'll send the items back, and once we receive them the "
        f"{rendered['total_amount']} refund goes to your original payment method"
    )
    return validate_confirmation_rendering(policy, rendered, phrase)


def _profile_confirmation_phrase(pending: PendingProfileChange) -> str:
    """The canonical, policy-validated description used by both profile-change prompts.

    The new VALUE is a declared critical field (one STT error = goods to the wrong street /
    an OTP factor the caller doesn't hold), so the loud-fail contract guarantees it is
    literally spoken. Spoken to the caller only — never logged (PII discipline)."""
    policy = PROFILE_CHANGE_POLICIES[pending.field]
    key = "new_address" if pending.field == "address" else "new_contact"
    rendered: dict[str, str] = {key: pending.new_value}
    noun = "delivery address" if pending.field == "address" else "contact number"
    phrase = f"update the {noun} on your account to {rendered[key]}"
    return validate_confirmation_rendering(policy, rendered, phrase)


class _ProposeRefund(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_key: str
    amount_usd: float
    destination: RefundDestination


class _ProposeCancel(BaseModel):
    # Batch-capable (F-16.2): one call carries EVERY order the caller named ("cancel both").
    # A single cancel is a one-element list. `scope` carries a semantic "all/both" selector
    # when the caller named no ids (resolved against the authorized candidate set).
    model_config = ConfigDict(extra="forbid")

    order_keys: list[str] | None = None
    scope: CancelScope | None = None

    @model_validator(mode="after")
    def exactly_one_selector(self) -> _ProposeCancel:
        """Reject malformed/stale calls instead of widening them to account scope."""
        normalized = [key.strip() for key in (self.order_keys or [])]
        if any(not key for key in normalized):
            raise ValueError("order_keys must contain only non-empty references")
        has_keys = bool(normalized)
        has_scope = self.scope is not None
        if has_keys == has_scope:
            raise ValueError("provide exactly one of order_keys or scope")
        self.order_keys = normalized
        return self


class _ProposeReturn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_key: str


class _ProposeProfileChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: ProfileField
    new_value: str


class _RequestSupportClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: SupportQuestionDetail


@dataclass(frozen=True)
class SupportNodes:
    """The support flow's node callables + its caller-facing (speakable) node names.

    Wiring is the graph builder's job (frontline/graph.py); this module owns only behavior.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    continuation: Callable[[ReasoningState], dict[str, object]]
    clarify: Callable[[ReasoningState], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    risk_check: Callable[[ReasoningState], dict[str, object]]
    dispatch: Callable[[ReasoningState], dict[str, object]]
    collect: Callable[[ReasoningState], dict[str, object]]
    confirm: Callable[[ReasoningState], dict[str, object]]
    place: Callable[[ReasoningState], dict[str, object]]
    cancel_guardrail: Callable[[ReasoningState], dict[str, object]]
    cancel_confirm: Callable[[ReasoningState], dict[str, object]]
    cancel_void: Callable[[ReasoningState], dict[str, object]]
    # Resolve a CancellableOrderScope into a batch after authorization. Reached from the typed
    # continuation or directly from an already-bound caller's assemble, and never speaks a list.
    resolve: Callable[[ReasoningState], dict[str, object]]
    abort: Callable[[ReasoningState], dict[str, object]]
    escape_human: Callable[[ReasoningState], dict[str, object]]
    # After assemble: "refund" | "cancel" | "return" | "profile" | "resolve" (bound scope) |
    # "needs_identity" (unbound scope -> verify first) | "leave" | "clarify".
    route_after_assemble: Callable[[ReasoningState], str]
    # After resolve: "confirm" (batch frozen -> cancel guardrail) | "clarify" (>2 for 'both',
    # or nothing cancellable: the node spoke + ends).
    route_after_resolve: Callable[[ReasoningState], str]
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
    recent_orders: RecentOrderContext,
    payment_instruments: PaymentInstrumentDirectory,
    *,
    identity_store: CallerIdentityStore,
    display_name: str,
) -> SupportNodes:
    """Build Support around session-bound stores, providers, and policy.

    `identity_store` is the same instance used by tools and Identity. `payment_instruments`
    supplies only the authorized account's masked alternative refund destination; Support never
    receives contact-matching data or accepts a customer ref from the model.
    """

    @tool
    def propose_refund(order_key: str, amount_usd: float, destination: str) -> str:
        """Propose a refund: which order (option number), how much, and where it goes."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def propose_cancel(
        order_keys: list[str] | None = None,
        scope: CancelScope | None = None,
    ) -> str:
        """Propose cancelling one or MORE orders the caller no longer wants.
        order_keys: the option number(s) - list EVERY order the caller named in ONE call
        (e.g. "cancel both" -> both option numbers). Do NOT cancel them one at a time across
        turns. For ONE order, pass a single-element list.
        scope: use "all_cancellable" (or "both_cancellable") ONLY when the caller says "cancel
        all/both my orders" without naming option numbers and you have no candidates to list."""
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
    def request_support_clarification(detail: SupportQuestionDetail) -> str:
        """Ask one platform-authored Support question for the selected missing detail."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_support() -> str:
        """Leave the support flow (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    support_tools = (
        propose_refund,
        propose_cancel,
        propose_return,
        propose_profile_change,
        request_support_clarification,
        leave_support,
    )
    bound_tool_names = frozenset(tool.name for tool in support_tools)
    model = reasoning_model.bind_tools(support_tools)

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
            "pending_request": None,
            "pending_clarification": None,
        }

    def continuation_node(state: ReasoningState) -> dict[str, object]:
        """Consume one typed post-identity request without replaying prior transcript.

        Every order reference is resolved against the current principal's live candidate set
        and authorized again. The request is cleared in the same checkpoint that mints any
        action-specific pending state, so no continuation can execute twice.
        """
        request = state.pending_request
        assert request is not None
        base: dict[str, object] = {
            "pending_request": None,
            "active_flow": "support",
        }
        full = order_store.actionable_orders()
        by_id = {order.order_id: order for order in full}
        authorized = [
            order
            for order in full
            if order_read_allowed(order.order_id, store=order_store, identity=identity_store)
        ]
        by_key = {order.key: order for order in authorized}

        def resolve(ref: str) -> OrderCandidate | None:
            return by_key.get(ref) or by_id.get(ref.strip().upper())

        def resolve_and_authorize(ref: str) -> OrderCandidate | None:
            chosen = resolve(ref)
            verdict = _authorize_target(
                chosen.order_id if chosen else ref,
                order_known=chosen is not None,
            )
            return chosen if verdict is None else None

        if isinstance(request, ListOrders) and request.scope == "account":
            bound = identity_store.current()
            if bound is None:
                return {
                    **base,
                    "active_flow": None,
                    "messages": [AIMessage("I couldn't confirm your account.")],
                }
            orders = order_store.owned_orders(bound.customer_ref)
            if orders:
                recent_orders.record([order.order_id for order in orders], operation="list")
            else:
                recent_orders.clear()
            return {
                **base,
                "active_flow": None,
                "messages": [AIMessage(render_order_list_line(orders))],
            }

        if isinstance(request, CancelOrders):
            if isinstance(request.target, CancellableOrderScope):
                return {**base, "pending_cancel": request.target}
            if isinstance(request.target, ExplicitOrderSet):
                resolved: list[OrderCandidate] = []
                seen: set[str] = set()
                for ref in request.target.order_refs:
                    chosen = resolve_and_authorize(ref)
                    if chosen is None:
                        return {
                            **base,
                            "active_flow": None,
                            "messages": [AIMessage(_CONTINUATION_NOT_FOUND)],
                        }
                    if chosen.order_id not in seen:
                        seen.add(chosen.order_id)
                        resolved.append(chosen)
                return {
                    **base,
                    "pending_cancel": _mint_cancel_batch(
                        [(order.order_id, order.summary) for order in resolved]
                    ),
                }

        if isinstance(request, RefundOrder) and isinstance(request.target, ExplicitOrderTarget):
            chosen = resolve_and_authorize(request.target.order_ref)
            if chosen is not None and request.amount_usd is None:
                return {
                    **base,
                    "messages": [AIMessage("What amount would you like refunded?")],
                }
            if chosen is not None and request.destination is None:
                return {
                    **base,
                    "messages": [
                        AIMessage("Should that refund go back to the original payment method?")
                    ],
                }
            if chosen is not None:
                assert request.amount_usd is not None and request.destination is not None
                pending = _mint_refund(
                    chosen,
                    amount_usd=request.amount_usd,
                    destination=request.destination,
                )
                if pending is None:
                    write_event(
                        {"event": "refund_destination_unavailable", "reason": request.destination}
                    )
                    return {
                        **base,
                        "active_flow": None,
                        "handover": HandoffRequest(
                            destination="human",
                            reason_code="refund",
                            source="gate",
                        ),
                    }
                return {
                    **base,
                    "pending_refund": pending,
                }

        if isinstance(request, ReturnOrder) and isinstance(request.target, ExplicitOrderTarget):
            chosen = resolve_and_authorize(request.target.order_ref)
            if chosen is not None:
                return {**base, "pending_return": _mint_return(chosen)}

        if isinstance(request, ChangeProfile) and request.new_value is not None:
            pending = _mint_profile_change(request)
            if pending is not None:
                return {**base, "pending_profile_change": pending}
            return {
                **base,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code=(
                        "address_change" if request.field == "address" else "contact_change"
                    ),
                    source="gate",
                ),
            }

        write_event(
            {
                "event": "identity_continuation_declined",
                "capability": request.kind,
                "reason": "unsupported_or_unresolved",
            }
        )
        return {
            **base,
            "active_flow": None,
            "messages": [AIMessage(_CONTINUATION_NOT_FOUND)],
        }

    def _enter_identity_for_action(new_messages: list, request: IntentRequest) -> dict[str, object]:
        """Retain one caller-stated typed request across the identity detour.

        The request contains no resolved target or authority. A fresh context resolves and
        authorizes it again; transcript replay and graph-node resume pointers are forbidden.
        """
        write_event({"event": "support_action_needs_identity"})
        return {
            "messages": new_messages,
            "active_flow": "identity",
            "pending_request": request,
            "pending_clarification": None,
            "identity_claim_misses": 0,
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

    # Failed-match count per order, this session only (nodes are built per session). NOT a
    # throttle — it never blocks an attempt; the 2nd+ denial switches the corrective line to
    # one that offers a person (the posture's "verification can't complete -> escalate").
    _auth_denials: dict[str, int] = {}

    def _authorize_target(
        order_ref: str, *, order_known: bool
    ) -> SupportAuthorizationDetail | Literal["<needs-identity>"] | None:
        """The support-selection AUTHORIZATION DECISION (SECURITY §7d): may THIS caller act on
        THAT order? A mutation (cancel/refund/return) requires RUNG-2 — the target is
        session-placed OR owned by the OTP-bound identity (`order_mutation_allowed`). A rung-1
        contact-match grant authorizes READS only; the account contact is guessable, so it must
        never let one caller mutate another's order. Three closed outcomes (no ToolMessage
        appended here, so a batch from ONE propose call aggregates every target into ONE
        ToolMessage — several against one id would malform history, F-4):

          - None                 — authorized (rung-2 holds).
          - _NEEDS_IDENTITY      — the caller is UNBOUND: route into the identity OTP flow to
                                   bind, then resume the action (support/flow.py detour). No
                                   order authority is spoken or granted; existence is never
                                   revealed (the caller simply verifies themselves).
          - a clarification key — the caller is BOUND but the order isn't theirs / doesn't
                                  resolve: the ONE combined not-found (existence-oracle; never
                                  which detail failed), with store-contact guidance after the
                                  policy denial threshold.

        `order_known` no longer drives a grant (the contact claim is not a mutation credential —
        it is not even read here). Unbound needs-identity telemetry may record whether resolution
        succeeded but never raw unresolved text; bound denials expose only the policy attempt.
        """
        if order_known and order_mutation_allowed(
            order_ref, store=order_store, identity=identity_store
        ):
            write_event(
                {"event": "support_action_authorized", "order_id": order_ref.strip().upper()}
            )
            return None
        known_fields: dict[str, object] = {"order_id_known": order_known}
        if order_known:
            known_fields["order_id"] = order_ref.strip().upper()
        # Unbound: no rung-2 authority is possible this turn — bind first (OTP), never grant.
        # This is the guest path; it reveals nothing about the order (an unresolved and a real
        # but unowned reference are indistinguishable to the caller — both just "verify you").
        if identity_store.current() is None:
            write_event({"event": "support_auth_needs_identity", **known_fields})
            return _NEEDS_IDENTITY
        # Bound, but the order is not this identity's (or doesn't resolve): fail closed with the
        # combined not-found (existence-oracle) — never reveal that another account owns it.
        counter_key = order_ref.strip().upper() if order_known else _UNRESOLVED_ORDER
        attempt = _auth_denials[counter_key] = _auth_denials.get(counter_key, 0) + 1
        write_event({"event": "support_auth_denied", "attempt": attempt})
        return (
            "order_match_human_help"
            if attempt >= policy.auth_denials_before_human_offer
            else "order_match"
        )

    def _authorize_action(
        order_ref: str, call_id: str, new_messages: list, *, order_known: bool
    ) -> SupportAuthorizationDetail | Literal["<needs-identity>"] | None:
        """The single-target authorization wrapper (refund/return paths): runs the decision and,
        on a bound-mismatch, pairs the propose call with a constant acknowledgement. Returns
        None if authorized, `_NEEDS_IDENTITY` if the caller must verify first, or the typed
        platform-owned clarification outcome."""
        verdict = _authorize_target(order_ref, order_known=order_known)
        if verdict is None or verdict == _NEEDS_IDENTITY:
            return verdict
        new_messages.append(ToolMessage(_ORDER_SELECTION_ACK, tool_call_id=call_id))
        return verdict

    def _mint_refund(
        chosen: OrderCandidate, *, amount_usd: float, destination: RefundDestination
    ) -> PendingRefund | None:
        if destination == "original":
            instrument_ref = "original payment method"
        elif destination == "new_instrument":
            customer_ref = order_store.order_owner(chosen.order_id)
            if customer_ref is None:
                bound = identity_store.current()
                customer_ref = bound.customer_ref if bound is not None else None
            instrument_ref = (
                payment_instruments.new_instrument_ref(customer_ref)
                if customer_ref is not None
                else None
            )
            if instrument_ref is None:
                return None
        else:
            # There is no typed payout-address reference in this build.
            return None
        return PendingRefund(
            order_id=chosen.order_id,
            summary=chosen.summary,
            amount_usd=round(amount_usd, 2),
            destination=destination,
            instrument_ref=instrument_ref,
            idempotency_key=uuid.uuid4().hex,
            attempt_key=uuid.uuid4().hex,
            created_at=time.time(),
        )

    def _mint_return(chosen: OrderCandidate) -> PendingReturn:
        refund_due = max(
            0.0,
            round(
                chosen.total_usd
                - order_store.refunded_so_far(chosen.order_id)
                - order_store.return_refund_due(chosen.order_id),
                2,
            ),
        )
        return PendingReturn(
            order_id=chosen.order_id,
            summary=chosen.summary,
            refund_due_usd=refund_due,
            idempotency_key=uuid.uuid4().hex,
            created_at=time.time(),
        )

    def _mint_profile_change(change: ChangeProfile) -> PendingProfileChange | None:
        bound = identity_store.current()
        if bound is None or not profile_store.has_profile(bound.customer_ref):
            return None
        assert change.new_value is not None
        return PendingProfileChange(
            customer_ref=bound.customer_ref,
            field=change.field,
            new_value=change.new_value,
            factor_ref=profile_store.contact_on_file(bound.customer_ref),
            idempotency_key=uuid.uuid4().hex,
            attempt_key=uuid.uuid4().hex,
            created_at=time.time(),
        )

    def _clarification_result(
        state: ReasoningState,
        new_messages: list,
        detail: SupportQuestionDetail | SupportAuthorizationDetail,
    ) -> dict[str, object]:
        step = advance_clarification(
            state,
            flow="support",
            max_reasks=policy.support_clarification_reask_max,
        )
        if step.exhausted:
            return {
                "messages": new_messages,
                "active_flow": None,
                "pending_refund": None,
                "pending_cancel": None,
                "pending_return": None,
                "pending_profile_change": None,
                "pending_request": None,
                "pending_clarification": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="other",
                    source="gate",
                ),
            }
        return {
            "messages": new_messages,
            "active_flow": "support",
            "pending_clarification": SupportClarification(detail=detail),
            "clarification_progress": step.progress,
        }

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE support: propose a REFUND (order+amount+destination) or a CANCEL
        (one/many explicit orders or a semantic account scope), or clarify, or leave. Mints
        PendingRefund/PendingCancel (per-intent keys) on a valid proposal — A10a rule 2: every
        idempotency key exists in state before any effect.

        The candidate list the model SEES is scoped to AUTHORIZED orders (live call #15: the
        assemble model's clarify question recited the unscoped list — ids, items, totals — to
        an unverified caller who had just failed OTP; the model can't speak what it never
        saw, same structural stance as the buffer-before-speak fix). Keys keep their
        full-list positions, so a key stays stable across the session as grants accumulate.
        `by_id` is CODE-side only (never rendered): it resolves a caller-STATED order number
        for the guest path — the model relays the caller's id exactly as `order_status`
        already does; it still never AUTHORS one.
        """
        full = order_store.actionable_orders()
        by_id = {o.order_id: o for o in full}
        orders = [
            o
            for o in full
            if order_read_allowed(o.order_id, store=order_store, identity=identity_store)
        ]
        by_key = {o.key: o for o in orders}

        def _resolve(stated: str) -> OrderCandidate | None:
            return by_key.get(stated) or by_id.get(stated.strip().upper())

        prompt = SystemMessage(
            compose_support_prompt(
                display_name, orders, policy, recent_orders.snapshot().focused_order_ref
            )
        )
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal gets ONE corrective re-prompt
            response = model.invoke(messages)
            if not response.tool_calls:
                return _clarification_result(state, new_messages, "action")
            new_messages.append(response)
            ack_extra_tool_calls(response, new_messages)
            call = response.tool_calls[0]
            if call["name"] == "leave_support":
                return _leave(new_messages, call["id"])

            if call["name"] == "request_support_clarification":
                try:
                    clarification = _RequestSupportClarification.model_validate(call["args"])
                except ValueError:
                    clarification = None
                if clarification is None:
                    new_messages.append(
                        ToolMessage("Invalid clarification request.", tool_call_id=call["id"])
                    )
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                new_messages.append(
                    ToolMessage("support clarification requested", tool_call_id=call["id"])
                )
                return _clarification_result(state, new_messages, clarification.detail)

            if call["name"] == "propose_cancel":
                try:
                    cancel = _ProposeCancel.model_validate(call["args"])
                except ValueError:
                    cancel = None
                if cancel is None:
                    feedback = (
                        "Invalid cancellation selection. Use either a non-empty order_keys "
                        "list or one supported scope, never both. Do not guess order numbers."
                    )
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                # A bare account-wide `scope` (no keys): "cancel all/both my orders" with no
                # option numbers (Milestone B). A BOUND caller resolves immediately (the
                # resolve node reads their cancellable orders); an UNBOUND caller must verify
                # first — retain a typed semantic scope with no ids across the identity detour.
                if not cancel.order_keys:
                    assert cancel.scope is not None  # enforced by _ProposeCancel
                    new_messages.append(
                        ToolMessage(f"proposed cancel of {cancel.scope}", tool_call_id=call["id"])
                    )
                    selection = CancellableOrderScope(scope=cancel.scope)
                    if identity_store.current() is not None:
                        # Already verified: resolve now (route_after_assemble -> resolve).
                        return {"messages": new_messages, "pending_cancel": selection}
                    # Unverified: the shared typed request is the sole continuation channel.
                    return _enter_identity_for_action(new_messages, CancelOrders(target=selection))
                # Resolve + AUTHORIZE every stated key, aggregating the outcome into EXACTLY ONE
                # ToolMessage for this single tool_call_id (never one per target — F-4). An
                # unresolved key is NOT an "invalid proposal" (that would enumerate keys): it is
                # a guest reference the authorization gate handles with order_known=False (the
                # existence-oracle-safe corrective). ANY denial fails the whole proposal closed
                # (no partial batch reaches a readback unauthorized). Dedup by order_id so
                # "cancel both" listing one order twice mints a single target.
                # Resolve + deduplicate BEFORE authorization so aliases (an option key and
                # its ORD-id) cannot consume multiple denial attempts or create repeated
                # authorization work for one object.
                unique: list[tuple[str, OrderCandidate | None]] = []
                seen_refs: set[str] = set()
                for raw_key in cancel.order_keys:
                    hit = _resolve(raw_key)
                    ref = hit.order_id if hit is not None else raw_key.strip().upper()
                    if ref not in seen_refs:
                        seen_refs.add(ref)
                        unique.append((ref, hit))

                resolved: list[OrderCandidate] = []
                verdict: SupportAuthorizationDetail | Literal["<needs-identity>"] | None = None
                for ref, hit in unique:
                    verdict = _authorize_target(
                        hit.order_id if hit else ref, order_known=hit is not None
                    )
                    if verdict is not None:
                        break  # identity detour or a code-authored denial clarification
                    assert hit is not None  # unresolved never authorizes (no owner, no grant)
                    resolved.append(hit)
                if verdict == _NEEDS_IDENTITY:
                    # Unbound caller: bind first, then resume this cancel (no pending minted).
                    stated_refs = tuple(
                        caller_stated_order_id(_last_user_text(state), ref) for ref, _hit in unique
                    )
                    if any(ref is None for ref in stated_refs):
                        new_messages.append(
                            ToolMessage(
                                _ORDER_REFERENCE_REQUIRED_ACK,
                                tool_call_id=call["id"],
                            )
                        )
                        return _clarification_result(state, new_messages, "order")
                    validated_refs = tuple(ref for ref in stated_refs if ref is not None)
                    new_messages.append(ToolMessage("verification needed", tool_call_id=call["id"]))
                    return _enter_identity_for_action(
                        new_messages,
                        CancelOrders(target=ExplicitOrderSet(order_refs=validated_refs)),
                    )
                if verdict is not None:
                    new_messages.append(ToolMessage(_ORDER_SELECTION_ACK, tool_call_id=call["id"]))
                    return _clarification_result(state, new_messages, verdict)
                new_messages.append(
                    ToolMessage(
                        f"proposed cancel on {len(resolved)} order(s)", tool_call_id=call["id"]
                    )
                )
                pending_batch = _mint_cancel_batch([(c.order_id, c.summary) for c in resolved])
                return {"messages": new_messages, "pending_cancel": pending_batch}

            if call["name"] == "propose_return":
                try:
                    proposed = _ProposeReturn.model_validate(call["args"])
                except ValueError:
                    proposed = None
                chosen = _resolve(proposed.order_key) if proposed else None
                if proposed is None:
                    feedback = f"Invalid order. Valid order numbers: {', '.join(sorted(by_key))}."
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                verdict = _authorize_action(
                    chosen.order_id if chosen else proposed.order_key,
                    call["id"],
                    new_messages,
                    order_known=chosen is not None,
                )
                if verdict == _NEEDS_IDENTITY:
                    stated_ref = caller_stated_order_id(
                        _last_user_text(state),
                        chosen.order_id if chosen is not None else proposed.order_key,
                    )
                    if stated_ref is None:
                        new_messages.append(
                            ToolMessage(
                                _ORDER_REFERENCE_REQUIRED_ACK,
                                tool_call_id=call["id"],
                            )
                        )
                        return _clarification_result(state, new_messages, "order")
                    new_messages.append(ToolMessage("verification needed", tool_call_id=call["id"]))
                    return _enter_identity_for_action(
                        new_messages,
                        ReturnOrder(target=ExplicitOrderTarget(order_ref=stated_ref)),
                    )
                if verdict is not None:
                    return _clarification_result(state, new_messages, verdict)
                assert chosen is not None  # unresolved never authorizes (no owner, no grant)
                new_messages.append(
                    ToolMessage(f"proposed return on order {chosen.key}", tool_call_id=call["id"])
                )
                pending_return = _mint_return(chosen)
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
                if not _caller_stated_profile_value(
                    _last_user_text(state), change.field, change.new_value
                ):
                    new_messages.append(
                        ToolMessage(
                            _PROFILE_VALUE_REQUIRED_ACK,
                            tool_call_id=call["id"],
                        )
                    )
                    return _clarification_result(state, new_messages, "profile_value")
                # A profile change is ACCOUNT-scoped, so it requires a BOUND identity (Fix 5
                # Milestone B): the customer whose profile is touched comes from the LIVE binding,
                # never a model argument. An UNBOUND caller detours into the identity OTP flow to
                # bind first, then re-proposes (reusing the mutation detour). A BOUND caller whose
                # customer has no profile on file fails CLOSED — one neutral human-handover
                # line, never falling back to (or revealing) another customer's profile.
                bound = identity_store.current()
                if bound is None:
                    new_messages.append(ToolMessage("verification needed", tool_call_id=call["id"]))
                    return _enter_identity_for_action(
                        new_messages,
                        ChangeProfile(field=change.field, new_value=change.new_value.strip()),
                    )
                if not profile_store.has_profile(bound.customer_ref):
                    write_event({"event": "profile_change_denied", "reason": "no_profile"})
                    new_messages.append(
                        ToolMessage("profile update unavailable", tool_call_id=call["id"])
                    )
                    return {
                        "messages": new_messages,
                        "pending_profile_change": None,
                        "active_flow": None,
                        "handover": HandoffRequest(
                            destination="human",
                            reason_code=(
                                "address_change" if change.field == "address" else "contact_change"
                            ),
                            source="gate",
                        ),
                    }
                # NO value echo in the tool result (PII: thread history is model-visible
                # context, but the persisted line must not carry the raw value a second
                # time beyond the pending itself).
                new_messages.append(
                    ToolMessage(f"proposed profile change: {change.field}", tool_call_id=call["id"])
                )
                pending_change = _mint_profile_change(
                    ChangeProfile(field=change.field, new_value=change.new_value.strip())
                )
                assert pending_change is not None
                return {"messages": new_messages, "pending_profile_change": pending_change}

            if call["name"] == "propose_refund":
                try:
                    proposal = _ProposeRefund.model_validate(call["args"])
                except ValueError:
                    proposal = None
                chosen = _resolve(proposal.order_key) if proposal else None
                if proposal is None or proposal.amount_usd <= 0:
                    feedback = (
                        f"Invalid proposal. Valid order numbers: {', '.join(sorted(by_key))}; "
                        "amount must be > 0."
                    )
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                verdict = _authorize_action(
                    chosen.order_id if chosen else proposal.order_key,
                    call["id"],
                    new_messages,
                    order_known=chosen is not None,
                )
                if verdict == _NEEDS_IDENTITY:
                    stated_ref = caller_stated_order_id(
                        _last_user_text(state),
                        chosen.order_id if chosen is not None else proposal.order_key,
                    )
                    if stated_ref is None:
                        new_messages.append(
                            ToolMessage(
                                _ORDER_REFERENCE_REQUIRED_ACK,
                                tool_call_id=call["id"],
                            )
                        )
                        return _clarification_result(state, new_messages, "order")
                    new_messages.append(ToolMessage("verification needed", tool_call_id=call["id"]))
                    return _enter_identity_for_action(
                        new_messages,
                        RefundOrder(
                            target=ExplicitOrderTarget(order_ref=stated_ref),
                            amount_usd=_caller_stated_refund_amount(
                                _last_user_text(state), proposal.amount_usd
                            ),
                            destination=_caller_stated_refund_destination(
                                _last_user_text(state), proposal.destination
                            ),
                        ),
                    )
                if verdict is not None:
                    return _clarification_result(state, new_messages, verdict)
                assert chosen is not None  # unresolved never authorizes (no owner, no grant)
                new_messages.append(
                    ToolMessage(
                        f"proposed refund on order {chosen.key} ${proposal.amount_usd:.2f} "
                        f"to {proposal.destination}",
                        tool_call_id=call["id"],
                    )
                )
                pending = _mint_refund(
                    chosen,
                    amount_usd=proposal.amount_usd,
                    destination=proposal.destination,
                )
                if pending is None:
                    write_event(
                        {"event": "refund_destination_unavailable", "reason": proposal.destination}
                    )
                    return {
                        "messages": new_messages,
                        "pending_refund": None,
                        "active_flow": None,
                        "handover": HandoffRequest(
                            destination="human",
                            reason_code="refund",
                            source="gate",
                        ),
                    }
                return {"messages": new_messages, "pending_refund": pending}

            if call["name"] not in bound_tool_names:
                new_messages.append(unknown_tool_result(call["id"], leave_tool=leave_support.name))
                messages = [prompt, *state.messages, *new_messages]
                continue
            # A bound name reaching here is code drift: preserve the loud developer failure.
            raise ValueError(f"support assemble: bound tool has no handler: {call['name']!r}")
        logger.warning("support assemble: two invalid tool calls; asking for action in code")
        return _clarification_result(state, new_messages, "action")

    def clarify_node(state: ReasoningState) -> dict[str, object]:
        clarification = state.pending_clarification
        if not isinstance(clarification, SupportClarification):
            raise TypeError("support clarify node requires SupportClarification")
        return {
            "pending_clarification": None,
            "active_flow": "support",
            "messages": [AIMessage(_SUPPORT_CLARIFICATION_LINES[clarification.detail])],
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
                # Remedy steer mints through the ONE batch helper (a one-target batch).
                "pending_cancel": _mint_cancel_batch(
                    [(pending.order_id, order_store.order_item_summary(pending.order_id))]
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
                    summary=order_store.order_item_summary(pending.order_id) or pending.order_id,
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
        max_otp_attempts=policy.otp_max_attempts,
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
        phrase = _refund_confirmation_phrase(pending, ISSUE_REFUND_POLICY)
        answer = interrupt(f"Just to confirm: {phrase}. Shall I go ahead?")
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(f"Sorry - just to be clear: {phrase}. Yes or no?")
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
                instrument_ref=pending.instrument_ref,
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
        recent_orders.record([record.order_id], operation="refund")
        write_event(
            {
                "event": "refund_confirmed",
                "refund_id": record.refund_id,
                "amount": record.amount_usd,
                "verification": [grant.method for grant in verification_store.grants],
            }
        )
        return {
            "pending_refund": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - your ${record.amount_usd:.2f} refund is on its way to your "
                    f"{record.instrument_ref}. Your reference is {record.refund_id}."
                )
            ],
        }

    # --- cancel sub-path (F-16.2 batch; single = batch-of-one: guardrail -> confirm -> void)

    def _preflight_target(target: CancelTarget) -> BatchCancelOutcome | None:
        """Per-target eligibility, the SAME ladder as the pre-batch single guardrail (LIVE
        store reads, never the model). Returns an ineligible outcome (closed code) or None if
        the target is cancellable. Risk is NOT here — it is a session-level signal handled
        once by the guardrail (a per-order loop can't re-decide a session flag)."""
        status = order_store.order_status(target.order_id)
        if status == CANCELLED_STATUS:
            return BatchCancelOutcome(
                order_id=target.order_id, summary=target.summary, outcome="already_cancelled"
            )
        if not order_store.is_cancellable(target.order_id):
            return BatchCancelOutcome(
                order_id=target.order_id, summary=target.summary, outcome="not_cancellable"
            )
        if order_store.refunded_so_far(target.order_id) > 0:
            return BatchCancelOutcome(
                order_id=target.order_id, summary=target.summary, outcome="has_refunds"
            )
        return None

    def cancel_guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced batch eligibility (never the model). Partitions the targets into the
        ELIGIBLE subset (survives to the readback) and the INELIGIBLE ones (stated at the
        readback so the caller hears the whole truth), each via `_preflight_target`'s live
        reads. Cancel is L1 (no step-up). Special cases:
          - RISK-flagged session: no silent void (§A4a) — clear + hand to a person (no spoken
            line; the handover deferral is the single voice). Batch-level, checked once.
          - ZERO eligible: author the honest all-ineligible line + END (no readback).
          - OVER the batch cap: a big batch can't fit the step budget — clear + ask the caller
            to narrow (never silently take the first N).
        The pending is REWRITTEN to carry only the eligible `targets` + the `ineligible`
        outcomes; on an eligible subset it takes no other side effect (router -> confirm)."""
        pending = state.pending_cancel
        assert isinstance(pending, PendingCancelBatch)
        if risk.check_sim_swap():
            write_event({"event": "cancel_stepup_to_human", "reason": "risk_flagged"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        if len(pending.targets) > policy.cancel_batch_max:
            write_event({"event": "cancel_batch_over_cap", "count": len(pending.targets)})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"That's more orders than I can cancel in one go - I can do up to "
                        f"{policy.cancel_batch_max} at a time. Which ones would you like me to "
                        "start with?"
                    )
                ],
            }
        eligible: list[CancelTarget] = []
        ineligible: list[BatchCancelOutcome] = []
        for target in pending.targets:
            verdict = _preflight_target(target)
            if verdict is None:
                eligible.append(target)
            else:
                ineligible.append(verdict)
        if not eligible:
            write_event(
                {
                    "event": "cancel_declined",
                    "reason": "batch_none_eligible",
                    "count": len(ineligible),
                }
            )
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [AIMessage(render_batch_cancel_outcome(ineligible))],
            }
        return {
            "pending_cancel": pending.model_copy(
                update={"targets": tuple(eligible), "ineligible": tuple(ineligible)}
            )
        }

    def cancel_confirm_node(state: ReasoningState) -> dict[str, object]:
        """HITL interrupt: the batch cancel readback + deterministic consent. NO side effects;
        §4a barge -> re-confirm once (a truncated 'yes' does not authorize an irreversible
        void). Clock-A TTL FIRST (clear-before-speak) — a stale 'yes' can't void."""
        pending = state.pending_cancel
        assert isinstance(pending, PendingCancelBatch)
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
        # Cancel-polarity consent: the question IS "shall I cancel?", so "yeah cancel them"
        # must read as yes (plain classify_consent reads the 'cancel' as a no).
        answer = interrupt(_cancel_readback_line(pending))
        verdict = classify_cancel_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            phrases = "; ".join(_cancel_target_phrase(t) for t in pending.targets)
            retry = interrupt(f"Sorry - just to be clear: cancel {phrases}? Yes or no?")
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
            decline_line = (
                "Okay, I'll leave that order as it is - nothing changed."
                if len(pending.targets) == 1
                else "Okay, I'll leave those orders as they are - nothing changed."
            )
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [AIMessage(decline_line)],
            }
        return {}  # yes: pending survives; router -> void

    def cancel_void_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Voids ONE eligible target
        per node completion, recording its outcome, then returns — the router loops back here
        while targets remain, so progress is CHECKPOINTED BETWEEN writes (a kill after target
        N leaves N cancelled, the rest pending; replay with the same per-target key returns the
        SAME CancelRecord — exactly N effects, never N+1). Each spoken clause comes from a
        CancelRecord or the preflight outcome (INV-25) — NO confirming re-read. An effect-time
        CancelError (stale since preflight) becomes a generic `store_refused` outcome; one
        stale target never sinks the batch. When the last target is done, renders the whole
        per-target result and ends."""
        pending = state.pending_cancel
        assert isinstance(pending, PendingCancelBatch)
        done = len(pending.outcomes)
        target = pending.targets[done]
        try:
            record = order_store.cancel_order(target.idempotency_key, order_id=target.order_id)
            outcome = BatchCancelOutcome(
                order_id=record.order_id,
                summary=record.summary,
                outcome="cancelled",
                amount_usd=record.total_usd,
            )
            write_event({"event": "cancel_confirmed", "order_id": record.order_id})
        except CancelError as exc:
            # No typed reason on CancelError, and parsing its text is forbidden — a rare
            # effect-time (stale-since-preflight) refusal is the honest generic store_refused.
            logger.warning("cancel refused by store: %s", exc)
            write_event(
                {"event": "cancel_denied", "reason": "store_refused", "order_id": target.order_id}
            )
            outcome = BatchCancelOutcome(
                order_id=target.order_id, summary=target.summary, outcome="store_refused"
            )
        outcomes = (*pending.outcomes, outcome)
        if done + 1 < len(pending.targets):
            # More targets remain: checkpoint progress, loop back (router -> void).
            return {"pending_cancel": pending.model_copy(update={"outcomes": outcomes})}
        # Last target done: speak the whole truth from the collected records (+ any ineligible
        # ones stated at the readback) and end.
        all_outcomes = [*pending.ineligible, *outcomes]
        recent_orders.record(
            [outcome.order_id for outcome in all_outcomes],
            operation="cancel",
            focused_order_ref=all_outcomes[-1].order_id,
            outcomes=[(outcome.order_id, outcome.outcome) for outcome in all_outcomes],
        )
        return {
            "pending_cancel": None,
            "active_flow": None,
            "messages": [AIMessage(render_batch_cancel_outcome(all_outcomes))],
        }

    def resolve_node(state: ReasoningState) -> dict[str, object]:
        """Resolve a retained CancellableOrderScope after the account is bound (Milestone
        B — the "cancel all my orders" continuation). This is TRANSACTION AUTHORIZATION, not
        authentication: it reads the LIVE binding and the caller's CURRENT cancellable orders
        (never a set computed before the OTP, which could have gone stale). No contact claim,
        no grant — the bind already granted account-wide access, so `owned_cancellable_orders`
        is the authorized universe. Freezes the exact PendingCancelBatch and routes into the
        SAME cancel guardrail/readback/void the explicit path uses. Speaks NO order list.

        Terminal (spoke + ends) cases: not actually bound (belt-and-suspenders — a resolver
        reached unbound clears + defers), nothing cancellable, or a 'both' scope that resolves
        to more than two candidates (ambiguous — ask which)."""
        selection = state.pending_cancel
        assert isinstance(selection, CancellableOrderScope)
        bound = identity_store.current()
        if bound is None:
            # Unreachable in the happy path (resolve is only routed to post-bind); fail closed.
            write_event({"event": "cancel_resolve_declined", "reason": "not_bound"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage("I couldn't confirm your account, so I haven't changed anything.")
                ],
            }
        candidates, has_more = order_store.owned_cancellable_orders(
            bound.customer_ref, limit=policy.cancel_batch_max + 1
        )
        unauthorized = [
            candidate.order_id
            for candidate in candidates
            if not order_read_allowed(
                candidate.order_id, store=order_store, identity=identity_store
            )
        ]
        if unauthorized:
            # The query narrows candidates; it is not an authorization authority. Reject
            # the whole semantic scope when even one result falls outside the live binding.
            write_event(
                {
                    "event": "cancel_resolve_declined",
                    "reason": "authorization_mismatch",
                    "count": len(unauthorized),
                }
            )
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "I couldn't safely match those orders to your account, so I haven't "
                        "changed anything."
                    )
                ],
            }
        if not candidates:
            write_event({"event": "cancel_resolve_declined", "reason": "none_cancellable"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage("None of your orders can be cancelled right now - nothing to do.")
                ],
            }
        if selection.scope == "both_cancellable" and len(candidates) != 2:
            # "both" is only unambiguous with exactly two cancellable orders. Keep the
            # response truthful on either side of two.
            write_event(
                {
                    "event": "cancel_resolve_declined",
                    "reason": "both_ambiguous",
                    "count": len(candidates),
                }
            )
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "I found only one order that can be cancelled, so I need you to tell "
                        "me which orders you meant."
                        if len(candidates) == 1
                        else "You've got more than two orders that can be cancelled - which "
                        "ones would you like me to cancel?"
                    )
                ],
            }
        if len(candidates) > policy.cancel_batch_max or has_more:
            # Over the batch cap: ask to narrow, never silently take the first N (mirrors the
            # guardrail's over-cap line; here the caller never named specifics).
            write_event({"event": "cancel_resolve_over_cap"})
            return {
                "pending_cancel": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"You've got more orders than I can cancel in one go - I can do up to "
                        f"{policy.cancel_batch_max} at a time. Which would you like to start with?"
                    )
                ],
            }
        batch = _mint_cancel_batch([(c.order_id, c.summary) for c in candidates])
        write_event(
            {
                "event": "cancel_resolved_from_scope",
                "scope": selection.scope,
                "count": len(candidates),
            }
        )
        return {"pending_cancel": batch, "active_flow": "support"}

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
                # Remedy steer mints through the ONE batch helper (a one-target batch).
                "pending_cancel": _mint_cancel_batch([(pending.order_id, pending.summary)]),
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
        phrase = _return_confirmation_phrase(pending, CREATE_RETURN_POLICY)
        answer = interrupt(f"Just to confirm - {phrase}. Shall I go ahead?")
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(f"Sorry - just to be clear: {phrase}. Yes or no?")
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
                "messages": [AIMessage("Okay, I won't set up a return - nothing has changed.")],
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
        recent_orders.record([record.order_id], operation="return")
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
                    f"Done - your return for {pending.summary} is set up; your reference is "
                    f"{record.rma_id}. Send the items back, and once we receive them the "
                    f"${record.refund_due_usd:.2f} refund goes to your original payment method."
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
        max_otp_attempts=policy.otp_max_attempts,
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
        phrase = _profile_confirmation_phrase(pending)
        answer = interrupt(f"Just to confirm - {phrase}. Shall I go ahead?")
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(f"Sorry - just to be clear: {phrase}? Yes or no?")
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
        # Re-validate BOTH legs live at effect time (§A4c): the level must still hold, AND the
        # session must still be bound to the SAME customer the change was proposed for (Fix 5
        # Milestone B — a binding that lapsed or switched between propose and here must not apply
        # a change to the wrong / an unauthorized account).
        bound = identity_store.current()
        binding_holds = bound is not None and bound.customer_ref == pending.customer_ref
        if verification_store.current_level() < required or not binding_holds:
            reason = "level_lapsed_at_place" if binding_holds else "binding_lapsed_at_place"
            write_event({"event": "profile_stepup_failed", "reason": reason})
            return {
                "pending_profile_change": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
            }
        try:
            record = profile_store.update_profile(
                pending.idempotency_key,
                customer_ref=pending.customer_ref,
                field=pending.field,
                new_value=pending.new_value,
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
                "verification": [grant.method for grant in verification_store.grants],
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
            "pending_request": None,
            "pending_clarification": None,
            "clarification_progress": None,
            "active_flow": None,
            "messages": [
                AIMessage("No problem - I've dropped that request. Your orders are unchanged.")
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
            "pending_request": None,
            "pending_clarification": None,
            "clarification_progress": None,
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
        if state.handover is not None:
            return "handover"
        if state.active_flow == "left_support":
            return "leave"  # model explicitly left; normal pipeline answers this turn
        if state.active_flow == "identity":
            return "needs_identity"  # unbound "cancel all" — verify first, then resolve
        if isinstance(state.pending_cancel, CancellableOrderScope):
            return "resolve"  # a BOUND caller's scope resolves immediately
        if state.pending_cancel is not None:  # a PendingCancelBatch
            return "cancel"
        if state.pending_return is not None:
            return "return"
        if state.pending_profile_change is not None:
            return "profile"
        if state.pending_refund is not None:
            return "refund"
        if state.pending_clarification is not None:
            if not isinstance(state.pending_clarification, SupportClarification):
                raise TypeError("support assemble produced a non-support clarification")
            return "clarify"
        return "clarify"  # support_continuation may have authored its own terminal line

    def route_after_resolve(state: ReasoningState) -> str:
        # "confirm" (a batch was frozen -> the cancel guardrail/readback/void) | "clarify"
        # (nothing cancellable / >2 for 'both' / over-cap: the node spoke its line + ends).
        return "confirm" if isinstance(state.pending_cancel, PendingCancelBatch) else "clarify"

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
        assemble=with_clarification_lifecycle(assemble_node),
        continuation=continuation_node,
        clarify=clarify_node,
        guardrail=guardrail_node,
        risk_check=refund_stepup.risk_check,
        dispatch=refund_stepup.dispatch,
        collect=refund_stepup.collect,
        confirm=confirm_node,
        place=place_node,
        cancel_guardrail=cancel_guardrail_node,
        cancel_confirm=cancel_confirm_node,
        cancel_void=cancel_void_node,
        resolve=resolve_node,
        abort=abort_node,
        escape_human=escape_human_node,
        route_after_assemble=route_after_assemble,
        route_after_resolve=route_after_resolve,
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
                "support_continuation",
                "support_clarify",
                "support_guardrail",  # authors the over-amount-threshold decline line
                "support_risk_check",
                "support_collect",
                "support_confirm",
                "support_place",
                "support_cancel_guardrail",  # authors the shipped/ineligible decline line
                "support_cancel_confirm",
                "support_cancel_void",
                "support_resolve",  # authors the none-cancellable / ambiguous-scope decline
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
