"""The identity gated flow (P7) — verify the caller, bind the session, speak THEIR orders.

Entered by handover (reason_code "list_orders"): enumeration is rung 2 of the P7
authorization split — a caller may guest-look-up ONE order with the order id + contact
pair (voice/tools.py order_status), but "what orders do I have?" requires an OTP-BOUND
identity. The chain is the SAME `_stepup.py` factory the refund/profile flows run:

    assemble ─(claim code-matched, pending minted)─▶ guardrail
        │ no match, within budget ─▶ reask (softened re-ask; never asserts not-on-file)
        │ no match, budget spent  ─▶ human (SILENT — the handover deferral is the voice)
        ▼
    guardrail ─(already bound to this customer)─▶ apply
        └─(unbound)─▶ risk_check ─▶ dispatch ─▶ collect[INT: OTP] ─▶ apply

THE BINDING INVARIANT (the P7 security-review catch): the factory's route_after_collect
confirms on `current_level() >= required` ALONE — correct for refund/profile (level IS
their requirement), insufficient for a BIND: a stale cross-family L2 (an earlier
profile-flow OTP) would let a WRONG identity OTP route "confirm". `VerificationStore.
verify_otp` appends a grant on every successful match, so "a NEW grant since the pending
was minted" ≡ "THIS chain's OTP succeeded": `route_after_collect` here wraps the factory
decision and downgrades a no-new-grant "confirm" back to "dispatch" (the factory's tries
counter still bounds the loop), and `apply` re-checks both conditions before binding.

Anti-enumeration posture (P7 decisions 4/5, ACCEPTED GAPS documented in
commerce/identity.py): the re-ask wording never confirms non-existence, the terminal
no-match hands to a human with no flow-authored line, and claim attempts are bounded
per-session only — the cross-session throttle is the platform rate/abuse layer's job.
A neutral "dispatch anyway" flow is NOT an option in the build phase: the stub OTP code
is global, so a doomed collect for an unmatched claim would grant L2 to an unverified
caller.

PII discipline: the raw spoken claim reaches `match_contact` and is never persisted —
the pending carries the MASKED contact only; ToolMessages are constant (no value echo);
telemetry carries closed slugs + the customer_ref fixture slug only.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from agnostic_market.agents._toolcalls import ack_extra_tool_calls
from agnostic_market.agents.identity.prompt import compose_identity_prompt
from agnostic_market.agents.support._stepup import build_stepup_nodes
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.identity import BoundIdentity, CallerIdentityStore, CustomerDirectory
from agnostic_market.commerce.orders import OrderStore, render_order_list_line
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.confirmation import identity_required_level
from agnostic_market.dtos.state import (
    CancelSelection,
    HandoffRequest,
    PendingIdentity,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.identity")

# The ONE softened re-ask (P7 decision 4): asks the caller to re-state the contact without
# asserting it is not on file — "I couldn't find that" confirms non-existence to a probing
# caller, and an STT mishear (not absence) is the likely cause anyway.
_REASK_LINE = "Could you double-check the email or phone number on the account for me?"


class _ProposeIdentity(BaseModel):
    contact_claim: str


@dataclass(frozen=True)
class IdentityNodes:
    """The identity flow's node callables + its caller-facing (speakable) node names.

    Wiring is the graph builder's job (frontline/graph.py); this module owns only behavior.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    reask: Callable[[ReasoningState], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    risk_check: Callable[[ReasoningState], dict[str, object]]
    dispatch: Callable[[ReasoningState], dict[str, object]]
    collect: Callable[[ReasoningState], dict[str, object]]
    apply: Callable[[ReasoningState], dict[str, object]]
    abort: Callable[[ReasoningState], dict[str, object]]
    escape_human: Callable[[ReasoningState], dict[str, object]]
    # "leave" | "handover" | "guardrail" | "reask" | "clarify" — from the assemble outcome.
    route_after_assemble: Callable[[ReasoningState], str]
    # "confirm" (already bound to this customer, level held) | "stepup".
    route_after_guardrail: Callable[[ReasoningState], str]
    # "confirm" | "dispatch" | "handover" — the factory decision, WRAPPED with the binding
    # invariant (a no-new-grant "confirm" re-collects instead).
    route_after_collect: Callable[[ReasoningState], str]
    speakable_nodes: frozenset[str]


def _flow_exit(update: dict[str, object]) -> dict[str, object]:
    """Every identity-flow exit clears the pending + the re-ask counter — AND any retained
    CancelSelection (Milestone B) or `identity_resume` marker (Fix 2): a "cancel all" scope OR
    an explicit-order mutation that entered identity to verify must leave ZERO action intent
    behind on ANY failure/leave/abort/human path (a live selector/marker surviving a FAILED
    verification is the security hazard). The success paths that resume support (bind -> resolve,
    bind -> support_assemble) return their own update and do NOT go through here."""
    return {
        "pending_identity": None,
        "identity_claim_misses": 0,
        "pending_cancel": None,
        "identity_resume": None,
        **update,
    }


def build_identity_nodes(
    reasoning_model: BaseChatModel,
    order_store: OrderStore,
    verification_store: VerificationStore,
    otp: OtpProvider,
    risk: RiskProvider,
    customers: CustomerDirectory,
    identity_store: CallerIdentityStore,
    policy: PolicyContext,
    *,
    display_name: str,
) -> IdentityNodes:
    """Build the identity flow's nodes, closed over the session's stores + providers +
    policy (§A5: everything bound in code at build time — never carried in state)."""

    @tool
    def propose_identity(contact_claim: str) -> str:
        """Submit the email address or phone number the caller says is on the account,
        EXACTLY as they stated it."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_identity() -> str:
        """Leave identity verification (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    model = reasoning_model.bind_tools([propose_identity, leave_identity])

    def _leave(new_messages: list, call_id: str, reason: str) -> dict[str, object]:
        new_messages.append(ToolMessage("left identity", tool_call_id=call_id))
        write_event({"event": "identity_left", "reason": reason})
        return _flow_exit({"messages": new_messages, "active_flow": "left_identity"})

    def _human_handover(update: dict[str, object]) -> dict[str, object]:
        return _flow_exit(
            {
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="verification_required", source="gate"
                ),
                **update,
            }
        )

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE identity: collect the claimed contact, or clarify, or leave.
        The claim is code-matched HERE (never model-judged); a match mints PendingIdentity
        with the grants-at-mint snapshot (the binding invariant's baseline)."""
        bound = identity_store.current()
        if bound is not None:
            # Already proven this session (a misrouted second enumeration handover): mint
            # from the binding directly — no model call, no re-claim; the guardrail's
            # bound-same-customer check routes straight to apply (no second OTP).
            pending = PendingIdentity(
                customer_ref=bound.customer_ref,
                masked_contact=bound.masked_contact,
                attempt_key=uuid.uuid4().hex,
                grants_at_mint=len(verification_store.grants),
            )
            return {"pending_identity": pending}
        prompt = SystemMessage(compose_identity_prompt(display_name, policy))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal gets ONE corrective re-prompt
            response = model.invoke(messages)
            new_messages.append(response)
            if not response.tool_calls:
                return {"messages": new_messages}  # clarifying question (streamed) — stay
            ack_extra_tool_calls(response, new_messages)
            call = response.tool_calls[0]
            if call["name"] == "leave_identity":
                return _leave(new_messages, call["id"], "left_flow")

            if call["name"] == "propose_identity":
                try:
                    proposed = _ProposeIdentity.model_validate(call["args"])
                except ValueError:
                    proposed = None
                if proposed is None or not proposed.contact_claim.strip():
                    feedback = (
                        "Invalid proposal. contact_claim must be the email or phone number "
                        "the caller stated, exactly as they said it."
                    )
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                # Constant tool result — the claim is never echoed into thread history
                # (the raw value's only journey is this stack frame -> match_contact).
                new_messages.append(
                    ToolMessage("identity claim received", tool_call_id=call["id"])
                )
                matched = customers.match_contact(proposed.contact_claim)
                if matched is not None:
                    pending = PendingIdentity(
                        customer_ref=matched.customer_ref,
                        masked_contact=matched.masked_contact,
                        attempt_key=uuid.uuid4().hex,
                        grants_at_mint=len(verification_store.grants),
                    )
                    return {"messages": new_messages, "pending_identity": pending}
                if state.identity_claim_misses < policy.contact_reask_max:
                    # Bounded re-ask (decision 4; budget = policy.contact_reask_max, was a
                    # hardcoded ONE) — spoken by the reask node, not here (a speakable
                    # assemble double-speaks its streamed clarifies). contact_reask_max=0
                    # skips this entirely: the first miss hands over.
                    write_event({"event": "identity_reask", "reason": "no_match"})
                    return {
                        "messages": new_messages,
                        "identity_claim_misses": state.identity_claim_misses + 1,
                    }
                # Budget exhausted: SILENT human handover — the handover deferral is the single
                # voice (the cancel-risk pattern); no flow-authored line to distinguish
                # outcomes for a probing caller.
                write_event({"event": "identity_stepup_failed", "reason": "no_match"})
                return _human_handover({"messages": new_messages})

            # Explicit terminal guard (not fall-through): every bound tool has a branch
            # above, so an unhandled name means a NEW tool was bound without a handler.
            raise ValueError(f"identity assemble: unhandled tool {call['name']!r}")
        logger.warning("identity assemble: two invalid proposals; leaving flow")
        write_event({"event": "identity_left", "reason": "invalid_proposals"})
        return _flow_exit({"messages": new_messages, "active_flow": "left_identity"})

    def reask_node(state: ReasoningState) -> dict[str, object]:
        """OWN speakable node for the code-authored re-ask (the cart_ack lesson: assemble
        must never author spoken lines). Sticky stays — next turn re-enters assemble."""
        return {"messages": [AIMessage(_REASK_LINE)]}

    def guardrail_node(state: ReasoningState) -> dict[str, object]:
        """Records whether a step-up is needed (triad leg 1 — no side effect; the router,
        closed over the LIVE stores, decides). Enumeration's L2 floor is platform code
        (`identity_required_level`), no merchant knob."""
        pending = state.pending_identity
        assert pending is not None
        bound = identity_store.current()
        already_proven = (
            bound is not None
            and bound.customer_ref == pending.customer_ref
            and verification_store.current_level() >= identity_required_level()
        )
        if not already_proven:
            write_event({"event": "identity_stepup_required", "required_level": 2})
        return {}

    def route_after_guardrail(state: ReasoningState) -> str:
        pending = state.pending_identity
        assert pending is not None
        bound = identity_store.current()
        if (
            bound is not None
            and bound.customer_ref == pending.customer_ref
            and verification_store.current_level() >= identity_required_level()
        ):
            return "confirm"  # already bound to THIS customer — no second OTP
        return "stepup"

    stepup = build_stepup_nodes(
        verification_store,
        otp,
        risk,
        pending_field="pending_identity",
        required_level=lambda p: identity_required_level(),
        event_prefix="identity",
        max_otp_attempts=policy.otp_max_attempts,
    )

    def route_after_collect(state: ReasoningState) -> str:
        """The factory decision, WRAPPED with the binding invariant: the factory confirms
        on level alone, but a bind requires THIS chain's OTP to have succeeded — a stale
        cross-family L2 with no NEW grant since mint re-collects instead (the factory's
        tries counter still bounds the loop: `max_otp_attempts` committed misses -> human)."""
        decision = stepup.route_after_collect(state)
        if decision == "confirm":
            pending = state.pending_identity
            assert pending is not None  # collect clears only via handover
            bound = identity_store.current()
            already_proven = bound is not None and bound.customer_ref == pending.customer_ref
            if not already_proven and len(verification_store.grants) <= pending.grants_at_mint:
                return "dispatch"
        return decision

    def apply_node(state: ReasoningState) -> dict[str, object]:
        """Bind the session to the verified customer (authentication), then branch on WHY
        identity was entered. Re-validates BOTH invariant legs live before a NEW bind (§A4c,
        triad leg 3): the level, and a grant NEWER than the pending's mint snapshot (this
        chain's OTP really succeeded). An already-bound same-customer pass (the misrouted-
        handover shortcut) proceeds without re-proving.

        Two continuations (the Milestone B split):
          - a RETAINED CancelSelection (a "cancel all my orders" that came here to verify):
            hand back to the support RESOLVER — keep the selection, set active_flow="support",
            speak NO order list (the resolver reads the live binding + current cancellable
            orders and freezes the batch). Does NOT go through `_flow_exit` (which would clear
            the selection); it clears `pending_identity`/misses itself.
          - otherwise (plain list_orders): speak the order list + exit (the enumeration ask)."""
        pending = state.pending_identity
        assert pending is not None
        bound = identity_store.current()
        needs_bind = bound is None or bound.customer_ref != pending.customer_ref
        if needs_bind:
            if verification_store.current_level() < identity_required_level():
                write_event({"event": "identity_stepup_failed", "reason": "level_lapsed_at_apply"})
                return _human_handover({})
            if len(verification_store.grants) <= pending.grants_at_mint:
                write_event({"event": "identity_stepup_failed", "reason": "unproven_binding"})
                return _human_handover({})
            identity_store.bind(
                BoundIdentity(
                    customer_ref=pending.customer_ref, masked_contact=pending.masked_contact
                )
            )
            write_event(
                {
                    "event": "identity_bound",
                    "customer_ref": pending.customer_ref,
                    "verification": [g.get("method") for g in verification_store.grants],
                }
            )
        if isinstance(state.pending_cancel, CancelSelection):
            # Cancel-SCOPE continuation ("cancel all"): bound, hand to the resolver (NOT
            # _flow_exit — keep the selection). Clear only identity's own turn-scoped state here.
            write_event(
                {"event": "identity_bound_for_cancel", "customer_ref": pending.customer_ref}
            )
            return {
                "pending_identity": None,
                "identity_claim_misses": 0,
                "active_flow": "support",
            }
        if state.identity_resume == "support_assemble":
            # Explicit-order mutation continuation (Fix 2): bound, hand back to support_assemble
            # so the model re-proposes the cancel/refund/return from history — now authorized as
            # the bound owner (`order_mutation_allowed`). Clears the marker; speaks NO list.
            write_event(
                {"event": "identity_bound_for_action", "customer_ref": pending.customer_ref}
            )
            return {
                "pending_identity": None,
                "identity_claim_misses": 0,
                "identity_resume": None,
                "active_flow": "support",
            }
        line = render_order_list_line(order_store.owned_orders(pending.customer_ref))
        return _flow_exit({"messages": [AIMessage(line)], "active_flow": None})

    def abort_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: explicit abort while identity was in flight. Names what was
        dropped (the verification) and what wasn't touched (the account)."""
        write_event({"event": "identity_cancelled", "reason": "aborted"})
        return _flow_exit(
            {
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "No problem - I've dropped that. Nothing on your account has changed."
                    )
                ],
            }
        )

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-identity (§A9 no-trap)."""
        write_event({"event": "identity_cancelled", "reason": "human_requested"})
        return _flow_exit(
            {
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="other", source="gate"
                ),
            }
        )

    def route_after_assemble(state: ReasoningState) -> str:
        # Which outcome did assemble reach? Order matters (terminal states first).
        if state.active_flow == "left_identity":
            return "leave"  # model left / two invalid proposals -> normal pipeline answers
        if state.handover is not None:
            return "handover"  # terminal second no-match -> the silent human path
        if state.pending_identity is not None:
            return "guardrail"
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                # A propose_identity call that minted nothing = the claim didn't match
                # (terminal was caught above) -> the ONE bounded re-ask.
                if any(call["name"] == "propose_identity" for call in msg.tool_calls):
                    return "reask"
                break
        return "clarify"  # the model's question streamed already; end the turn in-flow

    def _clear_selection_on_handover(
        node: Callable[[ReasoningState], dict[str, object]],
    ) -> Callable[[ReasoningState], dict[str, object]]:
        """Wrap a factory step-up node (risk_check/collect) so ANY handover it emits (SIM-swap
        risk, OTP exhaustion) also drops a retained CancelSelection (Milestone B) AND the
        `identity_resume` marker (Fix 2) — a FAILED verification must leave ZERO action intent,
        so neither the scope selector nor the explicit-order resume survives to a later turn.
        The factory clears only `pending_identity`; both live in channels the shared factory
        must not know about — so the clear is HERE, not in _stepup.py."""

        def wrapped(state: ReasoningState) -> dict[str, object]:
            update = node(state)
            if update.get("handover") is not None:
                update = {**update, "pending_cancel": None, "identity_resume": None}
            return update

        return wrapped

    return IdentityNodes(
        assemble=assemble_node,
        reask=reask_node,
        guardrail=guardrail_node,
        risk_check=_clear_selection_on_handover(stepup.risk_check),
        dispatch=stepup.dispatch,
        collect=_clear_selection_on_handover(stepup.collect),
        apply=apply_node,
        abort=abort_node,
        escape_human=escape_human_node,
        route_after_assemble=route_after_assemble,
        route_after_guardrail=route_after_guardrail,
        route_after_collect=route_after_collect,
        speakable_nodes=frozenset(
            {
                "identity_reask",  # authors the softened re-ask line
                "identity_risk_check",
                "identity_collect",
                "identity_apply",  # authors the scoped order list
                "identity_abort",
            }
        ),
    )
