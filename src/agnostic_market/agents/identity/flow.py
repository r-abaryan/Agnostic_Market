"""Verify a matched customer, bind the session, and continue the typed request.

Account-wide authority requires the proof minted by this identity challenge. An unrelated
verification level cannot bind a customer. Unknown contact claims remain non-enumerating.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from agnostic_market.agents._copy import (
    ACCOUNT_CONTACT_QUESTION,
    identity_status_line,
    principal_completion_line,
)
from agnostic_market.agents._toolcalls import (
    ack_extra_tool_calls,
    current_turn_called,
    unknown_tool_result,
)
from agnostic_market.agents.clarification import (
    advance_clarification,
    invocation_clarification_owner,
    with_clarification_lifecycle,
)
from agnostic_market.agents.identity.prompt import compose_identity_prompt
from agnostic_market.agents.support._stepup import build_stepup_nodes
from agnostic_market.agents.telemetry import TelemetryRecorder
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectoryPort,
)
from agnostic_market.commerce.verification import (
    RiskPort,
    VerificationStore,
    VerificationSubject,
)
from agnostic_market.dtos.confirmation import identity_required_level
from agnostic_market.dtos.orchestration import (
    IntentRequest,
    PrincipalTransition,
    SwitchAccount,
    VerificationProof,
    VerifyIdentity,
)
from agnostic_market.dtos.state import (
    HandoffRequest,
    HandoffSource,
    IdentityClarification,
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
    model_config = ConfigDict(extra="forbid")

    contact_claim: str


class _RequestIdentityContact(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class IdentityNodes:
    """The identity flow's node callables + its caller-facing (speakable) node names.

    Wiring is the graph builder's job (frontline/graph.py); this module owns only behavior.
    """

    capability_entry: Callable[[ReasoningState], dict[str, object]]
    assemble: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    ask_contact: Callable[[ReasoningState], dict[str, object]]
    reask: Callable[[ReasoningState], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    risk_check: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    dispatch: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    collect: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    apply: Callable[[ReasoningState], Awaitable[dict[str, object]]]
    # "leave" | "handover" | "guardrail" | "reask" | "clarify" — from the assemble outcome.
    route_after_assemble: Callable[[ReasoningState], str]
    # "confirm" | "dispatch" | "handover" after the current identity risk decision.
    route_after_risk: Callable[[ReasoningState], str]
    # "confirm" | "collect" | "handover" for the dispatched identity challenge.
    route_after_collect: Callable[[ReasoningState], str]
    speakable_nodes: frozenset[str]


def _flow_exit(update: dict[str, object]) -> dict[str, object]:
    """Clear identity state and every typed action request on an unsuccessful flow exit."""
    return {
        "pending_identity": None,
        "identity_claim_misses": 0,
        "pending_cancel": None,
        "active_invocation": None,
        "pending_clarification": None,
        "clarification_liveness": None,
        **update,
    }


def build_identity_nodes(
    reasoning_model: BaseChatModel,
    verification_store: VerificationStore,
    risk: RiskPort,
    customers: CustomerDirectoryPort,
    identity_store: CallerIdentityStore,
    policy: PolicyContext,
    transition_principal: Callable[
        [BoundIdentity, VerificationProof, IntentRequest], Awaitable[PrincipalTransition]
    ],
    *,
    display_name: str,
    telemetry: TelemetryRecorder,
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

    @tool
    def request_identity_contact() -> str:
        """Request the account email address or phone number from the caller."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    identity_tools = (propose_identity, leave_identity, request_identity_contact)
    bound_tool_names = frozenset(tool.name for tool in identity_tools)
    model = reasoning_model.bind_tools(identity_tools)

    def capability_entry_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyIdentity | SwitchAccount):
            raise TypeError("identity capability entry requires an identity invocation")
        return {"execution_owner": "identity"}

    def _leave(new_messages: list, call_id: str, reason: str) -> dict[str, object]:
        new_messages.append(ToolMessage("left identity", tool_call_id=call_id))
        telemetry.record({"event": "identity_left", "reason": reason})
        return _flow_exit({"messages": new_messages, "execution_owner": None})

    def _human_handover(update: dict[str, object]) -> dict[str, object]:
        return _flow_exit(
            {
                "execution_owner": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="verification_required",
                    source=HandoffSource.DETERMINISTIC_POLICY,
                ),
                **update,
            }
        )

    def _clarification_result(state: ReasoningState, new_messages: list) -> dict[str, object]:
        step = advance_clarification(
            state,
            owner=invocation_clarification_owner(state),
            max_reasks=policy.identity_clarification_reask_max,
            telemetry=telemetry,
        )
        if step.exhausted:
            return _human_handover({"messages": new_messages})
        return {
            "messages": new_messages,
            "pending_clarification": IdentityClarification(),
            "clarification_liveness": step.liveness,
        }

    def _pending_for(identity: BoundIdentity) -> PendingIdentity | None:
        factor_ref = customers.verification_factor_ref(identity.customer_ref)
        if factor_ref is None:
            return None
        return PendingIdentity(
            customer_ref=identity.customer_ref,
            masked_contact=identity.masked_contact,
            factor_ref=factor_ref,
            attempt_key=uuid.uuid4().hex,
            challenge_id=None,
        )

    def _identity_subject(pending: PendingIdentity) -> VerificationSubject:
        if pending.customer_ref is None or pending.factor_ref is None:
            raise ValueError("pending identity requires a complete verification subject")
        return verification_store.subject(
            customer_ref=pending.customer_ref,
            factor_ref=pending.factor_ref,
            purpose="identity",
        )

    async def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Collect a contact claim, clarify it, or leave the identity flow."""
        bound = identity_store.current()
        invocation = state.active_invocation
        request = invocation.request if invocation is not None else None
        switching = isinstance(request, SwitchAccount)
        if bound is not None and not switching:
            # Build from the existing binding without another model claim. Risk still runs
            # before the current proof is reused.
            pending = _pending_for(bound)
            if pending is None:
                return _human_handover({})
            return {"pending_identity": pending}
        prompt = SystemMessage(compose_identity_prompt(display_name, policy))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal gets ONE corrective re-prompt
            response = await model.ainvoke(messages)
            if not response.tool_calls:
                return _clarification_result(state, new_messages)
            new_messages.append(response)
            ack_extra_tool_calls(response, new_messages)
            call = response.tool_calls[0]
            if call["name"] == "leave_identity":
                return _leave(new_messages, call["id"], "left_flow")

            if call["name"] == "request_identity_contact":
                try:
                    _RequestIdentityContact.model_validate(call["args"])
                except ValueError:
                    feedback = "Invalid request. request_identity_contact takes no arguments."
                    new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                new_messages.append(
                    ToolMessage("contact clarification requested", tool_call_id=call["id"])
                )
                return _clarification_result(state, new_messages)

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
                # Constant result: do not add another raw-value echo beyond the caller
                # transcript and model tool arguments already in checkpoint history.
                new_messages.append(ToolMessage("identity claim received", tool_call_id=call["id"]))
                matched = customers.match_contact(proposed.contact_claim)
                if matched is not None:
                    pending = _pending_for(matched)
                    if pending is None:
                        return _human_handover({"messages": new_messages})
                    return {"messages": new_messages, "pending_identity": pending}
                if state.identity_claim_misses < policy.contact_reask_max:
                    # Bounded re-ask (decision 4; budget = policy.contact_reask_max, was a
                    # hardcoded ONE) — spoken by the reask node, not here (a speakable
                    # assemble double-speaks its streamed clarifies). contact_reask_max=0
                    # skips this entirely: the first miss hands over.
                    telemetry.record({"event": "identity_reask", "reason": "no_match"})
                    return {
                        "messages": new_messages,
                        "identity_claim_misses": state.identity_claim_misses + 1,
                    }
                # Budget exhausted: SILENT human handover — the handover deferral is the single
                # voice (the cancel-risk pattern); no flow-authored line to distinguish
                # outcomes for a probing caller.
                telemetry.record({"event": "identity_stepup_failed", "reason": "no_match"})
                return _human_handover({"messages": new_messages})

            if call["name"] not in bound_tool_names:
                new_messages.append(unknown_tool_result(call["id"], leave_tool=leave_identity.name))
                messages = [prompt, *state.messages, *new_messages]
                continue
            # A bound name reaching here is code drift: preserve the loud developer failure.
            raise ValueError(f"identity assemble: bound tool has no handler: {call['name']!r}")
        logger.warning("identity assemble: two invalid tool calls; asking for contact in code")
        return _clarification_result(state, new_messages)

    def ask_contact_node(state: ReasoningState) -> dict[str, object]:
        clarification = state.pending_clarification
        if not isinstance(clarification, IdentityClarification):
            raise TypeError("identity ask-contact node requires IdentityClarification")
        return {
            "pending_clarification": None,
            "messages": [AIMessage(ACCOUNT_CONTACT_QUESTION)],
        }

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
            and verification_store.authorization_satisfies(
                _identity_subject(pending),
                identity_required_level(),
            )
        )
        if not already_proven:
            telemetry.record({"event": "identity_stepup_required", "required_level": 2})
        return {}

    stepup = build_stepup_nodes(
        verification_store,
        risk,
        pending_field="pending_identity",
        pending_type=PendingIdentity,
        required_level=lambda p: identity_required_level(),
        subject=_identity_subject,
        event_prefix="identity",
        reuse_existing_proof=True,
        max_otp_attempts=policy.otp_max_attempts,
        telemetry=telemetry,
    )

    def route_after_risk(state: ReasoningState) -> str:
        """Reuse a proof only for the customer already bound to this session."""
        decision = stepup.route_after_risk(state)
        if decision != "confirm":
            return decision
        pending = state.pending_identity
        assert pending is not None
        bound = identity_store.current()
        return (
            "confirm"
            if bound is not None and bound.customer_ref == pending.customer_ref
            else "dispatch"
        )

    def route_after_collect(state: ReasoningState) -> str:
        """Advance only when this identity challenge satisfies the required level."""
        return stepup.route_after_collect(state)

    async def apply_node(state: ReasoningState) -> dict[str, object]:
        """Bind with the exact challenge proof and continue the typed request."""
        pending = state.pending_identity
        assert pending is not None
        invocation = state.active_invocation
        if invocation is None:
            raise RuntimeError("identity apply requires an active invocation")
        request = invocation.request
        bound = identity_store.current()
        needs_bind = bound is None or bound.customer_ref != pending.customer_ref
        if needs_bind:
            exact_challenge_holds = bool(
                pending.challenge_id is not None
                and verification_store.challenge_satisfies(
                    pending.challenge_id,
                    _identity_subject(pending),
                    identity_required_level(),
                )
            )
            if not exact_challenge_holds:
                telemetry.record(
                    {"event": "identity_stepup_failed", "reason": "level_lapsed_at_apply"}
                )
                return _human_handover({})
            fresh_proof = (
                verification_store.proof_for_challenge(pending.challenge_id)
                if pending.challenge_id is not None
                else None
            )
            if fresh_proof is None:
                telemetry.record(
                    {"event": "identity_stepup_failed", "reason": "missing_fresh_proof"}
                )
                return _human_handover({})
            new_identity = BoundIdentity(
                customer_ref=pending.customer_ref, masked_contact=pending.masked_contact
            )
            transition = await transition_principal(new_identity, fresh_proof, request)
            projection = transition.projection
            continuation = projection.continuation
            completion_line = principal_completion_line(projection.completion_kind)
            telemetry.record(
                {
                    "event": "identity_bound",
                    "customer_ref": pending.customer_ref,
                    "verification": [grant.method for grant in verification_store.grants],
                }
            )
            if continuation is not None:
                telemetry.record(
                    {
                        "event": "identity_bound_for_action",
                        "customer_ref": pending.customer_ref,
                        "capability": continuation.kind,
                    }
                )
            return {
                "pending_identity": None,
                "identity_claim_misses": 0,
                "active_invocation": None,
                "execution_owner": None,
                "messages": [AIMessage(completion_line)] if completion_line is not None else [],
            }
        if isinstance(request, SwitchAccount):
            telemetry.record({"event": "principal_transition_skipped", "reason": "same_customer"})
            return _flow_exit(
                {
                    "execution_owner": None,
                    "messages": [AIMessage("You're already verified on that account.")],
                }
            )
        if isinstance(request, VerifyIdentity):
            newly_verified = bool(
                pending.challenge_id is not None
                and verification_store.proof_for_challenge(pending.challenge_id) is not None
            )
            line = (
                principal_completion_line("verify_identity")
                if newly_verified
                else identity_status_line(verified=True)
            )
            assert line is not None
            return _flow_exit(
                {
                    "execution_owner": None,
                    "messages": [AIMessage(line)],
                }
            )
        telemetry.record(
            {"event": "identity_bound_for_action", "customer_ref": pending.customer_ref}
        )
        return {
            "pending_identity": None,
            "identity_claim_misses": 0,
            "execution_owner": "support",
        }

    def route_after_assemble(state: ReasoningState) -> str:
        # Which outcome did assemble reach? Order matters (terminal states first).
        if current_turn_called(state.messages, "leave_identity"):
            return "leave"  # model explicitly left; normal pipeline answers this turn
        if state.handover is not None:
            return "handover"  # terminal second no-match -> the silent human path
        if state.pending_identity is not None:
            return "guardrail"
        if state.pending_clarification is not None:
            if not isinstance(state.pending_clarification, IdentityClarification):
                raise TypeError("identity assemble produced a non-identity clarification")
            return "clarify"
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                # A propose_identity call that minted nothing = the claim didn't match
                # (terminal was caught above) -> the ONE bounded re-ask.
                if any(call["name"] == "propose_identity" for call in msg.tool_calls):
                    return "reask"
                break
        return "clarify"  # the model's question streamed already; end the turn in-flow

    def _clear_selection_on_handover(
        node: Callable[[ReasoningState], Awaitable[dict[str, object]]],
    ) -> Callable[[ReasoningState], Awaitable[dict[str, object]]]:
        """Wrap a factory step-up node (risk_check/collect) so ANY handover it emits (SIM-swap
        risk, OTP exhaustion) also drops every typed action request — a FAILED verification
        must leave ZERO action intent for a later turn.
        The factory clears only `pending_identity`; both live in channels the shared factory
        must not know about — so the clear is HERE, not in _stepup.py."""

        async def wrapped(state: ReasoningState) -> dict[str, object]:
            update = await node(state)
            if update.get("handover") is not None:
                update = {
                    **update,
                    "pending_cancel": None,
                    "active_invocation": None,
                    "pending_clarification": None,
                }
            return update

        return wrapped

    return IdentityNodes(
        capability_entry=capability_entry_node,
        assemble=with_clarification_lifecycle(assemble_node),
        ask_contact=ask_contact_node,
        reask=reask_node,
        guardrail=guardrail_node,
        risk_check=_clear_selection_on_handover(stepup.risk_check),
        dispatch=stepup.dispatch,
        collect=_clear_selection_on_handover(stepup.collect),
        apply=apply_node,
        route_after_assemble=route_after_assemble,
        route_after_risk=route_after_risk,
        route_after_collect=route_after_collect,
        speakable_nodes=frozenset(
            {
                "identity_ask_contact",
                "identity_reask",  # authors the softened re-ask line
                "identity_risk_check",
                "identity_collect",
                "identity_apply",  # authors the scoped order list
            }
        ),
    )
