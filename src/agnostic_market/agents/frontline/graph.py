"""Typed capability graph and lifecycle boundary for voice reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from agnostic_market.agents._consent import classify_confirmation
from agnostic_market.agents._copy import identity_status_line, warm_close
from agnostic_market.agents.capabilities import (
    CapabilityEntry,
    CapabilityRegistry,
    CapabilityRegistryError,
    CapabilitySpec,
)
from agnostic_market.agents.cart import build_cart_nodes
from agnostic_market.agents.clarification import advance_clarification
from agnostic_market.agents.frontline.read_flow import (
    ANSWER_CLARIFY_NODE,
    ANSWER_RESPONSE_NODE,
    ANSWER_UNSUPPORTED_NODE,
    CATALOG_ENTRY_NODE,
    CATALOG_QUERY_REJECT_NODE,
    CATALOG_RESPONSE_NODE,
    ORDER_STATUS_ENTRY_NODE,
    ORDER_STATUS_FULFILL_NODE,
    ORDER_STATUS_TARGET_ASK_NODE,
    ORDER_STATUS_TARGET_CONFIRM_NODE,
    ORDER_STATUS_TARGET_PROPOSE_NODE,
    ORDER_STATUS_TARGET_REJECT_NODE,
    READ_FLOW_MODEL_SPEECH_NODES,
    build_read_flow_nodes,
)
from agnostic_market.agents.identity import build_identity_nodes
from agnostic_market.agents.lifecycle import PrincipalTransitionLifecycle
from agnostic_market.agents.model_speech import CallerAudibleModelTextPolicy
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    RECOVERY_NODE_NAME,
    RECOVERY_TERMINALIZER_NODE_NAME,
    CommerceEffectFinishers,
    NodePolicyRegistry,
    build_node_error_handler,
    build_recovery_infrastructure_handler,
    build_recovery_node,
    build_recovery_terminalizer,
    clear_automation_state,
    validate_automation_state_clear,
)
from agnostic_market.agents.support import build_support_nodes
from agnostic_market.agents.telemetry import (
    SessionTelemetry,
    record_capability_answered,
)
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.catalog import CatalogPort
from agnostic_market.commerce.identity import CallerIdentityStore, CustomerDirectoryPort
from agnostic_market.commerce.orders import (
    GuestOrderScope,
    OrderPort,
    RecentOrderContext,
    render_cart_line,
)
from agnostic_market.commerce.payment_instruments import PaymentInstrumentPort
from agnostic_market.commerce.profile import ProfilePort
from agnostic_market.commerce.verification import RiskPort, VerificationStore
from agnostic_market.dtos.llm import StructuredOutputMethod
from agnostic_market.dtos.orchestration import (
    AbortCurrent,
    ActiveInvocation,
    AnswerQuestion,
    CancelOrders,
    CapabilityDispatchEnvelope,
    CapabilityId,
    ChangeProfile,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    RefundOrder,
    RequestPerson,
    ReturnOrder,
    RouterNoActionEnvelope,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import (
    HandoffRequest,
    HandoffSource,
    PolicyContext,
    ReasoningState,
)

_CAPABILITY_DISPATCH_NODE = "capability_dispatch"
_ROUTER_NO_ACTION_NODE = "router_no_action"
_ABORT_CURRENT_NODE = "abort_current"
_REQUEST_PERSON_NODE = "request_person"
_CAPABILITY_DISPATCH_REJECTED_LINE = "I couldn't complete that request. Please try again."
_ROUTER_NO_ACTION_LINES = {
    "ambiguous_intent": ("I'm not sure what you'd like me to do. Could you say that another way?"),
    "missing_target": "I'm not sure which item or order you mean. Could you be more specific?",
    "missing_value": "I'm missing a detail needed to route that request. What should it be?",
    "unsupported_workflow": (
        "I can handle one request at a time. Which part would you like help with first?"
    ),
    "unsupported_capability": _CAPABILITY_DISPATCH_REJECTED_LINE,
    "invalid_output": _CAPABILITY_DISPATCH_REJECTED_LINE,
    "routing_unavailable": _CAPABILITY_DISPATCH_REJECTED_LINE,
    "context_invalid": _CAPABILITY_DISPATCH_REJECTED_LINE,
    "decision_rejected": _CAPABILITY_DISPATCH_REJECTED_LINE,
}
_SUPPORT_CAPABILITY_ENTRY_NODE = "support_capability_entry"
_SUPPORT_CAPABILITY_RENDER_NODE = "support_capability_render"
_IDENTITY_CAPABILITY_ENTRY_NODE = "identity_capability_entry"
_CART_CAPABILITY_ENTRY_NODE = "cart_capability_entry"
# Pure code-authored read owners: no model, no tools, no flow coupling, so they live here beside
# the other code-authored reads rather than in the cart/identity packages. An owner that DOES
# couple to a flow (slot gathering, HITL) belongs in that flow's package, as Support's does.
_CART_VIEW_RENDER_NODE = "cart_view_render"
_IDENTITY_STATUS_RENDER_NODE = "identity_status_render"


@dataclass(frozen=True, slots=True)
class FrontlineGraphAssembly:
    """The compiled graph plus the exact registry its dispatcher resolves against.

    Returned together so runtime, evaluator, and tests share ONE instance; a second
    availability list built alongside the graph drifts the day a capability is registered.
    """

    graph: CompiledStateGraph
    capability_registry: CapabilityRegistry


def _build_frontline_capability_registry() -> CapabilityRegistry:
    support_entry = CapabilityEntry(_SUPPORT_CAPABILITY_ENTRY_NODE)
    identity_entry = CapabilityEntry(_IDENTITY_CAPABILITY_ENTRY_NODE)
    cart_entry = CapabilityEntry(_CART_CAPABILITY_ENTRY_NODE)
    cart_view_entry = CapabilityEntry(_CART_VIEW_RENDER_NODE)
    identity_status_entry = CapabilityEntry(_IDENTITY_STATUS_RENDER_NODE)
    catalog_entry = CapabilityEntry(CATALOG_ENTRY_NODE)
    answer_entry = CapabilityEntry(ANSWER_RESPONSE_NODE)
    order_status_entry = CapabilityEntry(ORDER_STATUS_ENTRY_NODE)
    abort_current_entry = CapabilityEntry(_ABORT_CURRENT_NODE)
    request_person_entry = CapabilityEntry(_REQUEST_PERSON_NODE)
    return CapabilityRegistry(
        (
            CapabilitySpec(CapabilityId.LIST_ORDERS, ListOrders, support_entry),
            CapabilitySpec(CapabilityId.CANCEL_ORDERS, CancelOrders, support_entry),
            CapabilitySpec(CapabilityId.REFUND_ORDER, RefundOrder, support_entry),
            CapabilitySpec(CapabilityId.RETURN_ORDER, ReturnOrder, support_entry),
            CapabilitySpec(CapabilityId.CHANGE_PROFILE, ChangeProfile, support_entry),
            CapabilitySpec(CapabilityId.VIEW_CART, ViewCart, cart_view_entry),
            CapabilitySpec(
                CapabilityId.VIEW_IDENTITY_STATUS,
                ViewIdentityStatus,
                identity_status_entry,
            ),
            CapabilitySpec(CapabilityId.VERIFY_IDENTITY, VerifyIdentity, identity_entry),
            CapabilitySpec(CapabilityId.SWITCH_ACCOUNT, SwitchAccount, identity_entry),
            CapabilitySpec(CapabilityId.MODIFY_CART, ModifyCart, cart_entry),
            CapabilitySpec(CapabilityId.PLACE_ORDER, PlaceOrder, cart_entry),
            CapabilitySpec(CapabilityId.SEARCH_CATALOG, SearchCatalog, catalog_entry),
            CapabilitySpec(CapabilityId.ANSWER_QUESTION, AnswerQuestion, answer_entry),
            CapabilitySpec(
                CapabilityId.VERIFY_ORDER_STATUS,
                VerifyOrderStatus,
                order_status_entry,
            ),
            CapabilitySpec(
                CapabilityId.ABORT_CURRENT,
                AbortCurrent,
                abort_current_entry,
            ),
            CapabilitySpec(
                CapabilityId.REQUEST_PERSON,
                RequestPerson,
                request_person_entry,
            ),
        )
    )


# Graph topology is the source of truth for model-authored speech provenance. These nodes invoke
# a model for typed proposals or transactional decisions but never own caller prose, so none may
# be caller-speakable (asserted at compile).
NON_SPEAKING_MODEL_NODES = frozenset(
    {
        _CART_CAPABILITY_ENTRY_NODE,
        _SUPPORT_CAPABILITY_ENTRY_NODE,
        "identity_assemble",
        ORDER_STATUS_TARGET_PROPOSE_NODE,
    }
)
MODEL_SPEECH_NODES = READ_FLOW_MODEL_SPEECH_NODES
# Frontline-owned code-speech subset, not the compiled production set. Cart, Support,
# Identity, and read-flow bundles retain ownership of their names and graph assembly unions them.
FRONTLINE_SPEAKABLE_NODES = frozenset(
    {
        "handover",
        "automation_terminal_response",
        _ABORT_CURRENT_NODE,
        "owner_declined",
        "principal_warning",
        _CAPABILITY_DISPATCH_NODE,
        _ROUTER_NO_ACTION_NODE,
        _CART_VIEW_RENDER_NODE,
        _IDENTITY_STATUS_RENDER_NODE,
        RECOVERY_NODE_NAME,
    }
)


def build_frontline_graph(
    chat_model: BaseChatModel,
    *,
    display_name: str,
    tenant_id: str,
    reasoning_model: BaseChatModel,
    store: OrderPort,
    catalog: CatalogPort,
    guest_orders: GuestOrderScope,
    policy: PolicyContext,
    cart_store: CartStore,
    verification_store: VerificationStore,
    risk: RiskPort,
    profile_store: ProfilePort,
    recent_orders: RecentOrderContext | None = None,
    identity_store: CallerIdentityStore,
    customers: CustomerDirectoryPort,
    payment_instruments: PaymentInstrumentPort,
    lifecycle: PrincipalTransitionLifecycle,
    structured_output_method: StructuredOutputMethod,
    caller_audible_model_text_max_chars: int,
    response_model_node_timeout_seconds: float,
    reasoning_model_node_timeout_seconds: float,
    session_telemetry: SessionTelemetry,
    checkpointer: BaseCheckpointSaver | None = None,
) -> FrontlineGraphAssembly:
    """Compile the reasoning graph (frontline routing tier + the cart, support, and identity
    flows) and return it with the capability registry its dispatcher resolves against.

    `checkpointer` is required for interrupt/resume paths. None keeps the stateless mode used
    by text evaluation. Verification and risk dependencies are injected explicitly.

    The stores are session-owned dependencies shared by routing context projection and typed
    capability owners.
    """
    if not tenant_id.strip():
        raise ValueError("frontline graph requires a tenant id")
    tenant_dependencies = {
        "catalog": catalog.tenant_id,
        "order store": store.tenant_id,
        "guest-order scope": guest_orders.tenant_id,
        "customer directory": customers.tenant_id,
        "payment instrument directory": payment_instruments.tenant_id,
        "profile store": profile_store.tenant_id,
        "verification store": verification_store.tenant_id,
        "risk provider": risk.tenant_id,
        "session telemetry": session_telemetry.tenant_id,
    }
    mismatched_tenants = [
        name
        for name, dependency_tenant in tenant_dependencies.items()
        if dependency_tenant != tenant_id
    ]
    if mismatched_tenants:
        raise ValueError(
            "frontline tenant dependencies do not match: " + ", ".join(mismatched_tenants)
        )
    if session_telemetry.session_id != guest_orders.session_id:
        raise ValueError("frontline session dependencies do not match: session telemetry")
    telemetry = session_telemetry.operational
    routing_telemetry = session_telemetry.routing_evidence
    model_text_policy = CallerAudibleModelTextPolicy(caller_audible_model_text_max_chars)
    # Production and tests pass the same session context used by routing projection.
    recent_orders = recent_orders or RecentOrderContext(max_refs=policy.cancel_batch_max)

    def _cart_view_line(close: str) -> str:
        """Render the live session cart with an optional close."""
        if cart_store.is_empty():
            return "Your cart's empty at the moment."
        return render_cart_line(cart_store.view(), cart_store.cart_total()) + close

    def cart_view_render_node(state: ReasoningState) -> dict[str, object]:
        """Typed `ViewCart` owner: speak the live cart. No model, no tools, no mutation.

        `execution_owner` is left untouched: nothing on this path sets one, so clearing would be a
        no-op today and a silent flow-exit the day it is reachable mid-flow.
        """
        if state.active_invocation is None or not isinstance(
            state.active_invocation.request, ViewCart
        ):
            raise TypeError("cart view render requires a view-cart invocation")
        line = _cart_view_line(f" {warm_close()}")
        record_capability_answered(
            routing_telemetry,
            state.last_user_text(),
            CapabilityId.VIEW_CART.value,
            answer_source="code_authored_read",
        )
        return {"active_invocation": None, "messages": [AIMessage(line)]}

    def identity_status_render_node(state: ReasoningState) -> dict[str, object]:
        """Typed `ViewIdentityStatus` owner: bound or unbound, read from the LIVE store.

        Never infers verification from transcript or order knowledge, and speaks neither the
        customer reference nor the contact on file.
        """
        if state.active_invocation is None or not isinstance(
            state.active_invocation.request, ViewIdentityStatus
        ):
            raise TypeError("identity status render requires a view-identity-status invocation")
        # Only the verified line takes a close; the unverified one already ends in an invitation.
        verified = identity_store.current() is not None
        line = identity_status_line(verified=verified)
        if verified:
            line = f"{line} {warm_close()}"
        record_capability_answered(
            routing_telemetry,
            state.last_user_text(),
            CapabilityId.VIEW_IDENTITY_STATUS.value,
            answer_source="code_authored_read",
        )
        return {"active_invocation": None, "messages": [AIMessage(line)]}

    def handover_node(state: ReasoningState) -> dict[str, object]:
        """Terminate automation and emit the bounded human-onramp package."""
        if state.handover is None or state.handover.destination != "human":
            raise TypeError("handover requires a human destination")
        handover = state.handover
        telemetry.record(
            {
                "event": "human_onramp",
                "handover_schema_version": 2,
                "verification_level": verification_store.current_level(),
                "execution_owner": state.execution_owner,
                "reason_code": handover.reason_code,
                "source": handover.source,
            }
        )
        return {**clear_automation_state(), "automation_terminal": True}

    def entry_node(state: ReasoningState) -> dict[str, object]:
        # Fresh-turn hygiene: with a checkpointer, LAST turn's handover signal, the
        # turn-scoped "left_*" marker, pending_ack, and pending_clarification persist in
        # thread state and must not affect THIS turn. Pending confirmations arrive as resumes
        # and never pass through entry.
        update: dict[str, object] = {
            "handover": None,
            "pending_ack": None,
            "pending_clarification": None,
        }
        return update

    def automation_terminal_response_node(_state: ReasoningState) -> dict[str, object]:
        telemetry.record({"event": "automation_terminal_response"})
        return {
            **clear_automation_state(),
            "messages": [AIMessage(AUTOMATION_TERMINAL_LINE)],
        }

    def request_person_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, RequestPerson):
            raise TypeError("request-person owner requires a request-person invocation")
        routing_telemetry.record({"event": "semantic_human_requested"})
        return {
            **clear_automation_state(),
            "handover": HandoffRequest(
                destination="human",
                reason_code="other",
                source=HandoffSource.SEMANTIC_ROUTER,
            ),
        }

    def abort_current_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, AbortCurrent):
            raise TypeError("abort-current owner requires an abort-current invocation")
        routing_telemetry.record({"event": "semantic_request_aborted"})
        return {
            **clear_automation_state(),
            "messages": [AIMessage("Okay. I've stopped that request.")],
        }

    def owner_declined_node(_state: ReasoningState) -> dict[str, object]:
        routing_telemetry.record({"event": "capability_owner_declined", "disposition": "closed"})
        return {
            **clear_automation_state(),
            "messages": [AIMessage(_ROUTER_NO_ACTION_LINES["ambiguous_intent"])],
        }

    def route_after_entry(state: ReasoningState) -> str:
        """Route lifecycle, committed control, and engine-authored instructions only."""
        if state.automation_terminal:
            return "automation_terminal_response"
        if state.pending_recovery is not None:
            return RECOVERY_NODE_NAME
        if state.pending_capability_dispatch is not None:
            return _CAPABILITY_DISPATCH_NODE
        if state.pending_router_no_action is not None:
            return _ROUTER_NO_ACTION_NODE
        if state.active_invocation is not None:
            return _CAPABILITY_DISPATCH_NODE
        raise RuntimeError("ordinary turn reached the graph without a routed instruction")

    def principal_warning_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        assert invocation is not None
        request = invocation.request
        switching = isinstance(request, SwitchAccount)
        prompt = (
            "If the new details belong to a different account, switching will clear this "
            "call's cart and recent order context. Orders already placed will remain placed, "
            "but they won't stay available in this conversation. Would you like to continue?"
            if switching
            else "Verifying an account will clear this call's cart and recent order context. "
            "Orders already placed will remain placed, but they won't stay available in this "
            "conversation. Would you like to continue?"
        )
        answer = interrupt(prompt)
        decision = classify_confirmation(answer)
        if answer.get("readback_interrupted") or decision.verdict == "unclear":
            action = "switch accounts" if switching else "verify the account"
            answer = interrupt(f"To {action} and clear this call's context, say yes or no.")
            decision = classify_confirmation(answer)
        if decision.verdict == "yes":
            return {"execution_owner": "identity"}
        if decision.verdict == "human":
            assert decision.handoff_source is not None
            return {
                "active_invocation": None,
                "execution_owner": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="switch_account" if switching else "verification_required",
                    source=decision.handoff_source,
                ),
            }
        declined = (
            "Okay, I'll keep you on the current account."
            if switching
            else "Okay, I won't start account verification."
        )
        return {
            "active_invocation": None,
            "execution_owner": None,
            "messages": [AIMessage(declined)],
        }

    def route_after_principal_warning(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"
        if state.active_invocation is not None:
            return "identity_assemble"
        return END

    def route_after_cart_capability_entry(state: ReasoningState) -> str:
        # The flow decides; graph construction maps its closed result to one node.
        decision = cart.route_after_capability_entry(state)
        return {
            "place": "cart_guardrail",
            "mutation_confirm": "cart_mutation_confirm",
            "ack": "cart_ack",
            "leave": "owner_declined",
            "clarify": "cart_clarify",
        }[decision]

    def route_after_cart_mutation_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"
        if state.pending_cart_mutation is not None:
            return "cart_mutation_apply"
        return END

    def route_after_cart_guardrail(state: ReasoningState) -> str:
        return "cart_confirm" if state.pending_placement is not None else END

    def route_after_cart_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_placement is not None:
            return "cart_place"  # explicit committed yes
        return END  # declined / expired (node spoke its line, clear-before-speak)

    cart = build_cart_nodes(
        reasoning_model,
        store,
        catalog,
        guest_orders,
        cart_store,
        policy,
        recent_orders,
        display_name=display_name,
        telemetry=telemetry,
    )
    support = build_support_nodes(
        reasoning_model,
        store,
        guest_orders,
        verification_store,
        risk,
        policy,
        profile_store,
        recent_orders,
        payment_instruments,
        identity_store=identity_store,
        display_name=display_name,
        telemetry=telemetry,
        routing_telemetry=routing_telemetry,
    )
    identity = build_identity_nodes(
        reasoning_model,
        verification_store,
        risk,
        customers,
        identity_store,
        policy,
        lifecycle.transition_principal,
        display_name=display_name,
        telemetry=telemetry,
    )
    reads = build_read_flow_nodes(
        chat_model,
        store,
        catalog,
        guest_orders,
        policy,
        display_name=display_name,
        structured_output_method=structured_output_method,
        model_text_policy=model_text_policy,
        recent_orders=recent_orders,
        identity_store=identity_store,
        customers=customers,
        telemetry=telemetry,
        routing_telemetry=routing_telemetry,
    )
    capability_registry = _build_frontline_capability_registry()

    def capability_dispatch(state: ReasoningState) -> Command:
        """Consume one admitted dispatch or resume one already-open invocation."""

        def continuation_destination(invocation: ActiveInvocation) -> str:
            if state.execution_owner == "identity":
                return "identity_assemble"
            return capability_registry.resolve(invocation.request).node_name

        def reject(reason: str) -> Command:
            telemetry.record(
                {
                    "event": "capability_dispatch_rejected",
                    "reason": reason,
                    "disposition": "closed",
                }
            )
            return Command(
                update={
                    **clear_automation_state(),
                    "messages": [AIMessage(_CAPABILITY_DISPATCH_REJECTED_LINE)],
                },
                goto=END,
            )

        envelope_value = state.pending_capability_dispatch
        if envelope_value is None:
            invocation = state.active_invocation
            if invocation is None:
                return reject("missing_invocation")
            try:
                destination = continuation_destination(invocation)
            except CapabilityRegistryError:
                return reject("unregistered")
            return Command(goto=destination)

        if not isinstance(envelope_value, CapabilityDispatchEnvelope):
            return reject("malformed_envelope")
        try:
            envelope = CapabilityDispatchEnvelope.model_validate(envelope_value)
        except (TypeError, ValueError):
            return reject("malformed_envelope")

        if not state.consumed_turn_ids or envelope.turn_id != state.consumed_turn_ids[-1]:
            return reject("stale_turn")

        if envelope.mode == "continue":
            invocation = state.active_invocation
            if invocation is None or envelope.observed_invocation_id != invocation.invocation_id:
                return reject("stale_invocation")
            try:
                destination = continuation_destination(invocation)
            except CapabilityRegistryError:
                return reject("unregistered")
            return Command(
                update={"pending_capability_dispatch": None},
                goto=destination,
            )

        request = envelope.request
        if request is None:
            return reject("malformed_envelope")
        try:
            destination = capability_registry.resolve(request).node_name
        except CapabilityRegistryError:
            return reject("unregistered")
        replacement = ActiveInvocation(
            request=request,
            opened_turn_id=envelope.turn_id,
        )
        return Command(
            update={
                **clear_automation_state(),
                "active_invocation": replacement,
            },
            goto=destination,
        )

    def router_no_action(state: ReasoningState) -> Command:
        """Consume one admitted non-executable route and bound repeated no-action turns."""

        def reject(reason: str) -> Command:
            telemetry.record(
                {
                    "event": "router_no_action_rejected",
                    "reason": reason,
                    "disposition": "closed",
                }
            )
            return Command(
                update={
                    **clear_automation_state(),
                    "messages": [AIMessage(_CAPABILITY_DISPATCH_REJECTED_LINE)],
                },
                goto=END,
            )

        envelope_value = state.pending_router_no_action
        if not isinstance(envelope_value, RouterNoActionEnvelope):
            return reject("malformed_envelope")
        try:
            envelope = RouterNoActionEnvelope.model_validate(envelope_value)
        except (TypeError, ValueError):
            return reject("malformed_envelope")
        if not state.consumed_turn_ids or envelope.turn_id != state.consumed_turn_ids[-1]:
            return reject("stale_turn")
        owner = envelope.owner
        if owner.kind == "invocation":
            invocation = state.active_invocation
            if invocation is None or owner.invocation_id != invocation.invocation_id:
                return reject("stale_invocation")
        elif state.active_invocation is not None:
            return reject("owner_mismatch")

        step = advance_clarification(
            state,
            owner=owner,
            max_reasks=policy.router_clarification_reask_max,
            telemetry=telemetry,
        )
        routing_telemetry.record(
            {
                "event": "semantic_route_no_action",
                "reason": envelope.reason,
                "owner_kind": owner.kind,
                "exhausted": step.exhausted,
            }
        )
        if step.exhausted:
            return Command(
                update={
                    **clear_automation_state(),
                    "handover": HandoffRequest(
                        destination="human",
                        reason_code="other",
                        source=HandoffSource.ROUTING_FAILURE_POLICY,
                    ),
                },
                goto="handover",
            )
        return Command(
            update={
                "pending_router_no_action": None,
                "clarification_liveness": step.liveness,
                "messages": [AIMessage(_ROUTER_NO_ACTION_LINES[envelope.reason])],
            },
            goto=END,
        )

    def route_after_identity_capability_entry(state: ReasoningState) -> str:
        invocation = state.active_invocation
        if invocation is None:
            raise TypeError("identity capability routing requires an active invocation")
        request = invocation.request
        if not isinstance(request, VerifyIdentity | SwitchAccount):
            raise TypeError("identity capability routing requires an identity request")
        changes_principal = isinstance(request, SwitchAccount) or identity_store.current() is None
        if changes_principal and lifecycle.has_discardable_state():
            return "principal_warning"
        return "identity_assemble"

    # --- identity flow routers (the flow owns the store-dependent decisions; the graph
    #     maps them to node names — same stance as the support routers below) ---
    def route_after_identity_assemble(state: ReasoningState) -> str:
        # leave | handover | guardrail | reask | clarify — from the assemble outcome.
        decision = identity.route_after_assemble(state)
        return {
            "leave": "owner_declined",
            "handover": "handover",  # terminal no-match -> the silent human path
            "guardrail": "identity_guardrail",
            "reask": "identity_reask",  # the ONE bounded re-ask (own speakable node)
            "clarify": "identity_ask_contact",
        }[decision]

    def route_after_identity_guardrail(state: ReasoningState) -> str:
        # "confirm" (already bound to THIS customer at level) goes straight to apply —
        # identity has no confirm interrupt (nothing irreversible happens at a re-list).
        return {
            "confirm": "identity_apply",
            "stepup": "identity_risk_check",
        }[identity.route_after_guardrail(state)]

    def route_after_identity_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "identity_dispatch"

    def route_after_identity_collect(state: ReasoningState) -> str:
        # The flow's WRAPPED factory decision (binding invariant: a stale cross-family L2
        # "confirm" with no new grant re-collects) mapped to THIS family's nodes.
        return {
            "confirm": "identity_apply",
            "dispatch": "identity_dispatch",
            "handover": "handover",
        }[identity.route_after_collect(state)]

    def route_after_identity_apply(state: ReasoningState) -> str:
        # Typed continuation routes to deterministic resolution; no transcript replay.
        if state.handover is not None:
            return "handover"
        if state.active_invocation is not None:
            return _CAPABILITY_DISPATCH_NODE
        return END

    # --- support flow routers (state-only; the level/status-dependent branches live INSIDE
    #     the flow, closed over the store — support.route_after_* ) ---
    def _route_support_outcome(state: ReasoningState, *, clarify_target: str) -> str:
        # refund | cancel | return | profile | resolve | needs_identity | handover | leave |
        # clarify | done.
        decision = support.route_after_capability_entry(state)
        if decision == "needs_identity":
            return "principal_warning" if lifecycle.has_discardable_state() else "identity_assemble"
        return {
            "refund": "support_guardrail",
            "cancel": "support_cancel_guardrail",
            "return": "support_return_guardrail",
            "profile": "support_profile_guardrail",
            "resolve": "support_resolve",  # a bound caller's "cancel all" -> resolve now
            "handover": "handover",  # deterministic fail-closed path (for example no profile)
            "leave": "owner_declined",
            "clarify": clarify_target,
            "done": END,
        }[decision]

    def route_after_support_capability_entry(state: ReasoningState) -> str:
        if (
            state.execution_owner != "identity"
            and state.active_invocation is not None
            and isinstance(state.active_invocation.request, ListOrders)
        ):
            return _SUPPORT_CAPABILITY_RENDER_NODE
        return _route_support_outcome(state, clarify_target="support_clarify")

    def route_after_support_resolve(state: ReasoningState) -> str:
        # confirm (a batch was frozen -> the shared cancel guardrail) | clarify (none
        # cancellable / >2 for 'both' / over-cap: the resolve node spoke + ends).
        return {
            "confirm": "support_cancel_guardrail",
            "clarify": END,
        }[support.route_after_resolve(state)]

    def route_after_support_cancel_guardrail(state: ReasoningState) -> str:
        # confirm (eligible) | handover (risk-flagged) | declined (shipped: guardrail spoke + ends).
        return {
            "confirm": "support_cancel_confirm",
            "handover": "handover",
            "declined": END,
        }[support.route_after_cancel_guardrail(state)]

    def route_after_support_cancel_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_cancel is not None:
            return "support_cancel_void"  # committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_cancel_void(state: ReasoningState) -> str:
        # The batch void processes ONE target per completion: while the pending survives (more
        # targets, progress checkpointed) it loops back to itself; when it clears (last target
        # done, the whole result spoken) it ends. A per-target store refusal is recorded as an
        # outcome and the batch continues — no handover from here.
        return "support_cancel_void" if state.pending_cancel is not None else END

    def route_support_guardrail(state: ReasoningState) -> str:
        # "confirm" (level ok) | "stepup" (-> risk_check) | "declined" (over amount /
        # cancelled / open return: guardrail spoke its line + ends) | "cancel" (remedy
        # steer) | "return" (return-first steer, Group C).
        decision = support.route_after_guardrail(state)
        return {
            "confirm": "support_confirm",
            "stepup": "support_risk_check",
            "declined": END,
            "cancel": "support_cancel_guardrail",
            "return": "support_return_guardrail",
        }[decision]

    def route_after_support_return_guardrail(state: ReasoningState) -> str:
        # confirm (eligible) | cancel (unshipped steer) | declined (guardrail spoke + ends).
        return {
            "confirm": "support_return_confirm",
            "cancel": "support_cancel_guardrail",
            "declined": END,
        }[support.route_after_return_guardrail(state)]

    def route_after_support_return_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_return is not None:
            return "support_return_place"  # committed yes
        return END  # declined/expired (node spoke its line, clear-before-speak)

    def route_after_support_return_place(state: ReasoningState) -> str:
        # place may hand to a human on a store refusal; else it spoke its outcome + ends.
        return "handover" if state.handover is not None else END

    def route_after_support_profile_guardrail(state: ReasoningState) -> str:
        # "confirm" (already L2) | "stepup" (OTP on the OLD factor) — flow-owned decision
        # (the LIVE level lives in the store, which graph.py can't see).
        return {
            "confirm": "support_profile_confirm",
            "stepup": "support_profile_risk_check",
        }[support.route_after_profile_guardrail(state)]

    def route_after_support_profile_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "support_profile_dispatch"

    def route_after_support_profile_collect(state: ReasoningState) -> str:
        # The factory's decision ("confirm" | "dispatch" | "handover") mapped to THIS
        # family's nodes — the confirm target is per-family (R5), never shared.
        decision = support.route_after_profile_collect(state)
        return {
            "confirm": "support_profile_confirm",
            "dispatch": "support_profile_dispatch",
            "handover": "handover",
        }[decision]

    def route_after_support_profile_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_profile_change is not None:
            return "support_profile_place"  # committed yes
        return END  # declined/expired (node spoke its line, clear-before-speak)

    def route_after_support_profile_place(state: ReasoningState) -> str:
        # place may hand to a human on a lapsed level / store refusal; else it spoke + ends.
        return "handover" if state.handover is not None else END

    def route_after_support_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "support_dispatch"

    def route_after_support_collect(state: ReasoningState) -> str:
        # "confirm" (raised to L2) | "dispatch" (re-collect) | "handover" (exhausted).
        decision = support.route_after_collect(state)
        return {
            "confirm": "support_confirm",
            "dispatch": "support_dispatch",
            "handover": "handover",
        }[decision]

    def route_after_support_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_refund is not None:
            return "support_place"  # explicit committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_place(state: ReasoningState) -> str:
        # place may hand to a human on a store refusal / lapsed level; else it spoke + ends.
        return "handover" if state.handover is not None else END

    graph = StateGraph(ReasoningState)
    validate_automation_state_clear()
    node_registry = NodePolicyRegistry(
        graph,
        error_handler_factory=build_node_error_handler,
        model_node_timeouts={
            **dict.fromkeys(MODEL_SPEECH_NODES, response_model_node_timeout_seconds),
            **dict.fromkeys(NON_SPEAKING_MODEL_NODES, reasoning_model_node_timeout_seconds),
        },
    )
    entry_node_name = "entry"
    node_registry.register(
        entry_node_name, entry_node, ExceptionAction.SAFE_ABORT, AbandonmentKind.PURE_ABORT
    )
    node_registry.register(
        "handover", handover_node, ExceptionAction.TERMINAL, AbandonmentKind.TERMINAL
    )
    node_registry.register(
        "automation_terminal_response",
        automation_terminal_response_node,
        ExceptionAction.ENGINE_LAST_RESORT,
        AbandonmentKind.TERMINAL,
    )
    node_registry.register(
        _REQUEST_PERSON_NODE,
        request_person_node,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _ABORT_CURRENT_NODE,
        abort_current_node,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "owner_declined",
        owner_declined_node,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "principal_warning",
        principal_warning_node,
        ExceptionAction.ABORT_PRINCIPAL_WARNING,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        _CART_CAPABILITY_ENTRY_NODE,
        cart.capability_entry,
        ExceptionAction.CART_REVIEW,
        AbandonmentKind.CART_REVIEW,
    )
    node_registry.register(
        "cart_ack", cart.ack, ExceptionAction.CART_REVIEW, AbandonmentKind.CART_REVIEW
    )
    node_registry.register(
        "cart_clarify", cart.clarify, ExceptionAction.SAFE_ABORT, AbandonmentKind.PURE_ABORT
    )
    node_registry.register(
        "cart_mutation_confirm",
        cart.mutation_confirm,
        ExceptionAction.CART_REVIEW,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        "cart_mutation_apply",
        cart.mutation_apply,
        ExceptionAction.CART_REVIEW,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        "cart_guardrail", cart.guardrail, ExceptionAction.SAFE_ABORT, AbandonmentKind.PURE_ABORT
    )
    node_registry.register(
        "cart_confirm",
        cart.confirm,
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        "cart_place",
        cart.place,
        ExceptionAction.RECONCILE_PLACEMENT,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        _CAPABILITY_DISPATCH_NODE,
        capability_dispatch,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        # DERIVED from the registry, never hand-listed: the rendering hint must name exactly the
        # nodes `resolve()` can return, and a parallel tuple would silently diverge the day a
        # capability is registered with a new owner.
        destinations=capability_registry.entry_nodes,
    )
    node_registry.register(
        _ROUTER_NO_ACTION_NODE,
        router_no_action,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _SUPPORT_CAPABILITY_ENTRY_NODE,
        support.capability_entry,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _IDENTITY_CAPABILITY_ENTRY_NODE,
        identity.capability_entry,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _SUPPORT_CAPABILITY_RENDER_NODE,
        support.capability_render,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _CART_VIEW_RENDER_NODE,
        cart_view_render_node,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        _IDENTITY_STATUS_RENDER_NODE,
        identity_status_render_node,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        CATALOG_ENTRY_NODE,
        reads.catalog_entry,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(
            CATALOG_RESPONSE_NODE,
            CATALOG_QUERY_REJECT_NODE,
        ),
    )
    node_registry.register(
        CATALOG_QUERY_REJECT_NODE,
        reads.catalog_query_reject,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        CATALOG_RESPONSE_NODE,
        reads.catalog_response,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(END,),
    )
    node_registry.register(
        ANSWER_RESPONSE_NODE,
        reads.answer_response,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(ANSWER_CLARIFY_NODE, ANSWER_UNSUPPORTED_NODE, END),
    )
    node_registry.register(
        ANSWER_CLARIFY_NODE,
        reads.answer_clarify,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        ANSWER_UNSUPPORTED_NODE,
        reads.answer_unsupported,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        ORDER_STATUS_ENTRY_NODE,
        reads.order_status_entry,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(
            ORDER_STATUS_TARGET_ASK_NODE,
            ORDER_STATUS_TARGET_PROPOSE_NODE,
            ORDER_STATUS_FULFILL_NODE,
        ),
    )
    node_registry.register(
        ORDER_STATUS_TARGET_ASK_NODE,
        reads.order_status_target_ask,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        ORDER_STATUS_TARGET_PROPOSE_NODE,
        reads.order_status_target_propose,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(
            ORDER_STATUS_TARGET_REJECT_NODE,
            ORDER_STATUS_FULFILL_NODE,
        ),
    )
    node_registry.register(
        ORDER_STATUS_TARGET_CONFIRM_NODE,
        reads.order_status_target_confirm,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
        destinations=(ORDER_STATUS_TARGET_REJECT_NODE, ORDER_STATUS_FULFILL_NODE, "handover"),
    )
    node_registry.register(
        ORDER_STATUS_TARGET_REJECT_NODE,
        reads.order_status_target_reject,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        ORDER_STATUS_FULFILL_NODE,
        reads.order_status_fulfill,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(
            ORDER_STATUS_TARGET_ASK_NODE,
            ORDER_STATUS_TARGET_CONFIRM_NODE,
            ORDER_STATUS_TARGET_REJECT_NODE,
            ORDER_STATUS_FULFILL_NODE,
            END,
        ),
    )
    node_registry.register(
        "support_clarify",
        support.clarify,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_guardrail",
        support.guardrail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_risk_check",
        support.risk_check,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_dispatch",
        support.dispatch,
        ExceptionAction.ABORT_REFUND_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "support_collect",
        support.collect,
        ExceptionAction.ABORT_REFUND_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "support_confirm",
        support.confirm,
        ExceptionAction.ABORT_REFUND_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        "support_place",
        support.place,
        ExceptionAction.RECONCILE_REFUND,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        "support_cancel_guardrail",
        support.cancel_guardrail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_cancel_confirm",
        support.cancel_confirm,
        ExceptionAction.ABORT_CANCEL_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="cancel",
    )
    node_registry.register(
        "support_cancel_void",
        support.cancel_void,
        ExceptionAction.RECONCILE_CANCEL,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        "support_resolve",
        support.resolve,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_return_guardrail",
        support.return_guardrail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_return_confirm",
        support.return_confirm,
        ExceptionAction.ABORT_RETURN_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        "support_return_place",
        support.return_place,
        ExceptionAction.RECONCILE_RETURN,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        "support_profile_guardrail",
        support.profile_guardrail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_profile_risk_check",
        support.profile_risk_check,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "support_profile_dispatch",
        support.profile_dispatch,
        ExceptionAction.ABORT_PROFILE_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "support_profile_collect",
        support.profile_collect,
        ExceptionAction.ABORT_PROFILE_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "support_profile_confirm",
        support.profile_confirm,
        ExceptionAction.ABORT_PROFILE_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )
    node_registry.register(
        "support_profile_place",
        support.profile_place,
        ExceptionAction.RECONCILE_PROFILE_CHANGE,
        AbandonmentKind.AUTHORITATIVE_RECONCILE,
    )
    node_registry.register(
        "identity_assemble",
        identity.assemble,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "identity_ask_contact",
        identity.ask_contact,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "identity_reask",
        identity.reask,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "identity_guardrail",
        identity.guardrail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "identity_risk_check",
        identity.risk_check,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    node_registry.register(
        "identity_dispatch",
        identity.dispatch,
        ExceptionAction.ABORT_IDENTITY_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "identity_collect",
        identity.collect,
        ExceptionAction.ABORT_IDENTITY_VERIFICATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register(
        "identity_apply",
        identity.apply,
        ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
    )
    node_registry.register_infrastructure(
        RECOVERY_NODE_NAME,
        build_recovery_node(
            node_registry.validated_policies,
            node_registry.validated_handled_nodes,
            entry_node_name,
            cart_store,
            store,
            profile_store,
            CommerceEffectFinishers(
                placement=cart.finish_placement,
                cart_mutation=cart.reconcile_mutation,
                refund=support.finish_refund,
                cancel=support.finish_cancel,
                return_=support.finish_return,
                profile_change=support.finish_profile_change,
            ),
            lifecycle.inspect_principal_transition,
            lifecycle.invalidate_principal_transition,
            telemetry,
        ),
        error_handler=build_recovery_infrastructure_handler,
        destinations=(entry_node_name, END),
    )
    node_registry.register_infrastructure(
        RECOVERY_TERMINALIZER_NODE_NAME,
        build_recovery_terminalizer,
    )
    # Checkpoint-only lifecycle seam: the engine uses this declared completed node to seed
    # replay-defense metadata onto a fresh principal thread when there is no continuation.
    # ``update_state(as_node=...)`` records the update without executing this callable, and
    # its END edge leaves no pending graph work. LangGraph does not accept END itself as an
    # update-state author on the installed version.
    principal_seed_complete_node = "principal_seed_complete"
    node_registry.register_infrastructure(
        principal_seed_complete_node,
        lambda _state: {},
    )

    graph.add_edge(START, entry_node_name)
    graph.add_conditional_edges(
        entry_node_name,
        route_after_entry,
        {
            "automation_terminal_response": "automation_terminal_response",
            RECOVERY_NODE_NAME: RECOVERY_NODE_NAME,
            _ROUTER_NO_ACTION_NODE: _ROUTER_NO_ACTION_NODE,
            _CAPABILITY_DISPATCH_NODE: _CAPABILITY_DISPATCH_NODE,
        },
    )
    graph.add_edge(_REQUEST_PERSON_NODE, "handover")
    graph.add_edge(_ABORT_CURRENT_NODE, END)
    graph.add_edge("owner_declined", END)
    graph.add_edge("handover", "automation_terminal_response")
    graph.add_conditional_edges(
        "principal_warning",
        route_after_principal_warning,
        {"identity_assemble": "identity_assemble", "handover": "handover", END: END},
    )
    cart_outcome_routes = {
        "owner_declined": "owner_declined",
        "cart_ack": "cart_ack",
        "cart_clarify": "cart_clarify",
        "cart_mutation_confirm": "cart_mutation_confirm",
        "cart_guardrail": "cart_guardrail",
    }
    graph.add_conditional_edges(
        _CART_CAPABILITY_ENTRY_NODE,
        route_after_cart_capability_entry,
        cart_outcome_routes,
    )
    graph.add_edge("cart_ack", END)
    graph.add_edge("cart_clarify", END)
    graph.add_conditional_edges(
        "cart_mutation_confirm",
        route_after_cart_mutation_confirm,
        {"handover": "handover", "cart_mutation_apply": "cart_mutation_apply", END: END},
    )
    graph.add_edge("cart_mutation_apply", "cart_ack")
    graph.add_conditional_edges(
        "cart_guardrail",
        route_after_cart_guardrail,
        {"cart_confirm": "cart_confirm", END: END},
    )
    graph.add_conditional_edges(
        "cart_confirm",
        route_after_cart_confirm,
        {"handover": "handover", "cart_place": "cart_place", END: END},
    )
    graph.add_edge("cart_place", END)
    graph.add_conditional_edges(
        _SUPPORT_CAPABILITY_ENTRY_NODE,
        route_after_support_capability_entry,
        {
            "owner_declined": "owner_declined",
            "support_guardrail": "support_guardrail",
            "support_cancel_guardrail": "support_cancel_guardrail",
            "support_return_guardrail": "support_return_guardrail",
            "support_profile_guardrail": "support_profile_guardrail",
            "support_resolve": "support_resolve",
            "identity_assemble": "identity_assemble",
            "principal_warning": "principal_warning",
            "support_clarify": "support_clarify",
            _SUPPORT_CAPABILITY_RENDER_NODE: _SUPPORT_CAPABILITY_RENDER_NODE,
            "handover": "handover",
            END: END,
        },
    )
    graph.add_conditional_edges(
        _IDENTITY_CAPABILITY_ENTRY_NODE,
        route_after_identity_capability_entry,
        {
            "identity_assemble": "identity_assemble",
            "principal_warning": "principal_warning",
        },
    )
    graph.add_edge(_SUPPORT_CAPABILITY_RENDER_NODE, END)
    graph.add_edge(_CART_VIEW_RENDER_NODE, END)  # code-authored typed read ENDs, no model pass
    graph.add_edge(_IDENTITY_STATUS_RENDER_NODE, END)
    graph.add_edge(CATALOG_QUERY_REJECT_NODE, END)
    graph.add_edge(ANSWER_CLARIFY_NODE, END)
    graph.add_edge(ANSWER_UNSUPPORTED_NODE, END)
    graph.add_edge(ORDER_STATUS_TARGET_ASK_NODE, END)
    graph.add_edge(ORDER_STATUS_TARGET_REJECT_NODE, END)
    graph.add_edge("support_clarify", END)
    graph.add_conditional_edges(
        "support_guardrail",
        route_support_guardrail,
        {
            "support_confirm": "support_confirm",
            "support_risk_check": "support_risk_check",
            "support_cancel_guardrail": "support_cancel_guardrail",
            "support_return_guardrail": "support_return_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_risk_check",
        route_after_support_risk,
        {"support_dispatch": "support_dispatch", "handover": "handover"},
    )
    graph.add_edge("support_dispatch", "support_collect")
    graph.add_conditional_edges(
        "support_collect",
        route_after_support_collect,
        {
            "support_confirm": "support_confirm",
            "support_dispatch": "support_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "support_confirm",
        route_after_support_confirm,
        {"handover": "handover", "support_place": "support_place", END: END},
    )
    graph.add_conditional_edges(
        "support_place",
        route_after_support_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_guardrail",
        route_after_support_cancel_guardrail,
        {"support_cancel_confirm": "support_cancel_confirm", "handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_confirm",
        route_after_support_cancel_confirm,
        {"handover": "handover", "support_cancel_void": "support_cancel_void", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_void",
        route_after_support_cancel_void,
        {"support_cancel_void": "support_cancel_void", END: END},  # self-loop per target
    )
    graph.add_conditional_edges(
        "support_resolve",
        route_after_support_resolve,
        {"support_cancel_guardrail": "support_cancel_guardrail", END: END},
    )
    graph.add_conditional_edges(
        "support_return_guardrail",
        route_after_support_return_guardrail,
        {
            "support_return_confirm": "support_return_confirm",
            "support_cancel_guardrail": "support_cancel_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_return_confirm",
        route_after_support_return_confirm,
        {"handover": "handover", "support_return_place": "support_return_place", END: END},
    )
    graph.add_conditional_edges(
        "support_return_place",
        route_after_support_return_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_profile_guardrail",
        route_after_support_profile_guardrail,
        {
            "support_profile_confirm": "support_profile_confirm",
            "support_profile_risk_check": "support_profile_risk_check",
        },
    )
    graph.add_conditional_edges(
        "support_profile_risk_check",
        route_after_support_profile_risk,
        {"support_profile_dispatch": "support_profile_dispatch", "handover": "handover"},
    )
    graph.add_edge("support_profile_dispatch", "support_profile_collect")
    graph.add_conditional_edges(
        "support_profile_collect",
        route_after_support_profile_collect,
        {
            "support_profile_confirm": "support_profile_confirm",
            "support_profile_dispatch": "support_profile_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "support_profile_confirm",
        route_after_support_profile_confirm,
        {"handover": "handover", "support_profile_place": "support_profile_place", END: END},
    )
    graph.add_conditional_edges(
        "support_profile_place",
        route_after_support_profile_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "identity_assemble",
        route_after_identity_assemble,
        {
            "owner_declined": "owner_declined",
            "handover": "handover",
            "identity_guardrail": "identity_guardrail",
            "identity_ask_contact": "identity_ask_contact",
            "identity_reask": "identity_reask",
            END: END,
        },
    )
    graph.add_edge("identity_ask_contact", END)
    graph.add_edge("identity_reask", END)
    graph.add_conditional_edges(
        "identity_guardrail",
        route_after_identity_guardrail,
        {
            "identity_apply": "identity_apply",
            "identity_risk_check": "identity_risk_check",
        },
    )
    graph.add_conditional_edges(
        "identity_risk_check",
        route_after_identity_risk,
        {"identity_dispatch": "identity_dispatch", "handover": "handover"},
    )
    graph.add_edge("identity_dispatch", "identity_collect")
    graph.add_conditional_edges(
        "identity_collect",
        route_after_identity_collect,
        {
            "identity_apply": "identity_apply",
            "identity_dispatch": "identity_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "identity_apply",
        route_after_identity_apply,
        {
            "handover": "handover",
            _CAPABILITY_DISPATCH_NODE: _CAPABILITY_DISPATCH_NODE,
            END: END,
        },
    )
    graph.add_edge(RECOVERY_TERMINALIZER_NODE_NAME, "automation_terminal_response")
    graph.add_edge(principal_seed_complete_node, END)
    graph.add_edge("automation_terminal_response", END)

    node_recovery_policies = node_registry.validated_policies()
    # A dispatch destination with no regular-node recovery policy would fail unhandled on a
    # caller's turn; catch it at construction instead.
    missing_entries = {
        node for node in capability_registry.entry_nodes if node not in node_recovery_policies
    }
    if missing_entries:
        raise RuntimeError(
            f"capability entries lack regular-node recovery policies: {sorted(missing_entries)!r}"
        )
    infrastructure_nodes = node_registry.validated_infrastructure_nodes()
    handled_nodes = node_registry.validated_handled_nodes()
    handled_infrastructure_nodes = node_registry.validated_handled_infrastructure_nodes()
    node_execution_tracker = node_registry.validated_execution_tracker()
    consent_interrupt_kinds = node_registry.validated_consent_interrupt_kinds()
    compiled = graph.compile(checkpointer=checkpointer)
    speakable_nodes = (
        FRONTLINE_SPEAKABLE_NODES
        | cart.speakable_nodes
        | support.speakable_nodes
        | identity.speakable_nodes
        | reads.speakable_nodes
    )
    model_speech_nodes = reads.model_speech_nodes
    # Stashed for tests/introspection + the engine (single source of truth for which
    # node-authored messages are caller-facing — the voice side never hard-codes names).
    compiled.speakable_nodes = speakable_nodes  # type: ignore[attr-defined]
    compiled.model_speech_nodes = model_speech_nodes  # type: ignore[attr-defined]
    compiled.model_execution_nodes = (  # type: ignore[attr-defined]
        reads.model_speech_nodes | NON_SPEAKING_MODEL_NODES
    )
    compiled.capability_registry = capability_registry  # type: ignore[attr-defined]
    compiled.node_recovery_policies = node_recovery_policies  # type: ignore[attr-defined]
    compiled.recovery_infrastructure_nodes = infrastructure_nodes  # type: ignore[attr-defined]
    compiled.recovery_handled_nodes = handled_nodes  # type: ignore[attr-defined]
    compiled.recovery_handled_infrastructure_nodes = (  # type: ignore[attr-defined]
        handled_infrastructure_nodes
    )
    compiled.terminal_takeover_node = "automation_terminal_response"  # type: ignore[attr-defined]
    compiled.principal_seed_complete_node = principal_seed_complete_node  # type: ignore[attr-defined]
    compiled.recovery_entry_node = entry_node_name  # type: ignore[attr-defined]
    compiled.node_execution_tracker = node_execution_tracker  # type: ignore[attr-defined]
    compiled.consent_interrupt_kinds = consent_interrupt_kinds  # type: ignore[attr-defined]
    overlap = speakable_nodes & model_speech_nodes
    if overlap:
        raise RuntimeError(f"code/model speech source sets overlap: {sorted(overlap)!r}")
    # A model-invoking node that is ALSO speakable could put model prose in front of the
    # caller without passing the code-authored path: the structural half of one-author.
    non_speaking_overlap = speakable_nodes & NON_SPEAKING_MODEL_NODES
    if non_speaking_overlap:
        raise RuntimeError(
            f"non-speaking model nodes cannot be caller-speakable: {sorted(non_speaking_overlap)!r}"
        )
    return FrontlineGraphAssembly(graph=compiled, capability_registry=capability_registry)
