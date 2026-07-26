"""Failure-lifecycle contracts shared by graph construction and recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from langchain_core.messages import AIMessage
from langgraph.errors import NodeError
from langgraph.graph import StateGraph
from langgraph.types import Command

from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import (
    CancelRecord,
    OrderStore,
    PlacedOrder,
    RefundRecord,
    ReturnRecord,
    cancelled_batch_outcome,
    render_cart_line,
)
from agnostic_market.commerce.profile import ProfileChangeRecord, ProfileStore
from agnostic_market.commerce.receipts import CommittedReceipt, NotCommittedReceipt
from agnostic_market.dtos.orchestration import (
    PrincipalTransitionInspection,
    SwitchAccount,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    BatchCancelOutcome,
    PendingCancelBatch,
    PendingIdentity,
    PendingPlacement,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    ReasoningState,
)

TURN_FALLBACK_LINE = "Sorry - I hit a snag on my end just now. Could you say that again?"
TURN_FALLBACK_AUTHOR = "turn_fallback"
AUTOMATION_TERMINAL_LINE = (
    "I can't continue with automated assistance on this call. "
    "Please contact the store directly for further help."
)
RECOVERY_NODE_NAME = "recover_node_exception"
RECOVERY_TERMINALIZER_NODE_NAME = "terminalize_recovery_failure"
PRINCIPAL_TRANSITION_FAILURE_LINE = "I couldn't finish that account request; please try it again."

_COMMERCE_RECONCILE_ACTIONS = frozenset(
    {
        ExceptionAction.RECONCILE_PLACEMENT,
        ExceptionAction.RECONCILE_REFUND,
        ExceptionAction.RECONCILE_CANCEL,
        ExceptionAction.RECONCILE_RETURN,
        ExceptionAction.RECONCILE_PROFILE_CHANGE,
    }
)
_DEFERRED_EXCEPTION_ACTIONS = frozenset(
    {
        ExceptionAction.ENGINE_LAST_RESORT,
    }
)
_ACTION_LINES: Mapping[ExceptionAction, str] = MappingProxyType(
    {
        ExceptionAction.ABORT_PRINCIPAL_WARNING: (
            "The account verification or switch confirmation did not complete. "
            "Please try that request again."
        ),
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION: (
            "That order placement request did not complete. Your cart is still saved for review."
        ),
        ExceptionAction.ABORT_REFUND_VERIFICATION: (
            "That refund request did not complete. Please try it again."
        ),
        ExceptionAction.ABORT_REFUND_CONFIRMATION: (
            "That refund request did not complete. Please try it again."
        ),
        ExceptionAction.ABORT_CANCEL_CONFIRMATION: (
            "That cancellation request did not complete. Your orders are unchanged."
        ),
        ExceptionAction.ABORT_RETURN_CONFIRMATION: (
            "That return request did not complete. No return was created."
        ),
        ExceptionAction.ABORT_PROFILE_VERIFICATION: (
            "That profile update did not complete. Your profile is unchanged."
        ),
        ExceptionAction.ABORT_PROFILE_CONFIRMATION: (
            "That profile update did not complete. Your profile is unchanged."
        ),
        ExceptionAction.ABORT_IDENTITY_VERIFICATION: (
            "Account verification did not complete. Please try that request again."
        ),
        ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION: PRINCIPAL_TRANSITION_FAILURE_LINE,
    }
)
_NOT_COMMITTED_LINES: Mapping[ExceptionAction, str] = MappingProxyType(
    {
        ExceptionAction.RECONCILE_PLACEMENT: (
            "That order placement request did not complete. Your cart is still saved for review."
        ),
        ExceptionAction.RECONCILE_REFUND: "That refund request did not complete.",
        ExceptionAction.RECONCILE_RETURN: "That return request did not complete.",
        ExceptionAction.RECONCILE_PROFILE_CHANGE: ("That profile update request did not complete."),
    }
)
_ORDINARY_EXCEPTION_ACTIONS = frozenset(ExceptionAction) - _DEFERRED_EXCEPTION_ACTIONS
_RENDERED_EXCEPTION_ACTIONS = (
    frozenset(_ACTION_LINES)
    | _COMMERCE_RECONCILE_ACTIONS
    | {
        ExceptionAction.SAFE_ABORT,
        ExceptionAction.CART_REVIEW,
        ExceptionAction.TERMINAL,
    }
)
if _RENDERED_EXCEPTION_ACTIONS != _ORDINARY_EXCEPTION_ACTIONS:
    raise RuntimeError("ordinary recovery actions do not have a complete rendering")


@dataclass(frozen=True, slots=True)
class NodeRecoveryPolicy:
    on_exception: ExceptionAction
    on_abandonment: AbandonmentKind


@dataclass(frozen=True, slots=True)
class CommerceEffectFinishers:
    """The complete flow-owned post-commit projection boundary used by normal and recovery."""

    placement: Callable[[PlacedOrder], dict[str, object]]
    refund: Callable[[RefundRecord], dict[str, object]]
    cancel: Callable[[PendingCancelBatch, BatchCancelOutcome, bool], dict[str, object]]
    return_: Callable[[ReturnRecord], dict[str, object]]
    profile_change: Callable[[ProfileChangeRecord], dict[str, object]]


NodeErrorHandler = Callable[[ReasoningState, NodeError], Command]
NodeErrorHandlerFactory = Callable[[str, NodeRecoveryPolicy], NodeErrorHandler | None]


class NodePolicyRegistry:
    """The only production seam for registering reasoning-graph nodes."""

    def __init__(
        self,
        graph: StateGraph,
        *,
        error_handler_factory: NodeErrorHandlerFactory | None = None,
    ) -> None:
        self._graph = graph
        self._error_handler_factory = error_handler_factory
        self._policies: dict[str, NodeRecoveryPolicy] = {}
        self._infrastructure: set[str] = set()
        self._handled: set[str] = set()
        self._handled_infrastructure: set[str] = set()
        self._validated: Mapping[str, NodeRecoveryPolicy] | None = None

    def _ensure_open_and_unique(self, name: str) -> None:
        if self._validated is not None:
            raise RuntimeError("recovery policy registry is already finalized")
        if name in self._policies or name in self._infrastructure:
            raise RuntimeError(f"duplicate recovery policy registration: {name!r}")

    def _add_node(
        self,
        name: str,
        node: object,
        *,
        error_handler: NodeErrorHandler | None = None,
    ) -> None:
        options: dict[str, object] = {}
        if error_handler is not None:
            options = {"error_handler": error_handler}
        self._graph.add_node(name, node, **options)

    def register(
        self,
        name: str,
        node: object,
        on_exception: ExceptionAction,
        on_abandonment: AbandonmentKind,
    ) -> None:
        self._ensure_open_and_unique(name)
        policy = NodeRecoveryPolicy(
            on_exception=on_exception,
            on_abandonment=on_abandonment,
        )
        handler = (
            self._error_handler_factory(name, policy)
            if self._error_handler_factory is not None
            else None
        )
        self._add_node(name, node, error_handler=handler)
        self._policies[name] = policy
        if handler is not None:
            self._handled.add(name)

    def register_infrastructure(
        self,
        name: str,
        node: object,
        *,
        error_handler: NodeErrorHandler | None = None,
    ) -> None:
        self._ensure_open_and_unique(name)
        self._add_node(name, node, error_handler=error_handler)
        self._infrastructure.add(name)
        if error_handler is not None:
            self._handled_infrastructure.add(name)

    def validated_policies(self) -> Mapping[str, NodeRecoveryPolicy]:
        registered = frozenset(self._policies)
        infrastructure = frozenset(self._infrastructure)
        actual_regular = frozenset(
            name for name, spec in self._graph.nodes.items() if not spec.is_error_handler
        )
        actual_handlers = frozenset(
            name for name, spec in self._graph.nodes.items() if spec.is_error_handler
        )
        all_handled = self._handled | self._handled_infrastructure
        handler_links = {name: self._graph.nodes[name].error_handler_node for name in all_handled}
        missing_handler_links = frozenset(
            name for name, handler_name in handler_links.items() if handler_name is None
        )
        expected_handlers = frozenset(
            handler_name for handler_name in handler_links.values() if handler_name is not None
        )
        expected = registered | infrastructure
        if (
            expected != actual_regular
            or missing_handler_links
            or expected_handlers != actual_handlers
        ):
            raise RuntimeError(
                "graph nodes bypassed the recovery-policy registry: "
                f"unregistered={sorted(actual_regular - expected)!r}, "
                f"missing={sorted(expected - actual_regular)!r}, "
                f"missing_handler_links={sorted(missing_handler_links)!r}, "
                f"unexpected_handlers={sorted(actual_handlers - expected_handlers)!r}, "
                f"missing_handlers={sorted(expected_handlers - actual_handlers)!r}"
            )
        if self._validated is None:
            self._validated = MappingProxyType(dict(self._policies))
        return self._validated

    def validated_infrastructure_nodes(self) -> frozenset[str]:
        self.validated_policies()
        return frozenset(self._infrastructure)

    def validated_handled_nodes(self) -> frozenset[str]:
        self.validated_policies()
        return frozenset(self._handled)

    def validated_handled_infrastructure_nodes(self) -> frozenset[str]:
        self.validated_policies()
        return frozenset(self._handled_infrastructure)


def ordinary_exception_handler_enabled(action: ExceptionAction) -> bool:
    return action in _ORDINARY_EXCEPTION_ACTIONS


def build_node_error_handler(
    origin_node: str,
    policy: NodeRecoveryPolicy,
) -> NodeErrorHandler | None:
    if not ordinary_exception_handler_enabled(policy.on_exception):
        return None

    def handle(_state: ReasoningState, error: NodeError) -> Command:
        action = policy.on_exception
        if error.node != origin_node:
            action = (
                ExceptionAction.TERMINAL
                if action != ExceptionAction.TERMINAL
                else ExceptionAction.SAFE_ABORT
            )
        return Command(
            update={
                "pending_recovery": PendingRecovery(
                    origin_node=origin_node,
                    action=action,
                    trigger="node_exception",
                )
            },
            goto=RECOVERY_NODE_NAME,
        )

    return handle


def build_recovery_infrastructure_handler(
    _state: ReasoningState,
    _error: NodeError,
) -> Command:
    return Command(goto=RECOVERY_TERMINALIZER_NODE_NAME)


def build_recovery_terminalizer(_state: ReasoningState) -> dict[str, object]:
    return {
        **clear_automation_state(),
        "automation_terminal": True,
    }


def build_recovery_node(
    policies: Callable[[], Mapping[str, NodeRecoveryPolicy]],
    handled_nodes: Callable[[], frozenset[str]],
    cart_store: CartStore,
    order_store: OrderStore,
    profile_store: ProfileStore,
    finishers: CommerceEffectFinishers,
    inspect_principal_transition: Callable[[], PrincipalTransitionInspection],
    invalidate_principal_transition: Callable[[str | None], bool],
) -> Callable[[ReasoningState], dict[str, object]]:
    def node_failure_event(marker: PendingRecovery) -> dict[str, object]:
        return {
            "event": "turn_failed",
            "reason": "node_exception",
            "node": marker.origin_node,
            "action": marker.action,
        }

    def terminal_result(event: dict[str, object] | None = None) -> dict[str, object]:
        write_event(
            event
            or {
                "event": "turn_failed",
                "reason": "recovery_contract_invalid",
                "action": ExceptionAction.TERMINAL,
            }
        )
        return {
            **clear_automation_state(),
            "automation_terminal": True,
            "messages": [AIMessage(AUTOMATION_TERMINAL_LINE)],
        }

    def complete(update: dict[str, object]) -> dict[str, object]:
        return {**clear_automation_state(), **update}

    def not_committed(marker: PendingRecovery) -> dict[str, object]:
        line = _NOT_COMMITTED_LINES.get(marker.action)
        if line is None:
            return terminal_result(node_failure_event(marker))
        write_event(node_failure_event(marker))
        return {**clear_automation_state(), "messages": [AIMessage(line)]}

    def reconcile_effect(marker: PendingRecovery, state: ReasoningState) -> dict[str, object]:
        action = marker.action
        if action == ExceptionAction.RECONCILE_PLACEMENT:
            pending = state.pending_placement
            if not isinstance(pending, PendingPlacement):
                return terminal_result(node_failure_event(marker))
            receipt = order_store.placement_receipt(
                pending.idempotency_key,
                lines=pending.lines,
                total_usd=pending.total_usd,
            )
            if isinstance(receipt, CommittedReceipt) and isinstance(receipt.record, PlacedOrder):
                write_event(node_failure_event(marker))
                return complete(finishers.placement(receipt.record))
            if isinstance(receipt, NotCommittedReceipt):
                return not_committed(marker)
            return terminal_result(node_failure_event(marker))

        if action == ExceptionAction.RECONCILE_REFUND:
            pending = state.pending_refund
            if not isinstance(pending, PendingRefund):
                return terminal_result(node_failure_event(marker))
            receipt = order_store.refund_receipt(
                pending.idempotency_key,
                order_id=pending.order_id,
                amount_usd=pending.amount_usd,
                destination=pending.destination,
                instrument_ref=pending.instrument_ref,
            )
            if isinstance(receipt, CommittedReceipt) and isinstance(receipt.record, RefundRecord):
                write_event(node_failure_event(marker))
                return complete(finishers.refund(receipt.record))
            if isinstance(receipt, NotCommittedReceipt):
                return not_committed(marker)
            return terminal_result(node_failure_event(marker))

        if action == ExceptionAction.RECONCILE_CANCEL:
            pending = state.pending_cancel
            if not isinstance(pending, PendingCancelBatch):
                return terminal_result(node_failure_event(marker))
            done = len(pending.outcomes)
            if done >= len(pending.targets) or any(
                outcome.order_id != target.order_id
                for outcome, target in zip(pending.outcomes, pending.targets, strict=False)
            ):
                return terminal_result(node_failure_event(marker))
            target = pending.targets[done]
            receipt = order_store.cancel_receipt(
                target.idempotency_key,
                order_id=target.order_id,
            )
            if isinstance(receipt, CommittedReceipt) and isinstance(receipt.record, CancelRecord):
                outcome = cancelled_batch_outcome(receipt.record)
            elif isinstance(receipt, NotCommittedReceipt):
                outcome = BatchCancelOutcome(
                    order_id=target.order_id,
                    summary=target.summary,
                    outcome="not_completed",
                )
            else:
                return terminal_result(node_failure_event(marker))
            write_event(node_failure_event(marker))
            return complete(finishers.cancel(pending, outcome, True))

        if action == ExceptionAction.RECONCILE_RETURN:
            pending = state.pending_return
            if not isinstance(pending, PendingReturn):
                return terminal_result(node_failure_event(marker))
            receipt = order_store.return_receipt(
                pending.idempotency_key,
                order_id=pending.order_id,
                refund_due_usd=pending.refund_due_usd,
                destination="original",
            )
            if isinstance(receipt, CommittedReceipt) and isinstance(receipt.record, ReturnRecord):
                write_event(node_failure_event(marker))
                return complete(finishers.return_(receipt.record))
            if isinstance(receipt, NotCommittedReceipt):
                return not_committed(marker)
            return terminal_result(node_failure_event(marker))

        if action == ExceptionAction.RECONCILE_PROFILE_CHANGE:
            pending = state.pending_profile_change
            if not isinstance(pending, PendingProfileChange):
                return terminal_result(node_failure_event(marker))
            receipt = profile_store.profile_change_receipt(
                pending.idempotency_key,
                customer_ref=pending.customer_ref,
                field=pending.field,
                new_value=pending.new_value,
            )
            if isinstance(receipt, CommittedReceipt) and isinstance(
                receipt.record, ProfileChangeRecord
            ):
                write_event(node_failure_event(marker))
                return complete(finishers.profile_change(receipt.record))
            if isinstance(receipt, NotCommittedReceipt):
                return not_committed(marker)
            return terminal_result(node_failure_event(marker))

        return terminal_result(node_failure_event(marker))

    def reconcile_principal_transition(
        state: ReasoningState,
    ) -> dict[str, object]:
        inspection = inspect_principal_transition()
        transition = inspection.transition
        event: dict[str, object] = {
            "event": "principal_transition_reconciled",
            "outcome": inspection.outcome,
        }
        if transition is not None:
            event["transition_id"] = transition.transition_id
        if inspection.outcome == "none":
            write_event(event)
            return {
                **clear_automation_state(),
                "messages": [AIMessage(PRINCIPAL_TRANSITION_FAILURE_LINE)],
            }
        assert transition is not None
        pending_identity = state.pending_identity
        identity_matches = bool(
            isinstance(pending_identity, PendingIdentity)
            and pending_identity.customer_ref == transition.customer_ref
            and pending_identity.masked_contact == transition.masked_contact
        )
        request_matches = (
            state.pending_request == transition.continuation
            if transition.continuation is not None
            else isinstance(state.pending_request, SwitchAccount)
        )
        if inspection.outcome == "coherent" and identity_matches and request_matches:
            write_event(event)
            update = clear_automation_state()
            if transition.continuation is None:
                update["messages"] = [AIMessage("You're now verified on the new account.")]
            return update
        invalidate_principal_transition(transition.transition_id)
        event["outcome"] = "inconsistent"
        return terminal_result(event)

    def recover(state: ReasoningState) -> dict[str, object]:
        marker = state.pending_recovery
        if not isinstance(marker, PendingRecovery):
            return terminal_result()
        policy = policies().get(marker.origin_node)
        if (
            marker.trigger != "node_exception"
            or marker.origin_node not in handled_nodes()
            or policy is None
            or marker.action != policy.on_exception
        ):
            return terminal_result()

        update = clear_automation_state()
        if marker.action == ExceptionAction.TERMINAL:
            write_event(node_failure_event(marker))
            return {
                **update,
                "automation_terminal": True,
                "messages": [AIMessage(AUTOMATION_TERMINAL_LINE)],
            }
        if marker.action in _COMMERCE_RECONCILE_ACTIONS:
            return reconcile_effect(marker, state)
        if marker.action == ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION:
            return reconcile_principal_transition(state)
        if marker.action == ExceptionAction.SAFE_ABORT:
            line = TURN_FALLBACK_LINE
        elif marker.action == ExceptionAction.CART_REVIEW:
            lines = cart_store.view()
            cart_line = (
                render_cart_line(lines, cart_store.cart_total()) if lines else "Your cart is empty."
            )
            line = f"{cart_line} Please review your cart before trying checkout again."
        else:
            line = _ACTION_LINES.get(marker.action)
            if line is None:
                return terminal_result()
        write_event(node_failure_event(marker))
        return {**update, "messages": [AIMessage(line)]}

    return recover


_NON_PREFIXED_AUTOMATION_FIELDS = frozenset(
    {
        "handover",
        "identity_claim_misses",
        "active_flow",
        "clarification_progress",
    }
)
_PROTECTED_STATE_FIELDS = frozenset({"messages", "automation_terminal"})
_AUTOMATION_STATE_RESET: Mapping[str, object] = MappingProxyType(
    {
        "handover": None,
        "pending_placement": None,
        "pending_refund": None,
        "pending_cancel": None,
        "pending_return": None,
        "pending_profile_change": None,
        "pending_identity": None,
        "pending_request": None,
        "pending_recovery": None,
        "identity_claim_misses": 0,
        "active_flow": None,
        "pending_ack": None,
        "pending_clarification": None,
        "clarification_progress": None,
    }
)


def validate_automation_state_clear(
    state_type: type[ReasoningState] = ReasoningState,
) -> None:
    pending_fields = frozenset(
        name for name in state_type.model_fields if name.startswith("pending_")
    )
    expected = pending_fields | _NON_PREFIXED_AUTOMATION_FIELDS
    reset_fields = frozenset(_AUTOMATION_STATE_RESET)
    protected = reset_fields & _PROTECTED_STATE_FIELDS
    if reset_fields != expected or protected:
        raise RuntimeError(
            "automation-state reset is incomplete or unsafe: "
            f"missing={sorted(expected - reset_fields)!r}, "
            f"unexpected={sorted(reset_fields - expected)!r}, "
            f"protected={sorted(protected)!r}"
        )


def clear_automation_state() -> dict[str, object]:
    return dict(_AUTOMATION_STATE_RESET)
