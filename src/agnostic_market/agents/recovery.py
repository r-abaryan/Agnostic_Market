"""Failure-lifecycle contracts shared by graph construction and recovery."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.errors import NodeError
from langgraph.graph import StateGraph
from langgraph.types import Command

from agnostic_market.agents._copy import principal_completion_line
from agnostic_market.agents.telemetry import TelemetryRecorder
from agnostic_market.commerce.cart import CartMutationRecord, CartStore
from agnostic_market.commerce.orders import (
    CancelRecord,
    OrderPort,
    PlacedOrder,
    RefundRecord,
    ReturnRecord,
    cancelled_batch_outcome,
    render_cart_line,
)
from agnostic_market.commerce.profile import ProfileChangeRecord, ProfilePort
from agnostic_market.commerce.receipts import CommittedReceipt, NotCommittedReceipt
from agnostic_market.dtos.orchestration import (
    PrincipalTransitionInspection,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    BatchCancelOutcome,
    PendingCancelBatch,
    PendingCartMutation,
    PendingIdentity,
    PendingPlacement,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    ReasoningState,
    StateSchemaError,
    validate_reasoning_state_keys,
)

logger = logging.getLogger(__name__)

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

    @property
    def on_cancellation(self) -> ExceptionAction:
        if self.on_abandonment == AbandonmentKind.TERMINAL:
            return ExceptionAction.TERMINAL
        return self.on_exception


@dataclass(frozen=True, slots=True)
class CommerceEffectFinishers:
    """The complete flow-owned post-commit projection boundary used by normal and recovery."""

    placement: Callable[[str, PlacedOrder], dict[str, object]]
    cart_mutation: Callable[[CartMutationRecord], dict[str, object]]
    refund: Callable[[str, RefundRecord], dict[str, object]]
    cancel: Callable[[PendingCancelBatch, BatchCancelOutcome, bool], dict[str, object]]
    return_: Callable[[str, ReturnRecord], dict[str, object]]
    profile_change: Callable[[str, ProfileChangeRecord], Awaitable[dict[str, object]]]


class NodeExecutionTracker:
    """Tracks admitted turns and mutable callables through their complete lifetimes.

    Async task cancellation can unwind LangGraph while a synchronous node is still running
    in a worker thread. Wrapping the runnable invocation itself keeps the tracker active
    until that worker actually exits. A separate whole-turn span prevents session teardown
    from observing the zero-count gaps between nodes. The policy registry remains the sole
    mutable-node wrapper owner.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_by_node: dict[str, int] = {}
        self._active_turns = 0
        self._turn_admission_open = True
        self._tracked_node_names: set[str] = set()
        self._fully_idle_callbacks: list[Callable[[], None]] = []

    @property
    def active_count(self) -> int:
        with self._condition:
            return sum(self._active_by_node.values())

    @property
    def active_node_names(self) -> frozenset[str]:
        with self._condition:
            return frozenset(self._active_by_node)

    @property
    def turn_admission_open(self) -> bool:
        with self._condition:
            return self._turn_admission_open

    @property
    def tracked_node_names(self) -> frozenset[str]:
        with self._condition:
            return frozenset(self._tracked_node_names)

    @staticmethod
    def _run_callbacks(callbacks: tuple[Callable[[], None], ...]) -> None:
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.critical("deferred quiescence callback failed", exc_info=True)

    def _fully_idle_callbacks_locked(self) -> tuple[Callable[[], None], ...]:
        full_callbacks: tuple[Callable[[], None], ...] = ()
        if not self._active_by_node:
            if not self._active_turns:
                full_callbacks = tuple(self._fully_idle_callbacks)
                self._fully_idle_callbacks.clear()
            self._condition.notify_all()
        return full_callbacks

    @contextmanager
    def _node_span(self, node_name: str) -> Iterator[None]:
        with self._condition:
            self._active_by_node[node_name] = self._active_by_node.get(node_name, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                remaining = self._active_by_node[node_name] - 1
                if remaining:
                    self._active_by_node[node_name] = remaining
                else:
                    del self._active_by_node[node_name]
                full_callbacks = self._fully_idle_callbacks_locked()
            self._run_callbacks(full_callbacks)

    def _run(
        self,
        node_name: str,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._node_span(node_name):
            return operation(*args, **kwargs)

    async def _arun(
        self,
        node_name: str,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._node_span(node_name):
            return await operation(*args, **kwargs)

    @contextmanager
    def turn_span(self) -> Iterator[bool]:
        with self._condition:
            admitted = self._turn_admission_open
            if admitted:
                self._active_turns += 1
        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._active_turns -= 1
                    full_callbacks = self._fully_idle_callbacks_locked()
                self._run_callbacks(full_callbacks)

    def stop_turn_admission(self) -> None:
        with self._condition:
            self._turn_admission_open = False

    def wrap(self, node_name: str, node: object) -> Callable[[object, RunnableConfig], Any]:
        with self._condition:
            if node_name in self._tracked_node_names:
                raise RuntimeError(f"node is already tracked: {node_name!r}")
            self._tracked_node_names.add(node_name)
        runnable = node if isinstance(node, Runnable) else RunnableLambda(node)

        if _is_async_callable(node):

            async def arun(state: object, config: RunnableConfig) -> Any:
                return await self._arun(node_name, runnable.ainvoke, state, config)

            return arun

        def run(state: object, config: RunnableConfig) -> Any:
            return self._run(node_name, runnable.invoke, state, config)

        return run

    def wait_until_mutable_idle(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("quiescence timeout must be positive")
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._active_by_node,
                timeout=timeout_seconds,
            )

    def defer_until_fully_idle(self, callback: Callable[[], None]) -> None:
        run_now = False
        with self._condition:
            if self._active_turns or self._active_by_node:
                self._fully_idle_callbacks.append(callback)
            else:
                run_now = True
        if run_now:
            callback()


NodeErrorHandler = Callable[[ReasoningState, NodeError], Command]
NodeErrorHandlerFactory = Callable[[str, NodeRecoveryPolicy], NodeErrorHandler | None]


def _validate_node_result(node_name: str, result: object) -> object:
    update = result.update if isinstance(result, Command) else result
    if update is None:
        return result
    if not isinstance(update, Mapping):
        raise StateSchemaError(f"node {node_name!r} returned a non-mapping state update")
    validate_reasoning_state_keys(update, source=f"node {node_name!r}")
    return result


def _is_async_callable(value: object) -> bool:
    return inspect.iscoroutinefunction(value)


def _wrap_state_node(node_name: str, node: object) -> Callable[[object, RunnableConfig], Any]:
    runnable = node if isinstance(node, Runnable) else RunnableLambda(node)

    if _is_async_callable(node):

        async def arun(state: object, config: RunnableConfig) -> Any:
            result = await runnable.ainvoke(state, config)
            return _validate_node_result(node_name, result)

        return arun

    def run(state: object, config: RunnableConfig) -> Any:
        return _validate_node_result(node_name, runnable.invoke(state, config))

    return run


class _ModelNodeExecutionBoundary:
    """Apply one application deadline to a complete native-async model node."""

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("model node timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def wrap(self, node_name: str, node: object) -> Callable[[object, RunnableConfig], Any]:
        if not _is_async_callable(node):
            raise TypeError(f"model node {node_name!r} must be native async")
        runnable = node if isinstance(node, Runnable) else RunnableLambda(node)

        async def run(state: object, config: RunnableConfig) -> Any:
            async with asyncio.timeout(self._timeout_seconds):
                return await runnable.ainvoke(state, config)

        return run


def _wrap_error_handler(node_name: str, handler: NodeErrorHandler) -> NodeErrorHandler:
    def run(state: ReasoningState, error: NodeError) -> Command:
        command = handler(state, error)
        _validate_node_result(node_name, command)
        return command

    return run


class NodePolicyRegistry:
    """The only production seam for registering reasoning-graph nodes."""

    def __init__(
        self,
        graph: StateGraph,
        *,
        error_handler_factory: NodeErrorHandlerFactory | None = None,
        model_node_timeouts: Mapping[str, float] | None = None,
    ) -> None:
        resolved_model_timeouts = dict(model_node_timeouts or {})
        model_node_names = frozenset(resolved_model_timeouts)
        if any(not name for name in model_node_names):
            raise ValueError("model node names must be non-empty")
        if any(timeout <= 0 for timeout in resolved_model_timeouts.values()):
            raise ValueError("model node timeouts must be positive")
        self._graph = graph
        self._error_handler_factory = error_handler_factory
        self._model_node_names = model_node_names
        self._model_boundaries = {
            name: _ModelNodeExecutionBoundary(timeout)
            for name, timeout in resolved_model_timeouts.items()
        }
        self._policies: dict[str, NodeRecoveryPolicy] = {}
        self._infrastructure: set[str] = set()
        self._handled: set[str] = set()
        self._handled_infrastructure: set[str] = set()
        self._consent_interrupt_kinds: dict[str, Literal["standard", "cancel"]] = {}
        self._execution_tracker = NodeExecutionTracker()
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
        destinations: tuple[str, ...] | None = None,
    ) -> None:
        options: dict[str, object] = {}
        if error_handler is not None:
            options["error_handler"] = _wrap_error_handler(name, error_handler)
        if destinations is not None:
            options["destinations"] = destinations
        self._graph.add_node(name, _wrap_state_node(name, node), **options)

    def register(
        self,
        name: str,
        node: object,
        on_exception: ExceptionAction,
        on_abandonment: AbandonmentKind,
        *,
        consent_interrupt_kind: Literal["standard", "cancel"] | None = None,
        destinations: tuple[str, ...] | None = None,
    ) -> None:
        self._ensure_open_and_unique(name)
        if (
            consent_interrupt_kind is not None
            and on_abandonment != AbandonmentKind.LIFECYCLE_SPECIAL
        ):
            raise ValueError("a consent interrupt must use lifecycle-special abandonment")
        policy = NodeRecoveryPolicy(
            on_exception=on_exception,
            on_abandonment=on_abandonment,
        )
        handler = (
            self._error_handler_factory(name, policy)
            if self._error_handler_factory is not None
            else None
        )
        boundary = self._model_boundaries.get(name)
        bounded_node = boundary.wrap(name, node) if boundary is not None else node
        registered_node = (
            bounded_node
            if on_abandonment == AbandonmentKind.PURE_ABORT
            else self._execution_tracker.wrap(name, bounded_node)
        )
        self._add_node(
            name,
            registered_node,
            error_handler=handler,
            destinations=destinations,
        )
        self._policies[name] = policy
        if consent_interrupt_kind is not None:
            self._consent_interrupt_kinds[name] = consent_interrupt_kind
        if handler is not None:
            self._handled.add(name)

    def register_infrastructure(
        self,
        name: str,
        node: object,
        *,
        error_handler: NodeErrorHandler | None = None,
        destinations: tuple[str, ...] | None = None,
    ) -> None:
        self._ensure_open_and_unique(name)
        self._add_node(
            name,
            node,
            error_handler=error_handler,
            destinations=destinations,
        )
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
        expected_tracked = frozenset(
            name
            for name, policy in self._policies.items()
            if policy.on_abandonment != AbandonmentKind.PURE_ABORT
        )
        actual_tracked = self._execution_tracker.tracked_node_names
        expected = registered | infrastructure
        missing_model_nodes = self._model_node_names - registered
        if (
            expected != actual_regular
            or missing_handler_links
            or expected_handlers != actual_handlers
            or expected_tracked != actual_tracked
            or missing_model_nodes
        ):
            raise RuntimeError(
                "graph nodes bypassed the recovery-policy registry: "
                f"unregistered={sorted(actual_regular - expected)!r}, "
                f"missing={sorted(expected - actual_regular)!r}, "
                f"missing_handler_links={sorted(missing_handler_links)!r}, "
                f"unexpected_handlers={sorted(actual_handlers - expected_handlers)!r}, "
                f"missing_handlers={sorted(expected_handlers - actual_handlers)!r}, "
                f"untracked_mutable={sorted(expected_tracked - actual_tracked)!r}, "
                f"unexpected_tracked={sorted(actual_tracked - expected_tracked)!r}, "
                f"missing_model_nodes={sorted(missing_model_nodes)!r}"
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

    def validated_execution_tracker(self) -> NodeExecutionTracker:
        self.validated_policies()
        return self._execution_tracker

    def validated_consent_interrupt_kinds(
        self,
    ) -> Mapping[str, Literal["standard", "cancel"]]:
        self.validated_policies()
        return MappingProxyType(dict(self._consent_interrupt_kinds))


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
    error: NodeError,
) -> Command:
    del error
    return Command(goto=RECOVERY_TERMINALIZER_NODE_NAME)


def build_recovery_terminalizer(_state: ReasoningState) -> dict[str, object]:
    return {
        **clear_automation_state(),
        "automation_terminal": True,
    }


def build_recovery_node(
    policies: Callable[[], Mapping[str, NodeRecoveryPolicy]],
    handled_nodes: Callable[[], frozenset[str]],
    safe_abort_continue_node: str,
    cart_store: CartStore,
    order_store: OrderPort,
    profile_store: ProfilePort,
    finishers: CommerceEffectFinishers,
    inspect_principal_transition: Callable[[], PrincipalTransitionInspection],
    invalidate_principal_transition: Callable[[str | None], Awaitable[bool]],
    telemetry: TelemetryRecorder,
) -> Callable[[ReasoningState], Awaitable[dict[str, object] | Command]]:
    def node_failure_event(marker: PendingRecovery) -> dict[str, object]:
        return {
            "event": "turn_failed",
            "reason": marker.trigger,
            "node": marker.origin_node,
            "action": marker.action,
        }

    def terminal_result(event: dict[str, object] | None = None) -> dict[str, object]:
        telemetry.record(
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
        telemetry.record(node_failure_event(marker))
        return {**clear_automation_state(), "messages": [AIMessage(line)]}

    async def reconcile_effect(
        marker: PendingRecovery,
        state: ReasoningState,
    ) -> dict[str, object]:
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
                telemetry.record(node_failure_event(marker))
                return complete(finishers.placement(pending.idempotency_key, receipt.record))
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
                telemetry.record(node_failure_event(marker))
                return complete(finishers.refund(pending.idempotency_key, receipt.record))
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
            telemetry.record(node_failure_event(marker))
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
                telemetry.record(node_failure_event(marker))
                return complete(finishers.return_(pending.idempotency_key, receipt.record))
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
                telemetry.record(node_failure_event(marker))
                return complete(
                    await finishers.profile_change(pending.idempotency_key, receipt.record)
                )
            if isinstance(receipt, NotCommittedReceipt):
                return not_committed(marker)
            return terminal_result(node_failure_event(marker))

        return terminal_result(node_failure_event(marker))

    async def reconcile_principal_transition(
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
            telemetry.record(event)
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
        invocation = state.active_invocation
        request_matches = bool(
            invocation is not None and invocation.request == transition.initiating_request
        )
        if inspection.outcome == "coherent" and identity_matches and request_matches:
            telemetry.record(event)
            update = clear_automation_state()
            completion_line = principal_completion_line(transition.projection.completion_kind)
            if completion_line is not None:
                update["messages"] = [AIMessage(completion_line)]
            return update
        await invalidate_principal_transition(transition.transition_id)
        event["outcome"] = "inconsistent"
        return terminal_result(event)

    async def recover(state: ReasoningState) -> dict[str, object] | Command:
        marker = state.pending_recovery
        if not isinstance(marker, PendingRecovery):
            return terminal_result()
        policy = policies().get(marker.origin_node)
        node_exception_valid = bool(
            marker.trigger == "node_exception"
            and marker.origin_node in handled_nodes()
            and policy is not None
            and marker.action == policy.on_exception
        )
        stream_cancellation_valid = bool(
            marker.trigger == "stream_cancelled"
            and policy is not None
            and marker.action == policy.on_cancellation
            and marker.abandoned_message_id in state.consumed_turn_ids
        )
        if not (node_exception_valid or stream_cancellation_valid):
            return terminal_result()

        update = clear_automation_state()
        if marker.action == ExceptionAction.TERMINAL:
            telemetry.record(node_failure_event(marker))
            return {
                **update,
                "automation_terminal": True,
                "messages": [AIMessage(AUTOMATION_TERMINAL_LINE)],
            }
        if marker.action in _COMMERCE_RECONCILE_ACTIONS:
            return await reconcile_effect(marker, state)
        if marker.action == ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION:
            return await reconcile_principal_transition(state)
        if marker.action == ExceptionAction.SAFE_ABORT:
            if marker.trigger == "stream_cancelled":
                turn_ids = state.consumed_turn_ids
                latest_message = state.messages[-1] if state.messages else None
                distinct_turn_admitted = bool(
                    turn_ids
                    and turn_ids[-1] != marker.abandoned_message_id
                    and isinstance(latest_message, HumanMessage)
                    and latest_message.id == turn_ids[-1]
                )
                if distinct_turn_admitted:
                    telemetry.record(node_failure_event(marker))
                    return Command(update=update, goto=safe_abort_continue_node)
            line = TURN_FALLBACK_LINE
        elif marker.action == ExceptionAction.CART_REVIEW:
            pending_mutation = state.pending_cart_mutation
            if isinstance(pending_mutation, PendingCartMutation):
                receipt = cart_store.mutation_receipt(
                    pending_mutation.idempotency_key,
                    operation=pending_mutation.operation,
                    sku=pending_mutation.sku,
                    name=pending_mutation.name,
                    price_usd=pending_mutation.price_usd,
                    quantity=pending_mutation.quantity,
                    pre_confirm_quantity=pending_mutation.pre_confirm_quantity,
                )
                if isinstance(receipt, CommittedReceipt) and isinstance(
                    receipt.record, CartMutationRecord
                ):
                    telemetry.record(node_failure_event(marker))
                    return complete(finishers.cart_mutation(receipt.record))
                if not isinstance(receipt, NotCommittedReceipt):
                    return terminal_result(node_failure_event(marker))
            lines = cart_store.view()
            cart_line = (
                render_cart_line(lines, cart_store.cart_total()) if lines else "Your cart is empty."
            )
            line = f"{cart_line} Please review your cart before trying again."
        else:
            line = _ACTION_LINES.get(marker.action)
            if line is None:
                return terminal_result()
        telemetry.record(node_failure_event(marker))
        return {**update, "messages": [AIMessage(line)]}

    return recover


_NON_PREFIXED_AUTOMATION_FIELDS = frozenset(
    {
        "active_invocation",
        "handover",
        "identity_claim_misses",
        "execution_owner",
        "clarification_liveness",
    }
)
_PROTECTED_STATE_FIELDS = frozenset({"messages", "automation_terminal"})
_AUTOMATION_STATE_RESET: Mapping[str, object] = MappingProxyType(
    {
        "handover": None,
        "pending_capability_dispatch": None,
        "pending_router_no_action": None,
        "pending_cart_mutation": None,
        "pending_placement": None,
        "pending_refund": None,
        "pending_cancel": None,
        "pending_return": None,
        "pending_profile_change": None,
        "pending_identity": None,
        "pending_recovery": None,
        "active_invocation": None,
        "identity_claim_misses": 0,
        "execution_owner": None,
        "pending_ack": None,
        "pending_clarification": None,
        "clarification_liveness": None,
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
