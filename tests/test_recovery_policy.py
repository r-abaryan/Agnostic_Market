"""Milestone 6C-a architecture contracts for recovery policy and state clearing."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from langgraph.errors import NodeError
from langgraph.graph import START, StateGraph

from agnostic_market.agents.recovery import (
    RECOVERY_NODE_NAME,
    RECOVERY_TERMINALIZER_NODE_NAME,
    NodePolicyRegistry,
    NodeRecoveryPolicy,
    build_node_error_handler,
    build_recovery_infrastructure_handler,
    clear_automation_state,
    validate_automation_state_clear,
)
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import ReasoningState

_ROOT = Path(__file__).parents[1]
_PRODUCTION_ROOT = _ROOT / "src" / "agnostic_market"
_MUTATORS = frozenset(
    {
        "place_cart",
        "apply_confirmed_mutation",
        "add_item",
        "set_quantity",
        "remove_item",
        "issue_refund",
        "cancel_order",
        "create_return",
        "update_profile",
    }
)


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, names: frozenset[str]) -> None:
        self._names = names
        self._functions: list[str] = []
        self.calls: list[tuple[str | None, str]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in self._names:
            owner = self._functions[-1] if self._functions else None
            self.calls.append((owner, node.func.attr))
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "partial"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and node.args[0].attr in self._names
        ):
            owner = self._functions[-1] if self._functions else None
            self.calls.append((owner, node.args[0].attr))
        self.generic_visit(node)


def _production_calls(names: frozenset[str]) -> set[tuple[str, str | None, str]]:
    found: set[tuple[str, str | None, str]] = set()
    for path in _PRODUCTION_ROOT.rglob("*.py"):
        visitor = _CallVisitor(names)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        relative = path.relative_to(_PRODUCTION_ROOT).as_posix()
        found.update((relative, owner, name) for owner, name in visitor.calls)
    return found


def test_all_production_graph_nodes_use_the_registration_seam() -> None:
    assert _production_calls(frozenset({"add_node"})) == {
        ("agents/recovery.py", "_add_node", "add_node")
    }


def test_production_mutators_remain_in_the_six_reconcile_boundaries() -> None:
    assert _production_calls(_MUTATORS) == {
        ("agents/cart/flow.py", "place_node", "place_cart"),
        ("agents/support/flow.py", "place_node", "issue_refund"),
        ("agents/support/flow.py", "cancel_void_node", "cancel_order"),
        ("agents/support/flow.py", "return_place_node", "create_return"),
        ("agents/support/flow.py", "profile_place_node", "update_profile"),
        ("durability/session_state.py", "apply_cart_mutation", "apply_confirmed_mutation"),
    }


def test_production_authorization_has_no_test_only_grant_path() -> None:
    assert (
        _production_calls(frozenset({"grant_mutation_for_test", "mutation_granted_for_test"}))
        == set()
    )


def test_automation_state_clear_is_total_and_preserves_persistent_state() -> None:
    validate_automation_state_clear()
    assert clear_automation_state() == {
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
        "active_invocation": None,
        "pending_recovery": None,
        "identity_claim_misses": 0,
        "execution_owner": None,
        "pending_ack": None,
        "pending_clarification": None,
        "clarification_liveness": None,
    }
    assert {"messages", "automation_terminal"}.isdisjoint(clear_automation_state())


def test_new_pending_state_fails_the_clear_contract() -> None:
    class ExpandedReasoningState(ReasoningState):
        pending_future_action: str | None = None

    with pytest.raises(RuntimeError, match=r"missing=\['pending_future_action'\]"):
        validate_automation_state_clear(ExpandedReasoningState)


def test_node_registry_rejects_duplicates_and_bypasses() -> None:
    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)
    registry.register(
        "entry",
        lambda _state: {},
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    with pytest.raises(RuntimeError, match="duplicate recovery policy registration"):
        registry.register(
            "entry",
            lambda _state: {},
            ExceptionAction.SAFE_ABORT,
            AbandonmentKind.PURE_ABORT,
        )

    graph.add_node("bypass", lambda _state: {})
    with pytest.raises(RuntimeError, match=r"unregistered=\['bypass'\]"):
        registry.validated_policies()


def test_validated_policy_mapping_is_immutable() -> None:
    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)
    registry.register(
        "entry",
        lambda _state: {},
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    policies = registry.validated_policies()

    with pytest.raises(TypeError):
        policies["entry"] = policies["entry"]  # type: ignore[index]
    with pytest.raises(RuntimeError, match="registry is already finalized"):
        registry.register(
            "late",
            lambda _state: {},
            ExceptionAction.SAFE_ABORT,
            AbandonmentKind.PURE_ABORT,
        )


def test_consent_interrupt_metadata_is_closed_and_immutable() -> None:
    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)
    registry.register(
        "confirm",
        lambda _state: {},
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION,
        AbandonmentKind.LIFECYCLE_SPECIAL,
        consent_interrupt_kind="standard",
    )

    kinds = registry.validated_consent_interrupt_kinds()

    assert kinds == {"confirm": "standard"}
    with pytest.raises(TypeError):
        kinds["confirm"] = "cancel"  # type: ignore[index]


def test_consent_interrupt_requires_lifecycle_special_abandonment() -> None:
    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)

    with pytest.raises(ValueError, match="consent interrupt"):
        registry.register(
            "confirm",
            lambda _state: {},
            ExceptionAction.SAFE_ABORT,
            AbandonmentKind.PURE_ABORT,
            consent_interrupt_kind="standard",
        )


def test_registry_destinations_are_rendering_only() -> None:
    visited: list[str] = []

    def source(_state: ReasoningState) -> dict:
        visited.append("source")
        return {}

    def rendered(_state: ReasoningState) -> dict:
        visited.append("rendered")
        return {}

    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)
    registry.register(
        "source",
        source,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=("rendered",),
    )
    registry.register_infrastructure(
        "rendered",
        rendered,
    )
    graph.add_edge(START, "source")
    registry.validated_policies()

    compiled = graph.compile()
    compiled.invoke({})
    rendered_edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}

    assert graph.nodes["source"].ends == ("rendered",)
    assert not any(source == "source" for source, _target in graph.edges)
    assert "source" not in graph.branches
    assert ("source", "rendered") in rendered_edges
    assert visited == ["source"]


def test_registry_rejects_unknown_state_update_before_langgraph_filters_it() -> None:
    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph)
    registry.register(
        "invalid_update",
        lambda _state: {"unexpected_checkpoint_field": "must not disappear"},
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
    )
    graph.add_edge(START, "invalid_update")
    registry.validated_policies()

    with pytest.raises(ValueError, match="unexpected_checkpoint_field"):
        graph.compile().invoke({})


def test_handler_origin_mismatch_mints_a_marker_the_consumer_must_reject() -> None:
    policy = NodeRecoveryPolicy(
        on_exception=ExceptionAction.SAFE_ABORT,
        on_abandonment=AbandonmentKind.PURE_ABORT,
    )
    handler = build_node_error_handler("model", policy)
    assert handler is not None

    command = handler(
        ReasoningState(),
        NodeError(node="different_node", error=RuntimeError("not persisted")),
    )
    marker = command.update["pending_recovery"]

    assert marker.origin_node == "model"
    assert marker.action == ExceptionAction.TERMINAL
    assert marker.trigger == "node_exception"
    assert "not persisted" not in marker.model_dump_json()


def test_recovery_infrastructure_failure_reaches_terminalizer() -> None:
    def fail(_state: ReasoningState) -> dict:
        raise RuntimeError("failed")

    graph = StateGraph(ReasoningState)
    registry = NodePolicyRegistry(graph, error_handler_factory=build_node_error_handler)
    registry.register(
        "origin",
        fail,
        ExceptionAction.SAFE_ABORT,
        AbandonmentKind.PURE_ABORT,
        destinations=(RECOVERY_NODE_NAME,),
    )
    registry.register_infrastructure(
        RECOVERY_NODE_NAME,
        fail,
        error_handler=build_recovery_infrastructure_handler,
        destinations=(RECOVERY_TERMINALIZER_NODE_NAME,),
    )
    registry.register_infrastructure(
        RECOVERY_TERMINALIZER_NODE_NAME,
        lambda _state: {"automation_terminal": True},
    )
    graph.add_edge(START, "origin")
    registry.validated_policies()

    result = graph.compile().invoke({})

    assert result["automation_terminal"] is True
