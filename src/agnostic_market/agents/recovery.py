"""Failure-lifecycle contracts shared by graph construction and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from langgraph.graph import StateGraph

from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import ReasoningState


@dataclass(frozen=True, slots=True)
class NodeRecoveryPolicy:
    on_exception: ExceptionAction
    on_abandonment: AbandonmentKind


class NodePolicyRegistry:
    """The only production seam for registering regular reasoning-graph nodes."""

    def __init__(self, graph: StateGraph) -> None:
        self._graph = graph
        self._policies: dict[str, NodeRecoveryPolicy] = {}
        self._validated: Mapping[str, NodeRecoveryPolicy] | None = None

    def register(
        self,
        name: str,
        node: object,
        on_exception: ExceptionAction,
        on_abandonment: AbandonmentKind,
    ) -> None:
        if self._validated is not None:
            raise RuntimeError("recovery policy registry is already finalized")
        if name in self._policies:
            raise RuntimeError(f"duplicate recovery policy registration: {name!r}")
        self._graph.add_node(name, node)
        self._policies[name] = NodeRecoveryPolicy(
            on_exception=on_exception,
            on_abandonment=on_abandonment,
        )

    def validated_policies(self) -> Mapping[str, NodeRecoveryPolicy]:
        registered = frozenset(self._policies)
        actual = frozenset(self._graph.nodes)
        if registered != actual:
            raise RuntimeError(
                "regular graph nodes bypassed the recovery-policy registry: "
                f"unregistered={sorted(actual - registered)!r}, "
                f"missing={sorted(registered - actual)!r}"
            )
        if self._validated is None:
            self._validated = MappingProxyType(dict(self._policies))
        return self._validated


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
