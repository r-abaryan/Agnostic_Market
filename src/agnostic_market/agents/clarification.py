"""Shared clarification-liveness accounting for sticky transactional flows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

from agnostic_market.agents.telemetry import write_event
from agnostic_market.dtos.state import (
    ClarificationOwner,
    ClarificationProgress,
    ReasoningState,
)

NodeUpdate = dict[str, object]
StateNode = Callable[[ReasoningState], NodeUpdate]


@dataclass(frozen=True)
class ClarificationStep:
    exhausted: bool
    progress: ClarificationProgress | None


def advance_clarification(
    state: ReasoningState,
    *,
    flow: ClarificationOwner,
    max_reasks: int,
) -> ClarificationStep:
    """Admit the initial question/re-ask or report that the next one exceeds policy."""

    current = state.clarification_progress
    if current is None or current.flow != flow:
        return ClarificationStep(
            exhausted=False,
            progress=ClarificationProgress(flow=flow, reasks=0),
        )
    if current.reasks >= max_reasks:
        write_event(
            {
                "event": "clarification_exhausted",
                "flow": flow,
                "consumed_reasks": current.reasks,
                "limit": max_reasks,
            }
        )
        return ClarificationStep(exhausted=True, progress=None)
    return ClarificationStep(
        exhausted=False,
        progress=current.model_copy(update={"reasks": current.reasks + 1}),
    )


def with_clarification_lifecycle(node: StateNode) -> StateNode:
    """Clear progress whenever an assemble result is not another clarification."""

    @wraps(node)
    def wrapped(state: ReasoningState) -> NodeUpdate:
        update = node(state)
        if update.get("pending_clarification") is None:
            return {**update, "clarification_progress": None}
        return update

    return wrapped
