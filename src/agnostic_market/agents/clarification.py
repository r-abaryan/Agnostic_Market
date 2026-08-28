"""Shared clarification-liveness accounting for explicit routing owners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps

from agnostic_market.agents.telemetry import write_event
from agnostic_market.dtos.orchestration import InvocationClarificationOwner
from agnostic_market.dtos.state import (
    ClarificationLiveness,
    ClarificationOwner,
    ReasoningState,
)

NodeUpdate = dict[str, object]
AsyncStateNode = Callable[[ReasoningState], Awaitable[NodeUpdate]]


@dataclass(frozen=True)
class ClarificationStep:
    exhausted: bool
    liveness: ClarificationLiveness | None


def invocation_clarification_owner(state: ReasoningState) -> InvocationClarificationOwner:
    """Return the only valid owner for capability-level clarification."""

    invocation = state.active_invocation
    if invocation is None:
        raise ValueError("capability clarification requires an active invocation")
    return InvocationClarificationOwner(invocation_id=invocation.invocation_id)


def advance_clarification(
    state: ReasoningState,
    *,
    owner: ClarificationOwner,
    max_reasks: int,
) -> ClarificationStep:
    """Admit the initial question/re-ask or report that the next one exceeds policy."""

    current = state.clarification_liveness
    if current is None or current.owner != owner:
        return ClarificationStep(
            exhausted=False,
            liveness=ClarificationLiveness(owner=owner, reasks=0),
        )
    if current.reasks >= max_reasks:
        write_event(
            {
                "event": "clarification_exhausted",
                "owner_kind": owner.kind,
                "consumed_reasks": current.reasks,
                "limit": max_reasks,
            }
        )
        return ClarificationStep(exhausted=True, liveness=None)
    return ClarificationStep(
        exhausted=False,
        liveness=current.model_copy(update={"reasks": current.reasks + 1}),
    )


def with_clarification_lifecycle(node: AsyncStateNode) -> AsyncStateNode:
    """Clear progress whenever an assemble result is not another clarification."""

    @wraps(node)
    async def wrapped(state: ReasoningState) -> NodeUpdate:
        update = await node(state)
        if update.get("pending_clarification") is None:
            return {**update, "clarification_liveness": None}
        return update

    return wrapped
