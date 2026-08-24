"""Shared clarification-liveness accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agnostic_market.agents.clarification import advance_clarification
from agnostic_market.dtos.orchestration import (
    InvocationClarificationOwner,
    RouterClarificationOwner,
)
from agnostic_market.dtos.state import ClarificationLiveness, ReasoningState


@pytest.mark.parametrize(
    ("limit", "admitted_reasks"),
    [
        (0, 0),
        (1, 1),
    ],
)
def test_limit_counts_additional_questions_after_the_initial_one(
    limit: int, admitted_reasks: int, tmp_path: Path
) -> None:
    owner = InvocationClarificationOwner(invocation_id="invocation-1")
    state = ReasoningState()
    initial = advance_clarification(state, owner=owner, max_reasks=limit)
    assert initial.exhausted is False
    assert initial.liveness == ClarificationLiveness(owner=owner, reasks=0)

    liveness = initial.liveness
    for expected in range(1, admitted_reasks + 1):
        state = ReasoningState(clarification_liveness=liveness)
        admitted = advance_clarification(state, owner=owner, max_reasks=limit)
        assert admitted.exhausted is False
        assert admitted.liveness == ClarificationLiveness(owner=owner, reasks=expected)
        liveness = admitted.liveness

    exhausted = advance_clarification(
        ReasoningState(clarification_liveness=liveness),
        owner=owner,
        max_reasks=limit,
    )
    assert exhausted.exhausted is True
    assert exhausted.liveness is None
    events = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        {
            "event": "clarification_exhausted",
            "owner_kind": "invocation",
            "consumed_reasks": admitted_reasks,
            "limit": limit,
        }
    ]


def test_owner_change_starts_a_fresh_engagement_without_spending_a_reask(
    tmp_path: Path,
) -> None:
    initial_owner = InvocationClarificationOwner(invocation_id="invocation-1")
    next_owner = RouterClarificationOwner(clarification_id="router-1")
    state = ReasoningState(
        clarification_liveness=ClarificationLiveness(owner=initial_owner, reasks=1)
    )

    switched = advance_clarification(state, owner=next_owner, max_reasks=0)

    assert switched.exhausted is False
    assert switched.liveness == ClarificationLiveness(owner=next_owner, reasks=0)
    assert not (tmp_path / "telemetry.jsonl").exists()
