"""Shared clarification-liveness accounting contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agnostic_market.agents.clarification import advance_clarification
from agnostic_market.dtos.state import ClarificationProgress, ReasoningState


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
    state = ReasoningState()
    initial = advance_clarification(state, flow="identity", max_reasks=limit)
    assert initial.exhausted is False
    assert initial.progress == ClarificationProgress(flow="identity", reasks=0)

    progress = initial.progress
    for expected in range(1, admitted_reasks + 1):
        state = ReasoningState(clarification_progress=progress)
        admitted = advance_clarification(state, flow="identity", max_reasks=limit)
        assert admitted.exhausted is False
        assert admitted.progress == ClarificationProgress(flow="identity", reasks=expected)
        progress = admitted.progress

    exhausted = advance_clarification(
        ReasoningState(clarification_progress=progress),
        flow="identity",
        max_reasks=limit,
    )
    assert exhausted.exhausted is True
    assert exhausted.progress is None
    events = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events == [
        {
            "event": "clarification_exhausted",
            "flow": "identity",
            "consumed_reasks": admitted_reasks,
            "limit": limit,
        }
    ]


def test_owner_change_starts_a_fresh_engagement_without_spending_a_reask(
    tmp_path: Path,
) -> None:
    state = ReasoningState(clarification_progress=ClarificationProgress(flow="identity", reasks=1))

    switched = advance_clarification(state, flow="cart", max_reasks=0)

    assert switched.exhausted is False
    assert switched.progress == ClarificationProgress(flow="cart", reasks=0)
    assert not (tmp_path / "telemetry.jsonl").exists()
