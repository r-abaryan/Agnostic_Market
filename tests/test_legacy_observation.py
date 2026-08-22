from __future__ import annotations

import pytest

from agnostic_market.agents.legacy_observation import (
    LEGACY_HANDOVER_CAPABILITIES,
    legacy_route_observation_scope,
    observe_legacy_answer,
    observe_legacy_capabilities,
    observe_legacy_model_tool_calls,
)
from agnostic_market.dtos.orchestration import CapabilityId


def test_legacy_observation_is_closed_deduplicated_and_redacted() -> None:
    with legacy_route_observation_scope("turn-1") as observation:
        observe_legacy_capabilities(LEGACY_HANDOVER_CAPABILITIES["cart_write"], source="gate")
        observe_legacy_model_tool_calls(
            (
                {"name": "go_to_checkout", "args": {"caller_value": "secret"}},
                {"name": "go_to_checkout"},
                {"name": "unknown_tool"},
            )
        )
        observe_legacy_answer(source="tool")

    assert observation.as_event() == {
        "event": "semantic_route_legacy_observation",
        "turn_id": "turn-1",
        "decision_source": "legacy",
        "capability_candidates": ["modify_cart", "place_order"],
        "owner_sources": ["gate", "model"],
        "answer_sources": ["tool"],
        "answered": True,
        "non_tool_ai_text_without_owner": False,
    }
    assert "secret" not in repr(observation.as_event())


def test_model_answer_without_an_owner_is_distinguished_from_no_action() -> None:
    observe_legacy_capabilities((CapabilityId.CANCEL_ORDERS,), source="gate")
    observe_legacy_answer(source="model")

    with legacy_route_observation_scope("answer-turn") as answer_observation:
        observe_legacy_answer(source="model")
    with legacy_route_observation_scope("no-action-turn") as no_action_observation:
        pass

    assert answer_observation.as_event()["non_tool_ai_text_without_owner"] is True
    assert no_action_observation.as_event()["answered"] is False
    assert no_action_observation.as_event()["capability_candidates"] == []


def test_legacy_observation_scope_rejects_invalid_or_nested_ownership() -> None:
    with pytest.raises(ValueError, match="nonblank"), legacy_route_observation_scope(" "):
        pass

    with (
        legacy_route_observation_scope("outer"),
        pytest.raises(RuntimeError, match="cannot be nested"),
        legacy_route_observation_scope("inner"),
    ):
        pass
