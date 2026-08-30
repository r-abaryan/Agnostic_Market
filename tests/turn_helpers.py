"""Deterministic committed-turn identities for direct ReasoningEngine test harnesses."""

from __future__ import annotations

from weakref import WeakKeyDictionary

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.dtos.events import CommittedTurn, TurnEvent, TurnFacts

TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS = 2.0

_TURN_COUNTS: WeakKeyDictionary[ReasoningEngine, int] = WeakKeyDictionary()


def next_committed_turn(engine: ReasoningEngine, text: str) -> CommittedTurn:
    """Return the next deterministic transport turn owned by this engine harness."""
    sequence = _TURN_COUNTS.get(engine, 0) + 1
    _TURN_COUNTS[engine] = sequence
    return CommittedTurn(
        text=text,
        message_id=f"test-turn-{sequence}",
    )


async def engine_events(
    engine: ReasoningEngine,
    text: str,
    facts: TurnFacts | None = None,
) -> list[TurnEvent]:
    turn = next_committed_turn(engine, text)
    return await committed_turn_events(engine, turn, facts)


async def committed_turn_events(
    engine: ReasoningEngine,
    turn: CommittedTurn,
    facts: TurnFacts | None = None,
) -> list[TurnEvent]:
    """Collect one caller turn while preserving its transport-supplied identity."""
    return [event async for event in engine.stream_turn(turn, facts or TurnFacts())]
