"""Behavior contracts for the concrete durable latency probe."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from agnostic_market.agents.engine import GraphTurnLatencyMeasurement
from agnostic_market.commerce.cart import CartStore
from agnostic_market.dtos.events import InterruptEvent
from agnostic_market.durability.latency import LatencyCartLine, LatencyJourneyContract
from agnostic_market.durability.latency_probe import (
    DeploymentLatencyProbeError,
    _OpenProbeExecution,
    _prepare_journey,
    _TurnProbeExecution,
)


class _JourneyEngine:
    def __init__(
        self,
        cart: CartStore,
        measurements: list[GraphTurnLatencyMeasurement],
    ) -> None:
        self._cart = cart
        self._measurements = measurements
        self.turns: list[str] = []

    async def stream_turn(self, turn, _facts):
        self.turns.append(turn.text)
        if turn.text == "prepare cart":
            self._cart.add_item(
                sku="SKU-BLU-07",
                name="waterproof rain jacket",
                price_usd=129,
                quantity=1,
            )
        self._measurements.append(
            GraphTurnLatencyMeasurement(
                total_seconds=0.25,
                time_to_first_model_seconds=0.1,
                tool_count=1,
                tool_to_next_model_seconds=None,
            )
        )
        if turn.text == "place my order":
            yield InterruptEvent(prompt="Confirm the $129.00 order")


def _execution() -> _OpenProbeExecution:
    cart = CartStore()
    measurements: list[GraphTurnLatencyMeasurement] = []
    engine = _JourneyEngine(cart, measurements)
    loop = SimpleNamespace(
        engine=engine,
        application=SimpleNamespace(state=SimpleNamespace(cart_store=cart)),
    )
    services = SimpleNamespace(order_store=SimpleNamespace(placed_count=0))
    return _OpenProbeExecution(
        loop=cast(Any, loop),
        platform_resources=cast(Any, SimpleNamespace()),
        services=cast(Any, services),
        measurements=measurements,
        startup_elapsed_seconds=0.5,
    )


async def test_turn_probe_prepares_state_outside_the_measured_turn() -> None:
    contract = LatencyJourneyContract(
        journey_id="checkout-placement-readback",
        setup_turns=("prepare cart",),
        utterance="place my order",
        initial_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
        expected_cart=(LatencyCartLine(sku="SKU-BLU-07", quantity=1),),
        expected_event_kind="interrupt",
        expected_event_text_contains=("$129.00",),
    )
    execution = _execution()

    await _prepare_journey(contract, execution)
    elapsed = await _TurnProbeExecution(execution, contract, "sample-1").run()

    assert cast(Any, execution.loop).engine.turns == ["prepare cart", "place my order"]
    assert len(execution.measurements) == 1
    assert elapsed == 0.25


async def test_turn_probe_rejects_a_fast_but_wrong_outcome() -> None:
    contract = LatencyJourneyContract(
        journey_id="wrong-output",
        utterance="place my order",
        initial_cart=(),
        expected_cart=(),
        expected_event_kind="interrupt",
        expected_event_text_contains=("expected phrase",),
    )
    execution = _execution()

    with pytest.raises(DeploymentLatencyProbeError, match="frozen contract"):
        await _TurnProbeExecution(execution, contract, "sample-2").run()
