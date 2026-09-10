"""Durable timing must observe work without becoming execution authority."""

from __future__ import annotations

import asyncio

import pytest

from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingOutcome,
    InMemoryDurabilityTimingObserver,
    observe_async_operation,
    observe_duration,
)


class _MeasuredOperation:
    def __init__(self, observer: object | None) -> None:
        self._durability_timing = observer

    @observe_async_operation(DurabilityOperation.REGISTRY_GET)
    async def run(self, *, fail: bool = False) -> str:
        if fail:
            raise RuntimeError("operation failed")
        return "completed"


async def test_async_timing_records_success_and_failure_without_changing_outcomes() -> None:
    observer = InMemoryDurabilityTimingObserver()
    operation = _MeasuredOperation(observer)

    assert await operation.run() == "completed"
    with pytest.raises(RuntimeError, match="operation failed"):
        await operation.run(fail=True)

    assert [sample.operation for sample in observer.samples] == [
        DurabilityOperation.REGISTRY_GET,
        DurabilityOperation.REGISTRY_GET,
    ]
    assert [sample.outcome for sample in observer.samples] == [
        DurabilityTimingOutcome.SUCCESS,
        DurabilityTimingOutcome.ERROR,
    ]
    assert all(sample.elapsed_seconds >= 0 for sample in observer.samples)


def test_disabled_timing_does_not_suppress_the_observed_failure() -> None:
    with (
        pytest.raises(RuntimeError, match="operation failed"),
        observe_duration(None, DurabilityOperation.REGISTRY_GET),
    ):
        raise RuntimeError("operation failed")


async def test_observer_failure_does_not_replace_the_operation_result() -> None:
    class FailingObserver:
        def observe(self, _sample: object) -> None:
            raise RuntimeError("observer failed")

    assert await _MeasuredOperation(FailingObserver()).run() == "completed"


async def test_cancelled_operation_is_measured_without_consuming_cancellation() -> None:
    observer = InMemoryDurabilityTimingObserver()

    class BlockingOperation:
        def __init__(self) -> None:
            self._durability_timing = observer
            self.entered = asyncio.Event()

        @observe_async_operation(DurabilityOperation.REGISTRY_GET)
        async def run(self) -> None:
            self.entered.set()
            await asyncio.Event().wait()

    operation = BlockingOperation()
    task = asyncio.create_task(operation.run())
    await operation.entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert observer.samples[-1].outcome is DurabilityTimingOutcome.CANCELLED
