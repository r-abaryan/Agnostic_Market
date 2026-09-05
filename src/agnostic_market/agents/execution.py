"""Cancellation-safe execution primitives for tracked work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol


class SyncEffectExecutor(Protocol):
    async def __call__[T](self, node_name: str, effect: Callable[[], T]) -> T: ...


async def await_resisting_cancellation[T](
    task: asyncio.Task[T],
) -> tuple[T, asyncio.CancelledError | None]:
    deferred_cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            if deferred_cancellation is None:
                deferred_cancellation = cancellation
    return task.result(), deferred_cancellation
