"""Typed timing observations for durable runtime certification."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar

logger = logging.getLogger("agnostic_market.durability.timing")

P = ParamSpec("P")
R = TypeVar("R")


class DurabilityOperation(StrEnum):
    POOL_OPEN = "pool.open"
    STARTUP_GATES = "platform.startup_gates"
    REGISTRY_REGISTER = "registry.register_and_acquire"
    REGISTRY_ACTIVATE = "registry.activate"
    REGISTRY_RENEW = "registry.renew"
    REGISTRY_RESTORE = "registry.restore"
    REGISTRY_RECONCILE = "registry.reconcile_checkpoint_revision"
    REGISTRY_PUBLISH = "registry.publish"
    REGISTRY_CLOSE = "registry.close"
    REGISTRY_REAP = "registry.reap"
    REGISTRY_ROTATE = "registry.rotate"
    REGISTRY_GET = "registry.get"
    CHECKPOINT_READ = "checkpoint.read"
    CHECKPOINT_LIST = "checkpoint.list"
    CHECKPOINT_WRITE = "checkpoint.write"
    CHECKPOINT_PENDING_WRITE = "checkpoint.pending_write"
    CHECKPOINT_DELETE = "checkpoint.delete"
    CHECKPOINT_BIND = "checkpoint.bind"
    SESSION_STORE_PUBLISH = "session_store.publish"
    ENVELOPE_ENCRYPT = "envelope.encrypt"
    ENVELOPE_DECRYPT = "envelope.decrypt"


class DurabilityTimingOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DurabilityTimingSample:
    operation: DurabilityOperation
    elapsed_seconds: float
    outcome: DurabilityTimingOutcome


class DurabilityTimingObserver(Protocol):
    def observe(self, sample: DurabilityTimingSample) -> None: ...


class InMemoryDurabilityTimingObserver:
    """Thread-safe sample collector for deterministic and deployment certification."""

    def __init__(self) -> None:
        self._samples: list[DurabilityTimingSample] = []
        self._lock = threading.Lock()

    def observe(self, sample: DurabilityTimingSample) -> None:
        with self._lock:
            self._samples.append(sample)

    @property
    def samples(self) -> tuple[DurabilityTimingSample, ...]:
        with self._lock:
            return tuple(self._samples)


@contextmanager
def observe_duration(
    observer: DurabilityTimingObserver | None,
    operation: DurabilityOperation,
) -> Iterator[None]:
    started = time.perf_counter()
    outcome = DurabilityTimingOutcome.SUCCESS
    try:
        yield
    except asyncio.CancelledError:
        outcome = DurabilityTimingOutcome.CANCELLED
        raise
    except BaseException:
        outcome = DurabilityTimingOutcome.ERROR
        raise
    finally:
        if observer is not None:
            sample = DurabilityTimingSample(
                operation=operation,
                elapsed_seconds=time.perf_counter() - started,
                outcome=outcome,
            )
            try:
                observer.observe(sample)
            except Exception:
                logger.exception("durability timing observer failed")


def observe_async_operation(
    operation: DurabilityOperation,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Measure an async owner method without changing its outcome."""

    def decorate(method: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(method)
        async def observed(*args: P.args, **kwargs: P.kwargs) -> R:
            observer = getattr(args[0], "_durability_timing", None) if args else None
            with observe_duration(observer, operation):
                return await method(*args, **kwargs)

        return observed

    return decorate
