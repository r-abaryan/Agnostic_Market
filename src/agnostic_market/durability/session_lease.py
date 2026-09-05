"""Process-local supervision for one authoritative session lease."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from agnostic_market.durability.session_registry import (
    LeaseAdmissionError,
    LeaseAdmissionReason,
    SessionLeaseRenewal,
    SessionRegistryRecord,
)

logger = logging.getLogger(__name__)


class LeaseRenewalPort(Protocol):
    async def renew(self, renewal: SessionLeaseRenewal) -> SessionRegistryRecord: ...


class LeaseLossReason(StrEnum):
    REJECTED = "rejected"
    LEASE_DEADLINE = "lease_deadline"
    RENEWAL_TIMEOUT = "renewal_timeout"
    RENEWAL_FAILED = "renewal_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class LeaseLoss:
    reason: LeaseLossReason
    rejection: LeaseAdmissionReason | None = None

    def __post_init__(self) -> None:
        if (self.reason is LeaseLossReason.REJECTED) != (self.rejection is not None):
            raise ValueError("only rejected lease loss carries a rejection reason")


type LeaseLossCallback = Callable[[LeaseLoss], Awaitable[None] | None]


class SessionLeaseSupervisor:
    """Renew one lease and report authority loss exactly once."""

    def __init__(
        self,
        registry: LeaseRenewalPort,
        renewal: SessionLeaseRenewal,
        *,
        renewal_interval_seconds: float,
        on_lease_lost: LeaseLossCallback,
    ) -> None:
        if (
            not math.isfinite(renewal_interval_seconds)
            or renewal_interval_seconds <= 0
            or renewal_interval_seconds >= renewal.duration_seconds
        ):
            raise ValueError("renewal interval must be positive and shorter than the lease")
        self._registry = registry
        self._renewal = renewal
        self._renewal_interval_seconds = renewal_interval_seconds
        self._on_lease_lost = on_lease_lost
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._loss: LeaseLoss | None = None

    @property
    def loss(self) -> LeaseLoss | None:
        return self._loss

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("session lease supervisor is already started")
        self._task = asyncio.create_task(self._run(), name="session-lease-supervisor")
        self._task.add_done_callback(self._observe_completion)

    async def wait(self) -> None:
        if self._task is None:
            raise RuntimeError("session lease supervisor has not started")
        await asyncio.shield(self._task)

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                if not self._task.cancelled():
                    raise

    async def _report_loss(self, loss: LeaseLoss) -> None:
        if self._loss is not None:
            return
        self._loss = loss
        result = self._on_lease_lost(loss)
        if inspect.isawaitable(result):
            await result

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    loop = asyncio.get_running_loop()
                    started_at = loop.time()
                    record = await self._registry.renew(self._renewal)
                    request_elapsed = loop.time() - started_at
                    if record.lease_expires_at is None:
                        raise ValueError("renewed session has no lease expiry")
                    authoritative_remaining = (
                        record.lease_expires_at - record.updated_at
                    ).total_seconds() - request_elapsed
                    renewal_delay = min(
                        self._renewal_interval_seconds,
                        authoritative_remaining - self._renewal_interval_seconds,
                    )
                except LeaseAdmissionError as exc:
                    await self._report_loss(
                        LeaseLoss(
                            reason=LeaseLossReason.REJECTED,
                            rejection=exc.reason,
                        )
                    )
                    return
                except TimeoutError:
                    await self._report_loss(LeaseLoss(reason=LeaseLossReason.RENEWAL_TIMEOUT))
                    return
                except Exception:
                    await self._report_loss(LeaseLoss(reason=LeaseLossReason.RENEWAL_FAILED))
                    return
                if renewal_delay <= 0:
                    await self._report_loss(LeaseLoss(reason=LeaseLossReason.LEASE_DEADLINE))
                    return
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=renewal_delay,
                    )
                    return
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            if not self._stop.is_set():
                await self._report_loss(LeaseLoss(reason=LeaseLossReason.CANCELLED))
            raise

    @staticmethod
    def _observe_completion(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if error := task.exception():
            logger.critical(
                "session lease supervisor failed",
                exc_info=(type(error), error, error.__traceback__),
            )
