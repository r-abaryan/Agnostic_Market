"""Fenced durable session close and expired-session reaping."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from psycopg_pool import AsyncConnectionPool
from pydantic import TypeAdapter

from agnostic_market.checkpoints import (
    SchemaValidatedCheckpointSaver,
    build_checkpoint_serializer,
    build_checkpointer,
)
from agnostic_market.dtos.session import AuthorityIdentifier
from agnostic_market.dtos.state import ReasoningState
from agnostic_market.durability.encryption import AesGcmSessionCipher
from agnostic_market.durability.postgres_checkpoints import FencedPostgresCheckpointSaver
from agnostic_market.durability.session_registry import (
    CheckpointGeneration,
    LeaseAdmissionError,
    LeaseAdmissionReason,
    SessionCloseAuthority,
    SessionCloseClaim,
    SessionCloseError,
    SessionCloseReason,
    SessionCloseRequest,
    SessionLeaseAuthority,
    SessionLifecycle,
    SessionRegistryPort,
    SessionRegistryRecord,
)
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingObserver,
    observe_duration,
)

logger = logging.getLogger("agnostic_market.durability.session_lifecycle")

_AUTHORITY_IDENTIFIER = TypeAdapter(AuthorityIdentifier)


class SessionReapCoordinator(Protocol):
    async def reap_expired(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
    ) -> TenantReapResult: ...


@dataclass(frozen=True, slots=True)
class TenantReapResult:
    """Committed outcome of one tenant-scoped close and purge sweep."""

    tenant_id: str
    sessions_closed: int
    tombstones_purged: int
    failure_count: int

    def __post_init__(self) -> None:
        _AUTHORITY_IDENTIFIER.validate_python(self.tenant_id)
        for field_name in ("sessions_closed", "tombstones_purged", "failure_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ReaperCycleResult:
    """Committed outcomes retained across every tenant attempted in one cycle."""

    tenants: tuple[TenantReapResult, ...]

    def __post_init__(self) -> None:
        tenant_ids = tuple(result.tenant_id for result in self.tenants)
        if len(set(tenant_ids)) != len(tenant_ids):
            raise ValueError("reaper cycle tenant results must be unique")

    @property
    def sessions_closed(self) -> int:
        return sum(result.sessions_closed for result in self.tenants)

    @property
    def tombstones_purged(self) -> int:
        return sum(result.tombstones_purged for result in self.tenants)

    @property
    def tenants_attempted(self) -> int:
        return len(self.tenants)

    @property
    def failure_count(self) -> int:
        return sum(result.failure_count for result in self.tenants)

    @property
    def failed_tenant_ids(self) -> tuple[str, ...]:
        return tuple(result.tenant_id for result in self.tenants if result.failure_count)


class TenantReapFailure(ExceptionGroup):
    """One tenant sweep failed after retaining its committed outcome."""

    result: TenantReapResult

    def __new__(
        cls,
        message: str,
        exceptions: Sequence[Exception],
        result: TenantReapResult,
    ) -> TenantReapFailure:
        if result.failure_count != len(exceptions):
            raise ValueError("tenant reaper failure count must match its retained causes")
        failure = super().__new__(cls, message, exceptions)
        failure.result = result
        return failure

    def __init__(
        self,
        _message: str,
        _exceptions: Sequence[Exception],
        _result: TenantReapResult,
    ) -> None:
        pass

    def __reduce__(self) -> tuple[object, ...]:
        # ExceptionGroup rebuilds from its message and exceptions alone, but this
        # subclass also requires its retained result, so without this copy and
        # pickle both raise TypeError instead of reporting the failure.
        return (
            self.__class__,
            (self.message, cast(Sequence[Exception], self.exceptions), self.result),
        )

    def derive(self, exceptions: Sequence[BaseException]) -> ExceptionGroup:
        if not all(isinstance(exception, Exception) for exception in exceptions):
            raise TypeError("tenant reaper failures can contain only Exception instances")
        # A split subgroup cannot truthfully retain the full sweep's failure count.
        return ExceptionGroup(self.message, cast(Sequence[Exception], exceptions))


class ReaperCycleFailure(ExceptionGroup):
    """A multi-tenant cycle failed after retaining every attempted outcome."""

    result: ReaperCycleResult

    def __new__(
        cls,
        message: str,
        exceptions: Sequence[Exception],
        result: ReaperCycleResult,
    ) -> ReaperCycleFailure:
        if len(result.failed_tenant_ids) != len(exceptions):
            raise ValueError("reaper cycle failed-tenant count must match its retained causes")
        failure = super().__new__(cls, message, exceptions)
        failure.result = result
        return failure

    def __init__(
        self,
        _message: str,
        _exceptions: Sequence[Exception],
        _result: ReaperCycleResult,
    ) -> None:
        pass

    def __reduce__(self) -> tuple[object, ...]:
        # ExceptionGroup rebuilds from its message and exceptions alone, but this
        # subclass also requires its retained result, so without this copy and
        # pickle both raise TypeError instead of reporting the failure.
        return (
            self.__class__,
            (self.message, cast(Sequence[Exception], self.exceptions), self.result),
        )

    def derive(self, exceptions: Sequence[BaseException]) -> ExceptionGroup:
        if not all(isinstance(exception, Exception) for exception in exceptions):
            raise TypeError("reaper cycle failures can contain only Exception instances")
        # A split subgroup cannot truthfully retain the full cycle's tenant results.
        return ExceptionGroup(self.message, cast(Sequence[Exception], exceptions))


class OperationalSessionReaper:
    """Schedule bounded tenant sweeps through the shared close coordinator."""

    def __init__(
        self,
        coordinator: SessionReapCoordinator,
        *,
        tenant_ids: Sequence[str],
        interval_seconds: float,
        batch_size: int,
    ) -> None:
        validated_tenants = tuple(
            _AUTHORITY_IDENTIFIER.validate_python(tenant_id) for tenant_id in tenant_ids
        )
        if len(set(validated_tenants)) != len(validated_tenants):
            raise ValueError("session reaper tenant ids must be unique")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("session reaper interval must be positive and finite")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("session reaper batch size must be a positive integer")
        self._coordinator = coordinator
        self._tenant_ids = tuple(sorted(validated_tenants))
        self._interval_seconds = interval_seconds
        self._batch_size = batch_size

    async def run_once(self) -> ReaperCycleResult:
        tenant_results: list[TenantReapResult] = []
        failures: list[Exception] = []
        for tenant_id in self._tenant_ids:
            try:
                result = await self._coordinator.reap_expired(tenant_id, limit=self._batch_size)
                if result.tenant_id != tenant_id or result.failure_count:
                    raise ValueError("successful tenant reap result does not match its sweep")
            except TenantReapFailure as exc:
                if exc.result.tenant_id == tenant_id:
                    exc.add_note(f"failed reaper tenant {tenant_id}")
                    tenant_results.append(exc.result)
                    failures.append(exc)
                else:
                    mismatch = ValueError("failed tenant reap result does not match its sweep")
                    mismatch.add_note(f"failed reaper tenant {tenant_id}")
                    tenant_results.append(
                        TenantReapResult(
                            tenant_id=tenant_id,
                            sessions_closed=0,
                            tombstones_purged=0,
                            failure_count=1,
                        )
                    )
                    failures.append(
                        ExceptionGroup(
                            "tenant reap result binding failed",
                            (exc, mismatch),
                        )
                    )
            except Exception as exc:
                exc.add_note(f"failed reaper tenant {tenant_id}")
                tenant_results.append(
                    TenantReapResult(
                        tenant_id=tenant_id,
                        sessions_closed=0,
                        tombstones_purged=0,
                        failure_count=1,
                    )
                )
                failures.append(exc)
            else:
                tenant_results.append(result)
        cycle = ReaperCycleResult(tenants=tuple(tenant_results))
        if failures:
            raise ReaperCycleFailure(
                "one or more tenant cleanup operations failed",
                failures,
                cycle,
            )
        return cycle

    async def serve(self, stop: asyncio.Event) -> None:
        """Run immediately, then at the configured cadence until stopped."""
        while not stop.is_set():
            try:
                result = await self.run_once()
            except ReaperCycleFailure as exc:
                result = exc.result
                logger.exception(
                    "durable session reaper cycle failed after closing %d sessions and purging "
                    "%d tombstones across %d tenants (%d failures)",
                    result.sessions_closed,
                    result.tombstones_purged,
                    result.tenants_attempted,
                    result.failure_count,
                )
            except Exception:
                logger.exception("durable session reaper cycle failed")
            else:
                logger.info(
                    "durable session reaper closed %d sessions and purged %d tombstones across "
                    "%d tenants",
                    result.sessions_closed,
                    result.tombstones_purged,
                    result.tenants_attempted,
                )
            if stop.is_set():
                return
            try:
                async with asyncio.timeout(self._interval_seconds):
                    await stop.wait()
            except TimeoutError:
                pass


class DurableSessionLifecycleCoordinator:
    """Use one fenced close contract for live teardown and background reaping."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        registry: SessionRegistryPort,
        cipher: AesGcmSessionCipher,
        *,
        checkpoint_io_timeout_seconds: float,
        close_lease_duration_seconds: float,
        tombstone_retention_seconds: float,
        durability_timing: DurabilityTimingObserver | None = None,
    ) -> None:
        if checkpoint_io_timeout_seconds <= 0:
            raise ValueError("checkpoint close timeout must be positive")
        if close_lease_duration_seconds <= 0:
            raise ValueError("session close lease duration must be positive")
        if tombstone_retention_seconds <= 0:
            raise ValueError("closed tombstone retention must be positive")
        self._pool = pool
        self._registry = registry
        self._cipher = cipher
        self._checkpoint_io_timeout_seconds = checkpoint_io_timeout_seconds
        self._close_lease_duration_seconds = close_lease_duration_seconds
        self._tombstone_retention_seconds = tombstone_retention_seconds
        self._durability_timing = durability_timing

    def bind(self, authority: SessionLeaseAuthority) -> BoundDurableSessionCloser:
        return BoundDurableSessionCloser(
            coordinator=self,
            authority=authority,
            operation_id=uuid.uuid4().hex,
            lease_owner_id=uuid.uuid4().hex,
        )

    async def begin_live_close(
        self,
        authority: SessionLeaseAuthority,
        operation_id: str,
        lease_owner_id: str,
    ) -> SessionCloseClaim:
        return await self._registry.begin_close(
            authority,
            SessionCloseRequest(
                operation_id=operation_id,
                lease_owner_id=lease_owner_id,
                duration_seconds=self._close_lease_duration_seconds,
            ),
        )

    async def close_completed(
        self,
        authority: SessionLeaseAuthority,
        operation_id: str,
    ) -> bool:
        record = await self._registry.get(
            authority.tenant_id,
            authority.authority.logical_session_id,
        )
        return bool(
            record is not None
            and record.lifecycle is SessionLifecycle.CLOSED
            and record.close_operation_id == operation_id
            and record.authority == authority.authority
            and record.deployment_id == authority.deployment_id
            and record.graph_contract == authority.graph_contract
            and record.config_version == authority.config_version
        )

    async def _delete_generation(
        self,
        claim: SessionCloseClaim,
        generation: CheckpointGeneration,
    ) -> SessionCloseClaim:
        claim = await self._refresh_claim(claim)
        checkpointer = self._generation_checkpointer(claim.authority, generation)
        await checkpointer.adelete_thread(generation.storage_thread_id)
        claim = await self._refresh_claim(claim)
        deleted = await self._registry.record_close_checkpoint_deletion(
            claim.authority,
            generation.checkpoint_namespace,
        )
        if deleted.state != "deleted" or deleted.storage_thread_id != generation.storage_thread_id:
            raise RuntimeError("session close recorded an inconsistent checkpoint deletion")
        return claim

    async def _refresh_claim(self, claim: SessionCloseClaim) -> SessionCloseClaim:
        return await self._registry.refresh_close(
            claim.authority,
            duration_seconds=self._close_lease_duration_seconds,
        )

    def _generation_checkpointer(
        self,
        authority: SessionCloseAuthority,
        generation: CheckpointGeneration,
    ) -> SchemaValidatedCheckpointSaver:
        with observe_duration(
            self._durability_timing,
            DurabilityOperation.CHECKPOINT_BIND,
        ):
            backend = FencedPostgresCheckpointSaver(
                self._pool,
                authority=authority,
                cipher=self._cipher,
                serde=build_checkpoint_serializer(),
                durability_timing=self._durability_timing,
            )
            checkpointer = build_checkpointer(
                backend,
                synchronous_operations=False,
                cipher=self._cipher,
            )
        checkpointer.bind_checkpoint_contract(
            ReasoningState.model_fields,
            binding=generation.binding,
            io_timeout_seconds=self._checkpoint_io_timeout_seconds,
        )
        return checkpointer

    async def checkpoint_has_pending_interrupt(self, claim: SessionCloseClaim) -> bool:
        """Inspect every retained generation through the acquired close fence."""
        for generation in claim.generations:
            if generation.state == "deleted":
                continue
            claim = await self._refresh_claim(claim)
            checkpointer = self._generation_checkpointer(claim.authority, generation)
            if await checkpointer.acheckpoint_has_pending_interrupt(generation.binding.config):
                return True
        return False

    async def finalize(self, claim: SessionCloseClaim) -> SessionRegistryRecord:
        try:
            claim = await self._refresh_claim(claim)
            generations = await self._registry.close_checkpoint_generations(claim.authority)
        except SessionCloseError as exc:
            if exc.reason is not SessionCloseReason.NOT_ELIGIBLE:
                raise
            return await self._finalize_registry_record(claim.authority)
        for generation in generations:
            claim = await self._delete_generation(claim, generation)
        claim = await self._refresh_claim(claim)
        return await self._finalize_registry_record(claim.authority)

    async def _finalize_registry_record(
        self,
        authority: SessionCloseAuthority,
    ) -> SessionRegistryRecord:
        closed = await self._registry.finalize_close(
            authority,
            tombstone_retention_seconds=self._tombstone_retention_seconds,
        )
        if closed.lifecycle is not SessionLifecycle.CLOSED or closed.envelope is not None:
            raise RuntimeError("session close returned an incomplete tombstone")
        return closed

    async def reap_expired(self, tenant_id: str, *, limit: int = 100) -> TenantReapResult:
        candidates = await self._registry.expired_sessions(tenant_id, limit=limit)
        closed = 0
        purged = 0
        failures: list[Exception] = []

        def record_failure(error: Exception, context: str) -> None:
            error.add_note(context)
            failures.append(error)

        def is_benign_race(error: SessionCloseError) -> bool:
            return error.reason in {
                SessionCloseReason.NOT_ELIGIBLE,
                SessionCloseReason.WRONG_CLOSE_OWNER,
            }

        for candidate in candidates:
            request = SessionCloseRequest(
                operation_id=candidate.close_operation_id or uuid.uuid4().hex,
                lease_owner_id=uuid.uuid4().hex,
                duration_seconds=self._close_lease_duration_seconds,
            )
            try:
                claim = await self._registry.claim_expired(candidate, request)
            except SessionCloseError as exc:
                if is_benign_race(exc):
                    continue
                record_failure(
                    exc,
                    f"failed reaper candidate {candidate.tenant_id}/{candidate.logical_session_id}",
                )
                continue
            except Exception as exc:
                record_failure(
                    exc,
                    f"failed reaper candidate {candidate.tenant_id}/{candidate.logical_session_id}",
                )
                continue
            try:
                await self.finalize(claim)
            except SessionCloseError as exc:
                if is_benign_race(exc):
                    continue
                record_failure(
                    exc,
                    f"failed reaper candidate {candidate.tenant_id}/{candidate.logical_session_id}",
                )
            except Exception as exc:
                record_failure(
                    exc,
                    f"failed reaper candidate {candidate.tenant_id}/{candidate.logical_session_id}",
                )
            else:
                closed += 1
        try:
            purged = await self._registry.purge_closed_tombstones(tenant_id, limit=limit)
        except Exception as exc:
            record_failure(exc, f"failed closed-session tombstone purge for tenant {tenant_id}")
        result = TenantReapResult(
            tenant_id=tenant_id,
            sessions_closed=closed,
            tombstones_purged=purged,
            failure_count=len(failures),
        )
        if failures:
            raise TenantReapFailure(
                "one or more durable session cleanup operations failed",
                failures,
                result,
            )
        return result


@dataclass(slots=True)
class BoundDurableSessionCloser:
    """Retain one operation identity across close retries and lost acknowledgements."""

    coordinator: DurableSessionLifecycleCoordinator
    authority: SessionLeaseAuthority
    operation_id: str
    lease_owner_id: str
    _claim: SessionCloseClaim | None = field(default=None, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)

    async def begin_close(self) -> None:
        if self._claim is None and not self._finalized:
            try:
                self._claim = await self.coordinator.begin_live_close(
                    self.authority,
                    self.operation_id,
                    self.lease_owner_id,
                )
            except LeaseAdmissionError as exc:
                if exc.reason not in {
                    LeaseAdmissionReason.LIFECYCLE_REJECTED,
                    LeaseAdmissionReason.SESSION_EXPIRED,
                } or not await self.coordinator.close_completed(
                    self.authority,
                    self.operation_id,
                ):
                    raise
                self._finalized = True

    async def finalize_close(self) -> None:
        if self._finalized:
            return
        if self._claim is None:
            raise RuntimeError("durable close was not begun")
        try:
            await self.coordinator.finalize(self._claim)
        except SessionCloseError as exc:
            if exc.reason is not SessionCloseReason.NOT_ELIGIBLE:
                raise
            self._claim = await self.coordinator.begin_live_close(
                self.authority,
                self.operation_id,
                self.lease_owner_id,
            )
            await self.coordinator.finalize(self._claim)
        self._finalized = True

    async def checkpoint_has_pending_interrupt(self) -> bool:
        if self._finalized:
            return False
        if self._claim is None:
            raise RuntimeError("durable close was not begun")
        return await self.coordinator.checkpoint_has_pending_interrupt(self._claim)
