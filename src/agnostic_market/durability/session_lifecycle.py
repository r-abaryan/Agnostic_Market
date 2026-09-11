"""Fenced durable session close and expired-session reaping."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from agnostic_market.checkpoints import (
    SchemaValidatedCheckpointSaver,
    build_checkpoint_serializer,
    build_checkpointer,
)
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
        backend = FencedPostgresCheckpointSaver(
            self._pool,
            authority=authority,
            cipher=self._cipher,
            serde=build_checkpoint_serializer(),
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

    async def reap_expired(self, tenant_id: str, *, limit: int = 100) -> int:
        candidates = await self._registry.expired_sessions(tenant_id, limit=limit)
        closed = 0
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
            await self._registry.purge_closed_tombstones(tenant_id, limit=limit)
        except Exception as exc:
            record_failure(exc, f"failed closed-session tombstone purge for tenant {tenant_id}")
        if failures:
            raise ExceptionGroup("one or more durable session cleanup operations failed", failures)
        return closed


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
