"""Job-owned PostgreSQL resources for the durable application runtime."""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agnostic_market.checkpoints import (
    SchemaValidatedCheckpointSaver,
    build_checkpoint_serializer,
    build_checkpointer,
)
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.dtos.platform import PlatformRuntimeConfig
from agnostic_market.dtos.session import AdmittedSessionAuthority
from agnostic_market.durability.encryption import AesGcmSessionCipher
from agnostic_market.durability.migrations import (
    require_platform_application_role,
    require_platform_schema_version,
)
from agnostic_market.durability.postgres_checkpoints import FencedPostgresCheckpointSaver
from agnostic_market.durability.session_lease import (
    LeaseLossCallback,
    SessionLeaseSupervisor,
)
from agnostic_market.durability.session_lifecycle import (
    BoundDurableSessionCloser,
    DurableSessionLifecycleCoordinator,
    OperationalSessionReaper,
)
from agnostic_market.durability.session_payload import DurableSessionPayload
from agnostic_market.durability.session_registry import (
    BoundCheckpointGenerationAuthority,
    CheckpointGeneration,
    CheckpointRevisionDisposition,
    CheckpointRevisionReconciliation,
    PostgresSessionRegistry,
    RestoredSessionState,
    SessionLeaseAuthority,
    SessionLeaseRenewal,
    SessionLeaseRequest,
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryPort,
    SessionRegistryRecord,
)
from agnostic_market.durability.session_state import BoundPostgresSessionStatePersistence
from agnostic_market.durability.timing import (
    DurabilityOperation,
    DurabilityTimingObserver,
    observe_duration,
)
from agnostic_market.secrets.base import SecretResolver

_AES_256_KEY_BYTES = 32


def load_platform_runtime_config(path: Path) -> PlatformRuntimeConfig:
    """Load strict deployment configuration without resolving its secrets."""
    return PlatformRuntimeConfig.model_validate(load_yaml_layer(path))


def _decode_session_key(encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("platform session key is not valid base64") from exc
    if len(key) != _AES_256_KEY_BYTES:
        raise ValueError("platform session key must decode to exactly 32 bytes")
    return key


def _milliseconds(seconds: float) -> str:
    return str(math.ceil(seconds * 1000))


async def _finish_pool_close(
    close_task: asyncio.Task[None],
) -> asyncio.CancelledError | None:
    deferred_cancellation: asyncio.CancelledError | None = None
    while not close_task.done():
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancellation:
            if close_task.cancelled():
                break
            if deferred_cancellation is None:
                deferred_cancellation = cancellation
            if close_task.done():
                break
        except BaseException:
            # The pool close itself failed. Report it through close_task.result() below so
            # a deferred cancellation is preserved and chained instead of being dropped.
            break
    try:
        close_task.result()
    except BaseException as cleanup_failure:
        if deferred_cancellation is not None:
            raise deferred_cancellation from cleanup_failure
        raise
    return deferred_cancellation


@dataclass(frozen=True, slots=True)
class DurableSessionResources:
    """Session-bound persistence assembled from one admitted registry lease."""

    authority: SessionLeaseAuthority
    registry: SessionRegistryPort
    restored: RestoredSessionState
    reconciliation: CheckpointRevisionReconciliation | None
    generation: CheckpointGeneration
    generation_authority: BoundCheckpointGenerationAuthority
    persistence: BoundPostgresSessionStatePersistence
    checkpointer: SchemaValidatedCheckpointSaver
    closer: BoundDurableSessionCloser
    lease_duration_seconds: float
    lease_renewal_interval_seconds: float

    def __post_init__(self) -> None:
        if not self.checkpointer.encryption_enabled or not self.checkpointer.uses_storage_backend(
            FencedPostgresCheckpointSaver
        ):
            raise ValueError("durable session requires encrypted, registry-fenced checkpoints")
        if self.reconciliation is not None and (
            self.reconciliation.disposition is not CheckpointRevisionDisposition.SEED_REQUIRED
            or self.reconciliation.record != self.restored.record
            or self.reconciliation.payload != self.restored.payload
        ):
            raise ValueError("fresh durable session has inconsistent seed evidence")
        if self.generation != self.generation_authority.current_generation:
            raise ValueError("durable session generation authority is inconsistent")
        if self.restored.record.checkpoint_namespace != self.generation.checkpoint_namespace:
            raise ValueError("durable session projection and checkpoint generation diverge")

    async def activate_seeded(self, checkpoint_revision: int) -> SessionRegistryRecord:
        """Activate only after the initial checkpoint matches registry revision zero."""
        if self.reconciliation is None:
            raise RuntimeError("restored session cannot enter the initial seed path")
        reconciled = await self.registry.reconcile_checkpoint_revision(
            self.authority,
            checkpoint_revision,
        )
        if (
            reconciled.disposition is not CheckpointRevisionDisposition.CURRENT
            or reconciled.record != self.restored.record
            or reconciled.payload != self.restored.payload
        ):
            raise RuntimeError("initial checkpoint does not match the session registry")
        return await self._activate(checkpoint_revision)

    async def _activate(self, checkpoint_revision: int) -> SessionRegistryRecord:
        activated = await self.registry.activate(self.authority)
        if (
            activated.lifecycle is not SessionLifecycle.ACTIVE
            or activated.session_revision != checkpoint_revision
            or activated.checkpoint_namespace != self.generation.checkpoint_namespace
            or activated.lease_owner_id != self.authority.lease_owner_id
            or activated.fencing_generation != self.authority.fencing_generation
        ):
            raise RuntimeError("session activation returned inconsistent authority")
        return activated

    async def reconcile_restored(
        self,
        checkpoint_revision: int,
    ) -> CheckpointRevisionReconciliation:
        """Join an existing checkpoint to the projection under the unchanged lease."""
        if self.reconciliation is not None:
            raise RuntimeError("fresh session cannot enter the existing-checkpoint path")
        reconciled = await self.registry.reconcile_checkpoint_revision(
            self.authority,
            checkpoint_revision,
        )
        if reconciled.record != self.restored.record or reconciled.payload != self.restored.payload:
            raise RuntimeError("restored session changed during checkpoint reconciliation")
        if (
            reconciled.disposition is CheckpointRevisionDisposition.CURRENT
            and reconciled.record.lifecycle is SessionLifecycle.OPENING
        ):
            await self._activate(checkpoint_revision)
        return reconciled

    def build_lease_supervisor(
        self,
        on_lease_lost: LeaseLossCallback,
    ) -> SessionLeaseSupervisor:
        """Bind lease renewal to this session's admitted owner and fence."""
        return SessionLeaseSupervisor(
            self.registry,
            SessionLeaseRenewal(
                **self.authority.model_dump(),
                duration_seconds=self.lease_duration_seconds,
            ),
            renewal_interval_seconds=self.lease_renewal_interval_seconds,
            on_lease_lost=on_lease_lost,
        )


class DurablePlatformResources:
    """Own one event-loop-local pool, cipher, and registry for a network job."""

    def __init__(
        self,
        config: PlatformRuntimeConfig,
        pool: AsyncConnectionPool,
        cipher: AesGcmSessionCipher,
        durability_timing: DurabilityTimingObserver | None = None,
    ) -> None:
        self.config = config
        self.pool = pool
        self.cipher = cipher
        self.registry = PostgresSessionRegistry(
            pool,
            cipher=cipher,
            operation_timeout_seconds=config.database.operation_timeout_seconds,
            durability_timing=durability_timing,
        )
        self._durability_timing = durability_timing
        self.lifecycle = DurableSessionLifecycleCoordinator(
            pool,
            self.registry,
            cipher,
            checkpoint_io_timeout_seconds=config.database.operation_timeout_seconds,
            close_lease_duration_seconds=config.sessions.lease_duration_seconds,
            tombstone_retention_seconds=config.sessions.closed_tombstone_retention_seconds,
            durability_timing=durability_timing,
        )
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._close_failure: BaseException | None = None

    @classmethod
    async def open(
        cls,
        config: PlatformRuntimeConfig,
        secrets: SecretResolver,
        *,
        durability_timing: DurabilityTimingObserver | None = None,
        application_dsn: str | None = None,
    ) -> Self:
        database = config.database
        encoded_key = secrets.resolve(config.encryption.key_ref.uri)
        cipher = AesGcmSessionCipher(
            active_key_version=config.encryption.key_version,
            keys={config.encryption.key_version: _decode_session_key(encoded_key)},
            durability_timing=durability_timing,
        )
        dsn = (
            secrets.resolve(database.application_dsn_ref.uri)
            if application_dsn is None
            else application_dsn
        )

        async def configure(connection: AsyncConnection) -> None:
            await connection.execute(
                "SELECT set_config('search_path', quote_ident(%s), false)",
                (database.schema_name,),
            )
            await connection.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (_milliseconds(database.statement_timeout_seconds),),
            )
            await connection.execute(
                "SELECT set_config('transaction_timeout', %s, false)",
                (_milliseconds(database.transaction_timeout_seconds),),
            )

        pool: AsyncConnectionPool = AsyncConnectionPool(
            dsn,
            min_size=database.minimum_pool_size,
            max_size=database.maximum_pool_size,
            timeout=database.pool_acquisition_timeout_seconds,
            kwargs={
                "autocommit": True,
                "connect_timeout": math.ceil(database.connection_timeout_seconds),
                "prepare_threshold": 0,
            },
            configure=configure,
            open=False,
        )
        try:
            with observe_duration(durability_timing, DurabilityOperation.POOL_OPEN):
                await pool.open(wait=True, timeout=database.connection_timeout_seconds)
            with observe_duration(durability_timing, DurabilityOperation.STARTUP_GATES):
                async with asyncio.timeout(database.operation_timeout_seconds):
                    async with pool.connection(
                        timeout=database.pool_acquisition_timeout_seconds
                    ) as connection:
                        await require_platform_schema_version(
                            connection,
                            database.expected_schema_version,
                        )
                        await require_platform_application_role(
                            connection,
                            schema_name=database.schema_name,
                        )
        except BaseException as failure:
            close_task = asyncio.create_task(pool.close(timeout=database.operation_timeout_seconds))
            try:
                deferred_cancellation = await _finish_pool_close(close_task)
            except BaseException as cleanup_failure:
                raise failure from cleanup_failure
            if deferred_cancellation is not None:
                if isinstance(failure, asyncio.CancelledError):
                    raise failure from deferred_cancellation
                raise deferred_cancellation from failure
            raise
        return cls(config, pool, cipher, durability_timing)

    async def acquire_fresh_session(
        self,
        *,
        tenant_id: str,
        admitted_authority: AdmittedSessionAuthority,
        deployment_id: str,
        config_version: str,
    ) -> DurableSessionResources:
        """Register one fresh session and bind its fenced persistence resources."""
        sessions = self.config.sessions
        lease_owner_id = uuid.uuid4().hex
        registration = SessionRegistration(
            tenant_id=tenant_id,
            authority=admitted_authority,
            deployment_id=deployment_id,
            graph_contract=self.config.graph_contract,
            config_version=config_version,
            principal_generation=0,
            session_revision=0,
            retention_seconds=sessions.session_retention_seconds,
        )
        record = await self.registry.register_and_acquire(
            registration,
            SessionLeaseRequest(
                lease_owner_id=lease_owner_id,
                duration_seconds=sessions.lease_duration_seconds,
            ),
            payload=DurableSessionPayload(),
        )
        authority = SessionLeaseAuthority(
            tenant_id=registration.tenant_id,
            authority=registration.authority,
            deployment_id=registration.deployment_id,
            graph_contract=registration.graph_contract,
            config_version=registration.config_version,
            lease_owner_id=lease_owner_id,
            fencing_generation=record.fencing_generation,
        )
        try:
            restored = await self.registry.restore(authority)
            reconciliation = await self.registry.reconcile_checkpoint_revision(authority, None)
            return await self._bind_session_resources(authority, restored, reconciliation)
        except BaseException as failure:
            # Registration was acknowledged; close only this confirmed lease authority.
            try:
                closer = self.lifecycle.bind(authority)
                await closer.begin_close()
                await closer.finalize_close()
            except BaseException as cleanup_failure:
                raise failure from cleanup_failure
            raise

    def build_reaper(self, tenant_ids: tuple[str, ...]) -> OperationalSessionReaper:
        sessions = self.config.sessions
        return OperationalSessionReaper(
            self.lifecycle,
            tenant_ids=tenant_ids,
            interval_seconds=sessions.reaper_interval_seconds,
            batch_size=sessions.reaper_batch_size,
        )

    async def restore_owned_session(
        self,
        authority: SessionLeaseAuthority,
    ) -> DurableSessionResources:
        """Restore without reacquiring or changing the admitted lease authority."""
        if authority.graph_contract != self.config.graph_contract:
            raise ValueError("restored session graph contract does not match deployment")
        restored = await self.registry.restore(authority)
        return await self._bind_session_resources(authority, restored, None)

    async def _bind_session_resources(
        self,
        authority: SessionLeaseAuthority,
        restored: RestoredSessionState,
        reconciliation: CheckpointRevisionReconciliation | None,
    ) -> DurableSessionResources:
        generations = await self.registry.checkpoint_generations(authority)
        current_generations = tuple(item for item in generations if item.state == "current")
        if len(current_generations) != 1:
            raise RuntimeError("session does not have one current checkpoint generation")
        generation = current_generations[0]
        generation_authority = BoundCheckpointGenerationAuthority(
            self.registry,
            authority,
            generation,
        )
        with observe_duration(
            self._durability_timing,
            DurabilityOperation.CHECKPOINT_BIND,
        ):
            serializer = build_checkpoint_serializer()
            backend = FencedPostgresCheckpointSaver(
                self.pool,
                authority=authority,
                cipher=self.cipher,
                serde=serializer,
                durability_timing=self._durability_timing,
            )
            checkpointer = build_checkpointer(
                backend,
                synchronous_operations=False,
                cipher=self.cipher,
            )
        return DurableSessionResources(
            authority=authority,
            registry=self.registry,
            restored=restored,
            reconciliation=reconciliation,
            generation=generation,
            generation_authority=generation_authority,
            persistence=BoundPostgresSessionStatePersistence(
                self.registry,
                authority,
                self._durability_timing,
            ),
            checkpointer=checkpointer,
            closer=self.lifecycle.bind(authority),
            lease_duration_seconds=self.config.sessions.lease_duration_seconds,
            lease_renewal_interval_seconds=(self.config.sessions.lease_renewal_interval_seconds),
        )

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if self._close_failure is not None:
                raise RuntimeError(
                    "durable platform pool close already failed and cannot be completed"
                ) from self._close_failure
            close_task = asyncio.create_task(
                self.pool.close(timeout=self.config.database.operation_timeout_seconds)
            )
            try:
                deferred_cancellation = await _finish_pool_close(close_task)
            except BaseException as failure:
                # The vendor pool marks itself closed before draining connections, so once
                # it reports closed a later call returns immediately and cannot finish the
                # interrupted drain. Retain the failure instead of letting that retry
                # report a clean shutdown. A pool that has not yet marked itself closed is
                # still genuinely retryable.
                if self.pool.closed:
                    self._close_failure = failure
                raise
            self._closed = True
            if deferred_cancellation is not None:
                raise deferred_cancellation
