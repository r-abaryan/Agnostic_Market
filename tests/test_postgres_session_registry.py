"""Real PostgreSQL contracts for the platform-session registry."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import uuid
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from llm_fakes import (
    TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
    TEST_STRUCTURED_OUTPUT_METHOD,
    FakeChatModel,
)
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import (
    CheckViolation,
    InsufficientPrivilege,
    NotNullViolation,
    UndefinedObject,
    UniqueViolation,
)
from psycopg_pool import AsyncConnectionPool
from routing_helpers import ArchitectureRoutingRecognizer
from telemetry_helpers import make_tenant_telemetry
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    committed_turn_events,
    engine_events,
)

from agnostic_market.application import (
    ApplicationModels,
    ApplicationSettings,
    build_application_session,
    build_fixture_tenant_services,
)
from agnostic_market.checkpoints import (
    CheckpointBinding,
    build_checkpoint_serializer,
    build_checkpointer,
    graph_contract_fingerprint,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.events import CommittedTurn, InterruptEvent, TurnFacts
from agnostic_market.dtos.platform import PlatformRuntimeConfig
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.dtos.state import PendingCartMutation, ReasoningState
from agnostic_market.durability.encryption import AesGcmSessionCipher, SessionEnvelopeContext
from agnostic_market.durability.migrations import (
    PLATFORM_SESSION_SCHEMA_VERSION,
    PlatformSchemaError,
    apply_platform_migrations,
    grant_platform_application_role,
    require_platform_application_role,
    require_platform_schema_version,
)
from agnostic_market.durability.platform_runtime import DurablePlatformResources
from agnostic_market.durability.postgres_checkpoints import FencedPostgresCheckpointSaver
from agnostic_market.durability.session_lifecycle import DurableSessionLifecycleCoordinator
from agnostic_market.durability.session_payload import (
    SESSION_OPERATION_RESULT_SCHEMA_VERSION,
    SESSION_PAYLOAD_SCHEMA_VERSION,
    DurableSessionPayload,
    EmptySessionOperationResult,
    PrincipalRetirementMarker,
    SessionOperationReceiptPayload,
)
from agnostic_market.durability.session_registry import (
    CheckpointRevisionDisposition,
    LeaseAdmissionError,
    LeaseAdmissionReason,
    PostgresSessionRegistry,
    SessionCloseError,
    SessionCloseReason,
    SessionCloseRequest,
    SessionLeaseAuthority,
    SessionLeaseRenewal,
    SessionLeaseRequest,
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryDataError,
    SessionRegistryError,
    SessionRegistryRecord,
    SessionRestoreError,
    SessionRestoreReason,
    SessionStatePublication,
    SessionStateWriteError,
    SessionStateWriteReason,
)
from agnostic_market.durability.session_state import (
    BoundPostgresSessionStatePersistence,
    SessionStateCoordinator,
)
from agnostic_market.tenancy.context import build_tenant_context

_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
_KEY = bytes(range(32))


def _cipher() -> AesGcmSessionCipher:
    return AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY})


def _empty_payload() -> DurableSessionPayload:
    return DurableSessionPayload()


async def _publish_retirement_marker(
    registry: PostgresSessionRegistry,
    authority: SessionLeaseAuthority,
    *,
    expected_revision: int,
    transition_id: str,
) -> None:
    await registry.publish(
        SessionStatePublication(
            **authority.model_dump(),
            expected_revision=expected_revision,
            operation_id=f"principal-retirement:{transition_id}",
            request_fingerprint=hashlib.sha256(
                f"begin_principal_retirement:{transition_id}".encode()
            ).hexdigest(),
            payload=DurableSessionPayload(
                principal_retirement=PrincipalRetirementMarker(transition_id=transition_id)
            ),
            operation_result=EmptySessionOperationResult(),
        )
    )


async def _complete_retirement_marker(
    registry: PostgresSessionRegistry,
    authority: SessionLeaseAuthority,
    *,
    expected_revision: int,
    transition_id: str,
) -> None:
    await registry.publish(
        SessionStatePublication(
            **authority.model_dump(),
            expected_revision=expected_revision,
            operation_id=f"principal-retirement-complete:{transition_id}",
            request_fingerprint=hashlib.sha256(
                f"finish_principal_retirement:{transition_id}".encode()
            ).hexdigest(),
            payload=_empty_payload(),
            operation_result=EmptySessionOperationResult(),
        )
    )


def _dsn() -> str:
    dsn = os.environ.get(_POSTGRES_DSN_ENV, "").strip()
    if not dsn:
        pytest.skip(f"{_POSTGRES_DSN_ENV} is provided by the disposable PostgreSQL harness")
    return dsn


def _schema_dsn(dsn: str, schema: str) -> str:
    return make_conninfo(dsn, options=f"-csearch_path={schema}")


async def _open_pool(stack: AsyncExitStack, dsn: str) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        dsn,
        min_size=0,
        max_size=2,
        timeout=0.5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await pool.open(wait=True, timeout=2.0)
    stack.push_async_callback(pool.close)
    return pool


async def _rewind_generation_schema_to_version_4(connection: AsyncConnection) -> None:
    await connection.execute("DROP TABLE platform_checkpoint_write_manifests")
    await connection.execute("DROP TABLE platform_checkpoint_generations CASCADE")
    await connection.execute(
        """
        ALTER TABLE platform_session_operations
        DROP CONSTRAINT platform_session_operations_one_receipt_per_revision,
        DROP CONSTRAINT platform_session_operations_current_result_schema
        """
    )
    await connection.execute(
        """
        ALTER TABLE platform_sessions
            DROP CONSTRAINT platform_sessions_checkpoint_matches_generation,
            DROP CONSTRAINT platform_sessions_close_operation_format,
            DROP CONSTRAINT platform_sessions_current_payload_schema,
            DROP COLUMN close_operation_id,
            ADD CONSTRAINT platform_sessions_checkpoint_matches_fence
                CHECK (
                    checkpoint_namespace =
                        logical_session_id || '::fence::' || fencing_generation
                )
        """
    )
    await connection.execute("DELETE FROM platform_schema_migrations WHERE version >= 5")


@pytest.mark.postgres
async def test_initial_checkpoint_generation_is_registered_with_its_session() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("generation_store", "AD_generation")
    authority = _lease_authority(registration, "generation-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        record = await registry.register_and_acquire(
            registration, _lease("generation-owner"), payload=_empty_payload()
        )
        generations = await registry.checkpoint_generations(authority)
        assert len(generations) == 1
        generation = generations[0]
        assert generation.checkpoint_namespace == record.checkpoint_namespace
        assert generation.fencing_generation == record.fencing_generation
        assert generation.principal_generation == record.principal_generation
        assert generation.state == "current"
        assert generation.transition_id is None
        assert generation.binding == CheckpointBinding(
            tenant_id=record.tenant_id,
            logical_session_id=record.authority.logical_session_id,
            deployment_id=record.deployment_id,
            graph_contract=record.graph_contract,
            thread_id=record.checkpoint_namespace,
        )
        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.checkpoint_generations(
                authority.model_copy(update={"lease_owner_id": "foreign-owner"})
            )
        assert rejected.value.reason is LeaseAdmissionReason.WRONG_LEASE_OWNER


@pytest.mark.postgres
async def test_checkpoint_rotation_allocation_is_atomic_and_replayable() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("rotation_store", "AD_rotation")
    authority = _lease_authority(registration, "rotation-owner")
    async with AsyncExitStack() as stack:
        registries = [
            PostgresSessionRegistry(
                await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
            )
            for _ in range(2)
        ]
        original = await registries[0].register_and_acquire(
            registration,
            _lease("rotation-owner"),
            payload=DurableSessionPayload(guest_order_refs=("ORD-OLD",)),
        )
        await registries[0].activate(authority)
        await _publish_retirement_marker(
            registries[0], authority, expected_revision=0, transition_id="identity-change"
        )
        first, duplicate = await asyncio.gather(
            *(
                registry.begin_checkpoint_rotation(
                    authority,
                    expected_revision=1,
                    expected_principal_generation=0,
                    transition_id="identity-change",
                )
                for registry in registries
            )
        )
        assert first == duplicate
        assert first.state == "pending"
        assert first.principal_generation == 1
        assert first.fencing_generation == original.fencing_generation
        assert first.checkpoint_namespace != original.checkpoint_namespace
        restored = await registries[1].restore(authority)
        assert restored.record.checkpoint_namespace == original.checkpoint_namespace
        assert restored.record.principal_generation == 0
        assert restored.record.session_revision == 1
        assert restored.payload.guest_order_refs == ()
        assert restored.payload.principal_retirement.transition_id == "identity-change"
        assert [g.state for g in await registries[1].checkpoint_generations(authority)] == [
            "current",
            "pending",
        ]
        for revision, principal, transition in (
            (2, 0, "identity-change"),
            (1, 1, "identity-change"),
            (1, 0, "other-change"),
        ):
            with pytest.raises(SessionStateWriteError):
                await registries[1].begin_checkpoint_rotation(
                    authority,
                    expected_revision=revision,
                    expected_principal_generation=principal,
                    transition_id=transition,
                )
        with pytest.raises(SessionStateWriteError):
            await registries[1].publish(
                SessionStatePublication(
                    **authority.model_dump(),
                    expected_revision=1,
                    operation_id="erase-marker",
                    request_fingerprint="a" * 64,
                    payload=_empty_payload(),
                    operation_result=EmptySessionOperationResult(),
                )
            )
        assert await registries[1].restore(authority) == restored


@pytest.mark.postgres
async def test_checkpoint_rotation_switches_atomically_and_records_verified_deletion() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("rotation_switch_store", "AD_rotation_switch")
    authority = _lease_authority(registration, "rotation-owner")
    async with AsyncExitStack() as stack:
        registries = [
            PostgresSessionRegistry(
                await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
            )
            for _ in range(2)
        ]
        source = await registries[0].register_and_acquire(
            registration,
            _lease("rotation-owner"),
            payload=DurableSessionPayload(guest_order_refs=("ORD-OLD",)),
        )
        await registries[0].activate(authority)
        await _publish_retirement_marker(
            registries[0], authority, expected_revision=0, transition_id="switch-identity"
        )
        destination = await registries[0].begin_checkpoint_rotation(
            authority,
            expected_revision=1,
            expected_principal_generation=0,
            transition_id="switch-identity",
        )

        switched, duplicate = await asyncio.gather(
            *(
                registry.switch_checkpoint_generation(authority, "switch-identity")
                for registry in registries
            )
        )
        assert switched == duplicate
        assert switched.source.checkpoint_namespace == source.checkpoint_namespace
        assert switched.source.state == "retired"
        assert switched.destination == destination.model_copy(update={"state": "current"})
        restored = await registries[1].restore(authority)
        assert restored.record.checkpoint_namespace == destination.checkpoint_namespace
        assert restored.record.principal_generation == 1
        assert restored.record.fencing_generation == authority.fencing_generation
        assert restored.record.session_revision == 1
        assert restored.payload.principal_retirement == PrincipalRetirementMarker(
            transition_id="switch-identity"
        )
        assert [
            generation.state for generation in await registries[0].checkpoint_generations(authority)
        ] == [
            "retired",
            "current",
        ]

        assert (
            await registries[0].begin_checkpoint_rotation(
                authority,
                expected_revision=1,
                expected_principal_generation=0,
                transition_id="switch-identity",
            )
            == switched.destination
        )
        await _complete_retirement_marker(
            registries[0], authority, expected_revision=1, transition_id="switch-identity"
        )
        published = await registries[0].publish(
            SessionStatePublication(
                **authority.model_dump(),
                expected_revision=2,
                operation_id="post-rotation-publication",
                request_fingerprint="b" * 64,
                payload=DurableSessionPayload(guest_order_refs=("ORD-NEW",)),
                operation_result=EmptySessionOperationResult(),
            )
        )
        assert published.record.session_revision == 3

        deleted, duplicate_deleted = await asyncio.gather(
            *(
                registry.record_checkpoint_deletion(authority, "switch-identity")
                for registry in registries
            )
        )
        assert deleted == duplicate_deleted
        assert deleted.checkpoint_namespace == source.checkpoint_namespace
        assert deleted.state == "deleted"
        assert [
            generation.state for generation in await registries[0].checkpoint_generations(authority)
        ] == [
            "deleted",
            "current",
        ]


@pytest.mark.postgres
async def test_rotation_preserves_receipt_encryption_context() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("rotation_receipt_store", "AD_rotation_receipt")
    authority = _lease_authority(registration, "rotation-owner")
    publication = SessionStatePublication(
        **authority.model_dump(),
        expected_revision=0,
        operation_id="pre-rotation-operation",
        request_fingerprint="c" * 64,
        payload=DurableSessionPayload(guest_order_refs=("ORD-BEFORE",)),
        operation_result=EmptySessionOperationResult(),
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration, _lease("rotation-owner"), payload=_empty_payload()
        )
        await registry.activate(authority)
        await registry.publish(publication)
        await _publish_retirement_marker(
            registry, authority, expected_revision=1, transition_id="receipt-identity"
        )
        await registry.begin_checkpoint_rotation(
            authority,
            expected_revision=2,
            expected_principal_generation=0,
            transition_id="receipt-identity",
        )
        await registry.switch_checkpoint_generation(authority, "receipt-identity")

        replayed = await registry.publish(publication)

    assert replayed.replayed
    assert replayed.operation_revision == 1
    assert replayed.operation_result == EmptySessionOperationResult()
    assert replayed.record.session_revision == 2


@pytest.mark.postgres
async def test_late_rotation_replay_and_cleanup_survive_a_newer_generation() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("rotation_history_store", "AD_rotation_history")
    authority = _lease_authority(registration, "rotation-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration, _lease("rotation-owner"), payload=_empty_payload()
        )
        await registry.activate(authority)
        await _publish_retirement_marker(
            registry, authority, expected_revision=0, transition_id="first-identity"
        )
        await registry.begin_checkpoint_rotation(
            authority,
            expected_revision=1,
            expected_principal_generation=0,
            transition_id="first-identity",
        )
        first = await registry.switch_checkpoint_generation(authority, "first-identity")
        await _complete_retirement_marker(
            registry, authority, expected_revision=1, transition_id="first-identity"
        )
        await _publish_retirement_marker(
            registry, authority, expected_revision=2, transition_id="second-identity"
        )
        await registry.begin_checkpoint_rotation(
            authority,
            expected_revision=3,
            expected_principal_generation=1,
            transition_id="second-identity",
        )
        await registry.switch_checkpoint_generation(authority, "second-identity")
        await _complete_retirement_marker(
            registry, authority, expected_revision=3, transition_id="second-identity"
        )

        delayed = await registry.switch_checkpoint_generation(authority, "first-identity")
        deleted = await registry.record_checkpoint_deletion(authority, "first-identity")

        assert delayed.source == first.source
        assert delayed.destination.state == "retired"
        assert deleted.state == "deleted"
        assert [
            (generation.principal_generation, generation.state)
            for generation in await registry.checkpoint_generations(authority)
        ] == [(0, "deleted"), (1, "retired"), (2, "current")]


@pytest.mark.postgres
async def test_checkpoint_rotation_switch_failure_rolls_back_inventory_and_marker() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
        async with AsyncExitStack() as stack:
            registry = PostgresSessionRegistry(
                await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
            )
            registration = _registration("rotation_switch_failure", "AD_rotation_switch_failure")
            authority = _lease_authority(registration, "rotation-owner")
            await registry.register_and_acquire(
                registration, _lease("rotation-owner"), payload=_empty_payload()
            )
            await registry.activate(authority)
            await _publish_retirement_marker(
                registry, authority, expected_revision=0, transition_id="failed-switch"
            )
            await registry.begin_checkpoint_rotation(
                authority,
                expected_revision=1,
                expected_principal_generation=0,
                transition_id="failed-switch",
            )
            before = await registry.restore(authority)
            await connection.execute(
                """
                ALTER TABLE platform_sessions ADD CONSTRAINT test_rotation_switch_failure
                CHECK (tenant_id <> 'rotation_switch_failure' OR principal_generation = 0)
                """
            )
            try:
                with pytest.raises(SessionRegistryError, match="rotation switch failed"):
                    await registry.switch_checkpoint_generation(authority, "failed-switch")
                assert await registry.restore(authority) == before
                assert [
                    generation.state
                    for generation in await registry.checkpoint_generations(authority)
                ] == ["current", "pending"]
            finally:
                await connection.execute(
                    "ALTER TABLE platform_sessions DROP CONSTRAINT test_rotation_switch_failure"
                )


@pytest.mark.postgres
async def test_rotation_marker_failure_rolls_back_destination() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
        async with AsyncExitStack() as stack:
            registry = PostgresSessionRegistry(
                await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
            )
            registration = _registration("rotation_failure_store", "AD_rotation_failure")
            authority = _lease_authority(registration, "rotation-owner")
            await registry.register_and_acquire(
                registration, _lease("rotation-owner"), payload=_empty_payload()
            )
            await registry.activate(authority)
            original = await registry.restore(authority)
            await _publish_retirement_marker(
                registry, authority, expected_revision=0, transition_id="failed-rotation"
            )
            marked = await registry.restore(authority)
            await connection.execute(
                """
                ALTER TABLE platform_checkpoint_generations
                ADD CONSTRAINT test_rotation_marker_failure
                CHECK (tenant_id <> 'rotation_failure_store' OR principal_generation = 0)
                """
            )
            try:
                with pytest.raises(SessionRegistryError, match="rotation allocation failed"):
                    await registry.begin_checkpoint_rotation(
                        authority,
                        expected_revision=1,
                        expected_principal_generation=0,
                        transition_id="failed-rotation",
                    )
                assert original.record.session_revision == 0
                assert await registry.restore(authority) == marked
                assert len(await registry.checkpoint_generations(authority)) == 1
            finally:
                await connection.execute(
                    "ALTER TABLE platform_checkpoint_generations "
                    "DROP CONSTRAINT test_rotation_marker_failure"
                )


@pytest.mark.postgres
async def test_rotation_rejects_stale_authority_before_allocation_or_replay() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        registration = _registration("rotation_authority_store", "AD_rotation_authority")
        authority = _lease_authority(registration, "rotation-owner")
        await registry.register_and_acquire(
            registration, _lease("rotation-owner"), payload=_empty_payload()
        )
        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.begin_checkpoint_rotation(
                authority,
                expected_revision=0,
                expected_principal_generation=0,
                transition_id="authority-rotation",
            )
        assert rejected.value.reason is LeaseAdmissionReason.LIFECYCLE_REJECTED
        await registry.activate(authority)
        with pytest.raises(SessionStateWriteError) as missing_marker:
            await registry.begin_checkpoint_rotation(
                authority,
                expected_revision=0,
                expected_principal_generation=0,
                transition_id="authority-rotation",
            )
        assert missing_marker.value.reason is SessionStateWriteReason.RETIREMENT_MARKER_MISSING
        await _publish_retirement_marker(
            registry, authority, expected_revision=0, transition_id="authority-rotation"
        )
        for revision, principal, reason in (
            (2, 0, SessionStateWriteReason.STALE_REVISION),
            (1, 1, SessionStateWriteReason.STALE_PRINCIPAL),
        ):
            with pytest.raises(SessionStateWriteError) as rejected:
                await registry.begin_checkpoint_rotation(
                    authority,
                    expected_revision=revision,
                    expected_principal_generation=principal,
                    transition_id="authority-rotation",
                )
            assert rejected.value.reason is reason
        for allocated in (False, True):
            if allocated:
                await registry.begin_checkpoint_rotation(
                    authority,
                    expected_revision=1,
                    expected_principal_generation=0,
                    transition_id="authority-rotation",
                )
            for update, reason in (
                ({"lease_owner_id": "foreign-worker"}, LeaseAdmissionReason.WRONG_LEASE_OWNER),
                ({"fencing_generation": 2}, LeaseAdmissionReason.STALE_FENCE),
            ):
                with pytest.raises(LeaseAdmissionError) as rejected:
                    await registry.begin_checkpoint_rotation(
                        authority.model_copy(update=update),
                        expected_revision=1,
                        expected_principal_generation=0,
                        transition_id="authority-rotation",
                    )
                assert rejected.value.reason is reason
            assert len(await registry.checkpoint_generations(authority)) == (2 if allocated else 1)


@pytest.mark.postgres
async def test_generation_insert_failure_rolls_back_initial_lease() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
        await connection.execute(
            """
            ALTER TABLE platform_checkpoint_generations ADD CONSTRAINT test_inventory_failure
            CHECK (tenant_id <> 'inventory_failure_store')
            """
        )
        try:
            async with AsyncExitStack() as stack:
                registry = PostgresSessionRegistry(
                    await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
                )
                registration = _registration("inventory_failure_store", "AD_inventory_failure")
                with pytest.raises(SessionRegistryError, match="registration failed"):
                    await registry.register_and_acquire(
                        registration, _lease("inventory-owner"), payload=_empty_payload()
                    )
                assert (
                    await registry.get(
                        registration.tenant_id, registration.authority.logical_session_id
                    )
                    is None
                )
        finally:
            await connection.execute(
                "ALTER TABLE platform_checkpoint_generations DROP CONSTRAINT test_inventory_failure"
            )


@pytest.mark.postgres
async def test_generation_upgrade_preserves_existing_sessions_and_encryption() -> None:
    dsn = _dsn()
    schema = f"generation_upgrade_{uuid.uuid4().hex[:16]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await apply_platform_migrations(connection)
            async with AsyncExitStack() as stack:
                registry = PostgresSessionRegistry(
                    await _open_pool(stack, isolated_dsn),
                    cipher=_cipher(),
                    operation_timeout_seconds=2.0,
                )
                originals = []
                for tenant in ("upgrade_acme", "café_store", "商店"):
                    registration = _registration(tenant, "AD_same_session")
                    originals.append(
                        await registry.register_and_acquire(
                            registration, _lease("upgrade-owner"), payload=_empty_payload()
                        )
                    )
                # Restore the schema-4 shape without changing either session or its ciphertext.
                await _rewind_generation_schema_to_version_4(connection)
                await apply_platform_migrations(connection)
                await apply_platform_migrations(connection)
                for original in originals:
                    authority = _lease_authority(
                        _registration(original.tenant_id, original.authority.logical_session_id),
                        "upgrade-owner",
                    )
                    restored = await registry.restore(authority)
                    assert restored.record == original
                    assert restored.payload == _empty_payload()
                    generations = await registry.checkpoint_generations(authority)
                    assert len(generations) == 1
                    assert generations[0].checkpoint_namespace == original.checkpoint_namespace
                    assert generations[0].binding == CheckpointBinding(
                        tenant_id=original.tenant_id,
                        logical_session_id=original.authority.logical_session_id,
                        deployment_id=original.deployment_id,
                        graph_contract=original.graph_contract,
                        thread_id=original.checkpoint_namespace,
                    )
                cursor = await connection.execute(
                    """
                    SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
                    WHERE oid IN ('platform_sessions'::regclass,
                                  'platform_checkpoint_generations'::regclass,
                                  'platform_checkpoint_write_manifests'::regclass)
                    ORDER BY relname
                    """
                )
                assert await cursor.fetchall() == [
                    ("platform_checkpoint_generations", True, True),
                    ("platform_checkpoint_write_manifests", True, True),
                    ("platform_sessions", True, True),
                ]
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
@pytest.mark.parametrize("payload_surface", ("session", "operation"))
async def test_generation_upgrade_rejects_unsupported_payloads_atomically(
    payload_surface: str,
) -> None:
    dsn = _dsn()
    schema = f"generation_payload_cutover_{uuid.uuid4().hex[:16]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await apply_platform_migrations(connection)
            async with AsyncExitStack() as stack:
                registry = PostgresSessionRegistry(
                    await _open_pool(stack, isolated_dsn),
                    cipher=_cipher(),
                    operation_timeout_seconds=2.0,
                )
                registration = _registration("payload_cutover", "AD_payload_cutover")
                await registry.register_and_acquire(
                    registration,
                    _lease("payload-cutover-owner"),
                    payload=_empty_payload(),
                )
                if payload_surface == "operation":
                    authority = _lease_authority(registration, "payload-cutover-owner")
                    await registry.activate(authority)
                    await registry.publish(
                        SessionStatePublication(
                            **authority.model_dump(),
                            expected_revision=0,
                            operation_id="payload-cutover-operation",
                            request_fingerprint="c" * 64,
                            payload=_empty_payload(),
                            operation_result=EmptySessionOperationResult(),
                        )
                    )
            await _rewind_generation_schema_to_version_4(connection)
            if payload_surface == "session":
                await connection.execute(
                    """
                    UPDATE platform_sessions SET payload_schema_version = 1
                    WHERE tenant_id = %s AND logical_session_id = %s
                    """,
                    (registration.tenant_id, registration.authority.logical_session_id),
                )
            else:
                await connection.execute(
                    """
                    UPDATE platform_session_operations SET result_schema_version = 1
                    WHERE tenant_id = %s AND logical_session_id = %s
                    """,
                    (registration.tenant_id, registration.authority.logical_session_id),
                )

            with pytest.raises(PlatformSchemaError, match="unsupported encrypted session"):
                await apply_platform_migrations(connection)

            cursor = await connection.execute("SELECT max(version) FROM platform_schema_migrations")
            assert await cursor.fetchone() == (4,)
            cursor = await connection.execute(
                "SELECT to_regclass('platform_checkpoint_generations')"
            )
            assert await cursor.fetchone() == (None,)
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_generation_backfill_runs_as_a_non_bypass_migration_owner() -> None:
    dsn = _dsn()
    suffix = uuid.uuid4().hex[:16]
    role_name = f"phase4c_migration_{suffix}"
    schema = f"generation_owner_{suffix}"
    role = sql.Identifier(role_name)
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN NOBYPASSRLS").format(role))
        await admin.execute(
            sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema),
                role,
            )
        )
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await connection.execute(sql.SQL("SET ROLE {}").format(role))
            await apply_platform_migrations(connection)
            await connection.execute("RESET ROLE")
            async with AsyncExitStack() as stack:
                registry = PostgresSessionRegistry(
                    await _open_pool(stack, isolated_dsn),
                    cipher=_cipher(),
                    operation_timeout_seconds=2.0,
                )
                registration = _registration("migration_owner", "AD_migration_owner")
                original = await registry.register_and_acquire(
                    registration,
                    _lease("migration-owner"),
                    payload=_empty_payload(),
                )
            await connection.execute(sql.SQL("SET ROLE {}").format(role))
            await _rewind_generation_schema_to_version_4(connection)
            await apply_platform_migrations(connection)
            await connection.execute("RESET ROLE")

            cursor = await connection.execute(
                """
                SELECT storage_thread_id FROM platform_checkpoint_generations
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
            expected = CheckpointBinding(
                tenant_id=registration.tenant_id,
                logical_session_id=registration.authority.logical_session_id,
                deployment_id=registration.deployment_id,
                graph_contract=registration.graph_contract,
                thread_id=original.checkpoint_namespace,
            ).storage_thread_id
            assert await cursor.fetchone() == (expected,)
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            await admin.execute(sql.SQL("DROP ROLE {}").format(role))


@pytest.mark.postgres
async def test_generation_upgrade_rejects_missing_schema4_checkpoint_constraint() -> None:
    dsn = _dsn()
    schema = f"generation_drift_{uuid.uuid4().hex[:16]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await apply_platform_migrations(connection)
            await connection.execute("DROP TABLE platform_checkpoint_write_manifests")
            await connection.execute("DROP TABLE platform_checkpoint_generations CASCADE")
            await connection.execute(
                """
                ALTER TABLE platform_session_operations
                DROP CONSTRAINT platform_session_operations_one_receipt_per_revision
                """
            )
            await connection.execute(
                """
                ALTER TABLE platform_sessions
                    DROP CONSTRAINT platform_sessions_checkpoint_matches_generation
                """
            )
            await connection.execute("DELETE FROM platform_schema_migrations WHERE version >= 5")

            with pytest.raises(UndefinedObject, match="checkpoint_matches_fence"):
                await apply_platform_migrations(connection)

            cursor = await connection.execute(
                "SELECT to_regclass('platform_checkpoint_generations')"
            )
            assert await cursor.fetchone() == (None,)
            cursor = await connection.execute("SELECT max(version) FROM platform_schema_migrations")
            assert await cursor.fetchone() == (4,)
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_platform_migration_preserves_an_idle_connection_transaction_mode() -> None:
    dsn = _dsn()
    schema = f"migration_tx_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn) as migration_connection:
            assert not migration_connection.autocommit
            await apply_platform_migrations(migration_connection)
            assert not migration_connection.autocommit
            async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as observer:
                await require_platform_schema_version(
                    observer,
                    PLATFORM_SESSION_SCHEMA_VERSION,
                )
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_platform_migration_rejects_an_active_caller_transaction() -> None:
    dsn = _dsn()
    schema = f"migration_active_tx_{uuid.uuid4().hex[:16]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn) as migration_connection:
            await migration_connection.execute("SELECT 1")
            with pytest.raises(PlatformSchemaError, match="must be idle"):
                await apply_platform_migrations(migration_connection)
            await migration_connection.rollback()
            await apply_platform_migrations(migration_connection)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as observer:
            await require_platform_schema_version(observer, PLATFORM_SESSION_SCHEMA_VERSION)
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_platform_migration_retries_after_vendor_signature_repair() -> None:
    dsn = _dsn()
    schema = f"migration_vendor_retry_{uuid.uuid4().hex[:16]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await connection.execute(
                "CREATE TABLE checkpoint_migrations (v integer PRIMARY KEY, unexpected text)"
            )
            with pytest.raises(PlatformSchemaError, match="signature"):
                await apply_platform_migrations(connection)
            cursor = await connection.execute("SELECT max(v) FROM checkpoint_migrations")
            assert await cursor.fetchone() == (9,)
            cursor = await connection.execute("SELECT to_regclass('platform_schema_migrations')")
            assert await cursor.fetchone() == (None,)

            await connection.execute("ALTER TABLE checkpoint_migrations DROP COLUMN unexpected")
            await apply_platform_migrations(connection)
            await require_platform_schema_version(connection, PLATFORM_SESSION_SCHEMA_VERSION)
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_database_rejects_legacy_session_and_receipt_payload_versions() -> None:
    dsn = _dsn()
    suffix = uuid.uuid4().hex
    registration = _registration(f"schema_v2_{suffix}", f"AD_schema_v2_{suffix}")
    authority = _lease_authority(registration, "schema-v2-owner")
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration, _lease("schema-v2-owner"), payload=_empty_payload()
        )
        await registry.activate(authority)
        await registry.publish(
            SessionStatePublication(
                **authority.model_dump(),
                expected_revision=0,
                operation_id="schema-v2-operation",
                request_fingerprint="d" * 64,
                payload=_empty_payload(),
                operation_result=EmptySessionOperationResult(),
            )
        )

    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await connection.execute(
            "SELECT set_config('agnostic_market.tenant_id', %s, false)",
            (registration.tenant_id,),
        )
        with pytest.raises(CheckViolation):
            await connection.execute(
                """
                UPDATE platform_sessions SET payload_schema_version = 1
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
        with pytest.raises(CheckViolation):
            await connection.execute(
                """
                UPDATE platform_session_operations SET result_schema_version = 1
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("column", "replacement"),
    (
        ("name", "divergent_platform_session_registry"),
        ("checksum", "0" * 64),
    ),
)
async def test_platform_schema_gate_rejects_divergent_migration_history(
    column: str,
    replacement: str,
) -> None:
    dsn = _dsn()
    schema = f"migration_drift_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await apply_platform_migrations(connection)
            await connection.execute(
                sql.SQL("UPDATE platform_schema_migrations SET {} = %s WHERE version = 1").format(
                    sql.Identifier(column)
                ),
                (replacement,),
            )
            with pytest.raises(PlatformSchemaError, match="does not match repository history"):
                await require_platform_schema_version(
                    connection,
                    PLATFORM_SESSION_SCHEMA_VERSION,
                )
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_room_authority_migration_refuses_to_invent_identity_for_v1_rows() -> None:
    dsn = _dsn()
    schema = f"room_authority_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await apply_platform_migrations(connection)

        async with AsyncExitStack() as stack:
            registry = PostgresSessionRegistry(
                await _open_pool(stack, isolated_dsn),
                cipher=_cipher(),
                operation_timeout_seconds=2.0,
            )
            await registry.register_and_acquire(
                _registration("acme_store", "AD_v1"),
                _lease("lease-v1"),
                payload=_empty_payload(),
            )

        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await connection.execute("DELETE FROM platform_schema_migrations WHERE version >= 2")
            await connection.execute(
                """
                ALTER TABLE platform_sessions
                    DROP CONSTRAINT platform_sessions_open_requires_lease,
                    DROP CONSTRAINT platform_sessions_open_requires_positive_fence,
                    DROP CONSTRAINT platform_sessions_checkpoint_matches_generation,
                    DROP CONSTRAINT platform_sessions_lease_starts_before_expiry
                """
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET lease_owner_id = NULL,
                    lease_expires_at = NULL,
                    fencing_generation = 0
                """
            )
            await connection.execute("ALTER TABLE platform_sessions DROP COLUMN transport_room_id")

            with pytest.raises(NotNullViolation):
                await apply_platform_migrations(connection)

            cursor = await connection.execute(
                "SELECT version FROM platform_schema_migrations ORDER BY version"
            )
            assert [row[0] for row in await cursor.fetchall()] == [1]
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.postgres
async def test_initial_lease_migration_refuses_ownerless_open_rows() -> None:
    dsn = _dsn()
    schema = f"initial_lease_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with AsyncExitStack() as stack:
            async with await AsyncConnection.connect(
                isolated_dsn,
                autocommit=True,
            ) as migration_connection:
                await apply_platform_migrations(migration_connection)
            registry = PostgresSessionRegistry(
                await _open_pool(stack, isolated_dsn),
                cipher=_cipher(),
                operation_timeout_seconds=2.0,
            )
            await registry.register_and_acquire(
                _registration("ownerless_store", "AD_ownerless"),
                _lease("ownerless-original-owner"),
                payload=_empty_payload(),
            )

        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await connection.execute("DELETE FROM platform_schema_migrations WHERE version >= 3")
            await connection.execute(
                """
                ALTER TABLE platform_sessions
                    DROP CONSTRAINT platform_sessions_open_requires_lease,
                    DROP CONSTRAINT platform_sessions_open_requires_positive_fence,
                    DROP CONSTRAINT platform_sessions_checkpoint_matches_generation,
                    DROP CONSTRAINT platform_sessions_lease_starts_before_expiry
                """
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET lease_owner_id = NULL,
                    lease_expires_at = NULL,
                    fencing_generation = 0
                """
            )

            with pytest.raises(CheckViolation):
                await apply_platform_migrations(connection)

            cursor = await connection.execute(
                "SELECT version FROM platform_schema_migrations ORDER BY version"
            )
            assert [row[0] for row in await cursor.fetchall()] == [1, 2]
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _registration(
    tenant_id: str,
    logical_session_id: str,
) -> SessionRegistration:
    return SessionRegistration(
        tenant_id=tenant_id,
        authority=AdmittedSessionAuthority(
            logical_session_id=logical_session_id,
            transport=TransportAuthority(
                provider="livekit",
                room_id=f"RM_{tenant_id}",
                assignment_id=f"AJ_{tenant_id}",
                worker_id=f"AW_{tenant_id}",
            ),
        ),
        deployment_id="deployment-a",
        graph_contract="graph-a",
        config_version="config-a",
        principal_generation=0,
        session_revision=0,
        retention_seconds=3600.0,
    )


def _lease(
    owner_id: str,
    *,
    duration_seconds: float = 30.0,
) -> SessionLeaseRequest:
    return SessionLeaseRequest(
        lease_owner_id=owner_id,
        duration_seconds=duration_seconds,
    )


def _renewal(
    registration: SessionRegistration,
    owner_id: str,
    *,
    duration_seconds: float = 30.0,
    fencing_generation: int = 1,
) -> SessionLeaseRenewal:
    return SessionLeaseRenewal(
        tenant_id=registration.tenant_id,
        authority=registration.authority,
        deployment_id=registration.deployment_id,
        graph_contract=registration.graph_contract,
        config_version=registration.config_version,
        lease_owner_id=owner_id,
        fencing_generation=fencing_generation,
        duration_seconds=duration_seconds,
    )


def _lease_authority(
    registration: SessionRegistration,
    owner_id: str,
    *,
    fencing_generation: int = 1,
) -> SessionLeaseAuthority:
    return SessionLeaseAuthority(
        tenant_id=registration.tenant_id,
        authority=registration.authority,
        deployment_id=registration.deployment_id,
        graph_contract=registration.graph_contract,
        config_version=registration.config_version,
        lease_owner_id=owner_id,
        fencing_generation=fencing_generation,
    )


@pytest.mark.postgres
async def test_durable_close_fences_writers_deletes_state_and_replays_safely() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("close_store", f"AD_close_{uuid.uuid4().hex}")
    live_authority = _lease_authority(registration, "live-owner")
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registry = PostgresSessionRegistry(
            pool,
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("live-owner"),
            payload=_empty_payload(),
        )
        await registry.activate(live_authority)
        await registry.publish(
            SessionStatePublication(
                **live_authority.model_dump(),
                expected_revision=0,
                operation_id="pre-close-publication",
                request_fingerprint="d" * 64,
                payload=DurableSessionPayload(guest_order_refs=("ORD-CLOSE",)),
                operation_result=EmptySessionOperationResult(),
            )
        )
        coordinator = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=30.0,
            tombstone_retention_seconds=3600.0,
        )

        claim = await coordinator.begin_live_close(
            live_authority,
            "close-operation",
            "close-lease-owner",
        )
        replayed = await coordinator.begin_live_close(
            live_authority,
            "close-operation",
            "close-lease-owner",
        )

        assert replayed == claim
        assert claim.record.lifecycle is SessionLifecycle.CLOSING
        assert claim.authority.fencing_generation == live_authority.fencing_generation + 1
        with pytest.raises(LeaseAdmissionError) as closing_restore:
            await registry.restore(live_authority)
        assert closing_restore.value.reason is LeaseAdmissionReason.LIFECYCLE_REJECTED
        with pytest.raises(LeaseAdmissionError) as stale_writer:
            await registry.publish(
                SessionStatePublication(
                    **live_authority.model_dump(),
                    expected_revision=1,
                    operation_id="stale-publication",
                    request_fingerprint="e" * 64,
                    payload=_empty_payload(),
                    operation_result=EmptySessionOperationResult(),
                )
            )
        assert stale_writer.value.reason is LeaseAdmissionReason.LIFECYCLE_REJECTED

        closed = await coordinator.finalize(claim)
        duplicate = await coordinator.finalize(claim)

        assert duplicate == closed
        assert closed.lifecycle is SessionLifecycle.CLOSED
        assert closed.envelope is None
        assert closed.lease_owner_id is None
        assert closed.close_operation_id == "close-operation"
        with pytest.raises(LeaseAdmissionError) as closed_restore:
            await registry.restore(live_authority)
        assert closed_restore.value.reason is LeaseAdmissionReason.LIFECYCLE_REJECTED
        async with pool.connection() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                (registration.tenant_id,),
            )
            operation_count = await (
                await connection.execute(
                    """
                    SELECT count(*) FROM platform_session_operations
                    WHERE tenant_id = %s AND logical_session_id = %s
                    """,
                    (registration.tenant_id, registration.authority.logical_session_id),
                )
            ).fetchone()
            generation_count = await (
                await connection.execute(
                    """
                    SELECT count(*) FROM platform_checkpoint_generations
                    WHERE tenant_id = %s AND logical_session_id = %s
                    """,
                    (registration.tenant_id, registration.authority.logical_session_id),
                )
            ).fetchone()
        assert operation_count == (0,)
        assert generation_count == (0,)


@pytest.mark.postgres
async def test_durable_close_reclaims_its_expired_lease_before_finalization() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("close_progress_store", f"AD_close_{uuid.uuid4().hex}")
    live_authority = _lease_authority(registration, "live-owner")
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registry = PostgresSessionRegistry(
            pool,
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("live-owner"),
            payload=_empty_payload(),
        )
        coordinator = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=30.0,
            tombstone_retention_seconds=3600.0,
        )
        claim = await coordinator.begin_live_close(
            live_authority,
            "close-operation",
            "close-lease-owner",
        )
        refreshed = await registry.refresh_close(
            claim.authority,
            duration_seconds=1.0,
        )
        assert refreshed.authority.fencing_generation == claim.authority.fencing_generation
        assert refreshed.record.lease_expires_at is not None
        assert claim.record.lease_expires_at is not None
        assert refreshed.record.lease_expires_at >= claim.record.lease_expires_at
        claim = refreshed
        async with pool.connection() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, true)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET created_at = created_at - interval '1 hour',
                    lease_expires_at = clock_timestamp() - interval '1 second'
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )

        closed = await coordinator.finalize(claim)

        assert closed.lifecycle is SessionLifecycle.CLOSED
        assert closed.close_operation_id == claim.authority.operation_id
        assert closed.fencing_generation == claim.authority.fencing_generation + 1


@pytest.mark.postgres
async def test_reaper_claims_only_expired_sessions_through_the_close_coordinator() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    expired = _registration("reaper_store", f"AD_expired_{uuid.uuid4().hex}")
    live = _registration("reaper_store", f"AD_live_{uuid.uuid4().hex}")
    foreign = _registration("foreign_reaper_store", expired.authority.logical_session_id)
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registry = PostgresSessionRegistry(
            pool,
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            expired,
            _lease("expired-owner", duration_seconds=0.01),
            payload=_empty_payload(),
        )
        await registry.register_and_acquire(
            live,
            _lease("live-owner"),
            payload=_empty_payload(),
        )
        await registry.register_and_acquire(
            foreign,
            _lease("foreign-expired-owner", duration_seconds=0.01),
            payload=_empty_payload(),
        )
        await asyncio.sleep(0.02)
        coordinator = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=30.0,
            tombstone_retention_seconds=3600.0,
        )

        assert await coordinator.reap_expired("reaper_store") == 1
        expired_record = await registry.get(
            expired.tenant_id,
            expired.authority.logical_session_id,
        )
        live_record = await registry.get(live.tenant_id, live.authority.logical_session_id)
        foreign_record = await registry.get(
            foreign.tenant_id,
            foreign.authority.logical_session_id,
        )

        assert expired_record is not None
        assert expired_record.lifecycle is SessionLifecycle.CLOSED
        assert live_record is not None
        assert live_record.lifecycle is SessionLifecycle.OPENING
        assert foreign_record is not None
        assert foreign_record.lifecycle is SessionLifecycle.OPENING


@pytest.mark.postgres
async def test_expired_close_reclaim_keeps_operation_identity_and_advances_owner_fence() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("close_reclaim_store", f"AD_reclaim_{uuid.uuid4().hex}")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("expired-live-owner", duration_seconds=0.01),
            payload=_empty_payload(),
        )
        await asyncio.sleep(0.02)
        candidate = (await registry.expired_sessions(registration.tenant_id, limit=10))[0]
        first = await registry.claim_expired(
            candidate,
            SessionCloseRequest(
                operation_id="stable-reap-operation",
                lease_owner_id="reaper-owner-a",
                duration_seconds=0.01,
            ),
        )

        assert await registry.expired_sessions(registration.tenant_id, limit=10) == ()
        await asyncio.sleep(0.02)
        abandoned = (await registry.expired_sessions(registration.tenant_id, limit=10))[0]
        assert abandoned.close_operation_id == "stable-reap-operation"
        second = await registry.claim_expired(
            abandoned,
            SessionCloseRequest(
                operation_id=abandoned.close_operation_id,
                lease_owner_id="reaper-owner-b",
                duration_seconds=30.0,
            ),
        )

        assert second.authority.operation_id == first.authority.operation_id
        assert second.authority.lease_owner_id == "reaper-owner-b"
        assert second.authority.fencing_generation == first.authority.fencing_generation + 1
        with pytest.raises(SessionCloseError) as stale_cleanup:
            await registry.close_checkpoint_generations(first.authority)
        assert stale_cleanup.value.reason is SessionCloseReason.WRONG_CLOSE_OWNER


@pytest.mark.postgres
async def test_lost_begin_acknowledgement_recognizes_reaper_completed_close() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("close_ack_store", f"AD_close_ack_{uuid.uuid4().hex}")
    authority = _lease_authority(registration, "close-ack-live-owner")
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registry = PostgresSessionRegistry(
            pool,
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("close-ack-live-owner"),
            payload=_empty_payload(),
        )
        coordinator = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=0.01,
            tombstone_retention_seconds=3600.0,
        )
        closer = coordinator.bind(authority)

        await coordinator.begin_live_close(
            authority,
            closer.operation_id,
            closer.lease_owner_id,
        )
        await asyncio.sleep(0.02)
        reaper = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=30.0,
            tombstone_retention_seconds=3600.0,
        )
        assert await reaper.reap_expired(registration.tenant_id) == 1

        await closer.begin_close()
        await closer.finalize_close()

        closed = await registry.get(
            registration.tenant_id,
            registration.authority.logical_session_id,
        )
        assert closed is not None
        assert closed.lifecycle is SessionLifecycle.CLOSED
        assert closed.close_operation_id == closer.operation_id


@pytest.mark.postgres
async def test_close_verifies_checkpoint_absence_before_accepting_a_deletion_marker() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("partial_close_store", f"AD_partial_{uuid.uuid4().hex}")
    authority = _lease_authority(registration, "partial-live-owner")
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registry = PostgresSessionRegistry(
            pool,
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("partial-live-owner"),
            payload=_empty_payload(),
        )
        generation = (await registry.checkpoint_generations(authority))[0]
        saver = build_checkpointer(
            FencedPostgresCheckpointSaver(
                pool,
                authority=authority,
                cipher=_cipher(),
                serde=build_checkpoint_serializer(),
            ),
            synchronous_operations=False,
            cipher=_cipher(),
        )
        saver.bind_checkpoint_contract(
            ReasoningState.model_fields,
            binding=generation.binding,
            io_timeout_seconds=2.0,
        )
        saved_config = await saver.aput(generation.binding.config, empty_checkpoint(), {}, {})
        await saver.aput_writes(
            saved_config,
            (("__interrupt__", "pending confirmation"),),
            "interrupted-task",
        )
        coordinator = DurableSessionLifecycleCoordinator(
            pool,
            registry,
            _cipher(),
            checkpoint_io_timeout_seconds=2.0,
            close_lease_duration_seconds=30.0,
            tombstone_retention_seconds=3600.0,
        )
        claim = await coordinator.begin_live_close(
            authority,
            "partial-close-operation",
            "partial-close-owner",
        )
        assert await coordinator.checkpoint_has_pending_interrupt(claim)
        await registry.record_close_checkpoint_deletion(
            claim.authority,
            generation.checkpoint_namespace,
        )

        with pytest.raises(SessionCloseError) as incomplete:
            await registry.finalize_close(
                claim.authority,
                tombstone_retention_seconds=3600.0,
            )
        assert incomplete.value.reason is SessionCloseReason.CLEANUP_INCOMPLETE

        closed = await coordinator.finalize(claim)
        assert closed.lifecycle is SessionLifecycle.CLOSED


@pytest.mark.postgres
async def test_concurrent_close_claims_have_one_fence_owner() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("close_race_store", f"AD_close_race_{uuid.uuid4().hex}")
    authority = _lease_authority(registration, "race-live-owner")
    async with AsyncExitStack() as stack:
        pool = await _open_pool(stack, dsn)
        registries = [
            PostgresSessionRegistry(pool, cipher=_cipher(), operation_timeout_seconds=2.0)
            for _ in range(2)
        ]
        await registries[0].register_and_acquire(
            registration,
            _lease("race-live-owner"),
            payload=_empty_payload(),
        )

        outcomes = await asyncio.gather(
            registries[0].begin_close(
                authority,
                SessionCloseRequest(
                    operation_id="close-a",
                    lease_owner_id="close-owner-a",
                    duration_seconds=30.0,
                ),
            ),
            registries[1].begin_close(
                authority,
                SessionCloseRequest(
                    operation_id="close-b",
                    lease_owner_id="close-owner-b",
                    duration_seconds=30.0,
                ),
            ),
            return_exceptions=True,
        )

        claims = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        assert len(claims) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], SessionCloseError)
        assert errors[0].reason is SessionCloseReason.WRONG_CLOSE_OWNER


@pytest.mark.postgres
async def test_platform_migrations_and_registry_are_cross_worker_and_tenant_scoped() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
        await apply_platform_migrations(migration_connection)
        await require_platform_schema_version(
            migration_connection,
            PLATFORM_SESSION_SCHEMA_VERSION,
        )
        with pytest.raises(PlatformSchemaError, match="runtime"):
            await require_platform_schema_version(
                migration_connection,
                PLATFORM_SESSION_SCHEMA_VERSION + 1,
            )

    sentinel = b"UNIQUE_PLAINTEXT_SENTINEL"
    sentinel_payload = DurableSessionPayload(guest_order_refs=(sentinel.decode(),))
    async with AsyncExitStack() as stack:
        first = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        second = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        acme = _registration(
            "acme_store",
            "AD_shared",
        )
        demo = _registration(
            "demo_shop",
            "AD_shared",
        )

        created = await first.register_and_acquire(
            acme,
            _lease("lease-acme"),
            payload=sentinel_payload,
        )
        await second.register_and_acquire(
            demo,
            _lease("lease-demo"),
            payload=_empty_payload(),
        )
        reread = await second.get("acme_store", "AD_shared")

        assert reread == created
        assert created.lifecycle is SessionLifecycle.OPENING
        assert created.fencing_generation == 1
        assert created.lease_owner_id == "lease-acme"
        assert created.lease_expires_at is not None
        assert created.checkpoint_namespace == "AD_shared::fence::1"
        assert reread.envelope is not None
        context = SessionEnvelopeContext(
            tenant_id="acme_store",
            logical_session_id="AD_shared",
            checkpoint_namespace="AD_shared::fence::1",
            payload_schema_version=SESSION_PAYLOAD_SCHEMA_VERSION,
            payload_purpose="session_projection",
            session_revision=0,
        )
        assert (
            AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY}).decrypt(
                reread.envelope,
                context,
            )
            == sentinel_payload.to_bytes()
        )
        assert await first.get("demo_shop", "AD_shared") is not None
        assert await first.get("unknown_store", "AD_shared") is None
        with pytest.raises(LeaseAdmissionError) as conflict:
            await second.register_and_acquire(
                acme,
                _lease("another-owner"),
                payload=sentinel_payload,
            )
        assert conflict.value.reason is LeaseAdmissionReason.SESSION_EXISTS

    async with await AsyncConnection.connect(dsn, autocommit=True) as inspection:
        cursor = await inspection.execute(
            """
            SELECT encrypted_payload
            FROM platform_sessions
            WHERE tenant_id = %s AND logical_session_id = %s
            """,
            ("acme_store", "AD_shared"),
        )
        row = await cursor.fetchone()

    assert row is not None
    assert sentinel not in bytes(row[0])


@pytest.mark.postgres
async def test_initial_registration_and_lease_have_one_concurrent_winner() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("lease_race_store", "AD_lease_race")
    async with AsyncExitStack() as stack:
        first = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        second = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        outcomes = await asyncio.gather(
            first.register_and_acquire(
                registration,
                _lease("lease-race-a"),
                payload=_empty_payload(),
            ),
            second.register_and_acquire(
                registration,
                _lease("lease-race-b"),
                payload=_empty_payload(),
            ),
            return_exceptions=True,
        )

        records = [outcome for outcome in outcomes if isinstance(outcome, SessionRegistryRecord)]
        rejections = [outcome for outcome in outcomes if isinstance(outcome, LeaseAdmissionError)]
        assert len(records) == 1
        assert len(rejections) == 1
        assert rejections[0].reason is LeaseAdmissionReason.SESSION_EXISTS
        assert records[0].lease_owner_id in {"lease-race-a", "lease-race-b"}
        assert records[0].fencing_generation == 1
        assert records[0].checkpoint_namespace == "AD_lease_race::fence::1"


@pytest.mark.postgres
async def test_renewal_uses_database_time_without_advancing_the_fence() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("lease_renew_store", "AD_lease_renew")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        created = await registry.register_and_acquire(
            registration,
            _lease("lease-renew-owner"),
            payload=_empty_payload(),
        )
        await asyncio.sleep(0.01)
        renewed = await registry.renew(_renewal(registration, "lease-renew-owner"))

    async with await AsyncConnection.connect(dsn, autocommit=True) as observer:
        cursor = await observer.execute("SELECT clock_timestamp()")
        row = await cursor.fetchone()
    assert row is not None
    database_now = row[0]
    assert renewed.lease_expires_at is not None
    assert created.lease_expires_at is not None
    assert renewed.lease_expires_at > created.lease_expires_at
    assert 0 < (renewed.lease_expires_at - database_now).total_seconds() <= 30.0
    assert renewed.fencing_generation == created.fencing_generation == 1
    assert renewed.checkpoint_namespace == created.checkpoint_namespace


@pytest.mark.postgres
async def test_opening_session_is_published_active_by_its_current_lease_owner() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("activate_store", "AD_activate")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        opening = await registry.register_and_acquire(
            registration,
            _lease("activate-owner"),
            payload=_empty_payload(),
        )
        active = await registry.activate(_lease_authority(registration, "activate-owner"))
        repeated = await registry.activate(_lease_authority(registration, "activate-owner"))

    assert opening.lifecycle is SessionLifecycle.OPENING
    assert active.lifecycle is SessionLifecycle.ACTIVE
    assert repeated == active
    assert active.fencing_generation == opening.fencing_generation
    assert active.checkpoint_namespace == opening.checkpoint_namespace


@pytest.mark.postgres
async def test_fenced_session_state_publication_is_visible_and_replay_safe() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("state_store", "AD_state_store")
    authority = _lease_authority(registration, "state-owner")
    published_payload = DurableSessionPayload(guest_order_refs=("ORD-9001",))
    publication = SessionStatePublication(
        **authority.model_dump(),
        expected_revision=0,
        operation_id="operation-1",
        request_fingerprint="a" * 64,
        payload=published_payload,
        operation_result=EmptySessionOperationResult(),
    )
    async with AsyncExitStack() as stack:
        first = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        second = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await first.register_and_acquire(
            registration,
            _lease("state-owner"),
            payload=_empty_payload(),
        )
        await first.activate(authority)

        committed = await first.publish(publication)
        restored = await second.restore(authority)
        replayed = await second.publish(publication)

        with pytest.raises(SessionStateWriteError) as conflict:
            await second.publish(publication.model_copy(update={"request_fingerprint": "b" * 64}))
        with pytest.raises(SessionStateWriteError) as stale:
            await second.publish(
                publication.model_copy(
                    update={
                        "operation_id": "operation-2",
                        "request_fingerprint": "c" * 64,
                    }
                )
            )

    assert committed.record.session_revision == 1
    assert committed.payload == published_payload
    assert restored.record.session_revision == 1
    assert restored.payload == published_payload
    assert replayed.replayed is True
    assert replayed.operation_revision == 1
    assert replayed.operation_result == EmptySessionOperationResult()
    assert replayed.record.session_revision == 1
    assert conflict.value.reason is SessionStateWriteReason.OPERATION_CONFLICT
    assert stale.value.reason is SessionStateWriteReason.STALE_REVISION


@pytest.mark.postgres
async def test_session_projection_cannot_replay_at_a_newer_revision() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("projection_replay_store", "AD_projection_replay")
    authority = _lease_authority(registration, "projection-replay-owner")
    publication = SessionStatePublication(
        **authority.model_dump(),
        expected_revision=0,
        operation_id="projection-replay-operation",
        request_fingerprint="a" * 64,
        payload=DurableSessionPayload(guest_order_refs=("ORD-9001",)),
        operation_result=EmptySessionOperationResult(),
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        initial = await registry.register_and_acquire(
            registration,
            _lease("projection-replay-owner"),
            payload=_empty_payload(),
        )
        assert initial.envelope is not None
        await registry.activate(authority)
        await registry.publish(publication)
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET envelope_format = %s,
                    envelope_key_version = %s,
                    payload_schema_version = %s,
                    envelope_nonce = %s,
                    encrypted_payload = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (
                    initial.envelope.format,
                    initial.envelope.key_version,
                    initial.envelope.payload_schema_version,
                    initial.envelope.nonce,
                    initial.envelope.ciphertext,
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )

        with pytest.raises(SessionRestoreError) as rejected:
            await registry.restore(authority)

    assert rejected.value.reason is SessionRestoreReason.DECRYPTION_FAILED


@pytest.mark.postgres
async def test_operation_receipt_revision_is_unique_and_authenticated() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("receipt_revision_store", "AD_receipt_revision")
    authority = _lease_authority(registration, "receipt-revision-owner")
    first_publication = SessionStatePublication(
        **authority.model_dump(),
        expected_revision=0,
        operation_id="receipt-revision-one",
        request_fingerprint="1" * 64,
        payload=DurableSessionPayload(guest_order_refs=("ORD-9001",)),
        operation_result=EmptySessionOperationResult(),
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("receipt-revision-owner"),
            payload=_empty_payload(),
        )
        await registry.activate(authority)
        first = await registry.publish(first_publication)
        await registry.publish(
            first_publication.model_copy(
                update={
                    "expected_revision": first.record.session_revision,
                    "operation_id": "receipt-revision-two",
                    "request_fingerprint": "2" * 64,
                }
            )
        )
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            with pytest.raises(UniqueViolation):
                await connection.execute(
                    """
                    INSERT INTO platform_session_operations (
                        tenant_id, logical_session_id, operation_id,
                        request_fingerprint, committed_revision,
                        committed_checkpoint_namespace, result_envelope_format,
                        result_envelope_key_version, result_schema_version,
                        result_envelope_nonce, encrypted_result
                    )
                    SELECT tenant_id, logical_session_id, 'receipt-revision-duplicate',
                        request_fingerprint, committed_revision,
                        committed_checkpoint_namespace, result_envelope_format,
                        result_envelope_key_version, result_schema_version,
                        result_envelope_nonce, encrypted_result
                    FROM platform_session_operations
                    WHERE tenant_id = %s AND logical_session_id = %s
                      AND operation_id = 'receipt-revision-one'
                    """,
                    (registration.tenant_id, registration.authority.logical_session_id),
                )
            await connection.execute(
                """
                DELETE FROM platform_session_operations
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'receipt-revision-two'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
            await connection.execute(
                """
                UPDATE platform_session_operations SET committed_revision = 2
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'receipt-revision-one'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )

        with pytest.raises(
            SessionRegistryDataError,
            match="operation result could not be authenticated",
        ):
            await registry.publish(first_publication)


@pytest.mark.postgres
async def test_checkpoint_revision_reconciliation_distinguishes_seed_and_missing_state() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
    registration = _registration("revision_seed_store", "AD_revision_seed")
    authority = _lease_authority(registration, "revision-seed-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration,
            _lease("revision-seed-owner"),
            payload=_empty_payload(),
        )

        seed = await registry.reconcile_checkpoint_revision(authority, None)
        assert seed.disposition is CheckpointRevisionDisposition.SEED_REQUIRED
        assert seed.checkpoint_revision is None
        assert seed.operations == ()

        await registry.activate(authority)
        with pytest.raises(SessionRestoreError) as missing:
            await registry.reconcile_checkpoint_revision(authority, None)
        with pytest.raises(SessionRestoreError) as ahead:
            await registry.reconcile_checkpoint_revision(authority, 1)
        with pytest.raises(SessionRestoreError) as malformed:
            await registry.reconcile_checkpoint_revision(authority, "0")
        current = await registry.reconcile_checkpoint_revision(authority, 0)

    assert missing.value.reason is SessionRestoreReason.CHECKPOINT_INVALID
    assert ahead.value.reason is SessionRestoreReason.REVISION_MISMATCH
    assert malformed.value.reason is SessionRestoreReason.CHECKPOINT_INVALID
    assert current.disposition is CheckpointRevisionDisposition.CURRENT
    assert current.operations == ()


@pytest.mark.postgres
async def test_session_ahead_reconciliation_requires_contiguous_typed_receipts() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
    registration = _registration("revision_gap_store", "AD_revision_gap")
    authority = _lease_authority(registration, "revision-gap-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration,
            _lease("revision-gap-owner"),
            payload=_empty_payload(),
        )
        await registry.activate(authority)
        first = await registry.publish(
            SessionStatePublication(
                **authority.model_dump(),
                expected_revision=0,
                operation_id="revision-operation-1",
                request_fingerprint="1" * 64,
                payload=DurableSessionPayload(guest_order_refs=("ORD-9001",)),
                operation_result=EmptySessionOperationResult(),
            )
        )
        await registry.publish(
            SessionStatePublication(
                **authority.model_dump(),
                expected_revision=first.record.session_revision,
                operation_id="revision-operation-2",
                request_fingerprint="2" * 64,
                payload=DurableSessionPayload(guest_order_refs=("ORD-9001", "ORD-9002")),
                operation_result=EmptySessionOperationResult(),
            )
        )

        reconciled = await registry.reconcile_checkpoint_revision(authority, 0)
        assert reconciled.disposition is CheckpointRevisionDisposition.SESSION_AHEAD
        assert reconciled.payload.guest_order_refs == ("ORD-9001", "ORD-9002")
        assert [item.operation_id for item in reconciled.operations] == [
            "revision-operation-1",
            "revision-operation-2",
        ]
        assert [item.committed_revision for item in reconciled.operations] == [1, 2]
        assert all(
            item.committed_checkpoint_namespace == reconciled.record.checkpoint_namespace
            for item in reconciled.operations
        )

        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            cursor = await connection.execute(
                """
                SELECT result_envelope_nonce, encrypted_result
                FROM platform_session_operations
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
            original_result = await cursor.fetchone()
            assert original_result is not None
            await connection.execute(
                """
                UPDATE platform_session_operations
                SET encrypted_result = set_byte(
                    encrypted_result, 0, get_byte(encrypted_result, 0) # 1
                )
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
        with pytest.raises(SessionRestoreError) as corrupted:
            await registry.reconcile_checkpoint_revision(authority, 0)
        assert corrupted.value.reason is SessionRestoreReason.DECRYPTION_FAILED
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_session_operations
                SET result_envelope_nonce = %s, encrypted_result = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (
                    original_result[0],
                    original_result[1],
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )
            operation_context = SessionEnvelopeContext(
                tenant_id=registration.tenant_id,
                logical_session_id=registration.authority.logical_session_id,
                checkpoint_namespace=reconciled.record.checkpoint_namespace,
                payload_schema_version=SESSION_OPERATION_RESULT_SCHEMA_VERSION,
                payload_purpose="operation_result",
                session_revision=2,
                operation_id="revision-operation-2",
                request_fingerprint="2" * 64,
            )
            malformed = _cipher().encrypt(b'{"schema_version":2}', operation_context)
            await connection.execute(
                """
                UPDATE platform_session_operations
                SET result_envelope_nonce = %s, encrypted_result = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (
                    malformed.nonce,
                    malformed.ciphertext,
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )
        with pytest.raises(SessionRestoreError) as malformed_payload:
            await registry.reconcile_checkpoint_revision(authority, 0)
        assert malformed_payload.value.reason is SessionRestoreReason.PAYLOAD_SCHEMA_INVALID

        mismatched_receipt = SessionOperationReceiptPayload(
            operation_id="different-operation",
            request_fingerprint="2" * 64,
            result=EmptySessionOperationResult(),
        )
        mismatched = _cipher().encrypt(mismatched_receipt.to_bytes(), operation_context)
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_session_operations
                SET result_envelope_nonce = %s, encrypted_result = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (
                    mismatched.nonce,
                    mismatched.ciphertext,
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )
        with pytest.raises(SessionRestoreError) as mismatched_authority:
            await registry.reconcile_checkpoint_revision(authority, 0)
        assert mismatched_authority.value.reason is SessionRestoreReason.RECONSTRUCTION_FAILED

        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_session_operations
                SET result_envelope_nonce = %s, encrypted_result = %s
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-2'
                """,
                (
                    original_result[0],
                    original_result[1],
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )
            await connection.execute(
                """
                DELETE FROM platform_session_operations
                WHERE tenant_id = %s AND logical_session_id = %s
                  AND operation_id = 'revision-operation-1'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
        with pytest.raises(SessionRestoreError) as unexplained:
            await registry.reconcile_checkpoint_revision(authority, 0)

    assert unexplained.value.reason is SessionRestoreReason.REVISION_GAP_UNEXPLAINED


@pytest.mark.postgres
async def test_checkpoint_reconciliation_rejects_a_principal_retirement_marker() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
    registration = _registration("retirement_reconcile_store", "AD_retirement_reconcile")
    authority = _lease_authority(registration, "retirement-reconcile-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration,
            _lease("retirement-reconcile-owner"),
            payload=_empty_payload(),
        )
        await registry.activate(authority)
        await _publish_retirement_marker(
            registry,
            authority,
            expected_revision=0,
            transition_id="retirement-reconcile-transition",
        )
        await registry.begin_checkpoint_rotation(
            authority,
            expected_revision=1,
            expected_principal_generation=0,
            transition_id="retirement-reconcile-transition",
        )

        with pytest.raises(SessionRestoreError) as pending:
            await registry.reconcile_checkpoint_revision(authority, 0)

    assert pending.value.reason is SessionRestoreReason.PRINCIPAL_RETIREMENT_PENDING


@pytest.mark.postgres
async def test_lost_retirement_acknowledgement_reconciles_the_committed_marker(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_principal_recovery import _CUST1, _identity_harness
    from verification_helpers import grant_verification

    from agnostic_market.dtos.orchestration import ListOrders

    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        await apply_platform_migrations(connection)
    registration = _registration("acme_store", "AD_lost_retirement_ack")
    authority = _lease_authority(registration, "retirement-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn), cipher=_cipher(), operation_timeout_seconds=2.0
        )
        await registry.register_and_acquire(
            registration, _lease("retirement-owner"), payload=_empty_payload()
        )
        await registry.activate(authority)
        context = _identity_harness(
            config_root, thread_id=registration.authority.logical_session_id
        ).caller_context
        persistence = BoundPostgresSessionStatePersistence(registry, authority)
        context.session_state = SessionStateCoordinator(
            context.cart_store,
            context.recent_orders,
            context.guest_orders,
            persistence=persistence,
        )
        publish = persistence.publish
        lose_ack = True

        async def publish_then_lose_ack(self, **kwargs):
            nonlocal lose_ack
            assert self is persistence
            committed = await publish(**kwargs)
            if lose_ack:
                lose_ack = False
                raise TimeoutError("injected lost acknowledgement")
            return committed

        monkeypatch.setattr(BoundPostgresSessionStatePersistence, "publish", publish_then_lose_ack)
        await grant_verification(context.verification_store)
        with pytest.raises(TimeoutError, match="lost acknowledgement"):
            await context.transition_principal(
                _CUST1, context.verification_store.grants[-1], ListOrders(scope="account")
            )

        committed = await registry.restore(authority)
        assert committed.record.session_revision == 1
        assert committed.payload.principal_retirement is not None
        assert context.session_state.principal_retirement is None
        assert context.session_revision == 0
        assert await context.invalidate_principal_transition() is True
        restored = await registry.restore(authority)
        assert restored.record.session_revision == context.session_revision == 2
        assert restored.payload.principal_retirement is None
        assert context.pending_transition() is None
        assert context.session_state.principal_retirement is None
        assert context.identity_store.current() is None
        assert context.verification_store.grants == []


@pytest.mark.postgres
async def test_session_state_coordinator_restores_a_committed_cross_worker_projection() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("coordinator_store", "AD_coordinator")
    authority = _lease_authority(registration, "coordinator-owner")
    async with AsyncExitStack() as stack:
        first_registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        second_registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await first_registry.register_and_acquire(
            registration,
            _lease("coordinator-owner"),
            payload=_empty_payload(),
        )
        await first_registry.activate(authority)
        initial = await first_registry.restore(authority)
        first = SessionStateCoordinator.reconstruct(
            initial.payload,
            tenant_id=registration.tenant_id,
            session_id=registration.authority.logical_session_id,
            recent_order_max_refs=3,
            session_revision=initial.record.session_revision,
            persistence=BoundPostgresSessionStatePersistence(first_registry, authority),
        )

        committed = await first.apply_cart_mutation(
            "cart-operation-1",
            operation="add",
            sku="SKU-1",
            name="Trail Jacket",
            price_usd="79.00",
            quantity=1,
            pre_confirm_quantity=0,
        )
        visible = await second_registry.restore(authority)
        second = SessionStateCoordinator.reconstruct(
            visible.payload,
            tenant_id=registration.tenant_id,
            session_id=registration.authority.logical_session_id,
            recent_order_max_refs=3,
            session_revision=visible.record.session_revision,
            persistence=BoundPostgresSessionStatePersistence(second_registry, authority),
        )

    assert committed.session_revision == visible.record.session_revision == second.revision == 1
    assert first.cart.view() == second.cart.view()
    assert second.cart.view()[0].quantity == 1


@pytest.mark.postgres
async def test_cross_worker_replay_returns_the_original_result_after_later_state_changes() -> None:
    dsn = _dsn()
    result_sentinel = "UNIQUE_OPERATION_RESULT_SENTINEL"
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("operation_replay_store", "AD_operation_replay")
    authority = _lease_authority(registration, "operation-replay-owner")
    async with AsyncExitStack() as stack:
        first_registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        second_registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await first_registry.register_and_acquire(
            registration,
            _lease("operation-replay-owner"),
            payload=_empty_payload(),
        )
        await first_registry.activate(authority)
        initial = await first_registry.restore(authority)
        first = SessionStateCoordinator.reconstruct(
            initial.payload,
            tenant_id=registration.tenant_id,
            session_id=registration.authority.logical_session_id,
            recent_order_max_refs=3,
            session_revision=initial.record.session_revision,
            persistence=BoundPostgresSessionStatePersistence(first_registry, authority),
        )
        await first.apply_cart_mutation(
            "seed-cart-operation",
            operation="add",
            sku="SKU-1",
            name=result_sentinel,
            price_usd="79.00",
            quantity=2,
            pre_confirm_quantity=0,
        )
        original = await first.apply_cart_mutation(
            "replayed-cart-operation",
            operation="add",
            sku="SKU-1",
            name=result_sentinel,
            price_usd="79.00",
            quantity=1,
            pre_confirm_quantity=2,
        )
        await first.complete_placement("later-placement", "ORD-9001")

        latest = await second_registry.restore(authority)
        async with await AsyncConnection.connect(dsn, autocommit=True) as inspection:
            await inspection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            cursor = await inspection.execute(
                """
                SELECT encrypted_result
                FROM platform_session_operations
                WHERE tenant_id = %s
                  AND logical_session_id = %s
                  AND operation_id = %s
                """,
                (
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                    "replayed-cart-operation",
                ),
            )
            stored_result = await cursor.fetchone()
        assert stored_result is not None
        assert result_sentinel.encode() not in bytes(stored_result[0])
        restored = SessionStateCoordinator.reconstruct(
            latest.payload,
            tenant_id=registration.tenant_id,
            session_id=registration.authority.logical_session_id,
            recent_order_max_refs=3,
            session_revision=latest.record.session_revision,
            persistence=BoundPostgresSessionStatePersistence(second_registry, authority),
        )
        replayed = await restored.apply_cart_mutation(
            "replayed-cart-operation",
            operation="add",
            sku="SKU-1",
            name=result_sentinel,
            price_usd="79.00",
            quantity=1,
            pre_confirm_quantity=2,
        )

    assert original.value == replayed.value
    assert (replayed.value.previous_quantity, replayed.value.final_quantity) == (2, 3)
    assert replayed.session_revision == latest.record.session_revision == 3
    assert restored.cart.is_empty()


@pytest.mark.postgres
async def test_operation_receipt_survives_a_later_session_fence() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("receipt_fence_store", "AD_receipt_fence")
    authority = _lease_authority(registration, "receipt-fence-owner")
    fingerprint = "a" * 64
    publication = SessionStatePublication(
        **authority.model_dump(),
        expected_revision=0,
        operation_id="receipt-operation",
        request_fingerprint=fingerprint,
        payload=_empty_payload(),
        operation_result=EmptySessionOperationResult(),
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("receipt-fence-owner"),
            payload=_empty_payload(),
        )
        await registry.activate(authority)
        committed = await registry.publish(publication)

        next_namespace = "AD_receipt_fence::fence::2"
        next_envelope = _cipher().encrypt(
            committed.payload.to_bytes(),
            SessionEnvelopeContext(
                tenant_id=registration.tenant_id,
                logical_session_id=registration.authority.logical_session_id,
                checkpoint_namespace=next_namespace,
                payload_schema_version=SESSION_PAYLOAD_SCHEMA_VERSION,
                payload_purpose="session_projection",
                session_revision=committed.record.session_revision,
            ),
        )
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET fencing_generation = 2,
                    checkpoint_namespace = %s,
                    envelope_format = %s,
                    envelope_key_version = %s,
                    payload_schema_version = %s,
                    envelope_nonce = %s,
                    encrypted_payload = %s,
                    updated_at = clock_timestamp()
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (
                    next_namespace,
                    next_envelope.format,
                    next_envelope.key_version,
                    next_envelope.payload_schema_version,
                    next_envelope.nonce,
                    next_envelope.ciphertext,
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )

            # Simulate the corresponding inventory switch, not a corrupt current pointer.
            await connection.execute(
                """
                UPDATE platform_checkpoint_generations SET state = 'retired'
                WHERE tenant_id = %s AND logical_session_id = %s AND state = 'current'
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )
            await connection.execute(
                """
                INSERT INTO platform_checkpoint_generations (
                    tenant_id, logical_session_id, checkpoint_namespace, storage_thread_id,
                    fencing_generation,
                    principal_generation, binding_version, state
                ) VALUES (%s, %s, %s, %s, 2, 0, 1, 'current')
                """,
                (
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                    next_namespace,
                    CheckpointBinding(
                        tenant_id=registration.tenant_id,
                        logical_session_id=registration.authority.logical_session_id,
                        deployment_id=registration.deployment_id,
                        graph_contract=registration.graph_contract,
                        thread_id=next_namespace,
                    ).storage_thread_id,
                ),
            )

        replayed = await registry.publish(publication.model_copy(update={"fencing_generation": 2}))

    assert replayed.replayed
    assert replayed.operation_revision == 1
    assert replayed.operation_result == EmptySessionOperationResult()


@pytest.mark.postgres
async def test_restore_classifies_an_unauthentic_session_payload_without_exposing_it() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("corrupt_state_store", "AD_corrupt_state")
    authority = _lease_authority(registration, "corrupt-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("corrupt-owner"),
            payload=_empty_payload(),
        )
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                """
                UPDATE platform_sessions
                SET encrypted_payload = set_byte(
                    encrypted_payload,
                    0,
                    (get_byte(encrypted_payload, 0) + 1) %% 256
                )
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (registration.tenant_id, registration.authority.logical_session_id),
            )

        with pytest.raises(SessionRestoreError) as rejected:
            await registry.restore(authority)

    assert rejected.value.reason is SessionRestoreReason.DECRYPTION_FAILED
    assert "payload" not in str(rejected.value)


@pytest.mark.postgres
async def test_database_rejects_an_unsupported_payload_schema_before_restore() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("schema_state_store", "AD_schema_state")
    authority = _lease_authority(registration, "schema-owner")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("schema-owner"),
            payload=_empty_payload(),
        )
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            with pytest.raises(CheckViolation):
                await connection.execute(
                    """
                    UPDATE platform_sessions
                    SET payload_schema_version = %s
                    WHERE tenant_id = %s AND logical_session_id = %s
                    """,
                    (
                        SESSION_PAYLOAD_SCHEMA_VERSION + 1,
                        registration.tenant_id,
                        registration.authority.logical_session_id,
                    ),
                )

        assert (await registry.restore(authority)).payload == _empty_payload()


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    (
        ("deployment_id", "deployment-b", LeaseAdmissionReason.WRONG_DEPLOYMENT),
        ("graph_contract", "graph-b", LeaseAdmissionReason.WRONG_GRAPH_CONTRACT),
        ("config_version", "config-b", LeaseAdmissionReason.STALE_CONFIG),
        ("lease_owner_id", "another-owner", LeaseAdmissionReason.WRONG_LEASE_OWNER),
        ("fencing_generation", 2, LeaseAdmissionReason.STALE_FENCE),
    ),
)
async def test_activation_rejects_each_mismatched_authority_dimension(
    field: str,
    replacement: object,
    reason: LeaseAdmissionReason,
) -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration(f"activate_{reason.value}", "AD_activate_reject")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("activate-owner"),
            payload=_empty_payload(),
        )
        authority = _lease_authority(registration, "activate-owner").model_copy(
            update={field: replacement}
        )

        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.activate(authority)

    assert rejected.value.reason is reason


@pytest.mark.postgres
async def test_renewal_never_shortens_the_current_lease() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("lease_monotonic_store", "AD_lease_monotonic")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        created = await registry.register_and_acquire(
            registration,
            _lease("lease-monotonic-owner"),
            payload=_empty_payload(),
        )
        renewed = await registry.renew(
            _renewal(
                registration,
                "lease-monotonic-owner",
                duration_seconds=1.0,
            )
        )

    assert renewed.lease_expires_at is not None
    assert created.lease_expires_at is not None
    assert renewed.lease_expires_at >= created.lease_expires_at


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    (
        ("deployment_id", "deployment-b", LeaseAdmissionReason.WRONG_DEPLOYMENT),
        ("graph_contract", "graph-b", LeaseAdmissionReason.WRONG_GRAPH_CONTRACT),
        ("config_version", "config-b", LeaseAdmissionReason.STALE_CONFIG),
        ("lease_owner_id", "another-owner", LeaseAdmissionReason.WRONG_LEASE_OWNER),
        ("fencing_generation", 2, LeaseAdmissionReason.STALE_FENCE),
    ),
)
async def test_renewal_rejects_each_mismatched_authority_dimension(
    field: str,
    replacement: object,
    reason: LeaseAdmissionReason,
) -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration(
        f"renew_mismatch_{reason.value}",
        "AD_renew_mismatch",
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("renew-owner"),
            payload=_empty_payload(),
        )
        renewal = _renewal(registration, "renew-owner").model_copy(update={field: replacement})

        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.renew(renewal)

    assert rejected.value.reason is reason


@pytest.mark.postgres
async def test_renewal_rejects_changed_physical_transport() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("renew_transport_store", "AD_renew_transport")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("transport-owner"),
            payload=_empty_payload(),
        )
        changed_authority = registration.authority.model_copy(
            update={
                "transport": registration.authority.transport.model_copy(
                    update={"worker_id": "AW_replacement"}
                )
            }
        )
        renewal = _renewal(registration, "transport-owner").model_copy(
            update={"authority": changed_authority}
        )

        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.renew(renewal)

    assert rejected.value.reason is LeaseAdmissionReason.WRONG_TRANSPORT


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("database_change", "reason"),
    (
        (
            "lifecycle = 'closing', close_operation_id = 'test-close'",
            LeaseAdmissionReason.LIFECYCLE_REJECTED,
        ),
        (
            "created_at = clock_timestamp() - interval '2 hours', "
            "lease_expires_at = clock_timestamp() - interval '90 minutes', "
            "expires_at = clock_timestamp() - interval '1 hour'",
            LeaseAdmissionReason.SESSION_EXPIRED,
        ),
        (
            "created_at = clock_timestamp() - interval '2 minutes', "
            "lease_expires_at = clock_timestamp() - interval '1 second'",
            LeaseAdmissionReason.LEASE_EXPIRED,
        ),
    ),
)
async def test_renewal_rejects_ineligible_durable_state(
    database_change: str,
    reason: LeaseAdmissionReason,
) -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration(
        f"renew_state_{reason.value}",
        "AD_renew_state",
    )
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("state-owner"),
            payload=_empty_payload(),
        )
        async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
            await connection.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                (registration.tenant_id,),
            )
            await connection.execute(
                sql.SQL(
                    """
                UPDATE platform_sessions
                SET {}
                WHERE tenant_id = %s AND logical_session_id = %s
                """
                ).format(sql.SQL(database_change)),
                (
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )

        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.renew(_renewal(registration, "state-owner"))

    assert rejected.value.reason is reason


@pytest.mark.postgres
async def test_renewal_does_not_create_a_missing_session() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("missing_renew_store", "AD_missing_renew")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        with pytest.raises(LeaseAdmissionError) as rejected:
            await registry.renew(_renewal(registration, "missing-owner"))

    assert rejected.value.reason is LeaseAdmissionReason.SESSION_NOT_FOUND


@pytest.mark.postgres
async def test_platform_application_role_can_check_but_not_modify_schema_history() -> None:
    dsn = _dsn()
    role_name = f"phase4c_app_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
        cursor = await migration_connection.execute("SELECT current_schema()")
        schema_row = await cursor.fetchone()
        assert schema_row is not None
        schema_name = str(schema_row[0])
        await migration_connection.execute(
            sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role_name))
        )
        await migration_connection.execute(
            sql.SQL("GRANT ALL PRIVILEGES ON TABLE platform_sessions TO {}").format(
                sql.Identifier(role_name)
            )
        )
        await grant_platform_application_role(
            migration_connection,
            schema_name=schema_name,
            role_name=role_name,
        )
        cursor = await migration_connection.execute(
            "SELECT rolconfig FROM pg_roles WHERE rolname = %s",
            (role_name,),
        )
        assert await cursor.fetchone() == (None,)

    try:
        async with AsyncExitStack() as stack:
            registry = PostgresSessionRegistry(
                await _open_pool(stack, dsn),
                cipher=_cipher(),
                operation_timeout_seconds=2.0,
            )
            await registry.register_and_acquire(
                _registration(
                    "role_test_acme",
                    "AD_role_test",
                ),
                _lease("role-owner-acme"),
                payload=_empty_payload(),
            )
            await registry.register_and_acquire(
                _registration(
                    "role_test_demo",
                    "AD_role_test",
                ),
                _lease("role-owner-demo"),
                payload=_empty_payload(),
            )

        async with await AsyncConnection.connect(dsn, autocommit=True) as application:
            await application.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role_name)))
            await application.execute(
                "SELECT set_config('agnostic_market.tenant_id', %s, false)",
                ("role_test_acme",),
            )
            cursor = await application.execute(
                "SELECT tenant_id FROM platform_sessions ORDER BY tenant_id"
            )
            assert await cursor.fetchall() == [("role_test_acme",)]
            cursor = await application.execute(
                "SELECT tenant_id FROM platform_checkpoint_generations ORDER BY tenant_id"
            )
            assert await cursor.fetchall() == [("role_test_acme",)]
            hidden_delete = await application.execute(
                """
                DELETE FROM platform_checkpoint_generations
                WHERE tenant_id = 'role_test_demo'
                """
            )
            assert hidden_delete.rowcount == 0
            cursor = await application.execute(
                "SELECT tenant_id FROM platform_checkpoint_write_manifests ORDER BY tenant_id"
            )
            assert await cursor.fetchall() == []
            with pytest.raises(InsufficientPrivilege):
                await application.execute(
                    """
                    INSERT INTO platform_checkpoint_generations (
                        tenant_id, logical_session_id, checkpoint_namespace, storage_thread_id,
                        fencing_generation,
                        principal_generation, binding_version, state
                    ) VALUES (
                        'role_test_demo', 'AD_role_test', 'foreign-generation',
                        'cp_0000000000000000000000000000000000000000000000000000000000000000',
                        2, 0, 1, 'pending'
                    )
                    """
                )
            await require_platform_schema_version(
                application,
                PLATFORM_SESSION_SCHEMA_VERSION,
            )
            await require_platform_application_role(
                application,
                schema_name=schema_name,
            )
            cursor = await application.execute(
                """
                SELECT
                    has_table_privilege(current_user, 'platform_sessions', 'SELECT'),
                    has_table_privilege(current_user, 'platform_sessions', 'INSERT'),
                    has_table_privilege(current_user, 'platform_sessions', 'UPDATE'),
                    has_table_privilege(current_user, 'platform_sessions', 'DELETE'),
                    has_table_privilege(current_user, 'platform_sessions', 'TRUNCATE'),
                    has_table_privilege(current_user, 'platform_sessions', 'REFERENCES'),
                    has_table_privilege(current_user, 'platform_sessions', 'TRIGGER'),
                    has_table_privilege(current_user, 'platform_sessions', 'MAINTAIN')
                """
            )
            assert await cursor.fetchone() == (
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            )
            cursor = await application.execute(
                """
                SELECT
                    has_table_privilege(
                        current_user, 'platform_checkpoint_generations', 'SELECT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_generations', 'INSERT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_generations', 'UPDATE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_generations', 'DELETE'
                    )
                """
            )
            assert await cursor.fetchone() == (True, True, True, True)
            cursor = await application.execute(
                """
                SELECT
                    has_table_privilege(
                        current_user, 'platform_checkpoint_write_manifests', 'SELECT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_write_manifests', 'INSERT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_write_manifests', 'UPDATE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_write_manifests', 'DELETE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_checkpoint_write_manifests', 'TRUNCATE'
                    )
                """
            )
            assert await cursor.fetchone() == (True, True, True, True, False)
            cursor = await application.execute(
                """
                SELECT
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'SELECT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'INSERT'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'UPDATE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'DELETE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'TRUNCATE'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'REFERENCES'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'TRIGGER'
                    ),
                    has_table_privilege(
                        current_user, 'platform_session_operations', 'MAINTAIN'
                    )
                """
            )
            assert await cursor.fetchone() == (
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
            )
            with pytest.raises(InsufficientPrivilege):
                await application.execute(
                    """
                    INSERT INTO platform_schema_migrations (version, name, checksum)
                    VALUES (%s, %s, %s)
                    """,
                    (99, "unauthorized", "0" * 64),
                )
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
            role = sql.Identifier(role_name)
            await migration_connection.execute(sql.SQL("DROP OWNED BY {}").format(role))
            await migration_connection.execute(sql.SQL("DROP ROLE {}").format(role))


@pytest.mark.postgres
async def test_durable_platform_resources_open_with_the_pinned_application_role(
    config_root: Path,
) -> None:
    dsn = _dsn()
    config_registry = ConfigRegistry(config_root).load()
    resolved = config_registry.get("acme_store")
    tenant = build_tenant_context(config_registry, "acme_store")
    models = ApplicationModels(
        response=FakeChatModel(),
        reasoning=FakeChatModel(),
        response_structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
    )
    settings = ApplicationSettings(
        display_name=resolved.config.display_name,
        caller_audible_model_text_max_chars=TEST_CALLER_AUDIBLE_MODEL_TEXT_MAX_CHARS,
        checkpoint_io_timeout_seconds=2.0,
        response_model_node_timeout_seconds=2.0,
        reasoning_model_node_timeout_seconds=6.0,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
    )
    probe_services = build_fixture_tenant_services(
        config_root,
        tenant,
        telemetry=make_tenant_telemetry(tenant.tenant_id),
    )
    probe = await build_application_session(
        tenant,
        settings,
        models,
        probe_services,
        deployment_id="deployment-a",
        routing_factory=lambda _registry: ArchitectureRoutingRecognizer(),
    )
    graph_contract = graph_contract_fingerprint(probe.assembly.graph)
    await probe.state.caller_context.aclose_session()
    role_name = f"phase4c_runtime_{uuid.uuid4().hex[:20]}"
    password = "synthetic-runtime-password"
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)
        cursor = await migration_connection.execute("SELECT current_schema()")
        schema_row = await cursor.fetchone()
        assert schema_row is not None
        schema_name = str(schema_row[0])
        await migration_connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role_name),
                sql.Literal(password),
            )
        )
        await grant_platform_application_role(
            migration_connection,
            schema_name=schema_name,
            role_name=role_name,
        )

    application_dsn = make_conninfo(dsn, user=role_name, password=password)
    key = base64.b64encode(bytes(range(32))).decode("ascii")
    resolved_secrets = {
        "env://PLATFORM_POSTGRES_DSN": application_dsn,
        "env://PLATFORM_SESSION_KEY": key,
    }

    class RuntimeSecrets:
        def resolve(self, ref: str) -> str:
            return resolved_secrets[ref]

    config = PlatformRuntimeConfig.model_validate(
        {
            "schema_version": 1,
            "graph_contract": graph_contract,
            "database": {
                "application_dsn_ref": {
                    "provider": "env",
                    "locator": "PLATFORM_POSTGRES_DSN",
                },
                "schema_name": schema_name,
                "minimum_pool_size": 1,
                "maximum_pool_size": 2,
                "connection_timeout_seconds": 2.0,
                "pool_acquisition_timeout_seconds": 0.5,
                "statement_timeout_seconds": 1.0,
                "transaction_timeout_seconds": 2.0,
                "operation_timeout_seconds": 3.0,
                "expected_schema_version": PLATFORM_SESSION_SCHEMA_VERSION,
            },
            "sessions": {
                "lease_duration_seconds": 30.0,
                "lease_renewal_interval_seconds": 10.0,
                "session_retention_seconds": 3600,
                "closed_tombstone_retention_seconds": 7200,
            },
            "encryption": {
                "envelope_format": "aes_256_gcm_v1",
                "key_ref": {
                    "provider": "env",
                    "locator": "PLATFORM_SESSION_KEY",
                },
                "key_version": "runtime-key-v1",
                "key_encoding": "base64",
            },
        }
    )

    resources: DurablePlatformResources | None = None
    try:
        resources = await DurablePlatformResources.open(config, RuntimeSecrets())
        async with resources.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT current_schema(), current_schemas(false),
                    current_setting('statement_timeout'),
                    current_setting('transaction_timeout')
                """
            )
            assert await cursor.fetchone() == (schema_name, [schema_name], "1s", "2s")
        assert resources.registry is not None
        assert resources.cipher.active_key_version == "runtime-key-v1"
        admitted_authority = AdmittedSessionAuthority(
            logical_session_id="AD_runtime_application",
            transport=TransportAuthority(
                provider="livekit",
                room_id="RM_runtime_application",
                assignment_id="AJ_runtime_application",
                worker_id="AW_runtime_application",
            ),
        )
        durable_session = await resources.acquire_fresh_session(
            tenant_id=tenant.tenant_id,
            admitted_authority=admitted_authority,
            deployment_id="deployment-a",
            config_version=tenant.config_version,
        )
        with pytest.raises(ValueError, match="registry-fenced checkpoints"):
            replace(
                durable_session,
                checkpointer=build_checkpointer(cipher=resources.cipher),
            )
        services = build_fixture_tenant_services(
            config_root,
            tenant,
            telemetry=make_tenant_telemetry(tenant.tenant_id),
            checkpointer=durable_session.checkpointer,
        )
        application = await build_application_session(
            tenant,
            settings,
            models,
            services,
            deployment_id="deployment-a",
            routing_factory=lambda _registry: ArchitectureRoutingRecognizer(),
            durable_session=durable_session,
        )
        snapshot = await application.assembly.graph.aget_state(application.engine._config)
        restored = await resources.registry.restore(durable_session.authority)

        assert application.state.session_id == admitted_authority.logical_session_id
        assert ReasoningState.from_checkpoint(snapshot.values) == ReasoningState()
        assert restored.record.lifecycle is SessionLifecycle.ACTIVE
        assert restored.record.session_revision == 0

        restored_session = await resources.restore_owned_session(durable_session.authority)
        reconstructed_services = build_fixture_tenant_services(
            config_root,
            tenant,
            telemetry=make_tenant_telemetry(tenant.tenant_id),
            checkpointer=restored_session.checkpointer,
        )
        reconstructed = await build_application_session(
            tenant,
            settings,
            models,
            reconstructed_services,
            deployment_id="deployment-a",
            routing_factory=lambda _registry: ArchitectureRoutingRecognizer(),
            durable_session=restored_session,
        )
        reconstructed_snapshot = await reconstructed.assembly.graph.aget_state(
            reconstructed.engine._config
        )

        assert reconstructed.state.session_id == application.state.session_id
        assert reconstructed.state.thread_id == application.state.thread_id
        assert ReasoningState.from_checkpoint(reconstructed_snapshot.values) == ReasoningState()

        pending_mutation = PendingCartMutation(
            operation="add",
            sku="SKU-SESSION-AHEAD",
            name="Synthetic Session Ahead Item",
            price_usd="17.25",
            quantity=1,
            pre_confirm_quantity=0,
            idempotency_key="session-ahead-cart-mutation",
            created_at=1.0,
        )
        await reconstructed.assembly.graph.aupdate_state(
            reconstructed.engine._config,
            {
                "pending_cart_mutation": pending_mutation,
                "execution_owner": "cart",
            },
            as_node="cart_mutation_confirm",
        )
        before_publication = await reconstructed.assembly.graph.aget_state(
            reconstructed.engine._config
        )
        assert before_publication.next == ("cart_mutation_apply",)
        assert ReasoningState.from_checkpoint(before_publication.values).session_revision == 0

        committed = await reconstructed.state.session_state.apply_cart_mutation(
            pending_mutation.idempotency_key,
            operation=pending_mutation.operation,
            sku=pending_mutation.sku,
            name=pending_mutation.name,
            price_usd=pending_mutation.price_usd,
            quantity=pending_mutation.quantity,
            pre_confirm_quantity=pending_mutation.pre_confirm_quantity,
        )
        assert committed.session_revision == 1

        session_ahead = await resources.restore_owned_session(durable_session.authority)
        recovery_services = build_fixture_tenant_services(
            config_root,
            tenant,
            telemetry=make_tenant_telemetry(tenant.tenant_id),
            checkpointer=session_ahead.checkpointer,
        )
        recovered = await build_application_session(
            tenant,
            settings,
            models,
            recovery_services,
            deployment_id="deployment-a",
            routing_factory=lambda _registry: ArchitectureRoutingRecognizer(),
            durable_session=session_ahead,
        )
        prepared = await recovered.assembly.graph.aget_state(recovered.engine._config)
        prepared_state = ReasoningState.from_checkpoint(prepared.values)

        assert prepared.next == ("entry",)
        assert prepared_state.session_revision == 1
        assert prepared_state.pending_recovery is not None
        assert prepared_state.pending_recovery.origin_node == "cart_mutation_apply"

        await engine_events(recovered.engine, "continue", TurnFacts())
        completed = ReasoningState.from_checkpoint(
            (await recovered.assembly.graph.aget_state(recovered.engine._config)).values
        )
        durable_after_recovery = await resources.registry.restore(durable_session.authority)

        assert completed.pending_recovery is None
        assert completed.pending_cart_mutation is None
        assert completed.session_revision == 1
        assert recovered.state.cart_store.view()[0].quantity == 1
        assert durable_after_recovery.record.session_revision == 1

        readback = await engine_events(recovered.engine, "place my order", TurnFacts())
        paused = ReasoningState.from_checkpoint(
            (await recovered.assembly.graph.aget_state(recovered.engine._config)).values
        )
        assert paused.pending_placement is not None
        assert len([event for event in readback if isinstance(event, InterruptEvent)]) == 1

        confirmation_restore = await resources.restore_owned_session(durable_session.authority)
        confirmation_services = build_fixture_tenant_services(
            config_root,
            tenant,
            telemetry=make_tenant_telemetry(tenant.tenant_id),
            checkpointer=confirmation_restore.checkpointer,
        )
        confirmation_application = await build_application_session(
            tenant,
            settings,
            models,
            confirmation_services,
            deployment_id="deployment-a",
            routing_factory=lambda _registry: ArchitectureRoutingRecognizer(),
            durable_session=confirmation_restore,
        )
        prepared_confirmation = ReasoningState.from_checkpoint(
            (
                await confirmation_application.assembly.graph.aget_state(
                    confirmation_application.engine._config
                )
            ).values
        )
        assert prepared_confirmation.pending_recovery is not None
        assert prepared_confirmation.pending_recovery.trigger == "session_restored"

        stale_consent = await committed_turn_events(
            confirmation_application.engine,
            CommittedTurn(text="yes", message_id="restored-stale-confirmation"),
            TurnFacts(),
        )
        after_stale_consent = ReasoningState.from_checkpoint(
            (
                await confirmation_application.assembly.graph.aget_state(
                    confirmation_application.engine._config
                )
            ).values
        )
        assert len([event for event in stale_consent if isinstance(event, InterruptEvent)]) == 1
        assert not any(
            isinstance(message, HumanMessage) and message.content == "yes"
            for message in after_stale_consent.messages
        )
        assert (
            confirmation_services.order_store.placement_receipt(
                paused.pending_placement.idempotency_key,
                lines=paused.pending_placement.lines,
                total_usd=paused.pending_placement.total_usd,
            ).kind
            == "not_committed"
        )

        await committed_turn_events(
            confirmation_application.engine,
            CommittedTurn(text="yes", message_id="restored-fresh-confirmation"),
            TurnFacts(),
        )
        assert (
            confirmation_services.order_store.placement_receipt(
                paused.pending_placement.idempotency_key,
                lines=paused.pending_placement.lines,
                total_usd=paused.pending_placement.total_usd,
            ).kind
            == "committed"
        )

        await confirmation_application.state.caller_context.aclose_session()
        closed = await resources.registry.get(
            tenant.tenant_id,
            admitted_authority.logical_session_id,
        )

        assert closed is not None
        assert closed.lifecycle is SessionLifecycle.CLOSED
        assert closed.envelope is None
        assert confirmation_application.state.caller_context._closed is True
        assert (
            confirmation_services.order_store.placement_receipt(
                paused.pending_placement.idempotency_key,
                lines=paused.pending_placement.lines,
                total_usd=paused.pending_placement.total_usd,
            ).kind
            == "committed"
        )
    finally:
        if resources is not None:
            await resources.aclose()
            await resources.aclose()
        async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
            role = sql.Identifier(role_name)
            await migration_connection.execute(sql.SQL("DROP OWNED BY {}").format(role))
            await migration_connection.execute(sql.SQL("DROP ROLE {}").format(role))


@pytest.mark.postgres
async def test_database_rejects_a_closed_session_with_an_active_lease() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            _registration(
                "closed_lease_store",
                "AD_closed_lease",
            ),
            _lease("closed-lease-owner"),
            payload=_empty_payload(),
        )

    async with await AsyncConnection.connect(dsn, autocommit=True) as application:
        with pytest.raises(CheckViolation):
            await application.execute(
                """
                UPDATE platform_sessions
                SET lifecycle = 'closed',
                    lease_owner_id = 'retired-worker',
                    lease_expires_at = clock_timestamp() + interval '1 minute',
                    envelope_format = NULL,
                    envelope_key_version = NULL,
                    payload_schema_version = NULL,
                    envelope_nonce = NULL,
                    encrypted_payload = NULL
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                ("closed_lease_store", "AD_closed_lease"),
            )


@pytest.mark.postgres
async def test_database_rejects_a_checkpoint_namespace_outside_the_issued_fence() -> None:
    dsn = _dsn()
    async with await AsyncConnection.connect(dsn, autocommit=True) as migration_connection:
        await apply_platform_migrations(migration_connection)

    registration = _registration("checkpoint_fence_store", "AD_checkpoint_fence")
    async with AsyncExitStack() as stack:
        registry = PostgresSessionRegistry(
            await _open_pool(stack, dsn),
            cipher=_cipher(),
            operation_timeout_seconds=2.0,
        )
        await registry.register_and_acquire(
            registration,
            _lease("checkpoint-fence-owner"),
            payload=_empty_payload(),
        )

    async with await AsyncConnection.connect(dsn, autocommit=True) as application:
        await application.execute(
            "SELECT set_config('agnostic_market.tenant_id', %s, false)",
            (registration.tenant_id,),
        )
        with pytest.raises(CheckViolation):
            await application.execute(
                """
                UPDATE platform_sessions
                SET checkpoint_namespace = 'caller-selected'
                WHERE tenant_id = %s AND logical_session_id = %s
                """,
                (
                    registration.tenant_id,
                    registration.authority.logical_session_id,
                ),
            )
