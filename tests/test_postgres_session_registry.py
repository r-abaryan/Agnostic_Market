"""Real PostgreSQL contracts for the platform-session registry."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import CheckViolation, InsufficientPrivilege, NotNullViolation
from psycopg_pool import AsyncConnectionPool

from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.encryption import AesGcmSessionCipher, SessionEnvelopeContext
from agnostic_market.durability.migrations import (
    PLATFORM_SESSION_SCHEMA_VERSION,
    PlatformSchemaError,
    apply_platform_migrations,
    grant_platform_application_role,
    require_platform_schema_version,
)
from agnostic_market.durability.session_payload import (
    SESSION_PAYLOAD_SCHEMA_VERSION,
    DurableSessionPayload,
    EmptySessionOperationResult,
)
from agnostic_market.durability.session_registry import (
    LeaseAdmissionError,
    LeaseAdmissionReason,
    PostgresSessionRegistry,
    SessionLeaseAuthority,
    SessionLeaseRenewal,
    SessionLeaseRequest,
    SessionLifecycle,
    SessionRegistration,
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

_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
_KEY = bytes(range(32))


def _cipher() -> AesGcmSessionCipher:
    return AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY})


def _empty_payload() -> DurableSessionPayload:
    return DurableSessionPayload()


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


@pytest.mark.postgres
async def test_platform_migration_commits_without_requiring_autocommit() -> None:
    dsn = _dsn()
    schema = f"migration_tx_{uuid.uuid4().hex[:20]}"
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        isolated_dsn = _schema_dsn(dsn, schema)
        async with await AsyncConnection.connect(isolated_dsn) as migration_connection:
            await apply_platform_migrations(migration_connection)
            async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as observer:
                await require_platform_schema_version(
                    observer,
                    PLATFORM_SESSION_SCHEMA_VERSION,
                )
    finally:
        async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


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
                    DROP CONSTRAINT platform_sessions_checkpoint_matches_fence,
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
                    DROP CONSTRAINT platform_sessions_checkpoint_matches_fence,
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
async def test_restore_classifies_an_unsupported_payload_schema_separately() -> None:
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

        with pytest.raises(SessionRestoreError) as rejected:
            await registry.restore(authority)

    assert rejected.value.reason is SessionRestoreReason.PAYLOAD_SCHEMA_INVALID


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
            "lifecycle = 'closing'",
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
            await require_platform_schema_version(
                application,
                PLATFORM_SESSION_SCHEMA_VERSION,
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
