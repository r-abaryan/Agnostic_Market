"""Real PostgreSQL contracts for the platform-session registry."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import AsyncExitStack

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
from agnostic_market.durability.session_registry import (
    LeaseAdmissionError,
    LeaseAdmissionReason,
    PostgresSessionRegistry,
    SessionLeaseRenewal,
    SessionLeaseRequest,
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryRecord,
)

_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
_KEY = bytes(range(32))


def _cipher() -> AesGcmSessionCipher:
    return AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY})


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
                payload=b'{"cart":[]}',
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
                payload=b'{"cart":[]}',
            )

        async with await AsyncConnection.connect(isolated_dsn, autocommit=True) as connection:
            await connection.execute("DELETE FROM platform_schema_migrations WHERE version = 3")
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
        payload_schema_version=1,
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
        lease=_lease(owner_id, duration_seconds=duration_seconds),
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

    sentinel = b'unique plaintext sentinel: {"cart":["SKU-1"]}'
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
            payload=sentinel,
        )
        await second.register_and_acquire(
            demo,
            _lease("lease-demo"),
            payload=b'{"cart":[]}',
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
            payload_schema_version=1,
        )
        assert (
            AesGcmSessionCipher(active_key_version="key-v1", keys={"key-v1": _KEY}).decrypt(
                reread.envelope,
                context,
            )
            == sentinel
        )
        assert await first.get("demo_shop", "AD_shared") is not None
        assert await first.get("unknown_store", "AD_shared") is None
        with pytest.raises(LeaseAdmissionError) as conflict:
            await second.register_and_acquire(
                acme,
                _lease("another-owner"),
                payload=sentinel,
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
                payload=b'{"cart":[]}',
            ),
            second.register_and_acquire(
                registration,
                _lease("lease-race-b"),
                payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
        ("lease", _lease("another-owner"), LeaseAdmissionReason.WRONG_LEASE_OWNER),
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
            payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
                payload=b'{"cart":[]}',
            )
            await registry.register_and_acquire(
                _registration(
                    "role_test_demo",
                    "AD_role_test",
                ),
                _lease("role-owner-demo"),
                payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
            payload=b'{"cart":[]}',
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
