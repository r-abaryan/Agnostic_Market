"""Real PostgreSQL contracts for the platform-session registry."""

from __future__ import annotations

import os
import uuid
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta

import pytest
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import CheckViolation, InsufficientPrivilege
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
    PostgresSessionRegistry,
    SessionLifecycle,
    SessionRegistration,
    SessionRegistryConflictError,
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


def _registration(
    tenant_id: str,
    logical_session_id: str,
    checkpoint_namespace: str,
) -> SessionRegistration:
    return SessionRegistration(
        tenant_id=tenant_id,
        authority=AdmittedSessionAuthority(
            logical_session_id=logical_session_id,
            transport=TransportAuthority(
                provider="livekit",
                assignment_id=f"AJ_{tenant_id}",
                worker_id=f"AW_{tenant_id}",
            ),
        ),
        checkpoint_namespace=checkpoint_namespace,
        deployment_id="deployment-a",
        graph_contract="graph-a",
        config_version="config-a",
        principal_generation=0,
        session_revision=0,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        payload_schema_version=1,
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
            "cp_acme_generation_0",
        )
        demo = _registration(
            "demo_shop",
            "AD_shared",
            "cp_demo_generation_0",
        )

        created = await first.create(acme, payload=sentinel)
        await second.create(demo, payload=b'{"cart":[]}')
        reread = await second.get("acme_store", "AD_shared")

        assert reread == created
        assert created.lifecycle is SessionLifecycle.OPENING
        assert created.fencing_generation == 0
        assert reread.envelope is not None
        context = SessionEnvelopeContext(
            tenant_id="acme_store",
            logical_session_id="AD_shared",
            checkpoint_namespace="cp_acme_generation_0",
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
        with pytest.raises(SessionRegistryConflictError):
            await second.create(acme, payload=sentinel)

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
            await registry.create(
                _registration(
                    "role_test_acme",
                    "AD_role_test",
                    "cp_role_test_acme",
                ),
                payload=b'{"cart":[]}',
            )
            await registry.create(
                _registration(
                    "role_test_demo",
                    "AD_role_test",
                    "cp_role_test_demo",
                ),
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
        await registry.create(
            _registration(
                "closed_lease_store",
                "AD_closed_lease",
                "cp_closed_lease",
            ),
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
