"""Explicit versioned migrations for platform-owned session state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import LiteralString

from psycopg import AsyncConnection, sql
from pydantic import TypeAdapter

from agnostic_market.dtos.session import AuthorityIdentifier

PLATFORM_SESSION_SCHEMA_VERSION = 4
_DATABASE_IDENTIFIER = TypeAdapter(AuthorityIdentifier)

_BOOTSTRAP_SQL: LiteralString = """
CREATE TABLE IF NOT EXISTS platform_schema_migrations (
    version integer PRIMARY KEY CHECK (version > 0),
    name text NOT NULL CHECK (btrim(name) <> ''),
    checksum text NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
)
"""

_CREATE_SESSION_REGISTRY_SQL: LiteralString = """
CREATE TABLE platform_sessions (
    tenant_id text NOT NULL CHECK (tenant_id = btrim(tenant_id) AND tenant_id <> ''),
    logical_session_id text NOT NULL
        CHECK (logical_session_id = btrim(logical_session_id) AND logical_session_id <> ''),
    lifecycle text NOT NULL
        CHECK (lifecycle IN ('opening', 'active', 'closing', 'closed')),
    checkpoint_namespace text NOT NULL
        CHECK (checkpoint_namespace = btrim(checkpoint_namespace) AND checkpoint_namespace <> ''),
    deployment_id text NOT NULL
        CHECK (deployment_id = btrim(deployment_id) AND deployment_id <> ''),
    graph_contract text NOT NULL
        CHECK (graph_contract = btrim(graph_contract) AND graph_contract <> ''),
    config_version text NOT NULL
        CHECK (config_version = btrim(config_version) AND config_version <> ''),
    principal_generation bigint NOT NULL CHECK (principal_generation >= 0),
    session_revision bigint NOT NULL CHECK (session_revision >= 0),
    fencing_generation bigint NOT NULL CHECK (fencing_generation >= 0),
    lease_owner_id text,
    lease_expires_at timestamptz,
    transport_provider text NOT NULL CHECK (transport_provider = 'livekit'),
    transport_assignment_id text NOT NULL
        CHECK (
            transport_assignment_id = btrim(transport_assignment_id)
            AND transport_assignment_id <> ''
        ),
    transport_worker_id text NOT NULL
        CHECK (transport_worker_id = btrim(transport_worker_id) AND transport_worker_id <> ''),
    expires_at timestamptz NOT NULL,
    envelope_format text,
    envelope_key_version text,
    payload_schema_version integer,
    envelope_nonce bytea,
    encrypted_payload bytea,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, logical_session_id),
    UNIQUE (tenant_id, checkpoint_namespace),
    CHECK ((lease_owner_id IS NULL) = (lease_expires_at IS NULL)),
    CHECK (lifecycle <> 'closed' OR lease_owner_id IS NULL),
    CHECK (
        lease_owner_id IS NULL
        OR (lease_owner_id = btrim(lease_owner_id) AND lease_owner_id <> '')
    ),
    CHECK (expires_at > created_at),
    CHECK (updated_at >= created_at),
    CHECK (
        (
            lifecycle = 'closed'
            AND envelope_format IS NULL
            AND envelope_key_version IS NULL
            AND payload_schema_version IS NULL
            AND envelope_nonce IS NULL
            AND encrypted_payload IS NULL
        )
        OR
        (
            lifecycle <> 'closed'
            AND envelope_format = 'aes_256_gcm_v1'
            AND envelope_key_version IS NOT NULL
            AND envelope_key_version = btrim(envelope_key_version)
            AND envelope_key_version <> ''
            AND payload_schema_version IS NOT NULL
            AND payload_schema_version > 0
            AND envelope_nonce IS NOT NULL
            AND octet_length(envelope_nonce) = 12
            AND encrypted_payload IS NOT NULL
            AND octet_length(encrypted_payload) >= 16
        )
    )
)
"""

_ENABLE_SESSION_RLS_SQL: LiteralString = """
ALTER TABLE platform_sessions ENABLE ROW LEVEL SECURITY
"""

_FORCE_SESSION_RLS_SQL: LiteralString = """
ALTER TABLE platform_sessions FORCE ROW LEVEL SECURITY
"""

_CREATE_SESSION_POLICY_SQL: LiteralString = """
CREATE POLICY platform_sessions_tenant_isolation ON platform_sessions
    USING (tenant_id = current_setting('agnostic_market.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('agnostic_market.tenant_id', true))
"""

_ADD_TRANSPORT_ROOM_ID_SQL: LiteralString = """
ALTER TABLE platform_sessions
    ADD COLUMN transport_room_id text NOT NULL
        CHECK (transport_room_id = btrim(transport_room_id) AND transport_room_id <> '')
"""

_REQUIRE_OPEN_LEASE_SQL: LiteralString = """
ALTER TABLE platform_sessions
    ADD CONSTRAINT platform_sessions_open_requires_lease
        CHECK ((lifecycle = 'closed') = (lease_owner_id IS NULL)),
    ADD CONSTRAINT platform_sessions_open_requires_positive_fence
        CHECK (lifecycle = 'closed' OR fencing_generation > 0),
    ADD CONSTRAINT platform_sessions_checkpoint_matches_fence
        CHECK (
            checkpoint_namespace = logical_session_id || '::fence::' || fencing_generation
        ),
    ADD CONSTRAINT platform_sessions_lease_starts_before_expiry
        CHECK (
            lease_expires_at IS NULL
            OR (lease_expires_at > created_at AND lease_expires_at <= expires_at)
        )
"""

_CREATE_SESSION_OPERATIONS_SQL: LiteralString = """
CREATE TABLE platform_session_operations (
    tenant_id text NOT NULL,
    logical_session_id text NOT NULL,
    operation_id text NOT NULL
        CHECK (operation_id = btrim(operation_id) AND operation_id <> ''),
    request_fingerprint text NOT NULL CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    committed_revision bigint NOT NULL CHECK (committed_revision > 0),
    committed_checkpoint_namespace text NOT NULL
        CHECK (
            committed_checkpoint_namespace = btrim(committed_checkpoint_namespace)
            AND committed_checkpoint_namespace <> ''
        ),
    result_envelope_format text NOT NULL CHECK (result_envelope_format = 'aes_256_gcm_v1'),
    result_envelope_key_version text NOT NULL
        CHECK (
            result_envelope_key_version = btrim(result_envelope_key_version)
            AND result_envelope_key_version <> ''
        ),
    result_schema_version integer NOT NULL CHECK (result_schema_version > 0),
    result_envelope_nonce bytea NOT NULL CHECK (octet_length(result_envelope_nonce) = 12),
    encrypted_result bytea NOT NULL CHECK (octet_length(encrypted_result) >= 16),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, logical_session_id, operation_id),
    FOREIGN KEY (tenant_id, logical_session_id)
        REFERENCES platform_sessions (tenant_id, logical_session_id)
        ON DELETE CASCADE
)
"""

_ENABLE_SESSION_OPERATIONS_RLS_SQL: LiteralString = """
ALTER TABLE platform_session_operations ENABLE ROW LEVEL SECURITY
"""

_FORCE_SESSION_OPERATIONS_RLS_SQL: LiteralString = """
ALTER TABLE platform_session_operations FORCE ROW LEVEL SECURITY
"""

_CREATE_SESSION_OPERATIONS_POLICY_SQL: LiteralString = """
CREATE POLICY platform_session_operations_tenant_isolation ON platform_session_operations
    USING (tenant_id = current_setting('agnostic_market.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('agnostic_market.tenant_id', true))
"""


class PlatformSchemaError(RuntimeError):
    """The installed platform schema is absent, divergent, or incompatible."""


@dataclass(frozen=True, slots=True)
class PlatformMigration:
    version: int
    name: str
    statements: tuple[LiteralString, ...]

    @property
    def checksum(self) -> str:
        payload = "\n-- statement --\n".join(statement.strip() for statement in self.statements)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PLATFORM_MIGRATIONS = (
    PlatformMigration(
        version=1,
        name="platform_session_registry",
        statements=(
            _CREATE_SESSION_REGISTRY_SQL,
            _ENABLE_SESSION_RLS_SQL,
            _FORCE_SESSION_RLS_SQL,
            _CREATE_SESSION_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=2,
        name="transport_room_authority",
        statements=(_ADD_TRANSPORT_ROOM_ID_SQL,),
    ),
    PlatformMigration(
        version=3,
        name="atomic_initial_session_lease",
        statements=(_REQUIRE_OPEN_LEASE_SQL,),
    ),
    PlatformMigration(
        version=4,
        name="session_state_operation_receipts",
        statements=(
            _CREATE_SESSION_OPERATIONS_SQL,
            _ENABLE_SESSION_OPERATIONS_RLS_SQL,
            _FORCE_SESSION_OPERATIONS_RLS_SQL,
            _CREATE_SESSION_OPERATIONS_POLICY_SQL,
        ),
    ),
)


def _validate_migration_inventory() -> None:
    versions = tuple(migration.version for migration in PLATFORM_MIGRATIONS)
    if versions != tuple(range(1, PLATFORM_SESSION_SCHEMA_VERSION + 1)):
        raise RuntimeError("platform migration versions must be contiguous and complete")
    if len({migration.name for migration in PLATFORM_MIGRATIONS}) != len(PLATFORM_MIGRATIONS):
        raise RuntimeError("platform migration names must be unique")


_validate_migration_inventory()


async def _read_installed_migrations(
    connection: AsyncConnection,
) -> dict[int, tuple[str, str]]:
    cursor = await connection.execute(
        "SELECT version, name, checksum FROM platform_schema_migrations ORDER BY version"
    )
    return {int(row[0]): (str(row[1]), str(row[2])) for row in await cursor.fetchall()}


def _validate_installed_migrations(installed: dict[int, tuple[str, str]]) -> None:
    installed_versions = tuple(installed)
    if installed_versions != tuple(range(1, len(installed_versions) + 1)):
        raise PlatformSchemaError("platform schema migration history is not contiguous")
    known = {migration.version: migration for migration in PLATFORM_MIGRATIONS}
    if unknown := set(installed) - set(known):
        raise PlatformSchemaError(
            f"database contains unknown platform migration versions: {sorted(unknown)!r}"
        )
    for version, recorded_identity in installed.items():
        migration = known[version]
        if recorded_identity != (migration.name, migration.checksum):
            raise PlatformSchemaError(
                f"platform migration {version} does not match repository history"
            )


async def apply_platform_migrations(connection: AsyncConnection) -> None:
    """Apply migrations through an explicit migration-owner connection."""
    async with connection.transaction():
        await connection.execute(_BOOTSTRAP_SQL)
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("agnostic_market_platform_migrations",),
        )
        installed = await _read_installed_migrations(connection)
        _validate_installed_migrations(installed)
        for migration in PLATFORM_MIGRATIONS:
            applied = installed.get(migration.version)
            if applied is not None:
                continue
            for statement in migration.statements:
                await connection.execute(sql.SQL(statement))
            await connection.execute(
                """
                INSERT INTO platform_schema_migrations (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (migration.version, migration.name, migration.checksum),
            )


async def read_platform_schema_version(connection: AsyncConnection) -> int:
    try:
        installed = await _read_installed_migrations(connection)
    except Exception as exc:
        raise PlatformSchemaError("platform schema migration history is unavailable") from exc
    _validate_installed_migrations(installed)
    versions = tuple(installed)
    return versions[-1] if versions else 0


async def require_platform_schema_version(
    connection: AsyncConnection,
    expected_version: int,
) -> None:
    if expected_version < 1:
        raise ValueError("expected platform schema version must be positive")
    if expected_version != PLATFORM_SESSION_SCHEMA_VERSION:
        raise PlatformSchemaError(
            "configured platform schema version does not match this runtime: "
            f"expected {expected_version}, runtime {PLATFORM_SESSION_SCHEMA_VERSION}"
        )
    actual_version = await read_platform_schema_version(connection)
    if actual_version != expected_version:
        raise PlatformSchemaError(
            f"platform schema version mismatch: expected {expected_version}, found {actual_version}"
        )


async def grant_platform_application_role(
    connection: AsyncConnection,
    *,
    schema_name: str,
    role_name: str,
) -> None:
    """Grant an existing application role its runtime and schema-gate access."""
    schema_name = _DATABASE_IDENTIFIER.validate_python(schema_name)
    role_name = _DATABASE_IDENTIFIER.validate_python(role_name)
    schema = sql.Identifier(schema_name)
    role = sql.Identifier(role_name)
    sessions = sql.Identifier(schema_name, "platform_sessions")
    operations = sql.Identifier(schema_name, "platform_session_operations")
    migrations = sql.Identifier(schema_name, "platform_schema_migrations")
    async with connection.transaction():
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM PUBLIC").format(sessions))
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM PUBLIC").format(operations))
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM PUBLIC").format(migrations))
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM {}").format(sessions, role))
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM {}").format(operations, role))
        await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM {}").format(migrations, role))
        await connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, role))
        await connection.execute(sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(migrations, role))
        await connection.execute(
            sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {} TO {}").format(
                sessions,
                role,
            )
        )
        await connection.execute(
            sql.SQL("GRANT SELECT, INSERT, DELETE ON TABLE {} TO {}").format(
                operations,
                role,
            )
        )
