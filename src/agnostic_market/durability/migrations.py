"""Explicit versioned migrations for platform-owned session state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import LiteralString, assert_never

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection, sql
from psycopg.pq import TransactionStatus
from pydantic import TypeAdapter

from agnostic_market.checkpoints import CheckpointBinding
from agnostic_market.dtos.session import AuthorityIdentifier
from agnostic_market.durability.session_payload import (
    SESSION_OPERATION_RESULT_SCHEMA_VERSION,
    SESSION_PAYLOAD_SCHEMA_VERSION,
)

PLATFORM_SESSION_SCHEMA_VERSION = 8
_DATABASE_IDENTIFIER = TypeAdapter(AuthorityIdentifier)

_TABLE_PRIVILEGES = (
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "TRUNCATE",
    "REFERENCES",
    "TRIGGER",
    "MAINTAIN",
)
_APPLICATION_TABLE_PRIVILEGES: dict[str, tuple[str, ...]] = {
    "platform_schema_migrations": ("SELECT",),
    "platform_sessions": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "platform_session_operations": ("SELECT", "INSERT", "DELETE"),
    "platform_checkpoint_generations": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "platform_checkpoint_write_manifests": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "checkpoints": ("SELECT", "INSERT", "UPDATE", "DELETE"),
    "checkpoint_blobs": ("SELECT", "INSERT", "DELETE"),
    "checkpoint_writes": ("SELECT", "INSERT", "UPDATE", "DELETE"),
}

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


_CREATE_CHECKPOINT_GENERATIONS_SQL: LiteralString = """
CREATE TABLE platform_checkpoint_generations (
    tenant_id text NOT NULL,
    logical_session_id text NOT NULL,
    checkpoint_namespace text NOT NULL
        CHECK (checkpoint_namespace = btrim(checkpoint_namespace) AND checkpoint_namespace <> ''),
    fencing_generation bigint NOT NULL CHECK (fencing_generation >= 0),
    principal_generation bigint NOT NULL CHECK (principal_generation >= 0),
    binding_version integer NOT NULL CHECK (binding_version = 1),
    state text NOT NULL CHECK (state IN ('pending', 'current', 'retired', 'deleted')),
    transition_id text
        CHECK (
            transition_id IS NULL
            OR (transition_id = btrim(transition_id) AND transition_id <> '')
        ),
    source_revision bigint CHECK (source_revision >= 0),
    CHECK ((transition_id IS NULL) = (source_revision IS NULL)),
    CHECK (state <> 'pending' OR transition_id IS NOT NULL),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, logical_session_id, checkpoint_namespace),
    UNIQUE (tenant_id, logical_session_id, fencing_generation, principal_generation),
    UNIQUE (tenant_id, logical_session_id, transition_id),
    FOREIGN KEY (tenant_id, logical_session_id)
        REFERENCES platform_sessions (tenant_id, logical_session_id) ON DELETE CASCADE
)
"""

_BACKFILL_CHECKPOINT_GENERATIONS_SQL: LiteralString = """
INSERT INTO platform_checkpoint_generations (
    tenant_id, logical_session_id, checkpoint_namespace, fencing_generation,
    principal_generation, binding_version, state
)
SELECT tenant_id, logical_session_id, checkpoint_namespace, fencing_generation,
    principal_generation, 1, CASE WHEN lifecycle = 'closed' THEN 'retired' ELSE 'current' END
FROM platform_sessions
"""

_CHECKPOINT_GENERATIONS_CURRENT_INDEX_SQL: LiteralString = """
CREATE UNIQUE INDEX platform_checkpoint_generations_one_current
    ON platform_checkpoint_generations (tenant_id, logical_session_id)
    WHERE state = 'current'
"""

_CHECKPOINT_GENERATIONS_PENDING_INDEX_SQL: LiteralString = """
CREATE UNIQUE INDEX platform_checkpoint_generations_one_pending
    ON platform_checkpoint_generations (tenant_id, logical_session_id)
    WHERE state = 'pending'
"""

_CHECKPOINT_GENERATIONS_RLS_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_generations ENABLE ROW LEVEL SECURITY
"""

_CHECKPOINT_GENERATIONS_FORCE_RLS_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_generations FORCE ROW LEVEL SECURITY
"""

_CHECKPOINT_GENERATIONS_POLICY_SQL: LiteralString = """
CREATE POLICY platform_checkpoint_generations_tenant_isolation ON platform_checkpoint_generations
    USING (tenant_id = current_setting('agnostic_market.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('agnostic_market.tenant_id', true))
"""

_ALLOW_PRINCIPAL_CHECKPOINT_GENERATIONS_SQL: LiteralString = """
ALTER TABLE platform_sessions
    DROP CONSTRAINT platform_sessions_checkpoint_matches_fence,
    ADD CONSTRAINT platform_sessions_checkpoint_matches_generation
        CHECK (
            checkpoint_namespace = logical_session_id || '::fence::' || fencing_generation
            OR checkpoint_namespace =
                logical_session_id || '::fence::' || fencing_generation
                || '::principal::' || principal_generation
        )
"""

_CREATE_CHECKPOINT_WRITE_MANIFESTS_SQL: LiteralString = """
CREATE TABLE platform_checkpoint_write_manifests (
    tenant_id text NOT NULL,
    logical_session_id text NOT NULL,
    checkpoint_generation_namespace text NOT NULL,
    langgraph_checkpoint_namespace text NOT NULL,
    checkpoint_id text NOT NULL CHECK (checkpoint_id <> ''),
    envelope_format text NOT NULL CHECK (envelope_format = 'aes_256_gcm_v1'),
    envelope_key_version text NOT NULL
        CHECK (envelope_key_version = btrim(envelope_key_version) AND envelope_key_version <> ''),
    payload_schema_version integer NOT NULL CHECK (payload_schema_version = 1),
    envelope_nonce bytea NOT NULL CHECK (octet_length(envelope_nonce) = 12),
    encrypted_manifest bytea NOT NULL CHECK (octet_length(encrypted_manifest) >= 16),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (
        tenant_id,
        logical_session_id,
        checkpoint_generation_namespace,
        langgraph_checkpoint_namespace,
        checkpoint_id
    ),
    FOREIGN KEY (tenant_id, logical_session_id, checkpoint_generation_namespace)
        REFERENCES platform_checkpoint_generations (
            tenant_id, logical_session_id, checkpoint_namespace
        ) ON DELETE CASCADE
)
"""

_CHECKPOINT_WRITE_MANIFESTS_RLS_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_write_manifests ENABLE ROW LEVEL SECURITY
"""

_CHECKPOINT_WRITE_MANIFESTS_FORCE_RLS_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_write_manifests FORCE ROW LEVEL SECURITY
"""

_CHECKPOINT_WRITE_MANIFESTS_POLICY_SQL: LiteralString = """
CREATE POLICY platform_checkpoint_write_manifests_tenant_isolation
    ON platform_checkpoint_write_manifests
    USING (tenant_id = current_setting('agnostic_market.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('agnostic_market.tenant_id', true))
"""

_ADD_CHECKPOINT_STORAGE_AUTHORITY_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_generations ADD COLUMN storage_thread_id text
"""

_CONSTRAIN_CHECKPOINT_STORAGE_AUTHORITY_SQL: LiteralString = """
ALTER TABLE platform_checkpoint_generations
    ALTER COLUMN storage_thread_id SET NOT NULL,
    ADD CONSTRAINT platform_checkpoint_generations_storage_thread_format
        CHECK (storage_thread_id ~ '^cp_[0-9a-f]{64}$'),
    ADD CONSTRAINT platform_checkpoint_generations_storage_thread_unique
        UNIQUE (storage_thread_id)
"""

_UNIQUE_OPERATION_REVISION_SQL: LiteralString = """
ALTER TABLE platform_session_operations
    ADD CONSTRAINT platform_session_operations_one_receipt_per_revision
    UNIQUE (tenant_id, logical_session_id, committed_revision)
"""

_REQUIRE_CURRENT_SESSION_PAYLOAD_SCHEMA_SQL: LiteralString = """
ALTER TABLE platform_sessions
    ADD CONSTRAINT platform_sessions_current_payload_schema
    CHECK (lifecycle = 'closed' OR payload_schema_version = 2)
"""

_REQUIRE_CURRENT_OPERATION_RESULT_SCHEMA_SQL: LiteralString = """
ALTER TABLE platform_session_operations
    ADD CONSTRAINT platform_session_operations_current_result_schema
    CHECK (result_schema_version = 2)
"""

_ALLOW_FENCED_SESSION_CLOSE_SQL: LiteralString = """
ALTER TABLE platform_sessions
    ADD COLUMN close_operation_id text,
    DROP CONSTRAINT platform_sessions_checkpoint_matches_generation,
    DROP CONSTRAINT platform_sessions_lease_starts_before_expiry,
    ADD CONSTRAINT platform_sessions_checkpoint_matches_generation
        CHECK (
            lifecycle IN ('closing', 'closed')
            OR checkpoint_namespace =
                logical_session_id || '::fence::' || fencing_generation
            OR checkpoint_namespace =
                logical_session_id || '::fence::' || fencing_generation
                || '::principal::' || principal_generation
        ),
    ADD CONSTRAINT platform_sessions_lease_starts_before_expiry
        CHECK (
            lease_expires_at IS NULL
            OR (
                lease_expires_at > created_at
                AND (
                    lifecycle = 'closing'
                    OR lease_expires_at <= expires_at
                )
            )
        ),
    ADD CONSTRAINT platform_sessions_close_operation_format
        CHECK (
            close_operation_id IS NULL
            OR (
                close_operation_id = btrim(close_operation_id)
                AND close_operation_id <> ''
            )
        )
"""

_CHECKPOINTS_RLS_SQL: LiteralString = """
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY
"""

_CHECKPOINTS_FORCE_RLS_SQL: LiteralString = """
ALTER TABLE checkpoints FORCE ROW LEVEL SECURITY
"""

_CHECKPOINTS_POLICY_SQL: LiteralString = """
CREATE POLICY checkpoints_tenant_isolation ON checkpoints
    USING (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoints.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoints.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
"""

_CHECKPOINT_BLOBS_RLS_SQL: LiteralString = """
ALTER TABLE checkpoint_blobs ENABLE ROW LEVEL SECURITY
"""

_CHECKPOINT_BLOBS_FORCE_RLS_SQL: LiteralString = """
ALTER TABLE checkpoint_blobs FORCE ROW LEVEL SECURITY
"""

_CHECKPOINT_BLOBS_POLICY_SQL: LiteralString = """
CREATE POLICY checkpoint_blobs_tenant_isolation ON checkpoint_blobs
    USING (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoint_blobs.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoint_blobs.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
"""

_CHECKPOINT_WRITES_RLS_SQL: LiteralString = """
ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY
"""

_CHECKPOINT_WRITES_FORCE_RLS_SQL: LiteralString = """
ALTER TABLE checkpoint_writes FORCE ROW LEVEL SECURITY
"""

_CHECKPOINT_WRITES_POLICY_SQL: LiteralString = """
CREATE POLICY checkpoint_writes_tenant_isolation ON checkpoint_writes
    USING (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoint_writes.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM platform_checkpoint_generations AS generation
            WHERE generation.storage_thread_id = checkpoint_writes.thread_id
              AND generation.tenant_id = current_setting('agnostic_market.tenant_id', true)
        )
    )
"""

_CHECKPOINT_VENDOR_COLUMNS = {
    "checkpoint_migrations": ("v",),
    "checkpoints": (
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
        "parent_checkpoint_id",
        "type",
        "checkpoint",
        "metadata",
    ),
    "checkpoint_blobs": ("thread_id", "checkpoint_ns", "channel", "version", "type", "blob"),
    "checkpoint_writes": (
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
        "task_id",
        "idx",
        "channel",
        "type",
        "blob",
        "task_path",
    ),
}
_CHECKPOINT_VENDOR_MIGRATION_VERSION = 9


class PlatformSchemaError(RuntimeError):
    """The installed platform schema is absent, divergent, or incompatible."""


class PlatformDataMigration(StrEnum):
    REQUIRE_CURRENT_ENCRYPTED_PAYLOADS = "require_current_encrypted_payloads"
    CHECKPOINT_STORAGE_AUTHORITY_V1 = "checkpoint_storage_authority_v1"


type PlatformMigrationStep = LiteralString | PlatformDataMigration


@dataclass(frozen=True, slots=True)
class PlatformMigration:
    version: int
    name: str
    steps: tuple[PlatformMigrationStep, ...]

    @property
    def checksum(self) -> str:
        payload = "\n-- statement --\n".join(
            f"data:{step.value}" if isinstance(step, PlatformDataMigration) else step.strip()
            for step in self.steps
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PLATFORM_MIGRATIONS = (
    PlatformMigration(
        version=1,
        name="platform_session_registry",
        steps=(
            _CREATE_SESSION_REGISTRY_SQL,
            _ENABLE_SESSION_RLS_SQL,
            _FORCE_SESSION_RLS_SQL,
            _CREATE_SESSION_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=2,
        name="transport_room_authority",
        steps=(_ADD_TRANSPORT_ROOM_ID_SQL,),
    ),
    PlatformMigration(
        version=3,
        name="atomic_initial_session_lease",
        steps=(_REQUIRE_OPEN_LEASE_SQL,),
    ),
    PlatformMigration(
        version=4,
        name="session_state_operation_receipts",
        steps=(
            _CREATE_SESSION_OPERATIONS_SQL,
            _ENABLE_SESSION_OPERATIONS_RLS_SQL,
            _FORCE_SESSION_OPERATIONS_RLS_SQL,
            _CREATE_SESSION_OPERATIONS_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=5,
        name="checkpoint_generation_inventory",
        steps=(
            _CREATE_CHECKPOINT_GENERATIONS_SQL,
            _ALLOW_PRINCIPAL_CHECKPOINT_GENERATIONS_SQL,
            # The migration owner inventories every tenant inside the migration transaction.
            "ALTER TABLE platform_sessions NO FORCE ROW LEVEL SECURITY",
            _BACKFILL_CHECKPOINT_GENERATIONS_SQL,
            "ALTER TABLE platform_sessions FORCE ROW LEVEL SECURITY",
            _CHECKPOINT_GENERATIONS_CURRENT_INDEX_SQL,
            _CHECKPOINT_GENERATIONS_PENDING_INDEX_SQL,
            _CHECKPOINT_GENERATIONS_RLS_SQL,
            _CHECKPOINT_GENERATIONS_FORCE_RLS_SQL,
            _CHECKPOINT_GENERATIONS_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=6,
        name="checkpoint_pending_write_manifests",
        steps=(
            _CREATE_CHECKPOINT_WRITE_MANIFESTS_SQL,
            _CHECKPOINT_WRITE_MANIFESTS_RLS_SQL,
            _CHECKPOINT_WRITE_MANIFESTS_FORCE_RLS_SQL,
            _CHECKPOINT_WRITE_MANIFESTS_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=7,
        name="checkpoint_storage_authority",
        steps=(
            _ADD_CHECKPOINT_STORAGE_AUTHORITY_SQL,
            "ALTER TABLE platform_checkpoint_generations NO FORCE ROW LEVEL SECURITY",
            "ALTER TABLE platform_sessions NO FORCE ROW LEVEL SECURITY",
            PlatformDataMigration.REQUIRE_CURRENT_ENCRYPTED_PAYLOADS,
            PlatformDataMigration.CHECKPOINT_STORAGE_AUTHORITY_V1,
            "ALTER TABLE platform_sessions FORCE ROW LEVEL SECURITY",
            "ALTER TABLE platform_checkpoint_generations FORCE ROW LEVEL SECURITY",
            _CONSTRAIN_CHECKPOINT_STORAGE_AUTHORITY_SQL,
            _UNIQUE_OPERATION_REVISION_SQL,
            _REQUIRE_CURRENT_SESSION_PAYLOAD_SCHEMA_SQL,
            _REQUIRE_CURRENT_OPERATION_RESULT_SCHEMA_SQL,
            _CHECKPOINTS_RLS_SQL,
            _CHECKPOINTS_FORCE_RLS_SQL,
            _CHECKPOINTS_POLICY_SQL,
            _CHECKPOINT_BLOBS_RLS_SQL,
            _CHECKPOINT_BLOBS_FORCE_RLS_SQL,
            _CHECKPOINT_BLOBS_POLICY_SQL,
            _CHECKPOINT_WRITES_RLS_SQL,
            _CHECKPOINT_WRITES_FORCE_RLS_SQL,
            _CHECKPOINT_WRITES_POLICY_SQL,
        ),
    ),
    PlatformMigration(
        version=8,
        name="fenced_session_close",
        steps=(_ALLOW_FENCED_SESSION_CLOSE_SQL,),
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


async def _backfill_checkpoint_storage_authority_v1(connection: AsyncConnection) -> None:
    cursor = await connection.execute(
        """
        SELECT generation.tenant_id, generation.logical_session_id,
            generation.checkpoint_namespace, session_row.deployment_id,
            session_row.graph_contract
        FROM platform_checkpoint_generations AS generation
        JOIN platform_sessions AS session_row
          ON session_row.tenant_id = generation.tenant_id
         AND session_row.logical_session_id = generation.logical_session_id
        WHERE generation.storage_thread_id IS NULL
        ORDER BY generation.tenant_id, generation.logical_session_id,
            generation.checkpoint_namespace
        """
    )
    rows = await cursor.fetchall()
    for tenant_id, logical_session_id, checkpoint_namespace, deployment_id, graph_contract in rows:
        binding = CheckpointBinding(
            tenant_id=tenant_id,
            logical_session_id=logical_session_id,
            deployment_id=deployment_id,
            graph_contract=graph_contract,
            thread_id=checkpoint_namespace,
        )
        cursor = await connection.execute(
            """
            UPDATE platform_checkpoint_generations
            SET storage_thread_id = %s
            WHERE tenant_id = %s AND logical_session_id = %s
              AND checkpoint_namespace = %s AND storage_thread_id IS NULL
            """,
            (
                binding.storage_thread_id,
                binding.tenant_id,
                binding.logical_session_id,
                binding.thread_id,
            ),
        )
        if cursor.rowcount != 1:
            raise PlatformSchemaError("checkpoint storage-authority backfill lost its target")


async def _require_current_encrypted_payloads(connection: AsyncConnection) -> None:
    cursor = await connection.execute(
        """
        SELECT
            EXISTS (
                SELECT 1 FROM platform_sessions
                WHERE lifecycle <> 'closed'
                  AND payload_schema_version IS DISTINCT FROM %s
            ),
            EXISTS (
                SELECT 1 FROM platform_session_operations
                WHERE result_schema_version IS DISTINCT FROM %s
            )
        """,
        (SESSION_PAYLOAD_SCHEMA_VERSION, SESSION_OPERATION_RESULT_SCHEMA_VERSION),
    )
    row = await cursor.fetchone()
    if row is None or row[0] or row[1]:
        raise PlatformSchemaError(
            "platform migration cannot retain unsupported encrypted session payloads"
        )


async def _apply_migration_step(
    connection: AsyncConnection,
    step: PlatformMigrationStep,
) -> None:
    if not isinstance(step, PlatformDataMigration):
        await connection.execute(sql.SQL(step))
        return
    match step:
        case PlatformDataMigration.REQUIRE_CURRENT_ENCRYPTED_PAYLOADS:
            await _require_current_encrypted_payloads(connection)
        case PlatformDataMigration.CHECKPOINT_STORAGE_AUTHORITY_V1:
            await _backfill_checkpoint_storage_authority_v1(connection)
        case _ as unhandled:
            assert_never(unhandled)


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
    if connection.info.transaction_status is not TransactionStatus.IDLE:
        raise PlatformSchemaError("platform migration connection must be idle")
    restore_transaction_mode = not connection.autocommit
    if restore_transaction_mode:
        await connection.set_autocommit(True)
    try:
        await AsyncPostgresSaver(connection).setup()
        await _require_checkpoint_vendor_schema(connection)
    finally:
        if restore_transaction_mode:
            await connection.set_autocommit(False)
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
            for step in migration.steps:
                await _apply_migration_step(connection, step)
            await connection.execute(
                """
                INSERT INTO platform_schema_migrations (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (migration.version, migration.name, migration.checksum),
            )


async def _require_checkpoint_vendor_schema(connection: AsyncConnection) -> None:
    cursor = await connection.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ANY(%s)
        ORDER BY table_name, ordinal_position
        """,
        (list(_CHECKPOINT_VENDOR_COLUMNS),),
    )
    observed: dict[str, list[str]] = {}
    for table_name, column_name in await cursor.fetchall():
        observed.setdefault(str(table_name), []).append(str(column_name))
    expected = {table: list(columns) for table, columns in _CHECKPOINT_VENDOR_COLUMNS.items()}
    if observed != expected:
        raise PlatformSchemaError("pinned checkpoint schema signature does not match the database")
    cursor = await connection.execute("SELECT max(v) FROM checkpoint_migrations")
    row = await cursor.fetchone()
    if row is None or row[0] != _CHECKPOINT_VENDOR_MIGRATION_VERSION:
        raise PlatformSchemaError("pinned checkpoint migration version does not match the database")


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


async def require_platform_application_role(
    connection: AsyncConnection,
    *,
    schema_name: str,
) -> None:
    """Verify the connected role and search path without changing database state."""
    schema_name = _DATABASE_IDENTIFIER.validate_python(schema_name)
    cursor = await connection.execute(
        """
        SELECT current_user, current_schema(), current_schemas(false),
            role.rolsuper, role.rolbypassrls,
            has_schema_privilege(current_user, %s, 'USAGE'),
            has_schema_privilege(current_user, %s, 'CREATE'),
            EXISTS (
                SELECT 1
                FROM pg_class AS relation
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = %s
                  AND pg_get_userbyid(relation.relowner) = current_user
            )
        FROM pg_roles AS role
        WHERE role.rolname = current_user
        """,
        (schema_name, schema_name, schema_name),
    )
    row = await cursor.fetchone()
    if row is None:
        raise PlatformSchemaError("platform application role is unavailable")
    (
        _role_name,
        current_schema,
        current_schemas,
        is_superuser,
        bypasses_rls,
        has_schema_usage,
        has_schema_create,
        owns_relation,
    ) = row
    if current_schema != schema_name or current_schemas != [schema_name]:
        raise PlatformSchemaError("platform application search path is not trusted")
    if is_superuser or bypasses_rls:
        raise PlatformSchemaError("platform application role bypasses row-level security")
    if not has_schema_usage or has_schema_create or owns_relation:
        raise PlatformSchemaError("platform application role has unsafe schema authority")

    privilege_contract = tuple(
        (table_name, privilege, privilege in allowed)
        for table_name, allowed in _APPLICATION_TABLE_PRIVILEGES.items()
        for privilege in _TABLE_PRIVILEGES
    )
    cursor = await connection.execute(
        """
        WITH expected(table_name, privilege, allowed) AS (
            SELECT * FROM unnest(%s::text[], %s::text[], %s::boolean[])
        ), observed AS (
            SELECT allowed,
                to_regclass(format('%%I.%%I', %s::text, table_name)) AS relation,
                privilege
            FROM expected
        )
        SELECT bool_and(
            relation IS NOT NULL
            AND has_table_privilege(current_user, relation, privilege)
                IS NOT DISTINCT FROM allowed
        )
        FROM observed
        """,
        (
            [item[0] for item in privilege_contract],
            [item[1] for item in privilege_contract],
            [item[2] for item in privilege_contract],
            schema_name,
        ),
    )
    privilege_row = await cursor.fetchone()
    if privilege_row is None or privilege_row[0] is not True:
        raise PlatformSchemaError(
            "platform application role privileges do not match the runtime contract"
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
    tables = {
        table_name: sql.Identifier(schema_name, table_name)
        for table_name in _APPLICATION_TABLE_PRIVILEGES
    }
    async with connection.transaction():
        cursor = await connection.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s",
            (role_name,),
        )
        role_attributes = await cursor.fetchone()
        if role_attributes is None:
            raise PlatformSchemaError("platform application role does not exist")
        if role_attributes[0] or role_attributes[1]:
            raise PlatformSchemaError("platform application role bypasses row-level security")
        cursor = await connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_class AS relation
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                JOIN pg_roles AS owner_role ON owner_role.oid = relation.relowner
                WHERE namespace.nspname = %s AND owner_role.rolname = %s
            )
            """,
            (schema_name, role_name),
        )
        owns_relation = await cursor.fetchone()
        if owns_relation is None or owns_relation[0]:
            raise PlatformSchemaError("platform application role must not own runtime relations")
        for table in tables.values():
            await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM PUBLIC").format(table))
            await connection.execute(sql.SQL("REVOKE ALL ON TABLE {} FROM {}").format(table, role))
        await connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM PUBLIC").format(schema))
        await connection.execute(sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(schema, role))
        await connection.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, role))
        for table_name, privileges in _APPLICATION_TABLE_PRIVILEGES.items():
            await connection.execute(
                sql.SQL("GRANT {} ON TABLE {} TO {}").format(
                    sql.SQL(", ").join(sql.SQL(privilege) for privilege in privileges),
                    tables[table_name],
                    role,
                )
            )
