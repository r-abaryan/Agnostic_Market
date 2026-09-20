"""Transactional storage boundary for merchant drafts and published versions."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

import agnostic_market.management.contracts as management_contracts
from agnostic_market.config.loader import ConfigError, config_version
from agnostic_market.config.registry import resolve_merchant_override
from agnostic_market.config.resolver import (
    ConfigResolutionError,
    PolicyBoundsViolationError,
    SafetyLockViolationError,
)
from agnostic_market.management.contracts import (
    MerchantAuditRecord,
    MerchantDraft,
    MerchantDraftValidationRequest,
    MerchantPublicationRequest,
    MerchantRetirementReceipt,
    MerchantRetirementRequest,
    MerchantRollbackRequest,
    MerchantValidationResult,
    PublicationReceipt,
    PublishedMerchantVersion,
    merchant_draft_fingerprint,
    merchant_publication_intent_fingerprint,
)

_REPOSITORY_SCHEMA_VERSION = 5
_REPOSITORY_SCHEMA_TABLE = "management_repository_schema"
_REPOSITORY_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE management_repository_schema (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE management_drafts (
        tenant_id TEXT NOT NULL,
        draft_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        payload TEXT NOT NULL,
        PRIMARY KEY (tenant_id, draft_id)
    )
    """,
    """
    CREATE TABLE management_versions (
        tenant_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        version_number INTEGER NOT NULL CHECK (version_number >= 1),
        payload TEXT NOT NULL,
        PRIMARY KEY (tenant_id, version_id),
        UNIQUE (tenant_id, version_number)
    )
    """,
    """
    CREATE TABLE management_active_versions (
        tenant_id TEXT PRIMARY KEY,
        version_id TEXT NOT NULL,
        version_number INTEGER NOT NULL CHECK (version_number >= 1),
        FOREIGN KEY (tenant_id, version_id)
            REFERENCES management_versions (tenant_id, version_id)
    )
    """,
    """
    CREATE TABLE management_publication_requests (
        tenant_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('publish', 'retire', 'rollback')),
        request_fingerprint TEXT NOT NULL,
        receipt_payload TEXT NOT NULL,
        PRIMARY KEY (tenant_id, request_id)
    )
    """,
    """
    CREATE TABLE management_audit_records (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        payload TEXT NOT NULL,
        UNIQUE (tenant_id, event_id)
    )
    """,
    """
    CREATE TABLE management_validation_requests (
        tenant_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        request_fingerprint TEXT NOT NULL,
        result_payload TEXT NOT NULL,
        audit_event_id TEXT NOT NULL,
        PRIMARY KEY (tenant_id, request_id),
        UNIQUE (tenant_id, audit_event_id),
        FOREIGN KEY (tenant_id, audit_event_id)
            REFERENCES management_audit_records (tenant_id, event_id)
    )
    """,
)
_REPOSITORY_TABLES = frozenset(
    (
        _REPOSITORY_SCHEMA_TABLE,
        "management_drafts",
        "management_versions",
        "management_active_versions",
        "management_publication_requests",
        "management_audit_records",
        "management_validation_requests",
    )
)
type Clock = Callable[[], datetime]
type VersionIdFactory = Callable[[], str]
type AuditEventIdFactory = Callable[[], str]


class ManagementRepositoryError(RuntimeError):
    """Base error for the merchant configuration repository."""


class ManagementRepositoryConflictError(ManagementRepositoryError):
    """An optimistic concurrency expectation no longer matches stored state."""


class ManagementRepositoryReplayConflictError(ManagementRepositoryError):
    """An idempotency key was reused for a different request."""


class ManagementRepositoryNotFoundError(ManagementRepositoryError):
    """A requested draft or immutable version does not exist."""


class ManagementRepositoryDataError(ManagementRepositoryError):
    """Persisted repository data does not satisfy the current typed contract."""


class MerchantConfigurationRepository(Protocol):
    """Storage-independent operations required by the management service."""

    def list_tenant_ids(self) -> tuple[str, ...]: ...

    def get_draft(self, tenant_id: str, draft_id: str) -> MerchantDraft | None: ...

    def save_draft(self, draft: MerchantDraft, *, expected_revision: int) -> MerchantDraft: ...

    def get_version(self, tenant_id: str, version_id: str) -> PublishedMerchantVersion | None: ...

    def list_versions(self, tenant_id: str) -> tuple[PublishedMerchantVersion, ...]: ...

    def get_active_version(self, tenant_id: str) -> PublishedMerchantVersion | None: ...

    def list_audit_records(self, tenant_id: str) -> tuple[MerchantAuditRecord, ...]: ...

    def record_validation(
        self,
        request: MerchantDraftValidationRequest,
        result: MerchantValidationResult,
    ) -> MerchantValidationResult: ...

    def replay_publication(
        self,
        *,
        tenant_id: str,
        request_id: str,
        request_fingerprint: str,
    ) -> PublicationReceipt | None: ...

    def publish(self, request: MerchantPublicationRequest) -> PublicationReceipt: ...

    def retire(self, request: MerchantRetirementRequest) -> MerchantRetirementReceipt: ...

    def rollback(self, request: MerchantRollbackRequest) -> PublicationReceipt: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _version_id() -> str:
    return f"version-{uuid.uuid4().hex}"


def _audit_event_id() -> str:
    return f"audit-{uuid.uuid4().hex}"


def _model_payload(model: BaseModel) -> str:
    return model.model_dump_json()


def _load_model[ModelT: BaseModel](
    payload: str, model_type: type[ModelT], *, subject: str
) -> ModelT:
    try:
        return model_type.model_validate_json(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ManagementRepositoryDataError(f"stored {subject} is invalid") from exc


def _load_published_version(payload: str, *, subject: str) -> PublishedMerchantVersion:
    try:
        return PublishedMerchantVersion.from_persisted_json(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ManagementRepositoryDataError(f"stored {subject} is invalid") from exc


def _request_fingerprint(kind: str, request: BaseModel) -> str:
    return config_version(
        {
            "kind": kind,
            "request": request.model_dump(mode="json"),
        }
    )


def _publication_request_fingerprint(request: MerchantPublicationRequest) -> str:
    return merchant_publication_intent_fingerprint(
        tenant_id=request.tenant_id,
        draft_id=request.preview.draft_id,
        draft_revision=request.preview.draft_revision,
        expected_preview_fingerprint=request.preview.preview_fingerprint,
        expected_active_version_id=request.expected_active_version_id,
        actor_id=request.actor_id,
        request_id=request.request_id,
    )


class SqliteMerchantConfigurationRepository:
    """SQLite-backed development repository with transactional compare-and-swap writes."""

    def __init__(
        self,
        database_path: Path,
        *,
        active_config_root: Path,
        busy_timeout_seconds: float = 5.0,
        clock: Clock = _utc_now,
        version_id_factory: VersionIdFactory = _version_id,
        audit_event_id_factory: AuditEventIdFactory = _audit_event_id,
    ) -> None:
        if busy_timeout_seconds <= 0:
            raise ValueError("repository busy timeout must be positive")
        resolved_database_path = database_path.resolve()
        resolved_config_root = active_config_root.resolve()
        if resolved_database_path == resolved_config_root or resolved_database_path.is_relative_to(
            resolved_config_root
        ):
            raise ValueError("management repository must live outside the active config tree")
        self._database_path = resolved_database_path
        self._active_config_root = resolved_config_root
        self._busy_timeout_ms = ceil(busy_timeout_seconds * 1000)
        self._clock = clock
        self._version_id_factory = version_id_factory
        self._audit_event_id_factory = audit_event_id_factory
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._transaction() as connection:
                existing_tables = frozenset(
                    row["name"]
                    for row in connection.execute(
                        """
                        SELECT name FROM sqlite_master
                        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                        """
                    ).fetchall()
                )
                if _REPOSITORY_SCHEMA_TABLE in existing_tables:
                    row = connection.execute(
                        """
                        SELECT schema_version FROM management_repository_schema
                        WHERE singleton = 1
                        """
                    ).fetchone()
                    if row is None or row["schema_version"] != _REPOSITORY_SCHEMA_VERSION:
                        raise ManagementRepositoryDataError(
                            "management repository schema version is unsupported"
                        )
                    missing_tables = _REPOSITORY_TABLES - existing_tables
                    if missing_tables:
                        raise ManagementRepositoryDataError(
                            "management repository schema is incomplete"
                        )
                    return
                if existing_tables:
                    raise ManagementRepositoryDataError(
                        "management repository schema version is unsupported"
                    )
                for statement in _REPOSITORY_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO management_repository_schema (singleton, schema_version)
                    VALUES (1, ?)
                    """,
                    (_REPOSITORY_SCHEMA_VERSION,),
                )
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("management repository initialization failed") from exc

    def list_tenant_ids(self) -> tuple[str, ...]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT tenant_id FROM management_drafts
                    UNION
                    SELECT tenant_id FROM management_versions
                    ORDER BY tenant_id
                    """
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("managed tenant listing failed") from exc
        return tuple(row["tenant_id"] for row in rows)

    def get_draft(self, tenant_id: str, draft_id: str) -> MerchantDraft | None:
        try:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT payload FROM management_drafts
                    WHERE tenant_id = ? AND draft_id = ?
                    """,
                    (tenant_id, draft_id),
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("management draft read failed") from exc
        if row is None:
            return None
        return _load_model(row["payload"], MerchantDraft, subject="merchant draft")

    def save_draft(self, draft: MerchantDraft, *, expected_revision: int) -> MerchantDraft:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected draft revision must be a non-negative integer")
        try:
            with self._transaction() as connection:
                row = connection.execute(
                    """
                    SELECT revision, payload FROM management_drafts
                    WHERE tenant_id = ? AND draft_id = ?
                    """,
                    (draft.tenant_id, draft.draft_id),
                ).fetchone()
                if row is None:
                    if expected_revision != 0 or draft.revision != 1:
                        raise ManagementRepositoryConflictError(
                            "draft creation no longer matches the expected revision"
                        )
                    connection.execute(
                        """
                        INSERT INTO management_drafts (tenant_id, draft_id, revision, payload)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            draft.tenant_id,
                            draft.draft_id,
                            draft.revision,
                            _model_payload(draft),
                        ),
                    )
                    self._store_audit(
                        connection,
                        MerchantAuditRecord(
                            event_id=self._audit_event_id_factory(),
                            event="draft_created",
                            tenant_id=draft.tenant_id,
                            actor_id=draft.actor_id,
                            request_id=draft.request_id,
                            occurred_at=self._clock(),
                            draft_id=draft.draft_id,
                            payload_fingerprint=config_version(draft.model_dump(mode="json")),
                        ),
                    )
                    return draft

                current = _load_model(row["payload"], MerchantDraft, subject="merchant draft")
                if (
                    current.tenant_id != draft.tenant_id
                    or current.draft_id != draft.draft_id
                    or current.revision != row["revision"]
                ):
                    raise ManagementRepositoryDataError(
                        "stored merchant draft does not match its repository key"
                    )
                if current == draft:
                    if expected_revision == draft.revision - 1:
                        return current
                    raise ManagementRepositoryConflictError(
                        "draft retry does not match its original expected revision"
                    )
                if row["revision"] != expected_revision or draft.revision != expected_revision + 1:
                    raise ManagementRepositoryConflictError(
                        "draft update no longer matches the expected revision"
                    )
                updated = connection.execute(
                    """
                    UPDATE management_drafts SET revision = ?, payload = ?
                    WHERE tenant_id = ? AND draft_id = ? AND revision = ?
                    """,
                    (
                        draft.revision,
                        _model_payload(draft),
                        draft.tenant_id,
                        draft.draft_id,
                        expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise ManagementRepositoryConflictError(
                        "draft update lost its optimistic concurrency race"
                    )
                self._store_audit(
                    connection,
                    MerchantAuditRecord(
                        event_id=self._audit_event_id_factory(),
                        event="draft_updated",
                        tenant_id=draft.tenant_id,
                        actor_id=draft.actor_id,
                        request_id=draft.request_id,
                        occurred_at=self._clock(),
                        draft_id=draft.draft_id,
                        payload_fingerprint=config_version(draft.model_dump(mode="json")),
                    ),
                )
                return draft
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("management draft write failed") from exc

    def get_version(self, tenant_id: str, version_id: str) -> PublishedMerchantVersion | None:
        try:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT payload FROM management_versions
                    WHERE tenant_id = ? AND version_id = ?
                    """,
                    (tenant_id, version_id),
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("published version read failed") from exc
        if row is None:
            return None
        return _load_published_version(row["payload"], subject="published merchant version")

    def list_versions(self, tenant_id: str) -> tuple[PublishedMerchantVersion, ...]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT version_id, version_number, payload FROM management_versions
                    WHERE tenant_id = ?
                    ORDER BY version_number
                    """,
                    (tenant_id,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("published version listing failed") from exc
        versions: list[PublishedMerchantVersion] = []
        for row in rows:
            version = _load_published_version(row["payload"], subject="published merchant version")
            if (
                version.tenant_id != tenant_id
                or version.version_id != row["version_id"]
                or version.version_number != row["version_number"]
            ):
                raise ManagementRepositoryDataError(
                    "stored merchant version does not match its repository key"
                )
            versions.append(version)
        return tuple(versions)

    def get_active_version(self, tenant_id: str) -> PublishedMerchantVersion | None:
        try:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT version.payload
                    FROM management_active_versions AS active
                    JOIN management_versions AS version
                      ON version.tenant_id = active.tenant_id
                     AND version.version_id = active.version_id
                    WHERE active.tenant_id = ?
                    """,
                    (tenant_id,),
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("active merchant version read failed") from exc
        if row is None:
            return None
        return _load_published_version(row["payload"], subject="active merchant version")

    def list_audit_records(self, tenant_id: str) -> tuple[MerchantAuditRecord, ...]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT event_id, payload FROM management_audit_records
                    WHERE tenant_id = ?
                    ORDER BY sequence
                    """,
                    (tenant_id,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("management audit history read failed") from exc
        records: list[MerchantAuditRecord] = []
        for row in rows:
            record = _load_model(
                row["payload"], MerchantAuditRecord, subject="management audit record"
            )
            if record.tenant_id != tenant_id or record.event_id != row["event_id"]:
                raise ManagementRepositoryDataError(
                    "stored management audit record does not match its repository key"
                )
            records.append(record)
        return tuple(records)

    def record_validation(
        self,
        request: MerchantDraftValidationRequest,
        result: MerchantValidationResult,
    ) -> MerchantValidationResult:
        if (
            result.tenant_id != request.tenant_id
            or result.draft_id != request.draft_id
            or result.draft_revision != request.draft_revision
        ):
            raise ManagementRepositoryDataError(
                "validation result does not match its command identity"
            )
        fingerprint = _request_fingerprint("validate", request)
        try:
            with self._transaction() as connection:
                row = connection.execute(
                    """
                    SELECT request_fingerprint, result_payload
                    FROM management_validation_requests
                    WHERE tenant_id = ? AND request_id = ?
                    """,
                    (request.tenant_id, request.request_id),
                ).fetchone()
                if row is not None:
                    if row["request_fingerprint"] != fingerprint:
                        raise ManagementRepositoryReplayConflictError(
                            "validation request id was reused with different parameters"
                        )
                    replay = _load_model(
                        row["result_payload"],
                        MerchantValidationResult,
                        subject="validation result",
                    )
                    if (
                        replay.tenant_id != request.tenant_id
                        or replay.draft_id != request.draft_id
                        or replay.draft_revision != request.draft_revision
                    ):
                        raise ManagementRepositoryDataError(
                            "stored validation result does not match its repository key"
                        )
                    return replay

                event_id = self._audit_event_id_factory()
                self._store_audit(
                    connection,
                    MerchantAuditRecord(
                        event_id=event_id,
                        event="draft_validated",
                        tenant_id=request.tenant_id,
                        actor_id=request.actor_id,
                        request_id=request.request_id,
                        occurred_at=result.validated_at,
                        draft_id=request.draft_id,
                        payload_fingerprint=config_version(result.model_dump(mode="json")),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO management_validation_requests (
                        tenant_id, request_id, request_fingerprint,
                        result_payload, audit_event_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        request.tenant_id,
                        request.request_id,
                        fingerprint,
                        _model_payload(result),
                        event_id,
                    ),
                )
                return result
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("merchant validation recording failed") from exc

    def _replayed_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        request_id: str,
        kind: str,
        fingerprint: str,
    ) -> PublicationReceipt | None:
        row = connection.execute(
            """
            SELECT kind, request_fingerprint, receipt_payload
            FROM management_publication_requests
            WHERE tenant_id = ? AND request_id = ?
            """,
            (tenant_id, request_id),
        ).fetchone()
        if row is None:
            return None
        if row["kind"] != kind or row["request_fingerprint"] != fingerprint:
            raise ManagementRepositoryReplayConflictError(
                "publication request id was reused with different parameters"
            )
        receipt = _load_model(
            row["receipt_payload"], PublicationReceipt, subject="publication receipt"
        )
        if (
            receipt.tenant_id != tenant_id
            or receipt.request_id != request_id
            or receipt.kind != kind
        ):
            raise ManagementRepositoryDataError(
                "stored publication receipt does not match its repository key"
            )
        return receipt.model_copy(update={"replayed": True})

    def _replayed_retirement_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        request_id: str,
        fingerprint: str,
    ) -> MerchantRetirementReceipt | None:
        row = connection.execute(
            """
            SELECT kind, request_fingerprint, receipt_payload
            FROM management_publication_requests
            WHERE tenant_id = ? AND request_id = ?
            """,
            (tenant_id, request_id),
        ).fetchone()
        if row is None:
            return None
        if row["kind"] != "retire" or row["request_fingerprint"] != fingerprint:
            raise ManagementRepositoryReplayConflictError(
                "publication request id was reused with different parameters"
            )
        receipt = _load_model(
            row["receipt_payload"],
            MerchantRetirementReceipt,
            subject="retirement receipt",
        )
        if receipt.tenant_id != tenant_id or receipt.request_id != request_id:
            raise ManagementRepositoryDataError(
                "stored retirement receipt does not match its repository key"
            )
        return receipt.model_copy(update={"replayed": True})

    def replay_publication(
        self,
        *,
        tenant_id: str,
        request_id: str,
        request_fingerprint: str,
    ) -> PublicationReceipt | None:
        """Return an exact committed publication before mutable draft resolution."""

        try:
            connection = self._connect()
            try:
                return self._replayed_receipt(
                    connection,
                    tenant_id=tenant_id,
                    request_id=request_id,
                    kind="publish",
                    fingerprint=request_fingerprint,
                )
            finally:
                connection.close()
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("publication replay lookup failed") from exc

    @staticmethod
    def _active_pointer(connection: sqlite3.Connection, tenant_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT version_id, version_number FROM management_active_versions
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()

    @staticmethod
    def _latest_version_pointer(
        connection: sqlite3.Connection, tenant_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT version_id, version_number FROM management_versions
            WHERE tenant_id = ?
            ORDER BY version_number DESC
            LIMIT 1
            """,
            (tenant_id,),
        ).fetchone()

    @staticmethod
    def _require_expected_active(
        active: sqlite3.Row | None, expected_active_version_id: str | None
    ) -> None:
        active_version_id = None if active is None else active["version_id"]
        if active_version_id != expected_active_version_id:
            raise ManagementRepositoryConflictError(
                "active merchant version no longer matches the publication expectation"
            )

    def _write_active_pointer(
        self,
        connection: sqlite3.Connection,
        version: PublishedMerchantVersion,
    ) -> None:
        connection.execute(
            """
            INSERT INTO management_active_versions (tenant_id, version_id, version_number)
            VALUES (?, ?, ?)
            ON CONFLICT (tenant_id) DO UPDATE SET
                version_id = excluded.version_id,
                version_number = excluded.version_number
            """,
            (version.tenant_id, version.version_id, version.version_number),
        )

    @staticmethod
    def _store_version(
        connection: sqlite3.Connection,
        version: PublishedMerchantVersion,
    ) -> None:
        connection.execute(
            """
            INSERT INTO management_versions (tenant_id, version_id, version_number, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                version.tenant_id,
                version.version_id,
                version.version_number,
                _model_payload(version),
            ),
        )

    @staticmethod
    def _store_audit(
        connection: sqlite3.Connection,
        record: MerchantAuditRecord,
    ) -> None:
        connection.execute(
            """
            INSERT INTO management_audit_records (tenant_id, event_id, payload)
            VALUES (?, ?, ?)
            """,
            (record.tenant_id, record.event_id, _model_payload(record)),
        )

    @staticmethod
    def _store_receipt(
        connection: sqlite3.Connection,
        *,
        fingerprint: str,
        receipt: PublicationReceipt,
    ) -> None:
        connection.execute(
            """
            INSERT INTO management_publication_requests (
                tenant_id, request_id, kind, request_fingerprint, receipt_payload
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                receipt.tenant_id,
                receipt.request_id,
                receipt.kind,
                fingerprint,
                _model_payload(receipt),
            ),
        )

    @staticmethod
    def _store_retirement_receipt(
        connection: sqlite3.Connection,
        *,
        fingerprint: str,
        receipt: MerchantRetirementReceipt,
    ) -> None:
        connection.execute(
            """
            INSERT INTO management_publication_requests (
                tenant_id, request_id, kind, request_fingerprint, receipt_payload
            ) VALUES (?, ?, 'retire', ?, ?)
            """,
            (
                receipt.tenant_id,
                receipt.request_id,
                fingerprint,
                _model_payload(receipt),
            ),
        )

    def publish(self, request: MerchantPublicationRequest) -> PublicationReceipt:
        fingerprint = _publication_request_fingerprint(request)
        try:
            with self._transaction() as connection:
                replay = self._replayed_receipt(
                    connection,
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    kind="publish",
                    fingerprint=fingerprint,
                )
                if replay is not None:
                    return replay

                draft_row = connection.execute(
                    """
                    SELECT revision, payload FROM management_drafts
                    WHERE tenant_id = ? AND draft_id = ?
                    """,
                    (request.tenant_id, request.preview.draft_id),
                ).fetchone()
                if draft_row is None:
                    raise ManagementRepositoryNotFoundError(
                        "publication source draft does not exist"
                    )
                if draft_row["revision"] != request.preview.draft_revision:
                    raise ManagementRepositoryConflictError(
                        "publication preview is stale relative to its source draft"
                    )
                source_draft = _load_model(
                    draft_row["payload"], MerchantDraft, subject="publication source draft"
                )
                if (
                    source_draft.tenant_id != request.tenant_id
                    or source_draft.draft_id != request.preview.draft_id
                    or source_draft.revision != draft_row["revision"]
                ):
                    raise ManagementRepositoryDataError(
                        "publication source draft does not match its repository key"
                    )
                if (
                    merchant_draft_fingerprint(source_draft)
                    != request.preview.source_draft_fingerprint
                ):
                    raise ManagementRepositoryConflictError(
                        "publication preview does not match the exact source draft"
                    )
                if source_draft.fixtures != request.preview.fixtures:
                    raise ManagementRepositoryConflictError(
                        "publication preview fixtures do not match the source draft"
                    )
                try:
                    resolved_source = resolve_merchant_override(
                        self._active_config_root,
                        source_draft.merchant_override,
                        source=f"merchant draft {source_draft.draft_id}",
                    )
                except (
                    ConfigError,
                    ConfigResolutionError,
                    PolicyBoundsViolationError,
                    SafetyLockViolationError,
                ) as exc:
                    raise ManagementRepositoryConflictError(
                        "publication source draft no longer resolves"
                    ) from exc
                if (
                    resolved_source.config != request.preview.config
                    or resolved_source.config_version != request.preview.config_fingerprint
                ):
                    raise ManagementRepositoryConflictError(
                        "publication preview config does not match the source draft"
                    )

                active = self._active_pointer(connection, request.tenant_id)
                self._require_expected_active(active, request.expected_active_version_id)
                latest = self._latest_version_pointer(connection, request.tenant_id)
                previous_version_id = None if latest is None else latest["version_id"]
                version_number = 1 if latest is None else latest["version_number"] + 1
                version = PublishedMerchantVersion(
                    tenant_id=request.tenant_id,
                    version_id=self._version_id_factory(),
                    version_number=version_number,
                    previous_version_id=previous_version_id,
                    source_draft_id=request.preview.draft_id,
                    source_draft_revision=request.preview.draft_revision,
                    published_at=self._clock(),
                    published_by=request.actor_id,
                    config=request.preview.config,
                    fixtures=request.preview.fixtures,
                    config_fingerprint=request.preview.config_fingerprint,
                    fixture_fingerprint=request.preview.fixture_fingerprint,
                    schema_fingerprint=request.preview.schema_fingerprint,
                )
                self._store_version(connection, version)
                self._write_active_pointer(connection, version)
                self._store_audit(
                    connection,
                    MerchantAuditRecord(
                        event_id=self._audit_event_id_factory(),
                        event="version_published",
                        tenant_id=request.tenant_id,
                        actor_id=request.actor_id,
                        request_id=request.request_id,
                        occurred_at=version.published_at,
                        draft_id=request.preview.draft_id,
                        version_id=version.version_id,
                        payload_fingerprint=config_version(version.model_dump(mode="json")),
                    ),
                )
                receipt = PublicationReceipt(
                    kind="publish",
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    version_id=version.version_id,
                    version_number=version.version_number,
                    committed_at=version.published_at,
                    replayed=False,
                )
                self._store_receipt(connection, fingerprint=fingerprint, receipt=receipt)
                return receipt
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("merchant publication failed") from exc

    def retire(self, request: MerchantRetirementRequest) -> MerchantRetirementReceipt:
        fingerprint = _request_fingerprint("retire", request)
        try:
            with self._transaction() as connection:
                replay = self._replayed_retirement_receipt(
                    connection,
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    fingerprint=fingerprint,
                )
                if replay is not None:
                    return replay

                active = self._active_pointer(connection, request.tenant_id)
                if active is None:
                    raise ManagementRepositoryNotFoundError(
                        "retirement requires an active merchant version"
                    )
                self._require_expected_active(active, request.expected_active_version_id)
                deleted = connection.execute(
                    """
                    DELETE FROM management_active_versions
                    WHERE tenant_id = ? AND version_id = ?
                    """,
                    (request.tenant_id, request.expected_active_version_id),
                )
                if deleted.rowcount != 1:
                    raise ManagementRepositoryConflictError(
                        "active merchant version changed during retirement"
                    )
                receipt = MerchantRetirementReceipt(
                    tenant_id=request.tenant_id,
                    retired_version_id=request.expected_active_version_id,
                    request_id=request.request_id,
                    retired_at=self._clock(),
                    replayed=False,
                )
                self._store_audit(
                    connection,
                    MerchantAuditRecord(
                        event_id=self._audit_event_id_factory(),
                        event="version_retired",
                        tenant_id=request.tenant_id,
                        actor_id=request.actor_id,
                        request_id=request.request_id,
                        occurred_at=receipt.retired_at,
                        version_id=receipt.retired_version_id,
                        payload_fingerprint=config_version(receipt.model_dump(mode="json")),
                    ),
                )
                self._store_retirement_receipt(
                    connection,
                    fingerprint=fingerprint,
                    receipt=receipt,
                )
                return receipt
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("merchant retirement failed") from exc

    def rollback(self, request: MerchantRollbackRequest) -> PublicationReceipt:
        fingerprint = _request_fingerprint("rollback", request)
        try:
            with self._transaction() as connection:
                replay = self._replayed_receipt(
                    connection,
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    kind="rollback",
                    fingerprint=fingerprint,
                )
                if replay is not None:
                    return replay

                active = self._active_pointer(connection, request.tenant_id)
                if active is None:
                    raise ManagementRepositoryNotFoundError(
                        "rollback requires an active merchant version"
                    )
                self._require_expected_active(active, request.expected_active_version_id)
                source_row = connection.execute(
                    """
                    SELECT payload FROM management_versions
                    WHERE tenant_id = ? AND version_id = ?
                    """,
                    (request.tenant_id, request.source_version_id),
                ).fetchone()
                if source_row is None:
                    raise ManagementRepositoryNotFoundError(
                        "rollback source version does not exist"
                    )
                source = _load_published_version(
                    source_row["payload"], subject="rollback source version"
                )
                version = PublishedMerchantVersion(
                    tenant_id=request.tenant_id,
                    version_id=self._version_id_factory(),
                    version_number=active["version_number"] + 1,
                    previous_version_id=active["version_id"],
                    source_draft_id=source.source_draft_id,
                    source_draft_revision=source.source_draft_revision,
                    published_at=self._clock(),
                    published_by=request.actor_id,
                    config=source.config,
                    fixtures=source.fixtures,
                    config_fingerprint=config_version(source.config.model_dump(mode="json")),
                    fixture_fingerprint=management_contracts.merchant_fixture_bundle_fingerprint(
                        source.fixtures
                    ),
                    schema_fingerprint=management_contracts.management_contract_schema_fingerprint(),
                )
                self._store_version(connection, version)
                self._write_active_pointer(connection, version)
                self._store_audit(
                    connection,
                    MerchantAuditRecord(
                        event_id=self._audit_event_id_factory(),
                        event="version_rolled_back",
                        tenant_id=request.tenant_id,
                        actor_id=request.actor_id,
                        request_id=request.request_id,
                        occurred_at=version.published_at,
                        version_id=version.version_id,
                        payload_fingerprint=config_version(version.model_dump(mode="json")),
                    ),
                )
                receipt = PublicationReceipt(
                    kind="rollback",
                    tenant_id=request.tenant_id,
                    request_id=request.request_id,
                    version_id=version.version_id,
                    version_number=version.version_number,
                    source_version_id=source.version_id,
                    committed_at=version.published_at,
                    replayed=False,
                )
                self._store_receipt(connection, fingerprint=fingerprint, receipt=receipt)
                return receipt
        except ManagementRepositoryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise ManagementRepositoryError("merchant rollback publication failed") from exc
