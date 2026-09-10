"""Deployment-owned durable runtime configuration contracts."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from agnostic_market.dtos.platform import (
    DurableSessionConfig,
    PlatformDatabaseConfig,
    PlatformRuntimeConfig,
    SessionEncryptionConfig,
)
from agnostic_market.dtos.session import AdmittedSessionAuthority, TransportAuthority
from agnostic_market.durability.latency import (
    DurableLatencyActivationError,
    deployment_runtime_contract_fingerprint,
)
from agnostic_market.durability.migrations import PLATFORM_SESSION_SCHEMA_VERSION
from agnostic_market.durability.platform_runtime import DurablePlatformResources


@pytest.mark.parametrize("stage", ("register", "restore", "reconcile", "bind", "success"))
@pytest.mark.parametrize("cancelled", (False, True))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_fresh_session_setup_failure_closes_confirmed_authority(
    stage: str,
    cancelled: bool,
    cleanup_fails: bool,
) -> None:
    failure = (
        asyncio.CancelledError("setup cancelled") if cancelled else RuntimeError("setup failed")
    )
    closer = SimpleNamespace(begin_close=AsyncMock(), finalize_close=AsyncMock())
    if cleanup_fails:
        closer.begin_close.side_effect = ValueError("close failed")
    registry = SimpleNamespace(
        register_and_acquire=AsyncMock(return_value=SimpleNamespace(fencing_generation=1)),
        restore=AsyncMock(),
        reconcile_checkpoint_revision=AsyncMock(),
    )
    resources = SimpleNamespace(
        config=PlatformRuntimeConfig.model_validate(_valid_config()),
        registry=registry,
        lifecycle=SimpleNamespace(bind=Mock(return_value=closer)),
        _bind_session_resources=AsyncMock(),
    )
    if stage != "success":
        target = {
            "register": registry.register_and_acquire,
            "restore": registry.restore,
            "reconcile": registry.reconcile_checkpoint_revision,
            "bind": resources._bind_session_resources,
        }[stage]
        target.side_effect = failure
    admitted = AdmittedSessionAuthority(
        logical_session_id="session-a",
        transport=TransportAuthority(
            provider="livekit", room_id="room-a", assignment_id="assignment-a", worker_id="worker-a"
        ),
    )

    async def acquire():
        return await DurablePlatformResources.acquire_fresh_session(
            resources,
            tenant_id="tenant-a",
            admitted_authority=admitted,
            deployment_id="deployment-a",
            config_version="config-a",
        )

    if stage == "success":
        assert await acquire() is resources._bind_session_resources.return_value
        resources.lifecycle.bind.assert_not_called()
        return
    with pytest.raises(type(failure)) as caught:
        await acquire()
    assert caught.value is failure
    if stage == "register":
        resources.lifecycle.bind.assert_not_called()
        registry.restore.assert_not_awaited()
        return
    closer.begin_close.assert_awaited_once()
    if cleanup_fails:
        assert isinstance(failure.__cause__, ValueError)
        closer.finalize_close.assert_not_awaited()
    else:
        closer.finalize_close.assert_awaited_once()
    authority = resources.lifecycle.bind.call_args.args[0]
    assert authority.authority == admitted
    assert (
        authority.lease_owner_id == registry.register_and_acquire.call_args.args[1].lease_owner_id
    )


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "graph_contract": "a" * 64,
        "database": {
            "application_dsn_ref": {
                "provider": "env",
                "locator": "PLATFORM_POSTGRES_DSN",
            },
            "schema_name": "agnostic_market",
            "minimum_pool_size": 2,
            "maximum_pool_size": 12,
            "connection_timeout_seconds": 3.0,
            "pool_acquisition_timeout_seconds": 0.5,
            "statement_timeout_seconds": 1.0,
            "transaction_timeout_seconds": 2.0,
            "operation_timeout_seconds": 3.0,
            "expected_schema_version": PLATFORM_SESSION_SCHEMA_VERSION,
        },
        "sessions": {
            "lease_duration_seconds": 30.0,
            "lease_renewal_interval_seconds": 10.0,
            "session_retention_seconds": 86_400,
            "closed_tombstone_retention_seconds": 604_800,
            "reaper_interval_seconds": 60.0,
            "reaper_batch_size": 100,
            "transport_retirement_timeout_seconds": 5.0,
        },
        "encryption": {
            "envelope_format": "aes_256_gcm_v1",
            "key_ref": {
                "provider": "env",
                "locator": "PLATFORM_SESSION_KEY",
            },
            "key_version": "key-2026-09",
            "key_encoding": "base64",
        },
    }


def test_latency_runtime_fingerprint_binds_endpoint_without_secret_material() -> None:
    config = PlatformRuntimeConfig.model_validate(_valid_config())
    first = deployment_runtime_contract_fingerprint(
        config,
        application_dsn=(
            "postgresql://platform_app:first-secret@db.example/platform?sslmode=require"
        ),
    )
    rotated_secret = deployment_runtime_contract_fingerprint(
        config,
        application_dsn=(
            "postgresql://platform_app:second-secret@db.example/platform?sslmode=require"
        ),
    )
    changed_endpoint = deployment_runtime_contract_fingerprint(
        config,
        application_dsn=(
            "postgresql://platform_app:first-secret@other.example/platform?sslmode=require"
        ),
    )

    assert first == rotated_secret
    assert first != changed_endpoint
    with pytest.raises(DurableLatencyActivationError, match="invalid database connection"):
        deployment_runtime_contract_fingerprint(config, application_dsn="not a connection string")


def test_platform_runtime_config_is_strict_and_uses_structured_secret_references() -> None:
    config = PlatformRuntimeConfig.model_validate(_valid_config())

    assert config.graph_contract == "a" * 64
    assert config.database.application_dsn_ref.uri == "env://PLATFORM_POSTGRES_DSN"
    assert config.database.schema_name == "agnostic_market"
    assert config.encryption.key_ref.uri == "env://PLATFORM_SESSION_KEY"
    assert config.encryption.key_encoding == "base64"
    assert config.sessions.lease_renewal_interval_seconds < config.sessions.lease_duration_seconds
    assert config.sessions.reaper_interval_seconds == 60.0
    assert config.sessions.reaper_batch_size == 100
    assert config.sessions.transport_retirement_timeout_seconds == 5.0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlatformRuntimeConfig.model_validate({**_valid_config(), "merchant_id": "acme_store"})


def test_platform_runtime_config_has_no_unmeasured_defaults() -> None:
    assert all(field.is_required() for field in PlatformRuntimeConfig.model_fields.values())
    assert all(
        field.is_required()
        for section in (
            PlatformDatabaseConfig,
            DurableSessionConfig,
            SessionEncryptionConfig,
        )
        for field in section.model_fields.values()
    )


@pytest.mark.parametrize(
    "section, field_name, value",
    (
        ("database", "application_dsn_ref", "postgresql://user:secret@database/service"),
        ("database", "application_dsn_ref", {"provider": "postgresql", "locator": "user:secret"}),
        ("encryption", "key_ref", "raw-key-material"),
        ("encryption", "key_ref", {"provider": "env", "locator": "secret value"}),
    ),
)
def test_platform_runtime_config_rejects_scalar_or_noncanonical_secret_references(
    section: str,
    field_name: str,
    value: object,
) -> None:
    candidate = _valid_config()
    candidate[section] = {**candidate[section], field_name: value}  # type: ignore[misc]

    with pytest.raises(ValidationError):
        PlatformRuntimeConfig.model_validate(candidate)


def test_platform_runtime_config_allows_a_lazy_database_pool() -> None:
    candidate = _valid_config()
    candidate["database"] = {  # type: ignore[assignment]
        **candidate["database"],  # type: ignore[misc]
        "minimum_pool_size": 0,
    }

    config = PlatformRuntimeConfig.model_validate(candidate)

    assert config.database.minimum_pool_size == 0


@pytest.mark.parametrize(
    "field_name",
    (
        "connection_timeout_seconds",
        "pool_acquisition_timeout_seconds",
        "statement_timeout_seconds",
        "transaction_timeout_seconds",
        "operation_timeout_seconds",
    ),
)
def test_platform_runtime_config_rejects_unbounded_database_budgets(
    field_name: str,
) -> None:
    candidate = _valid_config()
    database = dict(candidate["database"])  # type: ignore[arg-type]
    database[field_name] = float("inf")
    candidate["database"] = database

    with pytest.raises(ValidationError, match="finite_number"):
        PlatformRuntimeConfig.model_validate(candidate)


@pytest.mark.parametrize(
    "section, changes, expected_error",
    (
        (
            "database",
            {"minimum_pool_size": 8, "maximum_pool_size": 4},
            "maximum_pool_size",
        ),
        (
            "database",
            {"statement_timeout_seconds": 2.5, "transaction_timeout_seconds": 2.0},
            "statement timeout",
        ),
        (
            "database",
            {"transaction_timeout_seconds": 3.5, "operation_timeout_seconds": 3.0},
            "transaction timeout",
        ),
        (
            "sessions",
            {"lease_duration_seconds": 10.0, "lease_renewal_interval_seconds": 10.0},
            "renewal interval",
        ),
        (
            "sessions",
            {"lease_duration_seconds": 30.0, "session_retention_seconds": 20},
            "session retention",
        ),
    ),
)
def test_platform_runtime_config_rejects_incoherent_operational_budgets(
    section: str,
    changes: dict[str, object],
    expected_error: str,
) -> None:
    candidate = _valid_config()
    candidate[section] = {**candidate[section], **changes}  # type: ignore[misc]

    with pytest.raises(ValidationError, match=expected_error):
        PlatformRuntimeConfig.model_validate(candidate)


class _StartupPool:
    instance: _StartupPool | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.closed = False
        type(self).instance = self

    async def open(self, *, wait: bool, **options: float) -> None:
        assert wait is True
        assert options["timeout"] > 0

    def connection(self, **options: float):
        assert options["timeout"] > 0

        @asynccontextmanager
        async def connected():
            yield object()

        return connected()

    async def close(self, **options: float) -> None:
        assert options["timeout"] > 0
        self.closed = True


class _RetryableCloseStartupPool(_StartupPool):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.close_calls = 0

    async def close(self, **options: float) -> None:
        assert options["timeout"] > 0
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("pool close failed")
        self.closed = True


class _BlockingCloseStartupPool(_StartupPool):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def close(self, **options: float) -> None:
        assert options["timeout"] > 0
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True


class _CancellableVendorPool(AsyncConnectionPool):
    def __init__(self) -> None:
        super().__init__("", min_size=0, max_size=1, open=False)
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def _signal_stop_worker(self):
        self.close_started.set()
        await self.allow_close.wait()
        return await super()._signal_stop_worker()


class _RuntimeSecrets:
    def resolve(self, ref: str) -> str:
        if ref == "env://PLATFORM_POSTGRES_DSN":
            return "postgresql://runtime.invalid/platform"
        if ref == "env://PLATFORM_SESSION_KEY":
            return base64.b64encode(bytes(range(32))).decode("ascii")
        raise AssertionError(f"unexpected secret reference {ref}")


async def test_failed_platform_startup_closes_the_job_owned_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    async def fail_schema_gate(_connection: object, _version: int) -> None:
        raise RuntimeError("startup gate failed")

    monkeypatch.setattr(platform_runtime, "AsyncConnectionPool", _StartupPool)
    monkeypatch.setattr(platform_runtime, "require_platform_schema_version", fail_schema_gate)

    with pytest.raises(RuntimeError, match="startup gate failed"):
        await DurablePlatformResources.open(
            PlatformRuntimeConfig.model_validate(_valid_config()),
            _RuntimeSecrets(),
        )

    assert _StartupPool.instance is not None
    assert _StartupPool.instance.closed is True


async def test_cancelled_platform_startup_closes_the_job_owned_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    entered = asyncio.Event()

    async def block_schema_gate(_connection: object, _version: int) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(platform_runtime, "AsyncConnectionPool", _StartupPool)
    monkeypatch.setattr(platform_runtime, "require_platform_schema_version", block_schema_gate)

    opening = asyncio.create_task(
        DurablePlatformResources.open(
            PlatformRuntimeConfig.model_validate(_valid_config()),
            _RuntimeSecrets(),
        )
    )
    await entered.wait()
    opening.cancel("startup cancelled")

    with pytest.raises(asyncio.CancelledError, match="startup cancelled"):
        await opening

    assert _StartupPool.instance is not None
    assert _StartupPool.instance.closed is True


async def test_repeated_cancellation_cannot_interrupt_failed_startup_pool_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    entered = asyncio.Event()

    async def block_schema_gate(_connection: object, _version: int) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(platform_runtime, "AsyncConnectionPool", _BlockingCloseStartupPool)
    monkeypatch.setattr(platform_runtime, "require_platform_schema_version", block_schema_gate)

    opening = asyncio.create_task(
        DurablePlatformResources.open(
            PlatformRuntimeConfig.model_validate(_valid_config()),
            _RuntimeSecrets(),
        )
    )
    await entered.wait()
    opening.cancel("startup cancelled")
    pool = _BlockingCloseStartupPool.instance
    assert isinstance(pool, _BlockingCloseStartupPool)
    await pool.close_started.wait()
    opening.cancel("startup cancelled again")
    await asyncio.sleep(0)
    assert not opening.done()

    pool.allow_close.set()
    with pytest.raises(asyncio.CancelledError, match="startup cancelled"):
        await opening
    assert pool.closed is True


async def test_pool_cleanup_reports_cancellation_when_close_finishes_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    async def cancel_caller_after_close(close_task: asyncio.Task[None]) -> None:
        await close_task
        raise asyncio.CancelledError("late shutdown cancellation")

    monkeypatch.setattr(platform_runtime.asyncio, "shield", cancel_caller_after_close)
    close_task = asyncio.create_task(asyncio.sleep(0))

    cancellation = await platform_runtime._finish_pool_close(close_task)

    assert isinstance(cancellation, asyncio.CancelledError)
    assert str(cancellation) == "late shutdown cancellation"


async def test_platform_startup_preserves_the_gate_failure_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    async def fail_schema_gate(_connection: object, _version: int) -> None:
        raise ValueError("startup gate failed")

    monkeypatch.setattr(platform_runtime, "AsyncConnectionPool", _RetryableCloseStartupPool)
    monkeypatch.setattr(platform_runtime, "require_platform_schema_version", fail_schema_gate)

    with pytest.raises(ValueError, match="startup gate failed") as failure:
        await DurablePlatformResources.open(
            PlatformRuntimeConfig.model_validate(_valid_config()),
            _RuntimeSecrets(),
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert str(failure.value.__cause__) == "pool close failed"


async def test_platform_resource_close_can_be_retried_after_pool_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.durability import platform_runtime

    async def pass_schema_gate(_connection: object, _version: int) -> None:
        return None

    async def pass_role_gate(_connection: object, *, schema_name: str) -> None:
        assert schema_name == "agnostic_market"

    monkeypatch.setattr(platform_runtime, "AsyncConnectionPool", _RetryableCloseStartupPool)
    monkeypatch.setattr(platform_runtime, "require_platform_schema_version", pass_schema_gate)
    monkeypatch.setattr(platform_runtime, "require_platform_application_role", pass_role_gate)

    resources = await DurablePlatformResources.open(
        PlatformRuntimeConfig.model_validate(_valid_config()),
        _RuntimeSecrets(),
    )
    pool = _RetryableCloseStartupPool.instance
    assert isinstance(pool, _RetryableCloseStartupPool)

    with pytest.raises(RuntimeError, match="pool close failed"):
        await resources.aclose()
    await resources.aclose()

    assert pool.close_calls == 2
    assert pool.closed is True


async def test_platform_resource_close_resists_repeated_cancellation_until_pool_cleanup() -> None:
    pool = _CancellableVendorPool()
    await pool.open(wait=False)
    connection = SimpleNamespace(_pool=pool, close=AsyncMock())
    waiter = SimpleNamespace(fail=AsyncMock())
    cast(Any, pool)._pool.append(connection)
    cast(Any, pool)._waiting.append(waiter)
    resources = SimpleNamespace(
        config=PlatformRuntimeConfig.model_validate(_valid_config()),
        pool=pool,
        _close_lock=asyncio.Lock(),
        _closed=False,
    )

    closing = asyncio.create_task(DurablePlatformResources.aclose(resources))
    await pool.close_started.wait()
    closing.cancel("shutdown cancelled")
    await asyncio.sleep(0)
    assert not closing.done()
    closing.cancel("shutdown cancelled again")
    await asyncio.sleep(0)
    assert not closing.done()
    pool.allow_close.set()
    with pytest.raises(asyncio.CancelledError, match="shutdown cancelled"):
        await closing

    await DurablePlatformResources.aclose(resources)
    assert pool.closed
    connection.close.assert_awaited_once_with()
    waiter.fail.assert_awaited_once()
