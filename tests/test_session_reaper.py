"""Operational durable-session reaper entry-point contracts."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agnostic_market.durability.session_lifecycle import (
    ReaperCycleFailure,
    ReaperCycleResult,
    TenantReapResult,
)
from agnostic_market.durability.tenant_lifecycle import (
    TenantDurableRowCounts,
    TenantLifecycleEntry,
    TenantLifecycleInventory,
    TenantLifecycleInventoryError,
    TenantLifecycleState,
)
from scripts import session_reaper


def test_reaper_uses_a_psycopg_compatible_event_loop_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_reaper.sys, "platform", "win32")

    loop = session_reaper._event_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_reaper_signal_handler_requests_stop_and_restores_previous_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: dict[signal.Signals, object] = {}
    previous = object()

    class Loop:
        def add_signal_handler(self, _sig, _callback) -> None:
            raise NotImplementedError

        def call_soon_threadsafe(self, callback) -> None:
            callback()

    monkeypatch.setattr(session_reaper.signal, "getsignal", lambda _sig: previous)
    monkeypatch.setattr(
        session_reaper.signal,
        "signal",
        lambda sig, handler: handlers.__setitem__(sig, handler),
    )
    stop = asyncio.Event()
    restore = session_reaper._install_stop_signal_handlers(Loop(), stop)

    handler = handlers[signal.SIGTERM]
    assert callable(handler)
    handler(signal.SIGTERM, None)
    assert stop.is_set()

    restore()
    assert handlers[signal.SIGTERM] is previous


class _TenantRegistry:
    merchant_ids = frozenset({"demo_shop", "acme_store"})

    def __init__(self, _root: Path) -> None:
        pass

    def load(self) -> _TenantRegistry:
        return self


class _Reaper:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.runs = 0

    async def run_once(self) -> ReaperCycleResult:
        self.runs += 1
        result = ReaperCycleResult(
            tenants=(
                TenantReapResult(
                    tenant_id="acme_store",
                    sessions_closed=2 if self.fail else 3,
                    tombstones_purged=1,
                    failure_count=int(self.fail),
                ),
            )
        )
        if self.fail:
            raise ReaperCycleFailure(
                "cleanup failed",
                (RuntimeError("candidate failed"),),
                result,
            )
        return result

    async def serve(self, stop: asyncio.Event) -> None:
        await stop.wait()


class _Resources:
    instance: _Resources | None = None
    fail_reaper = False

    def __init__(self) -> None:
        self.closed = False
        self.tenant_ids: tuple[str, ...] | None = None
        self.reaper = _Reaper(fail=self.fail_reaper)
        self.registry = _DrainRegistry()
        type(self).instance = self

    @classmethod
    async def open(
        cls,
        _config: object,
        _secrets: object,
        *,
        application_dsn: str,
    ) -> _Resources:
        assert application_dsn == "postgresql://app@example.test/platform"
        return cls()

    def build_reaper(self, tenant_ids: tuple[str, ...]) -> _Reaper:
        self.tenant_ids = tenant_ids
        return self.reaper

    async def aclose(self) -> None:
        self.closed = True


class _DrainRegistry:
    async def tenant_durable_row_counts(self, tenant_id: str) -> TenantDurableRowCounts:
        assert tenant_id == "old_shop"
        return TenantDurableRowCounts(
            tenant_id=tenant_id,
            observed_at=datetime.now(UTC),
            platform_sessions=0,
            platform_session_operations=0,
            platform_checkpoint_generations=0,
            platform_checkpoint_write_manifests=0,
        )


class _Secrets:
    def resolve(self, uri: str) -> str:
        assert uri == "env://PLATFORM_POSTGRES_DSN"
        return "postgresql://app@example.test/platform"


def _arguments() -> argparse.Namespace:
    return argparse.Namespace(
        platform_config=Path("platform.yaml"),
        config_root=Path("config"),
        tenant_inventory=Path("tenant-lifecycle.yaml"),
        deployment_id="deployment-a",
        drain_evidence_dir=None,
        once=True,
    )


def _tenant_inventory() -> TenantLifecycleInventory:
    return TenantLifecycleInventory(
        schema_version=1,
        revision=1,
        previous_revision=None,
        previous_fingerprint=None,
        entries=(
            TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),
            TenantLifecycleEntry(tenant_id="demo_shop", state=TenantLifecycleState.ACTIVE),
            TenantLifecycleEntry(tenant_id="old_shop", state=TenantLifecycleState.RETIRING),
        ),
    )


def _drain_artifact_contents(path: Path) -> tuple[str, ...]:
    return tuple(
        artifact.read_text(encoding="utf-8")
        for artifact in path.glob("tenant-drain-old_shop-r1-*.json")
    )


def _drain_artifact_names(path: Path, pattern: str) -> tuple[str, ...]:
    return tuple(artifact.name for artifact in path.glob(pattern))


def _install_runtime_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = SimpleNamespace(
        database=SimpleNamespace(
            application_dsn_ref=SimpleNamespace(uri="env://PLATFORM_POSTGRES_DSN")
        )
    )
    monkeypatch.setattr(session_reaper, "load_platform_runtime_config", lambda _path: platform)
    monkeypatch.setattr(session_reaper, "EnvSecretResolver", _Secrets)
    monkeypatch.setattr(
        session_reaper,
        "deployment_runtime_contract_fingerprint",
        lambda _platform, *, application_dsn: "b" * 64,
    )
    monkeypatch.setattr(session_reaper, "DurablePlatformResources", _Resources)


@pytest.mark.parametrize("fail", (False, True))
async def test_reaper_entrypoint_owns_and_closes_its_resources(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fail: bool,
) -> None:
    _Resources.fail_reaper = fail
    monkeypatch.setattr(session_reaper, "ConfigRegistry", _TenantRegistry)
    monkeypatch.setattr(
        session_reaper,
        "load_tenant_lifecycle_inventory",
        lambda _path: _tenant_inventory(),
    )
    _install_runtime_fakes(monkeypatch)
    caplog.set_level(logging.INFO, logger="session_reaper")

    if fail:
        with pytest.raises(ReaperCycleFailure, match="cleanup failed"):
            await session_reaper._run(_arguments())
        assert "failed after closing 2 sessions and purging 1 tombstones" in caplog.text
    else:
        await session_reaper._run(_arguments())
        assert "closed 3 sessions and purged 1 tombstones" in caplog.text

    resources = _Resources.instance
    assert resources is not None
    assert resources.closed
    assert resources.tenant_ids is not None
    assert set(resources.tenant_ids) == {"acme_store", "demo_shop", "old_shop"}
    assert resources.reaper.runs == 1


async def test_continuous_reaper_uses_signal_stop_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored = False

    def install(_loop: asyncio.AbstractEventLoop, stop: asyncio.Event):
        stop.set()

        def restore() -> None:
            nonlocal restored
            restored = True

        return restore

    arguments = _arguments()
    arguments.once = False
    _Resources.fail_reaper = False
    monkeypatch.setattr(session_reaper, "ConfigRegistry", _TenantRegistry)
    monkeypatch.setattr(
        session_reaper,
        "load_tenant_lifecycle_inventory",
        lambda _path: _tenant_inventory(),
    )
    _install_runtime_fakes(monkeypatch)
    monkeypatch.setattr(session_reaper, "_install_stop_signal_handlers", install)

    await session_reaper._run(arguments)

    resources = _Resources.instance
    assert resources is not None and resources.closed
    assert restored


async def test_reaper_refuses_an_inventory_that_omits_an_admitted_merchant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = TenantLifecycleInventory(
        schema_version=1,
        revision=1,
        previous_revision=None,
        previous_fingerprint=None,
        entries=(TenantLifecycleEntry(tenant_id="acme_store", state=TenantLifecycleState.ACTIVE),),
    )
    monkeypatch.setattr(session_reaper, "ConfigRegistry", _TenantRegistry)
    monkeypatch.setattr(
        session_reaper,
        "load_tenant_lifecycle_inventory",
        lambda _path: invalid,
    )

    with pytest.raises(TenantLifecycleInventoryError, match="active tenant set"):
        await session_reaper._run(_arguments())


async def test_once_reaper_records_immutable_drain_evidence_for_retiring_tenants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = _arguments()
    arguments.drain_evidence_dir = tmp_path
    _Resources.fail_reaper = False
    monkeypatch.setattr(session_reaper, "ConfigRegistry", _TenantRegistry)
    monkeypatch.setattr(
        session_reaper,
        "load_tenant_lifecycle_inventory",
        lambda _path: _tenant_inventory(),
    )
    _install_runtime_fakes(monkeypatch)

    await session_reaper._run(arguments)

    artifacts = await asyncio.to_thread(_drain_artifact_contents, tmp_path)
    assert len(artifacts) == 1
    assert "old_shop" in artifacts[0]


async def test_drain_observation_retains_later_success_and_reports_aggregate_failure(
    tmp_path: Path,
) -> None:
    inventory = TenantLifecycleInventory(
        schema_version=1,
        revision=1,
        previous_revision=None,
        previous_fingerprint=None,
        entries=(
            TenantLifecycleEntry(tenant_id="blocked_shop", state=TenantLifecycleState.RETIRING),
            TenantLifecycleEntry(tenant_id="drained_shop", state=TenantLifecycleState.RETIRING),
        ),
    )
    calls: list[str] = []

    class Registry:
        async def tenant_durable_row_counts(self, tenant_id: str) -> TenantDurableRowCounts:
            calls.append(tenant_id)
            return TenantDurableRowCounts(
                tenant_id=tenant_id,
                observed_at=datetime(
                    2026,
                    9,
                    14,
                    12,
                    tzinfo=timezone(timedelta(hours=2)),
                ),
                platform_sessions=int(tenant_id == "blocked_shop"),
                platform_session_operations=0,
                platform_checkpoint_generations=0,
                platform_checkpoint_write_manifests=0,
            )

    with pytest.raises(ExceptionGroup, match="retiring tenant drain evidence failed") as caught:
        await session_reaper._record_retiring_tenant_drains(
            SimpleNamespace(registry=Registry()),
            inventory,
            tmp_path,
            deployment_id="deployment-a",
            runtime_contract_fingerprint="b" * 64,
        )

    assert calls == ["blocked_shop", "drained_shop"]
    assert len(caught.value.exceptions) == 1
    assert isinstance(caught.value.exceptions[0], TenantLifecycleInventoryError)
    blocked = await asyncio.to_thread(
        _drain_artifact_names, tmp_path, "tenant-drain-blocked_shop-*.json"
    )
    assert not blocked
    artifacts = await asyncio.to_thread(
        _drain_artifact_names, tmp_path, "tenant-drain-drained_shop-*.json"
    )
    assert len(artifacts) == 1
    assert "20260914T100000000000Z" in artifacts[0]


async def test_continuous_reaper_refuses_a_drain_evidence_directory() -> None:
    arguments = _arguments()
    arguments.once = False
    arguments.drain_evidence_dir = Path("evidence")

    with pytest.raises(ValueError, match="requires a bounded --once sweep"):
        await session_reaper._run(arguments)
