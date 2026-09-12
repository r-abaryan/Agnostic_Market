"""Operational durable-session reaper entry-point contracts."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

import pytest

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

    async def run_once(self) -> int:
        self.runs += 1
        if self.fail:
            raise RuntimeError("cleanup failed")
        return 3

    async def serve(self, stop: asyncio.Event) -> None:
        await stop.wait()


class _Resources:
    instance: _Resources | None = None
    fail_reaper = False

    def __init__(self) -> None:
        self.closed = False
        self.tenant_ids: tuple[str, ...] | None = None
        self.reaper = _Reaper(fail=self.fail_reaper)
        type(self).instance = self

    @classmethod
    async def open(cls, _config: object, _secrets: object) -> _Resources:
        return cls()

    def build_reaper(self, tenant_ids: tuple[str, ...]) -> _Reaper:
        self.tenant_ids = tenant_ids
        return self.reaper

    async def aclose(self) -> None:
        self.closed = True


def _arguments() -> argparse.Namespace:
    return argparse.Namespace(
        platform_config=Path("platform.yaml"),
        config_root=Path("config"),
        once=True,
    )


@pytest.mark.parametrize("fail", (False, True))
async def test_reaper_entrypoint_owns_and_closes_its_resources(
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    _Resources.fail_reaper = fail
    monkeypatch.setattr(session_reaper, "ConfigRegistry", _TenantRegistry)
    monkeypatch.setattr(session_reaper, "load_platform_runtime_config", lambda _path: object())
    monkeypatch.setattr(session_reaper, "DurablePlatformResources", _Resources)

    if fail:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await session_reaper._run(_arguments())
    else:
        await session_reaper._run(_arguments())

    resources = _Resources.instance
    assert resources is not None
    assert resources.closed
    assert resources.tenant_ids is not None
    assert set(resources.tenant_ids) == {"acme_store", "demo_shop"}
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
    monkeypatch.setattr(session_reaper, "load_platform_runtime_config", lambda _path: object())
    monkeypatch.setattr(session_reaper, "DurablePlatformResources", _Resources)
    monkeypatch.setattr(session_reaper, "_install_stop_signal_handlers", install)

    await session_reaper._run(arguments)

    resources = _Resources.instance
    assert resources is not None and resources.closed
    assert restored
