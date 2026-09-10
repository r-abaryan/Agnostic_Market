"""Run durable session expiry and tombstone cleanup outside voice jobs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import selectors
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.durability.platform_runtime import (
    DurablePlatformResources,
    load_platform_runtime_config,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"

logger = logging.getLogger("session_reaper")


def _install_stop_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    stop: asyncio.Event,
) -> Callable[[], None]:
    loop_handlers: list[signal.Signals] = []
    process_handlers = []

    def request_stop(_signum: int, _frame: object) -> None:
        loop.call_soon_threadsafe(stop.set)

    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled_signal, stop.set)
            loop_handlers.append(handled_signal)
        except NotImplementedError:
            previous = signal.getsignal(handled_signal)
            signal.signal(handled_signal, request_stop)
            process_handlers.append((handled_signal, previous))

    def restore() -> None:
        for handled_signal in loop_handlers:
            loop.remove_signal_handler(handled_signal)
        for handled_signal, previous in process_handlers:
            signal.signal(handled_signal, previous)

    return restore


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-config",
        type=Path,
        required=True,
        help="deployment-owned PlatformRuntimeConfig YAML",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=_CONFIG_ROOT,
        help="merchant configuration root used to enumerate trusted tenant ids",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one bounded cleanup cycle and exit",
    )
    return parser.parse_args()


async def _run(arguments: argparse.Namespace) -> None:
    stop = asyncio.Event()
    restore_signals = None
    if not arguments.once:
        restore_signals = _install_stop_signal_handlers(asyncio.get_running_loop(), stop)
    try:
        tenant_registry = ConfigRegistry(arguments.config_root).load()
        platform_config = load_platform_runtime_config(arguments.platform_config)
        resources = await DurablePlatformResources.open(platform_config, EnvSecretResolver())
        try:
            reaper = resources.build_reaper(tuple(tenant_registry.merchant_ids))
            if arguments.once:
                closed = await reaper.run_once()
                logger.info("durable session reaper closed %d sessions", closed)
                return
            await reaper.serve(stop)
        finally:
            await resources.aclose()
    finally:
        if restore_signals is not None:
            restore_signals()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(_arguments()), loop_factory=_event_loop_factory)


if __name__ == "__main__":
    main()
