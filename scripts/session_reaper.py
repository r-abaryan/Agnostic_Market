"""Run durable session expiry and tombstone cleanup outside voice jobs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import selectors
import signal
import sys
from collections.abc import Callable
from datetime import UTC
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.platform import ConfigIdentifier
from agnostic_market.durability.latency import deployment_runtime_contract_fingerprint
from agnostic_market.durability.platform_runtime import (
    DurablePlatformResources,
    load_platform_runtime_config,
)
from agnostic_market.durability.session_lifecycle import ReaperCycleFailure
from agnostic_market.durability.tenant_lifecycle import (
    TenantLifecycleInventory,
    build_tenant_drain_result,
    load_tenant_lifecycle_inventory,
    tenant_lifecycle_inventory_fingerprint,
    validate_tenant_lifecycle_inventory,
    write_tenant_drain_result,
)
from agnostic_market.secrets.env_resolver import EnvSecretResolver

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_CONFIG_IDENTIFIER = TypeAdapter(ConfigIdentifier)

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
        help="merchant configuration root used to validate active tenant admission",
    )
    parser.add_argument(
        "--tenant-inventory",
        type=Path,
        required=True,
        help="deployment-owned active and retiring tenant lifecycle inventory YAML",
    )
    parser.add_argument(
        "--deployment-id",
        required=True,
        help="immutable deployment identity bound into operational evidence",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one bounded cleanup cycle and exit",
    )
    parser.add_argument(
        "--drain-evidence-dir",
        type=Path,
        help="write immutable zero-state evidence for retiring tenants after a clean --once sweep",
    )
    return parser.parse_args()


async def _record_retiring_tenant_drains(
    resources: DurablePlatformResources,
    inventory: TenantLifecycleInventory,
    evidence_dir: Path,
    *,
    deployment_id: str,
    runtime_contract_fingerprint: str,
) -> None:
    fingerprint = tenant_lifecycle_inventory_fingerprint(inventory)
    failures: list[Exception] = []
    for tenant_id in inventory.retiring_tenant_ids:
        try:
            counts = await resources.registry.tenant_durable_row_counts(tenant_id)
            result = build_tenant_drain_result(
                inventory,
                counts,
                deployment_id=deployment_id,
                runtime_contract_fingerprint=runtime_contract_fingerprint,
            )
            timestamp = result.observed_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            path = evidence_dir / (
                f"tenant-drain-{tenant_id}-r{inventory.revision}-{fingerprint}-{timestamp}.json"
            )
            write_tenant_drain_result(path, result)
            logger.info("recorded retiring tenant drain evidence for %s at %s", tenant_id, path)
        except Exception as exc:
            exc.add_note(f"failed retiring tenant drain observation for {tenant_id}")
            failures.append(exc)
    if failures:
        raise ExceptionGroup("retiring tenant drain evidence failed", failures)


async def _run(arguments: argparse.Namespace) -> None:
    if arguments.drain_evidence_dir is not None and not arguments.once:
        raise ValueError("tenant drain evidence requires a bounded --once sweep")
    deployment_id = _CONFIG_IDENTIFIER.validate_python(arguments.deployment_id)
    stop = asyncio.Event()
    restore_signals = None
    if not arguments.once:
        restore_signals = _install_stop_signal_handlers(asyncio.get_running_loop(), stop)
    try:
        tenant_registry = ConfigRegistry(arguments.config_root).load()
        tenant_inventory = load_tenant_lifecycle_inventory(arguments.tenant_inventory)
        validate_tenant_lifecycle_inventory(tenant_inventory, tenant_registry.merchant_ids)
        platform_config = load_platform_runtime_config(arguments.platform_config)
        secrets = EnvSecretResolver()
        application_dsn = secrets.resolve(platform_config.database.application_dsn_ref.uri)
        runtime_contract_fingerprint = deployment_runtime_contract_fingerprint(
            platform_config,
            application_dsn=application_dsn,
        )
        resources = await DurablePlatformResources.open(
            platform_config,
            secrets,
            application_dsn=application_dsn,
        )
        try:
            reaper = resources.build_reaper(tenant_inventory.sweep_tenant_ids)
            if arguments.once:
                try:
                    result = await reaper.run_once()
                except ReaperCycleFailure as exc:
                    result = exc.result
                    logger.error(
                        "durable session reaper failed after closing %d sessions and purging "
                        "%d tombstones across %d tenants (%d failures)",
                        result.sessions_closed,
                        result.tombstones_purged,
                        result.tenants_attempted,
                        result.failure_count,
                    )
                    raise
                logger.info(
                    "durable session reaper closed %d sessions and purged %d tombstones across "
                    "%d tenants",
                    result.sessions_closed,
                    result.tombstones_purged,
                    result.tenants_attempted,
                )
                if arguments.drain_evidence_dir is not None:
                    await _record_retiring_tenant_drains(
                        resources,
                        tenant_inventory,
                        arguments.drain_evidence_dir,
                        deployment_id=deployment_id,
                        runtime_contract_fingerprint=runtime_contract_fingerprint,
                    )
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
