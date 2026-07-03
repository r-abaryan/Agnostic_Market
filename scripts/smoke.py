"""Phase-0 smoke: exercise the exit end-to-end — load a config, resolve a tenant, fail loudly.

Run: uv run python scripts/smoke.py
"""

from __future__ import annotations

from pathlib import Path

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.config.resolver import SafetyLockViolationError, resolve_merchant_config
from agnostic_market.tenancy.context import build_tenant_context
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def main() -> None:
    # 1) Load + resolve all merchant configs.
    registry = ConfigRegistry(_CONFIG_ROOT).load()
    print(f"loaded merchants: {sorted(registry.merchant_ids)}")

    acme = registry.get("acme_store")
    print(
        f"acme_store resolved: cap=${acme.config.policies.max_order_value_usd:.0f} "
        f"tier={acme.config.isolation.tier} config_version={acme.config_version[:12]}..."
    )

    # 2) Resolve a tenant by inbound number, build the immutable session context.
    resolver = TenantResolver(registry)
    merchant_id = resolver.resolve_by_number("+15551230001")
    ctx = build_tenant_context(registry, merchant_id)
    print(
        f"resolved +15551230001 -> {merchant_id}; ctx bound "
        f"(tenant={ctx.tenant_id}, handoff_allowed={ctx.policy.allow_ai_merchant_handoff})"
    )

    # 3) Fail loudly: an override touching a safety-locked key is rejected.
    base = {"_safety_locked": ["_platform"], "_platform": {"payment": {"out_of_band_only": True}}}
    try:
        resolve_merchant_config(base, {}, {"_platform": {"payment": {"out_of_band_only": False}}})
    except SafetyLockViolationError as exc:
        print(f"safety-lock correctly rejected an override: {exc}")
    else:
        raise SystemExit("FAIL: safety-lock did not reject a locked-key override")

    # 4) Fail loudly: an unknown tenant.
    try:
        resolver.resolve_by_number("+10000000000")
    except TenantResolutionError as exc:
        print(f"unknown tenant correctly rejected: {exc}")
    else:
        raise SystemExit("FAIL: unknown tenant was not rejected")

    print("\nPhase-0 smoke OK - load a config, resolve a tenant, fail loudly. [PASS]")


if __name__ == "__main__":
    main()
