"""Tenant resolution over the real fixture registry: known -> merchant_id; unknown -> loud."""

from __future__ import annotations

import pytest

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.tenancy.context import build_tenant_context
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver


def test_resolve_by_inbound_number(registry: ConfigRegistry) -> None:
    resolver = TenantResolver(registry)
    assert resolver.resolve_by_number("+15551230001") == "acme_store"
    assert resolver.resolve_by_number("+15551230002") == "demo_shop"


def test_unknown_number_fails_loudly(registry: ConfigRegistry) -> None:
    resolver = TenantResolver(registry)
    with pytest.raises(TenantResolutionError):
        resolver.resolve_by_number("+19999999999")


@pytest.mark.parametrize("number", (" +15551230001", "+15551230001 ", "15551230001"))
def test_noncanonical_inbound_number_fails_loudly(
    registry: ConfigRegistry,
    number: str,
) -> None:
    resolver = TenantResolver(registry)
    with pytest.raises(TenantResolutionError, match=r"canonical E\.164"):
        resolver.resolve_by_number(number)


def test_explicit_id_resolves_known_merchant(registry: ConfigRegistry) -> None:
    resolver = TenantResolver(registry)
    assert resolver.resolve_by_id("demo_shop") == "demo_shop"


def test_explicit_id_rejects_unknown_merchant(registry: ConfigRegistry) -> None:
    resolver = TenantResolver(registry)
    with pytest.raises(TenantResolutionError):
        resolver.resolve_by_id("ghost_store")


def test_tenant_context_is_frozen_and_bound(registry: ConfigRegistry) -> None:
    ctx = build_tenant_context(registry, "acme_store")
    assert ctx.tenant_id == "acme_store"
    assert ctx.config_version  # a hash string is present
    assert ctx.policy.max_order_value_usd == 2000
    # immutable per session (frozen dataclass)
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is the point
        ctx.tenant_id = "demo_shop"  # type: ignore[misc]


def test_tenant_context_normalizes_identity_and_requires_a_config_version(
    registry: ConfigRegistry,
) -> None:
    resolved = registry.get("acme_store")
    context = build_tenant_context(registry, "acme_store")

    normalized = type(context)(
        tenant_id="  acme_store  ",
        config_version=f"  {resolved.config_version}  ",
        policy=context.policy,
    )
    assert normalized.tenant_id == "acme_store"
    assert normalized.config_version == resolved.config_version

    with pytest.raises(ValueError, match="config version"):
        type(context)(tenant_id="acme_store", config_version=" ", policy=context.policy)
