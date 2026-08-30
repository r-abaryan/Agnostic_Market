"""Tenancy — resolve a caller/request to a merchant, and carry an immutable tenant context.

- resolver.py : phone# / domain / explicit button -> merchant_id (registry-backed).
- context.py  : the per-session TenantContext (tenant_id + config_version + policy),
                bound once at session start and read-only thereafter (AGENTS §A5/§A0).
"""

from agnostic_market.tenancy.context import (
    TenantBound,
    TenantContext,
    build_tenant_context,
    normalize_tenant_id,
)
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver

__all__ = [
    "TenantBound",
    "TenantContext",
    "TenantResolutionError",
    "TenantResolver",
    "build_tenant_context",
    "normalize_tenant_id",
]
