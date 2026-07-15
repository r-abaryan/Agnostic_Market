"""The per-session TenantContext — bound once at session start, read-only thereafter.

`tenant_id`, `config_version`, and the policy snapshot are fixed for the session's life
(AGENTS §A5): the tenant is bound in code at session start, never derived from conversation
(§A0, injection defense), and the config_version is pinned so a mid-session hot-reload can't
change authority under an in-flight action (DESIGN_REVIEW M4).
"""

from __future__ import annotations

from dataclasses import dataclass

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.state import PolicyContext


@dataclass(frozen=True)
class TenantContext:
    """Immutable per-session tenant binding. Stamp `tenant_id` on every store query/tool call."""

    tenant_id: str
    config_version: str
    policy: PolicyContext


def build_tenant_context(registry: ConfigRegistry, merchant_id: str) -> TenantContext:
    """Build the frozen context for a session from the registry's resolved config."""
    resolved = registry.get(merchant_id)
    config = resolved.config
    return TenantContext(
        tenant_id=config.merchant_id,
        config_version=resolved.config_version,
        # The ONE config->runtime policy mapping (dtos/config.py to_policy_context).
        policy=config.policies.to_policy_context(),
    )
