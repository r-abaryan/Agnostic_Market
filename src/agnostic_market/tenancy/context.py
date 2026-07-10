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
    policy = PolicyContext(
        max_order_value_usd=config.policies.max_order_value_usd,
        allow_ai_merchant_handoff=config.policies.allow_ai_merchant_handoff,
        refund_auto_approve_under_usd=config.policies.refunds.auto_approve_under_usd,
        refund_require_human_above_usd=config.policies.refunds.require_human_above_usd,
        refund_returnless_under_usd=config.policies.refunds.returnless_under_usd,
        pending_ttl_seconds=config.policies.pending_confirmation_ttl_seconds,
        spoken_policy_extra=config.policies.spoken_facts_extra,
    )
    return TenantContext(
        tenant_id=config.merchant_id,
        config_version=resolved.config_version,
        policy=policy,
    )
