"""ONE PolicyContext factory for the suites.

Was an 8-way copied literal — every new REQUIRED PolicyContext field had to be added in eight
files, the exact lockstep drift the required fields exist to catch (but displaced into the
tests). Security knobs default to the platform values (config/base/base.yaml); a suite
overrides any field by keyword. Imported as `from policy_helpers import make_policy` (tests/
is on sys.path, like llm_fakes / support_helpers)."""

from __future__ import annotations

from agnostic_market.dtos.state import PolicyContext

_DEFAULTS: dict[str, object] = {
    "max_order_value_usd": 500.0,
    "allow_ai_merchant_handoff": True,
    "refund_auto_approve_under_usd": 50.0,
    "refund_require_human_above_usd": 200.0,
    # High by default: the support amount-gate/step-up scenarios refund the SHIPPED ORD-1001
    # in isolation and tighten this per-test via model_copy; the returns/return-first suites
    # pass their own low value. Suites that need a low default just override it.
    "refund_returnless_under_usd": 500.0,
    "return_window_days": 30,
    "pending_ttl_seconds": 120.0,
    "otp_max_attempts": 2,
    "contact_reask_max": 1,
    "auth_denials_before_human_offer": 2,
    "max_tool_hops": 5,
}


def make_policy(**overrides: object) -> PolicyContext:
    """A PolicyContext with platform-default security knobs; override any field by keyword."""
    return PolicyContext(**{**_DEFAULTS, **overrides})
