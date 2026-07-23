"""3-layer resolution + the load-bearing safety-lock rejection (the critical Phase-0 test)."""

from __future__ import annotations

import pytest

from agnostic_market.config.resolver import (
    PolicyBoundsViolationError,
    SafetyLockViolationError,
    resolve_merchant_config,
)


def _base() -> dict:
    return {
        "_safety_locked": ["_platform", "schema_version", "policies.cancel_batch_max"],
        "_platform": {
            "payment": {"out_of_band_only": True},
            "limits": {
                "refund_require_human_ceiling_usd": 500,
                "pending_confirmation_ttl_max_seconds": 300,
                "refund_returnless_ceiling_usd": 100,
                "return_window_max_days": 90,
                "otp_max_attempts_ceiling": 3,
                "contact_reask_ceiling": 2,
                "auth_denials_ceiling": 3,
                "tool_hops_ceiling": 8,
                "clarification_reask_ceiling": {
                    "identity": 2,
                    "support": 4,
                    "cart": 4,
                },
            },
        },
        "schema_version": "0.2",
        "compliance": {"call_start_disclosure": "Hi, this is an AI assistant."},
        "policies": {
            "cancel_batch_max": 10,
            "clarification_reask_max": {
                "identity": 1,
                "support": 2,
                "cart": 2,
            },
        },
    }


def _template() -> dict:
    return {
        "llm": {
            "routing": {"provider": "anthropic", "model": "claude-haiku-4-5"},
            "reasoning": {"provider": "anthropic", "model": "claude-opus-4-8"},
        },
        "policies": {
            "max_order_value_usd": 1500,
            "refunds": {"auto_approve_under_usd": 50, "require_human_above_usd": 200},
            "allow_ai_merchant_handoff": True,
        },
        "integration": {"catalog": {"source": "ingested", "freshness_sla_min": 15}},
        "isolation": {"tier": "shared"},
    }


def _override() -> dict:
    return {
        "merchant_id": "m1",
        "extends_template": "fashion",
        "display_name": "M One",
        "locale": "en-US",
        "voice": {
            "stt": {"provider": "deepgram", "model": "nova-3"},
            "tts": {"provider": "cartesia", "model": "sonic-3.5-2026-05-04", "voice_id": "v1"},
        },
        "telephony": {"provider": "telnyx", "inbound_number": "+15550000000"},
        # partial override merges (deep-merge keeps template's refund defaults)
        "policies": {"max_order_value_usd": 2000},
        "prompts": {"persona_ref": "prompt://m1/persona@sha256-abc"},
        "integration": {
            "order_sor": {"type": "api", "ref": "vault://m1/order", "idempotency": "supported"},
        },
        "vector_namespace": "m1",
        "secrets_ref": "vault://m1",
    }


def test_three_layer_merge_is_last_wins_and_deep() -> None:
    config = resolve_merchant_config(_base(), _template(), _override())
    # override wins on max_order_value_usd ...
    assert config.policies.max_order_value_usd == 2000
    # ... but the template's refund defaults survive (deep merge, not whole-key replace)
    assert config.policies.refunds.auto_approve_under_usd == 50
    # template-only value carried through
    assert config.integration.catalog.freshness_sla_min == 15


def test_override_touching_safety_locked_key_is_rejected() -> None:
    bad = _override()
    bad["_platform"] = {"payment": {"out_of_band_only": False}}  # attempt to weaken PCI posture
    with pytest.raises(SafetyLockViolationError, match="_platform"):
        resolve_merchant_config(_base(), _template(), bad)


def test_platform_locked_even_if_base_forgets_to_declare_it() -> None:
    # F2: the lock floor is in code. A base that does NOT list `_platform` in _safety_locked
    # must STILL reject an override touching `_platform` (authority in code, not the fixture).
    base = {"_safety_locked": ["schema_version"], "schema_version": "0.2"}  # no _platform listed
    bad = _override()
    bad["_platform"] = {"payment": {"out_of_band_only": False}}
    with pytest.raises(SafetyLockViolationError, match="_platform"):
        resolve_merchant_config(base, _template(), bad)


def test_override_cannot_ship_its_own_safety_lock_declaration() -> None:
    # F2: `_safety_locked` is itself locked — an override shipping one is tampering, rejected
    # (not silently ignored).
    bad = _override()
    bad["_safety_locked"] = []  # attempt to blank the lock set
    with pytest.raises(SafetyLockViolationError, match="_safety_locked"):
        resolve_merchant_config(_base(), _template(), bad)


def test_template_cannot_touch_locked_key_either() -> None:
    bad_template = _template()
    bad_template["schema_version"] = "9.9"  # locked structural key
    with pytest.raises(SafetyLockViolationError, match="schema_version"):
        resolve_merchant_config(_base(), bad_template, _override())


def test_cancel_batch_max_is_safety_locked_from_merchant_override() -> None:
    # F-16.2: the cancel batch cap is a SAFETY bound (fits the LangGraph step budget), not a
    # merchant knob — a base that declares it locked rejects an override that raises it.
    base = _base()
    bad = _override()
    bad["policies"] = {"cancel_batch_max": 999}  # attempt to blow the step-budget bound open
    with pytest.raises(SafetyLockViolationError, match="cancel_batch_max"):
        resolve_merchant_config(base, _template(), bad)


def test_cancel_batch_max_comes_from_locked_base() -> None:
    config = resolve_merchant_config(_base(), _template(), _override())
    assert config.policies.cancel_batch_max == 10


def test_platform_block_not_leaked_into_effective_config() -> None:
    config = resolve_merchant_config(_base(), _template(), _override())
    # `_platform` / `_safety_locked` are stripped before validation; MerchantConfig forbids extras.
    assert not hasattr(config, "_platform")


# --- policy-within-bounds: the "tune within limits, can't disable the guard" half ----------


def test_refund_human_threshold_over_platform_ceiling_is_rejected() -> None:
    bad = _override()
    bad["policies"] = {"refunds": {"require_human_above_usd": 999}}  # ceiling is 500
    with pytest.raises(PolicyBoundsViolationError, match="require_human_above_usd"):
        resolve_merchant_config(_base(), _template(), bad)


def test_refund_human_threshold_at_ceiling_is_allowed() -> None:
    ok = _override()
    ok["policies"] = {"refunds": {"require_human_above_usd": 500}}  # == ceiling, not above
    config = resolve_merchant_config(_base(), _template(), ok)
    assert config.policies.refunds.require_human_above_usd == 500


def test_auto_approve_above_human_line_is_rejected() -> None:
    bad = _override()
    # auto 300 > human 200 (template default) — incoherent tiering
    bad["policies"] = {"refunds": {"auto_approve_under_usd": 300}}
    with pytest.raises(PolicyBoundsViolationError, match="auto_approve_under_usd"):
        resolve_merchant_config(_base(), _template(), bad)


def test_pending_ttl_over_platform_max_is_rejected() -> None:
    bad = _override()
    bad["policies"] = {"pending_confirmation_ttl_seconds": 9999}  # max is 300
    with pytest.raises(PolicyBoundsViolationError, match="pending_confirmation_ttl_seconds"):
        resolve_merchant_config(_base(), _template(), bad)


def test_returnless_window_over_platform_ceiling_is_rejected() -> None:
    # A merchant may widen the returnless-refund window only up to the platform bound —
    # above it, a shipped refund is always return-first (the caller must not be able to
    # keep goods AND money at arbitrary value).
    bad = _override()
    bad["policies"] = {"refunds": {"returnless_under_usd": 250}}  # ceiling is 100
    with pytest.raises(PolicyBoundsViolationError, match="returnless_under_usd"):
        resolve_merchant_config(_base(), _template(), bad)


def test_returnless_window_within_ceiling_is_accepted() -> None:
    ok = _override()
    ok["policies"] = {"refunds": {"returnless_under_usd": 75}}  # <= 100
    config = resolve_merchant_config(_base(), _template(), ok)
    assert config.policies.refunds.returnless_under_usd == 75


def test_return_window_over_platform_ceiling_is_rejected() -> None:
    bad = _override()
    bad["policies"] = {"returns": {"window_days": 365}}  # ceiling is 90
    with pytest.raises(PolicyBoundsViolationError, match="window_days"):
        resolve_merchant_config(_base(), _template(), bad)


def test_return_window_defaults_when_merchant_sets_none() -> None:
    # No merchant/template `returns` block -> the platform default (30), still bounded.
    config = resolve_merchant_config(_base(), _template(), _override())
    assert config.policies.returns.window_days == 30


def test_pending_ttl_within_max_resolves() -> None:
    ok = _override()
    ok["policies"] = {"pending_confirmation_ttl_seconds": 90}
    config = resolve_merchant_config(_base(), _template(), ok)
    assert config.policies.pending_confirmation_ttl_seconds == 90


def test_pending_ttl_defaults_when_unset() -> None:
    # No merchant sets it -> the platform default (from the DTO), still within the max.
    config = resolve_merchant_config(_base(), _template(), _override())
    assert config.policies.pending_confirmation_ttl_seconds == 120.0


# --- security attempt-budget ceilings (D1) — a LARGER value weakens; the ceiling caps it ---


@pytest.mark.parametrize(
    ("field", "over_value"),
    [
        ("otp_max_attempts", 4),  # ceiling 3
        ("contact_reask_max", 3),  # ceiling 2
        ("auth_denials_before_human_offer", 4),  # ceiling 3
        ("max_tool_hops", 9),  # ceiling 8
    ],
)
def test_security_knob_over_ceiling_is_rejected(field: str, over_value: int) -> None:
    bad = _override()
    bad["policies"] = {"security": {field: over_value}}
    with pytest.raises(PolicyBoundsViolationError, match=field):
        resolve_merchant_config(_base(), _template(), bad)


def test_security_knobs_at_ceiling_are_allowed() -> None:
    ok = _override()
    ok["policies"] = {
        "security": {
            "otp_max_attempts": 3,
            "contact_reask_max": 2,
            "auth_denials_before_human_offer": 3,
            "max_tool_hops": 8,
        }
    }
    config = resolve_merchant_config(_base(), _template(), ok)
    assert config.policies.security.otp_max_attempts == 3
    assert config.policies.security.max_tool_hops == 8


@pytest.mark.parametrize(
    ("flow", "over_value"),
    [
        ("identity", 3),
        ("support", 5),
        ("cart", 5),
    ],
)
def test_clarification_reask_over_flow_ceiling_is_rejected(flow: str, over_value: int) -> None:
    bad = _override()
    bad["policies"] = {
        "clarification_reask_max": {
            flow: over_value,
        }
    }
    with pytest.raises(PolicyBoundsViolationError, match=flow):
        resolve_merchant_config(_base(), _template(), bad)


def test_security_knobs_default_when_unset() -> None:
    # No merchant/template `security` block -> the platform defaults (2/1/2/5), still bounded.
    config = resolve_merchant_config(_base(), _template(), _override())
    sec = config.policies.security
    assert (sec.otp_max_attempts, sec.contact_reask_max) == (2, 1)
    assert (sec.auth_denials_before_human_offer, sec.max_tool_hops) == (2, 5)
    assert config.policies.clarification_reask_max.model_dump() == {
        "identity": 1,
        "support": 2,
        "cart": 2,
    }
