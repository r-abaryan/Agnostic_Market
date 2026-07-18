"""MerchantConfig validation: valid fixtures pass; bad types / out-of-bounds / extras fail."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agnostic_market.dtos.config import MerchantConfig, PolicyConfig, SecurityPolicy


def _valid_merchant_dict() -> dict:
    return {
        "schema_version": "0.2",
        "merchant_id": "m1",
        "extends_template": "fashion",
        "display_name": "M One",
        "locale": "en-US",
        "llm": {
            "routing": {"provider": "anthropic", "model": "claude-haiku-4-5"},
            "reasoning": {"provider": "anthropic", "model": "claude-opus-4-8"},
        },
        "voice": {
            "stt": {"provider": "deepgram", "model": "nova-3"},
            "tts": {"provider": "cartesia", "model": "sonic-3.5-2026-05-04", "voice_id": "v1"},
        },
        "telephony": {"provider": "telnyx", "inbound_number": "+15550000000"},
        "compliance": {"call_start_disclosure": "Hi, this is an AI assistant."},
        "policies": {
            "max_order_value_usd": 100,
            "refunds": {"auto_approve_under_usd": 10, "require_human_above_usd": 50},
            "allow_ai_merchant_handoff": True,
        },
        "prompts": {"persona_ref": "prompt://m1/persona@sha256-abc"},
        "integration": {
            "order_sor": {"type": "api", "ref": "vault://m1/order", "idempotency": "supported"},
            "catalog": {"source": "ingested", "freshness_sla_min": 10},
        },
        "isolation": {"tier": "shared"},
        "vector_namespace": "m1",
        "secrets_ref": "vault://m1",
    }


def test_valid_config_validates() -> None:
    config = MerchantConfig.model_validate(_valid_merchant_dict())
    assert config.merchant_id == "m1"
    assert config.isolation.tier == "shared"


def test_returns_policy_defaults_and_floor() -> None:
    # No `returns` block -> platform default window; the DTO floor rejects zero/negative.
    config = MerchantConfig.model_validate(_valid_merchant_dict())
    assert config.policies.returns.window_days == 30
    bad = _valid_merchant_dict()
    bad["policies"]["returns"] = {"window_days": 0}
    with pytest.raises(ValidationError):
        MerchantConfig.model_validate(bad)


def test_to_policy_context_carries_every_enforced_value() -> None:
    # The ONE config->runtime mapping (Group C consolidation): if a field is added to
    # PolicyContext without wiring it here, PolicyContext's no-default constructor makes
    # THIS test fail loudly — the lockstep guard for all production construction sites.
    config = MerchantConfig.model_validate(_valid_merchant_dict())
    context = config.policies.to_policy_context()
    assert context.max_order_value_usd == 100
    assert context.refund_auto_approve_under_usd == 10
    assert context.refund_require_human_above_usd == 50
    assert context.refund_returnless_under_usd == 0.0  # DTO default
    assert context.return_window_days == 30  # DTO default
    assert context.pending_ttl_seconds == 120.0  # platform default TTL
    assert context.spoken_policy_extra is None
    # Security knobs carried through the same one mapping (D1).
    assert context.otp_max_attempts == 2
    assert context.contact_reask_max == 1
    assert context.auth_denials_before_human_offer == 2
    assert context.max_tool_hops == 5


def test_security_policy_defaults() -> None:
    sec = SecurityPolicy()
    assert (sec.otp_max_attempts, sec.contact_reask_max) == (2, 1)
    assert (sec.auth_denials_before_human_offer, sec.max_tool_hops) == (2, 5)


def test_security_policy_rejects_unknown_key_and_bad_floor() -> None:
    with pytest.raises(ValidationError):
        SecurityPolicy(surprise=1)  # extra="forbid"
    with pytest.raises(ValidationError):
        SecurityPolicy(otp_max_attempts=0)  # ge=1 floor


def test_negative_order_cap_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyConfig.model_validate(
            {
                "max_order_value_usd": -1,
                "refunds": {"auto_approve_under_usd": 0, "require_human_above_usd": 0},
                "allow_ai_merchant_handoff": True,
            }
        )


def test_unknown_key_rejected() -> None:
    bad = _valid_merchant_dict()
    bad["surprise"] = "nope"
    with pytest.raises(ValidationError):
        MerchantConfig.model_validate(bad)


def test_bad_isolation_tier_rejected() -> None:
    bad = _valid_merchant_dict()
    bad["isolation"]["tier"] = "gold"
    with pytest.raises(ValidationError):
        MerchantConfig.model_validate(bad)


def test_bool_field_rejects_string_coercion() -> None:
    # StrictBool: a string must NOT be coerced to a bool (model-side footgun guard).
    bad = _valid_merchant_dict()
    bad["policies"]["allow_ai_merchant_handoff"] = "true"
    with pytest.raises(ValidationError):
        MerchantConfig.model_validate(bad)
