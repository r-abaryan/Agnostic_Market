"""MerchantConfig validation: valid fixtures pass; bad types / out-of-bounds / extras fail."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agnostic_market.dtos.config import MerchantConfig, PolicyConfig


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
            "tts": {"provider": "cartesia", "voice_id": "v1"},
        },
        "telephony": {"provider": "telnyx", "inbound_number": "+15550000000"},
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
