"""3-layer resolution + the load-bearing safety-lock rejection (the critical Phase-0 test)."""

from __future__ import annotations

import pytest

from agnostic_market.config.resolver import (
    SafetyLockViolationError,
    resolve_merchant_config,
)


def _base() -> dict:
    return {
        "_safety_locked": ["_platform", "schema_version"],
        "_platform": {"payment": {"out_of_band_only": True}},
        "schema_version": "0.2",
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
            "tts": {"provider": "cartesia", "voice_id": "v1"},
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


def test_override_touching_locked_descendant_is_rejected() -> None:
    bad = _override()
    bad["_platform"] = {"payment": {"out_of_band_only": False}}
    with pytest.raises(SafetyLockViolationError):
        resolve_merchant_config(_base(), _template(), bad)


def test_template_cannot_touch_locked_key_either() -> None:
    bad_template = _template()
    bad_template["schema_version"] = "9.9"  # locked structural key
    with pytest.raises(SafetyLockViolationError, match="schema_version"):
        resolve_merchant_config(_base(), bad_template, _override())


def test_platform_block_not_leaked_into_effective_config() -> None:
    config = resolve_merchant_config(_base(), _template(), _override())
    # `_platform` / `_safety_locked` are stripped before validation; MerchantConfig forbids extras.
    assert not hasattr(config, "_platform")
