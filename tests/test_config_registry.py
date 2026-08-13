"""Registry loading — including the F1 path-traversal guard on `extends_template`."""

from __future__ import annotations

from pathlib import Path

import pytest

from agnostic_market.config.loader import ConfigError
from agnostic_market.config.registry import ConfigRegistry

# A minimal but complete override so resolution reaches the template-name check / validation.
# `extends_template` is SINGLE-quoted so backslashes stay literal (double-quoted YAML would
# treat `\.` as an invalid escape and fail at parse time, before the guard we're testing).
_OVERRIDE_BODY = """\
merchant_id: "m1"
extends_template: '{template}'
display_name: "M One"
locale: "en-US"
voice:
  stt: {{ provider: "deepgram", model: "nova-3" }}
  tts: {{ provider: "cartesia", model: "sonic-3.5-2026-05-04", voice_id: "v1" }}
telephony: {{ provider: "telnyx", inbound_number: "+15550000000" }}
prompts: {{ persona_ref: "prompt://m1/persona@sha256-abc" }}
integration:
  order_sor: {{ type: "api", ref: "vault://m1/order", idempotency: "supported" }}
vector_namespace: "m1"
secrets_ref: "vault://m1"
"""

_BASE_BODY = (
    "_safety_locked: [policies.cancel_batch_max, "
    "runtime.cancellation_quiescence_timeout_seconds, "
    'runtime.caller_audible_model_text_max_chars]\nschema_version: "0.2"\n'
    "runtime: { cancellation_quiescence_timeout_seconds: 2.0, "
    "caller_audible_model_text_max_chars: 500 }\n"
    'compliance: { call_start_disclosure: "Hi, this is an AI assistant." }\n'
    "policies:\n"
    "  cancel_batch_max: 10\n"
    "  clarification_reask_max: { identity: 1, support: 2, cart: 2 }\n"
)

_TEMPLATE_BODY = """\
llm:
  routing: { provider: "anthropic", model: "claude-haiku-4-5" }
  reasoning: { provider: "anthropic", model: "claude-opus-4-8" }
policies:
  max_order_value_usd: 1500
  refunds: { auto_approve_under_usd: 50, require_human_above_usd: 200 }
  allow_ai_merchant_handoff: true
integration:
  catalog: { source: "ingested", freshness_sla_min: 15 }
isolation: { tier: "shared" }
"""


def _make_tree(root: Path, *, extends_template: str, template_dir: str = "fashion") -> None:
    (root / "base").mkdir(parents=True)
    (root / "base" / "base.yaml").write_text(_BASE_BODY, encoding="utf-8")
    template_dir_path = root / "templates" / template_dir
    template_dir_path.mkdir(parents=True)
    (template_dir_path / "template.yaml").write_text(_TEMPLATE_BODY, encoding="utf-8")
    (root / "merchants").mkdir(parents=True)
    (root / "merchants" / "m1.yaml").write_text(
        _OVERRIDE_BODY.format(template=extends_template), encoding="utf-8"
    )


def test_valid_template_name_loads(tmp_path: Path) -> None:
    _make_tree(tmp_path, extends_template="fashion")
    registry = ConfigRegistry(tmp_path).load()
    assert registry.get("m1").config.merchant_id == "m1"


@pytest.mark.parametrize(
    "evil",
    [
        "../../etc/secrets",  # parent traversal
        "..\\..\\windows",  # windows-style traversal
        "fashion/../grocery",  # embedded traversal
        "a/b",  # any separator
        "UPPER",  # outside the whitelist charset
    ],
)
def test_path_traversal_template_name_rejected(tmp_path: Path, evil: str) -> None:
    _make_tree(tmp_path, extends_template=evil)
    with pytest.raises(ConfigError, match="invalid extends_template"):
        ConfigRegistry(tmp_path).load()
