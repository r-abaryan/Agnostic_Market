"""Gateway: SecretResolver pass-through, provider whitelist, credentials-file validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from llm_fakes import RecordingResolver

from agnostic_market.dtos.config import ProviderModel
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.llm.gateway import GatewayError, LLMGateway, load_provider_credentials
from agnostic_market.secrets.base import SecretResolutionError

_REPO_CREDENTIALS = Path(__file__).resolve().parents[1] / "config" / "base" / "providers.yaml"


def _credentials() -> ProviderCredentialsConfig:
    return ProviderCredentialsConfig.model_validate(
        {
            "providers": {
                "anthropic": {
                    "api_key_ref": "env://ANTHROPIC_API_KEY",
                    "structured_output_method": "json_schema",
                },
                "openai": {
                    "api_key_ref": "env://OPENAI_API_KEY",
                    "structured_output_method": "function_calling",
                },
            }
        }
    )


def test_key_resolved_via_seam_and_passed_through() -> None:
    resolver = RecordingResolver()
    gateway = LLMGateway(_credentials(), resolver)

    model = gateway.chat_model(ProviderModel(provider="anthropic", model="claude-haiku-4-5"))
    assert isinstance(model, ChatAnthropic)
    assert resolver.resolved == ["env://ANTHROPIC_API_KEY"]
    assert model.anthropic_api_key.get_secret_value() == "sk-test-dummy"


def test_openai_selection_builds_openai_model() -> None:
    gateway = LLMGateway(_credentials(), RecordingResolver())
    model = gateway.chat_model(ProviderModel(provider="openai", model="gpt-5.4-mini"))
    assert isinstance(model, ChatOpenAI)


def test_model_kwargs_pass_through() -> None:
    gateway = LLMGateway(_credentials(), RecordingResolver())
    model = gateway.chat_model(
        ProviderModel(provider="anthropic", model="claude-haiku-4-5"), max_retries=5
    )
    assert model.max_retries == 5


def test_structured_output_method_comes_from_provider_config() -> None:
    gateway = LLMGateway(_credentials(), RecordingResolver())

    assert (
        gateway.structured_output_method(
            ProviderModel(provider="anthropic", model="claude-haiku-4-5")
        )
        == "json_schema"
    )
    assert (
        gateway.structured_output_method(ProviderModel(provider="openai", model="gpt-5.4-mini"))
        == "function_calling"
    )


def test_missing_structured_output_method_rejected_loudly() -> None:
    credentials = ProviderCredentialsConfig.model_validate(
        {"providers": {"fake": {"api_key_ref": "env://FAKE_API_KEY"}}}
    )
    gateway = LLMGateway(credentials, RecordingResolver())

    with pytest.raises(GatewayError, match="no structured_output_method configured"):
        gateway.structured_output_method(ProviderModel(provider="fake", model="model"))


def test_transient_retries_default_on(  # F-13.1: a live 529 died with no retry
) -> None:
    gateway = LLMGateway(_credentials(), RecordingResolver())
    model = gateway.chat_model(ProviderModel(provider="anthropic", model="claude-haiku-4-5"))
    assert model.max_retries == 3  # the gateway default (callers may still override)


def test_unknown_provider_rejected_loudly() -> None:
    gateway = LLMGateway(_credentials(), RecordingResolver())
    with pytest.raises(GatewayError, match="no credentials configured for provider 'google'"):
        gateway.chat_model(ProviderModel(provider="google", model="gemini"))


def test_secret_resolution_failure_propagates() -> None:
    class FailingResolver:
        def resolve(self, ref: str) -> str:
            raise SecretResolutionError(f"env var missing for {ref!r}")

    gateway = LLMGateway(_credentials(), FailingResolver())
    with pytest.raises(SecretResolutionError):
        gateway.chat_model(ProviderModel(provider="anthropic", model="claude-haiku-4-5"))


def test_repo_credentials_file_loads_and_validates() -> None:
    credentials = load_provider_credentials(_REPO_CREDENTIALS)
    assert set(credentials.providers) == {"anthropic", "openai", "deepgram", "cartesia"}
    for entry in credentials.providers.values():
        assert entry.api_key_ref.startswith("env://")  # refs only — never values
    assert credentials.providers["anthropic"].structured_output_method == "json_schema"
    assert credentials.providers["openai"].structured_output_method == "function_calling"


def test_bad_credentials_shape_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "providers.yaml"
    bad.write_text('providers:\n  anthropic: { api_key: "sk-inline-secret" }\n', encoding="utf-8")
    with pytest.raises(GatewayError, match="failed validation"):
        load_provider_credentials(bad)
