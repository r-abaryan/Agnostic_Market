"""LLM-agnostic gateway — a provider:model selection -> a LangChain chat model.

Per-merchant + per-node selection: callers pass `config.llm.routing` or `config.llm.reasoning`
(a `ProviderModel`) — there is no second selection shape. API keys resolve through the
SecretResolver seam from the platform provider-credentials file (config/base/providers.yaml)
and are passed to `init_chat_model` explicitly — never inline in config, and never assumed
from ambient env by this code (the `env://` scheme is just the dev resolver; a Vault resolver
swaps in with zero gateway change, SECURITY §5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.dtos.config import ProviderModel
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.secrets.base import SecretResolver


class GatewayError(RuntimeError):
    """A selection cannot be served (unknown provider / bad credentials config)."""


def load_provider_credentials(path: Path) -> ProviderCredentialsConfig:
    """Load + validate the platform provider-credentials file. Fails loudly."""
    try:
        return ProviderCredentialsConfig.model_validate(load_yaml_layer(path))
    except ValidationError as exc:
        raise GatewayError(f"provider credentials file {path} failed validation:\n{exc}") from exc


class LLMGateway:
    """Build chat models from `ProviderModel` selections, via `init_chat_model`."""

    def __init__(self, credentials: ProviderCredentialsConfig, secrets: SecretResolver) -> None:
        self._credentials = credentials
        self._secrets = secrets

    def chat_model(self, selection: ProviderModel, **model_kwargs: Any) -> BaseChatModel:
        """Chat model for a selection; `model_kwargs` pass through (e.g. `max_retries`).

        Raises GatewayError for a provider with no credentials entry (the credentials file
        is the provider whitelist); SecretResolutionError propagates from the seam.
        """
        entry = self._credentials.providers.get(selection.provider)
        if entry is None:
            known = ", ".join(sorted(self._credentials.providers)) or "(none)"
            raise GatewayError(
                f"no credentials configured for provider {selection.provider!r} "
                f"(configured providers: {known}) - add it to config/base/providers.yaml"
            )
        api_key = self._secrets.resolve(entry.api_key_ref)
        # Transient-error retries (live call #13 F-13.1: a 529 'overloaded' mid-turn died
        # with NO retry): the provider SDK backs off on 429/5xx up to this many attempts.
        # One choke point for every model the platform builds; callers may still override.
        # Retries reduce the failure rate — the engine's spoken turn-fallback is what
        # guarantees a caller never gets silence when they're exhausted.
        model_kwargs.setdefault("max_retries", 3)
        return init_chat_model(
            f"{selection.provider}:{selection.model}", api_key=api_key, **model_kwargs
        )
