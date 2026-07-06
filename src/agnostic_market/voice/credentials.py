"""Voice-provider credential lookup — providers.yaml whitelist + SecretResolver seam.

Mirrors the LLM gateway's rule (SECURITY §5): keys resolve from `api_key_ref` at build
time and are passed to the plugin explicitly — never inline in config, never assumed
from ambient env by this code. The credentials file doubles as the provider whitelist.
"""

from __future__ import annotations

from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.secrets.base import SecretResolver


class VoiceEngineError(RuntimeError):
    """A voice.stt / voice.tts selection cannot be served (unknown provider / no credentials)."""


def provider_api_key(
    credentials: ProviderCredentialsConfig, provider: str, secrets: SecretResolver
) -> str:
    """Resolve the API key for a whitelisted provider; loud error otherwise."""
    entry = credentials.providers.get(provider)
    if entry is None:
        known = ", ".join(sorted(credentials.providers)) or "(none)"
        raise VoiceEngineError(
            f"no credentials configured for voice provider {provider!r} "
            f"(configured providers: {known}) - add it to config/base/providers.yaml"
        )
    return secrets.resolve(entry.api_key_ref)
