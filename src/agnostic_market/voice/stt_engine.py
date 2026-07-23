"""Config -> LiveKit STT plugin (the STTEngine seam, VOICE_PIPELINE §1/§1b).

LiveKit's `stt.STT` IS the engine interface for v1 (adopt built, don't re-wrap — §1b);
this factory adds what it lacks: per-merchant config-driven provider selection and the
SecretResolver credential seam. Adding a provider = one builder row here + one
credentials row in config/base/providers.yaml (AssemblyAI/Flux land via the bake-off).
"""

from __future__ import annotations

from collections.abc import Callable

from livekit.agents import stt
from livekit.plugins import deepgram

from agnostic_market.dtos.config import STTConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.voice.credentials import VoiceEngineError, provider_api_key


def _build_deepgram(cfg: STTConfig, api_key: str) -> stt.STT:
    # keyterm is nova-3's supported bias mechanism (keywords is rejected on nova-3); pass it only
    # when configured so an untuned merchant keeps the provider's prior runtime behavior.
    extra = {"keyterm": list(cfg.keyterms)} if cfg.keyterms else {}
    return deepgram.STT(model=cfg.model, api_key=api_key, numerals=cfg.numerals, **extra)


_BUILDERS: dict[str, Callable[[STTConfig, str], stt.STT]] = {
    "deepgram": _build_deepgram,
}


def build_stt(
    cfg: STTConfig, credentials: ProviderCredentialsConfig, secrets: SecretResolver
) -> stt.STT:
    """STT engine for a merchant's `voice.stt` selection. Fails loudly on unknown providers."""
    builder = _BUILDERS.get(cfg.provider)
    if builder is None:
        supported = ", ".join(sorted(_BUILDERS))
        raise VoiceEngineError(
            f"unsupported STT provider {cfg.provider!r} (supported: {supported}) - "
            f"add a builder row in voice/stt_engine.py"
        )
    return builder(cfg, provider_api_key(credentials, cfg.provider, secrets))
