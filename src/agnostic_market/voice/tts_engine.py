"""Config -> LiveKit TTS plugin (the TTSEngine seam, VOICE_PIPELINE §1/§1a/§1b).

Same pattern as stt_engine.py: LiveKit's `tts.TTS` is the interface; this factory adds
config-driven selection + the credential seam. `cfg.model` is the PINNED immutable
snapshot (§1a) and `cfg.voice_id` the merchant's brand voice — both config, never code.
ElevenLabs (validated second source) lands as one builder row via the bake-off.
"""

from __future__ import annotations

from collections.abc import Callable

from livekit.agents import tts
from livekit.plugins import cartesia

from agnostic_market.dtos.config import TTSConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.voice.credentials import VoiceEngineError, provider_api_key

_BUILDERS: dict[str, Callable[[TTSConfig, str], tts.TTS]] = {
    "cartesia": lambda cfg, api_key: cartesia.TTS(
        model=cfg.model, voice=cfg.voice_id, api_key=api_key
    ),
}


def build_tts(
    cfg: TTSConfig, credentials: ProviderCredentialsConfig, secrets: SecretResolver
) -> tts.TTS:
    """TTS engine for a merchant's `voice.tts` selection. Fails loudly on unknown providers."""
    builder = _BUILDERS.get(cfg.provider)
    if builder is None:
        supported = ", ".join(sorted(_BUILDERS))
        raise VoiceEngineError(
            f"unsupported TTS provider {cfg.provider!r} (supported: {supported}) - "
            f"add a builder row in voice/tts_engine.py"
        )
    return builder(cfg, provider_api_key(credentials, cfg.provider, secrets))
