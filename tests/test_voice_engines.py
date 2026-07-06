"""STT/TTS factories: config-driven selection, SecretResolver seam, loud whitelist errors."""

from __future__ import annotations

import pytest
from livekit.plugins import cartesia, deepgram
from llm_fakes import RecordingResolver

from agnostic_market.dtos.config import STTConfig, TTSConfig
from agnostic_market.dtos.llm import ProviderCredentialsConfig
from agnostic_market.voice.credentials import VoiceEngineError
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tts_engine import build_tts


def _credentials() -> ProviderCredentialsConfig:
    return ProviderCredentialsConfig.model_validate(
        {
            "providers": {
                "deepgram": {"api_key_ref": "env://DEEPGRAM_API_KEY"},
                "cartesia": {"api_key_ref": "env://CARTESIA_API_KEY"},
            }
        }
    )


def _stt_config(provider: str = "deepgram") -> STTConfig:
    return STTConfig(provider=provider, model="nova-3")


def _tts_config(provider: str = "cartesia") -> TTSConfig:
    return TTSConfig(provider=provider, model="sonic-3.5-2026-05-04", voice_id="v1")


def test_stt_selection_is_config_driven_and_key_flows_through_seam() -> None:
    resolver = RecordingResolver()
    engine = build_stt(_stt_config(), _credentials(), resolver)
    assert isinstance(engine, deepgram.STT)
    assert resolver.resolved == ["env://DEEPGRAM_API_KEY"]


def test_tts_selection_is_config_driven_and_key_flows_through_seam() -> None:
    resolver = RecordingResolver()
    engine = build_tts(_tts_config(), _credentials(), resolver)
    assert isinstance(engine, cartesia.TTS)
    assert resolver.resolved == ["env://CARTESIA_API_KEY"]


def test_unsupported_stt_provider_rejected_loudly() -> None:
    with pytest.raises(VoiceEngineError, match="unsupported STT provider 'assemblyai'"):
        build_stt(_stt_config("assemblyai"), _credentials(), RecordingResolver())


def test_unsupported_tts_provider_rejected_loudly() -> None:
    with pytest.raises(VoiceEngineError, match="unsupported TTS provider 'elevenlabs'"):
        build_tts(_tts_config("elevenlabs"), _credentials(), RecordingResolver())


def test_supported_provider_without_credentials_rejected_loudly() -> None:
    empty = ProviderCredentialsConfig.model_validate({"providers": {}})
    with pytest.raises(VoiceEngineError, match="no credentials configured for voice provider"):
        build_stt(_stt_config(), empty, RecordingResolver())
