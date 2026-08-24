"""Voice plane (Plane 1) — the Phase-2 minimal loop (BUILD_PLAN Phase 2, VOICE_PIPELINE).

- credentials.py : provider whitelist lookup + SecretResolver seam (voice keys).
- stt_engine.py  : config -> LiveKit STT plugin (Deepgram default; add rows via bake-off).
- tts_engine.py  : config -> LiveKit TTS plugin (Cartesia default, pinned snapshot).
- graph.py       : GraphVoiceAdapter — the Plane-1 side of the ReasoningEngine seam.
- pipeline.py    : AgentSession assembly + disclosure + latency logging + thread reaper.

The reasoning graph + engine live in agents/ (frontline + checkout); order data in commerce/.
"""

from agnostic_market.voice.credentials import VoiceEngineError, provider_api_key
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.pipeline import DisclosureFirstAgent, VoiceLoop, build_voice_loop
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tts_engine import build_tts

__all__ = [
    "DisclosureFirstAgent",
    "GraphVoiceAdapter",
    "VoiceEngineError",
    "VoiceLoop",
    "build_stt",
    "build_tts",
    "build_voice_loop",
    "provider_api_key",
]
