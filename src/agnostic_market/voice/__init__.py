"""Voice plane (Plane 1) — the Phase-2 minimal loop (BUILD_PLAN Phase 2, VOICE_PIPELINE).

- credentials.py : provider whitelist lookup + SecretResolver seam (voice keys).
- stt_engine.py  : config -> LiveKit STT plugin (Deepgram default; add rows via bake-off).
- tts_engine.py  : config -> LiveKit TTS plugin (Cartesia default, pinned snapshot).
- tools.py       : read-only order_status/catalog_search over a preloaded fixture stub.
- graph.py       : SpeakableTokens — the transport filter between the graph and LLMAdapter.
- pipeline.py    : AgentSession assembly + disclosure + per-turn latency logging.

The reasoning graph itself lives in agents/ (Phase 3a: the frontline agent).
"""

from agnostic_market.voice.credentials import VoiceEngineError, provider_api_key
from agnostic_market.voice.graph import SpeakableTokens
from agnostic_market.voice.pipeline import DisclosureFirstAgent, VoiceLoop, build_voice_loop
from agnostic_market.voice.stt_engine import build_stt
from agnostic_market.voice.tools import OrdersFixture, build_voice_tools, load_orders_fixture
from agnostic_market.voice.tts_engine import build_tts

__all__ = [
    "DisclosureFirstAgent",
    "OrdersFixture",
    "SpeakableTokens",
    "VoiceEngineError",
    "VoiceLoop",
    "build_stt",
    "build_tts",
    "build_voice_loop",
    "build_voice_tools",
    "load_orders_fixture",
    "provider_api_key",
]
