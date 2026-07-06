"""build_voice_loop wiring + the compliance-critical disclosure (zero network).

Builds the WHOLE loop from the real repo config with a dummy-key resolver — the same
assembly the worker runs, minus audio. Pins: disclosure wording/formatting, structural
disclosure-first (on_enter), engine/adapter wiring, and the credential seam.
"""

from __future__ import annotations

from pathlib import Path

from livekit.agents import Agent
from livekit.plugins import cartesia, deepgram
from livekit.plugins import langchain as lk_langchain
from llm_fakes import RecordingResolver

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.voice.pipeline import DisclosureFirstAgent, VoiceLoop, build_voice_loop


async def _loop(config_root: Path, resolver: RecordingResolver) -> VoiceLoop:
    resolved = ConfigRegistry(config_root).load().get("acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    return build_voice_loop(resolved, credentials, resolver, config_root=config_root)


async def test_disclosure_is_formatted_and_bound_to_on_enter(config_root: Path) -> None:
    loop = await _loop(config_root, RecordingResolver())
    # Exact wording: the base-layer script with the merchant's display name substituted.
    assert loop.agent.disclosure == (
        "Hi, you've reached Acme Store. I'm an automated AI assistant. How can I help?"
    )
    assert "{display_name}" not in loop.agent.disclosure
    # Structural disclosure-first: the agent itself owns it via an on_enter override
    # (not the worker racing to call say() after session.start()).
    assert isinstance(loop.agent, DisclosureFirstAgent)
    assert DisclosureFirstAgent.on_enter is not Agent.on_enter


async def test_session_wiring_is_config_driven(config_root: Path) -> None:
    resolver = RecordingResolver()
    loop = await _loop(config_root, resolver)
    assert isinstance(loop.session.stt, deepgram.STT)
    assert isinstance(loop.session.tts, cartesia.TTS)
    assert isinstance(loop.session.llm, lk_langchain.LLMAdapter)
    assert "Acme Store" in loop.agent.instructions
    # One key per provider actually used, all through the SecretResolver seam.
    assert set(resolver.resolved) == {
        "env://ANTHROPIC_API_KEY",
        "env://DEEPGRAM_API_KEY",
        "env://CARTESIA_API_KEY",
    }
