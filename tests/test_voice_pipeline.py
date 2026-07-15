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

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.voice.graph import GraphVoiceAdapter
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
    # The Agent carries NO instructions — the prompt lives in the frontline graph (F1).
    assert loop.agent.instructions == ""
    # One key per provider actually used, all through the SecretResolver seam.
    assert set(resolver.resolved) == {
        "env://ANTHROPIC_API_KEY",
        "env://DEEPGRAM_API_KEY",
        "env://CARTESIA_API_KEY",
    }
    # Preemptive generation is OFF, explicitly (live call #9 P1): LiveKit's speculative
    # LLM run on the interim transcript would resume a HITL interrupt and fire the
    # place/cancel effect before the turn commits. Asserted on the resolved session
    # options so a LiveKit default change can never silently re-enable it.
    assert loop.session.options.preemptive_generation["enabled"] is False
    # Endpointing max_delay trialled to 1.5s (live call #10): shorter dead pause on
    # low-confidence turns without cutting natural multi-clause speech; min stays 0.3s.
    assert loop.session.options.endpointing["max_delay"] == 1.5
    assert loop.session.options.endpointing["min_delay"] == 0.3  # streaming default kept


async def test_background_audio_has_a_thinking_sound_and_no_ambient(config_root: Path) -> None:
    # The thinking-sound earcon masks LLM/tool dead-air (constructed here; started in the
    # worker entrypoint where the room lives). A call is not a storefront -> no ambient sound.
    # Wiring only: the behavioral check (stops before speech, no readback overlap) is a live
    # audio pass — zero-network tests can't observe playback.
    from livekit.agents.voice.background_audio import AudioConfig

    loop = await _loop(config_root, RecordingResolver())
    thinking = loop.background_audio._thinking_sound
    assert isinstance(thinking, AudioConfig)
    # Points at the repo's subtle-beep asset (a file path, not a LiveKit built-in clip).
    assert isinstance(thinking.source, str) and thinking.source.endswith("thinking_beep.wav")
    assert loop.background_audio._ambient_sound is None


async def test_engine_seam_wiring(config_root: Path) -> None:
    loop = await _loop(config_root, RecordingResolver())
    # LLMAdapter wraps the voice adapter, which wraps the engine (the two-layer seam) —
    # and the session is attached for the §4a fact source.
    adapter = loop.session.llm._graph
    assert isinstance(adapter, GraphVoiceAdapter)
    assert adapter.engine is loop.engine
    assert adapter._session is loop.session
    assert isinstance(loop.engine, ReasoningEngine)
    assert not loop.engine.pending_interrupt()  # fresh thread


def test_thread_reaper_is_reentrant_safe(tmp_path: Path, monkeypatch) -> None:
    # Clock B: a double-fired session close deletes the thread + emits the abandoned
    # event AT MOST once (the checkpointer delete is idempotent; the guard is for OUR
    # telemetry, and for savers whose repeat-call behavior is unverified).
    import json

    from agnostic_market.agents import telemetry
    from agnostic_market.voice.pipeline import _attach_thread_reaper

    monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", tmp_path / "telemetry.jsonl")

    class FakeEngine:
        def __init__(self) -> None:
            self.deletes = 0

        def pending_interrupt(self) -> bool:
            return True  # a live pending confirmation at teardown -> abandoned

        def delete_thread(self) -> None:
            self.deletes += 1

    class FakeSession:
        def __init__(self) -> None:
            self.handlers: dict[str, object] = {}

        def on(self, event: str):
            def _register(fn):
                self.handlers[event] = fn
                return fn

            return _register

    class FakeClearable:
        def __init__(self) -> None:
            self.clears = 0

        def clear(self) -> None:
            self.clears += 1

    session, engine = FakeSession(), FakeEngine()
    verification, cart, pointer = FakeClearable(), FakeClearable(), FakeClearable()
    _attach_thread_reaper(session, engine, verification, cart, pointer)  # type: ignore[arg-type]
    session.handlers["close"](object())
    session.handlers["close"](object())  # double fire
    assert engine.deletes == 1  # reaped once despite the double fire
    assert verification.clears == 1  # verification grant cleared once (re-entrant-safe)
    assert cart.clears == 1  # cart cleared once too
    assert pointer.clears == 1  # the "that order" pointer cleared once too (Group C L4)
    lines = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [rec["event"] for rec in lines] == ["flow_abandoned"]
