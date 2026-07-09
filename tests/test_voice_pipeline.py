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

    class FakeVerificationStore:
        def __init__(self) -> None:
            self.clears = 0

        def clear(self) -> None:
            self.clears += 1

    session, engine, verification = FakeSession(), FakeEngine(), FakeVerificationStore()
    _attach_thread_reaper(session, engine, verification)  # type: ignore[arg-type]
    session.handlers["close"](object())
    session.handlers["close"](object())  # double fire
    assert engine.deletes == 1  # reaped once despite the double fire
    assert verification.clears == 1  # verification grant cleared once (re-entrant-safe)
    lines = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [rec["event"] for rec in lines] == ["flow_abandoned"]
