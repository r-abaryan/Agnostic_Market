"""build_voice_loop wiring + the compliance-critical disclosure (zero network).

Builds the WHOLE loop from the real repo config with a dummy-key resolver — the same
assembly the worker runs, minus audio. Pins: disclosure wording/formatting, structural
disclosure-first (on_enter), engine/adapter wiring, and the credential seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from livekit.agents import Agent
from livekit.plugins import cartesia, deepgram
from livekit.plugins import langchain as lk_langchain
from llm_fakes import FakeChatModel, RecordingResolver
from routing_helpers import ArchitectureRoutingRecognizer

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.recovery import NodeExecutionTracker
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentEntry,
    PaymentInstrumentsFixture,
)
from agnostic_market.commerce.profile import ProfileFixture, load_profile_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.pipeline import DisclosureFirstAgent, VoiceLoop, build_voice_loop


async def _loop(config_root: Path, resolver: RecordingResolver) -> VoiceLoop:
    resolved = ConfigRegistry(config_root).load().get("acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    return build_voice_loop(
        resolved,
        credentials,
        resolver,
        config_root=config_root,
        routing_recognizer_factory=lambda _registry: ArchitectureRoutingRecognizer(),
    )


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


async def test_voice_graph_uses_only_response_and_reasoning_model_roles(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.voice import pipeline

    config = ConfigRegistry(config_root).load().get("acme_store").config
    response_model = FakeChatModel()
    reasoning_model = FakeChatModel()
    chat_selections = []
    method_selections = []

    class RecordingGateway:
        def __init__(self, *_args: object) -> None:
            pass

        def chat_model(self, selection):
            chat_selections.append(selection)
            if selection == config.llm.response:
                return response_model
            if selection == config.llm.reasoning:
                return reasoning_model
            raise AssertionError(f"unexpected voice-graph role {selection}")

        def structured_output_method(self, selection):
            method_selections.append(selection)
            if selection != config.llm.response:
                raise AssertionError(f"unexpected structured-output role {selection}")
            return "function_calling"

    monkeypatch.setattr(pipeline, "LLMGateway", RecordingGateway)
    await _loop(config_root, RecordingResolver())

    assert chat_selections == [config.llm.response, config.llm.reasoning]
    assert method_selections == [config.llm.response]


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
    assert loop.engine._cancellation_quiescence_timeout_seconds == 2.0
    assert loop.engine._node_execution_tracker is loop.engine._graph.node_execution_tracker
    assert not loop.engine.pending_interrupt()  # fresh thread


async def test_voice_runtime_shares_the_graph_capability_registry(config_root: Path) -> None:
    loop = await _loop(config_root, RecordingResolver())
    # The dispatcher closes over the registry the compiled graph carries. The runtime must
    # expose THAT instance; a second availability list built alongside it would drift the
    # day a capability is registered, and the ids guard against sharing an empty one.
    assert loop.capability_registry is loop.engine._graph.capability_registry
    assert loop.capability_registry.capability_ids


async def test_session_build_rejects_profile_for_unknown_customer(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agnostic_market.voice import pipeline

    loaded = load_profile_fixture(config_root, "acme_store")
    profile = next(iter(loaded.profiles.values()))
    monkeypatch.setattr(
        pipeline,
        "load_profile_fixture",
        lambda _root, _merchant: ProfileFixture(profiles={"CUST-UNKNOWN": profile}),
    )
    resolved = ConfigRegistry(config_root).load().get("acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    with pytest.raises(ConfigError, match="CUST-UNKNOWN"):
        build_voice_loop(
            resolved,
            credentials,
            RecordingResolver(),
            config_root=config_root,
            routing_recognizer_factory=lambda _registry: ArchitectureRoutingRecognizer(),
        )


async def test_session_build_rejects_payment_instrument_for_unknown_customer(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agnostic_market.voice import pipeline

    monkeypatch.setattr(
        pipeline,
        "load_payment_instruments_fixture",
        lambda _root, _merchant: PaymentInstrumentsFixture(
            payment_instruments={
                "CUST-UNKNOWN": PaymentInstrumentEntry(masked_ref="card ending 1234")
            }
        ),
    )
    resolved = ConfigRegistry(config_root).load().get("acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    with pytest.raises(ConfigError, match="CUST-UNKNOWN"):
        build_voice_loop(
            resolved,
            credentials,
            RecordingResolver(),
            config_root=config_root,
            routing_recognizer_factory=lambda _registry: ArchitectureRoutingRecognizer(),
        )


class _FakeEngine:
    def __init__(self, *, pending: bool = True) -> None:
        self.deletes = 0
        self._pending = pending

    def pending_interrupt(self) -> bool:
        return self._pending

    def delete_thread(self) -> None:
        self.deletes += 1


class _FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event: str):
        def _register(fn):
            self.handlers[event] = fn
            return fn

        return _register


class _FakeClearable:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


class _FakeOrderStore:
    def __init__(self) -> None:
        self.session_placed_clears = 0

    def clear_session_placed(self) -> None:
        self.session_placed_clears += 1


def _fake_caller_context(engine=None):
    from agnostic_market.voice.context import CallerContext

    return CallerContext(
        engine=engine or _FakeEngine(),
        verification_store=_FakeClearable(),  # type: ignore[arg-type]
        cart_store=_FakeClearable(),  # type: ignore[arg-type]
        recent_orders=_FakeClearable(),  # type: ignore[arg-type]
        identity_store=_FakeClearable(),  # type: ignore[arg-type]
        order_store=_FakeOrderStore(),  # type: ignore[arg-type]
    )


def test_close_session_clears_every_caller_store_and_thread() -> None:
    # Milestone A postcondition: close_session tears down ALL caller-ephemeral state (cart,
    # recent context, session-placed orders, verification, identity, and reasoning thread.
    ctx = _fake_caller_context()
    ctx.close_session()
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.recent_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.order_store.session_placed_clears == 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 1  # type: ignore[attr-defined]


def test_close_session_is_idempotent() -> None:
    # A double close (a race, or a reaper firing after a future transition) is harmless.
    ctx = _fake_caller_context()
    ctx.close_session()
    ctx.close_session()
    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


def test_close_stops_turn_admission_and_waits_for_the_whole_turn() -> None:
    ctx = _fake_caller_context()
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker)

    with tracker.turn_span() as admitted:
        assert admitted is True
        ctx.close_session()
        assert tracker.turn_admission_open is False
        with tracker.turn_span() as rejected:
            assert rejected is False
        assert ctx.engine.deletes == 0

    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


def test_close_waits_for_the_cancellation_takeover_lease_without_double_firing() -> None:
    ctx = _fake_caller_context()

    with ctx.cancellation_takeover_lease() as acquired:
        assert acquired is True
        ctx.close_session()
        ctx.close_session()
        assert ctx.engine.deletes == 0
        assert ctx.cart_store.clears == 0  # type: ignore[attr-defined]

    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


def test_cancellation_takeover_lease_is_rejected_after_close_starts() -> None:
    ctx = _fake_caller_context()

    with ctx.cancellation_takeover_lease() as acquired:
        assert acquired is True
        ctx.close_session()
        with ctx.cancellation_takeover_lease() as rejected:
            assert rejected is False

    assert ctx.engine.deletes == 1


def test_thread_reaper_is_reentrant_safe(tmp_path: Path, monkeypatch) -> None:
    # Clock B: a double-fired session close runs the teardown + emits the abandoned event AT
    # MOST once (the reaper's own re-entrant guard; the teardown itself is idempotent too).
    import json

    from agnostic_market.agents import telemetry
    from agnostic_market.voice.pipeline import _attach_thread_reaper

    monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", tmp_path / "telemetry.jsonl")

    session = _FakeSession()
    ctx = _fake_caller_context()
    _attach_thread_reaper(session, ctx)  # type: ignore[arg-type]
    session.handlers["close"](object())
    session.handlers["close"](object())  # double fire
    assert ctx.engine.deletes == 1  # type: ignore[attr-defined]  # reaped once despite double fire
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.recent_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.order_store.session_placed_clears == 1  # type: ignore[attr-defined]
    lines = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # flow_abandoned (reaper, pending interrupt) then caller_context_closed (close_session), once.
    assert [rec["event"] for rec in lines] == ["flow_abandoned", "caller_context_closed"]
