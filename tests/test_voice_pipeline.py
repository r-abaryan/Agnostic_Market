"""build_voice_loop wiring + the compliance-critical disclosure (zero network).

Builds the WHOLE loop from the real repo config with a dummy-key resolver — the same
assembly the worker runs, minus audio. Pins: disclosure wording/formatting, structural
disclosure-first (on_enter), engine/adapter wiring, and the credential seam.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit.agents import Agent
from livekit.plugins import cartesia, deepgram
from livekit.plugins import langchain as lk_langchain
from llm_fakes import FakeChatModel, RecordingResolver
from routing_helpers import ArchitectureRoutingRecognizer
from telemetry_helpers import make_session_telemetry, make_tenant_telemetry
from turn_helpers import engine_events

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.recovery import NodeExecutionTracker
from agnostic_market.agents.routing_activation import RoutingRecognizerFactory
from agnostic_market.agents.telemetry import InMemoryTelemetrySink
from agnostic_market.application import (
    PreparedApplicationRouting,
    build_fixture_tenant_services,
    prepare_application_routing,
)
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentEntry,
    PaymentInstrumentsFixture,
)
from agnostic_market.commerce.profile import ProfileFixture, load_profile_fixture
from agnostic_market.config.loader import ConfigError
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.durability.session_registry import (
    SessionStateWriteError,
    SessionStateWriteReason,
)
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.tenancy.context import build_tenant_context
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.pipeline import (
    DisclosureFirstAgent,
    TurnLatencyMeasurement,
    VoiceLoop,
    _attach_turn_metrics_logger,
    build_voice_loop,
    retire_voice_transport,
)


async def _loop(
    config_root: Path,
    resolver: RecordingResolver,
    routing: RoutingRecognizerFactory | PreparedApplicationRouting | None = None,
) -> VoiceLoop:
    registry = ConfigRegistry(config_root).load()
    resolved = registry.get("acme_store")
    tenant = build_tenant_context(registry, "acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    return await build_voice_loop(
        tenant,
        resolved,
        credentials,
        resolver,
        deployment_id="test-voice-artifact",
        tenant_services=build_fixture_tenant_services(
            config_root,
            tenant,
            telemetry=make_tenant_telemetry("acme_store"),
        ),
        routing_recognizer_factory=(
            routing if routing is not None else lambda _registry: ArchitectureRoutingRecognizer()
        ),
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


def test_latency_observer_separates_endpointing_from_assistant_processing() -> None:
    callbacks = {}
    observed: list[TurnLatencyMeasurement] = []

    class Session:
        def on(self, event_name: str):
            def register(callback):
                callbacks[event_name] = callback
                return callback

            return register

    _attach_turn_metrics_logger(Session(), observed.append)
    callback = callbacks["conversation_item_added"]

    callback(
        SimpleNamespace(
            item=SimpleNamespace(
                role="user",
                metrics={"end_of_turn_delay": 0.25, "transcription_delay": 0.05},
            )
        )
    )
    callback(
        SimpleNamespace(
            item=SimpleNamespace(
                role="assistant",
                metrics={"e2e_latency": 0.625, "llm_node_ttft": 0.2},
            )
        )
    )
    assert observed == [
        TurnLatencyMeasurement(
            end_to_end_seconds=0.625,
            endpointing_seconds=0.25,
            processing_seconds=0.375,
        )
    ]


def test_latency_observer_rejects_uncorrelated_or_contradictory_metrics() -> None:
    callbacks = {}
    observed: list[TurnLatencyMeasurement] = []

    class Session:
        def on(self, event_name: str):
            def register(callback):
                callbacks[event_name] = callback
                return callback

            return register

    _attach_turn_metrics_logger(Session(), observed.append)
    callback = callbacks["conversation_item_added"]

    callback(SimpleNamespace(item=SimpleNamespace(role="assistant", metrics={"e2e_latency": 1.0})))
    callback(SimpleNamespace(item=SimpleNamespace(role="user", metrics={})))
    callback(SimpleNamespace(item=SimpleNamespace(role="assistant", metrics={"e2e_latency": 1.0})))
    callback(SimpleNamespace(item=SimpleNamespace(role="user", metrics={"end_of_turn_delay": 0.7})))
    callback(SimpleNamespace(item=SimpleNamespace(role="user", metrics={"end_of_turn_delay": 0.2})))
    callback(SimpleNamespace(item=SimpleNamespace(role="assistant", metrics={"e2e_latency": 1.0})))
    callback(SimpleNamespace(item=SimpleNamespace(role="user", metrics={"end_of_turn_delay": 1.1})))
    callback(SimpleNamespace(item=SimpleNamespace(role="assistant", metrics={"e2e_latency": 1.0})))

    assert observed == []


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


async def test_voice_composition_rejects_a_mismatched_tenant_before_model_construction(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.voice import pipeline

    registry = ConfigRegistry(config_root).load()
    resolved = registry.get("acme_store")
    tenant = build_tenant_context(registry, "acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")

    def model_construction_started(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model construction started before tenant validation")

    monkeypatch.setattr(pipeline, "LLMGateway", model_construction_started)

    with pytest.raises(ValueError, match="tenant context"):
        await build_voice_loop(
            replace(tenant, tenant_id="demo_shop"),
            resolved,
            credentials,
            RecordingResolver(),
            deployment_id="test-voice-artifact",
            tenant_services=build_fixture_tenant_services(
                config_root,
                tenant,
                telemetry=make_tenant_telemetry("acme_store"),
            ),
            routing_recognizer_factory=lambda _registry: ArchitectureRoutingRecognizer(),
        )


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
    assert loop.application.engine is loop.engine
    assert adapter._session is loop.session
    assert isinstance(loop.engine, ReasoningEngine)
    assert loop.engine._cancellation_quiescence_timeout_seconds == 2.0
    assert loop.engine._node_execution_tracker is loop.engine._graph.node_execution_tracker
    assert not await loop.engine.apending_interrupt()  # fresh thread


async def test_voice_runtime_shares_the_graph_capability_registry(config_root: Path) -> None:
    prepared = prepare_application_routing(lambda _registry: ArchitectureRoutingRecognizer())
    loop = await _loop(config_root, RecordingResolver(), prepared)
    # The dispatcher closes over the registry the compiled graph carries. The runtime must
    # expose THAT instance; a second availability list built alongside it would drift the
    # day a capability is registered, and the ids guard against sharing an empty one.
    assert loop.capability_registry is loop.engine._graph.capability_registry
    assert loop.capability_registry is prepared.capability_registry
    assert loop.application.assembly.graph is loop.engine._graph
    assert loop.capability_registry.capability_ids


async def test_session_build_rejects_profile_for_unknown_customer(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agnostic_market import application

    loaded = load_profile_fixture(config_root, "acme_store")
    profile = next(iter(loaded.profiles.values()))
    monkeypatch.setattr(
        application,
        "load_profile_fixture",
        lambda _root, _merchant: ProfileFixture(profiles={"CUST-UNKNOWN": profile}),
    )
    with pytest.raises(ConfigError, match="CUST-UNKNOWN"):
        build_fixture_tenant_services(
            config_root,
            build_tenant_context(ConfigRegistry(config_root).load(), "acme_store"),
            telemetry=make_tenant_telemetry("acme_store"),
        )


async def test_session_build_rejects_payment_instrument_for_unknown_customer(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agnostic_market import application

    monkeypatch.setattr(
        application,
        "load_payment_instruments_fixture",
        lambda _root, _merchant: PaymentInstrumentsFixture(
            payment_instruments={
                "CUST-UNKNOWN": PaymentInstrumentEntry(masked_ref="card ending 1234")
            }
        ),
    )
    with pytest.raises(ConfigError, match="CUST-UNKNOWN"):
        build_fixture_tenant_services(
            config_root,
            build_tenant_context(ConfigRegistry(config_root).load(), "acme_store"),
            telemetry=make_tenant_telemetry("acme_store"),
        )


class _FakeEngine:
    def __init__(self, *, pending: bool = True) -> None:
        self.deletes = 0
        self._pending = pending

    async def apending_interrupt(self) -> bool:
        return self._pending

    async def acheckpoint_has_pending_interrupt(self) -> bool:
        return self._pending

    async def adelete_thread(self) -> None:
        self.deletes += 1


class _FailOnceDeleteEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.delete_entered = asyncio.Event()
        self.release_delete = asyncio.Event()

    async def adelete_thread(self) -> None:
        self.deletes += 1
        if self.deletes == 1:
            self.delete_entered.set()
            await self.release_delete.wait()
            raise TimeoutError("injected checkpoint deletion timeout")


class _FailPendingProbeEngine(_FakeEngine):
    async def acheckpoint_has_pending_interrupt(self) -> bool:
        raise TimeoutError("injected pending-interrupt observation timeout")


class _DelayedDeleteEngine(_FakeEngine):
    def __init__(self) -> None:
        super().__init__(pending=False)
        self.delete_entered = asyncio.Event()
        self.release_delete = asyncio.Event()

    async def adelete_thread(self) -> None:
        self.deletes += 1
        self.delete_entered.set()
        await self.release_delete.wait()


class _FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event: str):
        def _register(fn):
            self.handlers[event] = fn
            return fn

        return _register


class _FakeJobContext:
    def __init__(self) -> None:
        self.shutdown_callbacks = []

    def add_shutdown_callback(self, callback) -> None:
        self.shutdown_callbacks.append(callback)


class _FailingLeaseSupervisor:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("injected lease-supervisor close failure")


class _RetirementSession:
    def __init__(self, tracker: NodeExecutionTracker, *, fail_shutdown: bool = False) -> None:
        self.tracker = tracker
        self.fail_shutdown = fail_shutdown
        self.shutdown_calls: list[bool] = []

    def shutdown(self, *, drain: bool = True) -> None:
        assert self.tracker.turn_admission_open is False
        self.shutdown_calls.append(drain)
        if self.fail_shutdown:
            raise RuntimeError("injected output shutdown failure")


class _RetirementRoom:
    def __init__(self, tracker: NodeExecutionTracker, *, block: bool = False) -> None:
        self.tracker = tracker
        self.block = block
        self.disconnect_calls = 0

    async def disconnect(self) -> None:
        assert self.tracker.turn_admission_open is False
        self.disconnect_calls += 1
        if self.block:
            await asyncio.Event().wait()


class _FakeClearable:
    def __init__(self) -> None:
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1


class _FakeAsyncClearable:
    def __init__(self) -> None:
        self.clears = 0

    async def clear(self) -> None:
        self.clears += 1


class _FakeSessionState:
    def __init__(
        self,
        *,
        clear_error: Exception | None = None,
        discard_error: Exception | None = None,
    ) -> None:
        self.cart = _FakeClearable()
        self.recent_orders = _FakeClearable()
        self.guest_orders = _FakeClearable()
        self.clear_error = clear_error
        self.discard_error = discard_error

    async def discard_local_projection(self) -> None:
        if self.discard_error is not None:
            raise self.discard_error
        self.cart.clear()
        self.recent_orders.clear()
        self.guest_orders.clear()

    async def clear_ephemeral(self, _operation_id: str):
        if self.clear_error is not None:
            raise self.clear_error
        self.cart.clear()
        self.recent_orders.clear()
        self.guest_orders.clear()
        return SimpleNamespace(session_revision=1)


class _BlockingDiscardSessionState(_FakeSessionState):
    def __init__(self) -> None:
        super().__init__()
        self.discard_entered = asyncio.Event()

    async def discard_local_projection(self) -> None:
        self.discard_entered.set()
        await asyncio.Event().wait()


class _FakeDurableCloser:
    def __init__(
        self,
        events: list[str],
        *,
        finalize_error: Exception | None = None,
        pending_interrupt: bool = False,
    ) -> None:
        self.events = events
        self.finalize_error = finalize_error
        self.pending_interrupt = pending_interrupt

    async def begin_close(self) -> None:
        self.events.append("begin")

    async def finalize_close(self) -> None:
        self.events.append("finalize")
        if self.finalize_error is not None:
            raise self.finalize_error

    async def checkpoint_has_pending_interrupt(self) -> bool:
        return self.pending_interrupt


class _BlockingDurableCloser:
    def __init__(self, blocked_stage: str) -> None:
        self.blocked_stage = blocked_stage
        self.entered = asyncio.Event()

    async def begin_close(self) -> None:
        if self.blocked_stage == "begin":
            self.entered.set()
            await asyncio.Event().wait()

    async def finalize_close(self) -> None:
        if self.blocked_stage == "finalize":
            self.entered.set()
            await asyncio.Event().wait()

    async def checkpoint_has_pending_interrupt(self) -> bool:
        return False


def _fake_caller_context(engine=None, *, session_state=None, durable_closer=None):
    from agnostic_market.session import CallerContext

    telemetry = make_session_telemetry("acme_store", "fake-caller")
    session_state = session_state or _FakeSessionState()
    return CallerContext(
        engine=engine or _FakeEngine(),
        verification_store=_FakeAsyncClearable(),  # type: ignore[arg-type]
        session_state=session_state,  # type: ignore[arg-type]
        identity_store=_FakeClearable(),  # type: ignore[arg-type]
        telemetry=telemetry.operational,
        durable_closer=durable_closer,
    )


async def test_close_session_clears_every_caller_store_and_thread() -> None:
    # Milestone A postcondition: aclose_session tears down all caller-ephemeral state (cart,
    # recent context, guest-order scope, verification, identity, and reasoning thread.
    ctx = _fake_caller_context()
    await ctx.aclose_session()
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.recent_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.guest_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 1  # type: ignore[attr-defined]


async def test_durable_close_fences_before_quiescence_and_never_uses_live_cleanup() -> None:
    events: list[str] = []
    closer = _FakeDurableCloser(events)
    state = _FakeSessionState(clear_error=AssertionError("durable close must not publish"))
    ctx = _fake_caller_context(session_state=state, durable_closer=closer)
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=1.0)

    with tracker.turn_span() as admitted:
        assert admitted is True
        closing = asyncio.create_task(ctx.aclose_session())
        await asyncio.sleep(0)
        assert events == ["begin"]
        assert ctx.cart_store.clears == 0  # type: ignore[attr-defined]

    await closing

    assert events == ["begin", "finalize"]
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 0  # type: ignore[attr-defined]
    assert ctx._closed is True


async def test_durable_close_observes_an_interrupt_after_fencing_and_quiescence() -> None:
    events: list[str] = []
    closer = _FakeDurableCloser(events)
    ctx = _fake_caller_context(durable_closer=closer)
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=1.0)

    with tracker.turn_span() as admitted:
        assert admitted is True
        closing = asyncio.create_task(ctx.aclose_session())
        await asyncio.sleep(0)
        assert events == ["begin"]
        closer.pending_interrupt = True

    await closing

    assert ctx.close_had_pending_interrupt is True


async def test_durable_close_failure_still_destroys_local_authority() -> None:
    events: list[str] = []
    closer = _FakeDurableCloser(events, finalize_error=TimeoutError("lost close acknowledgement"))
    ctx = _fake_caller_context(durable_closer=closer)

    with pytest.raises(TimeoutError, match="lost close acknowledgement"):
        await ctx.aclose_session()

    assert events == ["begin", "finalize"]
    assert ctx.cart_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 0  # type: ignore[attr-defined]
    assert ctx._closed is False


async def test_local_disposal_failure_does_not_skip_durable_finalization() -> None:
    events: list[str] = []
    closer = _FakeDurableCloser(events)
    state = _FakeSessionState(discard_error=RuntimeError("injected local disposal failure"))
    ctx = _fake_caller_context(session_state=state, durable_closer=closer)

    with pytest.raises(RuntimeError, match="local disposal failure"):
        await ctx.aclose_session()

    assert events == ["begin", "finalize"]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx._closed is False


@pytest.mark.parametrize("blocked_stage", ["begin", "finalize"])
async def test_durable_close_cancellation_discards_local_authority(
    blocked_stage: str,
) -> None:
    closer = _BlockingDurableCloser(blocked_stage)
    ctx = _fake_caller_context(durable_closer=closer)
    closing = asyncio.create_task(ctx.aclose_session())
    await closer.entered.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert ctx.cart_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears >= 1  # type: ignore[attr-defined]
    assert ctx._closed is False


async def test_durable_close_cancellation_remains_direct_when_local_disposal_fails() -> None:
    closer = _BlockingDurableCloser("begin")
    disposal_error = RuntimeError("injected local disposal failure")
    ctx = _fake_caller_context(
        session_state=_FakeSessionState(discard_error=disposal_error),
        durable_closer=closer,
    )
    closing = asyncio.create_task(ctx.aclose_session())
    await closer.entered.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await closing

    assert cancelled.value.__cause__ is disposal_error


async def test_local_disposal_cancellation_remains_direct_when_durable_finalize_fails() -> None:
    finalize_error = RuntimeError("injected durable finalization failure")
    state = _BlockingDiscardSessionState()
    ctx = _fake_caller_context(
        session_state=state,
        durable_closer=_FakeDurableCloser([], finalize_error=finalize_error),
    )
    closing = asyncio.create_task(ctx.aclose_session())
    await state.discard_entered.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await closing

    assert cancelled.value.__cause__ is finalize_error


async def test_close_destroys_local_authority_when_durable_clear_is_rejected() -> None:
    session_state = _FakeSessionState(
        clear_error=SessionStateWriteError(SessionStateWriteReason.STALE_REVISION)
    )
    ctx = _fake_caller_context(session_state=session_state)

    with pytest.raises(SessionStateWriteError) as rejected:
        await ctx.aclose_session()

    assert rejected.value.reason is SessionStateWriteReason.STALE_REVISION
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 0  # type: ignore[attr-defined]
    assert ctx._closed is False


async def test_abandonment_never_writes_or_deletes_through_lost_authority() -> None:
    session_state = _FakeSessionState(clear_error=AssertionError("lost authority must not publish"))
    ctx = _fake_caller_context(session_state=session_state)

    await ctx.aabandon_session()

    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.recent_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.guest_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 0  # type: ignore[attr-defined]
    assert ctx._closed is True


async def test_close_hook_observes_authority_loss_before_transport_shutdown() -> None:
    session_state = _FakeSessionState(clear_error=AssertionError("lost authority must not publish"))
    ctx = _fake_caller_context(session_state=session_state)

    ctx.stop_for_authority_loss()
    await ctx.aclose_session()

    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.engine.deletes == 0  # type: ignore[attr-defined]
    assert ctx._closed is True


async def test_close_session_is_idempotent() -> None:
    # A double close (a race, or a reaper firing after a future transition) is harmless.
    ctx = _fake_caller_context()
    await asyncio.gather(ctx.aclose_session(), ctx.aclose_session())
    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


async def test_pending_interrupt_observation_failure_does_not_skip_teardown() -> None:
    engine = _FailPendingProbeEngine()
    ctx = _fake_caller_context(engine)

    await ctx.aclose_session()

    assert engine.deletes == 1
    assert ctx._closed is True
    assert ctx.close_had_pending_interrupt is False


async def test_concurrent_close_retries_after_checkpoint_delete_failure() -> None:
    engine = _FailOnceDeleteEngine()
    ctx = _fake_caller_context(engine)

    first = asyncio.create_task(ctx.aclose_session())
    await engine.delete_entered.wait()
    second = asyncio.create_task(ctx.aclose_session())
    await asyncio.sleep(0)
    engine.release_delete.set()

    with pytest.raises(TimeoutError, match="checkpoint deletion timeout"):
        await first
    await second

    assert engine.deletes == 2
    assert ctx._closed is True
    assert ctx.cart_store.clears == 2  # type: ignore[attr-defined]


async def test_close_stops_turn_admission_and_waits_for_the_whole_turn() -> None:
    ctx = _fake_caller_context()
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=1.0)

    with tracker.turn_span() as admitted:
        assert admitted is True
        closing = asyncio.create_task(ctx.aclose_session())
        await asyncio.sleep(0)
        assert tracker.turn_admission_open is False
        with tracker.turn_span() as rejected:
            assert rejected is False
        assert ctx.engine.deletes == 0

    await closing
    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


async def test_close_captures_abandonment_after_the_active_turn_becomes_idle() -> None:
    engine = _FakeEngine(pending=False)
    ctx = _fake_caller_context(engine)
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=1.0)

    with tracker.turn_span() as admitted:
        assert admitted is True
        closing = asyncio.create_task(ctx.aclose_session())
        await asyncio.sleep(0)
        engine._pending = True

    await closing
    assert ctx.close_had_pending_interrupt is True


async def test_close_quiescence_timeout_fails_without_clearing_live_state() -> None:
    ctx = _fake_caller_context()
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=0.01)

    with tracker.turn_span() as admitted:
        assert admitted is True
        with pytest.raises(TimeoutError):
            await ctx.aclose_session()
        assert ctx.engine.deletes == 0
        assert ctx.cart_store.clears == 0  # type: ignore[attr-defined]
        assert ctx._closed is False


async def test_takeover_wait_never_occupies_a_pool_thread(monkeypatch) -> None:
    ctx = _fake_caller_context()
    tracker = NodeExecutionTracker()
    ctx.attach_execution_quiescence(tracker, timeout_seconds=0.01)

    async def forbidden_to_thread(*_args, **_kwargs):
        raise AssertionError("close must not block a worker thread on takeover quiescence")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)
    with ctx.cancellation_takeover_lease() as acquired:
        assert acquired is True
        with pytest.raises(TimeoutError):
            await ctx.aclose_session()
        assert ctx.engine.deletes == 0


async def test_close_waits_for_the_cancellation_takeover_lease_without_double_firing() -> None:
    ctx = _fake_caller_context()

    with ctx.cancellation_takeover_lease() as acquired:
        assert acquired is True
        closings = (
            asyncio.create_task(ctx.aclose_session()),
            asyncio.create_task(ctx.aclose_session()),
        )
        await asyncio.sleep(0)
        assert ctx.engine.deletes == 0
        assert ctx.cart_store.clears == 0  # type: ignore[attr-defined]

    await asyncio.gather(*closings)
    assert ctx.engine.deletes == 1
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]


async def test_cancellation_takeover_lease_is_rejected_after_close_starts() -> None:
    ctx = _fake_caller_context()

    with ctx.cancellation_takeover_lease() as acquired:
        assert acquired is True
        closing = asyncio.create_task(ctx.aclose_session())
        await asyncio.sleep(0)
        with ctx.cancellation_takeover_lease() as rejected:
            assert rejected is False

    await closing
    assert ctx.engine.deletes == 1


async def test_voice_loop_registers_awaited_caller_teardown_for_job_shutdown(
    config_root: Path,
) -> None:
    loop = await _loop(config_root, RecordingResolver())
    engine = _DelayedDeleteEngine()
    loop.application.state.caller_context.engine = engine  # type: ignore[assignment]
    job = _FakeJobContext()
    resources_released = asyncio.Event()

    async def release_resources() -> None:
        assert engine.deletes == 1
        resources_released.set()

    loop.register_shutdown(  # type: ignore[arg-type]
        job,
        release_job_resources=release_resources,
    )
    assert len(job.shutdown_callbacks) == 1

    shutdown = asyncio.create_task(job.shutdown_callbacks[0]())
    await engine.delete_entered.wait()
    assert shutdown.done() is False
    assert not resources_released.is_set()

    engine.release_delete.set()
    await shutdown
    assert engine.deletes == 1
    assert resources_released.is_set()


async def test_job_shutdown_releases_session_after_lease_supervisor_failure(
    config_root: Path,
) -> None:
    loop = await _loop(config_root, RecordingResolver())
    engine = _FakeEngine(pending=False)
    loop.application.state.caller_context.engine = engine  # type: ignore[assignment]
    supervisor = _FailingLeaseSupervisor()
    loop.lease_supervisor = supervisor  # type: ignore[assignment]
    job = _FakeJobContext()
    resources_released = asyncio.Event()

    async def release_resources() -> None:
        resources_released.set()

    loop.register_shutdown(  # type: ignore[arg-type]
        job,
        release_job_resources=release_resources,
    )

    with pytest.raises(RuntimeError, match="lease-supervisor close failure"):
        await job.shutdown_callbacks[0]()

    assert supervisor.close_calls == 1
    assert engine.deletes == 1
    assert resources_released.is_set()


async def test_lease_loss_stops_turns_and_output_before_transport_disconnect() -> None:
    context = _fake_caller_context()
    tracker = NodeExecutionTracker()
    context.attach_execution_quiescence(tracker, timeout_seconds=1.0)
    session = _RetirementSession(tracker)
    room = _RetirementRoom(tracker)

    await retire_voice_transport(
        session,  # type: ignore[arg-type]
        room,  # type: ignore[arg-type]
        context,
        timeout_seconds=1.0,
    )

    assert session.shutdown_calls == [False]
    assert room.disconnect_calls == 1
    assert context.engine.deletes == 0
    assert context.cart_store.clears == 1  # type: ignore[attr-defined]
    assert context.verification_store.clears == 1  # type: ignore[attr-defined]
    sink = context.telemetry.sink
    assert isinstance(sink, InMemoryTelemetrySink)
    assert [(record.event, record.attributes.get("reason")) for record in sink.records] == [
        ("caller_context_closed", "authority_lost")
    ]


async def test_transport_retirement_timeout_still_closes_local_caller_state() -> None:
    context = _fake_caller_context()
    tracker = NodeExecutionTracker()
    context.attach_execution_quiescence(tracker, timeout_seconds=1.0)
    session = _RetirementSession(tracker)
    room = _RetirementRoom(tracker, block=True)

    with pytest.raises(TimeoutError):
        await retire_voice_transport(
            session,  # type: ignore[arg-type]
            room,  # type: ignore[arg-type]
            context,
            timeout_seconds=0.01,
        )

    assert session.shutdown_calls == [False]
    assert room.disconnect_calls == 1
    assert context.engine.deletes == 0
    assert context.cart_store.clears == 1  # type: ignore[attr-defined]


async def test_transport_retirement_attempts_disconnect_after_output_shutdown_failure() -> None:
    context = _fake_caller_context()
    tracker = NodeExecutionTracker()
    context.attach_execution_quiescence(tracker, timeout_seconds=1.0)
    session = _RetirementSession(tracker, fail_shutdown=True)
    room = _RetirementRoom(tracker)

    with pytest.raises(RuntimeError, match="output shutdown"):
        await retire_voice_transport(
            session,  # type: ignore[arg-type]
            room,  # type: ignore[arg-type]
            context,
            timeout_seconds=1.0,
        )

    assert session.shutdown_calls == [False]
    assert room.disconnect_calls == 1
    assert context.engine.deletes == 0
    assert context.cart_store.clears == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_thread_reaper_is_reentrant_safe() -> None:
    # Clock B: a double-fired session close runs the teardown + emits the abandoned event AT
    # MOST once (the reaper's own re-entrant guard; the teardown itself is idempotent too).
    from agnostic_market.voice.pipeline import _attach_thread_reaper

    session = _FakeSession()
    ctx = _fake_caller_context()
    _attach_thread_reaper(session, ctx)  # type: ignore[arg-type]
    session.handlers["close"](object())
    session.handlers["close"](object())  # double fire
    for _ in range(100):
        if ctx.engine.deletes:
            break
        await asyncio.sleep(0)
    assert ctx.engine.deletes == 1  # type: ignore[attr-defined]  # reaped once despite double fire
    assert ctx.verification_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.cart_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.recent_orders.clears == 1  # type: ignore[attr-defined]
    assert ctx.identity_store.clears == 1  # type: ignore[attr-defined]
    assert ctx.guest_orders.clears == 1  # type: ignore[attr-defined]
    sink = ctx.telemetry.sink
    assert isinstance(sink, InMemoryTelemetrySink)
    assert [record.event for record in sink.records] == [
        "caller_context_closed",
        "flow_abandoned",
    ]


async def test_abandoned_interrupt_is_recorded_when_another_close_path_runs_first(
    config_root: Path,
) -> None:
    from agnostic_market.voice.pipeline import _attach_thread_reaper

    loop = await _loop(config_root, RecordingResolver())
    context = loop.application.state.caller_context
    loop.application.state.cart_store.add_item(
        sku="SKU-GRN-15",
        name="merino hiking socks",
        price_usd=14.50,
        quantity=1,
    )
    await engine_events(loop.engine, "place my order")
    assert bool((await loop.engine._graph.aget_state(loop.engine._config)).interrupts)

    session = _FakeSession()
    _attach_thread_reaper(session, context)  # type: ignore[arg-type]
    with context.cancellation_takeover_lease() as acquired:
        assert acquired is True
        first_close = asyncio.create_task(context.aclose_session())
        await asyncio.sleep(0)
        session.handlers["close"](object())
        await asyncio.sleep(0)

    await first_close
    for _ in range(100):
        await asyncio.sleep(0)
        if context._closed:
            break

    sink = loop.application.services.telemetry.operational_sink
    assert isinstance(sink, InMemoryTelemetrySink)
    assert [record.event for record in sink.records].count("flow_abandoned") == 1
