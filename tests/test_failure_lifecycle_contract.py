"""LangGraph 1.2.7 contracts required by the engine failure-lifecycle design."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from livekit import rtc
from livekit.agents import Agent, AgentSession, CloseEvent, CloseReason
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics
from livekit.agents.voice.room_io import RoomIO
from livekit.plugins.langchain import LLMAdapter
from llm_fakes import RecordingResolver
from routing_helpers import ArchitectureRoutingRecognizer
from telemetry_helpers import make_tenant_telemetry

from agnostic_market.agents.telemetry import (
    InMemoryTelemetrySink,
    OperationalTelemetryEvent,
    TenantTelemetry,
)
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.orchestration import ViewCart
from agnostic_market.dtos.state import (
    ReasoningState,
    merge_consumed_turn_ids,
    open_active_invocation,
)
from agnostic_market.llm.gateway import load_provider_credentials
from agnostic_market.voice.pipeline import VoiceLoop, build_voice_loop
from scripts.close_evidence_recorder import (
    CLOSE_CERTIFICATION_CASE_ENV,
    CLOSE_CERTIFICATION_REPORT_ENV,
    CloseCaseConfig,
    CloseCertificationError,
    CloseCertificationRequest,
    CloseEvidenceRecorder,
    TelemetryCounts,
    load_close_certification_request,
)

_WAIT_TIMEOUT_SECONDS = 5.0


def test_close_telemetry_counts_are_registered_operational_events() -> None:
    registered = {event.value for event in OperationalTelemetryEvent}

    assert set(TelemetryCounts.model_fields) <= registered


def _append(left: list[str] | None, right: list[str] | None) -> list[str]:
    return (left or []) + (right or [])


class _ContractState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    automation_terminal: bool
    pending_recovery: str | None
    prior_intent: str | None
    disposition: str | None
    effect_count: int
    normal_count: int
    consumed_turn_ids: Annotated[tuple[str, ...], merge_consumed_turn_ids]
    pending_route: str | None
    visited: Annotated[list[str], _append]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class _ExecutionTracker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self.entered_threads: list[int] = []
        self.exited_threads: list[int] = []

    @property
    def active(self) -> int:
        with self._condition:
            return self._active

    def run(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        thread_id = threading.get_ident()
        with self._condition:
            self._active += 1
            self.entered_threads.append(thread_id)
        try:
            return operation(*args, **kwargs)
        finally:
            with self._condition:
                self.exited_threads.append(thread_id)
                self._active -= 1
                if self._active == 0:
                    self._condition.notify_all()

    def wait_until_idle(self, timeout: float) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: self._active == 0, timeout=timeout)


def _tracked_sync_node(
    tracker: _ExecutionTracker,
    node: Runnable | Callable[..., Any],
) -> Callable[[_ContractState, RunnableConfig], Any]:
    runnable = node if isinstance(node, Runnable) else RunnableLambda(node)

    def run(state: _ContractState, config: RunnableConfig) -> Any:
        return tracker.run(runnable.invoke, state, config)

    return run


class _BlockingProviderGraph:
    def __init__(self, timeline: list[str]) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finalized = asyncio.Event()
        self.cancelled = False
        self.calls = 0
        self._timeline = timeline

    async def astream(self, _state: object, *_args: object, **_kwargs: object):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
            yield "completed"
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self._timeline.append("provider_finalized")
            self.finalized.set()


async def _start_blocked_livekit_session(
    timeline: list[str],
) -> tuple[AgentSession, _BlockingProviderGraph, list[CloseEvent]]:
    graph = _BlockingProviderGraph(timeline)
    session = AgentSession(llm=LLMAdapter(graph))  # type: ignore[arg-type]
    close_events: list[CloseEvent] = []

    @session.on("close")
    def record_close(event: CloseEvent) -> None:
        close_events.append(event)
        timeline.append("close")

    await session.start(Agent(instructions=""), record=False)
    session.generate_reply(user_input="initial lifecycle turn")
    await asyncio.wait_for(graph.started.wait(), timeout=_WAIT_TIMEOUT_SECONDS)
    return session, graph, close_events


async def test_shielded_quiescence_reawaits_after_a_second_cancellation() -> None:
    tracker = _ExecutionTracker()
    worker_started = threading.Event()
    release_worker = threading.Event()
    cleanup_started = asyncio.Event()
    events: list[str] = []

    def blocking_worker() -> None:
        worker_started.set()
        if not release_worker.wait(timeout=_WAIT_TIMEOUT_SECONDS):
            raise TimeoutError("test did not release tracked worker")

    worker = asyncio.create_task(asyncio.to_thread(tracker.run, blocking_worker))
    assert await asyncio.to_thread(worker_started.wait, _WAIT_TIMEOUT_SECONDS)

    async def owned_turn() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError as original:
            cleanup_task = asyncio.create_task(
                asyncio.to_thread(tracker.wait_until_idle, _WAIT_TIMEOUT_SECONDS)
            )
            cleanup_started.set()
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            assert cleanup_task.result() is True
            events.append("takeover")
            raise original

    turn = asyncio.create_task(owned_turn())
    await asyncio.sleep(0)
    turn.cancel("original-cancellation")
    await cleanup_started.wait()
    turn.cancel("later-cancellation")
    release_worker.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await turn
    await worker

    assert cancelled.value.args == ("original-cancellation",)
    assert events == ["takeover"]


def test_node_error_handler_receives_typed_context_and_completes() -> None:
    handled: list[NodeError] = []

    def explode(_state: _ContractState) -> dict:
        raise RuntimeError("contract failure")

    def recover(_state: _ContractState, error: NodeError) -> Command:
        handled.append(error)
        return Command(update={"disposition": "recovered"}, goto=END)

    builder = StateGraph(_ContractState)
    builder.add_node("explode", explode, error_handler=recover)
    builder.add_edge(START, "explode")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("exception-handler")

    result = graph.invoke({}, config)

    assert result["disposition"] == "recovered"
    assert len(handled) == 1
    assert handled[0].node == "explode"
    assert isinstance(handled[0].error, RuntimeError)
    assert graph.get_state(config).next == ()


async def test_node_error_handler_completes_under_update_streaming() -> None:
    def explode(_state: _ContractState) -> dict:
        raise RuntimeError("contract failure")

    def recover(_state: _ContractState, error: NodeError) -> Command:
        assert error.node == "explode"
        return Command(update={"disposition": "recovered"}, goto=END)

    builder = StateGraph(_ContractState)
    builder.add_node("explode", explode, error_handler=recover)
    builder.add_edge(START, "explode")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("exception-handler-streaming")

    updates = [
        item
        async for item in graph.astream(
            {},
            config,
            stream_mode="updates",
        )
    ]

    assert updates
    assert graph.get_state(config).values["disposition"] == "recovered"
    assert graph.get_state(config).next == ()


async def test_message_stream_rethrows_a_handled_error_on_langgraph_1_2_7() -> None:
    def explode(_state: _ContractState) -> dict:
        raise RuntimeError("contract failure")

    def recover(_state: _ContractState, error: NodeError) -> Command:
        assert error.node == "explode"
        return Command(update={"disposition": "recovered"}, goto=END)

    builder = StateGraph(_ContractState)
    builder.add_node("explode", explode, error_handler=recover)
    builder.add_edge(START, "explode")
    graph = builder.compile(checkpointer=InMemorySaver())

    with pytest.raises(RuntimeError, match="contract failure"):
        async for _ in graph.astream(
            {},
            _config("message-stream-handler-defect"),
            stream_mode="messages",
        ):
            pass


async def test_external_stream_cancellation_bypasses_node_error_handler() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    handled: list[NodeError] = []

    async def slow(_state: _ContractState) -> dict:
        started.set()
        await release.wait()
        return {"disposition": "completed"}

    def recover(_state: _ContractState, error: NodeError) -> Command:
        handled.append(error)
        return Command(update={"disposition": "recovered"}, goto=END)

    builder = StateGraph(_ContractState)
    builder.add_node("slow", slow, error_handler=recover, destinations=(END,))
    builder.add_edge(START, "slow")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("external-cancellation")

    async def drain() -> None:
        async for _ in graph.astream({}, config):
            pass

    task = asyncio.create_task(drain())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = graph.get_state(config)
    assert handled == []
    assert snapshot.next == ("slow",)
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].error is None


async def test_livekit_langgraph_stream_preserves_the_committed_message_id() -> None:
    captured: dict[str, object] = {}

    class CaptureGraph:
        async def astream(self, state: dict[str, object], *_args: object, **_kwargs: object):
            captured.update(state)
            yield "acknowledged"

    chat_context = ChatContext.empty()
    committed = chat_context.add_message(
        role="user",
        content="cancel my order",
        id="transport-turn-1",
    )
    stream = LLMAdapter(CaptureGraph()).chat(chat_ctx=chat_context)  # type: ignore[arg-type]

    response = await stream.collect()

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert response.text == "acknowledged"
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].id == committed.id == "transport-turn-1"


async def test_livekit_langgraph_stream_cancellation_reaches_the_graph_generator() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finalized = asyncio.Event()
    release = asyncio.Event()

    class CancellationGraph:
        async def astream(self, _state: object, *_args: object, **_kwargs: object):
            started.set()
            try:
                await release.wait()
                yield "unexpected"
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                finalized.set()

    chat_context = ChatContext.empty()
    chat_context.add_message(role="user", content="cancel my order", id="transport-turn-2")
    stream = LLMAdapter(CancellationGraph()).chat(chat_ctx=chat_context)  # type: ignore[arg-type]

    await asyncio.wait_for(started.wait(), timeout=_WAIT_TIMEOUT_SECONDS)
    await stream.aclose()

    assert cancelled.is_set()
    assert finalized.is_set()


async def test_participant_disconnect_interrupts_generation_before_close_event() -> None:
    timeline: list[str] = []
    session, graph, close_events = await _start_blocked_livekit_session(timeline)
    participant = cast(
        rtc.RemoteParticipant,
        SimpleNamespace(
            identity="contract-caller",
            disconnect_reason=rtc.DisconnectReason.CLIENT_INITIATED,
        ),
    )
    room = cast(
        rtc.Room,
        SimpleNamespace(name="contract-room"),
    )
    room_io = RoomIO(
        agent_session=session,
        room=room,
        participant=participant.identity,
    )
    room_io._participant_available_fut.set_result(participant)
    try:
        room_io._on_participant_disconnected(participant)
        assert session._closing_task is not None
        await asyncio.wait_for(session._closing_task, timeout=_WAIT_TIMEOUT_SECONDS)

        assert graph.cancelled is True
        assert graph.finalized.is_set()
        assert len(close_events) == 1
        assert isinstance(close_events[0], CloseEvent)
        assert close_events[0].reason == CloseReason.PARTICIPANT_DISCONNECTED
        assert timeline == ["provider_finalized", "close"]
    finally:
        graph.release.set()
        await session.aclose()


async def test_graceful_close_retains_late_transcript_without_starting_another_turn() -> None:
    timeline: list[str] = []
    session, graph, close_events = await _start_blocked_livekit_session(timeline)
    late_transcript = "late committed transcript"

    @session.on("conversation_item_added")
    def record_late_transcript(event: object) -> None:
        item = getattr(event, "item", None)
        if isinstance(item, ChatMessage) and item.text_content == late_transcript:
            timeline.append("late_transcript")

    activity = session._activity
    assert activity is not None
    assert not activity._q_updated.is_set()

    try:
        session.shutdown(drain=True)
        assert session._closing_task is not None
        assert not session._closing_task.done()
        assert graph.cancelled is False
        await asyncio.wait_for(
            activity._q_updated.wait(),
            timeout=_WAIT_TIMEOUT_SECONDS,
        )
        assert session._closing is True
        assert activity.scheduling_paused is True

        accepted = activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript=late_transcript,
                transcript_confidence=0.9,
                metrics=_EndOfTurnMetrics(
                    started_speaking_at=None,
                    stopped_speaking_at=None,
                    transcription_delay=None,
                    end_of_turn_delay=None,
                ),
            )
        )

        assert accepted is True
        assert graph.calls == 1
        assert [
            item.text_content
            for item in session.history.items
            if isinstance(item, ChatMessage)
            and item.role == "user"
            and item.text_content == late_transcript
        ] == [late_transcript]

        graph.release.set()
        await asyncio.wait_for(session._closing_task, timeout=_WAIT_TIMEOUT_SECONDS)

        assert graph.cancelled is False
        assert graph.finalized.is_set()
        assert len(close_events) == 1
        assert isinstance(close_events[0], CloseEvent)
        assert close_events[0].reason == CloseReason.USER_INITIATED
        assert timeline == ["late_transcript", "provider_finalized", "close"]
    finally:
        graph.release.set()
        await session.aclose()


class _RecorderRoom:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable[[object], None]]] = {}

    def on(self, event: str, callback: Callable[[object], None]) -> Callable[[object], None]:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, value: object) -> None:
        for callback in tuple(self.handlers.get(event, ())):
            callback(value)


async def _recorder_loop(config_root: Path) -> VoiceLoop:
    from agnostic_market.application import build_fixture_tenant_services

    resolved = ConfigRegistry(config_root).load().get("acme_store")
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    return build_voice_loop(
        resolved,
        credentials,
        RecordingResolver(),
        deployment_id="test-close-evidence-artifact",
        tenant_services=build_fixture_tenant_services(
            config_root,
            "acme_store",
            telemetry=make_tenant_telemetry("acme_store"),
        ),
        routing_recognizer_factory=lambda _registry: ArchitectureRoutingRecognizer(),
    )


def _recorder_request(
    report_path: Path,
    *,
    shutdown_mode: str = "participant_disconnect",
) -> CloseCertificationRequest:
    case = (
        CloseCaseConfig(shutdown_mode="participant_disconnect")
        if shutdown_mode == "participant_disconnect"
        else CloseCaseConfig(
            shutdown_mode="graceful_drain",
            graceful_trigger="first_thinking_after_user",
        )
    )
    return CloseCertificationRequest(
        case_name=f"contract-{shutdown_mode}",
        case=case,
        report_path=report_path,
        finalization_timeout_seconds=_WAIT_TIMEOUT_SECONDS,
    )


async def _attached_recorder(
    config_root: Path,
    report_path: Path,
    *,
    shutdown_mode: str = "participant_disconnect",
) -> tuple[VoiceLoop, CloseEvidenceRecorder, _RecorderRoom]:
    loop = await _recorder_loop(config_root)
    room = _RecorderRoom()
    recorder = CloseEvidenceRecorder(
        _recorder_request(report_path, shutdown_mode=shutdown_mode),
        merchant_id="acme_store",
        telemetry=loop.application.services.telemetry.operational_sink,
    )
    recorder.attach(
        session=loop.session,
        room=cast(rtc.Room, room),
        engine=loop.engine,
        effect_source=loop.application.services.order_store,
        linked_participant_identity="contract-caller",
    )
    return loop, recorder, room


def _disconnect(room: _RecorderRoom) -> None:
    room.emit(
        "participant_disconnected",
        cast(
            rtc.RemoteParticipant,
            SimpleNamespace(
                identity="contract-caller",
                disconnect_reason=rtc.DisconnectReason.CLIENT_INITIATED,
            ),
        ),
    )


def test_close_certification_opt_in_is_complete_and_case_bound(
    config_root: Path,
    tmp_path: Path,
) -> None:
    assert load_close_certification_request(config_root, {}) is None
    with pytest.raises(CloseCertificationError, match="must be set together"):
        load_close_certification_request(
            config_root,
            {CLOSE_CERTIFICATION_CASE_ENV: "participant_slow_model"},
        )
    with pytest.raises(CloseCertificationError, match="unknown close-certification case"):
        load_close_certification_request(
            config_root,
            {
                CLOSE_CERTIFICATION_CASE_ENV: "not-a-case",
                CLOSE_CERTIFICATION_REPORT_ENV: str(tmp_path / "unknown.json"),
            },
        )

    request = load_close_certification_request(
        config_root,
        {
            CLOSE_CERTIFICATION_CASE_ENV: "participant_stt_fragment",
            CLOSE_CERTIFICATION_REPORT_ENV: str(tmp_path / "report.json"),
        },
    )
    assert request is not None
    assert request.case_name == "participant_stt_fragment"
    assert request.case.shutdown_mode == "participant_disconnect"


async def test_close_recorder_permits_only_one_active_certification_job(
    config_root: Path,
    tmp_path: Path,
) -> None:
    first_loop, first_recorder, _room = await _attached_recorder(
        config_root,
        tmp_path / "first.json",
    )
    second_loop = await _recorder_loop(config_root)
    second_recorder = CloseEvidenceRecorder(
        _recorder_request(tmp_path / "second.json"),
        merchant_id="acme_store",
        telemetry=InMemoryTelemetrySink(),
    )

    with pytest.raises(CloseCertificationError, match=r"another .* recorder is active"):
        second_recorder.attach(
            session=second_loop.session,
            room=cast(rtc.Room, _RecorderRoom()),
            engine=second_loop.engine,
            effect_source=second_loop.application.services.order_store,
            linked_participant_identity="other-contract-caller",
        )

    assert "_complete_close" not in second_loop.engine._lifecycle.__dict__
    first_recorder._release()
    assert "_complete_close" not in first_loop.engine._lifecycle.__dict__


@pytest.mark.parametrize("lifecycle_first", [False, True])
async def test_close_recorder_is_independent_of_lifecycle_listener_order_and_redacts_content(
    config_root: Path,
    tmp_path: Path,
    lifecycle_first: bool,
) -> None:
    report_path = tmp_path / f"order-{lifecycle_first}.json"
    loop, recorder, room = await _attached_recorder(config_root, report_path)
    context = loop.engine._lifecycle
    assert context is not None
    sink = loop.application.services.telemetry.operational_sink
    TenantTelemetry("other_store", sink, sink).bind_session("other-session").operational.record(
        {"event": "caller_context_closed", "reason": "unrelated_session"}
    )
    secret_text = "casey@example.com OTP 123456"
    loop.session._conversation_item_added(
        ChatMessage(id="user-contract-id", role="user", content=[secret_text])
    )
    loop.session._conversation_item_added(
        ChatMessage(id="assistant-contract-id", role="assistant", content=["private reply"])
    )
    _disconnect(room)
    event = CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED)

    if lifecycle_first:
        await context.aclose_session()
        recorder._on_close(event)
    else:
        recorder._on_close(event)
        await context.aclose_session()

    assert await recorder.wait_for_completion() == report_path
    raw_report = report_path.read_text(encoding="utf-8")
    assert secret_text not in raw_report
    assert "private reply" not in raw_report
    report = json.loads(raw_report)
    assert report["status"] == "complete"
    assert report["telemetry"]["caller_context_closed"] == 1
    assert report["all_observed_reasoning_threads_retired"] is True
    assert [message["message_id"] for message in report["messages"]] == [
        "user-contract-id",
        "assistant-contract-id",
    ]
    assert report["messages"][0]["text_present"] is True


@pytest.mark.parametrize("active_kind", ["turn", "mutable_worker"])
async def test_close_recorder_waits_for_full_idle_before_reading_or_writing(
    config_root: Path,
    tmp_path: Path,
    active_kind: str,
) -> None:
    report_path = tmp_path / f"idle-{active_kind}.json"
    loop, recorder, room = await _attached_recorder(config_root, report_path)
    tracker = loop.engine._node_execution_tracker
    _disconnect(room)
    event = CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED)

    if active_kind == "turn":
        with tracker.turn_span() as admitted:
            assert admitted is True
            recorder._on_close(event)
            await asyncio.sleep(0)
            assert not report_path.exists()
    else:
        worker_started = threading.Event()
        release_worker = threading.Event()

        def block_worker() -> None:
            worker_started.set()
            if not release_worker.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release recorder worker")

        worker = asyncio.create_task(
            asyncio.to_thread(tracker._run, "recorder_contract", block_worker)
        )
        assert await asyncio.to_thread(worker_started.wait, _WAIT_TIMEOUT_SECONDS)
        recorder._on_close(event)
        await asyncio.sleep(0)
        assert not report_path.exists()
        release_worker.set()
        await worker

    assert await recorder.wait_for_completion() == report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["telemetry"]["caller_context_closed"] == 1


async def test_close_recorder_waits_for_cancellation_takeover_lease(
    config_root: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "takeover-lease.json"
    loop, recorder, room = await _attached_recorder(config_root, report_path)
    context = loop.engine._lifecycle
    assert context is not None
    _disconnect(room)

    with context.cancellation_takeover_lease() as acquired:
        assert acquired is True
        recorder._on_close(CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
        await asyncio.sleep(0)
        assert not report_path.exists()

    assert await recorder.wait_for_completion() == report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["telemetry"]["caller_context_closed"] == 1


async def test_close_recorder_duplicate_close_is_one_failed_report_and_one_teardown(
    config_root: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "duplicate.json"
    _loop, recorder, room = await _attached_recorder(config_root, report_path)
    _disconnect(room)
    event = CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED)
    recorder._on_close(event)
    recorder._on_close(event)

    with pytest.raises(CloseCertificationError, match="failed its structural gate"):
        await recorder.wait_for_completion()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["failure_reasons"] == ["duplicate_close_event"]
    assert report["close"]["delivery_count"] == 2
    assert report["telemetry"]["caller_context_closed"] == 1


async def test_close_recorder_graceful_trigger_is_one_shot(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, recorder, _room = await _attached_recorder(
        config_root,
        tmp_path / "graceful.json",
        shutdown_mode="graceful_drain",
    )
    shutdowns: list[bool] = []
    monkeypatch.setattr(loop.session, "shutdown", lambda *, drain: shutdowns.append(drain))
    recorder._on_conversation_item(
        SimpleNamespace(item=ChatMessage(id="graceful-user", role="user", content=["start"]))
    )
    recorder._on_agent_state_changed(SimpleNamespace(new_state="thinking"))
    recorder._on_agent_state_changed(SimpleNamespace(new_state="thinking"))
    assert shutdowns == [True]
    recorder._release()


async def test_close_recorder_write_failure_never_leaves_a_half_green_report(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import close_evidence_recorder

    report_path = tmp_path / "write-failure.json"
    _loop, recorder, room = await _attached_recorder(config_root, report_path)
    _disconnect(room)

    def fail_write(_path: Path, _report: object) -> None:
        raise OSError("injected report write failure")

    monkeypatch.setattr(close_evidence_recorder, "_write_report_atomic", fail_write)
    recorder._on_close(CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
    with pytest.raises(CloseCertificationError, match="could not be finalized"):
        await recorder.wait_for_completion()
    assert not report_path.exists()


async def test_cancelled_sync_node_continues_after_the_astream_task_unwinds() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    external_writes: list[str] = []

    def slow_write(_state: _ContractState) -> dict[str, object]:
        started.set()
        try:
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release the synchronous node")
            external_writes.append("committed-after-cancel")
            return {"effect_count": 1}
        finally:
            finished.set()

    builder = StateGraph(_ContractState)
    builder.add_node("slow_write", slow_write)
    builder.add_edge(START, "slow_write")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("sync-worker-outlives-cancel")

    async def drain() -> None:
        async for _ in graph.astream({}, config, stream_mode="updates"):
            pass

    task = asyncio.create_task(drain())
    assert await asyncio.to_thread(started.wait, _WAIT_TIMEOUT_SECONDS)
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        assert external_writes == []
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)

    snapshot = graph.get_state(config)
    assert external_writes == ["committed-after-cancel"]
    assert snapshot.values.get("effect_count") is None
    assert snapshot.next == ("slow_write",)


async def test_in_worker_tracker_remains_active_until_the_sync_node_exits() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    tracker = _ExecutionTracker()
    event_loop_thread = threading.get_ident()

    def slow(_state: _ContractState) -> dict[str, object]:
        started.set()
        try:
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release the tracked node")
            return {"disposition": "completed"}
        finally:
            finished.set()

    builder = StateGraph(_ContractState)
    builder.add_node("slow", _tracked_sync_node(tracker, slow))
    builder.add_edge(START, "slow")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("in-worker-tracker")

    async def drain() -> None:
        async for _ in graph.astream({}, config, stream_mode="updates"):
            pass

    task = asyncio.create_task(drain())
    assert await asyncio.to_thread(started.wait, _WAIT_TIMEOUT_SECONDS)
    try:
        assert tracker.active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert tracker.active == 1
        assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)
        assert await asyncio.to_thread(tracker.wait_until_idle, _WAIT_TIMEOUT_SECONDS)

    assert tracker.entered_threads == tracker.exited_threads
    assert tracker.entered_threads[0] != event_loop_thread


def test_in_worker_tracker_wraps_a_real_tool_node() -> None:
    tracker = _ExecutionTracker()

    @tool
    def order_label(order_id: str) -> str:
        """Return a deterministic label for one order."""
        return f"order:{order_id}"

    tools = ToolNode([order_label])
    builder = StateGraph(_ContractState)
    builder.add_node("tools", _tracked_sync_node(tracker, tools))
    builder.add_edge(START, "tools")
    graph = builder.compile(checkpointer=InMemorySaver())

    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "order_label",
                            "args": {"order_id": "ORD-1002"},
                            "id": "tool-call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        _config("tracked-tool-node"),
    )

    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.content for message in tool_messages] == ["order:ORD-1002"]
    assert tracker.active == 0
    assert tracker.entered_threads == tracker.exited_threads
    assert len(tracker.entered_threads) == 1


async def test_runnable_error_listener_fires_before_the_sync_worker_exits() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    listener_called = threading.Event()

    def slow(_state: _ContractState) -> dict[str, object]:
        started.set()
        try:
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release the listened node")
            return {"disposition": "completed"}
        finally:
            finished.set()

    listened = RunnableLambda(slow).with_listeners(
        on_error=lambda _run: listener_called.set(),
    )
    builder = StateGraph(_ContractState)
    builder.add_node("slow", listened)
    builder.add_edge(START, "slow")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("listener-is-not-quiescence")

    async def drain() -> None:
        async for _ in graph.astream({}, config, stream_mode="updates"):
            pass

    task = asyncio.create_task(drain())
    assert await asyncio.to_thread(started.wait, _WAIT_TIMEOUT_SECONDS)
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(listener_called.wait, _WAIT_TIMEOUT_SECONDS)
        assert not finished.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)


async def test_start_takeover_waits_for_tracked_sync_work_to_be_quiescent() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    wait_started = asyncio.Event()
    external_writes: list[str] = []
    tracker = _ExecutionTracker()

    def entry(_state: _ContractState) -> dict[str, object]:
        return {"visited": ["entry"]}

    def route_after_entry(state: _ContractState) -> str:
        return "recover" if state.get("pending_recovery") else "slow_write"

    def slow_write(_state: _ContractState) -> dict[str, object]:
        started.set()
        try:
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release the takeover node")
            external_writes.append("committed")
            return {"effect_count": 1}
        finally:
            finished.set()

    def recover(_state: _ContractState) -> dict[str, object]:
        return {"disposition": "recovered"}

    builder = StateGraph(_ContractState)
    builder.add_node("entry", entry)
    builder.add_node("slow_write", _tracked_sync_node(tracker, slow_write))
    builder.add_node("recover", recover)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        route_after_entry,
        {"recover": "recover", "slow_write": "slow_write"},
    )
    builder.add_edge("slow_write", END)
    builder.add_edge("recover", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("takeover-after-quiescence")

    async def drain() -> None:
        async for _ in graph.astream({}, config, stream_mode="updates"):
            pass

    async def takeover() -> None:
        wait_started.set()
        idle = await asyncio.to_thread(tracker.wait_until_idle, _WAIT_TIMEOUT_SECONDS)
        if not idle:
            raise TimeoutError("tracked node did not become quiescent")
        graph.update_state(
            config,
            {"pending_recovery": "safe_abort"},
            as_node=START,
        )

    task = asyncio.create_task(drain())
    assert await asyncio.to_thread(started.wait, _WAIT_TIMEOUT_SECONDS)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    takeover_task = asyncio.create_task(takeover())
    await asyncio.wait_for(wait_started.wait(), timeout=_WAIT_TIMEOUT_SECONDS)
    try:
        assert not takeover_task.done()
        assert graph.get_state(config).next == ("slow_write",)
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)

    await asyncio.wait_for(takeover_task, timeout=_WAIT_TIMEOUT_SECONDS)
    snapshot = graph.get_state(config)
    assert external_writes == ["committed"]
    assert snapshot.values["pending_recovery"] == "safe_abort"
    assert snapshot.next == ("entry",)


async def test_cancellation_after_external_write_leaves_ambiguous_checkpoint() -> None:
    committed = asyncio.Event()
    release = asyncio.Event()
    ledger: dict[str, str] = {}

    async def write_then_wait(_state: _ContractState) -> dict:
        ledger["effect-key"] = "committed"
        committed.set()
        await release.wait()
        return {"effect_count": 1}

    builder = StateGraph(_ContractState)
    builder.add_node("effect", write_then_wait)
    builder.add_edge(START, "effect")
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("write-cancellation")

    async def drain() -> None:
        async for _ in graph.astream({}, config):
            pass

    task = asyncio.create_task(drain())
    await committed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = graph.get_state(config)
    assert ledger == {"effect-key": "committed"}
    assert snapshot.next == ("effect",)
    assert snapshot.values.get("effect_count") is None
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].error is None


def test_command_goto_adds_to_a_static_edge() -> None:
    def source(_state: _ContractState) -> Command:
        return Command(update={"visited": ["source"]}, goto="dynamic")

    builder = StateGraph(_ContractState)
    builder.add_node("source", source)
    builder.add_node("static", lambda _state: {"visited": ["static"]})
    builder.add_node("dynamic", lambda _state: {"visited": ["dynamic"]})
    builder.add_edge(START, "source")
    builder.add_edge("source", "static")

    result = builder.compile().invoke({})

    assert sorted(result["visited"]) == ["dynamic", "source", "static"]


def test_command_goto_adds_to_a_conditional_edge() -> None:
    def source(_state: _ContractState) -> Command:
        return Command(update={"visited": ["source"]}, goto="dynamic")

    builder = StateGraph(_ContractState)
    builder.add_node("source", source)
    builder.add_node("conditional", lambda _state: {"visited": ["conditional"]})
    builder.add_node("dynamic", lambda _state: {"visited": ["dynamic"]})
    builder.add_edge(START, "source")
    builder.add_conditional_edges(
        "source",
        lambda _state: "conditional",
        {"conditional": "conditional"},
    )

    result = builder.compile().invoke({})

    assert sorted(result["visited"]) == ["conditional", "dynamic", "source"]


def test_command_goto_end_does_not_suppress_a_static_edge() -> None:
    def source(_state: _ContractState) -> Command:
        return Command(update={"visited": ["source"]}, goto=END)

    builder = StateGraph(_ContractState)
    builder.add_node("source", source)
    builder.add_node("static", lambda _state: {"visited": ["static"]})
    builder.add_edge(START, "source")
    builder.add_edge("source", "static")

    result = builder.compile().invoke({})

    assert result["visited"] == ["source", "static"]


def test_destinations_render_an_edge_without_executing_it() -> None:
    builder = StateGraph(_ContractState)
    builder.add_node(
        "source",
        lambda _state: {"visited": ["source"]},
        destinations=("rendered",),
    )
    builder.add_node("rendered", lambda _state: {"visited": ["rendered"]})
    builder.add_edge(START, "source")
    graph = builder.compile()

    result = graph.invoke({})
    rendered_edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert result["visited"] == ["source"]
    assert ("source", "rendered") in rendered_edges


async def test_pending_route_survives_cancellation_before_its_consumer_commits() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def no_action(_state: _ContractState) -> dict:
        started.set()
        await release.wait()
        return {"pending_route": None, "visited": ["no_action"]}

    builder = StateGraph(_ContractState)
    builder.add_node("entry", lambda _state: {})
    builder.add_node("no_action", no_action)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        lambda state: "no_action" if state.get("pending_route") else END,
        {"no_action": "no_action", END: END},
    )
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("pending-route-cancellation")
    graph.update_state(config, {"pending_route": "clarify"}, as_node="__start__")

    async def drain() -> None:
        async for _ in graph.astream(None, config):
            pass

    first = asyncio.create_task(drain())
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    cancelled = graph.get_state(config)
    assert cancelled.values["pending_route"] == "clarify"
    assert cancelled.next == ("no_action",)

    release.set()
    await graph.ainvoke(None, config)
    finished = graph.get_state(config)
    assert finished.values.get("pending_route") is None
    assert finished.values["visited"] == ["no_action"]
    assert finished.next == ()


def _recovery_contract_graph(
    *,
    old_work_started: asyncio.Event | None = None,
    release_old_work: asyncio.Event | None = None,
):
    def entry(_state: _ContractState) -> dict:
        return {"visited": ["entry"]}

    def route_after_entry(state: _ContractState) -> str:
        if state.get("pending_recovery"):
            return "recover"
        return "old_work" if state.get("prior_intent") else "normal"

    def recover(state: _ContractState) -> dict | Command:
        if state["pending_recovery"] == "safe_abort":
            return Command(
                update={
                    "pending_recovery": None,
                    "prior_intent": None,
                    "disposition": "aborted",
                    "visited": ["recover"],
                },
                goto="entry",
            )
        return {
            "pending_recovery": None,
            "prior_intent": None,
            "disposition": "consumed",
            "visited": ["recover"],
        }

    def normal(state: _ContractState) -> dict:
        assert state.get("prior_intent") is None
        return {
            "normal_count": state.get("normal_count", 0) + 1,
            "visited": ["normal"],
        }

    async def old_work(_state: _ContractState) -> dict:
        if old_work_started is not None:
            old_work_started.set()
        if release_old_work is not None:
            await release_old_work.wait()
        return {"effect_count": 1, "visited": ["old_work"]}

    builder = StateGraph(_ContractState)
    builder.add_node("entry", entry)
    builder.add_node("recover", recover, destinations=("entry", END))
    builder.add_node("normal", normal)
    builder.add_node("old_work", old_work)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        route_after_entry,
        {"recover": "recover", "old_work": "old_work", "normal": "normal"},
    )
    builder.add_edge("normal", END)
    builder.add_edge("old_work", END)
    return builder.compile(checkpointer=InMemorySaver())


def _human_texts(snapshot) -> tuple[str, ...]:
    return tuple(
        str(message.content)
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    )


async def test_start_seeded_safe_abort_supersedes_stranded_work_and_admits_fresh_message() -> None:
    old_work_started = asyncio.Event()
    release_old_work = asyncio.Event()
    graph = _recovery_contract_graph(
        old_work_started=old_work_started,
        release_old_work=release_old_work,
    )
    config = _config("safe-abort-admission")

    async def drain() -> None:
        async for _ in graph.astream({"prior_intent": "cancel order"}, config):
            pass

    abandoned = asyncio.create_task(drain())
    await old_work_started.wait()
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    stranded = graph.get_state(config)
    assert stranded.next == ("old_work",)
    pre_takeover_visits = list(stranded.values["visited"])

    graph.update_state(
        config,
        {"pending_recovery": "safe_abort"},
        as_node=START,
    )
    assert graph.get_state(config).next == ("entry",)

    result = graph.invoke({"messages": [HumanMessage("never mind")]}, config)
    snapshot = graph.get_state(config)

    assert result["disposition"] == "aborted"
    assert result["normal_count"] == 1
    assert result.get("effect_count", 0) == 0
    assert result["visited"] == [
        *pre_takeover_visits,
        "entry",
        "recover",
        "entry",
        "normal",
    ]
    assert _human_texts(snapshot) == ("never mind",)
    assert snapshot.next == ()


def test_start_seeded_reconcile_consumes_fresh_turn_without_storing_text() -> None:
    graph = _recovery_contract_graph()
    config = _config("reconcile-consumption")
    graph.update_state(
        config,
        {"pending_recovery": "reconcile", "prior_intent": "refund order"},
        as_node=START,
    )

    result = graph.invoke(
        Command(update={"consumed_turn_ids": ("fresh-reconcile-turn",)}),
        config,
    )
    snapshot = graph.get_state(config)

    assert result["disposition"] == "consumed"
    assert result.get("normal_count", 0) == 0
    assert result.get("effect_count", 0) == 0
    assert result["visited"] == ["entry", "recover"]
    assert _human_texts(snapshot) == ()
    assert tuple(snapshot.values["consumed_turn_ids"]) == ("fresh-reconcile-turn",)
    assert snapshot.next == ()


def test_nonempty_noop_command_preserves_start_seeded_recovery_and_ledger() -> None:
    graph = _recovery_contract_graph()
    config = _config("recovery-noop-ledger")
    graph.update_state(
        config,
        {
            "pending_recovery": "reconcile",
            "prior_intent": "refund order",
            "consumed_turn_ids": ("abandoned-turn",),
        },
        as_node=START,
    )

    result = graph.invoke(Command(update={"consumed_turn_ids": ()}), config)
    snapshot = graph.get_state(config)

    assert result["disposition"] == "consumed"
    assert result["visited"] == ["entry", "recover"]
    assert tuple(snapshot.values["consumed_turn_ids"]) == ("abandoned-turn",)
    assert snapshot.next == ()


def test_empty_command_update_is_not_a_valid_recovery_advance() -> None:
    graph = _recovery_contract_graph()
    config = _config("empty-recovery-command")
    graph.update_state(
        config,
        {"pending_recovery": "reconcile", "prior_intent": "refund order"},
        as_node=START,
    )

    with pytest.raises(UnboundLocalError):
        graph.invoke(Command(update={}), config)


def test_interrupt_is_not_intercepted_by_node_error_handler() -> None:
    handled: list[NodeError] = []

    def confirm(_state: _ContractState) -> dict:
        decision = interrupt({"ask": "continue?"})
        return {"disposition": str(decision), "visited": ["confirm"]}

    def recover(_state: _ContractState, error: NodeError) -> Command:
        handled.append(error)
        return Command(update={"disposition": "recovered"}, goto=END)

    def effect(state: _ContractState) -> dict:
        return {
            "effect_count": state.get("effect_count", 0) + 1,
            "visited": ["effect"],
        }

    builder = StateGraph(_ContractState)
    builder.add_node("confirm", confirm, error_handler=recover, destinations=(END,))
    builder.add_node("effect", effect)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", "effect")
    builder.add_edge("effect", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("interrupt-contract")

    graph.invoke({}, config)
    paused = graph.get_state(config)
    assert paused.next == ("confirm",)
    assert [item.value for item in paused.interrupts] == [{"ask": "continue?"}]
    assert handled == []

    result = graph.invoke(Command(resume="yes"), config)
    assert result["disposition"] == "yes"
    assert result["effect_count"] == 1
    assert result["visited"] == ["confirm", "effect"]
    assert handled == []
    assert graph.get_state(config).next == ()


def test_interrupt_resume_atomically_updates_the_consumed_turn_ledger() -> None:
    def confirm(_state: _ContractState) -> dict[str, object]:
        decision = interrupt({"ask": "continue?"})
        return {"disposition": str(decision), "visited": ["confirm"]}

    def effect(state: _ContractState) -> dict[str, object]:
        return {
            "effect_count": state.get("effect_count", 0) + 1,
            "visited": ["effect"],
        }

    builder = StateGraph(_ContractState)
    builder.add_node("confirm", confirm)
    builder.add_node("effect", effect)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", "effect")
    builder.add_edge("effect", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("interrupt-ledger-update")

    graph.invoke({"consumed_turn_ids": ("transport-turn-1",)}, config)
    paused = graph.get_state(config)
    assert paused.next == ("confirm",)
    assert len(paused.interrupts) == 1

    result = graph.invoke(
        Command(
            resume="yes",
            update={"consumed_turn_ids": ("transport-turn-1", "transport-turn-2")},
        ),
        config,
    )
    snapshot = graph.get_state(config)

    assert result["disposition"] == "yes"
    assert result["effect_count"] == 1
    assert tuple(snapshot.values["consumed_turn_ids"]) == (
        "transport-turn-1",
        "transport-turn-2",
    )
    assert snapshot.next == ()


def test_principal_seed_can_atomically_open_an_invocation_on_the_merged_ledger() -> None:
    builder = StateGraph(ReasoningState)
    builder.add_node("entry", lambda _state: {})
    builder.add_edge(START, "entry")
    builder.add_edge("entry", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("active-invocation-seed")
    consumed_turn_ids = ("identity-completion-turn",)
    invocation = open_active_invocation(
        ViewCart(),
        consumed_turn_ids=consumed_turn_ids,
    )

    graph.update_state(
        config,
        {
            "consumed_turn_ids": consumed_turn_ids,
            "active_invocation": invocation,
        },
        as_node="__start__",
    )
    snapshot = graph.get_state(config)
    restored = ReasoningState.model_validate(snapshot.values)

    assert restored.consumed_turn_ids == consumed_turn_ids
    assert restored.active_invocation == invocation
    assert snapshot.next == ("entry",)


def test_resume_turn_opens_an_invocation_from_the_ledger_not_message_history() -> None:
    def confirm(_state: ReasoningState) -> dict[str, object]:
        interrupt({"ask": "continue?"})
        return {}

    def open_invocation(state: ReasoningState) -> dict[str, object]:
        return {
            "active_invocation": open_active_invocation(
                ViewCart(),
                consumed_turn_ids=state.consumed_turn_ids,
            )
        }

    builder = StateGraph(ReasoningState)
    builder.add_node("confirm", confirm)
    builder.add_node("open_invocation", open_invocation)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", "open_invocation")
    builder.add_edge("open_invocation", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("active-invocation-resume")

    graph.invoke(
        {
            "messages": [HumanMessage(content="checkout", id="turn-A")],
            "consumed_turn_ids": ("turn-A",),
        },
        config,
    )
    graph.invoke(
        Command(
            resume={"text": "cancel all instead"},
            update={"consumed_turn_ids": ("turn-B",)},
        ),
        config,
    )
    restored = ReasoningState.model_validate(graph.get_state(config).values)
    human_ids = [message.id for message in restored.messages if isinstance(message, HumanMessage)]

    assert human_ids == ["turn-A"]
    assert restored.consumed_turn_ids == ("turn-A", "turn-B")
    assert restored.active_invocation is not None
    assert restored.active_invocation.opened_turn_id == "turn-B"


def test_interrupt_update_without_resume_consumes_id_and_preserves_interrupt() -> None:
    def confirm(_state: _ContractState) -> dict[str, object]:
        decision = interrupt({"ask": "continue?"})
        return {"disposition": str(decision)}

    builder = StateGraph(_ContractState)
    builder.add_node("confirm", confirm)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("interrupt-ledger-only-update")

    graph.invoke({"consumed_turn_ids": ("transport-turn-1",)}, config)
    before = graph.get_state(config)
    assert before.next == ("confirm",)
    interrupt_id = before.interrupts[0].id

    graph.invoke(
        Command(update={"consumed_turn_ids": ("transport-turn-1", "transport-turn-2")}),
        config,
    )
    after = graph.get_state(config)

    assert tuple(after.values["consumed_turn_ids"]) == (
        "transport-turn-1",
        "transport-turn-2",
    )
    assert after.next == ("confirm",)
    assert after.interrupts[0].id == interrupt_id


def test_terminal_node_takeover_supersedes_a_failed_checkpoint() -> None:
    def explode(_state: _ContractState) -> dict:
        raise RuntimeError("contract failure")

    def terminal(_state: _ContractState) -> dict:
        return {"disposition": "terminal"}

    builder = StateGraph(_ContractState)
    builder.add_node("explode", explode)
    builder.add_node("terminal", terminal)
    builder.add_edge(START, "explode")
    builder.add_edge("terminal", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _config("terminal-takeover")

    with pytest.raises(RuntimeError, match="contract failure"):
        graph.invoke({}, config)
    failed = graph.get_state(config)
    assert failed.next == ("explode",)
    assert len(failed.tasks) == 1
    assert failed.tasks[0].error is not None

    graph.update_state(
        config,
        {"automation_terminal": True, "disposition": "terminal"},
        as_node="terminal",
    )
    snapshot = graph.get_state(config)

    assert snapshot.values["automation_terminal"] is True
    assert snapshot.values["disposition"] == "terminal"
    assert snapshot.next == ()
    assert snapshot.tasks == ()
