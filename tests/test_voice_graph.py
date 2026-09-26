"""GraphVoiceAdapter — the Plane-1 side of the seam, tested against a SCRIPTED fake
engine (the §A0 mockability promise, proven: no graph, no LiveKit session, no network)."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from livekit.agents import Agent, AgentSession
from livekit.agents.voice.io import TextOutput
from livekit.plugins.langchain import LLMAdapter

from agnostic_market.dtos.events import (
    CommittedTurn,
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnFacts,
)
from agnostic_market.voice.graph import GraphVoiceAdapter


class ScriptedEngine:
    """ReasoningEngine double: replays canned TurnEvents, records what it was asked."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.calls: list[tuple[CommittedTurn, TurnFacts]] = []

    async def stream_turn(self, turn: CommittedTurn, facts: TurnFacts):
        self.calls.append((turn, facts))
        for event in self._events:
            yield event


class _FakeHistoryItem:
    def __init__(self, role: str, interrupted: object, item_id: str | None = None) -> None:
        self.type = "message"
        self.role = role
        self.interrupted = interrupted
        self.id = item_id


class _FakeSession:
    """AgentSession double: just the history surface the adapter reads."""

    def __init__(self, items: list[object] | None = None) -> None:
        class _History:
            pass

        self.history = _History()
        self.history.items = items or []


def test_unsupplied_playback_status_is_unknown() -> None:
    assert TurnFacts().readback_interrupted is None


async def _spoken(adapter: GraphVoiceAdapter, state: dict) -> list[str]:
    return [text async for text in adapter.astream(state, None)]


def _turn_after_reply(text: str = "yes") -> dict:
    return {
        "messages": [
            HumanMessage("previous request", id="previous-turn"),
            AIMessage("Previous reply", id="reply-1"),
            HumanMessage(text, id="turn-1"),
        ]
    }


async def test_adapter_renders_all_event_kinds_as_text() -> None:
    engine = ScriptedEngine(
        [
            TokenEvent(text="Your order "),
            TokenEvent(text="shipped."),
            SpokenMessageEvent(text="I'll pass it to support.", node="handover"),
            InterruptEvent(prompt="2 x rain jacket, $258.00 total. Shall I place it?"),
        ]
    )
    adapter = GraphVoiceAdapter(engine)
    out = await _spoken(adapter, {"messages": [HumanMessage("where is it", id="turn-1")]})
    assert out == [
        "Your order ",
        "shipped.",
        "I'll pass it to support.",
        "2 x rain jacket, $258.00 total. Shall I place it?",
    ]


async def test_adapter_reports_the_exact_events_it_renders() -> None:
    events = [
        SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render"),
    ]
    observed: list[object] = []
    adapter = GraphVoiceAdapter(ScriptedEngine(events), turn_event_observer=observed.append)

    assert await _spoken(
        adapter,
        {"messages": [HumanMessage("what is in my cart?", id="turn-1")]},
    ) == ["Your cart is empty."]
    assert observed == events


async def test_diagnostic_observer_failure_does_not_interrupt_caller_output() -> None:
    event = SpokenMessageEvent(text="Your cart is empty.", node="cart_view_render")
    started: list[bool] = []
    failures: list[Exception] = []

    def fail_observation(_event: object) -> None:
        raise RuntimeError("diagnostic observer failed")

    adapter = GraphVoiceAdapter(
        ScriptedEngine([event]),
        turn_started_observer=lambda: started.append(True),
        turn_event_observer=fail_observation,
        observer_failure_observer=failures.append,
    )

    assert await _spoken(
        adapter,
        {"messages": [HumanMessage("what is in my cart?", id="turn-1")]},
    ) == ["Your cart is empty."]
    assert started == [True]
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)


async def test_adapter_feeds_only_the_last_user_turn() -> None:
    # The transport hands FULL history each call; the engine must get just the new turn
    # (the thread checkpoint carries history — delta contract, verified 2026-07-08).
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    await _spoken(
        adapter,
        {
            "messages": [
                SystemMessage("sys"),
                HumanMessage("first turn", id="turn-1"),
                AIMessage("answer"),
                HumanMessage("second turn", id="turn-2"),
            ]
        },
    )
    assert [call[0] for call in engine.calls] == [
        CommittedTurn(text="second turn", message_id="turn-2")
    ]


async def test_empty_transport_input_is_an_empty_turn() -> None:
    engine = ScriptedEngine([TokenEvent(text="should not appear")])
    adapter = GraphVoiceAdapter(engine)
    assert await _spoken(adapter, {"messages": []}) == []
    assert engine.calls == []  # engine never invoked


async def test_adapter_passes_the_transport_interruption_fact_without_checkpoint_io() -> None:
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(
        _FakeSession(
            [
                _FakeHistoryItem("user", interrupted=False, item_id="previous-turn"),
                _FakeHistoryItem("assistant", interrupted=True, item_id="reply-1"),
            ]
        )
    )
    await _spoken(adapter, _turn_after_reply())
    assert engine.calls[0][1].readback_interrupted is True


async def test_transport_interruption_fact_is_perceptual_not_resume_authority() -> None:
    # The adapter reports transport perception only. The engine decides whether the current
    # checkpoint makes that fact relevant to a confirmation resume.
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(_FakeSession([_FakeHistoryItem("assistant", True, "reply-1")]))
    await _spoken(adapter, _turn_after_reply("what's the status"))
    assert engine.calls[0][1].readback_interrupted is True


async def test_4a_fact_false_when_readback_played_out() -> None:
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(_FakeSession([_FakeHistoryItem("assistant", False, "reply-1")]))
    await _spoken(adapter, _turn_after_reply())
    assert engine.calls[0][1].readback_interrupted is False


async def test_skipped_latest_reply_cannot_borrow_older_playback_completion() -> None:
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(
        _FakeSession(
            [
                _FakeHistoryItem("user", False, "older-turn"),
                _FakeHistoryItem("assistant", False, "older-reply"),
                _FakeHistoryItem("user", False, "previous-turn"),
            ]
        )
    )
    await _spoken(
        adapter,
        {
            "messages": [
                HumanMessage("older request", id="older-turn"),
                AIMessage("Older completed reply", id="older-reply"),
                HumanMessage("previous request", id="previous-turn"),
                HumanMessage("yes", id="current-turn"),
            ]
        },
    )

    assert engine.calls[0][1].readback_interrupted is None


async def test_missing_or_invalid_interruption_evidence_fails_closed() -> None:
    class MissingInterruption:
        type = "message"
        role = "assistant"
        id = "reply-1"

    for item in (MissingInterruption(), _FakeHistoryItem("assistant", "false", "reply-1")):
        engine = ScriptedEngine([])
        adapter = GraphVoiceAdapter(engine)
        adapter.attach_session(_FakeSession([item]))

        await _spoken(adapter, _turn_after_reply())

        assert engine.calls[0][1].readback_interrupted is None


async def test_absent_session_history_or_assistant_is_not_playback_completion() -> None:
    for session in (None, object(), _FakeSession([])):
        engine = ScriptedEngine([])
        adapter = GraphVoiceAdapter(engine)
        if session is not None:
            adapter.attach_session(session)

        await _spoken(adapter, _turn_after_reply())

        assert engine.calls[0][1].readback_interrupted is None


async def test_playback_history_reply_id_must_match_transport_reply() -> None:
    engine = ScriptedEngine([])
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(_FakeSession([_FakeHistoryItem("assistant", False, "older-reply")]))

    await _spoken(adapter, _turn_after_reply())

    assert engine.calls[0][1].readback_interrupted is None


async def test_livekit_session_correlates_a_completed_reply_with_the_next_turn() -> None:
    engine = ScriptedEngine([SpokenMessageEvent(text="Anything else?", node="test_reply")])
    adapter = GraphVoiceAdapter(engine)
    session = AgentSession(llm=LLMAdapter(adapter))  # type: ignore[arg-type]
    adapter.attach_session(session)

    await session.start(Agent(instructions=""), record=False)
    try:
        first = session.generate_reply(user_input="first request")
        await asyncio.wait_for(first.wait_for_playout(), timeout=10)
        second = session.generate_reply(user_input="second request")
        await asyncio.wait_for(second.wait_for_playout(), timeout=10)

        assert len(engine.calls) == 2
        assert engine.calls[1][1].readback_interrupted is False
    finally:
        await session.aclose()


async def test_livekit_session_does_not_borrow_status_across_an_extra_reply() -> None:
    engine = ScriptedEngine([SpokenMessageEvent(text="Anything else?", node="test_reply")])
    adapter = GraphVoiceAdapter(engine)
    session = AgentSession(llm=LLMAdapter(adapter))  # type: ignore[arg-type]
    adapter.attach_session(session)

    await session.start(Agent(instructions=""), record=False)
    try:
        first = session.generate_reply(user_input="first request")
        await asyncio.wait_for(first.wait_for_playout(), timeout=10)
        extra = session.say("Unrelated assistant speech.")
        await asyncio.wait_for(extra.wait_for_playout(), timeout=10)
        assistant_items = [
            item for item in session.history.items if getattr(item, "role", None) == "assistant"
        ]
        assert len(assistant_items) == 2
        assert all(item.interrupted is False for item in assistant_items)
        second = session.generate_reply(user_input="second request")
        await asyncio.wait_for(second.wait_for_playout(), timeout=10)

        assert len(engine.calls) == 2
        assert engine.calls[1][1].readback_interrupted is None
    finally:
        await session.aclose()


async def test_livekit_session_correlates_an_interrupted_reply_with_the_next_turn() -> None:
    class CapturedText(TextOutput):
        def __init__(self) -> None:
            super().__init__(label="test-output", next_in_chain=None)
            self.captured = asyncio.Event()

        async def capture_text(self, text: str) -> None:
            if text.strip():
                self.captured.set()

        def flush(self) -> None:
            pass

    class StreamingEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[CommittedTurn, TurnFacts]] = []
            self.streaming = asyncio.Event()

        async def stream_turn(self, turn: CommittedTurn, facts: TurnFacts):
            self.calls.append((turn, facts))
            if len(self.calls) == 1:
                yield SpokenMessageEvent(text="Anything else?", node="test_reply")
                self.streaming.set()
                await asyncio.Future()
            else:
                yield SpokenMessageEvent(text="What can I help with?", node="test_reply")

    engine = StreamingEngine()
    adapter = GraphVoiceAdapter(engine)
    session = AgentSession(llm=LLMAdapter(adapter))  # type: ignore[arg-type]
    adapter.attach_session(session)

    await session.start(Agent(instructions=""), record=False)
    try:
        output = CapturedText()
        session.output.transcription = output
        first = session.generate_reply(user_input="previous request")
        await asyncio.wait_for(engine.streaming.wait(), timeout=10)
        await asyncio.wait_for(output.captured.wait(), timeout=10)
        await asyncio.wait_for(session.interrupt(), timeout=10)
        await asyncio.wait_for(first.wait_for_playout(), timeout=10)
        interrupted = [
            item for item in session.history.items if getattr(item, "role", None) == "assistant"
        ]
        assert len(interrupted) == 1
        assert interrupted[0].interrupted is True
        second = session.generate_reply(user_input="yes")
        await asyncio.wait_for(second.wait_for_playout(), timeout=10)

        assert len(engine.calls) == 2
        assert engine.calls[1][1].readback_interrupted is True
    finally:
        await session.aclose()


async def test_unconsumed_turn_never_reaches_the_engine() -> None:
    # The interim-transcript discard path (live call #9 P1 family): when the voice layer
    # creates a turn but CANCELS it before consuming any output (a superseded/discarded
    # generation), the engine — and therefore the stateful graph and its interrupts — must
    # not have run at all. astream is lazy: creating the iterator is not execution.
    engine = ScriptedEngine([TokenEvent(text="never spoken")])
    adapter = GraphVoiceAdapter(engine)
    adapter.astream(
        {"messages": [HumanMessage("yes", id="turn-1")]},
        None,
    )  # created, never iterated
    assert engine.calls == []  # nothing reached the engine
    # The next, consumed turn runs normally against untouched state.
    spoken = await _spoken(
        adapter,
        {"messages": [HumanMessage("no, wait", id="turn-2")]},
    )
    assert engine.calls == [
        (
            CommittedTurn(text="no, wait", message_id="turn-2"),
            TurnFacts(readback_interrupted=None),
        )
    ]
    assert spoken == ["never spoken"]
