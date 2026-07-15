"""GraphVoiceAdapter — the Plane-1 side of the seam, tested against a SCRIPTED fake
engine (the §A0 mockability promise, proven: no graph, no LiveKit session, no network)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.voice.graph import GraphVoiceAdapter


class ScriptedEngine:
    """ReasoningEngine double: replays canned TurnEvents, records what it was asked."""

    def __init__(self, events: list, *, pending: bool = False) -> None:
        self._events = events
        self._pending = pending
        self.calls: list[tuple[str, TurnFacts]] = []

    def pending_interrupt(self) -> bool:
        return self._pending

    async def stream_turn(self, user_text: str, facts: TurnFacts):
        self.calls.append((user_text, facts))
        for event in self._events:
            yield event


class _FakeHistoryItem:
    def __init__(self, role: str, interrupted: bool) -> None:
        self.type = "message"
        self.role = role
        self.interrupted = interrupted


class _FakeSession:
    """AgentSession double: just the history surface the adapter reads."""

    def __init__(self, items: list[_FakeHistoryItem]) -> None:
        class _History:
            pass

        self.history = _History()
        self.history.items = items


async def _spoken(adapter: GraphVoiceAdapter, state: dict) -> list[str]:
    return [text async for text in adapter.astream(state, None)]


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
    out = await _spoken(adapter, {"messages": [HumanMessage("where is it")]})
    assert out == [
        "Your order ",
        "shipped.",
        "I'll pass it to support.",
        "2 x rain jacket, $258.00 total. Shall I place it?",
    ]


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
                HumanMessage("first turn"),
                AIMessage("answer"),
                HumanMessage("second turn"),
            ]
        },
    )
    assert [call[0] for call in engine.calls] == ["second turn"]


async def test_empty_transport_input_is_an_empty_turn() -> None:
    engine = ScriptedEngine([TokenEvent(text="should not appear")])
    adapter = GraphVoiceAdapter(engine)
    assert await _spoken(adapter, {"messages": []}) == []
    assert engine.calls == []  # engine never invoked


async def test_4a_fact_set_when_pending_and_readback_barged() -> None:
    engine = ScriptedEngine([], pending=True)
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(
        _FakeSession(
            [
                _FakeHistoryItem("user", interrupted=False),
                _FakeHistoryItem("assistant", interrupted=True),  # the barged readback
            ]
        )
    )
    await _spoken(adapter, {"messages": [HumanMessage("yes")]})
    assert engine.calls[0][1].readback_interrupted is True


async def test_4a_fact_false_when_no_pending_interrupt() -> None:
    # An interrupted PAST utterance is irrelevant on a normal turn: the fact is only
    # asserted while the engine is paused at a confirmation.
    engine = ScriptedEngine([], pending=False)
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(_FakeSession([_FakeHistoryItem("assistant", interrupted=True)]))
    await _spoken(adapter, {"messages": [HumanMessage("what's the status")]})
    assert engine.calls[0][1].readback_interrupted is False


async def test_4a_fact_false_when_readback_played_out() -> None:
    engine = ScriptedEngine([], pending=True)
    adapter = GraphVoiceAdapter(engine)
    adapter.attach_session(_FakeSession([_FakeHistoryItem("assistant", interrupted=False)]))
    await _spoken(adapter, {"messages": [HumanMessage("yes")]})
    assert engine.calls[0][1].readback_interrupted is False


async def test_unconsumed_turn_never_reaches_the_engine() -> None:
    # The interim-transcript discard path (live call #9 P1 family): when the voice layer
    # creates a turn but CANCELS it before consuming any output (a superseded/discarded
    # generation), the engine — and therefore the stateful graph and its interrupts — must
    # not have run at all. astream is lazy: creating the iterator is not execution.
    engine = ScriptedEngine([TokenEvent(text="never spoken")])
    adapter = GraphVoiceAdapter(engine)
    adapter.astream({"messages": [HumanMessage("yes")]}, None)  # created, never iterated
    assert engine.calls == []  # nothing reached the engine
    # The next, consumed turn runs normally against untouched state.
    spoken = await _spoken(adapter, {"messages": [HumanMessage("no, wait")]})
    assert engine.calls == [("no, wait", TurnFacts(readback_interrupted=False))]
    assert spoken == ["never spoken"]
