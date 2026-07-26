"""LangGraph 1.2.7 contracts required by the engine failure-lifecycle design."""

from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from agnostic_market.agents.engine import build_checkpointer


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
    visited: Annotated[list[str], _append]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())

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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())
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

    def recover(state: _ContractState) -> Command:
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
        return Command(
            update={
                "pending_recovery": None,
                "prior_intent": None,
                "disposition": "consumed",
                "visited": ["recover"],
            },
            goto=END,
        )

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
    return builder.compile(checkpointer=build_checkpointer())


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


def test_start_seeded_reconcile_consumes_fresh_message_before_normal_routing() -> None:
    graph = _recovery_contract_graph()
    config = _config("reconcile-consumption")
    graph.update_state(
        config,
        {"pending_recovery": "reconcile", "prior_intent": "refund order"},
        as_node=START,
    )

    result = graph.invoke({"messages": [HumanMessage("never mind")]}, config)
    snapshot = graph.get_state(config)

    assert result["disposition"] == "consumed"
    assert result.get("normal_count", 0) == 0
    assert result.get("effect_count", 0) == 0
    assert result["visited"] == ["entry", "recover"]
    assert _human_texts(snapshot) == ("never mind",)
    assert snapshot.next == ()


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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())
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
