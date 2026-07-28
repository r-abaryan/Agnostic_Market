"""LangGraph 1.2.7 contracts required by the engine failure-lifecycle design."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from livekit.agents.llm import ChatContext
from livekit.plugins.langchain import LLMAdapter

from agnostic_market.agents.engine import build_checkpointer
from agnostic_market.dtos.state import merge_consumed_turn_ids

_WAIT_TIMEOUT_SECONDS = 5.0


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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())

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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    graph = builder.compile(checkpointer=build_checkpointer())
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
    builder.add_node("recover", recover)
    builder.add_node("normal", normal)
    builder.add_node("old_work", old_work)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        route_after_entry,
        {"recover": "recover", "old_work": "old_work", "normal": "normal"},
    )
    builder.add_edge("recover", END)
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
    graph = builder.compile(checkpointer=build_checkpointer())
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


def test_interrupt_update_without_resume_consumes_id_and_preserves_interrupt() -> None:
    def confirm(_state: _ContractState) -> dict[str, object]:
        decision = interrupt({"ask": "continue?"})
        return {"disposition": str(decision)}

    builder = StateGraph(_ContractState)
    builder.add_node("confirm", confirm)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    graph = builder.compile(checkpointer=build_checkpointer())
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
