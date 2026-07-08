"""ReasoningEngine — the Plane-2 seam (AGENTS §A0): run a committed turn against a
durable thread, yielding typed TurnEvents.

ONE coherent job, deliberately narrow:
  - thread management (one thread_id per session, checkpointer-backed);
  - dispatch: a fresh turn is fed as a DELTA (`{"messages": [<new user message>]}` — the
    thread's checkpoint carries history; feeding full transport history would duplicate
    it, verified 2026-07-08), a turn arriving while the graph is paused at an interrupt
    resumes it with `Command(resume={text, readback_interrupted})`;
  - interrupt detection (the `__interrupt__` update) → an InterruptEvent wrapping the
    GRAPH-authored payload.

What it is NOT (the seam's honesty): it authors no text (one-author rule — the graph
writes every spoken string; this engine relays), and it imports NOTHING from the voice
plane — perception facts (§4a readback_interrupted) are passed IN via TurnFacts by the
caller. That one-way dependency (voice -> engine) is what makes the engine mockable, which
is the seam's documented purpose.

Speakable filtering: the compiled graph carries `speakable_nodes` (single source of
truth, stashed at build); node-authored AIMessages become SpokenMessageEvents only for
those nodes — the model node's own non-streamed echo and ToolMessages never surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agnostic_market.dtos.events import (
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnEvent,
    TurnFacts,
)


class ReasoningEngine:
    """Per-session engine over a compiled, checkpointer-backed reasoning graph."""

    def __init__(self, graph: CompiledStateGraph, *, thread_id: str) -> None:
        if graph.checkpointer is None:
            raise ValueError(
                "ReasoningEngine requires a checkpointer-backed graph "
                "(interrupt/resume needs a durable thread)"
            )
        self._graph = graph
        self._thread_id = thread_id
        self._config = {"configurable": {"thread_id": thread_id}}
        self._speakable: frozenset[str] = getattr(graph, "speakable_nodes", frozenset())

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def pending_interrupt(self) -> bool:
        """True if the thread is paused at a HITL interrupt (next turn must resume it)."""
        return bool(self._graph.get_state(self._config).interrupts)

    async def stream_turn(self, user_text: str, facts: TurnFacts) -> AsyncIterator[TurnEvent]:
        """Run one committed user turn; yield the caller-audible events it produces."""
        if self.pending_interrupt():
            # The graph's confirm node classifies consent; the engine only relays facts.
            payload: object = Command(
                resume={"text": user_text, "readback_interrupted": facts.readback_interrupted}
            )
        else:
            payload = {"messages": [HumanMessage(user_text)]}
        streamed_ids: set[str] = set()
        chunk_buffer: list[str] = []
        async for mode, item in self._graph.astream(
            payload, config=self._config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                token, meta = item
                if isinstance(token, AIMessageChunk):
                    if token.id:
                        streamed_ids.add(token.id)
                    text = str(token.text)
                    if text:
                        chunk_buffer.append(text)
                        yield TokenEvent(text=text)
                elif isinstance(token, AIMessage):
                    node = meta.get("langgraph_node") if isinstance(meta, dict) else None
                    text = str(token.text)
                    if not text:
                        continue
                    if node in self._speakable:
                        # Node-authored caller-facing line (deferral, checkout outcome) —
                        # never streamed, always spoken.
                        yield SpokenMessageEvent(text=text, node=node)
                    elif token.id not in streamed_ids and text not in "".join(chunk_buffer):
                        # A model answer that did NOT stream (non-streaming provider or
                        # test fake): speak it once. Already-streamed answers are echoes
                        # (the 3a double-speak class) — deduped by id AND by text.
                        yield TokenEvent(text=text)
            elif mode == "updates" and "__interrupt__" in item:
                yield InterruptEvent(prompt=str(item["__interrupt__"][0].value))

    def delete_thread(self) -> None:
        """Remove this session's thread from the checkpointer.

        The engine EXPOSES the reap but never decides when — session lifecycle (Clock B)
        belongs to the voice plane's teardown hook.
        """
        self._graph.checkpointer.delete_thread(self._thread_id)
