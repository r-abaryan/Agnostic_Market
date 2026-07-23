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

Speakable filtering + buffer-before-speak: the compiled graph carries two disjoint source
sets — `speakable_nodes` for code-authored lines and `model_speech_nodes` for approved
model prose. Model tokens are NEVER relayed as they stream: they buffer per message and
speak only once the completed message is known to carry no tool calls and an approved,
unambiguous source (`_TurnSpeech`). A tool-call message's narration is dropped outright:
the downstream node (guardrail decline, readback, outcome line) is the one author of what
the caller hears, so streamed narration can never contradict the validation that runs
after it (live call #9 P2: "shall I refund?" streamed over the guardrail's return-first
decline). ToolMessages never surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agnostic_market.agents.telemetry import write_event
from agnostic_market.dtos.events import (
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnEvent,
    TurnFacts,
)
from agnostic_market.dtos.orchestration import (
    AnswerQuestion,
    CancellableOrderScope,
    CancelOrders,
    CapabilityId,
    ChangeProfile,
    ExplicitOrderSet,
    ExplicitOrderTarget,
    FocusedOrderSet,
    FocusedOrderTarget,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    PrincipalTransition,
    RecentOrderSet,
    RefundOrder,
    RequestPerson,
    ReturnOrder,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
)
from agnostic_market.dtos.state import (
    BatchCancelOutcome,
    CancelTarget,
    CartClarification,
    CartLine,
    ClarificationProgress,
    HandoffRequest,
    IdentityClarification,
    PendingCancelBatch,
    PendingIdentity,
    PendingPlacement,
    PendingProfileChange,
    PendingRefund,
    PendingReturn,
    ReasoningState,
    SupportClarification,
)

logger = logging.getLogger("agnostic_market.agents.engine")


class PrincipalTransitionLifecycle(Protocol):
    def pending_transition(self) -> PrincipalTransition | None: ...

    def complete_transition(self, transition_id: str) -> None: ...


# The custom (non-message) types we checkpoint into graph state. langgraph's default
# msgpack serde is permissive (deserializes anything with a warning), but that path is
# slated to be BLOCKED — an explicit allowlist registers our DTOs as trusted so the
# checkpoint roundtrip is future-proof AND stops trusting arbitrary types (the security
# posture the warning is nudging toward). langchain messages stay covered by the built-in
# safe types; this ADDS ours. One source of truth — pipeline + tests build via this.
# PendingPlacement embeds a tuple[CartLine, ...], so CartLine must be registered too (the
# serde allowlists by MODULE — nested custom types are checked independently).
_CHECKPOINTED_DTOS = (
    PendingPlacement,
    CartLine,
    PendingRefund,
    # The cancel lifecycle (F-16.2): the batch + its nested targets/outcomes are each checked
    # independently by the serde allowlist (like CartLine inside PendingPlacement), and the
    # pre-auth selector is its own checkpointed type.
    CancellableOrderScope,
    CapabilityId,
    PendingCancelBatch,
    CancelTarget,
    BatchCancelOutcome,
    PendingReturn,
    PendingProfileChange,
    PendingIdentity,
    IdentityClarification,
    SupportClarification,
    CartClarification,
    ClarificationProgress,
    HandoffRequest,
    ReasoningState,
    AnswerQuestion,
    SearchCatalog,
    VerifyOrderStatus,
    ExplicitOrderSet,
    FocusedOrderSet,
    RecentOrderSet,
    ListOrders,
    ViewCart,
    ModifyCart,
    PlaceOrder,
    CancelOrders,
    ExplicitOrderTarget,
    FocusedOrderTarget,
    RefundOrder,
    ReturnOrder,
    ChangeProfile,
    VerifyIdentity,
    SwitchAccount,
    RequestPerson,
)


# The turn-failure fallback (F-13.1). EXCEPTION to the one-author rule (the graph authors
# all caller-facing text): when the GRAPH ITSELF is what died, there is no graph author —
# a fixed platform failure line is the only alternative to dead air. Honest (admits the
# hiccup, asks to repeat), promises nothing.
_TURN_FALLBACK_LINE = "Sorry - I hit a snag on my end just now. Could you say that again?"
_TURN_FALLBACK_NODE = "turn_fallback"  # not a graph node; names the author in the event


def build_checkpointer() -> InMemorySaver:
    """An InMemorySaver whose serde trusts our checkpointed DTOs (no 'unregistered type'
    warning, and not silently permissive to arbitrary types). Redis swap at Phase 4 wires
    the same allowlist into its serde."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=list(_CHECKPOINTED_DTOS)))


@dataclass
class _MsgBuffer:
    """Accumulated stream state for one in-flight model message."""

    parts: list[str] = field(default_factory=list)
    has_tool_calls: bool = False
    node: str | None = None
    node_conflict: bool = False

    @property
    def text(self) -> str:
        return "".join(self.parts)


class _TurnSpeech:
    """Buffer-before-speak (live call #9 P2). Model tokens never reach the caller as they
    stream: chunks accumulate per message id, and text speaks only once the COMPLETED
    message is known to carry no tool calls. A tool-call message's narration is dropped —
    the downstream node (guardrail decline, readback, outcome line) is the one author of
    what the caller hears, so streamed narration can never contradict the validation that
    runs after it. Cost accepted by decision: clarify turns speak at message completion
    instead of first token.

    Code-authored AIMessages speak only from `speakable`; model prose speaks only from
    `model_speech`. Unknown, missing, or conflicting provenance fails closed. `flush()`
    covers a streamed message whose completed AIMessage never arrived, but only when its
    buffered source is unambiguous and model-speakable. Text-level dedup guards an id
    change between a chunk and its completed message from double-speaking.
    """

    def __init__(self, speakable: frozenset[str], model_speech: frozenset[str]) -> None:
        overlap = speakable & model_speech
        if overlap:
            raise ValueError(f"speech source sets overlap: {sorted(overlap)!r}")
        self._speakable = speakable
        self._model_speech = model_speech
        self._buffers: dict[str | None, _MsgBuffer] = {}
        self._spoken_texts: set[str] = set()

    @staticmethod
    def _node(meta: object) -> str | None:
        value = meta.get("langgraph_node") if isinstance(meta, dict) else None
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _source_matches(buf: _MsgBuffer | None, node: str | None) -> bool:
        return bool(
            node is not None
            and (buf is None or (not buf.node_conflict and buf.node in (None, node)))
        )

    def feed(self, token: object, meta: object) -> TurnEvent | None:
        """Consume one `messages`-mode stream item; return the event to speak, if any."""
        if isinstance(token, AIMessageChunk):
            buf = self._buffers.setdefault(token.id, _MsgBuffer())
            node = self._node(meta)
            if node is not None:
                if buf.node is None:
                    buf.node = node
                elif buf.node != node:
                    buf.node_conflict = True
            if token.tool_call_chunks:
                buf.has_tool_calls = True
            if text := str(token.text):
                buf.parts.append(text)
            return None
        if not isinstance(token, AIMessage):
            return None  # ToolMessages and anything else never surface
        buf = self._buffers.pop(token.id, None)
        text = str(token.text)
        node = self._node(meta)
        # Tool narration is never authoritative, even if metadata accidentally labels the
        # message as a code-speakable node. Suppression therefore precedes source checks.
        if token.tool_calls or (buf is not None and buf.has_tool_calls):
            return None  # tool-call message: narration dropped; the next node speaks
        if not self._source_matches(buf, node):
            return None
        if node in self._speakable:
            return SpokenMessageEvent(text=text, node=node) if text else None
        if node in self._model_speech and text:
            self._spoken_texts.add(text)
            return TokenEvent(text=text)
        return None

    def flush(self) -> Iterator[TokenEvent]:
        """Speak only unambiguous, approved model buffers lacking a completed message."""
        for buf in self._buffers.values():
            text = buf.text
            if (
                text
                and not buf.has_tool_calls
                and not buf.node_conflict
                and buf.node in self._model_speech
                and text not in self._spoken_texts
            ):
                yield TokenEvent(text=text)
        self._buffers.clear()


class _GraphSpans:
    """Graph-internal latency timeline for one turn. The voice plane's `llm_node_ttft`/
    `e2e_latency` metrics treat the whole graph as one opaque "LLM", so they cannot say
    whether a slow turn is model generation, a second model pass after a tool read, or
    graph orchestration (live call #10). This measures the boundaries the graph stream
    exposes, from turn start:
      - `ttf_model`: to the first model output (chunk or message) — generation-start delay;
      - `tools`: how many tool calls ran (0 = single pass; >=1 = a read/mutation loop);
      - `tool_to_next_model`: from the last tool result to the model activity AFTER it — the
        cost of the second reasoning pass that renders a tool result into speech. STAYS None
        when a deterministic read renderer (the `read_render` node) authored the post-tool
        line INSTEAD of a second model pass (L3) — only the `model` node runs the LLM, so a
        node-authored AIMessage is not counted as model activity;
      - `total`: whole in-graph time.
    Passive: it only observes the stream the engine already consumes; it authors nothing and
    speaks nothing (the one-author rule holds). Logged at DEBUG — a telemetry backend is
    Phase 6; this is the diagnostic the renderer-bypass decision (T2 latency pass) needs.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._ttf_model: float | None = None
        self._tool_count = 0
        self._last_tool_at: float | None = None
        self._tool_to_next_model: float | None = None

    def observe(self, token: object, meta: object) -> None:
        now = time.perf_counter()
        if isinstance(token, ToolMessage):
            self._tool_count += 1
            self._last_tool_at = now
            return
        # Only the `model` node runs the LLM; a node-authored AIMessage (a read render, a
        # guardrail decline, a readback) is NOT a model pass and must not be timed as one —
        # else L3's rendered turns would show a phantom second-model-pass cost.
        node = meta.get("langgraph_node") if isinstance(meta, dict) else None
        is_model = (
            node == "model"
            and isinstance(token, AIMessageChunk | AIMessage)
            and (bool(str(token.text)) or bool(getattr(token, "tool_calls", None)))
        )
        if not is_model:
            return
        if self._ttf_model is None:
            self._ttf_model = now - self._start
        if self._last_tool_at is not None and self._tool_to_next_model is None:
            self._tool_to_next_model = now - self._last_tool_at

    def log(self) -> None:
        parts = [f"total={time.perf_counter() - self._start:.3f}s", f"tools={self._tool_count}"]
        if self._ttf_model is not None:
            parts.append(f"ttf_model={self._ttf_model:.3f}s")
        if self._tool_to_next_model is not None:
            parts.append(f"tool_to_next_model={self._tool_to_next_model:.3f}s")
        logger.debug("graph spans: %s", ", ".join(parts))


class ReasoningEngine:
    """Per-session engine over a compiled, checkpointer-backed reasoning graph."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        thread_id: str,
        lifecycle: PrincipalTransitionLifecycle | None = None,
    ) -> None:
        if graph.checkpointer is None:
            raise ValueError(
                "ReasoningEngine requires a checkpointer-backed graph "
                "(interrupt/resume needs a durable thread)"
            )
        self._graph = graph
        self._thread_id = thread_id
        self._config = {"configurable": {"thread_id": thread_id}}
        self._speakable: frozenset[str] = getattr(graph, "speakable_nodes", frozenset())
        self._model_speech: frozenset[str] = getattr(graph, "model_speech_nodes", frozenset())
        # A session is a serial conversation. A waiter re-checks the checkpoint so two
        # concurrent deliveries of one resume cannot both produce final result speech.
        self._turn_lock = asyncio.Lock()
        self._lifecycle = lifecycle

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def pending_interrupt(self) -> bool:
        """True if the thread is paused at a HITL interrupt (next turn must resume it)."""
        return bool(self._graph.get_state(self._config).interrupts)

    def _rotate_pending_transition(self) -> PrincipalTransition | None:
        if self._lifecycle is None:
            return None
        transition = self._lifecycle.pending_transition()
        if transition is None:
            return None
        old_thread_id = self._thread_id
        self._graph.checkpointer.delete_thread(old_thread_id)
        self._thread_id = uuid.uuid4().hex
        self._config = {"configurable": {"thread_id": self._thread_id}}
        if transition.continuation is not None:
            self._graph.update_state(
                self._config,
                {"pending_request": transition.continuation},
                as_node="__start__",
            )
        self._lifecycle.complete_transition(transition.transition_id)
        write_event(
            {
                "event": "reasoning_context_rotated",
                "transition_id": transition.transition_id,
            }
        )
        return transition

    async def stream_turn(self, user_text: str, facts: TurnFacts) -> AsyncIterator[TurnEvent]:
        """Run one committed user turn; yield the caller-audible events it produces.

        A turn that DIES mid-graph (provider outage, a bug) must never leave the caller in
        silence (live call #13 F-13.1: an Anthropic 529 survived the SDK retries, the turn
        errored into a log line, and the caller sat in dead air until they re-asked). The
        failure boundary here logs + telemeters loudly, then speaks ONE fixed fallback line
        and ends the turn cleanly — the thread stays usable (nothing mid-node was
        committed; a pending interrupt stays paused and the next turn resumes it).
        """
        arrived_thread_id = self._thread_id
        arrived = self._graph.get_state(self._config)
        arrived_during_continuation = bool(arrived.interrupts or arrived.next)
        async with self._turn_lock:
            transition = self._rotate_pending_transition()
            snapshot = self._graph.get_state(self._config)
            if (
                transition is None
                and arrived_thread_id == self._thread_id
                and arrived_during_continuation
                and not (snapshot.interrupts or snapshot.next)
            ):
                write_event({"event": "duplicate_turn_ignored", "reason": "continuation_done"})
                return
            if snapshot.interrupts:
                # The graph's confirm node classifies consent; the engine only relays facts.
                payload: object = Command(
                    resume={
                        "text": user_text,
                        "readback_interrupted": facts.readback_interrupted,
                    }
                )
            elif snapshot.next:
                # A prior attempt died after a checkpointed step, potentially after an
                # idempotent write. Continue that task with no new message; treating the
                # caller's repeat as a fresh turn would strand or duplicate the old intent.
                payload = None
                write_event({"event": "turn_recovering", "reason": "unfinished_checkpoint"})
            else:
                payload = {"messages": [HumanMessage(user_text)]}
            speech = _TurnSpeech(self._speakable, self._model_speech)
            spans = _GraphSpans()
            try:
                while True:
                    async for mode, item in self._graph.astream(
                        payload, config=self._config, stream_mode=["messages", "updates"]
                    ):
                        if mode == "messages":
                            token, meta = item
                            spans.observe(token, meta)
                            if (event := speech.feed(token, meta)) is not None:
                                yield event
                        elif mode == "updates" and "__interrupt__" in item:
                            yield InterruptEvent(prompt=str(item["__interrupt__"][0].value))
                    transition = self._rotate_pending_transition()
                    if transition is None or transition.continuation is None:
                        break
                    payload = None
            except Exception:
                # The degradation boundary, not error-swallowing: the full traceback is logged
                # and the event telemetered (exception CLASS only — provider error messages are
                # free text, not for telemetry); the caller hears the fallback instead of
                # nothing. CancelledError is BaseException and passes through untouched.
                logger.exception("turn failed mid-graph; speaking the fallback line")
                write_event({"event": "turn_failed", "reason": "graph_exception"})
                yield SpokenMessageEvent(text=_TURN_FALLBACK_LINE, node=_TURN_FALLBACK_NODE)
                return
            finally:
                self._rotate_pending_transition()
            for flushed in speech.flush():
                yield flushed
            spans.log()

    def delete_thread(self) -> None:
        """Remove this session's thread from the checkpointer.

        The engine EXPOSES the reap but never decides when — session lifecycle (Clock B)
        belongs to the voice plane's teardown hook.
        """
        self._graph.checkpointer.delete_thread(self._thread_id)
