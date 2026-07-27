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
model prose. The engine consumes completed node updates, not provider token chunks, so text
speaks only once the completed message is known to carry no tool calls and an approved,
unambiguous source (`_TurnSpeech`). A tool-call message's narration is dropped outright:
the downstream node (guardrail decline, readback, outcome line) is the one author of what
the caller hears, so narration can never contradict the validation that runs after it
(live call #9 P2: "shall I refund?" streamed over the guardrail's return-first decline).
ToolMessages never surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agnostic_market.agents.lifecycle import PrincipalTransitionLifecycle
from agnostic_market.agents.recovery import (
    AUTOMATION_TERMINAL_LINE,
    TURN_FALLBACK_AUTHOR,
    TURN_FALLBACK_LINE,
    NodeExecutionTracker,
    clear_automation_state,
)
from agnostic_market.agents.telemetry import write_event
from agnostic_market.dtos.events import (
    CommittedTurn,
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
from agnostic_market.dtos.recovery import PendingRecovery
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
    merge_consumed_turn_ids,
)

logger = logging.getLogger("agnostic_market.agents.engine")


def _new_thread_id() -> str:
    return uuid.uuid4().hex


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
    PendingRecovery,
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
      - `ttf_model`: to the completed model update. The historical key is retained for log
        continuity, but update-only streaming cannot observe provider generation-start;
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
        cancellation_quiescence_timeout_seconds: float,
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
        if cancellation_quiescence_timeout_seconds <= 0:
            raise ValueError("cancellation quiescence timeout must be positive")
        self._cancellation_quiescence_timeout_seconds = cancellation_quiescence_timeout_seconds
        terminal_node = getattr(graph, "terminal_takeover_node", None)
        if not isinstance(terminal_node, str) or not terminal_node:
            raise ValueError("ReasoningEngine requires a graph-declared terminal takeover node")
        self._terminal_takeover_node = terminal_node
        principal_seed_node = getattr(graph, "principal_seed_complete_node", None)
        if not isinstance(principal_seed_node, str) or not principal_seed_node:
            raise ValueError("ReasoningEngine requires a graph-declared principal seed node")
        self._principal_seed_complete_node = principal_seed_node
        execution_tracker = getattr(graph, "node_execution_tracker", None)
        if not isinstance(execution_tracker, NodeExecutionTracker):
            raise ValueError("ReasoningEngine requires a graph-declared node execution tracker")
        self._node_execution_tracker = execution_tracker
        if lifecycle is not None:
            lifecycle.attach_execution_quiescence(execution_tracker)
        self._terminal_latched = False

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def pending_interrupt(self) -> bool:
        """True if the thread is paused at a HITL interrupt (next turn must resume it)."""
        if self._terminal_latched:
            return False
        return bool(self._graph.get_state(self._config).interrupts)

    def _rotate_pending_transition(
        self,
        arriving_message_id: str,
    ) -> PrincipalTransition | None:
        if self._lifecycle is None:
            return None
        inspection = self._lifecycle.inspect_principal_transition()
        if inspection.outcome == "none":
            return None
        transition = inspection.transition
        assert transition is not None
        current = self._graph.get_state(self._config)
        current_state = ReasoningState.model_validate(current.values)
        if current_state.automation_terminal:
            self._lifecycle.invalidate_principal_transition(transition.transition_id)
            return None
        if inspection.outcome != "coherent":
            self._lifecycle.invalidate_principal_transition(transition.transition_id)
            raise RuntimeError("principal transition is inconsistent")

        old_thread_id = self._thread_id
        new_thread_id = _new_thread_id()
        new_config = {"configurable": {"thread_id": new_thread_id}}
        carried_turn_ids = merge_consumed_turn_ids(
            current_state.consumed_turn_ids,
            (arriving_message_id,),
        )
        switched = False
        try:
            seed: dict[str, object] = {"consumed_turn_ids": carried_turn_ids}
            if transition.continuation is not None:
                seed["pending_request"] = transition.continuation
                seed_author = "__start__"
            else:
                seed_author = self._principal_seed_complete_node
            self._graph.update_state(new_config, seed, as_node=seed_author)
            seeded = self._graph.get_state(new_config)
            seeded_state = ReasoningState.model_validate(seeded.values)
            expected_clear = clear_automation_state()
            expected_request = transition.continuation
            expected_next = ("entry",) if transition.continuation is not None else ()
            contaminated = bool(
                seeded_state.pending_request != expected_request
                or seeded_state.consumed_turn_ids != carried_turn_ids
                or seeded_state.messages
                or seeded_state.automation_terminal
                or seeded.next != expected_next
                or any(
                    getattr(seeded_state, field) != expected
                    for field, expected in expected_clear.items()
                    if field != "pending_request"
                )
            )
            if contaminated:
                raise RuntimeError("fresh principal thread seed is contaminated")
            rechecked = self._lifecycle.inspect_principal_transition()
            if rechecked.outcome != "coherent" or rechecked.transition != transition:
                raise RuntimeError("principal transition changed during rotation")
            self._graph.checkpointer.delete_thread(old_thread_id)
            self._thread_id = new_thread_id
            self._config = new_config
            switched = True
            self._lifecycle.complete_transition(transition.transition_id)
        except Exception:
            self._lifecycle.invalidate_principal_transition(transition.transition_id)
            if not switched:
                try:
                    self._graph.checkpointer.delete_thread(new_thread_id)
                except Exception:
                    logger.critical(
                        "failed to discard an incomplete principal-rotation seed",
                        exc_info=True,
                    )
            raise
        try:
            write_event(
                {
                    "event": "reasoning_context_rotated",
                    "transition_id": transition.transition_id,
                }
            )
        except Exception:
            logger.critical("principal rotation telemetry failed", exc_info=True)
        return transition

    def _invalidate_pending_transition_for_terminal(self) -> None:
        if self._lifecycle is None:
            return
        try:
            inspection = self._lifecycle.inspect_principal_transition()
            if inspection.transition is not None:
                self._lifecycle.invalidate_principal_transition(inspection.transition.transition_id)
        except Exception:
            logger.critical(
                "failed to inspect principal transition during terminal takeover",
                exc_info=True,
            )
            try:
                self._lifecycle.invalidate_principal_transition()
            except Exception:
                logger.critical(
                    "failed to invalidate caller authority during terminal takeover",
                    exc_info=True,
                )

    def _enter_last_resort(self) -> SpokenMessageEvent:
        self._terminal_latched = True
        self._invalidate_pending_transition_for_terminal()
        try:
            write_event({"event": "turn_failed", "reason": "engine_last_resort"})
        except Exception:
            logger.critical("last-resort telemetry failed", exc_info=True)
        try:
            self._graph.update_state(
                self._config,
                {
                    **clear_automation_state(),
                    "automation_terminal": True,
                    "messages": [AIMessage(AUTOMATION_TERMINAL_LINE)],
                },
                as_node=self._terminal_takeover_node,
            )
        except Exception:
            logger.critical("last-resort checkpoint takeover failed", exc_info=True)
        return SpokenMessageEvent(
            text=AUTOMATION_TERMINAL_LINE,
            node=self._terminal_takeover_node,
        )

    async def stream_turn(
        self,
        turn: CommittedTurn,
        facts: TurnFacts,
    ) -> AsyncIterator[TurnEvent]:
        """Run one committed user turn; yield the caller-audible events it produces.

        Registered ordinary node failures recover inside the graph. This outer boundary is
        the last resort for the six deliberately unhandled effect/principal nodes and for
        recovery-infrastructure failure; it logs, telemeters, and prevents dead air without
        claiming that an unfinished checkpoint is safe to abandon.
        """
        if self._terminal_latched:
            yield SpokenMessageEvent(
                text=AUTOMATION_TERMINAL_LINE,
                node=self._terminal_takeover_node,
            )
            return
        message_id = turn.message_id
        if message_id is None:
            try:
                write_event({"event": "ingress_turn_rejected", "reason": "missing_message_id"})
            except Exception:
                logger.critical("missing-message-id telemetry failed", exc_info=True)
            yield SpokenMessageEvent(
                text=TURN_FALLBACK_LINE,
                node=TURN_FALLBACK_AUTHOR,
            )
            return
        arrived_thread_id = self._thread_id
        try:
            arrived = self._graph.get_state(self._config)
        except Exception:
            logger.exception("initial checkpoint read failed; terminalizing the session")
            yield self._enter_last_resort()
            return
        arrived_during_continuation = bool(arrived.interrupts or arrived.next)
        async with self._turn_lock:
            try:
                if self._terminal_latched:
                    yield SpokenMessageEvent(
                        text=AUTOMATION_TERMINAL_LINE,
                        node=self._terminal_takeover_node,
                    )
                    return
                transition = self._rotate_pending_transition(message_id)
                snapshot = self._graph.get_state(self._config)
                snapshot_state = ReasoningState.model_validate(snapshot.values)
                if arrived_thread_id != self._thread_id:
                    write_event({"event": "cross_thread_turn_consumed"})
                    if not snapshot.next:
                        self._graph.update_state(
                            self._config,
                            {"consumed_turn_ids": (message_id,)},
                            as_node=self._principal_seed_complete_node,
                        )
                        yield SpokenMessageEvent(
                            text=TURN_FALLBACK_LINE,
                            node=TURN_FALLBACK_AUTHOR,
                        )
                        return
                    # A Command carrying only an update advances an unfinished node without
                    # supplying resume data. On an interrupt LangGraph checkpoints the ID and
                    # re-emits the same interrupt; on a non-interrupt continuation it runs the
                    # exact pending node. This avoids update_state(), whose inferred author can
                    # rewrite the pending route.
                    payload: object = Command(update={"consumed_turn_ids": (message_id,)})
                elif message_id in snapshot_state.consumed_turn_ids:
                    write_event({"event": "duplicate_turn_ignored", "reason": "consumed_turn_id"})
                    return
                elif (
                    transition is None
                    and arrived_during_continuation
                    and not (snapshot.interrupts or snapshot.next)
                ):
                    write_event({"event": "duplicate_turn_ignored", "reason": "continuation_done"})
                    return
                elif snapshot.interrupts:
                    # The graph's confirm node classifies consent; the engine only relays facts.
                    payload = Command(
                        resume={
                            "text": turn.text,
                            "readback_interrupted": facts.readback_interrupted,
                        },
                        update={"consumed_turn_ids": (message_id,)},
                    )
                elif snapshot.next:
                    # A prior attempt died after a checkpointed step, potentially after an
                    # idempotent write. Continue that task with no new message; treating the
                    # caller's repeat as a fresh turn would strand or duplicate the old intent.
                    payload = Command(update={"consumed_turn_ids": (message_id,)})
                    write_event({"event": "turn_recovering", "reason": "unfinished_checkpoint"})
                else:
                    payload = {
                        "messages": [HumanMessage(content=turn.text, id=message_id)],
                        "consumed_turn_ids": (message_id,),
                    }
                speech = _TurnSpeech(self._speakable, self._model_speech)
                spans = _GraphSpans()
                while True:
                    # LangGraph 1.2.7 re-raises an already-handled node exception when
                    # `messages` participates in an async stream. Its update-only path
                    # completes the handler correctly, so speech is relayed from the same
                    # completed message deltas that are committed to state.
                    async for item in self._graph.astream(
                        payload,
                        config=self._config,
                        stream_mode="updates",
                    ):
                        if "__interrupt__" in item:
                            yield InterruptEvent(prompt=str(item["__interrupt__"][0].value))
                            continue
                        for node, update in item.items():
                            if not isinstance(update, dict):
                                continue
                            messages = update.get("messages", ())
                            if not isinstance(messages, list | tuple):
                                messages = (messages,)
                            meta = {"langgraph_node": node}
                            for token in messages:
                                spans.observe(token, meta)
                                if (event := speech.feed(token, meta)) is not None:
                                    yield event
                    transition = self._rotate_pending_transition(message_id)
                    if transition is None or transition.continuation is None:
                        break
                    payload = None
            except Exception:
                logger.exception("unhandled turn failure; terminalizing the session")
                yield self._enter_last_resort()
                return
            finally:
                if not self._terminal_latched:
                    try:
                        self._rotate_pending_transition(message_id)
                    except Exception:
                        logger.exception("close-stream principal rotation failed")
                        # Generator close has no remaining consumer for a yielded event.
                        # Last resort persists and latches the terminal response for a later turn.
                        self._enter_last_resort()
            for flushed in speech.flush():
                yield flushed
            spans.log()

    def delete_thread(self) -> None:
        """Remove this session's thread from the checkpointer.

        The engine EXPOSES the reap but never decides when — session lifecycle (Clock B)
        belongs to the voice plane's teardown hook.
        """
        self._graph.checkpointer.delete_thread(self._thread_id)
