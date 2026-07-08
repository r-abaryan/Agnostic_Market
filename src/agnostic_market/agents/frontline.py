"""The frontline agent (Tier 1) — the fast path that answers or hands over (AGENTS.md §0).

The load-bearing safety invariant: the frontline holds NO state-changing tools. It can
answer (read-only tools) or hand over — so a wrong judgment is at worst "answered without
acting" (recoverable), never a dangerous mutation. Handover fires two ways:
  - the model's `request_handover` tool — the PRIMARY escalation decider (reads intent,
    including paraphrases; industry-consistent LLM-router stance);
  - the deterministic gate (agents/gate.py) — a slim pre-generation floor for high-certainty
    IRREVERSIBLE requests only (cancel/refund/place-order). NOT the router; a bonus fast-path.
Safety is STRUCTURAL (no sensitive tools), not dependent on either catcher hitting 100%.

Graph shape (LangGraph):
    gate -> (trip? handover : model) ; model -> tools -> (handover? handover : model) ; -> END

`request_handover` is a REAL executed tool that returns a `Command` routing to the handover
node — LangGraph's native handoff pattern: proper tool_use/tool_result pairing, no dangling
tool calls. Prompt + few-shot live HERE (inside the graph) so the T1 eval and production run
the identical prompt path.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agnostic_market.agents.gate import gate_check
from agnostic_market.dtos.state import (
    HandoffDestination,
    HandoffReasonCode,
    HandoffRequest,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.frontline")

# Dedicated telemetry sink (F6): utterance + gate/model verdicts, for the future escalation
# classifier. Separate from app logs. PRODUCTION USE REQUIRES the COMPLIANCE consent/
# retention framework before storing real-caller transcripts — fixture-data-only until then.
_TELEMETRY_PATH = Path(__file__).resolve().parents[3] / "config" / "telemetry" / "frontline.jsonl"

_INSTRUCTIONS = (
    "You are the voice assistant for {display_name}. You answer order-status questions "
    "(use order_status) and product questions (use catalog_search). You have NO other "
    "abilities: you cannot change addresses, payment, carts, or orders, and you cannot "
    "place orders, cancel, or issue refunds. When the caller wants any of those, call "
    "request_handover with the right destination and reason_code instead of trying to do "
    "it or saying you did. Keep spoken answers to one or two short sentences."
)

# Contrastive few-shot — PLATFORM safety content (not merchant persona), and strictly
# DISJOINT from the T1 eval set. Pairs teach the answer-vs-handover boundary on near-misses
# the gate cannot pattern (a read that mentions a sensitive noun vs an actual change intent).
_FEW_SHOT: tuple[tuple[str, str], ...] = (
    ("Did my address change go through?", "answer: this is a status read, not a change request"),
    ("Actually, send it to my work address instead.", "handover: address_change / support"),
    ("What's the status of my last order?", "answer: order_status read"),
    ("You know what, cancel that last order.", "handover: cancel_order / support"),
    ("Do you take Amex?", "answer: a question about accepted payment, not a change"),
    ("Put my new Amex on the account.", "handover: payment_change / support"),
    ("What's in my cart right now?", "answer: a cart READ, not a change (you may view the cart)"),
    ("Add the blue jacket to my cart.", "handover: cart_write / checkout"),
)

_HANDOVER_TOOL_NAME = "request_handover"

# Max read-only tool round-trips per turn before the graph ends (loop guard — the
# hand-built graph has no framework loop protection like create_agent had).
_MAX_TOOL_HOPS = 5

# 3a deferral copy — destination-keyed and HONEST: nothing downstream exists yet, so we do
# NOT promise a live connection. Structure carries into 3b/3c where the promises come true.
_DEFERRAL: dict[HandoffDestination, str] = {
    "support": (
        "That's something I'm not able to do on this call yet, but I'll make sure it reaches "
        "our support team."
    ),
    "checkout": (
        "I can't complete that change to your order on this call yet, but I'll pass it along "
        "so it can be handled."
    ),
    "planner": "That's a bit more than I can handle here yet, but I'll make sure it's picked up.",
    "human": "Let me get you to a person who can help with that.",
}


def _compose_system_prompt(display_name: str) -> str:
    """Platform instructions + contrastive few-shot as one SystemMessage body (F1: in-graph)."""
    lines = [_INSTRUCTIONS.format(display_name=display_name), "", "Examples of the boundary:"]
    for utterance, decision in _FEW_SHOT:
        lines.append(f'- Caller: "{utterance}" -> {decision}')
    return "\n".join(lines)


def _build_handover_tool() -> BaseTool:
    """The `request_handover` control tool: executes, returns a Command routing to the sink."""

    @tool(_HANDOVER_TOOL_NAME)
    def request_handover(
        destination: HandoffDestination,
        reason_code: HandoffReasonCode,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Hand this turn to a higher tier when the caller needs an action you cannot perform
        (changing an address/payment/cart/order, cancelling, refunding, or a multi-step task)."""
        # Sets `handover` + leaves a proper ToolMessage (clean tool_use/tool_result pairing).
        # Routing to the handover node is done by the graph's route_after_tools edge, which
        # reads this `handover` state — so ToolNode's normal batch handling stays intact.
        return Command(
            update={
                "handover": HandoffRequest(
                    destination=destination, reason_code=reason_code, source="model"
                ),
                "messages": [
                    ToolMessage("handover requested", tool_call_id=tool_call_id),
                ],
            }
        )

    return request_handover


def _model_spoke_this_turn(state: ReasoningState) -> bool:
    """True if the model produced spoken text since the last user turn.

    Scans back to the most recent HumanMessage; any AIMessage with non-empty string
    content in that window is the model's own narration (which reaches TTS as streamed
    tokens). Used to decide whether the handover node's canned deferral would double-speak.
    """
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return False
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return True
    return False


def _write_telemetry(record: dict[str, object]) -> None:
    try:
        _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # telemetry must never break a live call
        logger.warning("telemetry write failed: %s", exc)


def build_frontline_graph(
    chat_model: BaseChatModel,
    read_only_tools: Sequence[BaseTool],
    *,
    display_name: str,
) -> CompiledStateGraph:
    """Compile the frontline graph. `read_only_tools` are already audit-wrapped (tooling.py)."""
    handover_tool = _build_handover_tool()
    all_tools = [*read_only_tools, handover_tool]
    model_with_tools = chat_model.bind_tools(all_tools)
    system_prompt = _compose_system_prompt(display_name)
    read_only_names = {t.name for t in read_only_tools}

    def _last_user_text(state: ReasoningState) -> str:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def gate_node(state: ReasoningState) -> dict[str, object]:
        """Deterministic pre-generation check. Trip -> set handover, skip the model."""
        hit = gate_check(_last_user_text(state))
        if hit is None:
            return {}
        reason_code, destination = hit
        return {
            "handover": HandoffRequest(
                destination=destination,  # type: ignore[arg-type]
                reason_code=reason_code,
                source="gate",
            )
        }

    def route_after_gate(state: ReasoningState) -> str:
        return "handover" if state.handover is not None else "model"

    def model_node(state: ReasoningState) -> dict[str, object]:
        # Prompt lives inside the graph (single source; eval == production).
        messages = [SystemMessage(system_prompt), *state.messages]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_model(state: ReasoningState) -> str:
        last = state.messages[-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END
        # Bounded read-only tool loop: a model that never stops calling tools (or a
        # provider quirk) must not spin. After _MAX_TOOL_HOPS read-only round-trips in a
        # turn, end rather than loop. (request_handover routes itself out via Command, so
        # this only bounds legitimate read tools.)
        tool_hops = sum(1 for m in state.messages if isinstance(m, AIMessage) and m.tool_calls)
        if tool_hops > _MAX_TOOL_HOPS:
            logger.warning("frontline exceeded %d tool hops in a turn; ending", _MAX_TOOL_HOPS)
            return END
        return "tools"

    def handover_node(state: ReasoningState) -> dict[str, object]:
        assert state.handover is not None  # only reached with a handover set
        handover = state.handover
        _write_telemetry(
            {
                "utterance": _last_user_text(state),
                "gate_or_model": handover.source,
                "destination": handover.destination,
                "reason_code": handover.reason_code,
            }
        )
        # The canned deferral is a FALLBACK, spoken only if nothing else will be. On the
        # gate path the model never ran → speak it. On the model path the model usually
        # narrates the handover in its own (streamed) tokens; appending the canned line
        # there double-speaks it (observed live 2026-07-08) — so speak it only if the
        # model produced NO spoken text this turn (empty tool-call turn → not silent).
        if handover.source == "gate" or not _model_spoke_this_turn(state):
            return {"messages": [AIMessage(_DEFERRAL[handover.destination])]}
        return {}

    tool_node = ToolNode(all_tools)

    graph = StateGraph(ReasoningState)
    graph.add_node("gate", gate_node)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("handover", handover_node)

    def route_after_tools(state: ReasoningState) -> str:
        # request_handover's Command sets `handover` AND targets the handover node; but the
        # static tools->next edge still evaluates, so guard it: if a handover was set, go
        # there; otherwise a read-only result returns to the model.
        return "handover" if state.handover is not None else "model"

    graph.add_edge(START, "gate")
    graph.add_conditional_edges(
        "gate", route_after_gate, {"handover": "handover", "model": "model"}
    )
    graph.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    graph.add_conditional_edges(
        "tools", route_after_tools, {"handover": "handover", "model": "model"}
    )
    graph.add_edge("handover", END)

    compiled = graph.compile()
    # Stash for tests/introspection (the read-only tool names the frontline may call).
    compiled.frontline_read_only_tools = read_only_names  # type: ignore[attr-defined]
    return compiled
