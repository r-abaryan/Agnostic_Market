"""The reasoning graph: frontline agent (Tier 1) + the checkout & support gated flows (Tier 3).

The load-bearing safety invariant: the frontline holds NO state-changing tools. It can
answer (read-only tools) or hand over — so a wrong judgment is at worst "answered without
acting" (recoverable), never a dangerous mutation. Handover fires two ways:
  - the model's `request_handover` tool — the PRIMARY escalation decider (reads intent,
    including paraphrases; industry-consistent LLM-router stance);
  - the deterministic gate (agents/gate.py) — a slim pre-generation floor for high-certainty
    IRREVERSIBLE requests only (cancel/refund/place-order). NOT the router; a bonus fast-path.
Safety is STRUCTURAL (no sensitive tools), not dependent on either catcher hitting 100%.

Graph shape (3b/3c + Group A):
    entry -> (sticky in a flow? escapes: human -> abort -> CROSS-SWITCH -> assemble : gate)
    gate  -> (trip? handover : model) ; model -> tools -> (handover? handover : model)
    handover -> (checkout/support(refund|cancel)? enter flow : spoken deferral -> END)
    checkout: assemble -> guardrail -> confirm[INTERRUPT] -> place
    support:  assemble -> {refund: guardrail -> [step-up] -> confirm[INT] -> place |
                           cancel: guardrail -> confirm[INT] -> void}
The CROSS-SWITCH escape (cross_switch node): while sticky, a gate-certain intent for a
DIFFERENT flow deterministically abandons the current one and hands over — closes the
sticky-flow trap where e.g. a refund request while stuck in checkout reached the checkout
model (which cannot serve it) instead of the gate.

The ONLY state-changing tool in the whole graph is the checkout flow's place effect, and
it sits strictly behind the guardrail + HITL interrupt (agents/checkout.py, AGENTS §A10a).
`request_handover` is a REAL executed tool that returns a `Command` routing to the handover
node — proper tool_use/tool_result pairing. The model-facing prompt + few-shot live in
`prompt.py` and are composed into the graph's model node (F1: eval and production share one
prompt path). Node-authored caller copy (the deferral map) stays here — it is behavior, not
a model instruction.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agnostic_market.agents.checkout import build_checkout_nodes, is_abort, wants_human
from agnostic_market.agents.frontline.prompt import compose_system_prompt
from agnostic_market.agents.gate import gate_check
from agnostic_market.agents.support import build_support_nodes
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.orders import OrderStore
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.state import (
    HandoffDestination,
    HandoffReasonCode,
    HandoffRequest,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.frontline")

_HANDOVER_TOOL_NAME = "request_handover"

# Max read-only tool round-trips per turn before the graph ends (loop guard — the
# hand-built graph has no framework loop protection like create_agent had).
_MAX_TOOL_HOPS = 5

# 3a deferral copy — destination-keyed and HONEST: nothing downstream exists yet, so we do
# NOT promise a live connection. Structure carries into 3b/3c where the promises come true.
# (Node-authored caller-facing copy, not a model prompt — stays here with the graph.)
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


def build_frontline_graph(
    chat_model: BaseChatModel,
    read_only_tools: Sequence[BaseTool],
    *,
    display_name: str,
    reasoning_model: BaseChatModel,
    store: OrderStore,
    policy: PolicyContext,
    verification_store: VerificationStore | None = None,
    otp: OtpProvider | None = None,
    risk: RiskProvider | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the reasoning graph: frontline (routing tier) + checkout + support flows.

    `read_only_tools` are already audit-wrapped (tooling.py). `checkpointer` is REQUIRED for
    the interrupt/resume paths (the engine passes one per session); None keeps the per-turn
    stateless mode the text eval uses. The support flow's step-up seams
    (`verification_store`/`otp`/`risk`) default to fresh fakes when omitted (eval/tests that
    never enter support), so the frontline-only call sites don't have to build them.
    """
    # Build otp first so a defaulted VerificationStore shares the SAME provider the dispatch
    # node uses (else dispatch and verify would talk to different fakes).
    otp = otp or OtpProvider()
    verification_store = verification_store or VerificationStore(otp)
    risk = risk or RiskProvider()
    handover_tool = _build_handover_tool()
    all_tools = [*read_only_tools, handover_tool]
    model_with_tools = chat_model.bind_tools(all_tools)
    system_prompt = compose_system_prompt(display_name)
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
                destination=destination, reason_code=reason_code, source="gate"
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
            return "finalize"
        # Bounded read-only tool loop: a model that never stops calling tools (or a
        # provider quirk) must not spin. After _MAX_TOOL_HOPS read-only round-trips in
        # THIS TURN (counted back to the last user message — state may carry prior turns
        # once a checkpointer holds the thread), end rather than loop. (request_handover
        # routes itself out via Command, so this only bounds legitimate read tools.)
        tool_hops = 0
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_hops += 1
        if tool_hops > _MAX_TOOL_HOPS:
            logger.warning("frontline exceeded %d tool hops in a turn; ending", _MAX_TOOL_HOPS)
            return "finalize"
        return "tools"

    def finalize_node(state: ReasoningState) -> dict[str, object]:
        """Telemetry sink for ANSWERED turns — the classifier dataset needs negatives
        (turns that did NOT escalate) just as much as the handover positives."""
        write_event({"utterance": _last_user_text(state), "outcome": "answered"})
        return {}

    def handover_node(state: ReasoningState) -> dict[str, object]:
        assert state.handover is not None  # only reached with a handover set
        handover = state.handover
        write_event(
            {
                "utterance": _last_user_text(state),
                "outcome": "handover",
                "gate_or_model": handover.source,
                "destination": handover.destination,
                "reason_code": handover.reason_code,
            }
        )
        # A checkout/support handover ENTERS the flow (3b/3c) instead of speaking a deferral:
        # set the sticky flow marker, clear the handover signal, and let routing carry us into
        # the flow's assemble in this same turn. EXCEPT when that flow was already left this
        # turn ("left_*") — re-entering would cycle; fall through to the spoken deferral.
        if handover.destination == "checkout" and state.active_flow != "left_checkout":
            return {"active_flow": "checkout", "handover": None}
        # Support handles REFUND and CANCEL_ORDER (Group A). Other support-destined reason
        # codes (address/payment change) are still 3c-follow-up breadth — they must NOT enter
        # the flow (it would run the reasoning model, fail to propose, bail to left_support,
        # re-trip the gate and double-speak). They defer honestly until their flow is built.
        if (
            handover.destination == "support"
            and handover.reason_code in ("refund", "cancel_order")
            and state.active_flow != "left_support"
        ):
            return {"active_flow": "support", "handover": None}
        # The canned deferral is a FALLBACK, spoken only if nothing else will be. On the
        # gate path the model never ran → speak it. On the model path the model usually
        # narrates the handover in its own (streamed) tokens; appending the canned line
        # there double-speaks it (observed live 2026-07-08) — so speak it only if the
        # model produced NO spoken text this turn (empty tool-call turn → not silent).
        if handover.source == "gate" or not _model_spoke_this_turn(state):
            return {"messages": [AIMessage(_DEFERRAL[handover.destination])]}
        return {}

    def entry_node(state: ReasoningState) -> dict[str, object]:
        # Fresh-turn hygiene: with a checkpointer, LAST turn's handover signal and the
        # turn-scoped "left_checkout" marker persist in thread state and must not affect
        # THIS turn. (pending_action needs no such reset: while one exists the graph is
        # paused at confirm and turns arrive as resumes, never through here.)
        update: dict[str, object] = {"handover": None}
        if state.active_flow in ("left_checkout", "left_support"):
            update["active_flow"] = None
        return update

    def cross_switch_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: while sticky in one gated flow, the caller voiced a
        HIGH-CERTAINTY intent for a DIFFERENT one (the gate tripped cross-flow). The 3b
        escape design's "hard topic switch", now code-owned for gate-certain utterances —
        the model-owned switch (leave_checkout) proved an unreliable sole owner live
        (2026-07-09: a refund request while sticky in checkout was refused once and
        narrated-over once). Abandons the in-flight flow and hands the turn over; NOTHING
        is spoken here — the receiving flow owns the voice."""
        hit = gate_check(_last_user_text(state))
        assert hit is not None  # only routed here when the router's own gate_check tripped
        reason_code, destination = hit
        write_event(
            {
                "event": "flow_cross_switch",
                "from": state.active_flow,
                "destination": destination,
                "reason_code": reason_code,
            }
        )
        return {
            "active_flow": None,
            "pending_action": None,
            "pending_refund": None,
            "pending_cancel": None,
            "handover": HandoffRequest(
                destination=destination, reason_code=reason_code, source="gate"
            ),
        }

    def route_after_entry(state: ReasoningState) -> str:
        # Escape checks BEFORE the sticky flow re-engages (decision: no caller is ever
        # trapped in a gated flow — AGENTS §A9). Deterministic, committed transcript only.
        # ORDER MATTERS: human -> abort -> cross-switch -> assemble. Abort precedes the
        # gate so a mid-flow "cancel that/it" aborts the IN-FLIGHT thing locally; "cancel
        # my order" (not an abort phrasing) falls through to the gate and cross-switches
        # to support's cancel-order path. The cross-switch closes the sticky-flow trap:
        # without it, a gate-certain intent for ANOTHER flow (e.g. "refund me" while stuck
        # in checkout) reaches the checkout model, which cannot serve it.
        text = _last_user_text(state)
        if state.active_flow == "checkout":
            if wants_human(text):
                return "checkout_escape_human"
            if is_abort(text):
                return "checkout_abort"
            hit = gate_check(text)
            if hit is not None and hit[1] != "checkout":
                return "cross_switch"
            return "checkout_assemble"
        if state.active_flow == "support":
            if wants_human(text):
                return "support_escape_human"
            if is_abort(text):
                return "support_abort"
            hit = gate_check(text)
            if hit is not None and hit[1] != "support":
                return "cross_switch"
            return "support_assemble"
        return "gate"

    def route_after_handover(state: ReasoningState) -> str:
        # A checkout/support destination entered the flow (handover cleared, flow set);
        # everything else spoke its deferral and ends.
        if state.handover is None:
            if state.active_flow == "checkout":
                return "checkout_assemble"
            if state.active_flow == "support":
                return "support_assemble"
        return END

    def route_after_assemble(state: ReasoningState) -> str:
        if state.active_flow == "left_checkout":
            # Model left the flow (hard topic switch / repeated invalid proposals):
            # the normal pipeline answers this same turn (re-entry blocked by the marker).
            return "gate"
        if state.pending_action is not None:
            return "checkout_guardrail"
        return END  # clarifying question (streamed tokens already spoken)

    def route_after_guardrail(state: ReasoningState) -> str:
        return "checkout_confirm" if state.pending_action is not None else END

    def route_after_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_action is not None:
            return "checkout_place"  # explicit committed yes
        return END  # declined / expired (node spoke its line, clear-before-speak)

    checkout = build_checkout_nodes(
        reasoning_model, store, policy, display_name=display_name
    )
    support = build_support_nodes(
        reasoning_model, store, verification_store, otp, risk, policy, display_name=display_name
    )

    # --- support flow routers (state-only; the level/status-dependent branches live INSIDE
    #     the flow, closed over the store — support.route_after_* ) ---
    def route_after_support_assemble(state: ReasoningState) -> str:
        # refund | cancel | leave | clarify — the flow decides from which pending was minted.
        decision = support.route_after_assemble(state)
        return {
            "refund": "support_guardrail",
            "cancel": "support_cancel_guardrail",
            "leave": "gate",  # model left; normal pipeline answers this same turn
            "clarify": END,  # a clarifying question was streamed already
        }[decision]

    def route_after_support_cancel_guardrail(state: ReasoningState) -> str:
        # confirm (eligible) | handover (risk-flagged) | declined (shipped: guardrail spoke + ends).
        return {
            "confirm": "support_cancel_confirm",
            "handover": "handover",
            "declined": END,
        }[support.route_after_cancel_guardrail(state)]

    def route_after_support_cancel_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_cancel is not None:
            return "support_cancel_void"  # committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_cancel_void(state: ReasoningState) -> str:
        # void may hand to a human on a store refusal (order shipped since assemble); else ends.
        return "handover" if state.handover is not None else END

    def route_support_guardrail(state: ReasoningState) -> str:
        # "confirm" (level ok) | "stepup" (-> risk_check) | "declined" (over amount /
        # cancelled: guardrail spoke its line + ends) | "cancel" (remedy steer: full
        # money-back on an unshipped order converted to the cancel path).
        decision = support.route_after_guardrail(state)
        return {
            "confirm": "support_confirm",
            "stepup": "support_risk_check",
            "declined": END,
            "cancel": "support_cancel_guardrail",
        }[decision]

    def route_after_support_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "support_dispatch"

    def route_after_support_collect(state: ReasoningState) -> str:
        # "confirm" (raised to L2) | "dispatch" (re-collect) | "handover" (exhausted).
        decision = support.route_after_collect(state)
        return {"confirm": "support_confirm", "dispatch": "support_dispatch",
                "handover": "handover"}[decision]

    def route_after_support_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_refund is not None:
            return "support_place"  # explicit committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_place(state: ReasoningState) -> str:
        # place may hand to a human on a store refusal / lapsed level; else it spoke + ends.
        return "handover" if state.handover is not None else END

    tool_node = ToolNode(all_tools)

    graph = StateGraph(ReasoningState)
    graph.add_node("entry", entry_node)
    graph.add_node("cross_switch", cross_switch_node)
    graph.add_node("gate", gate_node)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("handover", handover_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("checkout_assemble", checkout.assemble)
    graph.add_node("checkout_guardrail", checkout.guardrail)
    graph.add_node("checkout_confirm", checkout.confirm)
    graph.add_node("checkout_place", checkout.place)
    graph.add_node("checkout_abort", checkout.abort)
    graph.add_node("checkout_escape_human", checkout.escape_human)
    graph.add_node("support_assemble", support.assemble)
    graph.add_node("support_guardrail", support.guardrail)
    graph.add_node("support_risk_check", support.risk_check)
    graph.add_node("support_dispatch", support.dispatch)
    graph.add_node("support_collect", support.collect)
    graph.add_node("support_confirm", support.confirm)
    graph.add_node("support_place", support.place)
    graph.add_node("support_cancel_guardrail", support.cancel_guardrail)
    graph.add_node("support_cancel_confirm", support.cancel_confirm)
    graph.add_node("support_cancel_void", support.cancel_void)
    graph.add_node("support_abort", support.abort)
    graph.add_node("support_escape_human", support.escape_human)

    def route_after_tools(state: ReasoningState) -> str:
        # request_handover's Command sets `handover` AND targets the handover node; but the
        # static tools->next edge still evaluates, so guard it: if a handover was set, go
        # there; otherwise a read-only result returns to the model.
        return "handover" if state.handover is not None else "model"

    graph.add_edge(START, "entry")
    graph.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "gate": "gate",
            "cross_switch": "cross_switch",
            "checkout_assemble": "checkout_assemble",
            "checkout_abort": "checkout_abort",
            "checkout_escape_human": "checkout_escape_human",
            "support_assemble": "support_assemble",
            "support_abort": "support_abort",
            "support_escape_human": "support_escape_human",
        },
    )
    graph.add_edge("cross_switch", "handover")
    graph.add_conditional_edges(
        "gate", route_after_gate, {"handover": "handover", "model": "model"}
    )
    graph.add_conditional_edges(
        "model", route_after_model, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools", route_after_tools, {"handover": "handover", "model": "model"}
    )
    graph.add_conditional_edges(
        "handover",
        route_after_handover,
        {
            "checkout_assemble": "checkout_assemble",
            "support_assemble": "support_assemble",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "checkout_assemble",
        route_after_assemble,
        {"gate": "gate", "checkout_guardrail": "checkout_guardrail", END: END},
    )
    graph.add_conditional_edges(
        "checkout_guardrail",
        route_after_guardrail,
        {"checkout_confirm": "checkout_confirm", END: END},
    )
    graph.add_conditional_edges(
        "checkout_confirm",
        route_after_confirm,
        {"handover": "handover", "checkout_place": "checkout_place", END: END},
    )
    graph.add_edge("checkout_place", END)
    graph.add_edge("checkout_abort", END)
    graph.add_edge("checkout_escape_human", "handover")
    graph.add_conditional_edges(
        "support_assemble",
        route_after_support_assemble,
        {
            "gate": "gate",
            "support_guardrail": "support_guardrail",
            "support_cancel_guardrail": "support_cancel_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_guardrail",
        route_support_guardrail,
        {
            "support_confirm": "support_confirm",
            "support_risk_check": "support_risk_check",
            "support_cancel_guardrail": "support_cancel_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_risk_check",
        route_after_support_risk,
        {"support_dispatch": "support_dispatch", "handover": "handover"},
    )
    graph.add_edge("support_dispatch", "support_collect")
    graph.add_conditional_edges(
        "support_collect",
        route_after_support_collect,
        {
            "support_confirm": "support_confirm",
            "support_dispatch": "support_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "support_confirm",
        route_after_support_confirm,
        {"handover": "handover", "support_place": "support_place", END: END},
    )
    graph.add_conditional_edges(
        "support_place",
        route_after_support_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_guardrail",
        route_after_support_cancel_guardrail,
        {"support_cancel_confirm": "support_cancel_confirm", "handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_confirm",
        route_after_support_cancel_confirm,
        {"handover": "handover", "support_cancel_void": "support_cancel_void", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_void",
        route_after_support_cancel_void,
        {"handover": "handover", END: END},
    )
    graph.add_edge("support_abort", END)
    graph.add_edge("support_escape_human", "handover")
    graph.add_edge("finalize", END)

    compiled = graph.compile(checkpointer=checkpointer)
    # Stashed for tests/introspection + the engine (single source of truth for which
    # node-authored messages are caller-facing — the voice side never hard-codes names).
    compiled.frontline_read_only_tools = read_only_names  # type: ignore[attr-defined]
    compiled.speakable_nodes = (  # type: ignore[attr-defined]
        frozenset({"handover"}) | checkout.speakable_nodes | support.speakable_nodes
    )
    return compiled
