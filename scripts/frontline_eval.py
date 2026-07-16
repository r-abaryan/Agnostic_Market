"""T1 eval — frontline under-escalation, live (AGENTS.md seam #1). MODEL-primary.

Run: uv run python scripts/frontline_eval.py
Needs ANTHROPIC_API_KEY (the acme routing model) in .env. Text-only — no voice.

Invokes the SAME graph production uses (prompt inside the graph, F1), one utterance per
turn. The MODEL is the primary escalation decider; the slim gate is a high-certainty
irreversible-only floor. Pre-registered:
  - PRIMARY BAR: escalation recall (gate OR model) >= 90% on should_escalate. Every miss
    triaged (a real answer-instead-of-escalate is a judgment gap to fix in prompt/few-shot).
  - Gate coverage REPORTED informationally (the free fast-path share) — never a mandate.
  - over-escalation on should_answer REPORTED (over-escalating is cheap, AGENTS §A2).
Safety note: recall < 100% is NOT a safety failure (frontline holds no state-changing
tools — a miss cannot mutate). This measures escalation QUALITY. Exit 1 if the bar is missed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agnostic_market.agents import telemetry
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import LastOrderPointer, OrderStore, load_orders_fixture
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.tools import build_voice_tools

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_t1.yaml"
_MERCHANT_ID = "acme_store"
_RECALL_BAR = 0.90


_THREAD_SEQ = 0

# The gated flows' model-facing tools: any of these appearing in the turn's messages means
# the turn LEFT the frontline and a flow's model ran — an escalation, even when the flow
# then ended in a terminal decline that clears its own state (see docstring below).
_FLOW_TOOLS = frozenset(
    {
        "propose_refund", "propose_cancel", "propose_return",  # support (returns: Group C)
        "propose_profile_change", "leave_support",  # support (profile: Group C)
        "add_to_cart", "remove_from_cart", "set_quantity", "review_cart", "buy_now",
        "go_to_checkout", "leave_cart",  # cart (Group B; view_cart is a FRONTLINE read, not here)
        "propose_identity", "leave_identity",  # identity (P7; list_orders is a FRONTLINE read)
    }
)


async def _outcome(graph, utterance: str) -> str | None:
    """How the turn left the frontline, else None (answered).

    3b semantics: a checkout/support-destination handover no longer ends at a spoken
    deferral — it ENTERS the flow (clearing the handover signal), so escalation shows up
    as `active_flow`/`pending_*` in the output, or as a paused confirm interrupt.
    3c-follow-up semantics: a flow can also run to a TERMINAL decline (return-first,
    amount gate, ineligible cancel) that clears `active_flow` before END — then the only
    trace is the flow model's tool activity in the turn's messages, so that counts as
    'flow' too. Returns 'gate'/'model' (handover survived in state) or 'flow' (a gated
    flow ran). Each utterance runs on a fresh thread (interrupts need a checkpointer).
    """
    global _THREAD_SEQ
    _THREAD_SEQ += 1
    config = {"configurable": {"thread_id": f"eval-{_THREAD_SEQ}"}}
    out = await graph.ainvoke({"messages": [HumanMessage(utterance)]}, config)
    handover = out.get("handover")
    if handover is not None:
        return handover.source
    if (
        out.get("active_flow") is not None
        or out.get("pending_placement") is not None
        or graph.get_state(config).interrupts
    ):
        return "flow"
    for msg in out["messages"]:
        if isinstance(msg, AIMessage) and any(
            call["name"] in _FLOW_TOOLS for call in msg.tool_calls
        ):
            return "flow"
    return None


async def _run() -> int:
    load_dotenv()
    # Eval runs must not pollute the LIVE telemetry dataset (classifier data): redirect
    # this process's sink to a sibling eval file (same local-only, gitignored dir).
    telemetry._TELEMETRY_PATH = telemetry._TELEMETRY_PATH.with_name("frontline_eval.jsonl")
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    resolved = ConfigRegistry(_CONFIG_ROOT).load().get(_MERCHANT_ID)
    chat_model = LLMGateway(credentials, secrets).chat_model(resolved.config.llm.routing)
    store = OrderStore(load_orders_fixture(_CONFIG_ROOT, _MERCHANT_ID))
    cart_store = CartStore()
    pointer = LastOrderPointer()
    customers = CustomerDirectory(load_customers_fixture(_CONFIG_ROOT, _MERCHANT_ID))
    identity_store = CallerIdentityStore()
    tools = [
        wrap_readonly_tool(t, _MERCHANT_ID)
        for t in build_voice_tools(store, cart_store, pointer, identity_store, customers)
    ]
    config = resolved.config
    graph = build_frontline_graph(
        chat_model,
        tools,
        display_name=config.display_name,
        tenant_id=config.merchant_id,
        # The T1 eval never enters checkout (text-only escalation probes), but the graph
        # compiles as ONE shape — same construction path as production (F1 discipline).
        reasoning_model=LLMGateway(credentials, secrets).chat_model(config.llm.reasoning),
        store=store,
        cart_store=cart_store,  # SAME instance as view_cart (no split-brain)
        pointer=pointer,  # SAME instance as order_status (Group C L4)
        identity_store=identity_store,  # SAME instance as the tools' gate (P7, no split-brain)
        customers=customers,
        # The ONE config->runtime policy mapping — identical to production (F1). The old
        # hand-built copy here had drifted (it silently omitted spoken_policy_extra).
        policy=config.policies.to_policy_context(),
        # An utterance that reaches the confirm readback pauses at an interrupt, which
        # needs a checkpointer even in the text eval (fresh thread per utterance).
        checkpointer=InMemorySaver(),
    )
    data = load_yaml_layer(_EVAL_PATH)

    # --- PRIMARY: escalation recall (gate OR model OR checkout-flow entry) ---
    escalate = data["should_escalate"]
    by_gate = by_model = by_flow = 0
    misses: list[str] = []
    for utt in escalate:
        source = await _outcome(graph, utt)
        if source == "gate":
            by_gate += 1
        elif source == "model":
            by_model += 1
        elif source == "flow":
            by_flow += 1
        else:
            misses.append(utt)
    escalated = by_gate + by_model + by_flow
    recall = escalated / len(escalate)
    print(f"[should_escalate] recall: {escalated}/{len(escalate)} ({recall:.0%})")
    print(
        f"    gate caught {by_gate} (informational fast-path) | model caught {by_model} "
        f"| entered a gated flow {by_flow}"
    )
    for miss in misses:
        print(f"    MISS (answered instead of escalated - triage prompt/few-shot): {miss!r}")

    # --- INFORMATIONAL: over-escalation on should_answer ---
    answer = data["should_answer"]
    over = []
    for utt in answer:
        source = await _outcome(graph, utt)
        if source is not None:
            over.append(f"{utt!r} (source={source})")
    print(f"[should_answer] over-escalation: {len(over)}/{len(answer)} (informational)")
    for o in over:
        print(f"    OVER-ESCALATED: {o}")

    if recall < _RECALL_BAR:
        print(f"\nT1 EVAL FAILED - escalation recall {recall:.0%} < {_RECALL_BAR:.0%}. [FAIL]")
        return 1
    print(f"\nT1 eval passed: escalation recall {recall:.0%} >= {_RECALL_BAR:.0%}. [PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
