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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agnostic_market.agents import telemetry
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.tools import build_voice_tools

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_t1.yaml"
_MERCHANT_ID = "acme_store"
_RECALL_BAR = 0.90


_THREAD_SEQ = 0


async def _outcome(graph, utterance: str) -> str | None:
    """How the turn left the frontline, else None (answered).

    3b semantics: a checkout-destination handover no longer ends at a spoken deferral —
    it ENTERS the checkout flow (clearing the handover signal), so escalation shows up as
    `active_flow`/`pending_action` in the output, or as a paused confirm interrupt.
    Returns 'gate'/'model' (handover survived in state) or 'flow' (entered checkout).
    Each utterance runs on a fresh thread (the flow's interrupt needs a checkpointer).
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
        or out.get("pending_action") is not None
        or graph.get_state(config).interrupts
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
    tools = [wrap_readonly_tool(t, _MERCHANT_ID) for t in build_voice_tools(store)]
    config = resolved.config
    graph = build_frontline_graph(
        chat_model,
        tools,
        display_name=config.display_name,
        # The T1 eval never enters checkout (text-only escalation probes), but the graph
        # compiles as ONE shape — same construction path as production (F1 discipline).
        reasoning_model=LLMGateway(credentials, secrets).chat_model(config.llm.reasoning),
        store=store,
        policy=PolicyContext(
            max_order_value_usd=config.policies.max_order_value_usd,
            allow_ai_merchant_handoff=config.policies.allow_ai_merchant_handoff,
            refund_auto_approve_under_usd=config.policies.refunds.auto_approve_under_usd,
            refund_require_human_above_usd=config.policies.refunds.require_human_above_usd,
            pending_ttl_seconds=config.policies.pending_confirmation_ttl_seconds,
        ),
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
        f"| entered checkout flow {by_flow}"
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
