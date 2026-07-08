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

from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.tools import build_voice_tools, load_orders_fixture

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_t1.yaml"
_MERCHANT_ID = "acme_store"
_RECALL_BAR = 0.90


async def _outcome(graph, utterance: str) -> str | None:
    """Handover source ('gate'/'model') if the turn escalated, else None (answered)."""
    out = await graph.ainvoke({"messages": [HumanMessage(utterance)]})
    handover = out.get("handover")
    return handover.source if handover else None


async def _run() -> int:
    load_dotenv()
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    resolved = ConfigRegistry(_CONFIG_ROOT).load().get(_MERCHANT_ID)
    chat_model = LLMGateway(credentials, secrets).chat_model(resolved.config.llm.routing)
    tools = [
        wrap_readonly_tool(t, _MERCHANT_ID)
        for t in build_voice_tools(load_orders_fixture(_CONFIG_ROOT, _MERCHANT_ID))
    ]
    graph = build_frontline_graph(chat_model, tools, display_name=resolved.config.display_name)
    data = load_yaml_layer(_EVAL_PATH)

    # --- PRIMARY: escalation recall (gate OR model) on should_escalate ---
    escalate = data["should_escalate"]
    by_gate = by_model = 0
    misses: list[str] = []
    for utt in escalate:
        source = await _outcome(graph, utt)
        if source == "gate":
            by_gate += 1
        elif source == "model":
            by_model += 1
        else:
            misses.append(utt)
    escalated = by_gate + by_model
    recall = escalated / len(escalate)
    print(f"[should_escalate] recall: {escalated}/{len(escalate)} ({recall:.0%})")
    print(f"    gate caught {by_gate} (informational fast-path) | model caught {by_model}")
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
