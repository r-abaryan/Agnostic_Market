"""Reasoning plane — the tiered agent graph + engine (AGENTS.md). Phase 3a/3b.

- gate.py       : deterministic pre-generation escalation triggers (platform safety floor).
- frontline/    : Tier-1 frontline + checkout wiring (graph.py) + its prompt (prompt.py).
- checkout/     : the Tier-3 checkout gated flow (flow.py) + its prompt (prompt.py).
- engine.py     : ReasoningEngine — the Plane-2 seam (thread + resume + TurnEvents).
- tooling.py    : the tool wrapper every agent tool call passes through (tenant + audit).
- telemetry.py  : the dedicated JSONL event sink (classifier dataset; consent-gated in prod).

Each agent is a package: graph/flow logic (CODE) is separated from its model-facing text
(prompt.py) so prompts are easy to read and iterate as they grow — but consent
classification, readbacks, and guardrails stay in the logic files, never in a prompt.

Support flow (3c) + planner tier (3d, trace-gated) come next.
"""

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.gate import gate_check
from agnostic_market.agents.tooling import wrap_readonly_tool

__all__ = [
    "ReasoningEngine",
    "build_frontline_graph",
    "gate_check",
    "wrap_readonly_tool",
]
