"""Reasoning plane: the tiered agent graph + engine (AGENTS.md).

- gate.py        : deterministic pre-generation escalation triggers (platform safety floor).
- frontline/     : Tier-1 routing + the compiled graph that wires every flow (graph.py).
- cart/, support/, identity/ : the Tier-3 gated flows (flow.py + prompt.py each).
- capabilities.py: the session capability registry the graph dispatcher resolves against.
- engine.py      : ReasoningEngine, the Plane-2 seam (thread + resume + TurnEvents).
- recovery.py    : node recovery policy, the registry every graph node registers through.
- tooling.py     : the tool wrapper every agent tool call passes through (tenant + audit).
- telemetry.py   : the dedicated JSONL event sink (classifier dataset; consent-gated in prod).

Each flow is a package: graph/flow logic (CODE) is separated from its model-facing text
(prompt.py) so prompts are easy to read and iterate as they grow — but consent
classification, readbacks, and guardrails stay in the logic files, never in a prompt.
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
