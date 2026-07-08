"""Reasoning plane — the tiered agent graph (AGENTS.md). Phase 3a: the frontline tier.

- gate.py      : deterministic pre-generation escalation triggers (platform safety floor).
- frontline.py : the Tier-1 frontline graph (read-only tools, no sensitive tools, handover).
- tooling.py   : the tool wrapper every agent tool call passes through (tenant + audit).

Higher tiers (checkout/support/planner) + the ReasoningEngine seam land in Phase 3b-3d.
"""

from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.gate import gate_check
from agnostic_market.agents.tooling import wrap_readonly_tool

__all__ = [
    "build_frontline_graph",
    "gate_check",
    "wrap_readonly_tool",
]
