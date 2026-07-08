"""The frontline agent (Tier 1) + the reasoning graph that wires in the checkout flow.

- graph.py  : the compiled StateGraph (frontline routing + checkout wiring), handover tool,
              node-authored deferral copy, speakable-node declaration. CODE.
- prompt.py : the frontline model-facing instructions + contrastive few-shot. PROMPT.
"""

from agnostic_market.agents.frontline.graph import _MAX_TOOL_HOPS, build_frontline_graph

__all__ = ["build_frontline_graph"]

# _MAX_TOOL_HOPS is re-exported for the loop-guard regression test (not public API).
_ = _MAX_TOOL_HOPS
