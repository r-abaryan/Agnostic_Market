"""The support gated flow (Tier 3, AGENTS §A4/§A9) — refunds + step-up verification (T3).

- flow.py   : the graph nodes (assemble -> guardrail -> [step-up loop] -> confirm[HITL] ->
              place), the §7a readback, escapes. Step-up: risk_check -> dispatch ->
              collect[HITL, verify] -> confirm. CODE.
- prompt.py : the assemble-node model-facing text + order rendering. PROMPT.
"""

from agnostic_market.agents.support.flow import SupportNodes, build_support_nodes

__all__ = ["SupportNodes", "build_support_nodes"]
