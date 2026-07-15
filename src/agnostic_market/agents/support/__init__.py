"""The support gated flow (Tier 3, AGENTS §A4/§A9) — refunds, cancellations, returns, and
profile changes (T3 + Groups A/C).

- flow.py    : the graph nodes — refund (assemble -> guardrail -> [step-up] -> confirm[HITL]
               -> place), cancel, returns (guardrail -> confirm[HITL] -> place), profile
               (guardrail -> [step-up] -> confirm[HITL] -> place) — the §7a readbacks,
               guardrail tiers, escapes. CODE.
- _stepup.py : the family-parametrized step-up chain factory (risk_check -> dispatch ->
               collect[HITL, verify]) — ONE body serving refund AND profile. CODE.
- prompt.py  : the assemble-node model-facing text + order rendering. PROMPT.
"""

from agnostic_market.agents.support.flow import SupportNodes, build_support_nodes

__all__ = ["SupportNodes", "build_support_nodes"]
