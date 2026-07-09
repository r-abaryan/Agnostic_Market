"""The checkout gated flow (Tier 3, AGENTS §A10/§A10a).

- flow.py   : the graph nodes (assemble → guardrail → confirm[HITL interrupt] → place),
              deterministic consent classification, the §7a readback, escapes. CODE.
- prompt.py : the assemble-node model-facing text + candidate rendering. PROMPT.
"""

from agnostic_market.agents._consent import is_abort, wants_human
from agnostic_market.agents.checkout.flow import (
    PLACE_ORDER_POLICY,
    CheckoutNodes,
    build_checkout_nodes,
    speak_quantity,
)

__all__ = [
    "PLACE_ORDER_POLICY",
    "CheckoutNodes",
    "build_checkout_nodes",
    "is_abort",
    "speak_quantity",
    "wants_human",
]
