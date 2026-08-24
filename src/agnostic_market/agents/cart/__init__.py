"""The cart gated flow (Tier 3, AGENTS §A10/§A10a) — Group B.

ONE flow owns cart MUTATION (reversible: add/remove/set-quantity/review) and the whole-cart
PLACEMENT tail (irreversible: the hardened checkout-3b assemble→guardrail→confirm[HITL]→
place, now over a frozen whole-cart snapshot). Direct-buy normalizes through the cart.

- flow.py   : graph nodes, deterministic consent, and the §7a readback. CODE.
- prompt.py : the assemble-node model-facing text + candidate rendering. PROMPT.
"""

from agnostic_market.agents.cart.flow import (
    PLACE_ORDER_POLICY,
    CartNodes,
    build_cart_nodes,
)

__all__ = [
    "PLACE_ORDER_POLICY",
    "CartNodes",
    "build_cart_nodes",
]
