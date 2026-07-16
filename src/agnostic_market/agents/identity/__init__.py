"""The identity gated flow (P7) — verify WHO the caller is, then speak THEIR orders.

- flow.py   : the graph nodes — assemble (contact claim, code-matched) -> guardrail ->
              [the _stepup.py chain: risk_check -> dispatch -> collect[HITL, OTP]] ->
              apply (bind + speak the scoped list), plus the bounded re-ask and escapes.
              Carries the BINDING INVARIANT (a stale cross-family L2 must not bind —
              PendingIdentity.grants_at_mint). CODE.
- prompt.py : the assemble-node model-facing text (collect ONE fact: the contact on the
              account; never promise outcomes or name orders). PROMPT.
"""

from agnostic_market.agents.identity.flow import IdentityNodes, build_identity_nodes

__all__ = ["IdentityNodes", "build_identity_nodes"]
