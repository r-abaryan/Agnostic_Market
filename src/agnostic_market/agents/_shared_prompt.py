"""The shared prompt context EVERY agent model reads (frontline, checkout, support).

One source of truth for knowledge that must not be tier-local (live 2026-07-10: the
refund-policy facts lived only in the frontline prompt, so a policy question that
gate-routed into support met a model with zero policy knowledge). Two blocks:
  - persona continuity: one continuous assistant, the machinery is never voiced, and no
    tier tells the caller a thing is impossible (it hands off instead);
  - the merchant policy summary, DERIVED from the enforced policy values + the merchant's
    free-text extras (agents/spoken_policy.py) — the ONLY policy statements any model may
    make.

Deliberately NOT here: a prose list of "what the assistant can do." That would be a second
copy of behavior owned by code + routing, drifting silently the day a capability changes.
The persona rule ("if it isn't your job, hand off - never say it's impossible") gets the
same caller outcome without duplicating the capability set.
"""

from __future__ import annotations

from agnostic_market.agents.spoken_policy import compose_spoken_policy
from agnostic_market.dtos.state import PolicyContext

_PERSONA = (
    "You are part of the single voice assistant for {display_name} - the caller "
    "experiences ONE continuous assistant for the whole call. Never mention teams, "
    "departments, handovers, transfers, or other systems - the caller must never hear the "
    "machinery. Everything already said in this conversation (order confirmations, refund "
    "or cancellation outcomes) was said by you, and it is true. If the caller wants "
    "something that isn't YOUR job, never tell them it's impossible - it is handled "
    "elsewhere in the call without the caller doing anything."
)

_POLICY_FACTS = (
    "Merchant policy facts - the ONLY policy statements you may make:\n{facts}\n"
    "If a policy detail is not covered above, say you don't have that detail on hand - "
    "NEVER invent policy (refund timelines, return windows, conditions)."
)


def compose_shared_context(display_name: str, policy: PolicyContext) -> str:
    """The common prompt prefix: persona continuity + the derived policy summary. The policy
    sentences come from the enforced values (single source), so no model can state a
    threshold that contradicts the guardrail."""
    facts = compose_spoken_policy(policy)
    facts_block = _POLICY_FACTS.format(facts=facts)
    return f"{_PERSONA.format(display_name=display_name)}\n{facts_block}"
