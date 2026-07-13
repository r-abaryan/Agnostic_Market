"""The shared prompt context EVERY agent model reads (frontline, checkout, support).

One source of truth for knowledge that must not be tier-local (live 2026-07-10: the
refund-policy facts lived only in the frontline prompt, so a policy question that
gate-routed into support met a model with zero policy knowledge). Three blocks:
  - persona continuity: one continuous assistant, the machinery is never voiced, and no
    tier tells the caller a thing is impossible (it hands off instead);
  - today's date + the past-date framing rule (live call #9 P6: with no "today", a stored
    ETA of July 9 was spoken as a FUTURE arrival on July 13);
  - the merchant policy summary, DERIVED from the enforced policy values + the merchant's
    free-text extras (agents/spoken_policy.py) — the ONLY policy statements any model may
    make.

Deliberately NOT here: a prose list of "what the assistant can do." That would be a second
copy of behavior owned by code + routing, drifting silently the day a capability changes.
The persona rule ("if it isn't your job, hand off - never say it's impossible") gets the
same caller outcome without duplicating the capability set.
"""

from __future__ import annotations

from datetime import datetime

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

_DATE_LINE = (
    "Today's date: {today}. A ship or arrival date BEFORE today is in the PAST - never "
    "speak it as an upcoming arrival; say it was expected by that date and offer to check "
    "the latest status."
)

_POLICY_FACTS = (
    "Merchant policy facts - the ONLY policy statements you may make:\n{facts}\n"
    "If a policy detail is not covered above, say you don't have that detail on hand - "
    "NEVER invent policy (refund timelines, return windows, conditions)."
)


def compose_shared_context(display_name: str, policy: PolicyContext) -> str:
    """The common prompt prefix: persona continuity + today's date + the derived policy
    summary. The policy sentences come from the enforced values (single source), so no
    model can state a threshold that contradicts the guardrail. The date is read at
    compose time (prompts are composed per turn, so it stays right across midnight);
    server-local — per-merchant timezone binding is a Phase-4 tenancy concern."""
    facts = compose_spoken_policy(policy)
    facts_block = _POLICY_FACTS.format(facts=facts)
    date_block = _DATE_LINE.format(today=f"{datetime.now():%A %d %B %Y}")
    return f"{_PERSONA.format(display_name=display_name)}\n{date_block}\n{facts_block}"
