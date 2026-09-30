"""The question asked before a routed person request ends automated help.

Ending automated help cannot be undone on the call, and the router's person decision is a model
judgment, so it is confirmed like any effect. Code-policy handovers are not asked about.
"""

from __future__ import annotations

from langgraph.types import interrupt

from agnostic_market.agents._consent import ConfirmationDecision, classify_person_confirmation
from agnostic_market.dtos.state import HandoffSource

PERSON_QUESTION = "Do you want me to stop the automated help so you can reach someone at the store?"
PERSON_QUESTION_RETRY = "To stop the automated help and reach someone at the store, say yes or no."
# The engine reads replies to these with the person grammar, whichever node asked.
PERSON_QUESTIONS = frozenset({PERSON_QUESTION, PERSON_QUESTION_RETRY})


def ask_person_question() -> ConfirmationDecision:
    """Ask, and ask once more if the reply is unclear."""
    decision = classify_person_confirmation(interrupt(PERSON_QUESTION))
    if decision.verdict == "unclear":
        decision = classify_person_confirmation(interrupt(PERSON_QUESTION_RETRY))
    return decision


def settle_person_request(decision: ConfirmationDecision) -> ConfirmationDecision:
    """Confirm a routed person request at a readback; a decline reads as unclear to the owner."""
    if decision.verdict != "human" or decision.handoff_source is not HandoffSource.SEMANTIC_ROUTER:
        return decision
    confirmed = ask_person_question()
    if confirmed.verdict in {"yes", "human"}:
        return decision
    return ConfirmationDecision("unclear")
