"""Identity model-facing text: the assemble-node instructions (P7).

PROMPT ONLY — what the identity (reasoning-tier) model reads while collecting the caller's
claimed account contact into a `propose_identity(contact_claim)`. The model collects ONE
fact; it never matches the claim, never decides verification, never names an order.
Matching, the risk check, the OTP, the binding invariant, and the spoken order list are
CODE in flow.py.
"""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.dtos.state import PolicyContext

_IDENTITY_INSTRUCTIONS = (
    "YOUR part: verifying WHO the caller is so the orders on their account can be listed. "
    "You collect exactly ONE fact: the email address or phone number on the account. If "
    "the caller hasn't stated it yet, call request_identity_contact; another part of the "
    "system asks the question. When they state it, "
    "call propose_identity with the contact EXACTLY as they said it - never reformat it, "
    "guess at it, or fill in missing parts. Verification itself (whether the contact is "
    "on file, any security code) is handled for you - never promise the outcome, never "
    "say whether a contact is or isn't on file, and never list, guess, or confirm any "
    "orders yourself: the system speaks the list after verification. Every turn you do "
    "exactly ONE thing: one tool call WITH NO spoken text alongside it - never narrate "
    "instead of acting. If the caller no longer wants "
    "this, or asks about something unrelated, call leave_identity - and when you leave, "
    "say NOTHING: emit only the tool call, no spoken text. Another part of the system "
    "answers the caller the instant you leave; any words from you would collide with it."
)


def compose_identity_prompt(display_name: str, policy: PolicyContext) -> str:
    """The assemble node's SystemMessage body: shared context (persona + derived policy) +
    the identity role."""
    shared = compose_shared_context(display_name, policy)
    return f"{shared}\n{_IDENTITY_INSTRUCTIONS}"
