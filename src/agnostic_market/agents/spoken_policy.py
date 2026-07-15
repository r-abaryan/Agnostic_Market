"""Spoken policy summary — DERIVED from the enforced policy values, never retyped.

The drift trap this closes (caught 2026-07-10): a merchant's `returnless_under_usd: 50` is
the number the refund guardrail ENFORCES; a hand-written "refunds over $50 need a return"
in the spoken summary is a SECOND copy of that 50 that a merchant can bump out of sync. The
enforced values have one source of truth (the typed PolicyContext the guardrail reads), so
the spoken sentence about them is generated FROM those values here — change the number, the
sentence changes, they cannot contradict.

Free-text facts with NO enforcing field (refund timeline "5-7 business days", "items must
be in original condition") have nothing to drift against; those stay merchant-authored
prose in `policies.spoken_facts_extra` and are appended after the derived sentences. The
return WINDOW moved from prose to a derived sentence when Group C made it enforced
(`returns.window_days` gates eligibility in the returns guardrail).
"""

from __future__ import annotations

from agnostic_market.dtos.state import PolicyContext


def _returnless_sentence(policy: PolicyContext) -> str:
    """How a SHIPPED/DELIVERED refund is handled, worded from `refund_returnless_under_usd`
    (the same value the guardrail's return-first gate reads — one source, no drift)."""
    threshold = policy.refund_returnless_under_usd
    if threshold <= 0:
        # Return-first for every shipped refund (the default).
        return (
            "For an order that has already shipped, a refund is issued once the return "
            "is arranged."
        )
    return (
        "For an order that has already shipped, refunds over "
        f"${threshold:.0f} are issued once the return is arranged; smaller amounts may "
        "be refunded without a return."
    )


def _human_review_sentence(policy: PolicyContext) -> str:
    """The amount above which a refund goes to a person, from `refund_require_human_above_usd`."""
    return (
        f"Refunds above ${policy.refund_require_human_above_usd:.0f} are handled by our "
        "support team rather than on this call."
    )


def _return_window_sentence(policy: PolicyContext) -> str:
    """The return-eligibility window, from `return_window_days` (the same value the returns
    guardrail enforces — one source, no drift)."""
    return f"Returns are accepted within {policy.return_window_days} days of delivery."


def compose_spoken_policy(policy: PolicyContext) -> str:
    """The spoken policy summary: DERIVED enforced-value sentences + the merchant's free-text
    extras. Always non-empty (the derived sentences exist for every merchant)."""
    parts = [
        _returnless_sentence(policy),
        _return_window_sentence(policy),
        _human_review_sentence(policy),
    ]
    if policy.spoken_policy_extra:
        parts.append(policy.spoken_policy_extra.strip())
    return " ".join(parts)
