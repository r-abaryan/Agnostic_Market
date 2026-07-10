"""The slim irreversible-only gate — high precision (never false-trip a read). Zero net.

The gate is a NARROW deterministic floor, not the router: only high-certainty irreversible
actions (cancel / refund / place-order). Reversible changes (address/payment/cart) + all
paraphrases are the MODEL's job — the gate returns None for them, which is correct.
"""

from __future__ import annotations

import pytest

from agnostic_market.agents.gate import gate_check

# Irreversible-action REQUESTS the slim gate must catch.
_IRREVERSIBLE = [
    "cancel my order please",
    "cancel this order",
    "I want a refund",
    "can I get a refund on this",
    "I'd like my money back",
    "refund me for the order",
    "I need to return this",
    "start a return for my order",
    "return these shoes",
    "I'd like to send them back",  # return intent with no 'return'/'refund' token
    "send back everything I ordered",
    "checkout now please",
    "place my order",
    "go ahead and place the order",
    "buy it now",
    "complete my purchase",
]

# The gate deliberately does NOT own these (reversible / paraphrase-prone) — the MODEL does.
# gate_check returning None here is correct, not a miss.
_MODEL_OWNS = [
    "change my address to 12 Elm St",
    "update my shipping address",
    "change where this is being delivered",
    "use a different credit card",
    "add two of those to my cart",
    "take the jacket out of my basket",
    "I moved recently, make sure it goes to the right place",
]

# Reads / questions / policy asks — MUST NOT false-trip (the gate's precision guarantee).
_NO_TRIP = [
    "what's the status of order ORD-1001",  # F5 regression anchor
    "where's my order",
    "track my package",
    "tell me about your return policy",  # a policy question, not a return request
    "what's your refund policy",
    "why was my order cancelled",  # a read about a past cancel
    "what's in my cart right now",
    "do you have running shoes",
    "how much is the jacket",
    "what payment methods do you accept",
    "can you send me feedback on my order",  # 'send…feedback' must not read as 'send back'
    "when will my package be sent to me",
]


@pytest.mark.parametrize("text", _IRREVERSIBLE)
def test_irreversible_requests_trip(text: str) -> None:
    assert gate_check(text) is not None, f"irreversible request should trip: {text!r}"


@pytest.mark.parametrize("text", _MODEL_OWNS)
def test_reversible_and_paraphrase_defer_to_model(text: str) -> None:
    # None is correct: the gate has no high-certainty opinion, the model decides.
    assert gate_check(text) is None, f"gate should defer to model (None): {text!r}"


@pytest.mark.parametrize("text", _NO_TRIP)
def test_reads_never_false_trip(text: str) -> None:
    assert gate_check(text) is None, f"gate must NOT false-trip a read: {text!r}"


def test_trip_returns_reason_and_destination() -> None:
    assert gate_check("cancel my order") == ("cancel_order", "support")
    assert gate_check("I want a refund") == ("refund", "support")
    assert gate_check("checkout now") == ("cart_write", "checkout")


def test_empty_transcript_does_not_trip() -> None:
    assert gate_check("") is None
    assert gate_check("   ") is None
