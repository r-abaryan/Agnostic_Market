"""Deterministic closed-consent contracts."""

import pytest

from agnostic_market.agents._consent import classify_confirmation, classify_consent
from agnostic_market.dtos.state import HandoffSource


@pytest.mark.parametrize("cancel_action", (False, True))
def test_unheard_confirmation_cannot_authorize_yes(cancel_action: bool) -> None:
    decision = classify_confirmation(
        {"text": "yes", "readback_interrupted": True},
        cancel_action=cancel_action,
    )
    assert decision.verdict == "unclear"


@pytest.mark.parametrize("source", tuple(HandoffSource))
def test_unheard_readback_still_reaches_a_person(source: HandoffSource) -> None:
    """Hearing the readback gates consent, never escape.

    ROUTING_FAILURE_POLICY is set when routing is unavailable, which is exactly when leaving the
    caller with a declined action and no route to a person is least acceptable.
    """
    decision = classify_confirmation(
        {"text": "let me talk to someone", "handoff_source": source, "readback_interrupted": True},
    )

    assert decision.verdict == "human"
    assert decision.handoff_source is source


@pytest.mark.parametrize(
    "utterance",
    (
        "yes",
        "yes please",
        "Yes, please.",
        "yeah",
        "yep",
        "yup",
        "confirm",
        "confirmed",
        "go ahead",
        "do it",
        "sounds good",
        "please do",
        "ok",
        "okay",
        "sure",
        "sure go ahead",
        "confirm it",
        "alright",
        "that's correct",
        "yes sir",
        "yes ma'am",
        "certainly",
        "that works",
        "let's do it",
    ),
)
def test_bounded_whole_reply_affirmations_authorize(utterance: str) -> None:
    assert classify_consent(utterance) == "yes"


@pytest.mark.parametrize(
    "utterance",
    (
        "yes do it",
        "yeah go ahead",
        "yes that's right",
        "yes that\u2019s right",
        "yes that is right",
        "yes, please do it",
        "yes and go ahead",
        "okay, yes",
        "yes, absolutely",
        "definitely yes",
        "yes, thanks",
    ),
)
def test_composed_affirmations_authorize(utterance: str) -> None:
    assert classify_consent(utterance) == "yes"


@pytest.mark.parametrize(
    "utterance",
    (
        "that is not right",
        "that is not correct",
        "yes, that is not correct",
        "I did not say yes",
        "I didn't say yes",
        "I cannot confirm that",
        "I can't confirm that",
        "I am not saying yes",
        "you should not do it",
        "I would not go ahead",
        "I am not sure, okay",
        "yes, but wait",
        "yes, if the price has not changed",
        "yes, unless it has already shipped",
        "yes, I guess",
        "yes, place the order",
        "yes, refund it instead",
        "yes, update my address",
        "yes, not yet",
        "right",
        "correct",
        "uh huh",
        "mhm",
        "please",
        "absolutely",
        "definitely",
        "thanks",
        "thank you",
        "not sure",
        "are you sure",
        "sure, but wait",
        "I think that's correct",
        "yes sir, but not yet",
        "certainly not",
        "that works if the total changes",
        "let's do it later",
    ),
)
def test_mixed_or_ambiguous_replies_never_authorize(utterance: str) -> None:
    assert classify_consent(utterance) != "yes"


@pytest.mark.parametrize("utterance", ("place it", "yes, place it"))
def test_placement_specific_language_is_not_generic_consent(utterance: str) -> None:
    assert classify_consent(utterance) != "yes"
