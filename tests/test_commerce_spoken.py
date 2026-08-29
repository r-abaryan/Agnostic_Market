"""Contact redaction units and the telemetry-recorder persistence boundary. Zero network.
(The listen-direction normalizers spoken_email/spoken_digits are pinned where they're
consumed — test_commerce_identity / test_commerce_verification.)"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agnostic_market.commerce.spoken import (
    caller_stated_order_ids,
    caller_stated_phone,
    redact_contact,
    scan_contact_candidates,
)


@pytest.mark.parametrize(
    ("utterance", "expected"),
    (
        ("cancel order ORD-1002", ("ORD-1002",)),
        ("cancel order one zero zero two", ("ORD-1002",)),
        ("Cancel order. O r d one zero zero two.", ("ORD-1002",)),
        ("cancel ord1002", ("ORD-1002",)),
        ("cancel ORD 1002", ("ORD-1002",)),
        ("cancel order 1002", ("ORD-1002",)),
        ("Cancel O R D dash one zero zero two", ("ORD-1002",)),
        ("cancel or d one zero zero two", ("ORD-1002",)),
        ("cancel ORD-1002 and order ORD-1001", ("ORD-1002", "ORD-1001")),
        ("cancel ORD-1002, ORD-1001 plus ORD-1002", ("ORD-1002", "ORD-1001")),
    ),
)
def test_caller_stated_order_ids_accepts_only_complete_labelled_sets(
    utterance: str, expected: tuple[str, ...]
) -> None:
    assert caller_stated_order_ids(utterance) == expected


@pytest.mark.parametrize(
    "utterance",
    (
        "cancel my rain jacket order",
        "cancel one zero zero two",
        "cancel option two",
        "cancel order ORD-1002 or ORD-1001",
        "cancel order ORD-1002 and 1001",
        "cancel order ORD-1002 and my other order",
        "do not cancel order ORD-1002",
        "cancel order ORD-1002, actually order ORD-1001 instead",
        "refund $1002",
        "refund my order for $1001",
    ),
)
def test_caller_stated_order_ids_rejects_weak_or_ambiguous_sets(utterance: str) -> None:
    assert caller_stated_order_ids(utterance) == ()


def test_caller_stated_phone_requires_one_complete_phone_shaped_run() -> None:
    assert caller_stated_phone("Use five five five one one one two two two two", "555 111 2222")
    assert not caller_stated_phone("Change the contact for order 1002", "1002")
    assert not caller_stated_phone("Order 1002, then use 555 111 2222", "10025551112222")


def test_typed_email_span_is_redacted() -> None:
    assert redact_contact("it's casey@example.com thanks") == "it's [email] thanks"
    assert "[email]" in redact_contact("casey@example.com.")


def test_spoken_email_form_is_redacted() -> None:
    assert "example" not in redact_contact("casey at example dot com")
    assert "example" not in redact_contact("it's k c at example dot com thanks")


def test_digit_phone_run_is_redacted() -> None:
    assert redact_contact("my number is 555 010 0119 okay") == "my number is [phone] okay"
    assert "[phone]" in redact_contact("call me on 555-010-0119")


def test_digit_word_phone_is_redacted() -> None:
    out = redact_contact("it's five five five oh one oh oh one one nine")
    assert "[phone]" in out
    assert "five" not in out


def test_order_ids_and_small_numbers_survive() -> None:
    # 4 digits per id, connector words between mentions — under the phone line by design.
    assert redact_contact("cancel ORD-1002 please") == "cancel ORD-1002 please"
    assert redact_contact("what about ORD-1001 and ORD-1002") == "what about ORD-1001 and ORD-1002"
    assert redact_contact("I want two of them") == "I want two of them"


def test_plain_utterances_pass_through() -> None:
    assert redact_contact("hi there") == "hi there"
    assert redact_contact("cancel my rain jacket order") == "cancel my rain jacket order"


@pytest.mark.parametrize(
    ("utterance", "kind", "claim"),
    (
        ("casey@example.com", "email", "casey@example.com"),
        ("my email is casey at example dot com", "email", "casey@example.com"),
        ("contact me at casey at example dot com", "email", "casey@example.com"),
        ("order ORD-1002 and casey at example dot com", "email", "casey@example.com"),
        ("my number is five five five zero one zero zero one one nine", "phone", "5550100119"),
    ),
)
def test_contact_scanner_extracts_one_bounded_syntax_candidate(
    utterance: str, kind: str, claim: str
) -> None:
    observed = tuple(
        (candidate.kind, candidate.claim) for candidate in scan_contact_candidates(utterance)
    )
    assert observed == ((kind, claim),)


def test_contact_scanner_does_not_turn_an_order_reference_into_a_phone() -> None:
    assert scan_contact_candidates("check order ORD-10020001") == ()
    assert tuple(
        (candidate.kind, candidate.claim)
        for candidate in scan_contact_candidates("order 10020001, phone 555 010 0119")
    ) == (("phone", "5550100119"),)


def test_telemetry_recorder_redacts_the_utterance_field(tmp_path: Path) -> None:
    # The chokepoint: no telemetry writer can persist a raw contact, whatever the caller.
    from agnostic_market.agents.telemetry import JsonlTelemetrySink, TenantTelemetry

    telemetry = TenantTelemetry(
        "acme_store",
        JsonlTelemetrySink(tmp_path / "telemetry.jsonl"),
        JsonlTelemetrySink(tmp_path / "routing-telemetry.jsonl"),
    ).bind_session("spoken-test")
    telemetry.routing_evidence.record(
        {
            "event": "capability_answered",
            "utterance": "sure, casey@example.com",
            "outcome": "answered",
        }
    )
    telemetry.operational.record({"event": "support_left", "reason": "left_flow"})
    routing_lines = [
        json.loads(line)
        for line in (tmp_path / "routing-telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    operational_lines = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert routing_lines[0]["utterance"] == "sure, [email]"
    assert operational_lines[0]["event"] == "support_left"
    assert operational_lines[0]["reason"] == "left_flow"


def test_telemetry_recorder_suppresses_profile_change_utterances(tmp_path: Path) -> None:
    from agnostic_market.agents.telemetry import JsonlTelemetrySink, TenantTelemetry

    telemetry = (
        TenantTelemetry(
            "acme_store",
            JsonlTelemetrySink(tmp_path / "telemetry.jsonl"),
            JsonlTelemetrySink(tmp_path / "routing-telemetry.jsonl"),
        )
        .bind_session("spoken-test")
        .operational
    )
    telemetry.record(
        {
            "event": "human_onramp",
            "utterance": "send future orders to 7 Elm Street, Dover",
            "outcome": "handover",
            "reason_code": "address_change",
        }
    )
    record = json.loads((tmp_path / "telemetry.jsonl").read_text(encoding="utf-8"))
    assert record["utterance"] == "[redacted]"
