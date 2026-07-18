"""commerce/spoken.py redaction units (SECURITY §7d transcript-telemetry gap) + the
write_event chokepoint: persisted utterances never carry a contact span. Zero network.
(The listen-direction normalizers spoken_email/spoken_digits are pinned where they're
consumed — test_commerce_identity / test_commerce_verification.)"""

from __future__ import annotations

import json
from pathlib import Path

from agnostic_market.commerce.spoken import redact_contact


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


def test_write_event_redacts_the_utterance_field(tmp_path: Path) -> None:
    # The chokepoint: no telemetry writer can persist a raw contact, whatever the caller.
    from agnostic_market.agents import telemetry

    telemetry.write_event({"utterance": "sure, casey@example.com", "outcome": "answered"})
    telemetry.write_event({"event": "support_left", "reason": "left_flow"})  # no utterance
    lines = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["utterance"] == "sure, [email]"
    assert lines[1] == {"event": "support_left", "reason": "left_flow"}
