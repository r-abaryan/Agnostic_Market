"""Dedicated telemetry sink (F6) — the classifier/audit dataset, separate from app logs.

One JSONL line per event: turn outcomes (answered/handover — positives AND negatives) and
checkout lifecycle events (confirmed/cancelled/denied/expired/abandoned). PRODUCTION USE
REQUIRES the COMPLIANCE consent/retention framework before storing real-caller
transcripts — fixture-data-only until then. Reason codes only, never free model prose.

The `utterance` field is redacted at this chokepoint (SECURITY §7d): contact-shaped spans are
removed generally, and profile-change handovers suppress the whole utterance because an address
has no reliable shape. Everything else in a record is closed slugs by the callers' own discipline
(never tool-arg values).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from agnostic_market.agents.legacy_observation import (
    observe_legacy_answer,
    observe_legacy_capabilities,
)
from agnostic_market.commerce.spoken import redact_contact
from agnostic_market.dtos.orchestration import CapabilityId

logger = logging.getLogger("agnostic_market.agents.telemetry")

_TELEMETRY_PATH = Path(__file__).resolve().parents[3] / "config" / "telemetry" / "frontline.jsonl"
_SENSITIVE_UTTERANCE_REASONS = frozenset({"address_change", "contact_change"})
_REDACTED_UTTERANCE = "[redacted]"


def write_event(record: dict[str, object]) -> None:
    utterance = record.get("utterance")
    if isinstance(utterance, str):
        redacted = (
            _REDACTED_UTTERANCE
            if record.get("reason_code") in _SENSITIVE_UTTERANCE_REASONS
            else redact_contact(utterance)
        )
        record = {**record, "utterance": redacted}
    try:
        _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # telemetry must never break a live call
        logger.warning("telemetry write failed: %s", exc)


CapabilityAnswerSource = Literal[
    "code_authored_read",
    "grounded_model_response",
    "general_model_response",
]


def write_capability_answered(
    utterance: str,
    capability: str,
    *,
    answer_source: CapabilityAnswerSource,
) -> None:
    """The one answered-turn record shared by every capability-dispatched answer owner.

    Call it only once the spoken line exists, as the tool path does: a row written ahead of a
    failing render claims "answered" for a turn the caller never heard. A blank utterance writes
    NOTHING - an allowlisted read resumed after principal rotation runs in a fresh thread with no
    messages, and an empty-string row is a mislabelled negative, not a usable one.
    """
    if not utterance.strip():
        return
    observe_legacy_capabilities((CapabilityId(capability),), source="typed_owner")
    observe_legacy_answer(source="typed_owner")
    write_event(
        {
            "utterance": utterance,
            "outcome": "answered",
            "outcome_detail": "capability_answer",
            "capability": capability,
            "answer_source": answer_source,
        }
    )
