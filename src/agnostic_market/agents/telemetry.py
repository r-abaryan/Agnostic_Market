"""Dedicated telemetry sink (F6) — the classifier/audit dataset, separate from app logs.

One JSONL line per event: turn outcomes (answered/handover — positives AND negatives) and
checkout lifecycle events (confirmed/cancelled/denied/expired/abandoned). PRODUCTION USE
REQUIRES the COMPLIANCE consent/retention framework before storing real-caller
transcripts — fixture-data-only until then. Reason codes only, never free model prose.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("agnostic_market.agents.telemetry")

_TELEMETRY_PATH = Path(__file__).resolve().parents[3] / "config" / "telemetry" / "frontline.jsonl"


def write_event(record: dict[str, object]) -> None:
    try:
        _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # telemetry must never break a live call
        logger.warning("telemetry write failed: %s", exc)
