from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.agents.telemetry import (
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    OperationalTelemetryEvent,
    RoutingEvidenceTelemetryEvent,
    SessionTelemetry,
    TelemetryPurpose,
    TelemetryRecorder,
    TenantTelemetry,
)

_PRODUCTION_ROOT = Path(__file__).parents[1] / "src" / "agnostic_market"
_EVENT_ENUMS = {
    "OperationalTelemetryEvent": OperationalTelemetryEvent,
    "RoutingEvidenceTelemetryEvent": RoutingEvidenceTelemetryEvent,
}


def _production_event_inventory() -> tuple[dict[str, set[str]], list[str], list[str]]:
    event_sites: dict[str, set[str]] = {}
    interpolated: list[str] = []
    unknown_members: list[str] = []
    for path in _PRODUCTION_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(_PRODUCTION_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if not isinstance(key, ast.Constant) or key.value != "event":
                        continue
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        event_sites.setdefault(value.value, set()).add(f"{relative}:{value.lineno}")
                    elif isinstance(value, (ast.BinOp, ast.Call, ast.JoinedStr)):
                        interpolated.append(f"{relative}:{value.lineno}")
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            event_enum = _EVENT_ENUMS.get(node.value.id)
            if event_enum is None:
                continue
            try:
                event = event_enum[node.attr].value
                event_sites.setdefault(event, set()).add(f"{relative}:{node.lineno}")
            except KeyError:
                unknown_members.append(f"{relative}:{node.lineno}:{node.value.id}.{node.attr}")
    return event_sites, interpolated, unknown_members


def test_production_event_inventory_is_closed_and_complete() -> None:
    event_sites, interpolated, unknown_members = _production_event_inventory()
    registered = {event.value for event_type in _EVENT_ENUMS.values() for event in event_type}
    unregistered = {
        event: sorted(sites) for event, sites in event_sites.items() if event not in registered
    }

    assert unknown_members == []
    assert interpolated == []
    assert unregistered == {}
    assert registered - event_sites.keys() == set()


def test_session_telemetry_binds_scope_and_purpose() -> None:
    operational_sink = InMemoryTelemetrySink()
    routing_sink = InMemoryTelemetrySink()
    session = TenantTelemetry("acme_store", operational_sink, routing_sink).bind_session(
        "session-1"
    )

    session.operational.record({"event": "caller_context_closed", "reason": "complete"})
    session.routing_evidence.record(
        {"event": "semantic_route", "decision": "request", "capability": "view_cart"}
    )

    (operational,) = operational_sink.records
    (routing,) = routing_sink.records
    assert operational.tenant_id == "acme_store"
    assert operational.session_id == "session-1"
    assert operational.purpose is TelemetryPurpose.OPERATIONAL
    assert operational.attributes == {"reason": "complete"}
    assert routing.purpose is TelemetryPurpose.ROUTING_EVIDENCE
    assert routing.attributes["capability"] == "view_cart"


@pytest.mark.parametrize(
    "routing",
    (
        TenantTelemetry("other_store", InMemoryTelemetrySink(), InMemoryTelemetrySink())
        .bind_session("session-1")
        .routing_evidence,
        TenantTelemetry("acme_store", InMemoryTelemetrySink(), InMemoryTelemetrySink())
        .bind_session("session-2")
        .routing_evidence,
    ),
)
def test_session_telemetry_rejects_split_scope(routing: TelemetryRecorder) -> None:
    operational = (
        TenantTelemetry("acme_store", InMemoryTelemetrySink(), InMemoryTelemetrySink())
        .bind_session("session-1")
        .operational
    )

    with pytest.raises(ValueError, match="telemetry scope"):
        SessionTelemetry(operational=operational, routing_evidence=routing)


def test_session_telemetry_rejects_swapped_purposes() -> None:
    session = TenantTelemetry(
        "acme_store", InMemoryTelemetrySink(), InMemoryTelemetrySink()
    ).bind_session("session-1")

    with pytest.raises(ValueError, match="telemetry purpose"):
        SessionTelemetry(
            operational=session.routing_evidence,
            routing_evidence=session.operational,
        )


def test_session_telemetry_isolates_concurrent_callers() -> None:
    sink = InMemoryTelemetrySink()
    tenant = TenantTelemetry("acme_store", sink, sink)
    first = tenant.bind_session("session-a").operational
    second = tenant.bind_session("session-b").operational

    def emit(recorder, session_id: str) -> None:
        for sequence in range(50):
            recorder.record(
                {
                    "event": "flow_abandoned",
                    "expected_session": session_id,
                    "sequence": sequence,
                }
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(emit, first, "session-a")
        two = executor.submit(emit, second, "session-b")
        one.result()
        two.result()

    assert len(sink.records) == 100
    assert all(
        record.session_id == record.attributes["expected_session"] for record in sink.records
    )


def test_replay_safe_telemetry_emits_one_event_under_concurrent_redelivery() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational

    def emit(_sequence: int) -> None:
        recorder.record_once(
            "checkout_confirmed:placement-key",
            {"event": "checkout_confirmed", "order_id": "ORD-9001"},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(emit, range(50)))

    assert [record.event for record in sink.records] == ["checkout_confirmed"]
    assert sink.records[0].attributes == {"order_id": "ORD-9001"}


def test_recorder_redacts_sensitive_utterances_before_the_sink() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational

    recorder.record(
        {
            "event": "human_onramp",
            "reason_code": "address_change",
            "utterance": "Send it to 12 Market Street",
        }
    )
    recorder.record(
        {
            "event": "human_onramp",
            "reason_code": "other",
            "utterance": "Call me on +44 7700 900123",
        }
    )

    assert sink.records[0].attributes["utterance"] == "[redacted]"
    assert "+44 7700 900123" not in str(sink.records[1].attributes["utterance"])


def test_recorder_rejects_an_unknown_event_name() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational

    with pytest.raises(ValidationError, match="event"):
        recorder.record({"event": "capabilty_answered"})

    assert sink.records == ()


@pytest.mark.parametrize(
    ("purpose", "event"),
    [
        (TelemetryPurpose.OPERATIONAL, "semantic_route"),
        (TelemetryPurpose.ROUTING_EVIDENCE, "caller_context_closed"),
    ],
)
def test_recorder_rejects_an_event_from_the_wrong_purpose(
    purpose: TelemetryPurpose,
    event: str,
) -> None:
    sink = InMemoryTelemetrySink()
    session = TenantTelemetry("acme_store", sink, sink).bind_session("session-1")
    recorder = (
        session.operational if purpose is TelemetryPurpose.OPERATIONAL else session.routing_evidence
    )

    with pytest.raises(ValidationError, match="purpose"):
        recorder.record({"event": event})

    assert sink.records == ()


def test_inventory_includes_closed_dynamic_event_families() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational
    events = (
        "cart_item_added",
        "cart_item_removed",
        "cart_quantity_set",
        "identity_stepup_ok",
        "refund_stepup_ok",
        "profile_stepup_ok",
    )

    for event in events:
        recorder.record({"event": event})

    assert tuple(record.event for record in sink.records) == events


@pytest.mark.parametrize("value", ["", "  padded", "padded  "])
def test_telemetry_scope_rejects_ambiguous_identifiers(value: str) -> None:
    sink = InMemoryTelemetrySink()
    if value:
        tenant = TenantTelemetry("acme_store", sink, sink)
        with pytest.raises(ValueError):
            tenant.bind_session(value)
    else:
        with pytest.raises(ValueError):
            TenantTelemetry(value, sink, sink)


def test_direct_recorder_construction_cannot_bypass_scope_validation() -> None:
    with pytest.raises(ValueError, match="tenant id"):
        TelemetryRecorder(
            tenant_id=" padded",
            session_id="session-1",
            purpose=TelemetryPurpose.OPERATIONAL,
            sink=InMemoryTelemetrySink(),
        )


def test_record_schema_rejects_non_normalized_scope_from_persisted_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"schema_version": 1, "purpose": "operational", '
        '"tenant_id": " acme_store", "session_id": "session-1", '
        '"event": "caller_context_closed"}\n',
        encoding="utf-8",
    )

    assert JsonlTelemetrySink(path).read_since(0).complete is False


def test_recorder_rejects_non_json_payloads() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational

    with pytest.raises(ValidationError):
        recorder.record({"event": "human_onramp", "value": object()})

    assert sink.records == ()


def test_recorder_rejects_scope_metadata_in_event_attributes() -> None:
    sink = InMemoryTelemetrySink()
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational

    with pytest.raises(ValueError, match="cannot replace envelope fields"):
        recorder.record({"event": "human_onramp", "tenant_id": "other_store"})

    assert sink.records == ()


def test_jsonl_evidence_reads_only_valid_records_after_the_mark(tmp_path: Path) -> None:
    sink = JsonlTelemetrySink(tmp_path / "telemetry.jsonl")
    recorder = TenantTelemetry("acme_store", sink, sink).bind_session("session-1").operational
    recorder.record({"event": "flow_abandoned", "position": "before"})
    mark = sink.mark()
    recorder.record({"event": "flow_abandoned", "position": "after"})

    read = sink.read_since(mark)

    assert read.complete is True
    assert [record.event for record in read.records] == ["flow_abandoned"]
    assert read.records[0].attributes == {"position": "after"}
