"""Support-selection AUTHORIZATION (SECURITY §7d) at the ENGINE level: the assemble-seam gate
that authorizes the ORDER before any pending is minted, plus the call-#15 candidate-scoping
closure (the model can't speak what it never saw).

**Fix 2 posture (the crux of this suite):** a MUTATION (cancel/refund/return) requires RUNG-2 —
the target is session-placed OR owned by the OTP-BOUND identity (`order_mutation_allowed`). A
rung-1 contact-match grant authorizes READS only; it never authorizes a mutation. So an UNBOUND
caller who asks to cancel/refund/return an order they can't yet mutate is routed into the
identity OTP flow to BIND, then the action RESUMES at support_assemble (the model re-proposes,
now the bound owner). A BOUND caller targeting a non-owned / unknown order fails closed with the
ONE combined not-found (existence-oracle). The contact claim is no longer a mutation credential —
`account_contact` is gone from the propose tools.

The pins that matter most: the steer-leak regressions (the refund guardrail RE-MINTS pendings
whose lines speak dollar amounts, so the gate must hold strictly UPSTREAM of the mint), the
prompt-content pins (no unauthorized order data ever enters the support model's context), and the
guest happy-path (detour → OTP bind → resume → cancel commits). Zero network (fake models +
InMemorySaver). Harnesses here are UNAUTHORIZED by construction (no authorize_fixture_orders —
that helper exists for suites pinning post-authorization money logic).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import TEST_OTP, SupportHarness, build_support_engine

from agnostic_market.agents.support.flow import (
    _SUPPORT_COMBINED_NOT_FOUND,
    _SUPPORT_NOT_FOUND_OFFER_HUMAN,
)
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.dtos.orchestration import (
    CancelOrders,
    ExplicitOrderSet,
    RefundOrder,
    ReturnOrder,
)
from agnostic_market.dtos.state import CartLine

_POLICY = make_policy()
_FACTS = TurnFacts()
# ORD-1002 is CUST-002's, "processing" — the canonical guest target. An unverified caller
# references it by ID (the guest path); its key never appears in their prompt.
_CANCEL_1002 = {"propose_cancel": {"order_keys": ["ORD-1002"]}}
_CONTACT_1001 = "+1 555 010 0119"
_CONTACT_1002 = "casey@example.com"  # CUST-002's on-file contact (the OTP recipient after bind)
# The three fixture orders' model-facing details — none may enter an unverified prompt.
_FIXTURE_DETAILS = ("ORD-100", "trail running", "waterproof rain jacket", "merino", "$129", "$179")

# The frontline hands over the production way (request_handover -> support); the deterministic
# gate deliberately does not own every phrasing. Reused by the guest-detour drives below.
_HANDOVER_TO_SUPPORT = {
    "request_handover": {"destination": "support", "reason_code": "cancel_order"}
}


def _harness(
    config_root: Path, reasoning: FakeChatModel, *, thread_id: str, policy=_POLICY, frontline=None
) -> SupportHarness:
    return build_support_engine(
        config_root, policy=policy, reasoning=reasoning, frontline=frontline, thread_id=thread_id
    )


async def _events(engine, text: str) -> list:
    return [e async for e in engine.stream_turn(text, _FACTS)]


def _telemetry(tmp_path: Path) -> list[dict]:
    """Event-shaped telemetry lines only (the file also carries gate-decision records with
    no 'event' key; normalizing here keeps every assertion a plain e["event"] compare)."""
    path = tmp_path / "telemetry.jsonl"
    if not path.exists():
        return []
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [e for e in lines if "event" in e]


def _state_values(harness: SupportHarness, thread_id: str) -> dict:
    return harness.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values


def _active_flow(harness: SupportHarness, thread_id: str) -> str | None:
    return _state_values(harness, thread_id).get("active_flow")


def _bind_cust2(harness: SupportHarness) -> None:
    """Bind the session to CUST-002 (rung-2) — the bound caller whose cross-customer / unknown
    targets exercise the fail-closed combined-not-found path."""
    harness.identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )


def _tool_messages(harness: SupportHarness, thread_id: str) -> list[ToolMessage]:
    return [m for m in _state_values(harness, thread_id)["messages"] if isinstance(m, ToolMessage)]


def _no_pendings(harness: SupportHarness, thread_id: str) -> bool:
    values = _state_values(harness, thread_id)
    return all(
        values.get(f) is None for f in ("pending_cancel", "pending_refund", "pending_return")
    )


def _no_details_spoken(events: list) -> bool:
    """No readback interrupt, and nothing node-authored speaks an amount or item summary
    for the unauthorized order (the leak surface the gate closes)."""
    if any(isinstance(e, InterruptEvent) for e in events):
        return False
    spoken = [e.text for e in events if isinstance(e, SpokenMessageEvent)]
    return not any("$" in t or "rain jacket" in t for t in spoken)


def _spoken(events: list) -> list[tuple[str, str]]:
    return [(event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)]


# --- THE call-#15 pins: the support model never SEES unauthorized order data --------------


async def test_unverified_support_prompt_contains_no_order_data(config_root: Path) -> None:
    # Live call #15: the assemble model's clarify question recited "the trail running
    # shoes, the rain jacket, or the merino hiking socks" to a caller who had FAILED OTP.
    # Structural closure: the data never enters the model's context.
    fake = FakeChatModel(emit_tool_calls=False, record_prompts=True)
    h = _harness(config_root, fake, thread_id="see-1")
    # The caller's own words may mention an item ("rain jacket") — the pins check the
    # FIXTURE-authored details (summaries/ids/totals), which only the prompt could add.
    await _events(h.engine, "cancel my order please")
    assert fake._seen_prompts  # the support assemble model ran
    for prompt in fake._seen_prompts:
        for detail in _FIXTURE_DETAILS:
            assert detail not in prompt
    assert "none verified for this caller yet" in fake._seen_prompts[-1]


async def test_bound_caller_prompt_lists_only_owned_orders(config_root: Path) -> None:
    fake = FakeChatModel(emit_tool_calls=False, record_prompts=True)
    h = _harness(config_root, fake, thread_id="see-2")
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    await _events(h.engine, "cancel my order please")
    prompt = fake._seen_prompts[-1]
    assert "ORD-1001" in prompt and "ORD-1003" in prompt  # CUST-001's own orders
    assert "ORD-1002" not in prompt  # CUST-002's — never rendered
    assert "waterproof rain jacket" not in prompt


# --- Fix 2: an UNBOUND caller's mutation detours into identity (never grants on contact) ----


async def test_unbound_cancel_detours_to_identity_without_leaking_details(
    config_root: Path, tmp_path: Path
) -> None:
    # A guest states an order id and asks to cancel it. The rung-1 contact path is GONE — the
    # flow can't authorize a mutation for an unbound caller, so it routes into the identity OTP
    # flow to verify. Nothing voided, nothing minted, no order details spoken, and the turn
    # leaves the session in the identity flow (the detour). Existence is never revealed.
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=_CANCEL_1002, tool_call_limit=1),
        thread_id="scope-1",
    )
    events = await _events(h.engine, "cancel order ORD-1002")
    assert _no_details_spoken(events)
    assert h.store.cancel_count == 0
    assert _no_pendings(h, "scope-1")
    assert _active_flow(h, "scope-1") == "identity"  # the detour is in flight
    tel = _telemetry(tmp_path)
    assert any(e["event"] == "support_action_needs_identity" for e in tel)
    # The rung-1 grant path never runs: no order was granted, no auth-denial recorded.
    assert not any(e["event"] == "support_order_granted" for e in tel)
    assert not h.identity.order_granted("ORD-1002")


async def test_model_only_order_reference_cannot_cross_identity(config_root: Path) -> None:
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["ORD-1002"]}},
            tool_call_limit=2,
        ),
        thread_id="scope-model-ref",
    )
    await _events(h.engine, "cancel my rain jacket order")
    state = _state_values(h, "scope-model-ref")
    assert state.get("pending_request") is None
    assert state.get("pending_cancel") is None
    assert state.get("active_flow") != "identity"
    assert h.identity.current() is None
    assert h.store.cancel_count == 0


async def test_guest_cancel_verifies_then_resumes_and_commits(
    config_root: Path, tmp_path: Path
) -> None:
    # THE Fix 2 happy path (the live-trace scenario, now SAFE): a guest asks to cancel ORD-1002,
    # detours into identity, verifies via OTP to CUST-002's on-file contact, then resumes from
    # a typed continuation without another model reconstruction call. A guest who cannot
    # complete the OTP never mutates. The frontline hands over the production way.
    frontline = FakeChatModel(
        force_tool="request_handover", canned_args=_HANDOVER_TO_SUPPORT, tool_call_limit=1
    )
    scripted = [
        [("propose_cancel", {"order_keys": ["ORD-1002"]})],  # support assemble -> detour
        [],  # identity clarify turn 1 (no contact yet)
        [("propose_identity", {"contact_claim": _CONTACT_1002})],  # identity assemble turn 2
    ]
    h = _harness(
        config_root,
        FakeChatModel(scripted_calls=scripted),
        thread_id="scope-resume",
        frontline=frontline,
    )
    await _events(h.engine, "cancel order ORD-1002")  # -> identity (asks for contact)
    assert _active_flow(h, "scope-resume") == "identity"
    e1 = await _events(h.engine, _CONTACT_1002)  # -> OTP dispatched
    assert any(isinstance(e, InterruptEvent) and "code" in e.prompt for e in e1)
    e2 = await _events(h.engine, TEST_OTP)  # OTP -> bind -> resume assemble -> readback
    interrupts = [e for e in e2 if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1 and "ORD-1002" in interrupts[0].prompt
    assert h.store.cancel_count == 0  # nothing voided before consent
    assert h.identity.current() is not None  # the session bound (rung-2)
    await _events(h.engine, "yes")
    assert h.store.cancel_count == 1
    assert h.store.order_status("ORD-1002") == "cancelled"
    tel = _telemetry(tmp_path)
    assert any(e["event"] == "identity_bound_for_action" for e in tel)


@pytest.mark.xfail(
    strict=True,
    reason="strong-labelled letter-spelled order IDs do not reach guest mutation detours",
)
@pytest.mark.parametrize(
    ("tool_name", "tool_args", "utterance", "request_type"),
    (
        (
            "propose_cancel",
            {"order_keys": ["ORD-1002"]},
            "Cancel order. O r d one zero zero two.",
            CancelOrders,
        ),
        (
            "propose_return",
            {"order_key": "ORD-1001"},
            "Return order. O r d one zero zero one.",
            ReturnOrder,
        ),
        (
            "propose_refund",
            {"order_key": "ORD-1001", "amount_usd": 20.0, "destination": "original"},
            "Refund $20 for order. O r d one zero zero one.",
            RefundOrder,
        ),
    ),
)
async def test_strong_labelled_stt_enters_each_guest_mutation_identity_detour(
    config_root: Path,
    tool_name: str,
    tool_args: dict,
    utterance: str,
    request_type: type,
) -> None:
    thread_id = f"scope-strong-stt-{tool_name}"
    harness = _harness(
        config_root,
        FakeChatModel(
            force_tool=tool_name,
            canned_args={tool_name: tool_args},
            tool_call_limit=1,
        ),
        thread_id=thread_id,
    )
    await _events(harness.engine, utterance)
    state = _state_values(harness, thread_id)
    assert isinstance(state.get("pending_request"), request_type)
    assert state.get("active_flow") == "identity"
    assert state.get("pending_cancel") is None
    assert state.get("pending_return") is None
    assert state.get("pending_refund") is None
    assert harness.identity.current() is None
    assert harness.store.cancel_count == 0
    assert harness.store.return_count == 0
    assert harness.store.refund_count == 0


async def test_guest_batch_requires_every_deduplicated_target_in_caller_speech(
    config_root: Path,
) -> None:
    thread_id = "scope-batch-all-targets"
    harness = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["ORD-1002", "ORD-1001"]}},
            tool_call_limit=2,
        ),
        thread_id=thread_id,
    )
    await _events(harness.engine, "Cancel order ORD-1002.")
    state = _state_values(harness, thread_id)
    assert state.get("pending_request") is None
    assert state.get("pending_cancel") is None
    assert state.get("active_flow") != "identity"
    assert harness.store.cancel_count == 0


async def test_guest_batch_preserves_all_exact_caller_stated_targets(
    config_root: Path,
) -> None:
    thread_id = "scope-batch-exact-targets"
    harness = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["ORD-1002", "ORD-1001"]}},
            tool_call_limit=1,
        ),
        thread_id=thread_id,
    )
    await _events(harness.engine, "Cancel order ORD-1002 and ORD-1001.")
    request = _state_values(harness, thread_id).get("pending_request")
    assert isinstance(request, CancelOrders)
    assert isinstance(request.target, ExplicitOrderSet)
    assert request.target.order_refs == ("ORD-1002", "ORD-1001")
    assert harness.store.cancel_count == 0


@pytest.mark.parametrize(
    ("tool_name", "tool_args", "utterance", "reason_code"),
    (
        (
            "propose_cancel",
            {"order_keys": ["ORD-1002"]},
            "Cancel my rain jacket order.",
            "cancel_order",
        ),
        (
            "propose_return",
            {"order_key": "ORD-1001"},
            "Return my trail running shoes.",
            "refund",
        ),
        (
            "propose_refund",
            {"order_key": "ORD-1001", "amount_usd": 20.0, "destination": "original"},
            "Refund $20 for my trail running shoes.",
            "refund",
        ),
    ),
)
async def test_model_only_target_fails_closed_for_every_guest_mutation_consumer(
    config_root: Path,
    tool_name: str,
    tool_args: dict,
    utterance: str,
    reason_code: str,
) -> None:
    thread_id = f"scope-model-target-{tool_name}"
    harness = _harness(
        config_root,
        FakeChatModel(
            force_tool=tool_name,
            canned_args={tool_name: tool_args},
            tool_call_limit=2,
        ),
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {
                    "destination": "support",
                    "reason_code": reason_code,
                }
            },
            tool_call_limit=1,
        ),
        thread_id=thread_id,
    )
    await _events(harness.engine, utterance)
    state = _state_values(harness, thread_id)
    assert state.get("pending_request") is None
    assert state.get("pending_cancel") is None
    assert state.get("pending_return") is None
    assert state.get("pending_refund") is None
    assert harness.identity.current() is None
    assert harness.store.cancel_count == 0
    assert harness.store.return_count == 0
    assert harness.store.refund_count == 0


async def _drive_explicit_action_continuation(
    config_root: Path,
    *,
    tool_name: str,
    tool_args: dict,
    utterance: str,
    thread_id: str,
) -> tuple[SupportHarness, FakeChatModel, list]:
    reasoning = FakeChatModel(
        scripted_calls=[
            [(tool_name, tool_args)],
            [],
            [("propose_identity", {"contact_claim": _CONTACT_1001})],
        ]
    )
    h = _harness(
        config_root,
        reasoning,
        thread_id=thread_id,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={"request_handover": {"destination": "support", "reason_code": "refund"}},
            tool_call_limit=1,
        ),
    )
    await _events(h.engine, utterance)
    assert _state_values(h, thread_id).get("pending_request") is not None
    dispatched = await _events(h.engine, _CONTACT_1001)
    assert any(
        isinstance(event, InterruptEvent) and "6-digit code" in event.prompt for event in dispatched
    )
    model_calls_before_continuation = reasoning._tool_calls_made
    old_thread_id = h.engine.thread_id
    events = await _events(h.engine, TEST_OTP)
    assert h.identity.current() is not None
    assert h.engine.thread_id != old_thread_id
    assert reasoning._tool_calls_made == model_calls_before_continuation
    return h, reasoning, events


async def test_explicit_refund_continuation_resolves_without_model_replay(
    config_root: Path,
) -> None:
    h, _reasoning, events = await _drive_explicit_action_continuation(
        config_root,
        tool_name="propose_refund",
        tool_args={"order_key": "ORD-1001", "amount_usd": 20.0, "destination": "original"},
        utterance="refund $20 for order ORD-1001",
        thread_id="scope-refund-continuation",
    )
    prompts = [event.prompt for event in events if isinstance(event, InterruptEvent)]
    assert len(prompts) == 1 and "$20.00" in prompts[0]
    await _events(h.engine, "yes")
    assert h.store.refund_count == 1


async def test_new_instrument_refund_continuation_uses_bound_customer_reference(
    config_root: Path,
) -> None:
    h, _reasoning, events = await _drive_explicit_action_continuation(
        config_root,
        tool_name="propose_refund",
        tool_args={
            "order_key": "ORD-1001",
            "amount_usd": 20.0,
            "destination": "new_instrument",
        },
        utterance="refund $20 for order ORD-1001 to a different card",
        thread_id="scope-new-instrument-continuation",
    )
    owner_ref = h.payment_instruments.new_instrument_ref("CUST-001")
    other_ref = h.payment_instruments.new_instrument_ref("CUST-002")
    assert owner_ref is not None and other_ref is not None
    prompts = [event.prompt for event in events if isinstance(event, InterruptEvent)]
    assert len(prompts) == 1
    assert owner_ref in prompts[0]
    assert other_ref not in prompts[0]

    completed = await _events(h.engine, "yes")
    assert h.store.refund_count == 1
    spoken = [
        event.text
        for event in completed
        if isinstance(event, SpokenMessageEvent) and event.node == "support_place"
    ]
    assert len(spoken) == 1
    assert owner_ref in spoken[0]
    assert other_ref not in spoken[0]


async def test_model_only_refund_amount_cannot_reach_confirmation(
    config_root: Path,
) -> None:
    h, _reasoning, events = await _drive_explicit_action_continuation(
        config_root,
        tool_name="propose_refund",
        tool_args={"order_key": "ORD-1001", "amount_usd": 20.0, "destination": "original"},
        utterance="refund order ORD-1001",
        thread_id="scope-refund-model-amount",
    )
    assert not any(isinstance(event, InterruptEvent) for event in events)
    assert any(
        "what amount" in event.text.lower()
        for event in events
        if isinstance(event, SpokenMessageEvent)
    )
    assert h.store.refund_count == 0


async def test_explicit_return_continuation_resolves_without_model_replay(
    config_root: Path,
) -> None:
    h, _reasoning, events = await _drive_explicit_action_continuation(
        config_root,
        tool_name="propose_return",
        tool_args={"order_key": "ORD-1001"},
        utterance="return order ORD-1001",
        thread_id="scope-return-continuation",
    )
    prompts = [event.prompt for event in events if isinstance(event, InterruptEvent)]
    assert len(prompts) == 1 and "ORD-1001" in prompts[0]
    await _events(h.engine, "yes")
    assert h.store.return_count == 1


async def test_failed_otp_on_action_detour_leaves_zero_resume_intent(
    config_root: Path,
) -> None:
    # SECURITY: a FAILED verification during an explicit-order mutation detour must leave ZERO
    # continuation intent is cleared on the handover so a later turn
    # can never silently resume the cancel. OTP exhausted -> human, nothing voided.
    frontline = FakeChatModel(
        force_tool="request_handover", canned_args=_HANDOVER_TO_SUPPORT, tool_call_limit=1
    )
    scripted = [
        [("propose_cancel", {"order_keys": ["ORD-1002"]})],  # detour
        [],
        [("propose_identity", {"contact_claim": _CONTACT_1002})],  # bind attempt
    ]
    h = _harness(
        config_root,
        FakeChatModel(scripted_calls=scripted),
        thread_id="scope-fail",
        frontline=frontline,
        policy=make_policy(otp_max_attempts=1),  # one wrong code exhausts
    )
    await _events(h.engine, "cancel order ORD-1002")
    await _events(h.engine, _CONTACT_1002)  # OTP dispatched
    await _events(h.engine, "000000")  # WRONG code -> exhausted -> human handover
    assert h.store.cancel_count == 0
    assert _state_values(h, "scope-fail").get("pending_request") is None
    assert _state_values(h, "scope-fail").get("pending_cancel") is None


async def test_guest_batch_detours_to_identity(config_root: Path, tmp_path: Path) -> None:
    # A guest explicit batch ("cancel ORD-1002 and this placed one") also detours — no target is
    # rung-2 authorized, so the whole proposal enters identity; nothing voided.
    args = {"propose_cancel": {"order_keys": ["ORD-1002", "ORD-9001"]}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-batch-guest",
    )
    # A placed order IS rung-2 (session-placed) — but ORD-1002 is not, and ANY unauthorized
    # target short-circuits the batch into the detour (fail-closed, no partial batch).
    h.store.place_cart(
        "guest-batch-seed",
        lines=[CartLine(sku="SKU-SOCK-01", name="wool socks", price_usd=24.0, quantity=1)],
        total_usd=24.0,
    )
    events = await _events(h.engine, "cancel order ORD-1002 and order ORD-9001")
    assert any(isinstance(e, InterruptEvent) for e in events)
    assert h.store.cancel_count == 0
    assert _active_flow(h, "scope-batch-guest") == "identity"
    assert any(e["event"] == "support_action_needs_identity" for e in _telemetry(tmp_path))


# --- a BOUND caller targeting a non-owned / unknown order: fail closed (existence-oracle) ----


async def test_bound_cross_customer_target_is_combined_not_found(
    config_root: Path, tmp_path: Path
) -> None:
    # A caller BOUND to CUST-002 targets ORD-1001 (CUST-001). Bound, so no detour — but not
    # theirs, so the ONE combined not-found (never reveals another account owns it), no void.
    args = {"propose_cancel": {"order_keys": ["ORD-1001"]}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-xcust",
    )
    _bind_cust2(h)
    events = await _events(h.engine, "cancel order ORD-1001")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert h.store.cancel_count == 0
    assert _no_pendings(h, "scope-xcust")
    denied = [e for e in _telemetry(tmp_path) if e["event"] == "support_auth_denied"]
    assert denied == [{"event": "support_auth_denied", "attempt": 1}]
    assert _spoken(events) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]


async def test_bound_unknown_order_id_is_combined_not_found_and_never_logged(
    config_root: Path, tmp_path: Path
) -> None:
    # A BOUND caller probes an unknown id: unknown-order collapses into the same combined
    # not-found as a cross-customer order, and the raw stated id (caller free text) never reaches
    # telemetry — events carry order_id_known: False instead.
    args = {"propose_cancel": {"order_keys": ["ORD-9999"]}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-4",
    )
    _bind_cust2(h)
    events = await _events(h.engine, "cancel order ORD-9999")
    assert _no_details_spoken(events)
    assert _no_pendings(h, "scope-4")
    tel = _telemetry(tmp_path)
    denied = [e for e in tel if e["event"] == "support_auth_denied"]
    assert denied == [{"event": "support_auth_denied", "attempt": 1}]
    for event in tel:
        assert not any(isinstance(v, str) and "ORD-9999" in v for v in event.values())
    assert _spoken(events) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]


async def test_bound_second_denial_offers_human(config_root: Path, tmp_path: Path) -> None:
    # The deterministic escalation on-ramp: a BOUND caller's 2nd failed target for the same order
    # switches the corrective to one that OFFERS a person (the caller accepting rides the
    # existing entry-router escape_human path). Unbound callers never reach this — they detour.
    cross = {"order_keys": ["ORD-1001"]}  # CUST-001's, not this bound CUST-002 caller's
    scripted = [
        [("propose_cancel", dict(cross))],
        [("propose_cancel", dict(cross))],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-7")
    _bind_cust2(h)
    first = await _events(h.engine, "cancel order ORD-1001")
    second = await _events(h.engine, "yes really cancel it")
    attempts = [e["attempt"] for e in _telemetry(tmp_path) if e["event"] == "support_auth_denied"]
    assert attempts == [1, 2]
    assert _spoken(first) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]
    assert _spoken(second) == [("support_clarify", _SUPPORT_NOT_FOUND_OFFER_HUMAN)]
    assert _no_pendings(h, "scope-7")


async def test_bound_denial_offer_threshold_tracks_the_policy_knob(config_root: Path) -> None:
    # auth_denials_before_human_offer is config-driven: raised to 3, the 2nd denial still
    # speaks the plain combined-not-found (the human offer waits for the 3rd).
    cross = {"order_keys": ["ORD-1001"]}
    scripted = [[("propose_cancel", dict(cross))], [("propose_cancel", dict(cross))]]
    h = _harness(
        config_root,
        FakeChatModel(scripted_calls=scripted),
        thread_id="scope-denial3",
        policy=_POLICY.model_copy(update={"auth_denials_before_human_offer": 3}),
    )
    _bind_cust2(h)
    first = await _events(h.engine, "cancel order ORD-1001")
    second = await _events(h.engine, "yes really cancel it")
    assert _spoken(first) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]
    assert _spoken(second) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "utterance"),
    [
        (
            "propose_cancel",
            {"order_keys": ["ORD-1001"]},
            "cancel order ORD-1001",
        ),
        (
            "propose_return",
            {"order_key": "ORD-1001"},
            "return order ORD-1001",
        ),
        (
            "propose_refund",
            {
                "order_key": "ORD-1001",
                "amount_usd": 10.0,
                "destination": "original",
            },
            "refund ten dollars on order ORD-1001 to my original payment method",
        ),
    ],
)
async def test_one_caller_turn_consumes_at_most_one_order_denial(
    config_root: Path,
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    utterance: str,
) -> None:
    repeated = [(tool_name, dict(arguments))]
    reasoning = FakeChatModel(scripted_calls=[repeated, repeated])
    thread_id = f"one-denial-{tool_name}"
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {
                "destination": "support",
                "reason_code": "cancel_order" if tool_name == "propose_cancel" else "refund",
            }
        },
        tool_call_limit=1,
    )
    harness = _harness(
        config_root,
        reasoning,
        thread_id=thread_id,
        frontline=frontline,
    )
    _bind_cust2(harness)

    first = await _events(harness.engine, utterance)

    assert reasoning.invoke_count == 1
    assert [
        event["attempt"]
        for event in _telemetry(tmp_path)
        if event["event"] == "support_auth_denied"
    ] == [1]
    assert _spoken(first) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]
    assert not any(isinstance(event, TokenEvent) for event in first)
    assert _active_flow(harness, thread_id) == "support"
    assert _no_pendings(harness, thread_id)
    assert (
        harness.store.cancel_count == harness.store.return_count == harness.store.refund_count == 0
    )

    second = await _events(harness.engine, "Please try that order again")

    assert reasoning.invoke_count == 2
    assert [
        event["attempt"]
        for event in _telemetry(tmp_path)
        if event["event"] == "support_auth_denied"
    ] == [1, 2]
    assert _spoken(second) == [("support_clarify", _SUPPORT_NOT_FOUND_OFFER_HUMAN)]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "utterance"),
    [
        (
            "propose_cancel",
            {"order_keys": ["ORD-1002"]},
            "cancel my order",
        ),
        (
            "propose_return",
            {"order_key": "ORD-1002"},
            "return my order",
        ),
        (
            "propose_refund",
            {
                "order_key": "ORD-1002",
                "amount_usd": 10.0,
                "destination": "original",
            },
            "refund ten dollars to my original payment method",
        ),
    ],
)
async def test_unbound_model_only_order_target_asks_for_caller_stated_number(
    config_root: Path,
    tool_name: str,
    arguments: dict[str, object],
    utterance: str,
) -> None:
    reasoning = FakeChatModel(
        force_tool=tool_name,
        canned_args={tool_name: arguments},
        tool_call_limit=99,
    )
    thread_id = f"missing-stated-{tool_name}"
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {
                "destination": "support",
                "reason_code": "cancel_order" if tool_name == "propose_cancel" else "refund",
            }
        },
        tool_call_limit=1,
    )
    harness = _harness(
        config_root,
        reasoning,
        thread_id=thread_id,
        frontline=frontline,
    )

    events = await _events(harness.engine, utterance)

    assert reasoning.invoke_count == 1
    assert _spoken(events) == [
        ("support_clarify", "What is the order number, for example ORD-1234?")
    ]
    assert not any(isinstance(event, (TokenEvent, InterruptEvent)) for event in events)
    state = _state_values(harness, thread_id)
    assert state["active_flow"] == "support"
    assert state.get("pending_request") is None
    assert state.get("pending_clarification") is None
    assert harness.identity.current() is None
    assert harness.otp.dispatch_count == 0
    assert _no_pendings(harness, thread_id)


async def test_cancel_batch_authorizes_all_targets_before_minting_any_pending(
    config_root: Path, tmp_path: Path
) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_cancel",
        canned_args={"propose_cancel": {"order_keys": ["ORD-1002", "ORD-1001"]}},
        tool_call_limit=99,
    )
    thread_id = "cancel-partial-denied"
    harness = _harness(config_root, reasoning, thread_id=thread_id)
    _bind_cust2(harness)

    events = await _events(
        harness.engine,
        "cancel order ORD-1002 and order ORD-1001",
    )

    assert reasoning.invoke_count == 1
    assert _spoken(events) == [("support_clarify", _SUPPORT_COMBINED_NOT_FOUND)]
    assert [
        event["attempt"]
        for event in _telemetry(tmp_path)
        if event["event"] == "support_auth_denied"
    ] == [1]
    assert _state_values(harness, thread_id).get("pending_cancel") is None
    assert harness.store.cancel_count == 0


# --- fast paths: session-placed / bound owner (rung-2 sails, no detour) --------------------


async def test_session_placed_order_needs_no_verification(
    config_root: Path, tmp_path: Path
) -> None:
    # Key "4" = the first PLACED order (3 fixture orders precede it in the full list); a
    # session-placed order IS rung-2 by construction, so it cancels with no detour.
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["4"]}},
            tool_call_limit=1,
        ),
        thread_id="scope-9",
    )
    placed = h.store.place_cart(
        "pk-1",
        lines=[CartLine(sku="SKU-SOCK-01", name="wool socks", price_usd=24.0, quantity=1)],
        total_usd=24.0,
    )
    events = await _events(h.engine, "cancel that order I just placed")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert placed.order_id in interrupts[0].prompt
    tel = _telemetry(tmp_path)
    assert not any(e["event"].startswith("support_auth_") for e in tel)
    assert not any(e["event"] == "support_action_needs_identity" for e in tel)
    assert any(
        e["event"] == "support_action_authorized" and e["order_id"] == placed.order_id for e in tel
    )


async def test_bound_identity_owned_sails_non_owned_detours_or_denies(
    config_root: Path, tmp_path: Path
) -> None:
    bound = BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119")
    # Owned: ORD-1001 (key "1", CUST-001, shipped) is LISTED — a return proposal by key reaches
    # its readback with no verification asked (rung-2 bound owner).
    h1 = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_return",
            canned_args={"propose_return": {"order_key": "1"}},
            tool_call_limit=1,
        ),
        thread_id="scope-10a",
    )
    h1.identity.bind(bound)
    e1 = await _events(h1.engine, "I need to return this order")
    assert any(isinstance(e, InterruptEvent) for e in e1)
    assert not any(e["event"].startswith("support_auth_") for e in _telemetry(tmp_path))


async def test_rung1_read_grant_does_not_authorize_a_mutation(
    config_root: Path, tmp_path: Path
) -> None:
    # THE crux pin: a rung-1 grant earned earlier this session (an order_status guest lookup)
    # authorizes READS — the order is LISTED (key "2") — but NOT a mutation. A cancel on it
    # detours into identity; nothing is voided (Fix 2: rung-1 != mutation authority).
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["2"]}},
            tool_call_limit=1,
        ),
        thread_id="scope-11",
    )
    h.identity.grant_order("ORD-1002")  # rung-1 READ grant only
    events = await _events(h.engine, "cancel order ORD-1002")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert h.store.cancel_count == 0
    assert _active_flow(h, "scope-11") == "identity"
    assert any(e["event"] == "support_action_needs_identity" for e in _telemetry(tmp_path))


# --- the KEY steer-leak pins (the reason the gate sits at assemble) -----------------------


async def test_unshipped_refund_steer_does_not_leak_for_unauthorized_order(
    config_root: Path, tmp_path: Path
) -> None:
    # A FULL refund-to-original on the unauthorized processing ORD-1002 would trip the
    # cancel-steer, whose line speaks "the full $129.00 goes back". The gate must hold
    # UPSTREAM: no steer event, no dollar spoken, no pending of either kind — and (Fix 2) an
    # unbound caller detours to identity instead.
    args = {
        "propose_refund": {"order_key": "ORD-1002", "amount_usd": 129.0, "destination": "original"}
    }
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_refund", canned_args=args, tool_call_limit=1),
        thread_id="scope-12",
    )
    events = await _events(h.engine, "I want my money back for order ORD-1002")
    assert _no_details_spoken(events)
    assert _no_pendings(h, "scope-12")
    tel = _telemetry(tmp_path)
    assert not any(e["event"] == "refund_steered_to_cancel" for e in tel)
    assert any(e["event"] == "support_action_needs_identity" for e in tel)
    assert h.store.cancel_count == 0 and h.store.refund_count == 0


async def test_shipped_refund_return_steer_does_not_leak_for_unauthorized_order(
    config_root: Path, tmp_path: Path
) -> None:
    # A $150 refund on the unauthorized shipped ORD-1001, over a tightened returnless line,
    # would trip the return-first steer ("the $150.00 refund is issued once the return is
    # set up"). Same pin: gate upstream, nothing steered, nothing spoken, nothing minted.
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 50.0})
    h = build_support_engine(
        config_root,
        policy=tight,
        reasoning=FakeChatModel(
            force_tool="propose_refund",
            canned_args={
                "propose_refund": {
                    "order_key": "ORD-1001",
                    "amount_usd": 150.0,
                    "destination": "original",
                }
            },
            tool_call_limit=1,
        ),
        thread_id="scope-13",
    )
    events = await _events(h.engine, "Refund $150 for order ORD-1001 to my original card")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert not any("$" in e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert _no_pendings(h, "scope-13")
    tel = _telemetry(tmp_path)
    assert not any(e["event"] == "refund_steered_to_return" for e in tel)
    assert h.store.return_count == 0 and h.store.refund_count == 0


async def test_return_gated_at_assemble(config_root: Path, tmp_path: Path) -> None:
    # An unbound return proposal detours to identity (no rung-2), nothing minted, nothing done.
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_return",
            canned_args={"propose_return": {"order_key": "ORD-1001"}},
            tool_call_limit=1,
        ),
        thread_id="scope-14",
    )
    events = await _events(h.engine, "I need to return order ORD-1001")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert _no_pendings(h, "scope-14")
    assert h.store.return_count == 0
    assert any(e["event"] == "support_action_needs_identity" for e in _telemetry(tmp_path))


# --- scope boundaries + PII ----------------------------------------------------------------


def _profile_harness(
    config_root: Path,
    thread_id: str,
    *,
    new_value: str = "555 111 2222",
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=_POLICY,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {"destination": "support", "reason_code": "contact_change"}
            },
            tool_call_limit=1,
        ),
        reasoning=FakeChatModel(
            force_tool="propose_profile_change",
            canned_args={"propose_profile_change": {"field": "contact", "new_value": new_value}},
            tool_call_limit=1,
        ),
        thread_id=thread_id,
    )


async def test_profile_change_not_gated_by_order_scope(config_root: Path, tmp_path: Path) -> None:
    # Profile changes target the ACCOUNT (L2 step-up on the old factor), not an order — the
    # ORDER gate never runs. A BOUND caller (Fix 5 M-B: profile requires a bound identity) whose
    # customer has a profile proceeds; the OTP dispatch proves the flow was not order-gated.
    h = _profile_harness(config_root, thread_id="scope-15")
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    await _events(h.engine, "Change my phone number to 555 111 2222")
    assert h.otp.dispatch_count == 1  # the step-up chain ran — the flow was not order-gated
    tel = _telemetry(tmp_path)
    assert not any(e["event"].startswith("support_auth_") for e in tel)
    assert not any(e["event"] == "support_action_authorized" for e in tel)


async def test_unbound_profile_change_detours_to_identity(
    config_root: Path, tmp_path: Path
) -> None:
    # Fix 5 M-B: a profile change is account-scoped, so an UNBOUND caller cannot mutate — it
    # detours into the identity OTP flow (no profile step-up dispatched, nothing changed).
    h = _profile_harness(config_root, thread_id="scope-prof-unbound")
    await _events(h.engine, "Change my phone number to 555 111 2222")
    assert _active_flow(h, "scope-prof-unbound") == "identity"
    assert h.otp.dispatch_count == 0  # no profile OTP — we're verifying identity first
    assert h.profile.change_count == 0
    assert any(e["event"] == "support_action_needs_identity" for e in _telemetry(tmp_path))


async def test_model_only_profile_value_cannot_cross_identity(config_root: Path) -> None:
    h = _profile_harness(config_root, thread_id="scope-prof-model-value")
    events = await _events(h.engine, "I have a new phone number for my account")
    state = _state_values(h, "scope-prof-model-value")
    assert _spoken(events) == [
        (
            "support_clarify",
            "What new delivery address or contact number would you like to use?",
        )
    ]
    assert not any(isinstance(event, TokenEvent) for event in events)
    assert state.get("pending_request") is None
    assert state.get("pending_profile_change") is None
    assert state.get("active_flow") == "support"
    assert state.get("pending_clarification") is None
    assert h.otp.dispatch_count == 0
    assert h.profile.change_count == 0


async def test_order_number_cannot_become_a_model_proposed_contact(
    config_root: Path,
) -> None:
    h = _profile_harness(
        config_root,
        thread_id="scope-prof-order-number",
        new_value="1002",
    )
    await _events(h.engine, "Change the contact for order 1002")
    state = _state_values(h, "scope-prof-order-number")
    assert state.get("pending_request") is None
    assert state.get("pending_profile_change") is None
    assert state.get("active_flow") != "identity"
    assert h.otp.dispatch_count == 0
    assert h.profile.change_count == 0


async def test_bound_customer_without_profile_fails_closed(
    config_root: Path, tmp_path: Path
) -> None:
    # A caller bound to a customer with NO profile on file (CUST-002) is refused with the
    # neutral not-available line — never a fallback to CUST-001's profile, never an oracle.
    h = _profile_harness(config_root, thread_id="scope-prof-noprofile")
    h.identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    events = await _events(h.engine, "Change my phone number to 555 111 2222")
    assert h.otp.dispatch_count == 0
    assert h.profile.change_count == 0
    assert [e.text for e in events if isinstance(e, SpokenMessageEvent)] == [
        "I can't continue with automated assistance on this call. "
        "Please contact the store directly for further help."
    ]
    state = _state_values(h, "scope-prof-noprofile")
    assert state.get("active_flow") is None
    assert state.get("pending_profile_change") is None
    assert state.get("handover") is None
    assert state.get("automation_terminal") is True
    telemetry = _telemetry(tmp_path)
    assert sum(e["event"] == "profile_change_denied" for e in telemetry) == 1
    assert sum(e["event"] == "human_onramp" for e in telemetry) == 1


async def test_needs_identity_detour_never_echoes_a_contact(
    config_root: Path, tmp_path: Path
) -> None:
    # The honest PII invariant, now trivial by construction: the contact claim is no longer a
    # mutation credential (account_contact is gone from propose_cancel), so a guest cancel that
    # detours to identity never carries a contact into a support ToolMessage or telemetry.
    frontline = FakeChatModel(
        force_tool="request_handover", canned_args=_HANDOVER_TO_SUPPORT, tool_call_limit=1
    )
    scripted = [
        [("propose_cancel", {"order_keys": ["ORD-1002"]})],  # -> detour (no contact anywhere)
        [],
    ]
    h = _harness(
        config_root,
        FakeChatModel(scripted_calls=scripted),
        thread_id="scope-16",
        frontline=frontline,
    )
    await _events(h.engine, "cancel order ORD-1002")
    for event in _telemetry(tmp_path):
        assert not any(isinstance(v, str) and _CONTACT_1002 in v for v in event.values())
    tool_msgs = _tool_messages(h, "scope-16")
    assert not any(_CONTACT_1002 in str(m.content) for m in tool_msgs)
