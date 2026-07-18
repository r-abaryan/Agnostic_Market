"""Support-selection scoping (SECURITY §7d, the P7 follow-up) at the ENGINE level: the
assemble-seam gate that authorizes the ORDER before any pending is minted, and the
call-#15 closure — the candidate list the support model SEES is scoped to authorized
orders (its clarify speech once recited the unscoped list to an unverified caller; the
model can't speak what it never saw).

The pins that matter most: the steer-leak regressions (the refund guardrail RE-MINTS
pendings whose lines speak dollar amounts, so the gate must hold strictly UPSTREAM of the
mint) and the prompt-content pins (no unauthorized order data ever enters the support
model's context). An unverified caller acts on an order by STATING its number (the guest
path: order_key carries the caller's stated id, code-resolved fail-closed). Zero network
(fake models + InMemorySaver). Harnesses here are UNAUTHORIZED by construction (no
authorize_fixture_orders — that helper exists for suites pinning post-authorization money
logic).
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import ToolMessage
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import SupportHarness, build_support_engine

from agnostic_market.agents.support.flow import (
    _SUPPORT_ASK_CONTACT,
    _SUPPORT_ASK_ORDER_AND_CONTACT,
    _SUPPORT_COMBINED_NOT_FOUND,
    _SUPPORT_NOT_FOUND_OFFER_HUMAN,
)
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.dtos.state import CartLine

_POLICY = make_policy()
_FACTS = TurnFacts()
# ORD-1002 is CUST-002's, "processing" — the canonical UNOWNED target. An unverified
# caller references it by ID (the guest path); its key never appears in their prompt.
_CANCEL_1002 = {"propose_cancel": {"order_key": "ORD-1002"}}
_CONTACT_1002 = "casey@example.com"
# The three fixture orders' model-facing details — none may enter an unverified prompt.
_FIXTURE_DETAILS = ("ORD-100", "trail running", "waterproof rain jacket", "merino", "$129", "$179")


def _harness(
    config_root: Path, reasoning: FakeChatModel, *, thread_id: str, policy=_POLICY
) -> SupportHarness:
    return build_support_engine(
        config_root, policy=policy, reasoning=reasoning, thread_id=thread_id
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
    return harness.engine._graph.get_state(
        {"configurable": {"thread_id": thread_id}}
    ).values


def _tool_messages(harness: SupportHarness, thread_id: str) -> list[ToolMessage]:
    return [m for m in _state_values(harness, thread_id)["messages"] if isinstance(m, ToolMessage)]


def _no_pendings(harness: SupportHarness, thread_id: str) -> bool:
    values = _state_values(harness, thread_id)
    return all(
        values.get(f) is None
        for f in ("pending_cancel", "pending_refund", "pending_return")
    )


def _no_details_spoken(events: list) -> bool:
    """No readback interrupt, and nothing node-authored speaks an amount or item summary
    for the unauthorized order (the leak surface the gate closes)."""
    if any(isinstance(e, InterruptEvent) for e in events):
        return False
    spoken = [e.text for e in events if isinstance(e, SpokenMessageEvent)]
    return not any("$" in t or "rain jacket" in t for t in spoken)


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


# --- the gate: ask / deny / grant (guest path — the caller STATES the order number) --------


async def test_unauthorized_cancel_asks_for_contact_without_leaking_details(
    config_root: Path, tmp_path: Path
) -> None:
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=_CANCEL_1002, tool_call_limit=1),
        thread_id="scope-1",
    )
    events = await _events(h.engine, "cancel my rain jacket order")
    assert _no_details_spoken(events)
    assert h.store.cancel_count == 0
    assert _no_pendings(h, "scope-1")
    tel = _telemetry(tmp_path)
    assert any(
        e["event"] == "support_auth_required" and e.get("order_id") == "ORD-1002" for e in tel
    )
    # The corrective ToolMessage answered the propose call (no dangling tool_use) and is
    # the ASK string — the id resolved, only the contact is missing.
    assert any(m.content == _SUPPORT_ASK_CONTACT for m in _tool_messages(h, "scope-1"))


async def test_contact_match_grants_then_proceeds_to_readback(
    config_root: Path, tmp_path: Path
) -> None:
    # Turn 1: propose by stated id without a claim -> the gate asks (the empty script entry
    # is the ask-turn: the model speaks its question, no tool call). Turn 2: the caller's
    # contact rides the re-propose -> rung-1 grant -> the cancel readback.
    scripted = [
        [("propose_cancel", {"order_key": "ORD-1002"})],
        [],
        [("propose_cancel", {"order_key": "ORD-1002", "account_contact": _CONTACT_1002})],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-2")
    first = await _events(h.engine, "cancel my rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in first)
    events = await _events(h.engine, _CONTACT_1002)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt
    assert h.identity.order_granted("ORD-1002")
    tel = _telemetry(tmp_path)
    assert any(
        e["event"] == "support_order_granted"
        and e["order_id"] == "ORD-1002"
        and e["method"] == "contact_match"
        for e in tel
    )
    # The granted order cancels on a committed yes — the authorized path is unchanged.
    await _events(h.engine, "yes")
    assert h.store.cancel_count == 1


async def test_wrong_contact_returns_combined_not_found(
    config_root: Path, tmp_path: Path
) -> None:
    args = {"propose_cancel": {"order_key": "ORD-1002", "account_contact": "mallory@example.com"}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-3",
    )
    events = await _events(h.engine, "cancel my rain jacket order")
    assert _no_details_spoken(events)
    assert _no_pendings(h, "scope-3")
    denied = [e for e in _telemetry(tmp_path) if e["event"] == "support_auth_denied"]
    assert denied and denied[0].get("order_id") == "ORD-1002" and denied[0]["attempt"] == 1
    # ONE combined line — never which detail failed, never order details.
    assert any(m.content == _SUPPORT_COMBINED_NOT_FOUND for m in _tool_messages(h, "scope-3"))


async def test_unknown_order_id_is_combined_not_found_and_never_logged(
    config_root: Path, tmp_path: Path
) -> None:
    # An id probe with a valid contact learns NOTHING: unknown-order + claim collapses into
    # the same combined not-found as a wrong pair, and the raw stated id (caller free text)
    # never reaches telemetry — events carry order_id_known: False instead.
    args = {"propose_cancel": {"order_key": "ORD-9999", "account_contact": _CONTACT_1002}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-4",
    )
    events = await _events(h.engine, "cancel order ORD-9999")
    assert _no_details_spoken(events)
    assert _no_pendings(h, "scope-4")
    assert not h.identity.order_granted("ORD-9999")
    tel = _telemetry(tmp_path)
    denied = [e for e in tel if e["event"] == "support_auth_denied"]
    assert denied and denied[0]["order_id_known"] is False and "order_id" not in denied[0]
    for event in tel:
        assert not any(isinstance(v, str) and "ORD-9999" in v for v in event.values())
    assert any(m.content == _SUPPORT_COMBINED_NOT_FOUND for m in _tool_messages(h, "scope-4"))


async def test_unresolved_reference_without_claim_asks_for_order_number(
    config_root: Path,
) -> None:
    # A hallucinated key with nothing verified: the corrective asks for the order number
    # AND stops — never "valid order numbers are..." (that would enumerate).
    args = {"propose_cancel": {"order_key": "1"}}
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=args, tool_call_limit=1),
        thread_id="scope-5",
    )
    await _events(h.engine, "cancel my order")
    assert _no_pendings(h, "scope-5")
    assert any(
        m.content == _SUPPORT_ASK_ORDER_AND_CONTACT for m in _tool_messages(h, "scope-5")
    )


async def test_wrong_then_right_contact_grants(config_root: Path, tmp_path: Path) -> None:
    # A denial must not poison the next attempt: wrong claim -> denied; right claim -> grant.
    scripted = [
        [("propose_cancel", {"order_key": "ORD-1002", "account_contact": "mallory@example.com"})],
        [],
        [("propose_cancel", {"order_key": "ORD-1002", "account_contact": _CONTACT_1002})],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-6")
    await _events(h.engine, "cancel my rain jacket order")
    events = await _events(h.engine, _CONTACT_1002)
    assert any(isinstance(e, InterruptEvent) for e in events)  # reached the readback
    tel = _telemetry(tmp_path)
    assert any(e["event"] == "support_auth_denied" for e in tel)
    assert any(e["event"] == "support_order_granted" for e in tel)


async def test_second_denial_offers_human(config_root: Path, tmp_path: Path) -> None:
    # The deterministic escalation on-ramp: the 2nd failed match for the same order switches
    # the corrective line to one that OFFERS a person (the caller accepting rides the
    # existing entry-router escape_human path).
    wrong = {"order_key": "ORD-1002", "account_contact": "mallory@example.com"}
    scripted = [
        [("propose_cancel", dict(wrong))],
        [],
        [("propose_cancel", dict(wrong))],
        [],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-7")
    await _events(h.engine, "cancel my rain jacket order")
    await _events(h.engine, "it's mallory at example dot com")
    attempts = [
        e["attempt"] for e in _telemetry(tmp_path) if e["event"] == "support_auth_denied"
    ]
    assert attempts == [1, 2]
    assert any(
        m.content == _SUPPORT_NOT_FOUND_OFFER_HUMAN for m in _tool_messages(h, "scope-7")
    )
    assert _no_pendings(h, "scope-7")


async def test_denial_offer_threshold_tracks_the_policy_knob(
    config_root: Path,
) -> None:
    # auth_denials_before_human_offer is config-driven: raised to 3, the 2nd denial still
    # speaks the plain combined-not-found (the human offer waits for the 3rd).
    wrong = {"order_key": "ORD-1002", "account_contact": "mallory@example.com"}
    scripted = [[("propose_cancel", dict(wrong))], [], [("propose_cancel", dict(wrong))], []]
    h = _harness(
        config_root,
        FakeChatModel(scripted_calls=scripted),
        thread_id="scope-denial3",
        policy=_POLICY.model_copy(update={"auth_denials_before_human_offer": 3}),
    )
    await _events(h.engine, "cancel my rain jacket order")
    await _events(h.engine, "it's mallory at example dot com")
    msgs = [m.content for m in _tool_messages(h, "scope-denial3")]
    assert msgs.count(_SUPPORT_COMBINED_NOT_FOUND) == 2  # both denials plain
    assert _SUPPORT_NOT_FOUND_OFFER_HUMAN not in msgs  # the offer has NOT come yet


async def test_gate_exhaustion_leaves_flow_frontline_answers(
    config_root: Path, tmp_path: Path
) -> None:
    # A model that ignores the STOP clause and re-proposes without a claim twice in one
    # assemble pass hits the existing invalid-proposals terminal: it leaves the flow and the
    # frontline answers the SAME turn — never dead air, never a pending.
    scripted = [
        [("propose_cancel", {"order_key": "ORD-1002"})],
        [("propose_cancel", {"order_key": "ORD-1002"})],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-8")
    events = await _events(h.engine, "cancel my rain jacket order")
    assert _no_pendings(h, "scope-8")
    assert any(isinstance(e, TokenEvent | SpokenMessageEvent) for e in events)  # not silent
    tel = _telemetry(tmp_path)
    assert any(
        e["event"] == "support_left" and e["reason"] == "invalid_proposals" for e in tel
    )
    assert [e["event"] for e in tel].count("support_auth_required") == 2


# --- fast paths: session-placed / bound owner / rung-1 reuse ------------------------------


async def test_session_placed_order_needs_no_contact(
    config_root: Path, tmp_path: Path
) -> None:
    # Key "4" = the first PLACED order (3 fixture orders precede it in the full list); a
    # session-placed order IS authorized, so it appears in the model's list under that key.
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_key": "4"}},
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
    assert any(
        e["event"] == "support_action_authorized" and e["order_id"] == placed.order_id
        for e in tel
    )


async def test_bound_identity_owned_sails_non_owned_asks(
    config_root: Path, tmp_path: Path
) -> None:
    bound = BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119")
    # Owned: ORD-1001 (key "1", CUST-001, shipped) is LISTED — a return proposal by key
    # reaches its readback with no contact asked.
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
    events = await _events(h1.engine, "I need to return this order")
    assert any(isinstance(e, InterruptEvent) for e in events)
    assert not any(
        e["event"].startswith("support_auth_") for e in _telemetry(tmp_path)
    )
    # Non-owned: ORD-1002 (CUST-002) is NOT listed — bound or not, targeting it by stated
    # id makes the gate ask (the rungs compose).
    h2 = _harness(
        config_root,
        FakeChatModel(force_tool="propose_cancel", canned_args=_CANCEL_1002, tool_call_limit=1),
        thread_id="scope-10b",
    )
    h2.identity.bind(bound)
    events = await _events(h2.engine, "cancel my rain jacket order")
    assert _no_details_spoken(events)
    assert any(
        e["event"] == "support_auth_required" for e in _telemetry(tmp_path)
    )


async def test_rung1_grant_from_order_status_reused_in_support(
    config_root: Path, tmp_path: Path
) -> None:
    # A grant earned earlier this session (an order_status guest lookup) authorizes the
    # support action too — the order is LISTED (key "2") and sails with no re-ask.
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_key": "2"}},
            tool_call_limit=1,
        ),
        thread_id="scope-11",
    )
    h.identity.grant_order("ORD-1002")
    events = await _events(h.engine, "cancel my rain jacket order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt
    assert not any(
        e["event"].startswith("support_auth_") for e in _telemetry(tmp_path)
    )


# --- the KEY steer-leak pins (the reason the gate sits at assemble) -----------------------


async def test_unshipped_refund_steer_does_not_leak_for_unauthorized_order(
    config_root: Path, tmp_path: Path
) -> None:
    # A FULL refund-to-original on the unauthorized processing ORD-1002 would trip the
    # cancel-steer, whose line speaks "the full $129.00 goes back". The gate must hold
    # UPSTREAM: no steer event, no dollar spoken, no pending of either kind.
    args = {
        "propose_refund": {"order_key": "ORD-1002", "amount_usd": 129.0, "destination": "original"}
    }
    h = _harness(
        config_root,
        FakeChatModel(force_tool="propose_refund", canned_args=args, tool_call_limit=1),
        thread_id="scope-12",
    )
    events = await _events(h.engine, "I want my money back for the rain jacket order")
    assert _no_details_spoken(events)
    assert _no_pendings(h, "scope-12")
    tel = _telemetry(tmp_path)
    assert not any(e["event"] == "refund_steered_to_cancel" for e in tel)
    assert any(e["event"] == "support_auth_required" for e in tel)
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
                    "order_key": "ORD-1001", "amount_usd": 150.0, "destination": "original"
                }
            },
            tool_call_limit=1,
        ),
        thread_id="scope-13",
    )
    events = await _events(h.engine, "I want a refund to my original card")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert not any(
        "$" in e.text for e in events if isinstance(e, SpokenMessageEvent)
    )
    assert _no_pendings(h, "scope-13")
    tel = _telemetry(tmp_path)
    assert not any(e["event"] == "refund_steered_to_return" for e in tel)
    assert h.store.return_count == 0 and h.store.refund_count == 0


async def test_return_gated_at_assemble(config_root: Path, tmp_path: Path) -> None:
    h = _harness(
        config_root,
        FakeChatModel(
            force_tool="propose_return",
            canned_args={"propose_return": {"order_key": "ORD-1001"}},
            tool_call_limit=1,
        ),
        thread_id="scope-14",
    )
    events = await _events(h.engine, "I need to return this order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert _no_pendings(h, "scope-14")
    assert h.store.return_count == 0
    assert any(
        e["event"] == "support_auth_required" and e.get("order_id") == "ORD-1001"
        for e in _telemetry(tmp_path)
    )


# --- scope boundaries + PII ----------------------------------------------------------------


async def test_profile_change_not_gated_by_order_scope(
    config_root: Path, tmp_path: Path
) -> None:
    # Profile changes target the ACCOUNT (L2 step-up on the old factor), not an order —
    # the gate never runs; the OTP dispatch proves the flow proceeded.
    h = build_support_engine(
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
            canned_args={
                "propose_profile_change": {"field": "contact", "new_value": "555 111 2222"}
            },
            tool_call_limit=1,
        ),
        thread_id="scope-15",
    )
    await _events(h.engine, "I have a new phone number for my account")
    assert h.otp.dispatch_count == 1  # the step-up chain ran — the flow was not gated
    tel = _telemetry(tmp_path)
    assert not any(e["event"].startswith("support_auth_") for e in tel)
    assert not any(e["event"] == "support_action_authorized" for e in tel)


async def test_authorized_action_never_leaks_the_claim(
    config_root: Path, tmp_path: Path
) -> None:
    # The honest PII invariant: the claim is never telemetered and never echoed into a
    # ToolMessage. (It DOES live in the HumanMessage transcript and the tool-call args —
    # the caller spoke it — exactly as order_status works today.)
    scripted = [
        [("propose_cancel", {"order_key": "ORD-1002"})],
        [],
        [("propose_cancel", {"order_key": "ORD-1002", "account_contact": _CONTACT_1002})],
    ]
    h = _harness(config_root, FakeChatModel(scripted_calls=scripted), thread_id="scope-16")
    await _events(h.engine, "cancel my rain jacket order")
    await _events(h.engine, _CONTACT_1002)
    for event in _telemetry(tmp_path):
        assert not any(
            isinstance(v, str) and _CONTACT_1002 in v for v in event.values()
        )
    tool_msgs = _tool_messages(h, "scope-16")
    assert tool_msgs and not any(_CONTACT_1002 in str(m.content) for m in tool_msgs)
