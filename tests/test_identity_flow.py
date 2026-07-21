"""Identity flow (P7) at the ENGINE level: enumeration behind an OTP-bound identity, the
scoped order list, the bounded re-ask, THE BINDING INVARIANT (a stale cross-family L2 must
never bind on a failed OTP), and the anti-oracle posture. Zero network."""

from __future__ import annotations

from pathlib import Path

from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import SupportHarness, build_support_engine

from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TurnFacts

_POLICY = make_policy(refund_returnless_under_usd=50.0)
_FACTS = TurnFacts()
_VALID_OTP = "482913"
_CUST1_REF = "CUST-001"
_CUST2_REF = "CUST-002"
_CUST1_MASK = "number ending 0119"
_CUST1_PHONE = "+1 555 010 0119"  # CUST-001 on file: owns ORD-1001 + ORD-1003
_CUST2_EMAIL = "casey@example.com"
_UNKNOWN_CLAIM = "nobody@nowhere.example"
# Enumeration has NO gate patterns by design — entry is the MODEL's request_handover.
_REQUEST = "what orders do I have on my account?"


def _identity_harness(
    config_root: Path,
    claim: str = _CUST1_PHONE,
    *,
    risk_flagged: bool = False,
    thread_id: str = "ident-1",
    propose_limit: int = 99,
    policy=_POLICY,
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=policy,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {"destination": "support", "reason_code": "list_orders"}
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(
            force_tool="propose_identity",
            canned_args={"propose_identity": {"contact_claim": claim}},
            tool_call_limit=propose_limit,
        ),
        risk_flagged=risk_flagged,
        thread_id=thread_id,
    )


def _switch_harness(
    config_root: Path,
    *,
    claim: str,
    thread_id: str,
    otp_max_attempts: int = 3,
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=_POLICY.model_copy(update={"otp_max_attempts": otp_max_attempts}),
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {
                    "destination": "support",
                    "reason_code": "switch_account",
                }
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(
            force_tool="propose_identity",
            canned_args={"propose_identity": {"contact_claim": claim}},
            tool_call_limit=99,
        ),
        thread_id=thread_id,
    )


def _establish_customer_one(harness: SupportHarness) -> str:
    assert harness.verification.verify_otp(_VALID_OTP)
    proof_id = harness.verification.grants[-1].proof_id
    harness.identity.bind(BoundIdentity(customer_ref=_CUST1_REF, masked_contact=_CUST1_MASK))
    return proof_id


def _seed_cart(harness: SupportHarness) -> None:
    harness.caller_context.cart_store.add_item(
        sku="SKU-BLU-07",
        name="blue wool hat",
        price_usd=29.0,
        quantity=1,
    )


async def _events(engine, text: str, facts: TurnFacts = _FACTS) -> list:
    return [e async for e in engine.stream_turn(text, facts)]


def _telemetry(tmp_path: Path) -> str:
    path = tmp_path / "telemetry.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _spoken(events: list) -> list[SpokenMessageEvent]:
    return [e for e in events if isinstance(e, SpokenMessageEvent)]


# --- the happy path: claim -> OTP -> bind -> the SCOPED list -------------------------------


async def test_enumeration_ask_dispatches_otp_before_naming_any_order(
    config_root: Path,
) -> None:
    h = _identity_harness(config_root)
    events = await _events(h.engine, _REQUEST)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "6-digit code" in interrupts[0].prompt
    assert h.otp.dispatch_count == 1
    assert h.identity.current() is None  # nothing bound before verification
    assert h.verification.current_level() == 1
    # NO order id spoken or prompted pre-verification (the enumeration leak this closes).
    all_text = " ".join([e.text for e in _spoken(events)] + [i.prompt for i in interrupts])
    assert "ORD-" not in all_text


async def test_committed_otp_binds_and_speaks_only_their_orders(config_root: Path) -> None:
    h = _identity_harness(config_root)
    old_thread_id = h.engine.thread_id
    await _events(h.engine, _REQUEST)
    events = await _events(h.engine, _VALID_OTP)
    bound = h.identity.current()
    assert bound is not None and bound.customer_ref == "CUST-001"
    assert h.verification.current_level() == 2
    assert h.engine.thread_id != old_thread_id
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}
    lines = [e for e in _spoken(events) if e.node == "support_continuation"]
    assert len(lines) == 1
    # THE privacy property: CUST-001 hears 1001 + 1003 and NEVER CUST-002's 1002.
    assert "ORD-1001" in lines[0].text and "ORD-1003" in lines[0].text
    assert "ORD-1002" not in lines[0].text
    assert "CUST-" not in lines[0].text  # closed slugs never spoken


async def test_spoken_email_and_spoken_otp_verify_end_to_end(config_root: Path) -> None:
    # THE live-call-#12 replay: the claim arrives as STT's spoken email form (no '@') and
    # the code as digit words — both must verify (F-12.1 + F-12.2 together, engine level).
    h = _identity_harness(config_root, claim="casey at example dot com", thread_id="ident-spoken-1")
    await _events(h.engine, _REQUEST)
    assert h.otp.dispatch_count == 1  # the spoken email MATCHED (no re-ask, straight to OTP)
    events = await _events(h.engine, "It should be four eight two nine one three.")
    bound = h.identity.current()
    assert bound is not None and bound.customer_ref == "CUST-002"
    line = next(e for e in _spoken(events) if e.node == "support_continuation")
    assert "ORD-1002" in line.text and "ORD-1001" not in line.text


async def test_verified_account_switch_rotates_all_principal_context(
    config_root: Path,
) -> None:
    h = _switch_harness(
        config_root,
        claim=_CUST2_EMAIL,
        thread_id="switch-success",
    )
    old_proof_id = _establish_customer_one(h)
    h.identity.grant_order("ORD-1001")
    h.identity.grant_mutation_for_test("ORD-1001")
    _seed_cart(h)
    h.recent_orders.record(["ORD-1001"], operation="read")
    old_thread_id = h.engine.thread_id

    warning = await _events(h.engine, "I want to use another account")
    prompts = [event.prompt for event in warning if isinstance(event, InterruptEvent)]
    assert len(prompts) == 1 and "different account" in prompts[0]
    assert h.identity.current().customer_ref == _CUST1_REF

    dispatched = await _events(h.engine, "yes")
    assert any(
        isinstance(event, InterruptEvent) and "6-digit code" in event.prompt for event in dispatched
    )
    assert h.identity.current().customer_ref == _CUST1_REF
    assert not h.caller_context.cart_store.is_empty()

    completed = await _events(h.engine, _VALID_OTP)
    assert any(
        event.node == "identity_apply" and "new account" in event.text
        for event in _spoken(completed)
    )
    assert h.engine.thread_id != old_thread_id
    assert h.identity.current().customer_ref == _CUST2_REF
    assert h.caller_context.cart_store.is_empty()
    assert h.recent_orders.snapshot().order_refs == ()
    assert not h.identity.order_granted("ORD-1001")
    assert not h.identity.mutation_granted_for_test("ORD-1001")
    assert len(h.verification.grants) == 1
    assert h.verification.grants[0].proof_id != old_proof_id
    old_state = h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}})
    assert old_state.values == {}
    assert old_state.interrupts == ()


async def test_failed_account_switch_preserves_original_principal(
    config_root: Path,
) -> None:
    h = _switch_harness(
        config_root,
        claim=_CUST2_EMAIL,
        thread_id="switch-failed",
        otp_max_attempts=1,
    )
    old_proof_id = _establish_customer_one(h)
    _seed_cart(h)
    old_thread_id = h.engine.thread_id

    await _events(h.engine, "switch my account")
    await _events(h.engine, "yes")
    await _events(h.engine, "000000")

    assert h.engine.thread_id == old_thread_id
    assert h.identity.current().customer_ref == _CUST1_REF
    assert not h.caller_context.cart_store.is_empty()
    assert [proof.proof_id for proof in h.verification.grants] == [old_proof_id]
    state = h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}})
    assert state.values.get("pending_request") is None


async def test_same_account_switch_skips_rotation_and_preserves_context(
    config_root: Path,
) -> None:
    h = _switch_harness(
        config_root,
        claim=_CUST1_PHONE,
        thread_id="switch-same",
    )
    old_proof_id = _establish_customer_one(h)
    _seed_cart(h)
    old_thread_id = h.engine.thread_id

    await _events(h.engine, "switch my account")
    completed = await _events(h.engine, "yes")

    assert any("already verified" in event.text for event in _spoken(completed))
    assert h.engine.thread_id == old_thread_id
    assert h.identity.current().customer_ref == _CUST1_REF
    assert not h.caller_context.cart_store.is_empty()
    assert [proof.proof_id for proof in h.verification.grants] == [old_proof_id]


async def test_closing_turn_stream_completes_a_pending_context_rotation(
    config_root: Path,
) -> None:
    h = _switch_harness(
        config_root,
        claim=_CUST2_EMAIL,
        thread_id="switch-stream-close",
    )
    _establish_customer_one(h)
    old_thread_id = h.engine.thread_id
    dispatched = await _events(h.engine, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)

    stream = h.engine.stream_turn(_VALID_OTP, _FACTS)
    try:
        while True:
            event = await anext(stream)
            if isinstance(event, SpokenMessageEvent) and event.node == "identity_apply":
                break
        assert h.caller_context.pending_transition() is not None
    finally:
        await stream.aclose()

    assert h.caller_context.pending_transition() is None
    assert h.engine.thread_id != old_thread_id
    assert h.identity.current().customer_ref == _CUST2_REF
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}


async def test_session_placed_order_appears_in_the_identified_list(config_root: Path) -> None:
    # A caller who verifies AND has a session-placed order hears BOTH their account orders and the
    # one they placed this call (the account list = fixture-owned + session-placed). We bind first,
    # THEN place + list: an unbound "list my orders" with a session order now answers as a GUEST
    # (Fix 3) and never reaches identity, so the bind must precede the placement to exercise the
    # identified-list path this test protects.
    from agnostic_market.commerce.identity import BoundIdentity
    from agnostic_market.dtos.state import CartLine

    h = _identity_harness(config_root, thread_id="ident-placed-1")
    h.identity.bind(BoundIdentity(customer_ref="CUST-001", masked_contact="number ending 0119"))
    h.store.place_cart(
        "k1",
        lines=[CartLine(sku="SKU-GRN-15", name="merino hiking socks", price_usd=14.5, quantity=2)],
        total_usd=29.0,
    )
    # Bound -> the enumeration ask code-renders the account list directly (handover node), no OTP.
    events = await _events(h.engine, _REQUEST)
    line = next(e for e in _spoken(events) if e.node == "handover")
    assert "ORD-9001" in line.text  # placed THIS call, listed alongside the account's own orders
    assert "ORD-1001" in line.text  # CUST-001's fixture order too


async def test_already_bound_reask_lists_without_a_second_otp(config_root: Path) -> None:
    h = _identity_harness(config_root, thread_id="ident-rebound-1")
    await _events(h.engine, _REQUEST)
    await _events(h.engine, _VALID_OTP)
    assert h.otp.dispatch_count == 1
    # A SECOND enumeration request re-lists directly from the existing binding — no Identity
    # re-entry, no re-claim, and no second OTP.
    events = await _events(h.engine, "tell me my orders again please")
    line = next(e for e in _spoken(events) if e.node == "handover")
    assert "ORD-1001" in line.text
    assert h.otp.dispatch_count == 1  # unchanged


# --- THE BINDING INVARIANT (the P7 security-review catch) ----------------------------------


async def test_stale_cross_family_l2_cannot_bind_on_a_wrong_otp(config_root: Path) -> None:
    # An EARLIER profile-flow OTP left the store at L2 (verify_otp = exactly what that flow
    # commits). The factory's route_after_collect confirms on level ALONE — without the
    # wrapper, a WRONG identity code would bind. The wrapper demands a grant NEWER than the
    # pending's mint snapshot, so the wrong code re-collects instead.
    h = _identity_harness(config_root, thread_id="ident-stale-1")
    assert h.verification.verify_otp(_VALID_OTP)  # the stale cross-family grant
    assert h.verification.current_level() == 2
    await _events(h.engine, _REQUEST)  # -> OTP interrupt (level alone must NOT skip it)
    events = await _events(h.engine, "000000")  # WRONG code, level already 2
    assert h.identity.current() is None  # NOT bound — the invariant held
    assert any(isinstance(e, InterruptEvent) for e in events)  # re-collect, not confirm
    assert h.otp.dispatch_count == 2  # a legitimate re-dispatch (new attempt key)


async def test_stale_l2_then_correct_otp_binds(config_root: Path) -> None:
    # The counterpart: with the stale grant present, the CORRECT code appends a NEW grant
    # (verify_otp records every success) and the bind proceeds.
    h = _identity_harness(config_root, thread_id="ident-stale-2")
    assert h.verification.verify_otp(_VALID_OTP)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, "000000")  # one miss
    events = await _events(h.engine, _VALID_OTP)  # correct on the re-collect
    bound = h.identity.current()
    assert bound is not None and bound.customer_ref == "CUST-001"
    assert any(e.node == "support_continuation" for e in _spoken(events))


async def test_stale_l2_wrong_otp_twice_exhausts_to_human(config_root: Path) -> None:
    # The wrapper preserves the factory's bounded-retry semantics: two committed misses
    # exhaust to a human even when the stale level would have "confirmed" each time.
    h = _identity_harness(config_root, thread_id="ident-stale-3")
    assert h.verification.verify_otp(_VALID_OTP)
    await _events(h.engine, _REQUEST)
    await _events(h.engine, "000000")
    events = await _events(h.engine, "111111")
    assert h.identity.current() is None
    assert not any(isinstance(e, InterruptEvent) for e in events)  # no third collect
    texts = [e.text for e in _spoken(events)]
    assert any("person" in t for t in texts)  # the human deferral is the single voice


# --- security branches ----------------------------------------------------------------------


async def test_wrong_otp_twice_stays_unbound_at_l1(config_root: Path, tmp_path: Path) -> None:
    h = _identity_harness(config_root, thread_id="ident-wrong-1")
    await _events(h.engine, _REQUEST)
    first = await _events(h.engine, "000000")
    assert any(isinstance(e, InterruptEvent) for e in first)  # ONE re-collect
    await _events(h.engine, "111111")
    assert h.identity.current() is None
    assert h.verification.current_level() == 1
    telemetry = _telemetry(tmp_path)
    assert "identity_stepup_failed" in telemetry and "otp_exhausted" in telemetry
    assert _VALID_OTP not in telemetry  # no code value in telemetry
    assert "000000" not in telemetry and "111111" not in telemetry


async def test_sim_swap_flag_escalates_before_any_dispatch(config_root: Path) -> None:
    h = _identity_harness(config_root, risk_flagged=True, thread_id="ident-risk-1")
    events = await _events(h.engine, _REQUEST)
    assert h.otp.dispatch_count == 0  # blocked BEFORE any code was sent
    assert h.identity.current() is None
    texts = [e.text for e in _spoken(events)]
    assert any("person" in t for t in texts)


# --- the bounded re-ask + anti-oracle posture ------------------------------------------------


async def test_no_match_first_claim_gets_one_softened_reask(
    config_root: Path, tmp_path: Path
) -> None:
    h = _identity_harness(config_root, claim=_UNKNOWN_CLAIM, thread_id="ident-nomatch-1")
    events = await _events(h.engine, _REQUEST)
    reasks = [e for e in _spoken(events) if e.node == "identity_reask"]
    assert len(reasks) == 1
    assert "double-check" in reasks[0].text
    # The softened wording NEVER asserts absence (existence-oracle discipline).
    assert "find" not in reasks[0].text.lower() and "on file" not in reasks[0].text.lower()
    assert h.otp.dispatch_count == 0  # a doomed dispatch would grant L2 on the stub code
    assert _UNKNOWN_CLAIM not in _telemetry(tmp_path)  # the claim is never persisted


async def test_no_match_second_claim_hands_to_human_silently(
    config_root: Path, tmp_path: Path
) -> None:
    h = _identity_harness(config_root, claim=_UNKNOWN_CLAIM, thread_id="ident-nomatch-2")
    await _events(h.engine, _REQUEST)
    events = await _events(h.engine, "it's the same address I told you")
    # SILENT terminal: no flow-authored line — the handover deferral is the single voice.
    spoken = _spoken(events)
    assert all(e.node == "handover" for e in spoken)
    assert any("person" in e.text for e in spoken)
    assert h.identity.current() is None
    telemetry = _telemetry(tmp_path)
    assert "no_match" in telemetry
    assert _UNKNOWN_CLAIM not in telemetry


async def test_contact_reask_budget_tracks_the_policy_knob(config_root: Path) -> None:
    # contact_reask_max is config-driven: a merchant that raised it to 2 gets a SECOND
    # softened re-ask where the default (1) would already have handed to a human.
    h = _identity_harness(
        config_root,
        claim=_UNKNOWN_CLAIM,
        thread_id="ident-reask2",
        policy=_POLICY.model_copy(update={"contact_reask_max": 2}),
    )
    await _events(h.engine, _REQUEST)  # miss #1 -> re-ask #1
    events = await _events(h.engine, "still the wrong one")  # miss #2 -> re-ask #2 (budget 2)
    reasks = [e for e in _spoken(events) if e.node == "identity_reask"]
    assert len(reasks) == 1  # a re-ask this turn, not a handover
    assert h.identity.current() is None


async def test_contact_reask_zero_hands_over_on_first_miss(config_root: Path) -> None:
    # contact_reask_max=0: no re-ask at all — the first no-match hands to a human.
    h = _identity_harness(
        config_root,
        claim=_UNKNOWN_CLAIM,
        thread_id="ident-reask0",
        policy=_POLICY.model_copy(update={"contact_reask_max": 0}),
    )
    events = await _events(h.engine, _REQUEST)
    spoken = _spoken(events)
    assert not any(e.node == "identity_reask" for e in spoken)  # no re-ask
    assert any(e.node == "handover" for e in spoken)  # straight to a person


# --- escapes + crossover isolation ----------------------------------------------------------


async def test_abort_mid_identity_leaves_nothing_bound(config_root: Path) -> None:
    # From the sticky-but-not-interrupted state (the identity model asked a clarify), an
    # explicit abort drops the verification without touching anything.
    h = build_support_engine(
        config_root,
        policy=_POLICY,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {"destination": "support", "reason_code": "list_orders"}
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(emit_tool_calls=False),  # clarifies; stays sticky
        thread_id="ident-abort-1",
    )
    await _events(h.engine, _REQUEST)
    events = await _events(h.engine, "never mind, forget it")
    texts = [e.text for e in _spoken(events)]
    assert any("dropped" in t for t in texts)
    assert h.identity.current() is None


async def test_cross_switch_out_of_identity_on_a_gate_certain_intent(
    config_root: Path,
) -> None:
    # While sticky in identity, "cancel my order please" is NOT an abort phrasing — it falls
    # through to the gate and cross-switches to support (no caller is trapped, §A9).
    h = build_support_engine(
        config_root,
        policy=_POLICY,
        frontline=FakeChatModel(
            force_tool="request_handover",
            canned_args={
                "request_handover": {"destination": "support", "reason_code": "list_orders"}
            },
            tool_call_limit=99,
        ),
        reasoning=FakeChatModel(emit_tool_calls=False),  # clarifies in BOTH flows
        thread_id="ident-cross-1",
    )
    await _events(h.engine, _REQUEST)
    await _events(h.engine, "cancel my order please")
    state = h.engine._graph.get_state({"configurable": {"thread_id": "ident-cross-1"}})
    assert state.values.get("active_flow") == "support"  # switched, not trapped
    assert state.values.get("pending_identity") is None


async def test_identity_stepup_never_touches_refund_or_profile_state(
    config_root: Path,
) -> None:
    # Third-family isolation (the factory's zero-crossover claim): a failed identity
    # step-up leaves the refund/profile families untouched.
    h = _identity_harness(config_root, thread_id="ident-iso-1")
    await _events(h.engine, _REQUEST)
    await _events(h.engine, "000000")
    await _events(h.engine, "111111")  # exhaust -> human
    state = h.engine._graph.get_state({"configurable": {"thread_id": "ident-iso-1"}})
    assert state.values.get("pending_refund") is None
    assert state.values.get("pending_profile_change") is None
    assert h.profile.change_count == 0
    assert h.store.placed_count == 0
