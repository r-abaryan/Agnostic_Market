"""Identity flow (P7) at the ENGINE level: enumeration behind an OTP-bound identity, the
scoped order list, the bounded re-ask, THE BINDING INVARIANT (a stale cross-family L2 must
never bind on a failed OTP), and the anti-oracle posture. Zero network."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from support_helpers import SupportHarness, build_support_engine
from turn_helpers import engine_events, next_committed_turn

from agnostic_market.agents._copy import ACCOUNT_CONTACT_QUESTION
from agnostic_market.agents.recovery import AUTOMATION_TERMINAL_LINE
from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.events import InterruptEvent, SpokenMessageEvent, TokenEvent, TurnFacts
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    CancelOrders,
    ChangeProfile,
    ListOrders,
    VerifyIdentity,
)
from agnostic_market.dtos.state import (
    CHECKPOINT_SCHEMA_VERSION,
    PendingIdentity,
    ReasoningState,
    open_active_invocation,
)

_POLICY = make_policy(refund_returnless_under_usd=50.0)
_FACTS = TurnFacts()
_VALID_OTP = "482913"
_CUST1_REF = "CUST-001"
_CUST2_REF = "CUST-002"
_CUST1_MASK = "number ending 0119"
_CUST1_PHONE = "+1 555 010 0119"  # CUST-001 on file: owns ORD-1001 + ORD-1003
_CUST2_EMAIL = "casey@example.com"
_UNKNOWN_CLAIM = "nobody@nowhere.example"
_VERIFY_CONTEXT_WARNING = (
    "Verifying an account will clear this call's cart and recent order context. "
    "Orders already placed will remain placed, but they won't stay available in this "
    "conversation. Would you like to continue?"
)
_REQUEST = "what orders do I have on my account?"


def _active_request(values: dict) -> object | None:
    invocation = values.get("active_invocation")
    if invocation is None:
        return None
    assert isinstance(invocation, ActiveInvocation)
    assert invocation.opened_turn_id in tuple(values["consumed_turn_ids"])
    return invocation.request


def _identity_harness(
    config_root: Path,
    claim: str = _CUST1_PHONE,
    *,
    risk_flagged: bool = False,
    thread_id: str = "ident-1",
    propose_limit: int = 99,
    policy=_POLICY,
    reasoning: FakeChatModel | None = None,
) -> SupportHarness:
    return build_support_engine(
        config_root,
        policy=policy,
        reasoning=(
            reasoning
            if reasoning is not None
            else FakeChatModel(
                force_tool="propose_identity",
                canned_args={"propose_identity": {"contact_claim": claim}},
                tool_call_limit=propose_limit,
            )
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


def _seed_typed_verification(harness: SupportHarness, *, turn_id: str) -> None:
    consumed_turn_ids = (turn_id,)
    harness.engine._graph.update_state(
        harness.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "consumed_turn_ids": consumed_turn_ids,
            "active_invocation": open_active_invocation(
                VerifyIdentity(),
                consumed_turn_ids=consumed_turn_ids,
            ),
        },
        as_node="__start__",
    )


async def _events(engine, text: str, facts: TurnFacts = _FACTS) -> list:
    return await engine_events(engine, text, facts)


def _telemetry(tmp_path: Path) -> str:
    path = tmp_path / "telemetry.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _spoken(events: list) -> list[SpokenMessageEvent]:
    return [e for e in events if isinstance(e, SpokenMessageEvent)]


def _assert_identity_contact_ask(
    harness: SupportHarness,
    events: list,
    *,
    thread_id: str,
    expected_misses: int,
) -> None:
    assert [(event.node, event.text) for event in _spoken(events)] == [
        ("identity_ask_contact", ACCOUNT_CONTACT_QUESTION)
    ]
    assert not any(isinstance(event, TokenEvent | InterruptEvent) for event in events)
    snapshot = harness.engine._graph.get_state({"configurable": {"thread_id": thread_id}})
    state = snapshot.values
    assert snapshot.next == ()
    assert state.get("execution_owner") == "identity"
    assert state.get("identity_claim_misses") == expected_misses
    assert _active_request(state) == ListOrders(scope="account")
    assert state.get("pending_clarification") is None
    for field in (
        "pending_placement",
        "pending_refund",
        "pending_cancel",
        "pending_return",
        "pending_profile_change",
        "pending_identity",
    ):
        assert state.get(field) is None
    assert harness.identity.current() is None
    assert harness.verification.grants == []
    assert harness.otp.dispatch_count == 0
    assert harness.store.placed_count == 0
    assert harness.store.refund_count == 0
    assert harness.store.cancel_count == 0
    assert harness.store.return_count == 0
    assert harness.profile.change_count == 0
    assert harness.caller_context.cart_store.view() == ()


# --- code-authored missing-contact clarification (Milestone 3b) ---------------------------


async def test_explicit_identity_contact_request_is_code_authored(config_root: Path) -> None:
    thread_id = "ident-clarify-tool"
    reasoning = FakeChatModel(
        force_tool="request_identity_contact",
        tool_call_limit=99,
        record_prompts=True,
    )
    h = _identity_harness(
        config_root,
        thread_id=thread_id,
        reasoning=reasoning,
    )
    events = await _events(h.engine, _REQUEST)
    _assert_identity_contact_ask(h, events, thread_id=thread_id, expected_misses=0)
    assert "call request_identity_contact" in reasoning._seen_prompts[-1]
    assert "one tool call WITH NO spoken text" in reasoning._seen_prompts[-1]


async def test_identity_no_tool_prose_falls_back_to_code_question(config_root: Path) -> None:
    thread_id = "ident-clarify-no-tool"
    raw_model_text = "The account is already identified on my end."
    h = _identity_harness(
        config_root,
        thread_id=thread_id,
        reasoning=FakeChatModel(emit_tool_calls=False, text_response=raw_model_text),
    )
    events = await _events(h.engine, _REQUEST)
    _assert_identity_contact_ask(h, events, thread_id=thread_id, expected_misses=0)
    state = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert not any(raw_model_text in str(message.content) for message in state["messages"])


@pytest.mark.parametrize(
    ("second_call", "unknown_results"),
    [
        ([("request_identity_contact", {})], 1),
        ([("catalog_lookup", {"query": "orders"})], 2),
    ],
)
async def test_unknown_identity_tool_uses_bounded_correction_without_authority(
    config_root: Path,
    second_call: list[tuple[str, dict]],
    unknown_results: int,
) -> None:
    thread_id = f"ident-unknown-{unknown_results}"
    reasoning = FakeChatModel(
        scripted_calls=[
            [("catalog_lookup", {"query": "orders"})],
            second_call,
        ]
    )
    h = _identity_harness(config_root, thread_id=thread_id, reasoning=reasoning)

    events = await _events(h.engine, _REQUEST)

    _assert_identity_contact_ask(h, events, thread_id=thread_id, expected_misses=0)
    state = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert reasoning.invoke_count == 2
    assert (
        sum(
            isinstance(message, ToolMessage)
            and str(message.content).startswith("Unavailable action.")
            for message in state["messages"]
        )
        == unknown_results
    )


async def test_repeated_identity_clarification_exhausts_to_one_terminal_handover(
    config_root: Path, tmp_path: Path
) -> None:
    thread_id = "ident-clarify-exhausted"
    h = _identity_harness(
        config_root,
        thread_id=thread_id,
        reasoning=FakeChatModel(emit_tool_calls=False),
    )

    first = await _events(h.engine, _REQUEST)
    second = await _events(h.engine, "I don't have an email address or phone number.")
    exhausted = await _events(h.engine, "I still can't provide either one.")

    assert [event.node for event in _spoken(first)] == ["identity_ask_contact"]
    assert [event.node for event in _spoken(second)] == ["identity_ask_contact"]
    assert [event.node for event in _spoken(exhausted)] == ["automation_terminal_response"]
    state = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert state.get("automation_terminal") is True
    assert state.get("execution_owner") is None
    assert state.get("active_invocation") is None
    assert state.get("clarification_liveness") is None
    assert h.identity.current() is None
    assert h.otp.dispatch_count == 0
    assert h.store.cancel_count == h.store.refund_count == h.store.return_count == 0
    exhausted_events = [
        line
        for line in _telemetry(tmp_path).splitlines()
        if '"event": "clarification_exhausted"' in line
    ]
    assert exhausted_events == [
        '{"event": "clarification_exhausted", "owner_kind": "invocation", '
        '"consumed_reasks": 1, "limit": 1}'
    ]


async def test_valid_identity_claim_clears_prior_clarification_liveness(
    config_root: Path,
) -> None:
    thread_id = "ident-clarify-progress"
    reasoning = FakeChatModel(
        scripted_calls=[
            [("request_identity_contact", {})],
            [("propose_identity", {"contact_claim": _CUST1_PHONE})],
        ]
    )
    h = _identity_harness(config_root, thread_id=thread_id, reasoning=reasoning)

    await _events(h.engine, _REQUEST)
    before = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert before["clarification_liveness"].reasks == 0
    events = await _events(h.engine, _CUST1_PHONE)

    assert any(isinstance(event, InterruptEvent) for event in events)
    after = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert after.get("clarification_liveness") is None
    assert h.otp.dispatch_count == 1


async def test_malformed_identity_tools_preserve_miss_budget_and_continuation(
    config_root: Path,
) -> None:
    thread_id = "ident-clarify-malformed"
    reasoning = FakeChatModel(
        scripted_calls=[
            [("propose_identity", {"contact_claim": _UNKNOWN_CLAIM})],
            [("request_identity_contact", {"unexpected": "value"})],
            [
                (
                    "propose_identity",
                    {"contact_claim": _CUST2_EMAIL, "unexpected": "value"},
                )
            ],
        ]
    )
    h = _identity_harness(config_root, thread_id=thread_id, reasoning=reasoning)

    await _events(h.engine, _REQUEST)
    after_miss = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert after_miss.get("identity_claim_misses") == 1
    assert _active_request(after_miss) == ListOrders(scope="account")

    events = await _events(h.engine, "It is the same contact I gave you.")
    _assert_identity_contact_ask(h, events, thread_id=thread_id, expected_misses=1)
    assert reasoning.invoke_count == 3
    state = h.engine._graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert state["clarification_liveness"].reasks == 0


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


async def test_committed_otp_binds_and_speaks_only_their_orders(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _identity_harness(config_root)
    old_thread_id = h.engine.thread_id
    await _events(h.engine, _REQUEST)
    before_rotation = ReasoningState.model_validate(
        h.engine._graph.get_state(h.engine._config).values
    )
    old_invocation = before_rotation.active_invocation
    assert old_invocation is not None
    rotation_seeds: list[dict[str, object]] = []
    real_update_state = h.engine._graph.update_state

    def observe_rotation_seed(config, values, *, as_node=None):
        invocation = values.get("active_invocation")
        if config["configurable"]["thread_id"] != old_thread_id and isinstance(
            invocation, ActiveInvocation
        ):
            rotation_seeds.append(dict(values))
        return real_update_state(config, values, as_node=as_node)

    monkeypatch.setattr(h.engine._graph, "update_state", observe_rotation_seed)
    events = await _events(h.engine, _VALID_OTP)
    bound = h.identity.current()
    assert bound is not None and bound.customer_ref == "CUST-001"
    assert h.verification.current_level() == 2
    assert h.engine.thread_id != old_thread_id
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}
    new_state = h.engine._graph.get_state(h.engine._config)
    assert tuple(new_state.values["consumed_turn_ids"]) == ("test-turn-1", "test-turn-2")
    assert len(rotation_seeds) == 1
    rotated_invocation = rotation_seeds[0]["active_invocation"]
    assert isinstance(rotated_invocation, ActiveInvocation)
    assert rotated_invocation.request == old_invocation.request == ListOrders(scope="account")
    assert rotated_invocation.invocation_id != old_invocation.invocation_id
    carried_turn_ids = tuple(rotation_seeds[0]["consumed_turn_ids"])
    assert rotated_invocation.opened_turn_id == carried_turn_ids[-1] == "test-turn-2"
    lines = [e for e in _spoken(events) if e.node == "support_capability_render"]
    assert len(lines) == 1
    # THE privacy property: CUST-001 hears 1001 + 1003 and NEVER CUST-002's 1002.
    assert "ORD-1001" in lines[0].text and "ORD-1003" in lines[0].text
    assert "ORD-1002" not in lines[0].text
    assert "CUST-" not in lines[0].text  # closed slugs never spoken


def test_rotation_preserves_populated_profile_slots_exactly(config_root: Path) -> None:
    h = _identity_harness(config_root, thread_id="ident-profile-slot-rotation")
    assert h.verification.verify_otp(_VALID_OTP)
    request = ChangeProfile(field="address", new_value="10 High Street")
    transition = h.caller_context.transition_principal(
        BoundIdentity(customer_ref=_CUST1_REF, masked_contact=_CUST1_MASK),
        h.verification.grants[-1],
        request,
    )
    old_thread_id = h.engine.thread_id

    rotated = h.engine._rotate_pending_transition("profile-rotation-turn")

    assert rotated == transition
    assert h.engine.thread_id != old_thread_id
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}
    snapshot = h.engine._graph.get_state(h.engine._config)
    state = ReasoningState.model_validate(snapshot.values)
    invocation = state.active_invocation
    assert invocation is not None
    assert invocation.request == request
    assert isinstance(invocation.request, ChangeProfile)
    assert invocation.request.new_value == "10 High Street"
    assert invocation.opened_turn_id == state.consumed_turn_ids[-1] == "profile-rotation-turn"
    assert snapshot.next == ("entry",)


def test_same_principal_apply_preserves_invocation_until_continuation(
    config_root: Path,
) -> None:
    h = _identity_harness(config_root, thread_id="ident-same-principal-invocation")
    h.identity.bind(BoundIdentity(customer_ref=_CUST1_REF, masked_contact=_CUST1_MASK))
    consumed_turn_ids = ("same-principal-opening-turn",)
    invocation = open_active_invocation(
        ListOrders(scope="account"),
        consumed_turn_ids=consumed_turn_ids,
    )
    state = ReasoningState(
        consumed_turn_ids=consumed_turn_ids,
        execution_owner="identity",
        active_invocation=invocation,
        pending_identity=PendingIdentity(
            customer_ref=_CUST1_REF,
            masked_contact=_CUST1_MASK,
            attempt_key="same-principal-attempt",
            grants_at_mint=0,
        ),
    )

    apply_update = h.engine._graph.nodes["identity_apply"].invoke(state)
    after_apply = ReasoningState.model_validate({**state.model_dump(mode="python"), **apply_update})

    assert "active_invocation" not in apply_update
    assert after_apply.active_invocation == invocation
    assert after_apply.active_invocation.invocation_id == invocation.invocation_id
    assert after_apply.active_invocation.opened_turn_id == invocation.opened_turn_id
    # The tool-capable entry hands a read off untouched; only the code-only render node,
    # which authors the caller-audible line, consumes the invocation.
    entry_update = h.engine._graph.nodes["support_capability_entry"].invoke(after_apply)
    assert "active_invocation" not in entry_update
    after_entry = ReasoningState.model_validate(
        {**after_apply.model_dump(mode="python"), **entry_update}
    )
    assert after_entry.active_invocation == invocation
    render_update = h.engine._graph.nodes["support_capability_render"].invoke(after_entry)
    assert render_update["active_invocation"] is None


async def test_typed_verification_binds_rotates_and_completes_once(config_root: Path) -> None:
    h = _identity_harness(config_root, thread_id="typed-verify")
    _seed_typed_verification(h, turn_id="typed-verify-opening")
    old_thread_id = h.engine.thread_id

    dispatched = await _events(h.engine, "continue")
    assert [event.prompt for event in dispatched if isinstance(event, InterruptEvent)] == [
        "For security, please read me the 6-digit code we just sent you."
    ]
    assert h.identity.current() is None
    assert h.otp.dispatch_count == 1

    completed = await _events(h.engine, _VALID_OTP)

    assert [(event.node, event.text) for event in _spoken(completed)] == [
        ("identity_apply", "You're now verified.")
    ]
    assert h.identity.current() == BoundIdentity(
        customer_ref=_CUST1_REF,
        masked_contact=_CUST1_MASK,
    )
    assert h.engine.thread_id != old_thread_id
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}
    new_state = h.engine._graph.get_state(h.engine._config)
    assert new_state.next == ()
    assert new_state.values.get("active_invocation") is None
    assert new_state.values.get("messages", []) == []


async def test_typed_verification_decline_preserves_discardable_context(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_identity",
        canned_args={"propose_identity": {"contact_claim": _CUST1_PHONE}},
        tool_call_limit=99,
    )
    h = _identity_harness(
        config_root,
        thread_id="typed-verify-decline",
        reasoning=reasoning,
    )
    _seed_cart(h)
    _seed_typed_verification(h, turn_id="typed-verify-decline-opening")
    old_thread_id = h.engine.thread_id

    warning = await _events(h.engine, "continue")
    assert [event.prompt for event in warning if isinstance(event, InterruptEvent)] == [
        _VERIFY_CONTEXT_WARNING
    ]
    declined = await _events(h.engine, "no")

    assert [(event.node, event.text) for event in _spoken(declined)] == [
        ("principal_warning", "Okay, I won't start account verification.")
    ]
    assert h.engine.thread_id == old_thread_id
    assert h.identity.current() is None
    assert not h.caller_context.cart_store.is_empty()
    assert h.otp.dispatch_count == 0
    assert reasoning.invoke_count == 0
    state = ReasoningState.model_validate(h.engine._graph.get_state(h.engine._config).values)
    assert state.active_invocation is None
    assert state.execution_owner is None


async def test_typed_verification_accepts_context_warning_then_rotates(
    config_root: Path,
) -> None:
    h = _identity_harness(config_root, thread_id="typed-verify-accept")
    _seed_cart(h)
    _seed_typed_verification(h, turn_id="typed-verify-accept-opening")
    old_thread_id = h.engine.thread_id

    warning = await _events(h.engine, "continue")
    assert [event.prompt for event in warning if isinstance(event, InterruptEvent)] == [
        _VERIFY_CONTEXT_WARNING
    ]
    dispatched = await _events(h.engine, "yes")
    assert [event.prompt for event in dispatched if isinstance(event, InterruptEvent)] == [
        "For security, please read me the 6-digit code we just sent you."
    ]
    assert h.identity.current() is None
    assert not h.caller_context.cart_store.is_empty()

    completed = await _events(h.engine, _VALID_OTP)

    assert [(event.node, event.text) for event in _spoken(completed)] == [
        ("identity_apply", "You're now verified.")
    ]
    assert h.identity.current() == BoundIdentity(
        customer_ref=_CUST1_REF,
        masked_contact=_CUST1_MASK,
    )
    assert h.engine.thread_id != old_thread_id
    assert h.caller_context.cart_store.is_empty()
    assert h.engine._graph.get_state({"configurable": {"thread_id": old_thread_id}}).values == {}
    new_state = h.engine._graph.get_state(h.engine._config)
    assert new_state.next == ()
    assert new_state.values.get("active_invocation") is None
    assert new_state.values.get("messages", []) == []


async def test_spoken_email_and_spoken_otp_verify_end_to_end(config_root: Path) -> None:
    # THE live-call-#12 replay: the claim arrives as STT's spoken email form (no '@') and
    # the code as digit words — both must verify (F-12.1 + F-12.2 together, engine level).
    h = _identity_harness(config_root, claim="casey at example dot com", thread_id="ident-spoken-1")
    await _events(h.engine, _REQUEST)
    assert h.otp.dispatch_count == 1  # the spoken email MATCHED (no re-ask, straight to OTP)
    events = await _events(h.engine, "It should be four eight two nine one three.")
    bound = h.identity.current()
    assert bound is not None and bound.customer_ref == "CUST-002"
    line = next(e for e in _spoken(events) if e.node == "support_capability_render")
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
    h.identity.grant_orders("ORD-1001")
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
    new_state = h.engine._graph.get_state(h.engine._config)
    assert tuple(new_state.values["consumed_turn_ids"]) == (
        "test-turn-1",
        "test-turn-2",
        "test-turn-3",
    )
    assert new_state.values.get("active_invocation") is None
    assert new_state.values.get("messages", []) == []
    assert new_state.next == ()


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
    assert state.values.get("active_invocation") is None


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

    stream = h.engine.stream_turn(next_committed_turn(h.engine, _VALID_OTP), _FACTS)
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


async def test_rotation_failure_during_stream_close_latches_terminal_for_next_turn(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _switch_harness(
        config_root,
        claim=_CUST2_EMAIL,
        thread_id="switch-stream-close-failure",
    )
    _establish_customer_one(h)
    old_thread_id = h.engine.thread_id
    dispatched = await _events(h.engine, "switch my account")
    assert any(isinstance(event, InterruptEvent) for event in dispatched)

    stream = h.engine.stream_turn(next_committed_turn(h.engine, _VALID_OTP), _FACTS)
    try:
        while True:
            event = await anext(stream)
            if isinstance(event, SpokenMessageEvent) and event.node == "identity_apply":
                break
        assert h.caller_context.pending_transition() is not None

        real_delete = h.engine._graph.checkpointer.delete_thread

        def fail_old_thread_delete(thread_id: str) -> None:
            if thread_id == old_thread_id:
                raise RuntimeError("injected close-stream rotation failure")
            real_delete(thread_id)

        monkeypatch.setattr(h.engine._graph.checkpointer, "delete_thread", fail_old_thread_delete)
    finally:
        await stream.aclose()

    next_turn = await _events(h.engine, "try again")
    snapshot = h.engine._graph.get_state(h.engine._config)

    assert [event.text for event in _spoken(next_turn)] == [AUTOMATION_TERMINAL_LINE]
    assert h.engine.thread_id == old_thread_id
    assert h.caller_context.pending_transition() is None
    assert h.identity.current() is None
    assert h.verification.current_level() == 1
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()


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
    line = next(e for e in _spoken(events) if e.node == "support_capability_render")
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
    line = next(e for e in _spoken(events) if e.node == "support_capability_render")
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
    assert any(e.node == "support_capability_render" for e in _spoken(events))


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
    assert any(
        e.node == "automation_terminal_response" and "contact the store" in e.text.lower()
        for e in _spoken(events)
    )


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
    assert any(
        e.node == "automation_terminal_response" and "contact the store" in e.text.lower()
        for e in _spoken(events)
    )


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
    # No flow-authored line: the shared terminal node is the single voice.
    spoken = _spoken(events)
    assert all(e.node == "automation_terminal_response" for e in spoken)
    assert any("contact the store" in e.text.lower() for e in spoken)
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
    assert any(e.node == "automation_terminal_response" for e in spoken)


# --- escapes + crossover isolation ----------------------------------------------------------


async def test_abort_mid_identity_leaves_nothing_bound(config_root: Path) -> None:
    # From the sticky-but-not-interrupted state (the identity model asked a clarify), an
    # explicit abort drops the verification without touching anything.
    h = build_support_engine(
        config_root,
        policy=_POLICY,
        reasoning=FakeChatModel(emit_tool_calls=False),  # clarifies; stays sticky
        thread_id="ident-abort-1",
    )
    await _events(h.engine, _REQUEST)
    events = await _events(h.engine, "never mind, forget it")
    texts = [e.text for e in _spoken(events)]
    assert texts == ["Okay. I've stopped that request."]
    assert h.identity.current() is None


async def test_direct_semantic_route_replaces_an_identity_invocation(
    config_root: Path,
) -> None:
    # While sticky in identity, "cancel my order please" is NOT an abort phrasing — it falls
    # through to the gate and cross-switches to support (no caller is trapped, §A9).
    h = build_support_engine(
        config_root,
        policy=_POLICY,
        reasoning=FakeChatModel(emit_tool_calls=False),  # clarifies in BOTH flows
        thread_id="ident-cross-1",
    )
    await _events(h.engine, _REQUEST)
    before_switch = h.engine._graph.get_state(
        {"configurable": {"thread_id": "ident-cross-1"}}
    ).values
    old_invocation = before_switch["active_invocation"]
    assert before_switch["clarification_liveness"].owner.invocation_id == (
        old_invocation.invocation_id
    )
    await _events(h.engine, "cancel my order please")
    state = h.engine._graph.get_state({"configurable": {"thread_id": "ident-cross-1"}})
    assert state.values.get("execution_owner") == "support"
    assert state.values.get("pending_identity") is None
    new_invocation = state.values["active_invocation"]
    assert new_invocation.invocation_id != old_invocation.invocation_id
    assert isinstance(new_invocation.request, CancelOrders)
    assert state.values["clarification_liveness"].owner.invocation_id == (
        new_invocation.invocation_id
    )
    assert state.values["clarification_liveness"].reasks == 0


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
