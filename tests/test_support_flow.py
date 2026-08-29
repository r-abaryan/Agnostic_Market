"""Support flow (T3) at the ENGINE level: the step-up verification loop, refund-to-new-
instrument, and its exit behaviors. Zero network (fake models + InMemorySaver).

Drives the real graph through the ReasoningEngine so the interrupt/resume path is exercised
exactly as production does. The reasoning fake proposes a refund via `propose_refund`; the
frontline gate trips on "refund ..." and enters the support flow.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from llm_fakes import FakeChatModel
from policy_helpers import make_policy
from pydantic import ValidationError
from support_helpers import authorize_customer, build_support_engine
from turn_helpers import engine_events

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.recovery import RECOVERY_NODE_NAME
from agnostic_market.agents.support import flow as support_flow
from agnostic_market.agents.telemetry import InMemoryTelemetrySink
from agnostic_market.checkpoints import CheckpointScopeError
from agnostic_market.commerce.orders import OrdersFixture, OrderStore, load_orders_fixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentsFixture,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.dtos.events import (
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnFacts,
)
from agnostic_market.dtos.orchestration import (
    AnswerQuestion,
    CancellableOrderScope,
    CancelOrders,
    ExplicitOrderSet,
    ExplicitOrderTarget,
    IntentRequest,
    RefundOrder,
    RouteDecision,
    RouteResolution,
)
from agnostic_market.dtos.state import CartLine, PolicyContext, ReasoningState

# returnless high on purpose (the default): the legacy amount-gate/step-up scenarios run
# refunds against the SHIPPED ORD-1001 in isolation; the return-first tests tighten via model_copy.
_POLICY = make_policy()
_FACTS = TurnFacts()
_VALID_OTP = "482913"
_ORIGINAL_INSTRUMENT = "original payment method"
# The reasoning fake proposes a refund of $129.00 on order key "2" (ORD-1002) to a NEW card.
_PROPOSE = {
    "propose_refund": {"order_key": "2", "amount_usd": 129.0, "destination": "new_instrument"}
}
_REFUND_REQUEST = "I'd like a refund to a different card"


def _instrument_ref(config_root: Path, customer_ref: str) -> str:
    return (
        load_payment_instruments_fixture(config_root, "acme_store")
        .payment_instruments[customer_ref]
        .masked_ref
    )


def _engine(
    config_root: Path,
    *,
    reasoning: FakeChatModel | None = None,
    risk_flagged: bool = False,
    policy: PolicyContext = _POLICY,
    thread_id: str = "support-1",
    routing_resolution: RouteResolution | None = None,
    authorized_customer_ref: str = "CUST-002",
) -> tuple[ReasoningEngine, OrderStore, VerificationStore, OtpProvider]:
    """Thin wrapper over the shared harness (support_helpers) preserving this file's
    4-tuple unpacking + its propose_refund default. Fixture orders are PRE-AUTHORIZED
    (rung-1) — this suite pins the post-authorization money logic; the selection gate has
    its own suite (test_support_scoping.py)."""
    harness = authorize_customer(
        build_support_engine(
            config_root,
            policy=policy,
            reasoning=reasoning
            or FakeChatModel(force_tool="propose_refund", canned_args=_PROPOSE, tool_call_limit=1),
            risk_flagged=risk_flagged,
            thread_id=thread_id,
            routing_resolution=routing_resolution
            or RouteDecision.direct(
                RefundOrder(
                    target=ExplicitOrderTarget(order_ref="ORD-1002"),
                    amount_usd=129.0,
                    destination="new_instrument",
                )
            ),
        ),
        authorized_customer_ref,
    )
    return harness.engine, harness.store, harness.verification, harness.otp


async def _events(engine: ReasoningEngine, text: str, facts: TurnFacts = _FACTS) -> list:
    return await engine_events(engine, text, facts)


def _telemetry_events(engine: ReasoningEngine) -> list[dict[str, object]]:
    sink = engine._telemetry.sink
    assert isinstance(sink, InMemoryTelemetrySink)
    return [{"event": record.event, **record.attributes} for record in sink.records]


async def _assert_refund_destination_failed_closed(
    *,
    engine: ReasoningEngine,
    store: OrderStore,
    events: list,
    reason: str,
) -> None:
    assert store.refund_count == 0
    assert not any(isinstance(event, InterruptEvent | TokenEvent) for event in events)
    spoken = [event for event in events if isinstance(event, SpokenMessageEvent)]
    assert [event.node for event in spoken] == ["automation_terminal_response"]
    assert "contact the store" in spoken[0].text.lower()
    assert not await engine.apending_interrupt()

    snapshot = engine._graph.get_state(engine._config)
    assert snapshot.next == ()
    assert snapshot.values.get("automation_terminal") is True
    assert snapshot.values.get("execution_owner") is None
    assert snapshot.values.get("handover") is None
    assert snapshot.values.get("pending_refund") is None

    telemetry = _telemetry_events(engine)
    unavailable = [
        event for event in telemetry if event["event"] == "refund_destination_unavailable"
    ]
    assert unavailable == [{"event": "refund_destination_unavailable", "reason": reason}]
    onramps = [event for event in telemetry if event["event"] == "human_onramp"]
    assert len(onramps) == 1
    assert onramps[0]["reason_code"] == "refund"
    assert onramps[0]["source"] == "deterministic_policy"
    assert onramps[0]["execution_owner"] is None
    assert [event for event in telemetry if event["event"] == "automation_terminal_response"] == [
        {"event": "automation_terminal_response"}
    ]


async def _pause_at_otp(engine: ReasoningEngine) -> list:
    return await _events(engine, _REFUND_REQUEST)


# --- the happy T3 path -------------------------------------------------------------------


async def test_stepup_asks_for_otp_before_touching_money(config_root: Path) -> None:
    engine, store, verification, otp = _engine(config_root)
    events = await _pause_at_otp(engine)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "6-digit code" in interrupts[0].prompt
    assert otp.dispatch_count == 1  # OTP dispatched exactly once
    assert verification.current_level() == 1  # not raised yet
    assert store.refund_count == 0  # nothing refunded before verification


async def test_committed_otp_raises_level_then_reads_back_the_refund(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    events = await _events(engine, _VALID_OTP)
    assert verification.current_level() == 2  # raised mid-flow
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # GRAPH-authored, speech-native readback (§7a): order + amount + masked instrument, no PAN.
    # The order id/summary is named so a wrong owned-order refund is caught at consent (ownership
    # alone can't tell WHICH owned order the caller meant — INV-32).
    assert "ORD-1002" in interrupts[0].prompt
    assert "$129.00" in interrupts[0].prompt
    assert _instrument_ref(config_root, "CUST-002") in interrupts[0].prompt
    assert store.refund_count == 0  # still nothing placed before the yes


async def test_yes_after_stepup_issues_exactly_one_refund(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    events = await _events(engine, "yes please")
    assert store.refund_count == 1
    assert store.refunded_so_far("ORD-1002") == 129.0
    assert not await engine.apending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "support_place" and "refund" in e.text.lower() for e in spoken)


async def test_double_resume_of_the_confirm_never_double_refunds(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    await _events(engine, "yes")
    # A stray extra turn after the flow completed must not re-place (no interrupt pending).
    await _events(engine, "yes")
    assert store.refund_count == 1


# --- the security branches (T3 failure variants) ----------------------------------------


async def test_sim_swap_flag_blocks_otp_and_hands_to_human(config_root: Path) -> None:
    engine, store, verification, otp = _engine(config_root, risk_flagged=True)
    events = await _events(engine, _REFUND_REQUEST)
    # No OTP dispatched, no interrupt (we don't collect a code we can't trust), no refund.
    assert otp.dispatch_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.refund_count == 0
    assert verification.current_level() == 1


async def test_wrong_otp_twice_stays_l1_and_never_refunds(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    first = await _events(engine, "000000")  # wrong -> re-collect
    assert any(isinstance(e, InterruptEvent) for e in first)  # a second OTP ask
    await _events(engine, "111111")  # wrong again -> exhausted
    assert verification.current_level() == 1
    assert store.refund_count == 0
    assert not await engine.apending_interrupt()  # not trapped


async def test_otp_attempt_budget_tracks_the_policy_knob(config_root: Path) -> None:
    # otp_max_attempts is config-driven (policies.security): a merchant that raised it to 3
    # gets a THIRD OTP ask where the default (2) would already have handed to a human. Pins
    # that the value threads through build_stepup_nodes.
    engine, store, verification, _ = _engine(
        config_root, policy=_POLICY.model_copy(update={"otp_max_attempts": 3}), thread_id="otp3"
    )
    await _pause_at_otp(engine)
    first = await _events(engine, "000000")  # wrong #1 -> re-collect
    assert any(isinstance(e, InterruptEvent) for e in first)
    second = await _events(engine, "111111")  # wrong #2 -> STILL re-collect (budget is 3)
    assert any(isinstance(e, InterruptEvent) for e in second)
    await _events(engine, "222222")  # wrong #3 -> exhausted
    assert verification.current_level() == 1
    assert store.refund_count == 0
    assert not await engine.apending_interrupt()


async def test_live_read_blocks_a_lapsed_grant_at_place(config_root: Path) -> None:
    # The T3 security property: the level is read LIVE. If the grant is revoked after step-up
    # but before the effect, the place node re-validates and refuses to move money.
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    assert verification.current_level() == 2
    verification.clear()  # session grant revoked (SIM-swap flagged / session invalidated)
    await _events(engine, "yes")
    assert store.refund_count == 0  # money did NOT move on a lapsed level


# --- §4a committed-consent ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "facts", "thread_id"),
    [
        ("yes", TurnFacts(readback_interrupted=True), "refund-reconfirm-barged"),
        ("I'm not sure", _FACTS, "refund-reconfirm-unclear"),
    ],
)
async def test_refund_reconfirmation_repeats_policy_fields_before_refunding(
    config_root: Path,
    response: str,
    facts: TurnFacts,
    thread_id: str,
) -> None:
    engine, store, _, _ = _engine(config_root, thread_id=thread_id)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    events = await _events(engine, response, facts)
    assert store.refund_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    assert "ORD-1002" in reconfirms[0].prompt
    assert "$129.00" in reconfirms[0].prompt
    assert _instrument_ref(config_root, "CUST-002") in reconfirms[0].prompt
    await _events(engine, "yes")  # a clean committed yes places it
    assert store.refund_count == 1


async def test_refund_accepts_supported_natural_affirmation(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root, thread_id="refund-natural-affirmation")
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)

    await _events(engine, "sure")

    assert store.refund_count == 1
    assert not await engine.apending_interrupt()


async def test_no_at_readback_cancels_without_refunding(config_root: Path) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)
    events = await _events(engine, "no, don't")
    assert store.refund_count == 0
    assert not await engine.apending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("nothing has changed" in e.text.lower() for e in spoken)


# --- kill mid-step-up (Clock B) ----------------------------------------------------------


async def test_kill_mid_stepup_leaves_no_ghost_refund_and_no_free_level(config_root: Path) -> None:
    engine, store, verification, _ = _engine(config_root)
    await _pause_at_otp(engine)  # paused at the OTP collect interrupt
    assert await engine.apending_interrupt()
    await engine.adelete_thread()  # Clock-B teardown
    verification.clear()  # pipeline reaper also clears the grant
    assert store.refund_count == 0
    assert verification.current_level() == 1
    # A fresh session starts clean.
    engine2, store2, verification2, _ = _engine(
        config_root,
        thread_id="support-2",
        routing_resolution=RouteDecision.direct(AnswerQuestion(topic="general")),
    )
    events = await _events(engine2, "hi there")
    assert store2.refund_count == 0
    assert verification2.current_level() == 1
    assert any(isinstance(e, TokenEvent | SpokenMessageEvent) for e in events)


# --- Group A: refund-to-ORIGINAL (L1, no step-up) ----------------------------------------

# ORD-1002 is "processing" (cancellable), $129.00 captured.
_REFUND_ORIGINAL = {
    "propose_refund": {"order_key": "2", "amount_usd": 50.0, "destination": "original"}
}
_CANCEL_PROCESSING = {"propose_cancel": {"order_keys": ["2"]}}  # ORD-1002, processing
_CANCEL_SHIPPED = {"propose_cancel": {"order_keys": ["1"]}}  # ORD-1001, shipped
_ORDER_BY_KEY = {
    "1": "ORD-1001",
    "2": "ORD-1002",
    "3": "ORD-1003",
    "4": "ORD-9001",
    "5": "ORD-1004",
}


def _cancel_engine(config_root, args, *, thread_id, risk_flagged=False):
    keys = args["propose_cancel"]["order_keys"]
    customer_ref = "CUST-001" if keys and keys[0] in {"1", "3"} else "CUST-002"
    return _engine(
        config_root,
        risk_flagged=risk_flagged,
        thread_id=thread_id,
        authorized_customer_ref=customer_ref,
        routing_resolution=RouteDecision.direct(
            CancelOrders(
                target=ExplicitOrderSet(order_refs=tuple(_ORDER_BY_KEY[key] for key in keys))
            )
        ),
    )


async def test_refund_to_original_is_l1_no_stepup(config_root: Path) -> None:
    engine, store, verification, otp = _engine(
        config_root,
        routing_resolution=RouteDecision.direct(
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-1002"),
                amount_usd=50.0,
                destination="original",
            )
        ),
        thread_id="ro-1",
    )
    events = await _events(engine, "I want a refund to my original card")
    # NO OTP dispatched (L1 already satisfies refund-to-original) — straight to the readback.
    assert otp.dispatch_count == 0
    assert verification.current_level() == 1
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$50.00" in interrupts[0].prompt
    assert "your your" not in interrupts[0].prompt  # the double-'your' bug stays fixed
    await _events(engine, "yes")
    assert store.refund_count == 1
    assert store.refunded_so_far("ORD-1002") == 50.0


# --- Group A: cancel-order ---------------------------------------------------------------


async def test_cancel_processing_order_voids_after_readback(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-1")
    events = await _events(engine, "cancel my rain jacket order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt and "can't be undone" in interrupts[0].prompt
    assert store.cancel_count == 0  # nothing voided before consent
    events = await _events(engine, "yes")
    assert store.cancel_count == 1
    assert store.order_status("ORD-1002") == "cancelled"
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(e.node == "support_cancel_void" and "cancelled" in e.text.lower() for e in spoken)


async def test_cancel_shipped_order_declines_without_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_SHIPPED, thread_id="cx-2")
    events = await _events(engine, "cancel my shoes order")
    # No interrupt (we don't read back a cancel we can't do), no void, one honest line.
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 0
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("already shipped" in e.text.lower() for e in spoken)


async def test_cancel_risk_flagged_hands_to_human_without_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(
        config_root, _CANCEL_PROCESSING, thread_id="cx-3", risk_flagged=True
    )
    events = await _events(engine, "cancel my rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 0  # a risk-flagged session gets no silent void
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(
        e.node == "automation_terminal_response" and "contact the store" in e.text.lower()
        for e in spoken
    )


async def test_cancel_no_at_readback_leaves_order_untouched(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-4")
    await _events(engine, "cancel my rain jacket order")
    events = await _events(engine, "no, leave it")
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # untouched
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("leave that order" in e.text.lower() for e in spoken)


async def test_cancel_barged_readback_reconfirms_before_voiding(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-5")
    await _events(engine, "cancel my rain jacket order")
    # "yes" over a barged-over readback is not consent (§4a) -> re-confirm, not void.
    events = await _events(engine, "yes", TurnFacts(readback_interrupted=True))
    assert store.cancel_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    await _events(engine, "yes")  # a clean committed yes voids it
    assert store.cancel_count == 1


async def test_cancel_is_idempotent_across_double_resume(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-6")
    await _events(engine, "cancel my rain jacket order")
    await _events(engine, "yes")
    await _events(engine, "yes")  # a stray extra turn after completion must not re-void
    assert store.cancel_count == 1


async def test_kill_mid_cancel_leaves_no_ghost(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="cx-7")
    await _events(engine, "cancel my rain jacket order")  # paused at the cancel readback
    assert await engine.apending_interrupt()
    await engine.adelete_thread()
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # nothing voided on the drop


# --- F-16.2 batch cancel (single = batch-of-one; the above single-cancel tests still pass) -


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"order_keys": []},
        {"order_keys": [" "]},
        {"scope": "everything"},
        {"order_keys": ["2"], "scope": "all_cancellable"},
        {"order_key": "2"},  # stale scalar schema must not be silently ignored
    ],
)
def test_cancel_proposal_rejects_invalid_or_ambiguous_selectors(args: dict) -> None:
    with pytest.raises(ValueError):
        support_flow._ProposeCancel.model_validate(args)


def test_cancel_proposal_accepts_exactly_one_normalized_selector() -> None:
    explicit = support_flow._ProposeCancel.model_validate({"order_keys": [" 2 "]})
    scoped = support_flow._ProposeCancel.model_validate(
        {"order_keys": None, "scope": "both_cancellable"}
    )
    assert explicit.order_keys == ["2"] and explicit.scope is None
    assert scoped.order_keys == [] and scoped.scope == "both_cancellable"


async def test_stale_scalar_cancel_call_cannot_widen_to_account_scope(
    config_root: Path,
) -> None:
    from agnostic_market.commerce.identity import BoundIdentity

    harness = build_support_engine(
        config_root,
        policy=_POLICY,
        reasoning=FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_key": "2"}},
            tool_call_limit=1,
        ),
        thread_id="batch-stale-scalar",
    )
    harness.identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    _place_second_cancellable(harness)

    events = await _events(harness.engine, "cancel my order")
    assert not any(isinstance(event, InterruptEvent) for event in events)
    assert harness.store.cancel_count == 0
    assert _pending_cancel(harness, "batch-stale-scalar") is None


def _place_second_cancellable(harness) -> str:
    """Place a session cart order (ORD-9001, 'processing' => cancellable, session-authorized
    with no grant needed) so a batch has TWO cancellable targets. Its candidate KEY is '4'
    (after the three fixture orders in actionable_orders). Returns the order id."""
    from agnostic_market.dtos.state import CartLine

    placed = harness.store.place_cart(
        "seed-batch",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=60.0, quantity=1)],
        total_usd=60.0,
    )
    harness.guest_orders.record(placed.order_id)
    return placed.order_id


def _place_second_cancellable_extra(harness) -> str:
    """A THIRD cancellable session order (ORD-9002) — for the 'both' >2 clarify test."""
    from agnostic_market.dtos.state import CartLine

    placed = harness.store.place_cart(
        "seed-batch-2",
        lines=[CartLine(sku="SKU-GRN-15", name="socks", price_usd=14.5, quantity=1)],
        total_usd=14.5,
    )
    harness.guest_orders.record(placed.order_id)
    return placed.order_id


def _batch_engine(
    config_root,
    keys,
    *,
    thread_id,
    policy=_POLICY,
    place_second=False,
    same_customer_shipped=False,
):
    """Build a typed cancellation engine and optionally seed another cancellable order."""
    orders_fixture: OrdersFixture | None = None
    if same_customer_shipped:
        loaded = load_orders_fixture(config_root, "acme_store")
        shipped = loaded.orders["ORD-1001"].model_copy(update={"customer_ref": "CUST-002"})
        orders_fixture = loaded.model_copy(
            update={"orders": {**loaded.orders, "ORD-1004": shipped}}
        )
    customer_ref = "CUST-002" if any(key in {"2", "4", "5"} for key in keys) else "CUST-001"
    harness = authorize_customer(
        build_support_engine(
            config_root,
            policy=policy,
            orders_fixture=orders_fixture,
            routing_resolution=RouteDecision.direct(
                CancelOrders(
                    target=ExplicitOrderSet(order_refs=tuple(_ORDER_BY_KEY[key] for key in keys))
                )
            ),
            thread_id=thread_id,
        ),
        customer_ref,
    )
    if place_second:
        _place_second_cancellable(harness)
    return harness.engine, harness.store


async def test_cancel_both_voids_both_in_one_flow(config_root: Path) -> None:
    # THE F-16.2 pin: "cancel both" stages BOTH orders in ONE confirmation and voids both in
    # ONE flow — no "and the other one" turn for the model to fabricate a completion into.
    engine, store = _batch_engine(config_root, ["2", "4"], thread_id="batch-1", place_second=True)
    events = await _events(engine, "cancel both of my orders")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1  # ONE readback covers both
    assert "ORD-1002" in interrupts[0].prompt and "ORD-9001" in interrupts[0].prompt
    assert store.cancel_count == 0  # nothing voided before consent (INV-25)
    events = await _events(engine, "yes")
    assert store.cancel_count == 2  # BOTH voided
    assert store.order_status("ORD-1002") == "cancelled"
    assert store.order_status("ORD-9001") == "cancelled"
    spoken = " ".join(e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert "ORD-1002" in spoken and "ORD-9001" in spoken  # each named in the result


async def test_batch_cancel_mixes_eligible_and_shipped(config_root: Path) -> None:
    # Mixed batch under one principal: ORD-1002 processing + ORD-1004 shipped.
    # The readback names only the eligible order; yes voids only that order.
    engine, store = _batch_engine(
        config_root,
        ["2", "5"],
        thread_id="batch-2",
        same_customer_shipped=True,
    )
    events = await _events(engine, "cancel both")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt
    assert "already shipped" in interrupts[0].prompt.lower()  # the ineligible one stated
    events = await _events(engine, "yes")
    assert store.cancel_count == 1
    assert store.order_status("ORD-1002") == "cancelled"
    assert store.order_status("ORD-1004") == "shipped"  # untouched
    spoken = " ".join(e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert "ORD-1002" in spoken and "already shipped" in spoken.lower()


async def test_batch_cancel_all_ineligible_declines_without_readback(config_root: Path) -> None:
    # Both shipped/delivered: the guardrail speaks the whole-truth decline + ends, NO readback.
    engine, store = _batch_engine(config_root, ["1", "3"], thread_id="batch-3")  # shipped+delivered
    events = await _events(engine, "cancel both")
    assert not any(isinstance(e, InterruptEvent) for e in events)  # no readback
    assert store.cancel_count == 0
    spoken = " ".join(e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert "already shipped" in spoken.lower()


async def test_batch_cancel_dedups_repeated_key(config_root: Path) -> None:
    # "cancel both" that lists the SAME order twice mints ONE target -> ONE void.
    engine, store = _batch_engine(config_root, ["2", "2"], thread_id="batch-4")
    await _events(engine, "cancel both duplicate order references")
    await _events(engine, "yes")
    assert store.cancel_count == 1


async def test_batch_cancel_over_cap_asks_to_narrow(config_root: Path) -> None:
    # cancel_batch_max=1, a two-order batch: no void, ask the caller to narrow (never first-N).
    engine, store = _batch_engine(
        config_root,
        ["2", "4"],
        thread_id="batch-5",
        policy=make_policy(cancel_batch_max=1),
        place_second=True,
    )
    events = await _events(engine, "cancel both of my orders")
    assert not any(isinstance(e, InterruptEvent) for e in events)  # no readback
    assert store.cancel_count == 0  # never silently takes the first N
    spoken = " ".join(e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert "up to 1" in spoken


async def test_batch_cancel_idempotent_across_double_resume(config_root: Path) -> None:
    # A stray extra turn after a two-order batch completes must not re-void either target.
    engine, store = _batch_engine(config_root, ["2", "4"], thread_id="batch-6", place_second=True)
    await _events(engine, "cancel both")
    await _events(engine, "yes")
    await _events(engine, "yes")  # stray extra turn after completion
    assert store.cancel_count == 2  # exactly N, never N+1


async def test_batch_cancel_checkpoints_between_voids(config_root: Path) -> None:
    # The void self-loop checkpoints replay-safe progress between targets.
    engine, store = _batch_engine(config_root, ["2", "4"], thread_id="batch-7", place_second=True)
    await _events(engine, "cancel both")
    await _events(engine, "yes")  # drives the void self-loop across both targets
    assert store.cancel_count == 2


async def test_cancel_effect_revalidates_status_changed_after_readback(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store = _batch_engine(config_root, ["2"], thread_id="batch-stale-status")
    await _events(engine, "cancel my order")
    current_status = store.order_status

    def changed_status(order_id: str) -> str | None:
        return "shipped" if order_id.strip().upper() == "ORD-1002" else current_status(order_id)

    monkeypatch.setattr(store, "order_status", changed_status)

    events = await _events(engine, "yes")
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "shipped"
    spoken = " ".join(
        event.text for event in events if isinstance(event, SpokenMessageEvent)
    ).lower()
    assert "couldn't cancel" in spoken and "nothing changed" in spoken


async def test_engine_recovers_cancel_after_write_before_checkpoint(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = _batch_engine(config_root, ["2"], thread_id="batch-crash-replay")
    await _events(engine, "cancel my order")
    original = store.cancel_order
    crashed = False

    def write_then_crash(idempotency_key: str, *, order_id: str):
        nonlocal crashed
        record = original(idempotency_key, order_id=order_id)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated crash after the store committed")
        return record

    monkeypatch.setattr(store, "cancel_order", write_then_crash)
    recovered = await _events(engine, "yes")
    assert store.cancel_count == 1
    result_lines = [
        event
        for event in recovered
        if isinstance(event, SpokenMessageEvent) and event.node == RECOVERY_NODE_NAME
    ]
    assert len(result_lines) == 1 and "cancelled" in result_lines[0].text.lower()
    assert engine._graph.get_state(engine._config).next == ()


async def test_concurrent_double_resume_has_n_effects_and_one_result_line(
    config_root: Path,
) -> None:
    engine, store = _batch_engine(
        config_root, ["2", "4"], thread_id="batch-concurrent", place_second=True
    )
    await _events(engine, "cancel both")
    first, second = await asyncio.gather(
        _events(engine, "yes"),
        _events(engine, "yes"),
    )
    assert store.cancel_count == 2
    result_lines = [
        event
        for event in [*first, *second]
        if isinstance(event, SpokenMessageEvent) and event.node == "support_cancel_void"
    ]
    assert len(result_lines) == 1


# --- F-16.2 Milestone B: unbound "cancel all my orders" -> identity -> resolve -> batch ----


def _scope_engine(
    config_root,
    *,
    scope="all_cancellable",
    thread_id,
    bound=False,
    otp_max=2,
    cancel_batch_max=10,
    risk_flagged=False,
):
    """Engine for a 'cancel all/both my orders' scope. The frontline hands over to
    support/cancel_order; ONE reasoning fake serves BOTH support (propose_cancel scope) and
    identity (a turn-1 clarify, then propose_identity when the caller gives their contact).
    A second cancellable order (key '4') is seeded so a resolve yields two targets. If `bound`,
    the session is pre-verified so the scope resolves WITHOUT an identity detour."""
    from support_helpers import TEST_OTP  # noqa: F401  (the valid code the OtpProvider holds)

    scripted = [
        [],  # identity clarify on turn 1 (caller hasn't given a contact yet)
        [("propose_identity", {"contact_claim": "casey@example.com"})],  # identity assemble turn 2
    ]
    harness = build_support_engine(
        config_root,
        policy=make_policy(otp_max_attempts=otp_max, cancel_batch_max=cancel_batch_max),
        reasoning=FakeChatModel(scripted_calls=scripted),
        risk_flagged=risk_flagged,
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(
            CancelOrders(target=CancellableOrderScope(scope=scope))
        ),
    )
    _place_second_cancellable(harness)  # ORD-9001, session-owned, cancellable
    if bound:
        from agnostic_market.commerce.identity import BoundIdentity

        harness.identity.bind(
            BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
        )
    return harness


def _pending_cancel(harness, thread_id):
    assert harness.engine.thread_id == thread_id
    snap = harness.engine._graph.get_state(harness.engine._config)
    return snap.values.get("pending_cancel")


def _validated_state(harness, thread_id) -> ReasoningState:
    assert harness.engine.thread_id == thread_id
    snap = harness.engine._graph.get_state(harness.engine._config)
    return ReasoningState.model_validate(snap.values)


def _active_request(harness, thread_id):
    state = _validated_state(harness, thread_id)
    invocation = state.active_invocation
    if invocation is None:
        return None
    assert invocation.opened_turn_id in state.consumed_turn_ids
    return invocation.request


async def _enter_unbound_cancel_scope(harness) -> None:
    events = await _events(harness.engine, "cancel all my orders")
    warnings = [event for event in events if isinstance(event, InterruptEvent)]
    assert len(warnings) == 1 and "clear this call's" in warnings[0].prompt
    state = _validated_state(harness, harness.engine.thread_id)
    invocation = state.active_invocation
    assert invocation is not None
    assert invocation.opened_turn_id == state.consumed_turn_ids[-1]
    assert isinstance(invocation.request, CancelOrders)
    assert isinstance(invocation.request.target, CancellableOrderScope)
    await _events(harness.engine, "yes")


async def test_guest_context_warning_decline_preserves_guest_state(config_root: Path) -> None:
    h = _scope_engine(config_root, thread_id="mb-warning-decline")
    old_thread_id = h.engine.thread_id
    events = await _events(h.engine, "cancel all my orders")
    assert any(isinstance(event, InterruptEvent) for event in events)

    declined = await _events(h.engine, "no")

    assert any(
        "won't start" in event.text for event in declined if isinstance(event, SpokenMessageEvent)
    )
    assert h.engine.thread_id == old_thread_id
    assert h.identity.current() is None
    assert h.store.guest_orders(h.guest_orders)
    assert _active_request(h, old_thread_id) is None
    assert h.store.cancel_count == 0


async def test_guest_context_warning_human_exit_clears_invocation(config_root: Path) -> None:
    h = _scope_engine(config_root, thread_id="mb-warning-human")
    events = await _events(h.engine, "cancel all my orders")
    assert any(isinstance(event, InterruptEvent) for event in events)

    await _events(h.engine, "I want a person")

    state = _validated_state(h, "mb-warning-human")
    assert state.active_invocation is None
    assert state.automation_terminal is True
    assert h.store.cancel_count == 0


async def test_unbound_cancel_all_verifies_then_resolves_to_a_batch(config_root: Path) -> None:
    # THE Milestone B pin: unverified "cancel all my orders" -> OTP -> NO order-list speech ->
    # the resolver freezes a batch over the caller's cancellable orders -> one confirmation ->
    # both voided. Identity establishes the binding; support re-resolves + authorizes live.
    h = _scope_engine(config_root, thread_id="mb-1")
    old_thread_id = h.engine.thread_id
    old_config = h.engine._config
    await _enter_unbound_cancel_scope(h)
    e1 = await _events(h.engine, "casey@example.com")  # -> OTP dispatched
    assert any(isinstance(e, InterruptEvent) and "code" in e.prompt for e in e1)
    e2 = await _events(h.engine, "482913")  # OTP -> bind -> resolve -> readback
    # NO order-list line spoken on the continuation (the list-speech branch is skipped).
    spoken2 = [e for e in e2 if isinstance(e, SpokenMessageEvent)]
    assert not any("you've got" in e.text.lower() for e in spoken2)
    interrupts = [e for e in e2 if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1002" in interrupts[0].prompt
    assert "ORD-9001" not in interrupts[0].prompt
    assert h.engine.thread_id != old_thread_id
    assert await h.engine.apending_interrupt()
    with pytest.raises(CheckpointScopeError, match="namespace"):
        h.engine._graph.get_state(old_config)
    assert h.store.cancel_count == 0  # nothing voided before consent
    await _events(h.engine, "yes")
    assert h.store.cancel_count == 1
    assert h.store.order_status("ORD-1002") == "cancelled"
    assert h.store.order_status("ORD-9001") == "processing"
    assert h.store.guest_orders(h.guest_orders) == []


async def test_bound_cancel_all_skips_identity(config_root: Path) -> None:
    # An ALREADY-verified caller's "cancel all" resolves immediately — no OTP, straight to the
    # batch readback on the SAME turn.
    h = _scope_engine(config_root, thread_id="mb-2", bound=True)
    events = await _events(h.engine, "cancel all my orders")
    assert not any(isinstance(e, InterruptEvent) and "code" in e.prompt for e in events)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1 and "ORD-1002" in interrupts[0].prompt
    await _events(h.engine, "yes")
    assert h.store.cancel_count == 2


async def test_failed_otp_clears_the_retained_cancel_selection(config_root: Path) -> None:
    # SECURITY: a FAILED verification during a "cancel all" detour must leave ZERO cancel
    # intent — the retained typed request is cleared on OTP exhaustion, nothing voided.
    h = _scope_engine(config_root, thread_id="mb-3", otp_max=2)
    await _enter_unbound_cancel_scope(h)
    await _events(h.engine, "casey@example.com")
    await _events(h.engine, "000000")  # wrong OTP 1 -> re-collect
    await _events(h.engine, "111111")  # wrong OTP 2 -> exhausted -> human
    assert _pending_cancel(h, "mb-3") is None  # selector dropped
    assert h.store.cancel_count == 0
    assert h.identity.current() is None  # never bound


async def test_cancel_both_scope_with_more_than_two_asks_to_narrow(config_root: Path) -> None:
    # "cancel both" is only unambiguous with exactly two cancellable orders. With three
    # (ORD-1002 + two placed), the resolver asks which — no readback, nothing voided.
    h = _scope_engine(config_root, scope="both_cancellable", thread_id="mb-4", bound=True)
    _place_second_cancellable_extra(h)  # a THIRD cancellable order
    events = await _events(h.engine, "cancel both of my orders")
    assert not any(isinstance(e, InterruptEvent) for e in events)  # no readback
    assert h.store.cancel_count == 0
    spoken = " ".join(e.text for e in events if isinstance(e, SpokenMessageEvent))
    assert "which" in spoken.lower()


async def test_cancel_both_scope_with_one_candidate_is_truthful(config_root: Path) -> None:
    h = _scope_engine(config_root, scope="both_cancellable", thread_id="mb-one", bound=True)
    h.store.cancel_order("already-done", order_id="ORD-1002")
    events = await _events(h.engine, "cancel both of my orders")
    assert not any(isinstance(event, InterruptEvent) for event in events)
    spoken = " ".join(
        event.text for event in events if isinstance(event, SpokenMessageEvent)
    ).lower()
    assert "only one" in spoken
    assert "more than two" not in spoken
    assert h.store.cancel_count == 1


async def test_cancel_all_exactly_cap_plus_one_never_truncates(config_root: Path) -> None:
    h = _scope_engine(config_root, thread_id="mb-cap-plus-one", bound=True, cancel_batch_max=1)
    events = await _events(h.engine, "cancel all my orders")
    assert not any(isinstance(event, InterruptEvent) for event in events)
    spoken = " ".join(event.text for event in events if isinstance(event, SpokenMessageEvent))
    assert "up to 1" in spoken
    assert h.store.cancel_count == 0
    assert _pending_cancel(h, "mb-cap-plus-one") is None


async def test_scope_resolver_reauthorizes_every_query_result(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agnostic_market.commerce.orders import OrderCandidate

    h = _scope_engine(config_root, thread_id="mb-cross-owner", bound=True)
    foreign = OrderCandidate(
        key="1",
        order_id="ORD-1001",
        summary="trail running shoes",
        total_usd=179.98,
        status="processing",
    )
    monkeypatch.setattr(
        h.store, "owned_cancellable_orders", lambda *_args, **_kwargs: ([foreign], False)
    )
    events = await _events(h.engine, "cancel all my orders")
    assert not any(isinstance(event, InterruptEvent) for event in events)
    spoken = " ".join(
        event.text for event in events if isinstance(event, SpokenMessageEvent)
    ).lower()
    assert "safely match" in spoken and "haven't changed" in spoken
    assert h.store.cancel_count == 0
    assert _pending_cancel(h, "mb-cross-owner") is None


async def test_risk_after_identity_clears_scope_without_voiding(config_root: Path) -> None:
    h = _scope_engine(config_root, thread_id="mb-risk", risk_flagged=True)
    await _enter_unbound_cancel_scope(h)
    await _events(h.engine, "casey@example.com")
    events = await _events(h.engine, "482913")
    assert not any(isinstance(event, InterruptEvent) for event in events)
    assert _pending_cancel(h, "mb-risk") is None
    assert _validated_state(h, "mb-risk").active_invocation is None
    assert h.store.cancel_count == 0


@pytest.mark.parametrize(
    ("utterance", "thread_id"),
    [
        ("never mind, forget it", "mb-abort"),
        ("I want a person", "mb-human"),
    ],
)
async def test_identity_exit_paths_clear_retained_cancel_scope(
    config_root: Path, utterance: str, thread_id: str
) -> None:
    h = _scope_engine(config_root, thread_id=thread_id)
    await _enter_unbound_cancel_scope(h)
    assert _active_request(h, thread_id) is not None

    await _events(h.engine, utterance)
    assert _active_request(h, thread_id) is None
    assert _pending_cancel(h, thread_id) is None
    assert h.store.cancel_count == 0


# --- F-1: refund amount gate (merchant policy, within platform bounds) --------------------


def _refund_engine(config_root, amount, dest, *, thread_id, policy=_POLICY):
    # ORD-1001 (key "1") is shipped but total 179.98; refunds don't need it cancellable.
    return _engine(
        config_root,
        policy=policy,
        thread_id=thread_id,
        authorized_customer_ref="CUST-001",
        routing_resolution=RouteDecision.direct(
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-1001"),
                amount_usd=amount,
                destination=dest,
            )
        ),
    )


async def test_new_instrument_reference_follows_the_authorized_order_owner(
    config_root: Path,
) -> None:
    engine, store, _, _ = _refund_engine(
        config_root,
        100.0,
        "new_instrument",
        thread_id="instrument-owner",
    )
    await _events(engine, "refund my order to a different card")
    events = await _events(engine, _VALID_OTP)
    interrupts = [event for event in events if isinstance(event, InterruptEvent)]
    assert len(interrupts) == 1
    assert _instrument_ref(config_root, "CUST-001") in interrupts[0].prompt
    assert _instrument_ref(config_root, "CUST-002") not in interrupts[0].prompt
    completed = await _events(engine, "yes")
    assert store.refund_count == 1
    spoken = [event for event in completed if isinstance(event, SpokenMessageEvent)]
    assert len(spoken) == 1 and spoken[0].node == "support_place"
    assert _instrument_ref(config_root, "CUST-001") in spoken[0].text
    assert _instrument_ref(config_root, "CUST-002") not in spoken[0].text


async def test_unmodelled_refund_destination_fails_closed(
    config_root: Path,
) -> None:
    engine, store, _, _ = _refund_engine(
        config_root,
        100.0,
        "new_address",
        thread_id="refund-new-address",
    )
    events = await _events(engine, "refund my order to a different address")
    await _assert_refund_destination_failed_closed(
        engine=engine,
        store=store,
        events=events,
        reason="new_address",
    )


async def test_missing_new_instrument_fails_closed(config_root: Path) -> None:
    harness = authorize_customer(
        build_support_engine(
            config_root,
            policy=_POLICY,
            reasoning=FakeChatModel(
                force_tool="propose_refund",
                canned_args={
                    "propose_refund": {
                        "order_key": "1",
                        "amount_usd": 100.0,
                        "destination": "new_instrument",
                    }
                },
                tool_call_limit=1,
            ),
            thread_id="refund-missing-instrument",
            payment_instruments_fixture=PaymentInstrumentsFixture(payment_instruments={}),
            routing_resolution=RouteDecision.direct(
                RefundOrder(
                    target=ExplicitOrderTarget(order_ref="ORD-1001"),
                    amount_usd=100.0,
                    destination="new_instrument",
                )
            ),
        ),
        "CUST-001",
    )

    events = await _events(harness.engine, "refund $100 for order ORD-1001 to a different card")
    await _assert_refund_destination_failed_closed(
        engine=harness.engine,
        store=harness.store,
        events=events,
        reason="new_instrument",
    )


async def test_session_order_without_account_owner_cannot_select_new_instrument(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        force_tool="propose_refund",
        canned_args={
            "propose_refund": {
                "order_key": "ORD-9001",
                "amount_usd": 20.0,
                "destination": "new_instrument",
            }
        },
        tool_call_limit=1,
    )
    harness = build_support_engine(
        config_root,
        policy=_POLICY,
        reasoning=reasoning,
        thread_id="refund-session-order-no-owner",
        routing_resolution=RouteDecision.direct(
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-9001"),
                amount_usd=20.0,
                destination="new_instrument",
            )
        ),
    )
    placed = harness.store.place_cart(
        "session-order",
        lines=[
            CartLine(
                sku="SKU-SESSION",
                name="session item",
                price_usd=20.0,
                quantity=1,
            )
        ],
        total_usd=20.0,
    )
    harness.guest_orders.record(placed.order_id)
    assert placed.order_id == "ORD-9001"

    events = await _events(
        harness.engine,
        "refund $20 for order ORD-9001 to a different card",
    )
    await _assert_refund_destination_failed_closed(
        engine=harness.engine,
        store=harness.store,
        events=events,
        reason="new_instrument",
    )


async def test_refund_in_band_reaches_readback(config_root: Path) -> None:
    # $150 <= require_human_above (200) -> agent processes behind the readback.
    engine, store, _, _ = _refund_engine(config_root, 150.0, "original", thread_id="amt-1")
    events = await _events(engine, "I want a refund to my original card")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$150.00" in interrupts[0].prompt
    assert store.refund_count == 0  # not until the yes


async def test_refund_over_human_threshold_routes_to_person_no_refund(config_root: Path) -> None:
    # $250 > require_human_above (200) -> a person, NO readback, NO refund, specific line.
    engine, store, _, otp = _refund_engine(config_root, 250.0, "original", thread_id="amt-2")
    events = await _events(engine, "I want a refund to my original card")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.refund_count == 0
    assert otp.dispatch_count == 0  # no step-up either — it just goes to a human
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("above what I can process" in e.text for e in spoken)


async def test_refund_amount_gate_precedes_stepup(config_root: Path) -> None:
    # An over-threshold refund to a NEW instrument (would need L2) must NOT enter the OTP
    # loop — the amount gate routes to a human FIRST (no point verifying a refund a human
    # must handle anyway).
    engine, store, _, otp = _refund_engine(config_root, 300.0, "new_instrument", thread_id="amt-3")
    events = await _events(engine, "refund my order to a different card")
    assert otp.dispatch_count == 0  # amount gate fired before dispatch
    assert store.refund_count == 0
    assert not any(isinstance(e, InterruptEvent) for e in events)


# --- F-2: Clock-A TTL on the support confirm nodes (clear-before-speak) -------------------


async def test_stale_refund_readback_expires_before_placing(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store, _, _ = _engine(config_root)
    await _pause_at_otp(engine)
    await _events(engine, _VALID_OTP)  # raised to L2, now paused at the refund readback
    # Jump the flow clock past the TTL; the resume must find the pending expired.
    future = time.time() + 10_000
    monkeypatch.setattr(support_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")  # a stale yes must NOT refund
    assert store.refund_count == 0
    assert not await engine.apending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)


async def test_stale_cancel_readback_expires_before_voiding(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="ttl-c")
    await _events(engine, "cancel my rain jacket order")  # paused at the cancel readback
    future = time.time() + 10_000
    monkeypatch.setattr(support_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")  # a stale yes must NOT void
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"
    assert not await engine.apending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("sat for a while" in e.text for e in spoken)


# --- remedy coherence: refund vs cancel (2026-07-10 live-call findings) --------------------

# A FULL refund to the ORIGINAL instrument on ORD-1002 (processing, $129.00 captured):
# money-back on an unshipped order IS a cancellation — the guardrail steers it.
_FULL_REFUND_ORIGINAL = {
    "propose_refund": {"order_key": "2", "amount_usd": 129.0, "destination": "original"}
}


def _refund_original_engine(config_root, args, *, thread_id, policy=_POLICY):
    proposal = args["propose_refund"]
    return _engine(
        config_root,
        policy=policy,
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(
            RefundOrder(
                target=ExplicitOrderTarget(order_ref=_ORDER_BY_KEY[proposal["order_key"]]),
                amount_usd=proposal["amount_usd"],
                destination=proposal["destination"],
            )
        ),
    )


async def test_full_refund_on_unshipped_order_steers_to_cancel(config_root: Path) -> None:
    engine, store, _, otp = _refund_original_engine(
        config_root, _FULL_REFUND_ORIGINAL, thread_id="steer-1"
    )
    events = await _events(engine, "I want my money back for the rain jacket order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("hasn't shipped yet" in e.text for e in spoken)  # the remedy explained
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1  # the CANCEL readback, not the refund one
    assert "cancel" in interrupts[0].prompt.lower()
    assert "ORD-1002" in interrupts[0].prompt
    assert otp.dispatch_count == 0  # a void needs no step-up
    await _events(engine, "yes")
    assert store.cancel_count == 1
    assert store.refund_count == 0  # the money comes back via the void, never twice
    assert store.order_status("ORD-1002") == "cancelled"


async def test_steer_precedes_the_amount_gate(config_root: Path) -> None:
    # The live incoherence (2026-07-10): a full "$258 refund" was sent to a person by the
    # amount gate while a self-serve cancel of the same order was available. The steer must
    # win over the threshold — a void is self-serve at any amount.
    tight = _POLICY.model_copy(update={"refund_require_human_above_usd": 100.0})
    engine, store, _, _ = _refund_original_engine(
        config_root, _FULL_REFUND_ORIGINAL, thread_id="steer-2", policy=tight
    )
    events = await _events(engine, "I want my money back for the rain jacket order")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert not any("above what I can process" in e.text for e in spoken)
    assert any(isinstance(e, InterruptEvent) for e in events)  # the cancel readback
    await _events(engine, "yes")
    assert store.cancel_count == 1


async def test_partial_refund_on_unshipped_order_is_not_steered(config_root: Path) -> None:
    # A PARTIAL amount is not "undo the order" — it stays a refund (price-adjustment shape).
    engine, store, _, _ = _refund_original_engine(
        config_root, _REFUND_ORIGINAL, thread_id="steer-3"
    )
    events = await _events(engine, "refund me fifty dollars on the rain jacket order")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$50.00" in interrupts[0].prompt  # the REFUND readback
    await _events(engine, "yes")
    assert store.refund_count == 1
    assert store.cancel_count == 0


async def test_refund_on_a_cancelled_order_declines_honestly(config_root: Path) -> None:
    # The live double-dip setup (2026-07-10): ORD cancelled, then an under-threshold refund
    # request against it — must decline with the honest line, never touch money.
    engine, store, _, _ = _refund_original_engine(
        config_root, _REFUND_ORIGINAL, thread_id="steer-4"
    )
    store.cancel_order("prior-void", order_id="ORD-1002")  # voided earlier in the session
    events = await _events(engine, "I want a refund for the rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.refund_count == 0
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("nothing left to refund" in e.text for e in spoken)


async def test_cancel_after_a_partial_refund_declines_without_voiding(config_root: Path) -> None:
    # The reverse double-dip: a partial refund exists, then a cancel request — a void would
    # return the full charge ON TOP of the refund. Declines to support, order untouched.
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="steer-5")
    store.issue_refund(
        "prior-refund",
        order_id="ORD-1002",
        amount_usd=30.0,
        destination="original",
        instrument_ref=_ORIGINAL_INSTRUMENT,
    )
    events = await _events(engine, "cancel my rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 0
    assert store.order_status("ORD-1002") == "processing"  # nothing voided
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("already has a refund" in e.text for e in spoken)


async def test_cancelled_order_cancel_request_says_nothing_more_to_do(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="steer-6")
    store.cancel_order("prior-void", order_id="ORD-1002")
    events = await _events(engine, "cancel my rain jacket order")
    assert not any(isinstance(e, InterruptEvent) for e in events)
    assert store.cancel_count == 1  # only the prior one; no second void, no wrong 'shipped' line
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("already cancelled" in e.text for e in spoken)
    assert not any("shipped" in e.text for e in spoken)


# --- return-first: a refund on a SHIPPED/DELIVERED order above the merchant's returnless
# ---  window waits for the return (2026-07-10 live: $179.98 paid out on shipped shoes with
# ---  no return created — money back AND goods kept). Group C: the terminal decline became
# ---  a STEER into the returns sub-path (the refund converts into an arranged return). ----


async def test_shipped_refund_over_returnless_window_steers_into_a_return(
    config_root: Path,
) -> None:
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 50.0})
    engine, store, _, _ = _refund_engine(
        config_root, 150.0, "original", thread_id="ret-1", policy=tight
    )
    events = await _events(engine, "I want a refund to my original card")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("once the return is set up" in e.text for e in spoken)  # the steer line
    # The steer pauses at the RETURN readback — order named, refund amount stated, original
    # payment method promised (v1 destination constant), no money moved yet.
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "ORD-1001" in interrupts[0].prompt
    assert "$150.00" in interrupts[0].prompt
    assert "original payment method" in interrupts[0].prompt
    assert store.refund_count == 0
    assert store.return_count == 0
    # Committed yes -> ONE return created, refund still zero (it releases at Phase 4).
    done = await _events(engine, "yes")
    assert store.return_count == 1
    assert store.refund_count == 0
    spoken = [e for e in done if isinstance(e, SpokenMessageEvent)]
    assert any("RMA-3001" in e.text and e.node == "support_return_place" for e in spoken)


async def test_shipped_refund_within_returnless_window_pays_out(config_root: Path) -> None:
    # At/under the merchant's returnless line the payout is a DELIBERATE policy (return
    # shipping would cost more than the goods) — proceeds to the normal readback.
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 50.0})
    engine, store, _, _ = _refund_engine(
        config_root, 30.0, "original", thread_id="ret-2", policy=tight
    )
    events = await _events(engine, "I want a refund to my original card")
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert "$30.00" in interrupts[0].prompt
    await _events(engine, "yes")
    assert store.refund_count == 1


async def test_amount_gate_precedes_return_first(config_root: Path) -> None:
    # $250 is over BOTH lines on a shipped order. The human-review line is the STRONGER gate
    # (an authorization ceiling) and must win — a person handling the amount also arranges
    # the return, whereas leading with "just set up a return" hides that the amount needs a
    # person (live 2026-07-10: with returnless=50 the return line masked the $200 human line
    # on every shipped refund over $50, so the $200 limit looked ignored).
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 50.0})
    engine, store, _, _ = _refund_engine(
        config_root, 250.0, "original", thread_id="ret-3", policy=tight
    )
    events = await _events(engine, "I want a refund to my original card")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("above what I can process" in e.text for e in spoken)
    assert not any("once the return is set up" in e.text for e in spoken)
    assert store.refund_count == 0


async def test_shipped_refund_between_returnless_and_human_lines_is_return_first(
    config_root: Path,
) -> None:
    # $150 is over returnless (50) but under the human line (200): the agent COULD process
    # it, so return-first is the right lead — "I can refund, but it needs a return first".
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 50.0})
    engine, store, _, _ = _refund_engine(
        config_root, 150.0, "original", thread_id="ret-3b", policy=tight
    )
    events = await _events(engine, "I want a refund to my original card")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("once the return is set up" in e.text for e in spoken)
    assert not any("above what I can process" in e.text for e in spoken)
    assert store.refund_count == 0


async def test_delivered_order_is_return_first_too(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ORD-1003 was DELIVERED 2026-07-01 (fixture): freeze the flow clock a few days after
    # delivery so the window check is deterministic (fixture dates age against wall clock).
    monkeypatch.setattr(support_flow.time, "time", lambda: 1783641600.0)  # 2026-07-10 UTC
    tight = _POLICY.model_copy(update={"refund_returnless_under_usd": 10.0})
    engine, store, _, _ = _engine(
        config_root,
        policy=tight,
        thread_id="ret-4",
        authorized_customer_ref="CUST-001",
        routing_resolution=RouteDecision.direct(
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-1003"),
                amount_usd=40.0,
                destination="original",
            )
        ),
    )
    events = await _events(engine, "I want a refund for my socks order")  # ORD-1003, delivered
    assert store.refund_count == 0
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("once the return is set up" in e.text for e in spoken)  # steered, in window


# --- cancel-consent polarity (2026-07-10 live: "yeah cancel it" hit the ABORT escape,
# ---  cancelled NOTHING, and sounded like it had) ------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yeah cancel it", "yes"),  # the live utterance — consent, not an abort/no
        ("yes, cancel the order", "yes"),
        ("go ahead, cancel it", "yes"),
        ("don't cancel", "no"),  # negation survives the neutralization
        ("no, leave it", "no"),
        ("cancel it", "unclear"),  # bare command -> the one §4a re-confirm asks yes/no
    ],
)
def test_classify_cancel_consent_polarity(text: str, expected: str) -> None:
    from agnostic_market.agents._consent import classify_cancel_consent

    assert classify_cancel_consent(text) == expected


async def test_cancel_phrase_during_typed_support_clarification_is_not_an_abort(
    config_root: Path,
) -> None:
    # Subject-matter "cancel" must remain with the active Support owner, not trigger abort.
    engine, store, _, _ = _engine(
        config_root,
        reasoning=FakeChatModel(emit_tool_calls=False),  # clarifies every turn -> sticky
        thread_id="pol-1",
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )
    await _events(engine, "I want a refund for my order")
    events = await _events(engine, "yeah cancel it")
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert not any("dropped that" in e.text for e in spoken)  # abort did NOT fire
    assert not any(isinstance(e, TokenEvent) for e in events)
    assert [(e.node, e.text) for e in spoken] == [
        (
            "support_clarify",
            "What is the order number, for example ORD-1234?",
        ),
    ]
    assert store.cancel_count == 0  # and nothing was silently voided either


@pytest.mark.parametrize(
    ("typed_request", "detail", "line"),
    [
        (
            RefundOrder(),
            "order",
            "What is the order number, for example ORD-1234?",
        ),
        (
            RefundOrder(target=ExplicitOrderTarget(order_ref="ORD-1001")),
            "amount",
            "What amount would you like refunded?",
        ),
        (
            RefundOrder(
                target=ExplicitOrderTarget(order_ref="ORD-1001"),
                amount_usd=20.0,
            ),
            "refund_destination",
            "Should the refund go back to the original payment method?",
        ),
    ],
)
async def test_incomplete_typed_support_request_selects_one_code_authored_line(
    config_root: Path, typed_request: IntentRequest, detail: str, line: str
) -> None:
    reasoning = FakeChatModel(emit_tool_calls=False)
    engine, store, _, _ = _engine(
        config_root,
        reasoning=reasoning,
        thread_id=f"support-clarify-{detail}",
        authorized_customer_ref=(
            "CUST-001"
            if isinstance(typed_request, RefundOrder)
            and isinstance(typed_request.target, ExplicitOrderTarget)
            else "CUST-002"
        ),
        routing_resolution=RouteDecision.direct(typed_request),
    )

    events = await _events(engine, "I need help with a refund")

    assert reasoning.invoke_count == 1
    assert not any(isinstance(e, TokenEvent) for e in events)
    assert [(e.node, e.text) for e in events if isinstance(e, SpokenMessageEvent)] == [
        ("support_clarify", line)
    ]
    state = engine._graph.get_state(engine._config).values
    assert state["execution_owner"] == "support"
    assert state.get("pending_clarification") is None
    assert store.refund_count == store.return_count == store.cancel_count == 0


async def test_typed_support_model_prose_falls_back_to_the_missing_slot_question(
    config_root: Path,
) -> None:
    reasoning = FakeChatModel(
        emit_tool_calls=False,
        text_response="I already cancelled that order.",
    )
    engine, store, _, _ = _engine(
        config_root,
        reasoning=reasoning,
        thread_id="support-clarify-no-tool",
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )

    events = await _events(engine, "I need a refund")

    assert reasoning.invoke_count == 1
    assert not any(isinstance(e, TokenEvent) for e in events)
    assert [(e.node, e.text) for e in events if isinstance(e, SpokenMessageEvent)] == [
        (
            "support_clarify",
            "What is the order number, for example ORD-1234?",
        )
    ]
    state = engine._graph.get_state(engine._config).values
    assert not any(
        isinstance(message, AIMessage) and message.content == reasoning.text_response
        for message in state["messages"]
    )
    assert store.refund_count == store.return_count == store.cancel_count == 0


@pytest.mark.parametrize(
    ("second_call", "unknown_results"),
    [
        ([("request_support_clarification", {"detail": "action"})], 1),
        ([("catalog_lookup", {"query": "shoes"})], 2),
    ],
)
async def test_unknown_support_tool_uses_bounded_correction_without_an_effect(
    config_root: Path,
    second_call: list[tuple[str, dict]],
    unknown_results: int,
) -> None:
    thread_id = f"support-unknown-{unknown_results}"
    reasoning = FakeChatModel(
        scripted_calls=[
            [("catalog_lookup", {"query": "shoes"})],
            second_call,
        ]
    )
    engine, store, verification, otp = _engine(
        config_root,
        reasoning=reasoning,
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )

    events = await _events(engine, "I need a refund")

    state = engine._graph.get_state(engine._config).values
    assert reasoning.invoke_count == 2
    assert (
        sum(
            isinstance(message, ToolMessage)
            and str(message.content).startswith("Unavailable action.")
            for message in state["messages"]
        )
        == unknown_results
    )
    assert [
        (event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)
    ] == [
        (
            "support_clarify",
            "What is the order number, for example ORD-1234?",
        )
    ]
    assert not any(isinstance(event, TokenEvent | InterruptEvent) for event in events)
    assert verification.current_level() == 1
    assert otp.dispatch_count == 0
    assert store.refund_count == store.return_count == store.cancel_count == 0


async def test_repeated_support_clarification_exhausts_without_an_effect(
    config_root: Path,
) -> None:
    thread_id = "support-clarify-exhausted"
    engine, store, verification, otp = _engine(
        config_root,
        reasoning=FakeChatModel(emit_tool_calls=False),
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )

    first = await _events(engine, "I need a refund")
    second = await _events(engine, "I'm not sure.")
    third = await _events(engine, "I still don't know.")
    exhausted = await _events(engine, "Can you just help?")

    assert [event.node for event in first if isinstance(event, SpokenMessageEvent)] == [
        "support_clarify"
    ]
    assert [event.node for event in second if isinstance(event, SpokenMessageEvent)] == [
        "support_clarify"
    ]
    assert [event.node for event in third if isinstance(event, SpokenMessageEvent)] == [
        "support_clarify"
    ]
    assert [event.node for event in exhausted if isinstance(event, SpokenMessageEvent)] == [
        "automation_terminal_response"
    ]
    state = ReasoningState.model_validate(engine._graph.get_state(engine._config).values)
    assert state.automation_terminal is True
    assert state.execution_owner is None
    assert state.active_invocation is None
    assert state.clarification_liveness is None
    assert verification.current_level() == 1
    assert otp.dispatch_count == 0
    assert store.refund_count == store.return_count == store.cancel_count == 0
    exhausted_events = [
        event for event in _telemetry_events(engine) if event["event"] == "clarification_exhausted"
    ]
    assert exhausted_events == [
        {
            "event": "clarification_exhausted",
            "owner_kind": "invocation",
            "consumed_reasks": 2,
            "limit": 2,
        }
    ]


async def test_changing_support_clarification_detail_does_not_reset_the_budget(
    config_root: Path,
) -> None:
    thread_id = "support-clarify-changing-detail"
    reasoning = FakeChatModel(
        scripted_calls=[
            [("request_support_clarification", {"detail": "action"})],
            [("request_support_clarification", {"detail": "order"})],
            [("request_support_clarification", {"detail": "amount"})],
            [("request_support_clarification", {"detail": "refund_destination"})],
        ]
    )
    engine, store, _, _ = _engine(
        config_root,
        reasoning=reasoning,
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )

    await _events(engine, "I need a refund")
    await _events(engine, "It is about an order.")
    await _events(engine, "Maybe part of the amount.")
    exhausted = await _events(engine, "I don't know where it should go.")

    assert [event.node for event in exhausted if isinstance(event, SpokenMessageEvent)] == [
        "automation_terminal_response"
    ]
    state = engine._graph.get_state(engine._config).values
    assert state.get("clarification_liveness") is None
    assert store.refund_count == store.return_count == store.cancel_count == 0
    telemetry = _telemetry_events(engine)
    assert sum(event["event"] == "clarification_exhausted" for event in telemetry) == 1


async def test_support_two_malformed_clarification_calls_are_paired_then_fall_back(
    config_root: Path,
) -> None:
    malformed = [
        (
            "request_support_clarification",
            {"detail": "order", "unexpected": "not allowed"},
        )
    ]
    reasoning = FakeChatModel(scripted_calls=[malformed, malformed])
    thread_id = "support-clarify-malformed"
    engine, store, _, _ = _engine(
        config_root,
        reasoning=reasoning,
        thread_id=thread_id,
        routing_resolution=RouteDecision.direct(RefundOrder()),
    )

    events = await _events(engine, "I need a refund")
    second_events = await _events(engine, "I still need help")

    assert reasoning.invoke_count == 2
    assert [
        (e.node, e.text) for e in [*events, *second_events] if isinstance(e, SpokenMessageEvent)
    ] == [
        (
            "support_clarify",
            "What is the order number, for example ORD-1234?",
        ),
        (
            "support_clarify",
            "What is the order number, for example ORD-1234?",
        ),
    ]
    state = engine._graph.get_state(engine._config).values
    tool_use_ids = {
        call["id"]
        for message in state["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    tool_result_ids = {
        message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)
    }
    assert tool_use_ids == tool_result_ids
    assert state["execution_owner"] == "support"
    assert state["clarification_liveness"].reasks == 1
    assert store.refund_count == store.return_count == store.cancel_count == 0


@pytest.mark.parametrize(
    ("proposal_model", "arguments"),
    [
        (
            support_flow._ProposeCancel,
            {"order_keys": ["1"], "unexpected": True},
        ),
        (
            support_flow._ProposeReturn,
            {"order_key": "1", "unexpected": True},
        ),
        (
            support_flow._ProposeProfileChange,
            {"field": "contact", "new_value": "+1 555 010 0000", "unexpected": True},
        ),
    ],
)
def test_support_proposal_models_reject_extra_fields(
    proposal_model: type, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        proposal_model.model_validate(arguments)


async def test_yeah_cancel_it_at_the_cancel_readback_is_consent(config_root: Path) -> None:
    # At the readback the question IS "shall I cancel?" — plain classify_consent would read
    # the 'cancel' as a NO and leave the order standing while the caller believes it's gone.
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="pol-2")
    await _events(engine, "cancel my rain jacket order")  # paused at the cancel readback
    await _events(engine, "yeah cancel it")
    assert store.cancel_count == 1
    assert store.order_status("ORD-1002") == "cancelled"


async def test_bare_cancel_it_at_readback_reconfirms_once(config_root: Path) -> None:
    engine, store, _, _ = _cancel_engine(config_root, _CANCEL_PROCESSING, thread_id="pol-3")
    await _events(engine, "cancel my rain jacket order")
    events = await _events(engine, "cancel it")  # neutralized -> unclear -> re-confirm
    assert store.cancel_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    # The retry names the order too — the caller may be here because they questioned
    # WHICH order it is (live: "isn't my most recent ORD-1001?").
    assert "ORD-1002" in reconfirms[0].prompt
    await _events(engine, "yes")
    assert store.cancel_count == 1


# --- F-4: a multi-tool-call response must not leave a dangling tool_use -------------------


async def test_support_double_tool_call_is_acked_in_thread_history(config_root: Path) -> None:
    engine, store, _, _ = _engine(
        config_root,
        reasoning=FakeChatModel(
            force_tool="propose_cancel",
            canned_args={"propose_cancel": {"order_keys": ["ORD-1002"]}},
            tool_call_limit=1,
            double_tool_calls=True,
        ),
        thread_id="multi-1",
        routing_resolution=RouteDecision.direct(CancelOrders()),
    )
    events = await _events(engine, "cancel order ORD-1002")
    assert any(isinstance(e, InterruptEvent) for e in events)  # flow proceeded to readback
    # Structural: EVERY persisted tool_use has a tool_result (a dangling pair fails
    # provider-side history validation on every later model call in the session). The
    # fakes never validate history, so this must be asserted on the thread state itself.
    state = engine._graph.get_state(engine._config)
    tool_use_ids = {
        c["id"] for m in state.values["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    tool_result_ids = {
        m.tool_call_id for m in state.values["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_use_ids <= tool_result_ids
    await _events(engine, "yes")
    assert store.cancel_count == 1  # the first call was honored, exactly once
