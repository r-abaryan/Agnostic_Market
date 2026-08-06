"""ReasoningEngine over the REAL graph (fake models, InMemorySaver, zero network).

Covers the exit behaviors: interrupt/resume, §4a re-confirm, TTL expiry (Clock A),
kill-mid-placement (Clock B reap), idempotent placement, and the seam's zero-LiveKit claim.
Group B: the placement path is the cart flow (buy_now → guardrail → confirm → place_cart).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.types import PregelTask, StateSnapshot
from llm_fakes import ExplodingOnceFakeChatModel, FakeChatModel
from policy_helpers import make_policy
from pydantic import PrivateAttr, ValidationError
from turn_helpers import (
    TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS,
    engine_events,
    next_committed_turn,
)

from agnostic_market.agents.cart import flow as cart_flow
from agnostic_market.agents.engine import (
    ReasoningEngine,
    _classify_cancelled_checkpoint,
    _GraphSpans,
    _TurnSpeech,
    build_checkpointer,
)
from agnostic_market.agents.frontline import MODEL_SPEECH_NODES, build_frontline_graph
from agnostic_market.agents.recovery import AUTOMATION_TERMINAL_LINE, TURN_FALLBACK_LINE
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    load_payment_instruments_fixture,
)
from agnostic_market.commerce.profile import ProfileStore, load_profile_fixture
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.dtos.events import (
    CommittedTurn,
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnFacts,
)
from agnostic_market.dtos.orchestration import (
    ActiveInvocation,
    DiscloseAiIdentity,
    ListOrders,
    ViewIdentityStatus,
)
from agnostic_market.dtos.recovery import ExceptionAction, PendingRecovery
from agnostic_market.dtos.state import (
    CartClarification,
    ClarificationProgress,
    IdentityClarification,
    ReasoningState,
    SupportClarification,
    open_active_invocation,
)
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.graph import GraphVoiceAdapter
from agnostic_market.voice.tools import build_voice_tools

# The reasoning fake buys option 2 (waterproof rain jacket, $129.00) x2 = $258.00 -> straight
# to the placement tail via buy_now.
_PROPOSE = {"buy_now": {"candidate_key": "2", "quantity": 2}}
_FACTS = TurnFacts()
_TEST_OTP = "482913"
_WAIT_TIMEOUT_SECONDS = 5.0


class _BlockingFirstResponseModel(FakeChatModel):
    _started: threading.Event = PrivateAttr(default_factory=threading.Event)
    _release: threading.Event = PrivateAttr(default_factory=threading.Event)
    _blocked: bool = PrivateAttr(default=False)

    @property
    def started(self) -> threading.Event:
        return self._started

    def release(self) -> None:
        self._release.set()

    def _respond(self, messages, **kwargs):
        if not self._blocked:
            self._blocked = True
            self._started.set()
            if not self._release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release the blocked model")
        return super()._respond(messages, **kwargs)


class _ObservedTurnLock:
    def __init__(self, lock: asyncio.Lock, *, task_name: str, arrived: asyncio.Event) -> None:
        self._lock = lock
        self._task_name = task_name
        self._arrived = arrived

    async def __aenter__(self) -> _ObservedTurnLock:
        task = asyncio.current_task()
        if task is not None and task.get_name() == self._task_name:
            self._arrived.set()
        await self._lock.acquire()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._lock.release()


def _engine(
    config_root: Path,
    *,
    frontline: FakeChatModel | None = None,
    reasoning: FakeChatModel | None = None,
    identity: CallerIdentityStore | None = None,
    thread_id: str = "session-1",
) -> tuple[ReasoningEngine, OrderStore]:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    policy = make_policy(refund_returnless_under_usd=50.0)
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    cart = CartStore()
    identity = identity or CallerIdentityStore()
    customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
    tools = [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, cart, recent_orders, identity, customers)
    ]
    otp = OtpProvider(valid_code=_TEST_OTP)
    verification = VerificationStore(otp)
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        order_store=store,
    )
    assembly = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning
        or FakeChatModel(force_tool="buy_now", canned_args=_PROPOSE, tool_call_limit=1),
        store=store,
        otp=otp,
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        customers=customers,
        payment_instruments=PaymentInstrumentDirectory(
            load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=ProfileStore(load_profile_fixture(config_root, "acme_store")),
        policy=policy,
        lifecycle=caller_context,
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(
        assembly.graph,
        thread_id=thread_id,
        cancellation_quiescence_timeout_seconds=(TEST_CANCELLATION_QUIESCENCE_TIMEOUT_SECONDS),
        lifecycle=caller_context,
    )
    caller_context.attach_engine(engine)
    return engine, store


async def _events(engine: ReasoningEngine, text: str, facts: TurnFacts = _FACTS) -> list:
    return await engine_events(engine, text, facts)


async def _adapter_turn(
    adapter: GraphVoiceAdapter,
    text: str,
    message_id: str,
) -> list[str]:
    state = {"messages": [HumanMessage(content=text, id=message_id)]}
    return [chunk async for chunk in adapter.astream(state)]


async def _pause_at_confirmation(engine: ReasoningEngine) -> list:
    """Drive the graph to the readback interrupt via the gate's checkout trigger."""
    return await _events(engine, "checkout now please")


def _block_placement(
    store: OrderStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    after_commit: bool,
) -> tuple[threading.Event, threading.Event, threading.Event]:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    real_place = store.place_cart

    def blocked_place(idempotency_key, *, lines, total_usd):
        try:
            if after_commit:
                result = real_place(idempotency_key, lines=lines, total_usd=total_usd)
                entered.set()
                if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                    raise TimeoutError("test did not release post-commit placement")
                return result
            entered.set()
            if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
                raise TimeoutError("test did not release pre-commit placement")
            return real_place(idempotency_key, lines=lines, total_usd=total_usd)
        finally:
            finished.set()

    monkeypatch.setattr(store, "place_cart", blocked_place)
    return entered, release, finished


# --- the turn-failure boundary (live call #13 F-13.1: a 529 died in SILENCE) -------------


async def test_failed_turn_recovers_with_the_fallback_never_silence(
    config_root: Path,
) -> None:
    import json

    from agnostic_market.agents import telemetry
    from agnostic_market.dtos.events import SpokenMessageEvent

    engine, _ = _engine(
        config_root,
        frontline=ExplodingOnceFakeChatModel(emit_tool_calls=False),
    )
    events = await _events(engine, "hi there")  # the graph dies mid-turn...
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert len(spoken) == 1 and spoken[0].node == "recover_node_exception"
    assert "say that again" in spoken[0].text  # ...but the caller hears the fallback
    lines = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any(rec.get("event") == "turn_failed" for rec in lines)  # loud, not swallowed
    # The session SURVIVES: the next turn runs normally on the same thread.
    retry = await _events(engine, "hi again")
    assert any(isinstance(e, TokenEvent) and e.text for e in retry)


async def test_failed_cart_turn_admits_changed_intent_instead_of_resuming_old_work(
    config_root: Path,
) -> None:
    engine, store = _engine(
        config_root,
        reasoning=ExplodingOnceFakeChatModel(emit_tool_calls=False),
        thread_id="failure-admission",
    )

    failed = await _events(engine, "checkout now please")
    assert [event.node for event in failed if isinstance(event, SpokenMessageEvent)] == [
        "recover_node_exception"
    ]
    assert engine._graph.get_state(engine._config).next == ()

    await _events(engine, "never mind")
    snapshot = engine._graph.get_state(engine._config)
    admitted = tuple(
        str(message.content)
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    )

    assert admitted == ("checkout now please", "never mind")
    assert store.placed_count == 0
    assert snapshot.values.get("active_flow") is None
    assert snapshot.next == ()


# --- plain turns -----------------------------------------------------------------------


async def test_plain_answer_is_spoken_exactly_once(config_root: Path) -> None:
    # A non-streaming model's answer arrives as ONE full message; the engine must speak it
    # once (fallback) and never twice (the 3a double-speak class).
    engine, _ = _engine(config_root)
    events = await _events(engine, "hi there")
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert len(tokens) == 1
    assert tokens[0].text  # the fake's canned answer


async def test_duplicate_normal_turn_id_is_admitted_only_once(config_root: Path) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    engine, _ = _engine(
        config_root,
        frontline=frontline,
        thread_id="duplicate-normal-turn",
    )
    adapter = GraphVoiceAdapter(engine)

    await _adapter_turn(adapter, "tell me about your shoes", "normal-turn-1")
    await _adapter_turn(adapter, "tell me about your shoes", "normal-turn-1")

    assert frontline.invoke_count == 1
    assert engine._graph.get_state(engine._config).values.get("active_invocation") is None


async def test_duplicate_list_turn_cannot_open_an_invocation(config_root: Path) -> None:
    frontline = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(raise_transport=True)
    engine, _ = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="duplicate-list-turn",
    )
    duplicate_id = "already-consumed-list-turn"
    engine._graph.update_state(
        engine._config,
        {"consumed_turn_ids": (duplicate_id,)},
        as_node="__start__",
    )

    output = await _adapter_turn(
        GraphVoiceAdapter(engine),
        "list my account orders",
        duplicate_id,
    )
    state = ReasoningState.model_validate(engine._graph.get_state(engine._config).values)

    assert output == []
    assert state.active_invocation is None
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0


async def test_duplicate_abandoned_turn_advances_safe_recovery_with_one_retry(
    config_root: Path,
    tmp_path: Path,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    engine, _ = _engine(
        config_root,
        frontline=frontline,
        thread_id="duplicate-abandoned-turn",
    )
    adapter = GraphVoiceAdapter(engine)
    abandoned_id = "abandoned-safe-turn"
    engine._graph.update_state(
        engine._config,
        {
            "messages": [HumanMessage(content="list my account orders", id=abandoned_id)],
            "consumed_turn_ids": (abandoned_id,),
            "pending_recovery": PendingRecovery(
                origin_node="model",
                action=ExceptionAction.SAFE_ABORT,
                trigger="stream_cancelled",
                abandoned_message_id=abandoned_id,
            ),
        },
        as_node="__start__",
    )

    output = await _adapter_turn(adapter, "list my account orders", abandoned_id)
    snapshot = engine._graph.get_state(engine._config)
    human_ids = [
        message.id
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    ]

    assert output == [TURN_FALLBACK_LINE]
    assert human_ids == [abandoned_id]
    assert snapshot.values["consumed_turn_ids"] == [abandoned_id]
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values.get("active_invocation") is None
    assert snapshot.next == ()
    assert frontline.invoke_count == 0
    import json

    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [
        record
        for record in records
        if record.get("event") == "turn_failed" and record.get("reason") == "stream_cancelled"
    ] == [
        {
            "event": "turn_failed",
            "reason": "stream_cancelled",
            "node": "model",
            "action": "safe_abort",
        }
    ]


def test_committed_turn_requires_an_explicit_nonblank_transport_identity() -> None:
    assert CommittedTurn.model_fields["message_id"].is_required()
    assert CommittedTurn(text="hello", message_id=None).message_id is None
    with pytest.raises(ValidationError, match="message_id"):
        CommittedTurn(text="hello", message_id=" ")
    with pytest.raises(ValidationError, match="text"):
        CommittedTurn(text=" ", message_id="turn-1")


async def test_missing_transport_id_rejects_before_checkpoint_or_model_work(
    config_root: Path,
) -> None:
    import json

    from agnostic_market.agents import telemetry

    frontline = FakeChatModel(emit_tool_calls=False)
    engine, store = _engine(
        config_root,
        frontline=frontline,
        thread_id="missing-message-id",
    )
    adapter = GraphVoiceAdapter(engine)

    output = [
        text
        async for text in adapter.astream(
            {"messages": [HumanMessage(content="place an order", id=None)]}
        )
    ]

    assert output == [TURN_FALLBACK_LINE]
    assert frontline.invoke_count == 0
    assert store.placed_count == 0
    assert engine._graph.get_state(engine._config).values == {}
    records = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [{"event": "ingress_turn_rejected", "reason": "missing_message_id"}]


# --- the interrupt path (V1/V2/V3) ------------------------------------------------------


async def test_checkout_pauses_with_graph_authored_readback(config_root: Path) -> None:
    engine, store = _engine(config_root)
    events = await _pause_at_confirmation(engine)
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    # The readback is GRAPH-authored, SPEECH-native (§7: "2 waterproof rain jackets", not
    # "2 x ..." which TTS voices "ex"), and covers the declared confirm fields (§7a):
    # quantity + total — total computed by CODE from the fixture price (2 x $129.00).
    assert "2 waterproof rain jackets" in interrupts[0].prompt
    assert "$258.00" in interrupts[0].prompt
    assert engine.pending_interrupt()
    assert store.placed_count == 0  # nothing placed before consent


async def test_resume_yes_places_exactly_once(config_root: Path) -> None:
    reasoning = FakeChatModel(force_tool="buy_now", canned_args=_PROPOSE, tool_call_limit=1)
    engine, store = _engine(config_root, reasoning=reasoning)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "yes please")
    assert store.placed_count == 1
    assert not engine.pending_interrupt()
    # The success line is node-authored by the place node and spoken.
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("ORD-9001" in e.text and e.node == "cart_place" for e in spoken)
    # No NEW interrupt fires on the resume (the readback is not re-spoken - V2).
    assert not any(isinstance(e, InterruptEvent) for e in events)
    # A10a/V4: assemble did NOT re-run on resume (its model was invoked exactly once).
    assert reasoning._tool_calls_made == 1


async def test_duplicate_consent_turn_id_cannot_be_reused_as_a_fresh_turn(
    config_root: Path,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    reasoning = FakeChatModel(force_tool="buy_now", canned_args=_PROPOSE, tool_call_limit=1)
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="duplicate-consent-turn",
    )
    adapter = GraphVoiceAdapter(engine)

    await _adapter_turn(adapter, "checkout now please", "checkout-turn-1")
    await _adapter_turn(adapter, "yes", "consent-turn-1")
    await _adapter_turn(adapter, "yes", "consent-turn-1")

    assert store.placed_count == 1
    assert frontline.invoke_count == 0


async def test_placed_order_is_queryable_same_session(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    await _events(engine, "yes")
    assert "rain jacket" in (store.order_summary("ORD-9001") or "")


async def test_resume_no_cancels_without_placing(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "no, don't do it")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("won't place it" in e.text.lower() for e in spoken)


# --- §4a + re-confirm-once (the user-decided consent UX) --------------------------------


async def test_barged_readback_makes_yes_invalid_and_reconfirms(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    # Caller said "yes" but LiveKit marked the readback barged-over: NOT consent (§4a).
    events = await _events(engine, "yes", TurnFacts(readback_interrupted=True))
    assert store.placed_count == 0
    reconfirms = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(reconfirms) == 1
    assert "yes or no" in reconfirms[0].prompt.lower()
    # A clean committed yes on the re-confirm places it.
    await _events(engine, "yes")
    assert store.placed_count == 1


async def test_unclear_answer_reconfirms_once_then_cancels(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    first = await _events(engine, "wait, do you have it in blue?")
    assert any(isinstance(e, InterruptEvent) for e in first)  # ONE re-confirm
    second = await _events(engine, "hmm what about the weather")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # cancelled, not trapped
    spoken = [e for e in second if isinstance(e, SpokenMessageEvent)]
    assert any("won't place it" in e.text.lower() for e in spoken)


async def test_human_request_at_confirmation_escapes(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    events = await _events(engine, "just get me a real person please")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any(
        e.node == "automation_terminal_response" and "contact the store" in e.text.lower()
        for e in spoken
    )


# --- Clock A: pending-confirmation TTL --------------------------------------------------


async def test_expired_pending_cancels_before_speaking(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    # TTL is policy-driven (default 120s); jump the flow's clock past it so the resume finds
    # the pending expired. Capture real now FIRST (checkout_flow.time is the global module —
    # a self-referential lambda would recurse). Clear-before-speak: a stale yes must NOT place.
    future = time.time() + 10_000
    monkeypatch.setattr(cart_flow.time, "time", lambda: future)
    events = await _events(engine, "yes")
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # cleared (clear-before-speak)
    spoken = [e for e in events if isinstance(e, SpokenMessageEvent)]
    assert any("haven't placed anything" in e.text for e in spoken)


# --- Clock B: kill mid-checkout (the BUILD_PLAN-named exit test) -------------------------


async def test_kill_mid_checkout_leaves_no_ghost_order(config_root: Path) -> None:
    engine, store = _engine(config_root)
    await _pause_at_confirmation(engine)
    assert engine.pending_interrupt()
    # The call drops: Clock-B teardown reaps the thread unconditionally.
    engine.delete_thread()
    assert store.placed_count == 0
    assert not engine.pending_interrupt()  # nothing resumable survives the drop
    # Reap is idempotent at the checkpointer (verified) - a double-fired hook is safe.
    engine.delete_thread()
    # A fresh session (new thread) starts clean: no ghost order, no stale pending.
    engine2, store2 = _engine(config_root, thread_id="session-2")
    events = await _events(engine2, "hi there")
    assert store2.placed_count == 0
    assert not engine2.pending_interrupt()
    assert any(isinstance(e, TokenEvent) for e in events)


# --- Milestone 6D-a: executable product-gap contracts -----------------------------------


async def test_distinct_turn_arriving_before_cancellation_is_admitted_once(
    config_root: Path,
) -> None:
    frontline = _BlockingFirstResponseModel(emit_tool_calls=False)
    engine, store = _engine(
        config_root,
        frontline=frontline,
        thread_id="late-distinct-turn",
    )
    adapter = GraphVoiceAdapter(engine)
    second_arrived = asyncio.Event()
    second_task_name = "late-distinct-turn-waiter"
    engine._turn_lock = _ObservedTurnLock(
        engine._turn_lock,
        task_name=second_task_name,
        arrived=second_arrived,
    )

    first = asyncio.create_task(
        _adapter_turn(adapter, "tell me about your shoes", "late-turn-1"),
        name="late-distinct-turn-owner",
    )
    assert await asyncio.to_thread(frontline.started.wait, _WAIT_TIMEOUT_SECONDS)
    second = asyncio.create_task(
        _adapter_turn(adapter, "never mind", "late-turn-2"),
        name=second_task_name,
    )
    await asyncio.wait_for(second_arrived.wait(), timeout=_WAIT_TIMEOUT_SECONDS)

    first.cancel()
    frontline.release()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, timeout=_WAIT_TIMEOUT_SECONDS)

    snapshot = engine._graph.get_state(engine._config)
    human_texts = [
        str(message.content)
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    ]
    assert human_texts.count("never mind") == 1
    assert store.placed_count == 0
    assert not engine.pending_interrupt()


async def test_distinct_turn_after_safe_abort_opens_on_the_new_ledger_tail(
    config_root: Path,
) -> None:
    frontline = _BlockingFirstResponseModel(emit_tool_calls=False)
    reasoning = FakeChatModel(emit_tool_calls=False)
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="late-list-after-safe-abort",
    )
    adapter = GraphVoiceAdapter(engine)
    second_arrived = asyncio.Event()
    second_task_name = "late-list-waiter"
    engine._turn_lock = _ObservedTurnLock(
        engine._turn_lock,
        task_name=second_task_name,
        arrived=second_arrived,
    )

    first = asyncio.create_task(
        _adapter_turn(adapter, "tell me about your shoes", "late-list-turn-1"),
        name="late-list-owner",
    )
    assert await asyncio.to_thread(frontline.started.wait, _WAIT_TIMEOUT_SECONDS)
    second = asyncio.create_task(
        _adapter_turn(adapter, "what orders do I have", "late-list-turn-2"),
        name=second_task_name,
    )
    await asyncio.wait_for(second_arrived.wait(), timeout=_WAIT_TIMEOUT_SECONDS)

    first.cancel()
    frontline.release()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, timeout=_WAIT_TIMEOUT_SECONDS)

    state = ReasoningState.model_validate(engine._graph.get_state(engine._config).values)
    invocation = state.active_invocation
    assert invocation is not None
    assert invocation.request == ListOrders(scope="account")
    assert invocation.opened_turn_id == state.consumed_turn_ids[-1] == "late-list-turn-2"
    assert store.placed_count == 0
    assert store.cancel_count == 0
    assert store.refund_count == 0
    assert store.return_count == 0


async def test_cancelled_cart_assemble_reviews_cart_and_consumes_queued_turn(
    config_root: Path,
) -> None:
    reasoning = _BlockingFirstResponseModel(
        force_tool="buy_now",
        canned_args=_PROPOSE,
        tool_call_limit=1,
    )
    engine, store = _engine(
        config_root,
        reasoning=reasoning,
        thread_id="cancelled-cart-review",
    )
    adapter = GraphVoiceAdapter(engine)
    second_arrived = asyncio.Event()
    second_task_name = "cancelled-cart-review-waiter"
    engine._turn_lock = _ObservedTurnLock(
        engine._turn_lock,
        task_name=second_task_name,
        arrived=second_arrived,
    )

    first = asyncio.create_task(
        _adapter_turn(adapter, "checkout now please", "cart-review-turn-1"),
        name="cancelled-cart-review-owner",
    )
    assert await asyncio.to_thread(reasoning.started.wait, _WAIT_TIMEOUT_SECONDS)
    second = asyncio.create_task(
        _adapter_turn(adapter, "never mind", "cart-review-turn-2"),
        name=second_task_name,
    )
    await asyncio.wait_for(second_arrived.wait(), timeout=_WAIT_TIMEOUT_SECONDS)

    first.cancel()
    reasoning.release()
    with pytest.raises(asyncio.CancelledError):
        await first
    output = await asyncio.wait_for(second, timeout=_WAIT_TIMEOUT_SECONDS)

    snapshot = engine._graph.get_state(engine._config)
    human_ids = [
        message.id
        for message in snapshot.values.get("messages", ())
        if isinstance(message, HumanMessage)
    ]
    assert "cart-review-turn-2" not in human_ids
    assert snapshot.values["consumed_turn_ids"] == ["cart-review-turn-1", "cart-review-turn-2"]
    assert snapshot.values.get("pending_recovery") is None
    assert engine._lifecycle is not None
    assert engine._lifecycle.has_discardable_state()
    assert store.placed_count == 0
    assert len(output) == 1 and "review your cart" in output[0].lower()


async def test_turn_arriving_before_a_new_interrupt_cannot_resume_it(
    config_root: Path,
) -> None:
    reasoning = _BlockingFirstResponseModel(
        force_tool="buy_now",
        canned_args=_PROPOSE,
        tool_call_limit=1,
    )
    engine, store = _engine(
        config_root,
        reasoning=reasoning,
        thread_id="new-interrupt-admission",
    )
    adapter = GraphVoiceAdapter(engine)
    second_arrived = asyncio.Event()
    second_task_name = "new-interrupt-waiter"
    engine._turn_lock = _ObservedTurnLock(
        engine._turn_lock,
        task_name=second_task_name,
        arrived=second_arrived,
    )

    first = asyncio.create_task(
        _adapter_turn(adapter, "checkout now please", "interrupt-turn-1"),
        name="new-interrupt-owner",
    )
    assert await asyncio.to_thread(reasoning.started.wait, _WAIT_TIMEOUT_SECONDS)
    second = asyncio.create_task(
        _adapter_turn(adapter, "yes", "interrupt-turn-2"),
        name=second_task_name,
    )
    await asyncio.wait_for(second_arrived.wait(), timeout=_WAIT_TIMEOUT_SECONDS)

    reasoning.release()
    await asyncio.wait_for(first, timeout=_WAIT_TIMEOUT_SECONDS)
    await asyncio.wait_for(second, timeout=_WAIT_TIMEOUT_SECONDS)

    assert store.placed_count == 0
    assert engine.pending_interrupt()


@pytest.mark.parametrize("after_commit", (False, True), ids=("before-commit", "after-commit"))
async def test_cancelled_sync_effect_seeds_recovery_only_after_worker_exit(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_commit: bool,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id=f"sync-effect-cancel-{after_commit}",
    )
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", "effect-checkout-turn")
    entered, release, finished = _block_placement(
        store,
        monkeypatch,
        after_commit=after_commit,
    )

    consent = asyncio.create_task(
        _adapter_turn(adapter, "yes", "effect-consent-turn"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)
    consent.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await consent
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)

    snapshot = engine._graph.get_state(engine._config)
    marker = snapshot.values.get("pending_recovery")
    assert isinstance(marker, PendingRecovery)
    assert marker.trigger == "stream_cancelled"
    assert store.placed_count == 1


async def test_second_cancellation_waits_for_takeover_then_repropagates_the_original(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _store = _engine(
        config_root,
        thread_id="repeated-cancellation",
    )
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", "repeat-cancel-checkout")
    entered, release, finished = _block_placement(
        _store,
        monkeypatch,
        after_commit=False,
    )
    wait_started = threading.Event()
    real_wait = engine._node_execution_tracker.wait_until_mutable_idle

    def observed_wait(timeout_seconds: float) -> bool:
        wait_started.set()
        return real_wait(timeout_seconds)

    monkeypatch.setattr(engine._node_execution_tracker, "wait_until_mutable_idle", observed_wait)
    consent = asyncio.create_task(
        _adapter_turn(adapter, "yes", "repeat-cancel-consent"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    consent.cancel("original-cancellation")
    assert await asyncio.to_thread(wait_started.wait, _WAIT_TIMEOUT_SECONDS)
    consent.cancel("later-cancellation")
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await consent
    assert cancelled.value.args == ("original-cancellation",)
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)

    snapshot = engine._graph.get_state(engine._config)
    marker = snapshot.values.get("pending_recovery")
    assert isinstance(marker, PendingRecovery)
    assert marker.trigger == "stream_cancelled"
    assert marker.abandoned_message_id == "repeat-cancel-consent"


async def test_cancelled_confirmation_resume_aborts_without_replaying_consent(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id="cancelled-confirmation-resume",
    )
    await _pause_at_confirmation(engine)
    entered = threading.Event()
    release = threading.Event()
    classify_calls = 0
    real_classify = cart_flow.classify_consent

    def classify_then_pause(text: str):
        nonlocal classify_calls
        classify_calls += 1
        verdict = real_classify(text)
        entered.set()
        if not release.wait(timeout=_WAIT_TIMEOUT_SECONDS):
            raise RuntimeError("test did not release consent classification")
        return verdict

    monkeypatch.setattr(cart_flow, "classify_consent", classify_then_pause)
    confirming = asyncio.create_task(_events(engine, "yes"))
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    confirming.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await confirming
    marker = engine._graph.get_state(engine._config).values.get("pending_recovery")
    assert isinstance(marker, PendingRecovery)
    assert marker.origin_node == "cart_confirm"
    assert marker.action == ExceptionAction.ABORT_PLACEMENT_CONFIRMATION

    recovered = await _events(engine, "continue")
    snapshot = engine._graph.get_state(engine._config)

    assert [event.text for event in recovered if isinstance(event, SpokenMessageEvent)] == [
        "That order placement request did not complete. Your cart is still saved for review."
    ]
    assert classify_calls == 1
    assert store.placed_count == 0
    assert engine._lifecycle is not None
    assert not engine._lifecycle.cart_store.is_empty()
    assert snapshot.values.get("pending_placement") is None
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.next == ()


async def test_quiescence_timeout_terminalizes_without_an_optimistic_recovery(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id="quiescence-timeout",
    )
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", "timeout-checkout")
    entered, release, finished = _block_placement(
        store,
        monkeypatch,
        after_commit=False,
    )
    monkeypatch.setattr(
        engine._node_execution_tracker,
        "wait_until_mutable_idle",
        lambda _timeout_seconds: False,
    )
    consent = asyncio.create_task(
        _adapter_turn(adapter, "yes", "timeout-consent"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    consent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consent
    unstable = engine._graph.get_state(engine._config)
    assert engine._terminal_latched is True
    assert unstable.values.get("pending_recovery") is None
    assert unstable.values.get("automation_terminal", False) is False
    assert unstable.next == ("cart_place",)

    cleanup_complete = threading.Event()
    engine._node_execution_tracker.defer_until_fully_idle(cleanup_complete.set)
    release.set()
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)
    assert await asyncio.to_thread(cleanup_complete.wait, _WAIT_TIMEOUT_SECONDS)
    snapshot = engine._graph.get_state(engine._config)
    assert store.placed_count == 1
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    repeated = await _adapter_turn(adapter, "try again", "timeout-retry")
    assert repeated == [AUTOMATION_TERMINAL_LINE]


@pytest.mark.parametrize("failure_point", ("seed", "readback"))
async def test_recovery_seed_or_readback_failure_terminalizes_and_preserves_cancellation(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id=f"recovery-{failure_point}-failure",
    )
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", f"{failure_point}-checkout")
    entered, release, finished = _block_placement(
        store,
        monkeypatch,
        after_commit=True,
    )
    real_update = engine._graph.update_state
    real_get = engine._graph.get_state
    fail_readback = False

    def intercept_update(config, values, *, as_node=None):
        nonlocal fail_readback
        if isinstance(values, dict) and values.get("pending_recovery") is not None:
            if failure_point == "seed":
                raise RuntimeError("injected recovery seed failure")
            result = real_update(config, values, as_node=as_node)
            fail_readback = True
            return result
        return real_update(config, values, as_node=as_node)

    def intercept_get(config):
        nonlocal fail_readback
        if fail_readback:
            fail_readback = False
            raise RuntimeError("injected recovery readback failure")
        return real_get(config)

    monkeypatch.setattr(engine._graph, "update_state", intercept_update)
    monkeypatch.setattr(engine._graph, "get_state", intercept_get)
    placing = asyncio.create_task(
        _adapter_turn(adapter, "yes", f"{failure_point}-consent"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    placing.cancel("transport-cancelled")
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await placing
    assert cancelled.value.args == ("transport-cancelled",)
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)
    snapshot = engine._graph.get_state(engine._config)

    assert store.placed_count == 1
    assert snapshot.values.get("pending_recovery") is None
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    repeated = await _adapter_turn(
        adapter,
        "try again",
        f"{failure_point}-retry",
    )
    assert repeated == [AUTOMATION_TERMINAL_LINE]


async def test_quiescence_task_failure_latches_terminal_without_unstable_checkpoint_work(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id="quiescence-task-failure",
    )
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", "wait-failure-checkout")
    entered, release, finished = _block_placement(
        store,
        monkeypatch,
        after_commit=False,
    )

    def fail_wait(_timeout_seconds: float) -> bool:
        raise RuntimeError("injected quiescence wait failure")

    monkeypatch.setattr(engine._node_execution_tracker, "wait_until_mutable_idle", fail_wait)
    placing = asyncio.create_task(
        _adapter_turn(adapter, "yes", "wait-failure-consent"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    placing.cancel("transport-cancelled")
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await placing
    assert cancelled.value.args == ("transport-cancelled",)
    assert engine._terminal_latched is True
    unstable = engine._graph.get_state(engine._config)
    assert unstable.values.get("pending_recovery") is None

    cleanup_complete = threading.Event()
    engine._node_execution_tracker.defer_until_fully_idle(cleanup_complete.set)
    release.set()
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)
    assert await asyncio.to_thread(cleanup_complete.wait, _WAIT_TIMEOUT_SECONDS)
    assert store.placed_count == 1
    snapshot = engine._graph.get_state(engine._config)
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    repeated = await _adapter_turn(adapter, "try again", "wait-failure-retry")
    assert repeated == [AUTOMATION_TERMINAL_LINE]


def test_repeated_last_resort_finalization_keeps_one_checkpoint_message(
    config_root: Path,
) -> None:
    engine, _ = _engine(config_root, thread_id="idempotent-last-resort")

    engine._enter_last_resort()
    engine._enter_last_resort()
    snapshot = engine._graph.get_state(engine._config)
    terminal_messages = [
        message
        for message in snapshot.values.get("messages", ())
        if isinstance(message, AIMessage) and message.content == AUTOMATION_TERMINAL_LINE
    ]

    assert len(terminal_messages) == 1
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()


def test_cancelled_checkpoint_classifier_rejects_every_ambiguous_shape(
    config_root: Path,
) -> None:
    engine, _ = _engine(
        config_root,
        thread_id="cancellation-classifier",
    )
    abandoned_id = "classifier-abandoned"
    state = ReasoningState(consumed_turn_ids=(abandoned_id,))

    def task(name: str, *, error: Exception | None = None) -> PregelTask:
        return PregelTask(
            id=f"task-{name}",
            name=name,
            path=(),
            error=error,
            interrupts=(),
            state=None,
            result=None,
        )

    def snapshot(
        next_nodes: tuple[str, ...],
        tasks: tuple[PregelTask, ...],
    ) -> StateSnapshot:
        return StateSnapshot(
            values=state,
            next=next_nodes,
            config=engine._config,
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=tasks,
            interrupts=(),
        )

    valid = _classify_cancelled_checkpoint(
        snapshot(("cart_place",), (task("cart_place"),)),
        policies=engine._node_recovery_policies,
        infrastructure_nodes=engine._recovery_infrastructure_nodes,
        active_node_names=frozenset({"cart_place"}),
        abandoned_message_id=abandoned_id,
        resumed_interrupt_node=None,
    )
    assert valid.outcome == "recover"
    assert valid.marker is not None
    assert valid.marker.action == ExceptionAction.RECONCILE_PLACEMENT

    invalid_cases = (
        snapshot(("cart_place",), ()),
        snapshot(("cart_place",), (task("model"),)),
        snapshot(("cart_place",), (task("cart_place", error=RuntimeError("failed")),)),
        snapshot(("cart_place", "model"), (task("cart_place"), task("model"))),
        snapshot(("recover_node_exception",), (task("recover_node_exception"),)),
    )
    for ambiguous in invalid_cases:
        result = _classify_cancelled_checkpoint(
            ambiguous,
            policies=engine._node_recovery_policies,
            infrastructure_nodes=engine._recovery_infrastructure_nodes,
            active_node_names=frozenset(),
            abandoned_message_id=abandoned_id,
            resumed_interrupt_node=None,
        )
        assert result.outcome == "invalid"

    tracker_mismatch = _classify_cancelled_checkpoint(
        snapshot(("cart_place",), (task("cart_place"),)),
        policies=engine._node_recovery_policies,
        infrastructure_nodes=engine._recovery_infrastructure_nodes,
        active_node_names=frozenset({"support_place"}),
        abandoned_message_id=abandoned_id,
        resumed_interrupt_node=None,
    )
    assert tracker_mismatch.outcome == "invalid"


async def test_untyped_bare_next_terminalizes_without_replaying_graph_work(
    config_root: Path,
) -> None:
    frontline = FakeChatModel()
    reasoning = FakeChatModel()
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="untyped-bare-next",
    )
    engine._graph.update_state(
        engine._config,
        {"active_flow": "cart"},
        as_node="__start__",
    )
    before = engine._graph.get_state(engine._config)
    assert before.next == ("entry",)

    events = await _events(engine, "continue")
    after = engine._graph.get_state(engine._config)

    assert [event.text for event in events if isinstance(event, SpokenMessageEvent)] == [
        AUTOMATION_TERMINAL_LINE
    ]
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0
    assert store.placed_count == 0
    assert after.values["automation_terminal"] is True
    assert after.next == ()


async def test_close_session_defers_teardown_until_the_mutable_worker_exits(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store = _engine(
        config_root,
        thread_id="deferred-close",
    )
    context = engine._lifecycle
    assert isinstance(context, CallerContext)
    adapter = GraphVoiceAdapter(engine)
    await _adapter_turn(adapter, "checkout now please", "close-checkout-turn")
    entered, release, finished = _block_placement(
        store,
        monkeypatch,
        after_commit=False,
    )
    cart_cleared = threading.Event()
    thread_deleted = threading.Event()
    real_clear = context.cart_store.clear
    real_delete_thread = engine.delete_thread

    def observed_clear() -> None:
        real_clear()
        cart_cleared.set()

    def observed_delete_thread() -> None:
        real_delete_thread()
        thread_deleted.set()

    monkeypatch.setattr(context.cart_store, "clear", observed_clear)
    monkeypatch.setattr(engine, "delete_thread", observed_delete_thread)
    consent = asyncio.create_task(
        _adapter_turn(adapter, "yes", "close-consent-turn"),
    )
    assert await asyncio.to_thread(entered.wait, _WAIT_TIMEOUT_SECONDS)

    consent.cancel()
    context.close_session()
    cleared_before_worker_exit = cart_cleared.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await consent
    assert await asyncio.to_thread(finished.wait, _WAIT_TIMEOUT_SECONDS)
    assert await asyncio.to_thread(cart_cleared.wait, _WAIT_TIMEOUT_SECONDS)
    assert await asyncio.to_thread(thread_deleted.wait, _WAIT_TIMEOUT_SECONDS)

    assert not cleared_before_worker_exit
    assert context.cart_store.is_empty()
    assert engine._graph.get_state(engine._config).values == {}


async def test_close_during_pure_abort_model_waits_then_deletes_permanently(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontline = _BlockingFirstResponseModel(emit_tool_calls=False)
    engine, _store = _engine(
        config_root,
        frontline=frontline,
        thread_id="close-during-pure-abort",
    )
    context = engine._lifecycle
    assert isinstance(context, CallerContext)
    adapter = GraphVoiceAdapter(engine)
    delete_calls = 0
    real_delete_thread = engine.delete_thread

    def observed_delete_thread() -> None:
        nonlocal delete_calls
        delete_calls += 1
        real_delete_thread()

    monkeypatch.setattr(engine, "delete_thread", observed_delete_thread)
    running = asyncio.create_task(
        _adapter_turn(adapter, "tell me about shoes", "pure-abort-close-turn")
    )
    assert await asyncio.to_thread(frontline.started.wait, _WAIT_TIMEOUT_SECONDS)

    context.close_session()
    assert delete_calls == 0
    assert engine._graph.get_state(engine._config).values

    frontline.release()
    await asyncio.wait_for(running, timeout=_WAIT_TIMEOUT_SECONDS)

    assert frontline.invoke_count == 1
    assert delete_calls == 1
    assert engine._graph.get_state(engine._config).values == {}


async def test_queued_turn_is_rejected_once_when_close_starts(
    config_root: Path,
    tmp_path: Path,
) -> None:
    import json

    frontline = _BlockingFirstResponseModel(emit_tool_calls=False)
    reasoning = FakeChatModel()
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="queued-turn-close",
    )
    context = engine._lifecycle
    assert isinstance(context, CallerContext)
    adapter = GraphVoiceAdapter(engine)
    second_arrived = asyncio.Event()
    second_task_name = "queued-turn-close-waiter"
    engine._turn_lock = _ObservedTurnLock(
        engine._turn_lock,
        task_name=second_task_name,
        arrived=second_arrived,
    )

    first = asyncio.create_task(
        _adapter_turn(adapter, "tell me about shoes", "queued-close-turn-1"),
        name="queued-turn-close-owner",
    )
    assert await asyncio.to_thread(frontline.started.wait, _WAIT_TIMEOUT_SECONDS)
    second = asyncio.create_task(
        _adapter_turn(adapter, "place an order", "queued-close-turn-2"),
        name=second_task_name,
    )
    await asyncio.wait_for(second_arrived.wait(), timeout=_WAIT_TIMEOUT_SECONDS)

    context.close_session()
    frontline.release()
    await asyncio.wait_for(first, timeout=_WAIT_TIMEOUT_SECONDS)
    second_output = await asyncio.wait_for(second, timeout=_WAIT_TIMEOUT_SECONDS)

    assert second_output == []
    assert frontline.invoke_count == 1
    assert reasoning.invoke_count == 0
    assert store.placed_count == 0
    assert engine._graph.get_state(engine._config).values == {}
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record for record in records if record.get("event") == "ingress_turn_rejected"] == [
        {"event": "ingress_turn_rejected", "reason": "session_closed"}
    ]
    closed_events = [record for record in records if record.get("event") == "caller_context_closed"]
    assert len(closed_events) == 1


@pytest.mark.parametrize("terminal_latched", (False, True))
async def test_post_close_turn_stops_before_every_engine_boundary(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_latched: bool,
) -> None:
    import json

    frontline = FakeChatModel(emit_tool_calls=False)
    reasoning = FakeChatModel()
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id=f"post-close-boundary-{terminal_latched}",
    )
    if terminal_latched:
        engine._terminal_latched = True
    context = engine._lifecycle
    assert isinstance(context, CallerContext)
    context.close_session()
    real_get_state = engine._graph.get_state
    checkpoint_reads = 0
    checkpoint_writes = 0

    def observed_get_state(config):
        nonlocal checkpoint_reads
        checkpoint_reads += 1
        return real_get_state(config)

    def fail_update_state(*_args, **_kwargs):
        nonlocal checkpoint_writes
        checkpoint_writes += 1
        pytest.fail("post-close turn attempted a checkpoint write")

    monkeypatch.setattr(engine._graph, "get_state", observed_get_state)
    monkeypatch.setattr(engine._graph, "update_state", fail_update_state)
    monkeypatch.setattr(
        engine,
        "_enter_last_resort",
        lambda: pytest.fail("post-close turn entered last-resort state handling"),
    )

    output = await _adapter_turn(
        GraphVoiceAdapter(engine),
        "late transcript",
        f"post-close-turn-{terminal_latched}",
    )

    assert output == []
    assert checkpoint_reads == 0
    assert checkpoint_writes == 0
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0
    assert store.placed_count == 0
    assert real_get_state(engine._config).values == {}
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record for record in records if record.get("event") == "ingress_turn_rejected"] == [
        {"event": "ingress_turn_rejected", "reason": "session_closed"}
    ]


# --- checkpoint serde: our DTOs are registered (no 'unregistered type' warning) ----------


async def test_checkpointed_dtos_deserialize_without_unregistered_warning(
    config_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # build_checkpointer's serde allowlists PendingPlacement/CartLine/PendingRefund/
    # HandoffRequest/ReasoningState — a checkpointed interrupt/resume (which deserializes
    # PendingPlacement + nested CartLine + HandoffRequest live) must NOT emit langgraph's
    # "Deserializing unregistered type" warning, which is slated to become a hard block.
    engine, _ = _engine(config_root)
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        await _pause_at_confirmation(engine)  # writes + reads PendingAction across the interrupt
        await _events(engine, "yes")  # resume re-reads the checkpoint
    assert not any("unregistered" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.parametrize(
    "intent_request",
    (ViewIdentityStatus(), DiscloseAiIdentity()),
)
def test_new_intent_requests_roundtrip_through_production_checkpoint_serde(
    config_root: Path,
    caplog: pytest.LogCaptureFixture,
    intent_request: ViewIdentityStatus | DiscloseAiIdentity,
) -> None:
    consumed_turn_ids = ("serde-turn",)
    invocation = open_active_invocation(intent_request, consumed_turn_ids=consumed_turn_ids)
    engine, _ = _engine(
        config_root,
        thread_id=f"serde-{intent_request.kind.value}",
    )
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        engine._graph.update_state(
            engine._config,
            {
                "consumed_turn_ids": consumed_turn_ids,
                "active_invocation": invocation,
            },
            as_node="__start__",
        )
        restored = ReasoningState.model_validate(engine._graph.get_state(engine._config).values)

    assert restored.active_invocation == invocation
    assert not any("unregistered" in record.getMessage().lower() for record in caplog.records)


async def test_incoherent_persisted_invocation_terminalizes_before_any_execution(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontline = FakeChatModel(emit_tool_calls=False)
    reasoning = FakeChatModel(emit_tool_calls=False)
    engine, store = _engine(
        config_root,
        frontline=frontline,
        reasoning=reasoning,
        thread_id="invalid-invocation-ledger",
    )
    graph_stream_calls = 0
    real_astream = engine._graph.astream

    def observe_astream(*args, **kwargs):
        nonlocal graph_stream_calls
        graph_stream_calls += 1
        return real_astream(*args, **kwargs)

    monkeypatch.setattr(engine._graph, "astream", observe_astream)
    engine._graph.update_state(
        engine._config,
        {
            "consumed_turn_ids": ("admitted-turn",),
            "active_invocation": ActiveInvocation(
                request=ViewIdentityStatus(),
                opened_turn_id="never-admitted",
            ),
        },
        as_node="__start__",
    )

    events = await _events(engine, "continue")
    snapshot = engine._graph.get_state(engine._config)

    assert [
        (event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)
    ] == [("automation_terminal_response", AUTOMATION_TERMINAL_LINE)]
    assert frontline.invoke_count == 0
    assert reasoning.invoke_count == 0
    assert graph_stream_calls == 0
    assert store.placed_count == 0
    assert store.cancel_count == 0
    assert store.refund_count == 0
    assert store.return_count == 0
    assert snapshot.values.get("active_invocation") is None
    assert snapshot.values["automation_terminal"] is True
    assert snapshot.next == ()
    assert engine._terminal_latched is True


@pytest.mark.parametrize(
    "clarification",
    (
        IdentityClarification(),
        SupportClarification(detail="order"),
        CartClarification(detail="quantity"),
    ),
)
def test_pending_clarification_is_a_closed_discriminated_union(clarification: object) -> None:
    state = ReasoningState(pending_clarification=clarification)
    assert state.pending_clarification == clarification


def test_pending_clarification_rejects_cross_flow_detail() -> None:
    with pytest.raises(ValidationError):
        ReasoningState.model_validate(
            {"pending_clarification": {"flow": "identity", "detail": "order"}}
        )


async def test_clarification_state_roundtrips_and_clears_without_an_active_owner(
    config_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine, _ = _engine(config_root, thread_id="clarification-hygiene")
    engine._graph.update_state(
        engine._config,
        {
            "pending_clarification": SupportClarification(detail="order"),
            "clarification_progress": ClarificationProgress(flow="support", reasks=1),
        },
        as_node="__start__",
    )
    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        seeded = engine._graph.get_state(engine._config)
        assert seeded.values["pending_clarification"] == SupportClarification(detail="order")
        assert seeded.values["clarification_progress"] == ClarificationProgress(
            flow="support", reasks=1
        )
        await _events(engine, "hello")
        finished = engine._graph.get_state(engine._config)
    assert finished.values.get("pending_clarification") is None
    assert finished.values.get("clarification_progress") is None
    assert not any("unregistered" in r.getMessage().lower() for r in caplog.records)


# --- buffer-before-speak (_TurnSpeech, live call #9 P2) ----------------------------------
# The chunk path can't be driven through the graph with non-streaming fakes, so the
# buffering unit is tested directly with synthetic stream items.


_SPEAKABLE = frozenset({"support_guardrail"})
_ASSEMBLE_META = {"langgraph_node": "support_assemble"}
_IDENTITY_ASSEMBLE_META = {"langgraph_node": "identity_assemble"}
_CART_ASSEMBLE_META = {"langgraph_node": "cart_assemble"}
_MODEL_META = {"langgraph_node": "model"}


def _chunk(text: str, msg_id: str, *, tool_call: bool = False) -> AIMessageChunk:
    chunks = [{"name": "propose_refund", "args": "", "id": "tc1", "index": 0}] if tool_call else []
    return AIMessageChunk(content=text, id=msg_id, tool_call_chunks=chunks)


def test_graph_declares_disjoint_code_and_model_speech_sources(config_root: Path) -> None:
    engine, _ = _engine(config_root)
    graph = engine._graph
    assert graph.model_speech_nodes == MODEL_SPEECH_NODES
    assert graph.model_speech_nodes == frozenset({"model"})
    assert frozenset(graph.nodes) >= MODEL_SPEECH_NODES
    assert "cart_clarify" in graph.speakable_nodes
    assert "identity_ask_contact" in graph.speakable_nodes
    assert graph.speakable_nodes.isdisjoint(graph.model_speech_nodes)


async def test_cart_clarification_never_routes_to_identity_contact(config_root: Path) -> None:
    engine, _ = _engine(
        config_root,
        reasoning=FakeChatModel(emit_tool_calls=False, text_response="Untrusted Cart prose."),
        thread_id="cart-identity-isolation",
    )
    events = [
        event
        async for event in engine.stream_turn(
            next_committed_turn(engine, "checkout now please"),
            _FACTS,
        )
    ]
    assert not any(
        isinstance(event, SpokenMessageEvent) and event.node == "identity_ask_contact"
        for event in events
    )
    assert [
        (event.node, event.text) for event in events if isinstance(event, SpokenMessageEvent)
    ] == [("cart_clarify", "What would you like to do with your cart?")]
    assert not any(isinstance(event, TokenEvent | InterruptEvent) for event in events)
    state = engine._graph.get_state(
        {"configurable": {"thread_id": "cart-identity-isolation"}}
    ).values
    assert state.get("active_flow") == "cart"
    assert state.get("pending_clarification") is None


def test_streamed_clarify_speaks_once_at_message_completion() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    assert speech.feed(_chunk("Which ", "m1"), _MODEL_META) is None
    assert speech.feed(_chunk("order?", "m1"), _MODEL_META) is None
    event = speech.feed(AIMessage(content="Which order?", id="m1"), _MODEL_META)
    assert isinstance(event, TokenEvent)
    assert event.text == "Which order?"
    assert list(speech.flush()) == []  # never re-spoken


def test_toolcall_narration_never_reaches_the_caller() -> None:
    # Live call #9 P2: "Shall I set the refund...?" streamed alongside propose_refund and
    # was heard OVER the guardrail's return-first decline. The narration must be dropped.
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Shall I refund?", "m1"), _ASSEMBLE_META)
    done = AIMessage(
        content="Shall I refund?",
        id="m1",
        tool_calls=[{"name": "propose_refund", "args": {}, "id": "tc1", "type": "tool_call"}],
    )
    assert speech.feed(done, _ASSEMBLE_META) is None
    assert list(speech.flush()) == []  # and not resurrected at flush


def test_toolcall_detected_from_chunks_alone_stays_dropped() -> None:
    # The completed message never arrives (or arrives under another id): chunks that
    # carried tool_call_chunks must stay dropped at flush.
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Refunding now.", "m1", tool_call=True), _ASSEMBLE_META)
    assert list(speech.flush()) == []


def test_unstreamed_answer_speaks_once() -> None:
    # Non-streaming provider / test fake: the answer arrives only as one full message.
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    event = speech.feed(AIMessage(content="Hello!"), _MODEL_META)
    assert isinstance(event, TokenEvent)
    assert event.text == "Hello!"


def test_approved_frontline_model_retains_plain_text_authority() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    event = speech.feed(AIMessage(content="Current behavior.", id="m1"), _MODEL_META)
    assert isinstance(event, TokenEvent)


def test_identity_assemble_completed_plain_text_is_not_caller_authoritative() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("What is your contact?", "m1"), _IDENTITY_ASSEMBLE_META)
    assert (
        speech.feed(
            AIMessage(content="What is your contact?", id="m1"),
            _IDENTITY_ASSEMBLE_META,
        )
        is None
    )
    assert list(speech.flush()) == []


def test_identity_assemble_orphan_plain_text_is_not_caller_authoritative() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("What is your contact?", "m1"), _IDENTITY_ASSEMBLE_META)
    assert list(speech.flush()) == []


def test_speakable_node_line_is_a_spoken_message() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    event = speech.feed(
        AIMessage(content="Refund declined.", id="g1"), {"langgraph_node": "support_guardrail"}
    )
    assert isinstance(event, SpokenMessageEvent)
    assert event.node == "support_guardrail"


def test_tool_call_text_is_dropped_even_with_a_speakable_node_label() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    message = AIMessage(
        content="I did it.",
        id="m1",
        tool_calls=[{"name": "effect", "args": {}, "id": "tc1", "type": "tool_call"}],
    )
    assert speech.feed(message, {"langgraph_node": "support_guardrail"}) is None


def test_code_and_model_speech_sources_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="speech source sets overlap"):
        _TurnSpeech(frozenset({"model"}), MODEL_SPEECH_NODES)


@pytest.mark.parametrize("meta", [{}, {"langgraph_node": "unknown"}])
def test_missing_or_unknown_completed_source_is_dropped(meta: dict[str, str]) -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    assert speech.feed(AIMessage(content="Untrusted.", id="m1"), meta) is None


def test_missing_chunk_source_can_be_resolved_by_approved_completed_message() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Approved answer.", "m1"), {})
    event = speech.feed(AIMessage(content="Approved answer.", id="m1"), _MODEL_META)
    assert isinstance(event, TokenEvent)


def test_conflicting_chunk_and_completed_sources_fail_closed() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Conflicting.", "m1"), _MODEL_META)
    assert speech.feed(AIMessage(content="Conflicting.", id="m1"), _ASSEMBLE_META) is None


def test_conflicting_chunk_sources_fail_closed() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Con", "m1"), _MODEL_META)
    speech.feed(_chunk("flict.", "m1"), _ASSEMBLE_META)
    assert speech.feed(AIMessage(content="Conflict.", id="m1"), _MODEL_META) is None


def test_missing_or_unknown_orphan_source_is_dropped() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Missing.", "m1"), {})
    speech.feed(_chunk("Unknown.", "m2"), {"langgraph_node": "unknown"})
    assert list(speech.flush()) == []


def test_orphan_streamed_clarify_flushes_exactly_once() -> None:
    # Defensive: chunks streamed but no completed message echoed — the clarify must not be
    # swallowed silently.
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Which order?", "m1"), _MODEL_META)
    assert [e.text for e in speech.flush()] == ["Which order?"]


def test_id_change_between_chunk_and_completion_does_not_double_speak() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Which order?", "m1"), _MODEL_META)
    event = speech.feed(AIMessage(content="Which order?", id="m2"), _MODEL_META)
    assert isinstance(event, TokenEvent)  # spoken at completion under the new id
    assert list(speech.flush()) == []  # the orphaned m1 buffer must not re-speak it


def test_tool_messages_never_surface() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    assert speech.feed(ToolMessage("raw", tool_call_id="t1"), _ASSEMBLE_META) is None
    assert list(speech.flush()) == []


def test_support_assemble_completed_plain_text_is_not_caller_authoritative() -> None:
    """Live-call #18 target: Support completion claims require code-backed authority."""
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Your order is cancelled.", "m1"), _ASSEMBLE_META)
    assert (
        speech.feed(AIMessage(content="Your order is cancelled.", id="m1"), _ASSEMBLE_META) is None
    )
    assert list(speech.flush()) == []


def test_support_assemble_orphan_plain_text_is_not_caller_authoritative() -> None:
    """Live-call #18 target: orphaned Support prose loses authority in Milestone 3c."""
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Your order is cancelled.", "m1"), _ASSEMBLE_META)
    assert list(speech.flush()) == []


def test_cart_assemble_completed_plain_text_is_not_caller_authoritative() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Your order is placed.", "m1"), _CART_ASSEMBLE_META)
    assert (
        speech.feed(
            AIMessage(content="Your order is placed.", id="m1"),
            _CART_ASSEMBLE_META,
        )
        is None
    )
    assert list(speech.flush()) == []


def test_cart_assemble_orphan_plain_text_is_not_caller_authoritative() -> None:
    speech = _TurnSpeech(_SPEAKABLE, MODEL_SPEECH_NODES)
    speech.feed(_chunk("Your order is placed.", "m1"), _CART_ASSEMBLE_META)
    assert list(speech.flush()) == []


# --- graph latency spans (_GraphSpans, live call #10) ------------------------------------


_MODEL = {"langgraph_node": "model"}  # the only node that runs the LLM


def test_graph_spans_counts_tools_and_ttf_model() -> None:
    spans = _GraphSpans()
    spans.observe(
        AIMessage(
            content="",
            tool_calls=[{"name": "order_status", "args": {}, "id": "t1", "type": "tool_call"}],
        ),
        _MODEL,
    )
    spans.observe(ToolMessage("processing", tool_call_id="t1"), {"langgraph_node": "tools"})
    spans.observe(AIMessage(content="It's processing."), _MODEL)  # a real second model pass
    # A tool ran, and the second model pass (rendering the result) is timed separately.
    assert spans._tool_count == 1
    assert spans._ttf_model is not None
    assert spans._tool_to_next_model is not None  # tool-result -> next model activity


def test_graph_spans_single_pass_has_no_tool_span() -> None:
    spans = _GraphSpans()
    spans.observe(AIMessage(content="Hello!"), _MODEL)
    assert spans._tool_count == 0
    assert spans._ttf_model is not None
    assert spans._tool_to_next_model is None  # no tool -> no second-pass span


def test_graph_spans_ignores_empty_and_tool_only_for_ttf() -> None:
    spans = _GraphSpans()
    spans.observe(AIMessage(content=""), _MODEL)  # empty: not model output
    assert spans._ttf_model is None
    spans.observe(
        AIMessage(
            content="", tool_calls=[{"name": "x", "args": {}, "id": "t1", "type": "tool_call"}]
        ),
        _MODEL,
    )
    assert spans._ttf_model is not None  # a tool-call message IS model activity


def test_graph_spans_render_node_is_not_a_model_pass() -> None:
    # L3: a deterministic read renderer (read_render node) authors the post-tool line INSTEAD
    # of a second model pass. That node-authored AIMessage must NOT be timed as model
    # activity, or every rendered turn shows a phantom tool_to_next_model cost.
    spans = _GraphSpans()
    spans.observe(
        AIMessage(
            content="",
            tool_calls=[{"name": "order_status", "args": {}, "id": "t1", "type": "tool_call"}],
        ),
        _MODEL,
    )
    spans.observe(ToolMessage("...", tool_call_id="t1"), {"langgraph_node": "tools"})
    spans.observe(
        AIMessage(content="Your order ORD-1001 is on its way."), {"langgraph_node": "read_render"}
    )
    assert spans._tool_count == 1
    assert spans._tool_to_next_model is None  # rendered, NOT a second model pass


# --- the seam's zero-LiveKit claim ------------------------------------------------------


def test_engine_module_imports_no_livekit() -> None:
    # The dependency arrow is voice -> engine ONLY (AGENTS §A0): the engine must be
    # importable and testable without the voice plane.
    import agnostic_market.agents.engine as engine_module

    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "livekit" not in source.lower()


# --- Fix 5 Milestone 0: LangGraph thread-rotation CONTRACT (framework behavior gate) --------
#
# This is a self-contained FRAMEWORK contract test (a mini-graph below; NO production graph, NO
# src/ dependency beyond build_checkpointer's production serde). It pins the LangGraph behaviors
# Fix 5's principal-context rotation (Milestone D) will rely on, so a LangGraph upgrade that
# changes any of them fails HERE instead of silently breaking rotation:
#   1. a FRESH thread is seeded with ONLY a typed continuation via update_state(as_node=START);
#   2. that seed routes DETERMINISTICALLY into the intended bootstrap node;
#   3. a confirmation interrupt is created AND resumed IN THE NEW thread, its effect running once;
#   4. a DELETED old thread cannot recover its messages/interrupt/continuation and does NOT run the
#      old effect (it may restart from START — pinned precisely so a behavior change is caught).
# The active-thread-id SWAP that makes "old cannot resume" a hard guarantee is PRODUCTION code
# (Milestone D) with its own test; deletion alone is what this contract covers.


def _rot_append(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


class _RotState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    continuation: str | None  # the typed continuation SEED (a proposal, not authority)
    resolved: str | None  # what the bootstrap froze from the continuation
    effect_done: int  # idempotence witness
    visited: Annotated[list[str], _rot_append]


def _rotation_contract_graph():
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def resolver(state: _RotState) -> dict:  # the continuation bootstrap node
        return {"resolved": f"target::{state.get('continuation')}", "visited": ["resolver"]}

    def confirm(state: _RotState) -> dict:
        decision = interrupt({"ask": f"confirm {state.get('resolved')}?"})
        return {"visited": ["confirm"], "messages": [("human", str(decision))]}

    def route_after_confirm(state: _RotState) -> str:
        last = state["messages"][-1]
        text = (last.content if hasattr(last, "content") else str(last)).lower()
        return "effect" if "yes" in text else END

    def effect(state: _RotState) -> dict:
        return {"effect_done": state.get("effect_done", 0) + 1, "visited": ["effect"]}

    g = StateGraph(_RotState)
    g.add_node("resolver", resolver)
    g.add_node("confirm", confirm)
    g.add_node("effect", effect)
    g.add_edge(START, "resolver")
    g.add_edge("resolver", "confirm")
    g.add_conditional_edges("confirm", route_after_confirm, {"effect": "effect", END: END})
    g.add_edge("effect", END)
    return g.compile(checkpointer=build_checkpointer())


def test_langgraph_thread_rotation_contract() -> None:
    from langgraph.types import Command

    graph = _rotation_contract_graph()
    old = {"configurable": {"thread_id": "OLD"}}
    new = {"configurable": {"thread_id": "NEW"}}

    # OLD thread accumulates prior-principal state and pauses at an interrupt.
    graph.update_state(old, {"messages": [("human", "A-private")], "continuation": "A"})
    graph.invoke(None, old)
    assert graph.get_state(old).next == ("confirm",)
    assert len(graph.get_state(old).interrupts) == 1

    # ROTATE: delete OLD, seed a FRESH thread with ONLY the typed continuation.
    graph.checkpointer.delete_thread("OLD")
    graph.update_state(new, {"continuation": "B-cancel-all", "effect_done": 0}, as_node="__start__")
    seeded = graph.get_state(new)
    assert seeded.values.get("continuation") == "B-cancel-all"
    assert seeded.values.get("messages") in (None, [])  # NO prior-principal message bleed
    assert seeded.next == ("resolver",)  # (2) deterministic bootstrap into the intended node

    # Drive NEW with no input: resolver -> confirm -> interrupt IN THE NEW THREAD.
    graph.invoke(None, new)
    st = graph.get_state(new)
    assert [i.value["ask"] for i in st.interrupts] == ["confirm target::B-cancel-all?"]
    assert st.next == ("confirm",)

    # (4) DELETED OLD cannot recover: a stray resume restarts from START — old messages/interrupt/
    # continuation gone, and the old effect is NOT executed.
    old_after = graph.invoke(Command(resume="yes"), old)
    assert old_after.get("effect_done") is None  # old effect never ran
    assert old_after.get("visited") == ["resolver"]  # restarted from START, not resumed at confirm
    assert old_after.get("messages", []) == []  # no "A-private" recovered

    # (3) Resume the NEW thread's interrupt -> effect runs EXACTLY once.
    final = graph.invoke(Command(resume="yes"), new)
    assert final.get("effect_done") == 1
    assert final.get("visited") == ["resolver", "confirm", "effect"]
    assert graph.get_state(new).next == ()  # completed
