"""Frontline graph: structural safety + routing (gate / read-only / model-handover). Zero net."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import MappingProxyType

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from llm_fakes import FakeChatModel
from policy_helpers import make_policy

from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.recovery import clear_automation_state
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
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
from agnostic_market.dtos.orchestration import CancelOrders, ListOrders, SwitchAccount
from agnostic_market.dtos.recovery import AbandonmentKind, ExceptionAction
from agnostic_market.dtos.state import (
    HandoffDestination,
    HandoffReasonCode,
    HandoffRequest,
    PolicyContext,
    ReasoningState,
)
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

# A DEFERRING destination (planner) — these tests exercise the destination-agnostic handover
# CONTROL mechanism (routing through Command, deferral speak, history hygiene), NOT a specific
# flow. checkout/support destinations ENTER their flows (3b/3c) instead of deferring, so a
# mechanism test must use a destination that still ends at the spoken deferral.
_HANDOVER_ARGS = {"request_handover": {"destination": "planner", "reason_code": "multi_step"}}
_READ_ARGS = {"order_status": {"order_id": "ORD-1001"}, "catalog_search": {"query": "shoes"}}
_TEST_OTP = "482913"


def _granted(*order_ids: str) -> CallerIdentityStore:
    """A session identity store with rung-1 grants — for tests exercising what happens
    AFTER an authorized order read (the L3 render path), not the gate itself."""
    identity = CallerIdentityStore()
    for oid in order_ids:
        identity.grant_order(oid)
    return identity


def _tools(
    config_root: Path,
    store: OrderStore,
    cart: CartStore,
    recent_orders: RecentOrderContext,
    identity: CallerIdentityStore,
) -> list:
    customers = CustomerDirectory(load_customers_fixture(config_root, "acme_store"))
    return [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, cart, recent_orders, identity, customers)
    ]


def _graph(config_root: Path, fake: FakeChatModel, **kwargs):
    store = kwargs.pop("store", None) or OrderStore(load_orders_fixture(config_root, "acme_store"))
    # The SAME cart instance must reach build_voice_tools AND the graph, or view_cart reads a
    # different cart than the render node (split-brain, graph docstring). Same for the
    # identity store (P7): the tool grants into it, the render router reads it.
    cart = kwargs.pop("cart_store", None) or CartStore()
    policy = kwargs.pop("policy", None) or make_policy(refund_returnless_under_usd=50.0)
    recent_orders = kwargs.pop("recent_orders", None) or RecentOrderContext(
        max_refs=policy.cancel_batch_max
    )
    identity = kwargs.pop("identity", None) or CallerIdentityStore()
    otp = kwargs.pop("otp", None) or OtpProvider(valid_code=_TEST_OTP)
    verification = kwargs.pop("verification_store", None) or VerificationStore(otp)
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart,
        recent_orders=recent_orders,
        identity_store=identity,
        order_store=store,
    )
    return build_frontline_graph(
        fake,
        _tools(config_root, store, cart, recent_orders, identity),
        display_name="Acme Store",
        tenant_id="acme_store",
        cart_store=cart,
        recent_orders=recent_orders,
        otp=otp,
        verification_store=verification,
        identity_store=identity,
        customers=CustomerDirectory(load_customers_fixture(config_root, "acme_store")),
        payment_instruments=PaymentInstrumentDirectory(
            load_payment_instruments_fixture(config_root, "acme_store")
        ),
        profile_store=ProfileStore(load_profile_fixture(config_root, "acme_store")),
        # Frontline-path tests never reach checkout; a default fake keeps one graph shape.
        reasoning_model=kwargs.pop("reasoning_model", None) or FakeChatModel(),
        store=store,
        policy=policy,
        transition_principal=caller_context.transition_principal,
        principal_state_will_be_discarded=caller_context.has_discardable_state,
        **kwargs,
    )


# --- the structural safety invariant (T1's structural half) --------------------------


def test_frontline_holds_no_sensitive_tool(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    # The only tools the frontline can call are the read-only ones + request_handover
    # (a control signal, not a mutation). NO cart-write / place-order / refund / profile.
    assert graph.frontline_read_only_tools == {
        "order_status",
        "list_orders",
        "catalog_search",
        "view_cart",
    }


def test_all_regular_nodes_have_the_reviewed_recovery_policy(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    policies = graph.node_recovery_policies
    expected_abandonment = {
        AbandonmentKind.PURE_ABORT: {
            "entry",
            "cross_switch",
            "gate",
            "model",
            "finalize",
            "read_render",
            "forced_status",
            "enumeration_gate",
            "cart_clarify",
            "cart_guardrail",
            "cart_abort",
            "support_assemble",
            "support_continuation",
            "support_clarify",
            "support_guardrail",
            "support_risk_check",
            "support_cancel_guardrail",
            "support_resolve",
            "support_return_guardrail",
            "support_profile_guardrail",
            "support_profile_risk_check",
            "support_abort",
            "identity_assemble",
            "identity_ask_contact",
            "identity_reask",
            "identity_guardrail",
            "identity_risk_check",
            "identity_abort",
        },
        AbandonmentKind.CART_REVIEW: {"cart_assemble", "cart_ack"},
        AbandonmentKind.AUTHORITATIVE_RECONCILE: {
            "cart_place",
            "support_place",
            "support_cancel_void",
            "support_return_place",
            "support_profile_place",
        },
        AbandonmentKind.LIFECYCLE_SPECIAL: {
            "tools",
            "principal_warning",
            "cart_confirm",
            "support_dispatch",
            "support_collect",
            "support_confirm",
            "support_cancel_confirm",
            "support_return_confirm",
            "support_profile_dispatch",
            "support_profile_collect",
            "support_profile_confirm",
            "identity_dispatch",
            "identity_collect",
            "identity_apply",
        },
        AbandonmentKind.TERMINAL: {
            "handover",
            "automation_terminal_response",
            "cart_escape_human",
            "support_escape_human",
            "identity_escape_human",
        },
    }
    expected_exception = {
        ExceptionAction.SAFE_ABORT: {
            *expected_abandonment[AbandonmentKind.PURE_ABORT],
            "tools",
        },
        ExceptionAction.CART_REVIEW: {"cart_assemble", "cart_ack"},
        ExceptionAction.RECONCILE_PLACEMENT: {"cart_place"},
        ExceptionAction.RECONCILE_REFUND: {"support_place"},
        ExceptionAction.RECONCILE_CANCEL: {"support_cancel_void"},
        ExceptionAction.RECONCILE_RETURN: {"support_return_place"},
        ExceptionAction.RECONCILE_PROFILE_CHANGE: {"support_profile_place"},
        ExceptionAction.ABORT_PRINCIPAL_WARNING: {"principal_warning"},
        ExceptionAction.ABORT_PLACEMENT_CONFIRMATION: {"cart_confirm"},
        ExceptionAction.ABORT_REFUND_VERIFICATION: {"support_dispatch", "support_collect"},
        ExceptionAction.ABORT_REFUND_CONFIRMATION: {"support_confirm"},
        ExceptionAction.ABORT_CANCEL_CONFIRMATION: {"support_cancel_confirm"},
        ExceptionAction.ABORT_RETURN_CONFIRMATION: {"support_return_confirm"},
        ExceptionAction.ABORT_PROFILE_VERIFICATION: {
            "support_profile_dispatch",
            "support_profile_collect",
        },
        ExceptionAction.ABORT_PROFILE_CONFIRMATION: {"support_profile_confirm"},
        ExceptionAction.ABORT_IDENTITY_VERIFICATION: {
            "identity_dispatch",
            "identity_collect",
        },
        ExceptionAction.RECONCILE_PRINCIPAL_TRANSITION: {"identity_apply"},
        ExceptionAction.TERMINAL: expected_abandonment[AbandonmentKind.TERMINAL],
    }

    regular_nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert isinstance(policies, MappingProxyType)
    assert len(policies) == 54
    assert set(policies) == regular_nodes
    assert Counter(policy.on_abandonment for policy in policies.values()) == {
        kind: len(nodes) for kind, nodes in expected_abandonment.items()
    }
    for kind, names in expected_abandonment.items():
        assert {name for name, policy in policies.items() if policy.on_abandonment == kind} == names
    for action, names in expected_exception.items():
        assert {name for name, policy in policies.items() if policy.on_exception == action} == names


def _handover_update(
    graph,
    destination: HandoffDestination,
    reason_code: HandoffReasonCode,
) -> dict[str, object]:
    state = ReasoningState(
        handover=HandoffRequest(
            destination=destination,
            reason_code=reason_code,
            source="gate",
        )
    )
    return graph.nodes["handover"].invoke(state)


def test_total_clear_is_used_only_for_terminal_handover_and_cross_switch(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel())
    assert _handover_update(graph, "human", "other") == {
        **clear_automation_state(),
        "automation_terminal": True,
    }

    cross_switch = graph.nodes["cross_switch"].invoke(
        ReasoningState(
            messages=[HumanMessage("refund my order")],
            active_flow="cart",
        )
    )
    assert cross_switch == {
        **clear_automation_state(),
        "handover": HandoffRequest(
            destination="support",
            reason_code="refund",
            source="gate",
        ),
    }


def test_non_human_handover_entries_remain_partial_updates(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())

    assert _handover_update(graph, "checkout", "cart_write") == {
        "active_flow": "cart",
        "handover": None,
        "clarification_progress": None,
    }
    assert _handover_update(graph, "support", "switch_account") == {
        "active_flow": "identity",
        "handover": None,
        "identity_claim_misses": 0,
        "clarification_progress": None,
        "pending_request": SwitchAccount(),
    }
    assert _handover_update(graph, "support", "list_orders") == {
        "active_flow": "identity",
        "handover": None,
        "identity_claim_misses": 0,
        "clarification_progress": None,
        "pending_request": ListOrders(scope="account"),
    }
    for reason_code in ("refund", "cancel_order", "address_change", "contact_change"):
        assert _handover_update(graph, "support", reason_code) == {
            "active_flow": "support",
            "handover": None,
            "clarification_progress": None,
        }


def test_bound_and_session_list_readbacks_remain_partial_updates(config_root: Path) -> None:
    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    bound_graph = _graph(config_root, FakeChatModel(), identity=identity)
    bound = _handover_update(bound_graph, "support", "list_orders")

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    _place_two_session_orders(store)
    session_graph = _graph(config_root, FakeChatModel(), store=store)
    session = _handover_update(session_graph, "support", "list_orders")

    expected_keys = {"messages", "active_flow", "handover", "identity_claim_misses"}
    assert set(bound) == expected_keys
    assert set(session) == expected_keys
    for update in (bound, session):
        assert update["active_flow"] is None
        assert update["handover"] is None
        assert update["identity_claim_misses"] == 0


# --- routing paths -------------------------------------------------------------------


async def test_gate_trip_skips_model_and_hands_over(config_root: Path) -> None:
    # The slim gate trips on high-certainty IRREVERSIBLE requests (here: cancel) BEFORE any
    # generation: the frontline model is never invoked, and the turn enters the support
    # flow directly (cancel is a BUILT capability — it no longer defers; and if support's
    # model bounces and leaves, the gate-skip hands the answer to the frontline model
    # rather than a canned deferral — see checkout's gate-skip test).
    fake = FakeChatModel(tool_call_limit=1)
    reasoning = FakeChatModel(emit_tool_calls=False)  # support clarifies; stays in flow
    graph = _graph(config_root, fake, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("cancel my order please")]})
    assert out["active_flow"] == "support"  # entered by the gate, pre-generation
    assert fake._tool_calls_made == 0  # the frontline model never ran


async def test_cancel_order_enters_the_support_flow(config_root: Path) -> None:
    # Group A: cancel_order is now a BUILT support capability, so a cancel_order handover
    # ENTERS the support flow (it no longer defers — that was the 3c-only behavior). The
    # support assemble model runs and proposes the cancel. (Group C: address/contact change
    # enter too; only payment_change defers.) Fix 2: ORD-1002 is CUST-002's and this session is
    # UNBOUND, so the mutation cannot authorize on a rung-1 pair — it DETOURS into the identity
    # OTP flow (no pending minted, "no pending before auth"); it resumes + mints after the bind.
    reasoning = FakeChatModel(
        force_tool="propose_cancel",
        canned_args={"propose_cancel": {"order_keys": ["ORD-1002"]}},
        tool_call_limit=1,
    )
    graph = _graph(config_root, FakeChatModel(tool_call_limit=1), reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("actually cancel order ORD-1002")]})
    assert reasoning._tool_calls_made == 1  # the support model ran and proposed a cancel
    assert out.get("active_flow") == "identity"  # detoured to verify (rung-2 required)
    assert out.get("pending_cancel") is None  # nothing staged before the caller is bound
    request = out.get("pending_request")
    assert isinstance(request, CancelOrders)
    assert request.target.order_refs == ("ORD-1002",)


async def test_address_change_enters_the_support_flow(config_root: Path) -> None:
    # Group C: address_change flipped from defer -> enter (the profile flow is built).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {"destination": "support", "reason_code": "address_change"}
        },
        tool_call_limit=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)  # support clarifies; stays in flow
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("I need to update my address")]})
    assert out.get("active_flow") == "support"  # ENTERED (no deferral line)
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("support team" in t for t in texts)  # the old deferral is gone


async def test_payment_change_still_defers(config_root: Path) -> None:
    # Phase 5 boundary pin: payment_change must NOT enter (its flow doesn't exist — entering
    # would bounce off the assemble and double-speak). The honest deferral speaks once.
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={
            "request_handover": {"destination": "support", "reason_code": "payment_change"}
        },
        tool_call_limit=1,
    )
    graph = _graph(config_root, frontline)
    out = await graph.ainvoke({"messages": [HumanMessage("put my new card on the account")]})
    assert out.get("active_flow") is None
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert sum("support team" in t for t in texts) == 1  # exactly one deferral line


async def test_read_only_turn_answers_without_handover(config_root: Path) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    assert out.get("handover") is None
    assert [type(m).__name__ for m in out["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]


async def test_model_handover_routes_through_command(config_root: Path) -> None:
    # Trigger-free phrasing the gate can't pattern -> the model calls request_handover.
    # This fake emits an EMPTY-content tool call (no narration), so the node's canned
    # deferral fires as the fallback — the caller is never left silent.
    fake = FakeChatModel(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("I moved recently, make sure it goes to the right place")]}
    )
    assert out["handover"].source == "model"
    # Proper tool_use/tool_result pairing (G1): the executed handover tool left a ToolMessage.
    assert any(isinstance(m, ToolMessage) for m in out["messages"])
    assert "picked up" in out["messages"][-1].content  # the planner deferral line


async def test_model_narration_is_not_double_spoken(config_root: Path) -> None:
    # Live 2026-07-08 bug: when the model narrates its handover AND the node appends the
    # canned deferral, the caller hears two deferrals. If the model already spoke, the
    # node must stay silent (its deferral is a fallback only).
    class NarratingFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            msg = super()._respond(messages, **kwargs)
            if msg.tool_calls:  # attach spoken narration alongside the handover tool call
                msg.content = "I'll connect you with our support team for that."
            return msg

    fake = NarratingFake(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("plan a whole trip outfit for me")]})
    assert out["handover"].source == "model"
    # The node did NOT append its canned deferral (the model's narration is the deferral).
    canned = "I'll make sure it's picked up"  # the planner deferral line
    assert not any(
        isinstance(m, AIMessage) and canned in (m.content or "") for m in out["messages"]
    )


async def test_two_turn_history_stays_clean(config_root: Path) -> None:
    # F2 regression: after a model-handover turn, the next turn runs without a dangling
    # tool_use breaking the model call.
    fake = FakeChatModel(
        force_tool="request_handover", tool_call_limit=1, canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    first = await graph.ainvoke({"messages": [HumanMessage("send it to my work address")]})
    second = await graph.ainvoke(
        {"messages": [*first["messages"], HumanMessage("what's the status of order ORD-1001")]}
    )
    assert second["messages"]  # completed without error


async def test_model_node_prepends_platform_system_prompt(config_root: Path) -> None:
    # F1: the prompt lives inside the graph, so eval and production share one prompt path.
    seen: dict[str, object] = {}

    class RecordingFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            seen["first"] = messages[0]
            return super()._respond(messages, **kwargs)

    fake = RecordingFake(tool_call_limit=0, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)
    await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert isinstance(seen["first"], SystemMessage)
    assert "Acme Store" in seen["first"].content


async def test_plain_answer_ends_without_tools(config_root: Path) -> None:
    fake = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("hi there")]})
    assert out.get("handover") is None
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content


async def test_runaway_tool_loop_is_bounded(config_root: Path) -> None:
    # A model that never stops calling read tools must not spin forever: the hop guard ends
    # the turn after policy.max_tool_hops round-trips (no framework loop protection here).
    # Forces catalog_search (NON-renderable — stays on the model->tools->model loop; a
    # single renderable read like order_status would divert to read_render and END after one
    # hop, which is a STRONGER bound, not the loop this guard protects).
    default_hops = make_policy().max_tool_hops
    fake = FakeChatModel(force_tool="catalog_search", canned_args=_READ_ARGS)  # no limit set
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    hops = sum(1 for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert hops == default_hops
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content  # limit produces a final answer, not a dangling call
    for index, message in enumerate(out["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            assert isinstance(out["messages"][index + 1], ToolMessage)


async def test_tool_hop_bound_tracks_the_policy_knob(config_root: Path) -> None:
    # The bound is config-driven (policies.security.max_tool_hops), not a hardcoded constant:
    # a tightened knob ends the turn sooner. Pins that the value actually THREADS to the guard.
    fake = FakeChatModel(force_tool="catalog_search", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, policy=make_policy(max_tool_hops=2))
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    hops = sum(1 for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert hops == 2


async def test_tool_hop_bound_strips_a_provider_tool_call_from_the_final_pass(
    config_root: Path,
) -> None:
    class ToolCallingWithoutToolsFake(FakeChatModel):
        def _respond(self, messages, **kwargs):  # type: ignore[override]
            if not kwargs.get("tools"):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "catalog_search",
                            "args": {"query": "shoes"},
                            "id": "invalid_final_call",
                            "type": "tool_call",
                        }
                    ],
                )
            return super()._respond(messages, **kwargs)

    fake = ToolCallingWithoutToolsFake(
        force_tool="catalog_search", canned_args=_READ_ARGS, tool_call_limit=None
    )
    graph = _graph(config_root, fake, policy=make_policy(max_tool_hops=1))
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    tool_messages = [message for message in out["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    final = out["messages"][-1]
    assert isinstance(final, AIMessage) and not final.tool_calls and final.content


async def test_state_followup_reads_recent_context_without_calling_the_model(
    config_root: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    store.cancel_order("cancel-1", order_id="ORD-1002")
    recent_orders = RecentOrderContext(max_refs=make_policy().cancel_batch_max)
    recent_orders.record(["ORD-1002"], operation="read")
    identity = _granted("ORD-1002")
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        store=store,
        recent_orders=recent_orders,
        identity=identity,
    )
    out = await graph.ainvoke({"messages": [HumanMessage("is it cancelled?")]})
    assert "ORD-1002" in out["messages"][-1].content
    assert "cancelled" in out["messages"][-1].content


async def test_plural_state_followup_corrects_a_memory_claim_from_live_store(
    config_root: Path,
) -> None:
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    store.cancel_order("cancel-1", order_id="ORD-1002")
    recent_orders = RecentOrderContext(max_refs=make_policy().cancel_batch_max)
    recent_orders.record(["ORD-1001", "ORD-1002"], operation="list")
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        store=store,
        recent_orders=recent_orders,
        identity=_granted("ORD-1001", "ORD-1002"),
    )
    out = await graph.ainvoke(
        {
            "messages": [
                AIMessage("ORD-1001 and ORD-1002 are both cancelled."),
                HumanMessage("so both are cancelled?"),
            ]
        }
    )
    line = out["messages"][-1].content
    assert "ORD-1001" in line and "ORD-1002" in line
    assert "ORD-1001" in line and "was expected" in line
    assert "ORD-1002" in line and "cancelled" in line


async def test_unverified_all_orders_state_check_enters_identity_flow(
    config_root: Path,
) -> None:
    graph = _graph(
        config_root,
        FakeChatModel(raise_transport=True),
        reasoning_model=FakeChatModel(emit_tool_calls=False),
    )
    out = await graph.ainvoke({"messages": [HumanMessage("are all my orders cancelled?")]})
    assert out.get("active_flow") == "identity"


async def test_unverified_explicit_state_check_does_not_bypass_order_authorization(
    config_root: Path,
) -> None:
    graph = _graph(config_root, FakeChatModel(emit_tool_calls=False))
    out = await graph.ainvoke({"messages": [HumanMessage("is ORD-1002 cancelled?")]})
    final = out["messages"][-1]
    assert final.content == _TEXT_RESPONSE
    assert "waterproof rain jacket" not in final.content


async def test_answered_turn_writes_telemetry_negative(config_root: Path, tmp_path) -> None:
    # The classifier dataset needs NEGATIVES: an answered (non-escalated) turn must leave
    # a telemetry line too, not only handovers. (Telemetry is redirected to tmp by conftest.)
    import json

    from agnostic_market.agents import telemetry

    fake = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, fake)
    await graph.ainvoke({"messages": [HumanMessage("hi there")]})
    lines = [
        json.loads(line)
        for line in telemetry._TELEMETRY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any(rec["outcome"] == "answered" and rec["utterance"] == "hi there" for rec in lines)


# --- policy grounding: DERIVED from enforced values (no drift), + free-text extras --------


def _policy(**over) -> PolicyContext:
    # The policy-grounding suite's default carries a spoken_policy_extra; everything else
    # (incl. the security knobs) comes from the shared factory. Any field overridable —
    # `over` wins over these defaults (a test may set spoken_policy_extra=None).
    return make_policy(
        **{
            "refund_returnless_under_usd": 50.0,
            "spoken_policy_extra": "Refunds take 5 to 7 business days.",
            **over,
        }
    )


def test_prompt_grounds_policy_from_enforced_values() -> None:
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy())
    assert "Refunds take 5 to 7 business days." in prompt  # the free-text extra
    assert "$50" in prompt  # the returnless threshold, DERIVED from the enforced value
    assert "$200" in prompt  # the human-review line, DERIVED
    assert "ONLY policy statements" in prompt


def test_spoken_policy_tracks_the_enforced_value_no_drift() -> None:
    # The whole point: change the ENFORCED number, the spoken sentence changes with it —
    # they are one source, so a merchant can't state a threshold the guardrail won't honor.
    from agnostic_market.agents.spoken_policy import compose_spoken_policy

    assert "$50" in compose_spoken_policy(_policy(refund_returnless_under_usd=50.0))
    assert "$120" in compose_spoken_policy(_policy(refund_returnless_under_usd=120.0))
    # returnless 0 => return-first for every shipped refund (no dollar threshold spoken).
    zero = compose_spoken_policy(_policy(refund_returnless_under_usd=0.0))
    assert "issued once the return is arranged" in zero
    # The return window (Group C: enforced by the returns guardrail) is derived the same way.
    assert "within 30 days of delivery" in compose_spoken_policy(_policy(return_window_days=30))
    assert "within 60 days of delivery" in compose_spoken_policy(_policy(return_window_days=60))


def test_prompt_speaks_derived_policy_even_without_free_text() -> None:
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy(spoken_policy_extra=None))
    assert "$50" in prompt  # derived sentences exist for every merchant
    assert "NEVER invent" in prompt


def test_prompt_forbids_order_status_from_confirming_account_ownership() -> None:
    # Fix 4 (live 2026-07-18): a guest read ORD-1002 via order#+email, then asked "is it on
    # this account, under that email?" and the model answered "Yes, it's on the account for
    # casey@example.com" - an ownership ORACLE (anyone with an order number could learn whose
    # account it is) + a PII echo. The frontline prompt must forbid confirming the account link
    # and repeating the caller's contact; an order_status read speaks order STATE only.
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy())
    assert "STATE ONLY" in prompt
    assert "NEVER confirms, denies, or discusses WHOSE account" in prompt
    assert "Never repeat the caller's email or phone number" in prompt
    # the contrastive few-shot for the ownership probe is present
    assert "that order is on this account" in prompt


def test_shared_context_carries_todays_date_and_past_eta_rule() -> None:
    # Live call #9 P6: with no "today" in any prompt, a stored ETA of July 9 was spoken as
    # a FUTURE arrival on July 13. The date is read at compose time (per turn).
    from datetime import datetime

    from agnostic_market.agents._shared_prompt import compose_shared_context

    context = compose_shared_context("Acme Store", _policy())
    assert f"Today's date: {datetime.now():%A %d %B %Y}" in context
    assert "BEFORE today is in the PAST" in context


def test_shared_context_reaches_every_agent_prompt() -> None:
    # Knowledge must not be tier-local (live 2026-07-10: policy facts lived only in the
    # frontline prompt, so a policy question that gate-routed into support met a model
    # with zero policy knowledge). All three composers carry the SAME shared block:
    # persona continuity + the DERIVED policy summary.
    from agnostic_market.agents.cart.prompt import compose_cart_prompt
    from agnostic_market.agents.frontline.prompt import compose_system_prompt
    from agnostic_market.agents.support.prompt import compose_support_prompt
    from agnostic_market.commerce.cart import CartStore
    from agnostic_market.commerce.orders import Candidate, OrderCandidate

    policy = _policy()
    prompts = [
        compose_system_prompt("Acme Store", policy),
        compose_cart_prompt(
            "Acme Store",
            [Candidate(key="1", sku="SKU-1", name="thing", price_usd=1.0)],
            CartStore(),
            policy,
        ),
        compose_support_prompt(
            "Acme Store",
            [
                OrderCandidate(
                    key="1",
                    order_id="ORD-1",
                    summary="a thing",
                    total_usd=1.0,
                    status="processing",
                )
            ],
            policy,
        ),
    ]
    for prompt in prompts:
        assert "ONE continuous assistant" in prompt
        assert "Refunds take 5 to 7 business days." in prompt
        assert "$50" in prompt  # the derived enforced sentence reaches every tier


def test_support_prompt_teaches_batch_cancel_tool() -> None:
    # F-16.2 batch: the support model is told to cancel MULTIPLE orders in ONE propose_cancel
    # call (not one-at-a-time across turns), and to never report an order done from memory. A
    # content pin — the structural fix is code; this guards the wording against regression to
    # the retired (and non-functional) one-at-a-time continuation promise.
    from agnostic_market.agents.support.prompt import compose_support_prompt
    from agnostic_market.commerce.orders import OrderCandidate

    orders = [
        OrderCandidate(key="1", order_id="ORD-1", summary="x", total_usd=1.0, status="processing")
    ]
    prompt = compose_support_prompt("Acme Store", orders, _policy())
    assert "propose_cancel ONCE with ALL their option numbers" in prompt
    assert "NEVER report an order as done from memory" in prompt
    # The retired false promise must not creep back in.
    assert "on the NEXT turn propose the next one" not in prompt


# --- L3 deterministic read renderers: a single order_status/view_cart read is rendered in
#     CODE and ENDs, skipping the second model pass (latency + grounding win). -------------

from llm_fakes import _TEXT_RESPONSE  # noqa: E402  the fake's narration text (single source)


async def test_single_order_status_renders_in_code_and_skips_second_model_pass(
    config_root: Path,
) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    # An AUTHORIZED read (P7): render tests exercise the L3 path, not the object-binding
    # gate — the gate's own pins live below and in test_voice_tools.py.
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    final = out["messages"][-1]
    # The final line is the CODE render (contains the order id + a store-derived status),
    # NOT the model's narration text — proving the second model pass was skipped.
    assert isinstance(final, AIMessage) and "ORD-1001" in final.content
    assert _TEXT_RESPONSE not in final.content
    # The model was invoked exactly ONCE (the tool-call turn); no narration invoke followed.
    assert fake._tool_calls_made == 1


async def test_render_node_is_speakable(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS))
    assert "read_render" in graph.speakable_nodes


async def test_render_appends_a_factual_close_no_product_opinion(config_root: Path) -> None:
    # A status read ends with a warm FACTUAL close — never a product opinion ("nice
    # choice"/"good pick"), which reads as scripted and is nonsensical after a read (live
    # 2026-07-15: appreciative closes dropped).
    from agnostic_market.agents._copy import all_closes

    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    line = out["messages"][-1].content
    assert any(line.endswith(c) for c in all_closes())
    assert "pick" not in line.lower() and "choice" not in line.lower()  # no product opinion


async def test_multi_intent_read_still_goes_to_the_model(config_root: Path) -> None:
    # "status of my order AND do you have socks?" -> the model emits TWO tool calls in one
    # response; the ==1 guard fails, so the turn does NOT divert to read_render after the
    # tools run — it returns to the model, which composes both (its narration text). The
    # tool_call_limit makes that post-tools model invoke return text (not loop).
    fake = FakeChatModel(
        scripted_calls=[
            [("order_status", {"order_id": "ORD-1001"}), ("catalog_search", {"query": "socks"})]
        ],
        tool_call_limit=0,  # after the scripted multi-call, the next invoke narrates
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("status of ORD-1001 and do you have socks")]}
    )
    # The model composed the answer (its narration text), NOT a code render.
    assert out["messages"][-1].content == _TEXT_RESPONSE


async def test_single_catalog_search_stays_model_narrated(config_root: Path) -> None:
    # catalog_search is NOT renderable (fuzzy discovery needs framing) — even a single call
    # routes back to the model.
    fake = FakeChatModel(tool_call_limit=1, force_tool="catalog_search", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("do you have shoes")]})
    assert out["messages"][-1].content == _TEXT_RESPONSE


async def test_single_view_cart_renders_in_code(config_root: Path) -> None:
    cart = CartStore()
    cart.add_item(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=2)
    fake = FakeChatModel(tool_call_limit=1, force_tool="view_cart", canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, cart_store=cart)
    out = await graph.ainvoke({"messages": [HumanMessage("what's in my cart")]})
    final = out["messages"][-1]
    assert "2 rain jackets" in final.content and "$258.00" in final.content
    assert _TEXT_RESPONSE not in final.content


async def test_render_cannot_invent_forward_state(config_root: Path) -> None:
    # A processing order must render "being prepared", never "on the way" — the phrase is
    # derived from the store field, so the embellishment class is structurally impossible.
    fake = FakeChatModel(
        tool_call_limit=1,
        canned_args={"order_status": {"order_id": "ORD-1002"}},
    )
    graph = _graph(config_root, fake, identity=_granted("ORD-1002"))
    out = await graph.ainvoke({"messages": [HumanMessage("status of ORD-1002")]})
    line = out["messages"][-1].content
    assert "being prepared" in line and "on its way" not in line


async def test_handover_turn_never_renders(config_root: Path) -> None:
    # A request_handover turn sets `handover`; the shared predicate excludes it, so the divert
    # never steals a handover turn — it routes to the handover sink (spoken deferral).
    fake = FakeChatModel(
        tool_call_limit=1, force_tool="request_handover", canned_args=_HANDOVER_ARGS
    )
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("I need a human")]})
    # Deferral spoken (planner destination), NOT a code render.
    assert "picked up" in out["messages"][-1].content.lower()


async def test_unauthorized_order_read_never_renders(config_root: Path) -> None:
    # THE P7 render-gate pin: read_render re-derives the line from the STORE, so a declined
    # order_status followed by the render divert would LEAK the order around the tool's
    # object-binding gate. `_render_ready` requires authorization — an unverified single
    # order_status falls back to the model, which narrates the tool's ask-for-contact
    # instruction (the fake's text), and NO store-derived order line is ever spoken.
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)  # identity store fresh: unverified session
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    final = out["messages"][-1]
    assert final.content == _TEXT_RESPONSE  # model narration, NOT a code render
    spoken = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("trail running shoes" in t for t in spoken)  # no order data leaked


async def test_list_orders_handover_enters_the_identity_flow(config_root: Path) -> None:
    # P7 rung 2: a list_orders handover ENTERS the identity flow (no deferral); the identity
    # model runs and asks for the contact on the account (clarify -> stays sticky).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={"request_handover": {"destination": "support", "reason_code": "list_orders"}},
        tool_call_limit=1,
    )
    reasoning = FakeChatModel(emit_tool_calls=False)  # identity model asks its ONE question
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    assert out.get("active_flow") == "identity"  # ENTERED (sticky, awaiting the claim)
    texts = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("support team" in t for t in texts)  # no stale deferral


async def test_identity_apply_is_speakable(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    assert "identity_apply" in graph.speakable_nodes
    assert "identity_reask" in graph.speakable_nodes
    assert "identity_assemble" not in graph.speakable_nodes  # double-speak (cart_ack lesson)


async def test_unverified_enumeration_diverts_without_a_relay_pass(config_root: Path) -> None:
    # THE deterministic enumeration divert (call #13 latency + F-12.3 closed structurally):
    # a single unverified list_orders probe routes STRAIGHT to the identity flow in code.
    # The high-confidence detector avoids the frontline model entirely, so it cannot relay
    # the tool's instruction, ask for the email itself, or narrate.
    frontline = FakeChatModel(force_tool="list_orders", tool_call_limit=1)
    reasoning = FakeChatModel(emit_tool_calls=False)  # identity model asks its ONE question
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    assert out.get("active_flow") == "identity"  # entered via the code-set handover
    assert frontline._tool_calls_made == 0
    spoken = [str(m.content) for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert not any("email or phone" in t and "verification" in t for t in spoken)


async def test_explicit_enumeration_phrase_skips_frontline_model(config_root: Path) -> None:
    # Live call #17: this exact intent was answered "I can't list orders" because only a
    # model-emitted list_orders tool call triggered the deterministic identity divert.
    frontline = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("tell me what order numbers are available")]}
    )
    assert out.get("active_flow") == "identity"


async def test_enumeration_cross_switches_out_of_sticky_support(config_root: Path) -> None:
    # The live failure occurred after a cancel authorization denial left support sticky.
    # Enumeration belongs to identity even though both use the "support" handover destination.
    frontline = FakeChatModel(raise_transport=True)
    reasoning = FakeChatModel(emit_tool_calls=False)
    graph = _graph(config_root, frontline, reasoning_model=reasoning)
    out = await graph.ainvoke(
        {
            "messages": [HumanMessage("tell me what orders are available")],
            "active_flow": "support",
        }
    )
    assert out.get("active_flow") == "identity"
    assert out.get("pending_cancel") is None


async def test_bound_enumeration_renders_the_list_in_code(config_root: Path) -> None:
    # A BOUND session's clear enumeration ask is code-authored from the scoped store view
    # and the turn ENDs without re-entering Identity or invoking the frontline model.
    from agnostic_market.commerce.identity import BoundIdentity

    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    fake = FakeChatModel(force_tool="list_orders", tool_call_limit=1)
    graph = _graph(config_root, fake, identity=identity)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    final = out["messages"][-1]
    assert "ORD-1002" in final.content and "ORD-1001" not in final.content  # scoped
    assert _TEXT_RESPONSE not in final.content  # code render, not model narration
    assert fake._tool_calls_made == 0


async def test_bound_enumeration_escapes_sticky_support_without_otp(config_root: Path) -> None:
    from agnostic_market.commerce.identity import BoundIdentity

    identity = CallerIdentityStore()
    identity.bind(
        BoundIdentity(customer_ref="CUST-002", masked_contact="email ending example dot com")
    )
    graph = _graph(config_root, FakeChatModel(raise_transport=True), identity=identity)
    out = await graph.ainvoke(
        {
            "messages": [HumanMessage("tell me what orders are available")],
            "active_flow": "support",
        }
    )
    final = out["messages"][-1]
    assert "ORD-1002" in final.content and "ORD-1001" not in final.content
    assert out.get("active_flow") is None
    assert out.get("pending_identity") is None


# --- Fix 3: a GUEST lists the orders they placed THIS session (no verification) ------------


def _place_two_session_orders(store: OrderStore) -> tuple[str, str]:
    from agnostic_market.dtos.state import CartLine

    a = store.place_cart(
        "s1",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=2)],
        total_usd=258.0,
    )
    b = store.place_cart(
        "s2",
        lines=[CartLine(sku="SKU-RED-42", name="trail shoes", price_usd=89.99, quantity=1)],
        total_usd=89.99,
    )
    return a.order_id, b.order_id


async def test_guest_lists_session_placed_orders_without_verification(
    config_root: Path, tmp_path: Path
) -> None:
    # THE Fix-3 pin (live trace 2026-07-18): an UNBOUND caller who placed orders this call asks
    # to list them and hears them read back from CODE — no identity handover, no OTP. The spoken
    # line DISCLOSES this-call scope (not "you've got N orders", which implies full history) and
    # carries exactly ONE closing invitation (the verify-for-more line, NOT also warm_close).
    import json

    from agnostic_market.agents._copy import GUEST_LIST_CLOSE, all_closes

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    a, b = _place_two_session_orders(store)
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("tell me what orders are there")]})
    final = out["messages"][-1]
    assert a in final.content and b in final.content  # both session orders read back
    assert out.get("active_flow") is None  # NO identity detour
    assert out.get("pending_identity") is None
    assert "on this call" in final.content  # scope disclosed (not full-history phrasing)
    assert GUEST_LIST_CLOSE in final.content  # the verify-for-more invite
    assert not any(c in final.content for c in all_closes())  # and NOT also a warm_close
    scope = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if '"order_scope"' in line
    ]
    assert scope and scope[-1]["order_scope"] == "session"


async def test_guest_enumeration_never_lists_fixture_orders(config_root: Path) -> None:
    # SECURITY: a guest's session list is drawn ONLY from what they placed — never an account's
    # fixture orders (a different code path). Place ONE order; the spoken list must not name any
    # fixture order id.
    from agnostic_market.dtos.state import CartLine

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    placed = store.place_cart(
        "one",
        lines=[CartLine(sku="SKU-BLU-07", name="rain jacket", price_usd=129.0, quantity=1)],
        total_usd=129.0,
    )
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    final = out["messages"][-1]
    assert placed.order_id in final.content
    assert not any(x in final.content for x in ("ORD-1001", "ORD-1002", "ORD-1003"))


async def test_guest_with_no_placed_orders_still_enters_identity(config_root: Path) -> None:
    # An unbound caller who placed NOTHING has no session orders to read — so enumeration still
    # enters the identity flow (today's behavior; nothing to list without an account).
    frontline = FakeChatModel(
        force_tool="request_handover",
        canned_args={"request_handover": {"destination": "support", "reason_code": "list_orders"}},
        tool_call_limit=1,
    )
    graph = _graph(config_root, frontline, reasoning_model=FakeChatModel(emit_tool_calls=False))
    out = await graph.ainvoke({"messages": [HumanMessage("what orders do I have")]})
    assert out.get("active_flow") == "identity"


async def test_guest_status_list_reads_session_placed(config_root: Path) -> None:
    # Symmetry with the state-verification path: an unbound guest asking "are they both shipped?"
    # reads the session-placed orders (forced-status), not nothing.
    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    a, b = _place_two_session_orders(store)
    graph = _graph(config_root, FakeChatModel(raise_transport=True), store=store)
    out = await graph.ainvoke({"messages": [HumanMessage("are both of my orders shipped?")]})
    final = out["messages"][-1]
    assert a in final.content and b in final.content
    assert out.get("active_flow") is None  # answered in code, no identity


async def test_render_emits_exactly_one_answered_event(config_root: Path, tmp_path: Path) -> None:
    # The render path ENDs at read_render (bypassing finalize_node); it must emit exactly ONE
    # answered-telemetry event, not zero and not a duplicate. (Telemetry is redirected to
    # tmp_path by the autouse conftest fixture.)
    import json

    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake, identity=_granted("ORD-1001"))
    await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    sink = tmp_path / "telemetry.jsonl"
    answered = [
        json.loads(line)
        for line in sink.read_text(encoding="utf-8").splitlines()
        if '"outcome": "answered"' in line
    ]
    assert len(answered) == 1
    assert answered[0]["outcome_detail"] == "code_render"
