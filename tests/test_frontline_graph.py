"""Frontline graph: structural safety + routing (gate / read-only / model-handover). Zero net."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from llm_fakes import FakeChatModel

from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import OrderStore, load_orders_fixture
from agnostic_market.dtos.state import PolicyContext
from agnostic_market.voice.tools import build_voice_tools

# A DEFERRING destination (planner) — these tests exercise the destination-agnostic handover
# CONTROL mechanism (routing through Command, deferral speak, history hygiene), NOT a specific
# flow. checkout/support destinations ENTER their flows (3b/3c) instead of deferring, so a
# mechanism test must use a destination that still ends at the spoken deferral.
_HANDOVER_ARGS = {"request_handover": {"destination": "planner", "reason_code": "multi_step"}}
_READ_ARGS = {"order_status": {"order_id": "ORD-1001"}, "catalog_search": {"query": "shoes"}}


def _tools(store: OrderStore) -> list:
    return [wrap_readonly_tool(t, "acme_store") for t in build_voice_tools(store, CartStore())]


def _graph(config_root: Path, fake: FakeChatModel, **kwargs):
    store = kwargs.pop("store", None) or OrderStore(load_orders_fixture(config_root, "acme_store"))
    return build_frontline_graph(
        fake,
        _tools(store),
        display_name="Acme Store",
        # Frontline-path tests never reach checkout; a default fake keeps one graph shape.
        reasoning_model=kwargs.pop("reasoning_model", None) or FakeChatModel(),
        store=store,
        policy=PolicyContext(
            max_order_value_usd=500.0,
            allow_ai_merchant_handoff=True,
            refund_auto_approve_under_usd=50.0,
            refund_require_human_above_usd=200.0,
            refund_returnless_under_usd=50.0,
            pending_ttl_seconds=120.0,
        ),
        **kwargs,
    )


# --- the structural safety invariant (T1's structural half) --------------------------


def test_frontline_holds_no_sensitive_tool(config_root: Path) -> None:
    graph = _graph(config_root, FakeChatModel())
    # The only tools the frontline can call are the read-only ones + request_handover
    # (a control signal, not a mutation). NO cart-write / place-order / refund / profile.
    assert graph.frontline_read_only_tools == {"order_status", "catalog_search", "view_cart"}


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
    # support assemble model runs and proposes the cancel; the flow reaches its readback
    # interrupt. (address/payment change still defer — only refund + cancel_order enter.)
    reasoning = FakeChatModel(
        force_tool="propose_cancel", canned_args={"propose_cancel": {"order_key": "2"}},
        tool_call_limit=1,
    )
    graph = _graph(config_root, FakeChatModel(tool_call_limit=1), reasoning_model=reasoning)
    out = await graph.ainvoke({"messages": [HumanMessage("actually cancel that order")]})
    assert out.get("active_flow") == "support"  # ENTERED the flow (paused at the readback)
    assert reasoning._tool_calls_made == 1  # the support model ran and proposed a cancel
    assert out.get("pending_cancel") is not None  # a cancel is staged, awaiting confirmation


async def test_read_only_turn_answers_without_handover(config_root: Path) -> None:
    fake = FakeChatModel(tool_call_limit=1, canned_args=_READ_ARGS)
    graph = _graph(config_root, fake)
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
    # A model that never stops calling read tools must not spin forever: the hop guard
    # ends the turn after _MAX_TOOL_HOPS round-trips (no framework loop protection here).
    from agnostic_market.agents.frontline import _MAX_TOOL_HOPS

    fake = FakeChatModel(force_tool="order_status", canned_args=_READ_ARGS)  # no limit set
    graph = _graph(config_root, fake)
    out = await graph.ainvoke({"messages": [HumanMessage("status of order ORD-1001")]})
    hops = sum(1 for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls)
    assert hops == _MAX_TOOL_HOPS + 1  # the hop that crossed the bound ended the turn


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
    base = {
        "max_order_value_usd": 500.0,
        "allow_ai_merchant_handoff": True,
        "refund_auto_approve_under_usd": 50.0,
        "refund_require_human_above_usd": 200.0,
        "refund_returnless_under_usd": 50.0,
        "pending_ttl_seconds": 120.0,
        "spoken_policy_extra": "Refunds take 5 to 7 business days.",
    }
    base.update(over)
    return PolicyContext(**base)


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


def test_prompt_speaks_derived_policy_even_without_free_text() -> None:
    from agnostic_market.agents.frontline.prompt import compose_system_prompt

    prompt = compose_system_prompt("Acme Store", _policy(spoken_policy_extra=None))
    assert "$50" in prompt  # derived sentences exist for every merchant
    assert "NEVER invent" in prompt


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
                    key="1", order_id="ORD-1", summary="a thing", total_usd=1.0,
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
