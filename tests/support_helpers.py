"""Shared support-flow engine harness — ONE builder for the support/returns/profile suites
(extracted from test_support_flow rather than copied per file)."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from llm_fakes import FakeChatModel

from agnostic_market.agents.engine import ReasoningEngine, build_checkpointer
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import LastOrderPointer, OrderStore, load_orders_fixture
from agnostic_market.commerce.profile import ProfileStore
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.state import PolicyContext


class SupportHarness(NamedTuple):
    """Everything a support-family test asserts against, by name."""

    engine: ReasoningEngine
    store: OrderStore
    verification: VerificationStore
    otp: OtpProvider
    profile: ProfileStore
    pointer: LastOrderPointer


def build_support_engine(
    config_root: Path,
    *,
    policy: PolicyContext,
    reasoning: FakeChatModel | None = None,
    frontline: FakeChatModel | None = None,
    risk_flagged: bool = False,
    thread_id: str = "support-1",
) -> SupportHarness:
    """The production graph shape behind a ReasoningEngine, with fakes + per-test stores.

    The pointer instance is SHARED between the order_status tool and the graph (the
    split-brain rule); the neutral reasoning default clarifies — suites pass their own
    force_tool/scripted fakes.
    """
    from agnostic_market.voice.tools import build_voice_tools

    store = OrderStore(load_orders_fixture(config_root, "acme_store"))
    pointer = LastOrderPointer()
    tools = [
        wrap_readonly_tool(t, "acme_store")
        for t in build_voice_tools(store, CartStore(), pointer)
    ]
    otp = OtpProvider()
    verification = VerificationStore(otp)
    profile = ProfileStore()
    graph = build_frontline_graph(
        frontline or FakeChatModel(emit_tool_calls=False),
        tools,
        display_name="Acme Store",
        tenant_id="acme_store",
        reasoning_model=reasoning or FakeChatModel(emit_tool_calls=False),
        store=store,
        policy=policy,
        verification_store=verification,
        otp=otp,
        risk=RiskProvider(flagged=risk_flagged),
        profile_store=profile,
        pointer=pointer,
        checkpointer=build_checkpointer(),
    )
    return SupportHarness(
        ReasoningEngine(graph, thread_id=thread_id), store, verification, otp, profile, pointer
    )
