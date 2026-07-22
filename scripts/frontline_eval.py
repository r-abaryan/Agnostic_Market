"""T1 routing-quality eval — frontline under-escalation (AGENTS.md seam #1).

Run: uv run python scripts/frontline_eval.py
Structural gate only: uv run python scripts/frontline_eval.py --preflight-only
Needs ANTHROPIC_API_KEY (the acme routing model) in .env. Text-only — no voice.

Invokes the SAME graph production uses (prompt inside the graph, F1), one utterance per
turn. The MODEL is the primary escalation decider; the slim gate is a high-certainty
irreversible-only floor. Pre-registered:
  - PRIMARY BAR: escalation recall (gate OR model) >= 90% on should_escalate. Every miss
    triaged (a real answer-instead-of-escalate is a judgment gap to fix in prompt/few-shot).
  - Gate coverage REPORTED informationally (the free fast-path share) — never a mandate.
  - over-escalation on should_answer REPORTED (over-escalating is cheap, AGENTS §A2).
This is NOT an overall commerce, speech, continuation, or voice-transport certification. The
live-call #18 structural preflight runs before provider access and blocks the routing score while
the shared speech-authority or exact-STT contracts are broken. The routing section then measures
escalation QUALITY only. Exit 1 if either the structural gate or routing bar is missed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agnostic_market.agents import telemetry
from agnostic_market.agents.engine import ReasoningEngine, _TurnSpeech
from agnostic_market.agents.frontline import build_frontline_graph
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    assert_orders_have_customers,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.commerce.profile import (
    ProfileStore,
    assert_profiles_have_customers,
    load_profile_fixture,
)
from agnostic_market.commerce.spoken import caller_stated_order_id
from agnostic_market.commerce.verification import (
    OtpProvider,
    VerificationStore,
    load_verification_fixture,
)
from agnostic_market.config.loader import load_yaml_layer
from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.dtos.events import (
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnEvent,
    TurnFacts,
)
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_t1.yaml"
_SAFETY_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_safety.yaml"
_MERCHANT_ID = "acme_store"
_RECALL_BAR = 0.90

_PENDING_FIELDS = (
    "pending_placement",
    "pending_refund",
    "pending_cancel",
    "pending_return",
    "pending_profile_change",
    "pending_identity",
    "pending_request",
    "pending_ack",
)


@dataclass(frozen=True)
class AudibleObservation:
    kind: str
    text: str
    node: str | None


@dataclass(frozen=True)
class CommerceObservation:
    placed: int
    refunded: int
    returned: int
    cancelled: int
    profile_changes: int
    otp_dispatches: int
    verification_level: int
    identity_bound: bool


@dataclass(frozen=True)
class GraphObservation:
    active_flow: str | None
    pending_fields: tuple[str, ...]
    handover_destination: str | None
    interrupted: bool
    unfinished: bool
    automation_terminal: bool


@dataclass(frozen=True)
class TurnObservation:
    utterance: str
    audible: tuple[AudibleObservation, ...]
    effects: CommerceObservation
    state: GraphObservation
    model_calls: int | None
    completed_tool_calls: int


@dataclass(frozen=True)
class ScenarioObservation:
    before: CommerceObservation
    turns: tuple[TurnObservation, ...]

    @property
    def final(self) -> TurnObservation:
        if not self.turns:
            raise ValueError("scenario observation has no turns")
        return self.turns[-1]


def _audible_observation(event: TurnEvent) -> AudibleObservation:
    if isinstance(event, InterruptEvent):
        return AudibleObservation(kind=event.kind, text=event.prompt, node="__interrupt__")
    if isinstance(event, SpokenMessageEvent):
        return AudibleObservation(kind=event.kind, text=event.text, node=event.node)
    assert isinstance(event, TokenEvent)
    # TokenEvent currently carries no graph-node provenance. Record that fact rather than
    # inventing an author; the speech-contract milestone owns any runtime DTO change.
    return AudibleObservation(kind=event.kind, text=event.text, node=None)


def _commerce_observation(
    store: OrderStore,
    profile_store: ProfileStore,
    otp: OtpProvider,
    verification: VerificationStore,
    identity_store: CallerIdentityStore,
) -> CommerceObservation:
    return CommerceObservation(
        placed=store.placed_count,
        refunded=store.refund_count,
        returned=store.return_count,
        cancelled=store.cancel_count,
        profile_changes=profile_store.change_count,
        otp_dispatches=otp.dispatch_count,
        verification_level=verification.current_level(),
        identity_bound=identity_store.current() is not None,
    )


def _checkpoint_observation(engine: ReasoningEngine) -> tuple[GraphObservation, int]:
    # Evaluator-only checkpoint inspection. Adding a production introspection API solely for
    # tests would widen ReasoningEngine's application contract.
    snapshot = engine._graph.get_state({"configurable": {"thread_id": engine.thread_id}})
    values = snapshot.values
    handover = values.get("handover")
    return (
        GraphObservation(
            active_flow=values.get("active_flow"),
            pending_fields=tuple(name for name in _PENDING_FIELDS if values.get(name) is not None),
            handover_destination=getattr(handover, "destination", None),
            interrupted=bool(snapshot.interrupts),
            unfinished=bool(snapshot.next),
            automation_terminal=bool(values.get("automation_terminal", False)),
        ),
        sum(isinstance(message, ToolMessage) for message in values.get("messages", ())),
    )


async def _observe_scenario(
    engine: ReasoningEngine,
    utterances: Sequence[str],
    *,
    store: OrderStore,
    profile_store: ProfileStore,
    otp: OtpProvider,
    verification: VerificationStore,
    identity_store: CallerIdentityStore,
    model_call_count: Callable[[], int] | None = None,
) -> ScenarioObservation:
    before = _commerce_observation(store, profile_store, otp, verification, identity_store)
    turns: list[TurnObservation] = []
    for utterance in utterances:
        events = tuple([event async for event in engine.stream_turn(utterance, TurnFacts())])
        state, completed_tool_calls = _checkpoint_observation(engine)
        turns.append(
            TurnObservation(
                utterance=utterance,
                audible=tuple(_audible_observation(event) for event in events),
                effects=_commerce_observation(
                    store, profile_store, otp, verification, identity_store
                ),
                state=state,
                model_calls=model_call_count() if model_call_count is not None else None,
                completed_tool_calls=completed_tool_calls,
            )
        )
    return ScenarioObservation(before=before, turns=tuple(turns))


def _score_safety_observation(
    observation: ScenarioObservation,
    *,
    expected_effects: CommerceObservation,
    expected_state: GraphObservation,
    forbidden_spoken: Sequence[str] = (),
) -> tuple[str, ...]:
    """Pure scripted scorer; this is not live-provider semantic conformance."""
    failures: list[str] = []
    final = observation.final
    if final.effects != expected_effects:
        failures.append("authoritative effect/store observation differed")
    if final.state != expected_state:
        failures.append("next-turn graph state observation differed")
    spoken = " ".join(part.text for turn in observation.turns for part in turn.audible).casefold()
    for phrase in forbidden_spoken:
        if phrase.casefold() in spoken:
            failures.append(f"forbidden scripted speech reached caller: {phrase!r}")
    return tuple(failures)


_THREAD_SEQ = 0

# The gated flows' model-facing tools: any of these appearing in the turn's messages means
# the turn LEFT the frontline and a flow's model ran — an escalation, even when the flow
# then ended in a terminal decline that clears its own state (see docstring below).
_FLOW_TOOLS = frozenset(
    {
        "propose_refund",
        "propose_cancel",
        "propose_return",  # support (returns: Group C)
        "propose_profile_change",
        "leave_support",  # support (profile: Group C)
        "add_to_cart",
        "remove_from_cart",
        "set_quantity",
        "review_cart",
        "buy_now",
        "go_to_checkout",
        "leave_cart",  # cart (Group B; view_cart is a FRONTLINE read, not here)
        "propose_identity",
        "leave_identity",  # identity (P7; list_orders is a FRONTLINE read)
    }
)


def _structural_preflight_failures(data: dict) -> tuple[str, ...]:
    """Zero-network gates for failures the old routing-only evaluator could not observe."""
    failures: list[str] = []

    completed_speech = _TurnSpeech(frozenset())
    event = completed_speech.feed(
        AIMessage(content="untrusted transactional model text", id="eval-message"),
        {"langgraph_node": "eval_unapproved_transactional_node"},
    )
    if event is not None:
        failures.append("completed unapproved transactional model text reached caller speech")

    orphan_speech = _TurnSpeech(frozenset())
    orphan_speech.feed(
        AIMessageChunk(content="untrusted transactional model text", id="eval-orphan"),
        {"langgraph_node": "eval_unapproved_transactional_node"},
    )
    if list(orphan_speech.flush()):
        failures.append("orphaned unapproved transactional model text reached caller speech")

    order_reference = data["order_reference"]
    expected = order_reference["expected_order_id"]
    for utterance in order_reference["accepted_labelled_stt"]:
        if caller_stated_order_id(utterance, expected) != expected:
            failures.append(f"labelled STT order reference was rejected: {utterance!r}")
    for case in order_reference["rejected_stt"]:
        if caller_stated_order_id(case["utterance"], case["proposed_order_id"]) is not None:
            failures.append(f"weak/conflicting STT order reference was accepted: {case!r}")
    return tuple(failures)


async def _outcome(graph, utterance: str) -> str | None:
    """How the turn left the frontline, else None (answered).

    3b semantics: a checkout/support-destination handover no longer ends at a spoken
    deferral — it ENTERS the flow (clearing the handover signal), so escalation shows up
    as `active_flow`/`pending_*` in the output, or as a paused confirm interrupt.
    3c-follow-up semantics: a flow can also run to a TERMINAL decline (return-first,
    amount gate, ineligible cancel) that clears `active_flow` before END — then the only
    trace is the flow model's tool activity in the turn's messages, so that counts as
    'flow' too. Returns 'gate'/'model' (handover survived in state) or 'flow' (a gated
    flow ran). Each utterance runs on a fresh thread (interrupts need a checkpointer).
    """
    global _THREAD_SEQ
    _THREAD_SEQ += 1
    config = {"configurable": {"thread_id": f"eval-{_THREAD_SEQ}"}}
    out = await graph.ainvoke({"messages": [HumanMessage(utterance)]}, config)
    handover = out.get("handover")
    if handover is not None:
        return handover.source
    if (
        out.get("active_flow") is not None
        or out.get("pending_placement") is not None
        or graph.get_state(config).interrupts
    ):
        return "flow"
    for msg in out["messages"]:
        if isinstance(msg, AIMessage) and any(
            call["name"] in _FLOW_TOOLS for call in msg.tool_calls
        ):
            return "flow"
    return None


async def _run(*, preflight_only: bool = False) -> int:
    safety_data = load_yaml_layer(_SAFETY_EVAL_PATH)
    failures = _structural_preflight_failures(safety_data)
    print("[coverage] structural speech authority + labelled STT order references")
    if failures:
        for failure in failures:
            print(f"    STRUCTURAL FAILURE: {failure}")
        print("\nFRONTLINE EVAL BLOCKED - structural safety preflight failed. [FAIL]")
        return 1
    print("[structural_preflight] all registered contracts passed. [PASS]")
    if preflight_only:
        return 0

    load_dotenv()
    # Eval runs must not pollute the LIVE telemetry dataset (classifier data): redirect
    # this process's sink to a sibling eval file (same local-only, gitignored dir).
    telemetry._TELEMETRY_PATH = telemetry._TELEMETRY_PATH.with_name("frontline_eval.jsonl")
    secrets = EnvSecretResolver()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    resolved = ConfigRegistry(_CONFIG_ROOT).load().get(_MERCHANT_ID)
    chat_model = LLMGateway(credentials, secrets).chat_model(resolved.config.llm.routing)
    store = OrderStore(load_orders_fixture(_CONFIG_ROOT, _MERCHANT_ID))
    cart_store = CartStore()
    customers_fixture = load_customers_fixture(_CONFIG_ROOT, _MERCHANT_ID)
    profile_fixture = load_profile_fixture(_CONFIG_ROOT, _MERCHANT_ID)
    assert_orders_have_customers(store.fixture, customers_fixture)
    assert_profiles_have_customers(profile_fixture, customers_fixture)
    customers = CustomerDirectory(customers_fixture)
    profile_store = ProfileStore(profile_fixture)
    identity_store = CallerIdentityStore()
    config = resolved.config
    policy = config.policies.to_policy_context()
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    verification_fixture = load_verification_fixture(_CONFIG_ROOT, _MERCHANT_ID)
    otp = OtpProvider(valid_code=verification_fixture.otp_code)
    verification = VerificationStore(otp)
    caller_context = CallerContext(
        verification_store=verification,
        cart_store=cart_store,
        recent_orders=recent_orders,
        identity_store=identity_store,
        order_store=store,
    )
    tools = [
        wrap_readonly_tool(t, _MERCHANT_ID)
        for t in build_voice_tools(store, cart_store, recent_orders, identity_store, customers)
    ]
    graph = build_frontline_graph(
        chat_model,
        tools,
        display_name=config.display_name,
        tenant_id=config.merchant_id,
        # The T1 eval never enters checkout (text-only escalation probes), but the graph
        # compiles as ONE shape — same construction path as production (F1 discipline).
        reasoning_model=LLMGateway(credentials, secrets).chat_model(config.llm.reasoning),
        store=store,
        otp=otp,
        verification_store=verification,
        cart_store=cart_store,  # SAME instance as view_cart (no split-brain)
        recent_orders=recent_orders,
        identity_store=identity_store,  # SAME instance as the tools' gate (P7, no split-brain)
        customers=customers,
        profile_store=profile_store,
        # The ONE config->runtime policy mapping — identical to production (F1). The old
        # hand-built copy here had drifted (it silently omitted spoken_policy_extra).
        policy=policy,
        transition_principal=caller_context.transition_principal,
        principal_state_will_be_discarded=caller_context.has_discardable_state,
        # An utterance that reaches the confirm readback pauses at an interrupt, which
        # needs a checkpointer even in the text eval (fresh thread per utterance).
        checkpointer=InMemorySaver(),
    )
    data = load_yaml_layer(_EVAL_PATH)

    # --- PRIMARY: escalation recall (gate OR model OR checkout-flow entry) ---
    escalate = data["should_escalate"]
    by_gate = by_model = by_flow = 0
    misses: list[str] = []
    for utt in escalate:
        source = await _outcome(graph, utt)
        if source == "gate":
            by_gate += 1
        elif source == "model":
            by_model += 1
        elif source == "flow":
            by_flow += 1
        else:
            misses.append(utt)
    escalated = by_gate + by_model + by_flow
    recall = escalated / len(escalate)
    print(f"[should_escalate] recall: {escalated}/{len(escalate)} ({recall:.0%})")
    print(
        f"    gate caught {by_gate} (informational fast-path) | model caught {by_model} "
        f"| entered a gated flow {by_flow}"
    )
    for miss in misses:
        print(f"    MISS (answered instead of escalated - triage prompt/few-shot): {miss!r}")

    # --- INFORMATIONAL: over-escalation on should_answer ---
    answer = data["should_answer"]
    over = []
    for utt in answer:
        source = await _outcome(graph, utt)
        if source is not None:
            over.append(f"{utt!r} (source={source})")
    print(f"[should_answer] over-escalation: {len(over)}/{len(answer)} (informational)")
    for o in over:
        print(f"    OVER-ESCALATED: {o}")

    if recall < _RECALL_BAR:
        print(
            f"\nT1 ROUTING EVAL FAILED - escalation recall {recall:.0%} < {_RECALL_BAR:.0%}. [FAIL]"
        )
        return 1
    print(
        f"\nT1 ROUTING eval passed: escalation recall {recall:.0%} "
        f">= {_RECALL_BAR:.0%}. [ROUTING-ONLY PASS]"
    )
    print(
        "Coverage limit: no multi-turn effect/speech/next-state or LiveKit transport "
        "certification was performed."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(preflight_only=args.preflight_only)))
