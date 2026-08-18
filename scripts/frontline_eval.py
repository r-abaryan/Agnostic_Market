"""T1 routing-quality eval — frontline under-escalation (AGENTS.md seam #1).

Run: uv run python scripts/frontline_eval.py
Structural gate only: uv run python scripts/frontline_eval.py --preflight-only
Order-target semantic corpus: uv run python scripts/frontline_eval.py --order-target-eval
Typed semantic-router corpus: uv run python scripts/frontline_eval.py --semantic-routing-eval
Offline 6E gate: uv run python scripts/frontline_eval.py --recovery-certification offline
Live retry gate: uv run python scripts/frontline_eval.py --recovery-certification live
The routing eval needs the merchant model credentials; the live recovery tier needs every
configured provider credential. The offline recovery tier uses a test-only resolver and no
upstream access. Text-only — no voice.

Invokes the SAME graph production uses (prompt inside the graph, F1), one utterance per
turn. The MODEL is the primary escalation decider; the slim gate is a high-certainty
irreversible-only floor. Pre-registered:
  - PRIMARY BAR: escalation recall (gate OR model) >= 90% on should_escalate. Every miss
    triaged (a real answer-instead-of-escalate is a judgment gap to fix in prompt/few-shot).
  - Gate coverage REPORTED informationally (the free fast-path share) — never a mandate.
  - over-escalation on should_answer REPORTED (over-escalating is cheap, AGENTS §A2).
The default routing mode is NOT an overall commerce, speech, continuation, or voice-transport
certification. The live-call #18 structural preflight runs before provider access and blocks the
routing score while the shared speech-authority or exact-STT contracts are broken. The routing
section then measures escalation QUALITY only. The separate 6E modes certify provider transport
failure against the production graph and existing effect/speech/state observer. Exit 1 when the
selected gate fails.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from agnostic_market.agents import telemetry
from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.engine import ReasoningEngine, _TurnSpeech, build_checkpointer
from agnostic_market.agents.frontline import (
    FRONTLINE_SPEAKABLE_NODES,
    MODEL_SPEECH_NODES,
    NON_SPEAKING_MODEL_NODES,
    build_frontline_graph,
)
from agnostic_market.agents.frontline.prompt import ORDER_TARGET_PROPOSAL_PROMPT
from agnostic_market.agents.frontline.read_flow import (
    ANSWER_CLARIFY_NODE,
    ANSWER_RESPONSE_NODE,
    ANSWER_UNSUPPORTED_NODE,
    CATALOG_RESPONSE_NODE,
    ORDER_STATUS_TARGET_PROPOSE_NODE,
)
from agnostic_market.agents.recovery import RECOVERY_NODE_NAME, TURN_FALLBACK_LINE
from agnostic_market.agents.routing import (
    ProviderCallOutcome,
    RoutingAttempt,
    SemanticRouter,
    project_routing_context,
    resolve_route,
)
from agnostic_market.agents.tooling import wrap_readonly_tool
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectory,
    assert_orders_have_customers,
    load_customers_fixture,
)
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext, load_orders_fixture
from agnostic_market.commerce.payment_instruments import (
    PaymentInstrumentDirectory,
    assert_payment_instruments_have_customers,
    load_payment_instruments_fixture,
)
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
from agnostic_market.dtos.config import MerchantConfig, ProviderModel
from agnostic_market.dtos.events import (
    CommittedTurn,
    InterruptEvent,
    SpokenMessageEvent,
    TokenEvent,
    TurnEvent,
    TurnFacts,
)
from agnostic_market.dtos.llm import ProviderCredentialsConfig, StructuredOutputMethod
from agnostic_market.dtos.orchestration import (
    AnswerQuestion,
    ChangeProfile,
    FocusedOrderSet,
    IntentRequest,
    ListOrders,
    ModifyCart,
    OrderTargetProposal,
    RecentOrderSet,
    RouteDecision,
    RouteProposal,
    RoutingContext,
    RoutingFailure,
    SearchCatalog,
    VerifyOrderStatus,
)
from agnostic_market.dtos.state import ReasoningState, open_active_invocation
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import load_conformance_targets
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.voice.context import CallerContext
from agnostic_market.voice.tools import build_voice_tools

if __package__:
    from .transport_fault_proxy import (
        PROVIDER_TRANSPORT_CONTRACTS,
        TRANSPORT_CONTRACT_VERSION,
        FaultKind,
        FaultMode,
        ProxyAttempt,
        TransportFaultProxy,
    )
else:
    from transport_fault_proxy import (
        PROVIDER_TRANSPORT_CONTRACTS,
        TRANSPORT_CONTRACT_VERSION,
        FaultKind,
        FaultMode,
        ProxyAttempt,
        TransportFaultProxy,
    )

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_t1.yaml"
_SAFETY_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_safety.yaml"
_READ_OWNER_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_read_owners.yaml"
_READ_OWNER_EVAL_SCHEMA_VERSION = "1"
_ORDER_TARGET_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_order_targets.yaml"
_ORDER_TARGET_EVAL_SCHEMA_VERSION = "1"
_SEMANTIC_ROUTE_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_semantic_routes.yaml"
_SEMANTIC_ROUTE_CORPUS_SCHEMA_VERSION = "3"
_SEMANTIC_ROUTE_REPORT_SCHEMA_VERSION = "4"
_SEMANTIC_ROUTE_REPORT_PATH = _CONFIG_ROOT / "telemetry" / "semantic_routing_report.json"
_SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS = 3
_SEMANTIC_ROUTE_MIN_ACCEPTANCE_EXACT_RATE = 0.95
_SEMANTIC_ROUTE_P50_QUANTILE = 0.50
_SEMANTIC_ROUTE_P95_QUANTILE = 0.95
_MERCHANT_ID = "acme_store"
_RECALL_BAR = 0.90
_TRANSPORT_REPORT_PATHS = {
    "offline": _CONFIG_ROOT / "telemetry" / "transport_recovery_offline_report.json",
    "live": _CONFIG_ROOT / "telemetry" / "transport_recovery_live_report.json",
}
_TRANSPORT_REQUEST_CEILING = 8
_TRANSPORT_FAULT_WINDOW_SECONDS = 30.0
_TRANSPORT_UPSTREAM_TIMEOUT_SECONDS = 60.0
_TRANSPORT_SCENARIO_TIMEOUT_SECONDS = 90.0

_AUTOMATION_CHANNELS = (
    "pending_placement",
    "pending_refund",
    "pending_cancel",
    "pending_return",
    "pending_profile_change",
    "pending_identity",
    "active_invocation",
    "pending_ack",
    "pending_clarification",
    "clarification_progress",
)


class ReadOwnerEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    owner: Literal["catalog", "answer"]
    utterance: str
    expected_disposition: Literal["answer", "clarify", "unsupported"]
    query: str | None = None
    topic: Literal["policy", "general"] | None = None
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _owner_matches_request(self) -> ReadOwnerEvalCase:
        text_fields = (self.case_id, self.utterance, *self.required_facts, *self.forbidden_facts)
        if any(not value.strip() for value in text_fields):
            raise ValueError("read-owner eval text fields must be non-empty")
        if self.owner == "catalog":
            if self.query is None or not self.query.strip() or self.topic is not None:
                raise ValueError("catalog cases require only a non-empty query")
            if self.expected_disposition != "answer":
                raise ValueError("catalog query clarification is a structural, not semantic, case")
        elif self.topic is None or self.query is not None:
            raise ValueError("answer cases require only a closed topic")
        return self


class ReadOwnerEvalCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    cases: tuple[ReadOwnerEvalCase, ...]

    @model_validator(mode="after")
    def _case_ids_are_unique(self) -> ReadOwnerEvalCorpus:
        if self.schema_version != _READ_OWNER_EVAL_SCHEMA_VERSION:
            raise ValueError("unsupported read-owner eval schema version")
        ids = [case.case_id for case in self.cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("read-owner eval requires non-empty unique case ids")
        return self


class OrderTargetEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    utterance: str
    expected_relationship: Literal["single", "plural", "alternative", "ambiguous"]
    expected_order_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _expected_proposal_is_valid(self) -> OrderTargetEvalCase:
        if not self.case_id.strip() or not self.utterance.strip():
            raise ValueError("order-target eval text fields must be non-empty")
        OrderTargetProposal(
            relationship=self.expected_relationship,
            order_refs=self.expected_order_refs,
        )
        return self


class OrderTargetEvalCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    cases: tuple[OrderTargetEvalCase, ...]

    @model_validator(mode="after")
    def _case_ids_are_unique(self) -> OrderTargetEvalCorpus:
        if self.schema_version != _ORDER_TARGET_EVAL_SCHEMA_VERSION:
            raise ValueError("unsupported order-target eval schema version")
        ids = [case.case_id for case in self.cases]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("order-target eval requires non-empty unique case ids")
        return self


SemanticRouteScenarioClass = Literal[
    "direct",
    "continuation",
    "clarification",
    "adversarial",
    "counterfactual",
    "asr_like",
]
SemanticRouteRiskClass = Literal["critical", "standard"]
SemanticRouteRiskDomain = Literal[
    "ordinary_intent",
    "commerce_read",
    "account_read",
    "account_control",
    "commerce_effect",
    "compliance_control",
]
SemanticRouteEvaluationSplit = Literal["development", "acceptance"]
SemanticRouteGate = Literal["diagnostic", "shadow", "cutover"]
SemanticRouteDisposition = Literal[
    "exact",
    "conservative_clarification",
    "closed_failure",
    "unsafe_executable_misroute",
]


class SemanticRouteEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    scenario_class: SemanticRouteScenarioClass
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    evaluation_split: SemanticRouteEvaluationSplit
    context: RoutingContext
    expected: RouteDecision

    @model_validator(mode="after")
    def expected_route_is_executable_in_context(self) -> SemanticRouteEvalCase:
        if isinstance(resolve_route(self.context, self.expected), RoutingFailure):
            raise ValueError("semantic route ground truth must be executable in its context")
        return self


class ProjectedSemanticRouteEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    scenario_class: SemanticRouteScenarioClass
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    evaluation_split: SemanticRouteEvaluationSplit
    turn: CommittedTurn
    expected_context: RoutingContext
    expected: RouteDecision

    @model_validator(mode="after")
    def projection_ground_truth_is_coherent(self) -> ProjectedSemanticRouteEvalCase:
        if self.turn.message_id is None:
            raise ValueError("projected semantic-route case requires a committed-turn id")
        if self.turn.text != self.expected_context.utterance:
            raise ValueError("projected turn text must equal its expected context utterance")
        if isinstance(resolve_route(self.expected_context, self.expected), RoutingFailure):
            raise ValueError("projected route ground truth must be executable")
        return self


class SemanticRouteEvalCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    projected_case: ProjectedSemanticRouteEvalCase
    cases: tuple[SemanticRouteEvalCase, ...]

    @model_validator(mode="after")
    def corpus_is_current_and_unique(self) -> SemanticRouteEvalCorpus:
        if self.schema_version != _SEMANTIC_ROUTE_CORPUS_SCHEMA_VERSION:
            raise ValueError("unsupported semantic-route eval schema version")
        ids = [self.projected_case.case_id, *(case.case_id for case in self.cases)]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("semantic-route eval requires non-empty unique case ids")
        splits = {
            self.projected_case.evaluation_split,
            *(case.evaluation_split for case in self.cases),
        }
        if splits != {"development", "acceptance"}:
            raise ValueError("semantic-route eval requires development and acceptance splits")
        if not any(
            case.evaluation_split == "acceptance" and case.risk_class == "critical"
            for case in self.cases
        ):
            raise ValueError("semantic-route eval requires critical acceptance cases")
        return self


@dataclass(frozen=True)
class _RoutingEvalSelection:
    config: MerchantConfig
    gateway: LLMGateway
    response_model: BaseChatModel
    response_structured_output_method: StructuredOutputMethod
    routing_model: BaseChatModel
    routing_structured_output_method: StructuredOutputMethod


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
    automation_channels: tuple[str, ...]
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
    admitted_user_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioObservation:
    before: CommerceObservation
    turns: tuple[TurnObservation, ...]

    @property
    def final(self) -> TurnObservation:
        if not self.turns:
            raise ValueError("scenario observation has no turns")
        return self.turns[-1]


@dataclass(frozen=True)
class EvalRuntime:
    graph: CompiledStateGraph
    capability_registry: CapabilityRegistry
    engine: ReasoningEngine
    store: OrderStore
    cart_store: CartStore
    profile_store: ProfileStore
    otp: OtpProvider
    verification: VerificationStore
    identity_store: CallerIdentityStore
    recent_orders: RecentOrderContext
    caller_context: CallerContext


@dataclass(frozen=True)
class TransportOwnerScenario:
    scenario_id: str
    owner: Literal["frontline", "cart", "identity", "support"]
    origin_node: str
    utterance: str
    initial_request: IntentRequest | None = None
    offline_faults: tuple[FaultKind, ...] = (FaultKind.PRE_RESPONSE_DISCONNECT,)
    live_faults: tuple[FaultKind, ...] = ()


@dataclass(frozen=True)
class TransportCaseResult:
    case_id: str
    tier: Literal["offline", "live"]
    provider: str
    model: str
    owner: str
    fault_kind: str
    passed: bool
    failures: tuple[str, ...]
    attempts: tuple[dict[str, object], ...]
    audible: tuple[dict[str, object], ...]
    before: dict[str, object]
    after: dict[str, object]
    state: dict[str, object]
    turn_failed: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SemanticRouteCaseResult:
    case_id: str
    scenario_class: str
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    evaluation_split: SemanticRouteEvaluationSplit
    expected: RouteDecision
    attempt: RoutingAttempt

    @property
    def disposition(self) -> SemanticRouteDisposition:
        return _semantic_route_disposition(self.expected, self.attempt.resolution)


@dataclass(frozen=True)
class ProjectedSemanticRouteCaseResult:
    case_id: str
    scenario_class: str
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    evaluation_split: SemanticRouteEvaluationSplit
    expected_context: RoutingContext
    actual_context: RoutingContext | RoutingFailure

    @property
    def passed(self) -> bool:
        return self.actual_context == self.expected_context


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


def _checkpoint_observation(
    engine: ReasoningEngine,
) -> tuple[GraphObservation, int, tuple[str, ...]]:
    # Evaluator-only checkpoint inspection. Adding a production introspection API solely for
    # tests would widen ReasoningEngine's application contract.
    snapshot = engine._graph.get_state({"configurable": {"thread_id": engine.thread_id}})
    values = snapshot.values
    handover = values.get("handover")
    return (
        GraphObservation(
            active_flow=values.get("active_flow"),
            automation_channels=tuple(
                name for name in _AUTOMATION_CHANNELS if values.get(name) is not None
            ),
            handover_destination=getattr(handover, "destination", None),
            interrupted=bool(snapshot.interrupts),
            unfinished=bool(snapshot.next),
            automation_terminal=bool(values.get("automation_terminal", False)),
        ),
        sum(isinstance(message, ToolMessage) for message in values.get("messages", ())),
        tuple(
            str(message.content)
            for message in values.get("messages", ())
            if isinstance(message, HumanMessage)
        ),
    )


def _build_eval_runtime(
    config: MerchantConfig,
    routing_model: BaseChatModel,
    reasoning_model: BaseChatModel,
    *,
    thread_id: str,
    structured_output_method: StructuredOutputMethod,
) -> EvalRuntime:
    store = OrderStore(load_orders_fixture(_CONFIG_ROOT, config.merchant_id))
    cart_store = CartStore()
    customers_fixture = load_customers_fixture(_CONFIG_ROOT, config.merchant_id)
    payment_instruments_fixture = load_payment_instruments_fixture(_CONFIG_ROOT, config.merchant_id)
    profile_fixture = load_profile_fixture(_CONFIG_ROOT, config.merchant_id)
    assert_orders_have_customers(store.fixture, customers_fixture)
    assert_profiles_have_customers(profile_fixture, customers_fixture)
    assert_payment_instruments_have_customers(payment_instruments_fixture, customers_fixture)
    customers = CustomerDirectory(customers_fixture)
    payment_instruments = PaymentInstrumentDirectory(payment_instruments_fixture)
    profile_store = ProfileStore(profile_fixture)
    identity_store = CallerIdentityStore()
    policy = config.policies.to_policy_context()
    recent_orders = RecentOrderContext(max_refs=policy.cancel_batch_max)
    verification_fixture = load_verification_fixture(_CONFIG_ROOT, config.merchant_id)
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
        wrap_readonly_tool(t, config.merchant_id)
        for t in build_voice_tools(
            store,
            cart_store,
            recent_orders,
            identity_store,
            customers,
        )
    ]
    assembly = build_frontline_graph(
        routing_model,
        tools,
        display_name=config.display_name,
        tenant_id=config.merchant_id,
        reasoning_model=reasoning_model,
        store=store,
        otp=otp,
        verification_store=verification,
        cart_store=cart_store,
        recent_orders=recent_orders,
        identity_store=identity_store,
        customers=customers,
        payment_instruments=payment_instruments,
        profile_store=profile_store,
        policy=policy,
        lifecycle=caller_context,
        structured_output_method=structured_output_method,
        caller_audible_model_text_max_chars=(config.runtime.caller_audible_model_text_max_chars),
        checkpointer=build_checkpointer(),
    )
    engine = ReasoningEngine(
        assembly.graph,
        thread_id=thread_id,
        cancellation_quiescence_timeout_seconds=(
            config.runtime.cancellation_quiescence_timeout_seconds
        ),
        lifecycle=caller_context,
    )
    caller_context.attach_engine(engine)
    return EvalRuntime(
        graph=assembly.graph,
        capability_registry=assembly.capability_registry,
        engine=engine,
        store=store,
        cart_store=cart_store,
        profile_store=profile_store,
        otp=otp,
        verification=verification,
        identity_store=identity_store,
        recent_orders=recent_orders,
        caller_context=caller_context,
    )


async def _observe_scenario(
    engine: ReasoningEngine,
    utterances: Sequence[str],
    *,
    scenario_key: str,
    store: OrderStore,
    profile_store: ProfileStore,
    otp: OtpProvider,
    verification: VerificationStore,
    identity_store: CallerIdentityStore,
    model_call_count: Callable[[], int] | None = None,
) -> ScenarioObservation:
    before = _commerce_observation(store, profile_store, otp, verification, identity_store)
    turns: list[TurnObservation] = []
    for turn_index, utterance in enumerate(utterances, start=1):
        turn = CommittedTurn(
            text=utterance,
            message_id=f"{scenario_key}:turn:{turn_index}",
        )
        events = tuple([event async for event in engine.stream_turn(turn, TurnFacts())])
        state, completed_tool_calls, admitted_user_messages = _checkpoint_observation(engine)
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
                admitted_user_messages=admitted_user_messages,
            )
        )
    return ScenarioObservation(before=before, turns=tuple(turns))


def _score_safety_observation(
    observation: ScenarioObservation,
    *,
    expected_effects: CommerceObservation,
    expected_state: GraphObservation,
    forbidden_spoken: Sequence[str] = (),
    expected_admitted_user_messages: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Pure scripted scorer; this is not live-provider semantic conformance."""
    failures: list[str] = []
    final = observation.final
    if final.effects != expected_effects:
        failures.append("authoritative effect/store observation differed")
    if final.state != expected_state:
        failures.append("next-turn graph state observation differed")
    if expected_admitted_user_messages is not None and final.admitted_user_messages != tuple(
        expected_admitted_user_messages
    ):
        failures.append("admitted caller-message history differed")
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
        "request_support_clarification",
        "leave_support",  # support (profile: Group C)
        "add_to_cart",
        "remove_from_cart",
        "set_quantity",
        "review_cart",
        "buy_now",
        "go_to_checkout",
        "request_cart_clarification",
        "leave_cart",  # cart (Group B; view_cart is a FRONTLINE read, not here)
        "propose_identity",
        "request_identity_contact",
        "leave_identity",  # identity (P7; list_orders is a FRONTLINE read)
    }
)


def _speech_authority_failures() -> tuple[str, ...]:
    """Exercise production speech-source policy against every transactional model node."""
    failures: list[str] = []

    for node in sorted(NON_SPEAKING_MODEL_NODES):
        meta = {"langgraph_node": node}
        completed_speech = _TurnSpeech(frozenset(), MODEL_SPEECH_NODES)
        event = completed_speech.feed(
            AIMessage(content="untrusted transactional model text", id=f"eval-{node}"),
            meta,
        )
        if event is not None:
            failures.append(f"completed transactional model text reached caller speech: {node!r}")

        orphan_speech = _TurnSpeech(frozenset(), MODEL_SPEECH_NODES)
        orphan_speech.feed(
            AIMessageChunk(content="untrusted transactional model text", id=f"orphan-{node}"),
            meta,
        )
        if list(orphan_speech.flush()):
            failures.append(f"orphaned transactional model text reached caller speech: {node!r}")

    # Positive controls keep the gate honest: every approved model source and one code-owned
    # line remain audible while unknown and missing provenance fail closed.
    for node in sorted(MODEL_SPEECH_NODES):
        model_control = _TurnSpeech(frozenset(), MODEL_SPEECH_NODES)
        if not isinstance(
            model_control.feed(
                AIMessage(content="approved model control", id=f"eval-approved-{node}"),
                {"langgraph_node": node},
            ),
            TokenEvent,
        ):
            failures.append(f"approved model text was not caller-speakable: {node!r}")

        orphan_control = _TurnSpeech(frozenset(), MODEL_SPEECH_NODES)
        orphan_control.feed(
            AIMessageChunk(content="approved orphan control", id=f"orphan-approved-{node}"),
            {"langgraph_node": node},
        )
        if len(list(orphan_control.flush())) != 1:
            failures.append(f"approved orphaned model text was not caller-speakable: {node!r}")

    code_node = "read_render"
    code_control = _TurnSpeech(FRONTLINE_SPEAKABLE_NODES, MODEL_SPEECH_NODES)
    if not isinstance(
        code_control.feed(
            AIMessage(content="approved code control", id="eval-code"),
            {"langgraph_node": code_node},
        ),
        SpokenMessageEvent,
    ):
        failures.append("approved code-authored text was not caller-speakable")

    for label, meta in (
        ("unknown", {"langgraph_node": "eval_unknown_node"}),
        ("missing", {}),
    ):
        denied = _TurnSpeech(frozenset(), MODEL_SPEECH_NODES)
        if denied.feed(AIMessage(content="untrusted", id=f"eval-{label}"), meta) is not None:
            failures.append(f"{label} model speech provenance did not fail closed")
    return tuple(failures)


def _order_reference_failures(data: dict) -> tuple[str, ...]:
    """Exercise strong-labelled order references without treating bare digits as authority."""
    failures: list[str] = []

    order_reference = data["order_reference"]
    expected = order_reference["expected_order_id"]
    for utterance in order_reference["accepted_labelled_stt"]:
        if caller_stated_order_id(utterance, expected) != expected:
            failures.append(f"labelled STT order reference was rejected: {utterance!r}")
    for case in order_reference["rejected_stt"]:
        if caller_stated_order_id(case["utterance"], case["proposed_order_id"]) is not None:
            failures.append(f"weak/conflicting STT order reference was accepted: {case!r}")
    return tuple(failures)


def _structural_preflight_failures(data: dict) -> tuple[str, ...]:
    """Aggregate the independently staged zero-network structural safety gates."""
    return _speech_authority_failures() + _order_reference_failures(data)


def _load_read_owner_corpus(path: Path = _READ_OWNER_EVAL_PATH) -> ReadOwnerEvalCorpus:
    return ReadOwnerEvalCorpus.model_validate(load_yaml_layer(path))


def _load_order_target_corpus(path: Path = _ORDER_TARGET_EVAL_PATH) -> OrderTargetEvalCorpus:
    return OrderTargetEvalCorpus.model_validate(load_yaml_layer(path))


def _load_semantic_route_corpus(
    path: Path = _SEMANTIC_ROUTE_EVAL_PATH,
) -> SemanticRouteEvalCorpus:
    return SemanticRouteEvalCorpus.model_validate(load_yaml_layer(path))


def _load_routing_eval_selection() -> _RoutingEvalSelection:
    load_dotenv()
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    config = ConfigRegistry(_CONFIG_ROOT).load().get(_MERCHANT_ID).config
    gateway = LLMGateway(credentials, EnvSecretResolver())
    return _RoutingEvalSelection(
        config=config,
        gateway=gateway,
        response_model=gateway.chat_model(config.llm.response),
        response_structured_output_method=gateway.structured_output_method(config.llm.response),
        routing_model=gateway.chat_model(config.llm.routing),
        routing_structured_output_method=gateway.structured_output_method(config.llm.routing),
    )


def _score_order_target_output(
    case: OrderTargetEvalCase,
    proposal: OrderTargetProposal,
) -> tuple[str, ...]:
    failures: list[str] = []
    if proposal.relationship != case.expected_relationship:
        failures.append(f"expected {case.expected_relationship}, got {proposal.relationship}")
    expected_refs = tuple(ref.casefold() for ref in case.expected_order_refs)
    actual_refs = tuple(ref.casefold() for ref in proposal.order_refs)
    if actual_refs != expected_refs:
        failures.append(f"expected refs {expected_refs!r}, got {actual_refs!r}")
    return tuple(failures)


async def _run_order_target_cases(
    chat_model: BaseChatModel,
    structured_output_method: StructuredOutputMethod,
    corpus: OrderTargetEvalCorpus,
) -> dict[str, tuple[str, ...]]:
    structured = chat_model.with_structured_output(
        OrderTargetProposal,
        method=structured_output_method,
    )
    failures: dict[str, tuple[str, ...]] = {}
    for case in corpus.cases:
        try:
            result = await structured.ainvoke(
                [SystemMessage(ORDER_TARGET_PROPOSAL_PROMPT), HumanMessage(case.utterance)]
            )
        except (OutputParserException, ValidationError) as exc:
            failures[case.case_id] = (f"output failed schema ({type(exc).__name__})",)
            continue
        if not isinstance(result, OrderTargetProposal):
            failures[case.case_id] = (f"expected OrderTargetProposal, got {type(result).__name__}",)
            continue
        case_failures = _score_order_target_output(case, result)
        if case_failures:
            failures[case.case_id] = case_failures
    return failures


async def _evaluate_order_targets(
    chat_model: BaseChatModel,
    structured_output_method: StructuredOutputMethod,
) -> dict[str, tuple[str, ...]]:
    corpus = _load_order_target_corpus()
    failures = await _run_order_target_cases(
        chat_model,
        structured_output_method,
        corpus,
    )
    print(
        f"[order_targets] semantic cases: {len(corpus.cases) - len(failures)}/{len(corpus.cases)}"
    )
    for case_id, case_failures in failures.items():
        for failure in case_failures:
            print(f"    {case_id}: {failure}")
    return failures


async def _run_order_target_eval() -> int:
    selection = _load_routing_eval_selection()
    failures = await _evaluate_order_targets(
        selection.response_model,
        selection.response_structured_output_method,
    )
    if failures:
        print("\nFRONTLINE ORDER-TARGET EVAL FAILED. [FAIL]")
        return 1
    print("\nFRONTLINE ORDER-TARGET EVAL PASSED. [SEMANTIC-QUALITY PASS]")
    print(
        "Coverage limit: text-only structured extraction; no graph, authorization, effect, or STT."
    )
    return 0


async def _run_semantic_route_cases(
    router: SemanticRouter,
    corpus: SemanticRouteEvalCorpus,
) -> tuple[SemanticRouteCaseResult, ...]:
    results: list[SemanticRouteCaseResult] = []
    for case in corpus.cases:
        results.append(
            SemanticRouteCaseResult(
                case_id=case.case_id,
                scenario_class=case.scenario_class,
                risk_class=case.risk_class,
                risk_domain=case.risk_domain,
                evaluation_split=case.evaluation_split,
                expected=case.expected,
                attempt=await router.route(case.context),
            )
        )
    return tuple(results)


_ROUTE_SIGNATURE_DISCRIMINATORS = frozenset(
    {
        "answer_topic",
        "list_scope",
        "cart_operation",
        "profile_field",
        "order_status_selector",
    }
)
_ACTUAL_ROUTE_PROPOSAL_DISCRIMINATORS = frozenset(RouteProposal.model_fields) - {
    "decision",
    "capability",
    "clarification_reason",
}
if _ACTUAL_ROUTE_PROPOSAL_DISCRIMINATORS != _ROUTE_SIGNATURE_DISCRIMINATORS:
    raise RuntimeError(
        "semantic-route report must explicitly classify every retained route discriminator"
    )


def _semantic_route_disposition(
    expected: RouteDecision,
    actual: RouteDecision | RoutingFailure,
) -> SemanticRouteDisposition:
    if actual == expected:
        return "exact"
    if isinstance(actual, RoutingFailure):
        return "closed_failure"
    if actual.decision == "clarify":
        return "conservative_clarification"
    if actual.decision in {"direct", "continue"}:
        return "unsafe_executable_misroute"
    raise AssertionError(f"unclassified route decision {actual.decision!r}")


def _route_signature(
    resolution: RouteDecision | RoutingFailure,
) -> dict[str, str | None]:
    signature: dict[str, str | None] = {
        "decision": None,
        "capability": None,
        "clarification_reason": None,
        "answer_topic": None,
        "list_scope": None,
        "cart_operation": None,
        "profile_field": None,
        "order_status_selector": None,
        "failure_reason": None,
    }
    if isinstance(resolution, RoutingFailure):
        signature["decision"] = "routing_failure"
        signature["failure_reason"] = resolution.reason
        return signature

    signature["decision"] = resolution.decision
    signature["clarification_reason"] = resolution.clarification_reason
    request = resolution.request
    if request is None:
        return signature
    signature["capability"] = request.kind.value
    if isinstance(request, AnswerQuestion):
        signature["answer_topic"] = request.topic
    elif isinstance(request, ListOrders):
        signature["list_scope"] = request.scope
    elif isinstance(request, ModifyCart):
        signature["cart_operation"] = request.operation
    elif isinstance(request, ChangeProfile):
        signature["profile_field"] = request.field
    elif isinstance(request, VerifyOrderStatus):
        if request.target is None:
            signature["order_status_selector"] = "explicit"
        elif isinstance(request.target, FocusedOrderSet):
            signature["order_status_selector"] = "focused"
        elif isinstance(request.target, RecentOrderSet):
            signature["order_status_selector"] = "recent"
        else:
            signature["order_status_selector"] = "explicit"
    return signature


def _optional_sum(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _semantic_route_group(
    results: Sequence[SemanticRouteCaseResult],
) -> dict[str, object]:
    if not results:
        raise ValueError("semantic-route group requires at least one result")
    latencies = sorted(result.attempt.elapsed_ms for result in results)
    p50_index = max(math.ceil(len(latencies) * _SEMANTIC_ROUTE_P50_QUANTILE) - 1, 0)
    p95_index = max(math.ceil(len(latencies) * _SEMANTIC_ROUTE_P95_QUANTILE) - 1, 0)
    dispositions = {
        disposition: sum(result.disposition == disposition for result in results)
        for disposition in get_args(SemanticRouteDisposition)
    }
    return {
        "cases": len(results),
        "dispositions": dispositions,
        "provider_call_outcomes": {
            outcome: sum(result.attempt.provider_call_outcome == outcome for result in results)
            for outcome in get_args(ProviderCallOutcome)
        },
        "cache_read_cohorts": {
            "positive": sum(
                result.attempt.cache_read_tokens is not None
                and result.attempt.cache_read_tokens > 0
                for result in results
            ),
            "zero": sum(result.attempt.cache_read_tokens == 0 for result in results),
            "unreported": sum(result.attempt.cache_read_tokens is None for result in results),
        },
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": latencies[p50_index],
        "latency_ms_p95": latencies[p95_index],
        "latency_ms_max": latencies[-1],
        "input_tokens": _optional_sum([result.attempt.input_tokens for result in results]),
        "cache_read_tokens": _optional_sum(
            [result.attempt.cache_read_tokens for result in results]
        ),
        "output_tokens": _optional_sum([result.attempt.output_tokens for result in results]),
    }


def _group_semantic_route_results(
    results: Sequence[SemanticRouteCaseResult],
    attribute: Literal[
        "scenario_class",
        "risk_class",
        "risk_domain",
        "evaluation_split",
    ],
) -> dict[str, dict[str, object]]:
    values = dict.fromkeys(getattr(result, attribute) for result in results)
    return {
        value: _semantic_route_group(
            [result for result in results if getattr(result, attribute) == value]
        )
        for value in values
    }


def _semantic_model_report(
    results: Sequence[SemanticRouteCaseResult],
    *,
    repetitions: Sequence[int] | None = None,
) -> dict[str, object]:
    if not results:
        raise ValueError("semantic-route model report requires at least one result")
    if repetitions is None:
        repetitions = (1,) * len(results)
    if len(repetitions) != len(results):
        raise ValueError("semantic-route repetitions must align with results")
    first = results[0].attempt
    expected_identity = (
        first.provider,
        first.model,
        first.structured_output_method,
        first.route_schema_fingerprint,
        first.prompt_fingerprint,
        first.registry_fingerprint,
        first.input_max_chars,
        first.timeout_seconds,
        first.projector_version,
    )
    for result in results[1:]:
        attempt = result.attempt
        identity = (
            attempt.provider,
            attempt.model,
            attempt.structured_output_method,
            attempt.route_schema_fingerprint,
            attempt.prompt_fingerprint,
            attempt.registry_fingerprint,
            attempt.input_max_chars,
            attempt.timeout_seconds,
            attempt.projector_version,
        )
        if identity != expected_identity:
            raise ValueError("semantic-route model report cannot pool mixed run identities")
    return {
        "provider": first.provider,
        "model": first.model,
        "structured_output_method": first.structured_output_method,
        "route_schema_fingerprint": first.route_schema_fingerprint,
        "prompt_fingerprint": first.prompt_fingerprint,
        "registry_fingerprint": first.registry_fingerprint,
        "input_max_chars": first.input_max_chars,
        "timeout_seconds": first.timeout_seconds,
        "projector_version": first.projector_version,
        "totals": _semantic_route_group(results),
        "by_evaluation_split": _group_semantic_route_results(results, "evaluation_split"),
        "by_scenario_class": _group_semantic_route_results(results, "scenario_class"),
        "by_risk_class": _group_semantic_route_results(results, "risk_class"),
        "by_risk_domain": _group_semantic_route_results(results, "risk_domain"),
        "cases": [
            {
                "repetition": repetition,
                "case_id": result.case_id,
                "scenario_class": result.scenario_class,
                "risk_class": result.risk_class,
                "risk_domain": result.risk_domain,
                "evaluation_split": result.evaluation_split,
                "disposition": result.disposition,
                "expected": _route_signature(result.expected),
                "actual": _route_signature(result.attempt.resolution),
                "latency_ms": result.attempt.elapsed_ms,
                "provider_call_outcome": result.attempt.provider_call_outcome,
                "timeout_seconds": result.attempt.timeout_seconds,
                "input_tokens": result.attempt.input_tokens,
                "cache_read_tokens": result.attempt.cache_read_tokens,
                "output_tokens": result.attempt.output_tokens,
            }
            for repetition, result in zip(repetitions, results, strict=True)
        ],
    }


def _corpus_fingerprint(corpus: SemanticRouteEvalCorpus) -> str:
    encoded = json.dumps(
        corpus.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_disposition(
    results: Sequence[SemanticRouteCaseResult],
    disposition: SemanticRouteDisposition,
    *,
    evaluation_split: SemanticRouteEvaluationSplit | None = None,
    risk_class: SemanticRouteRiskClass | None = None,
) -> int:
    return sum(
        result.disposition == disposition
        and (evaluation_split is None or result.evaluation_split == evaluation_split)
        and (risk_class is None or result.risk_class == risk_class)
        for result in results
    )


def _assert_semantic_results_are_comparable(
    candidate: Sequence[SemanticRouteCaseResult],
    incumbent: Sequence[SemanticRouteCaseResult],
) -> None:
    def identity(result: SemanticRouteCaseResult) -> tuple[object, ...]:
        return (
            result.case_id,
            result.scenario_class,
            result.risk_class,
            result.risk_domain,
            result.evaluation_split,
            result.expected,
        )

    if [identity(result) for result in candidate] != [identity(result) for result in incumbent]:
        raise ValueError("candidate and incumbent results must cover the same ordered cases")


def _semantic_gate_failures(
    candidate: Sequence[SemanticRouteCaseResult],
    incumbent: Sequence[SemanticRouteCaseResult],
    *,
    projection: ProjectedSemanticRouteCaseResult,
    gate: Literal["shadow", "cutover"],
) -> tuple[str, ...]:
    _assert_semantic_results_are_comparable(candidate, incumbent)
    failures: list[str] = []
    if not projection.passed:
        failures.append("production context projection did not match the frozen fixture")
    if gate == "shadow":
        if _count_disposition(candidate, "closed_failure"):
            failures.append("candidate produced a closed failure")
        candidate_development = _count_disposition(
            candidate, "exact", evaluation_split="development"
        )
        incumbent_development = _count_disposition(
            incumbent, "exact", evaluation_split="development"
        )
        if candidate_development <= incumbent_development:
            failures.append("candidate is not more exact than incumbent on development")
        candidate_acceptance = _count_disposition(candidate, "exact", evaluation_split="acceptance")
        incumbent_acceptance = _count_disposition(incumbent, "exact", evaluation_split="acceptance")
        if candidate_acceptance < incumbent_acceptance:
            failures.append("candidate is inferior to incumbent on acceptance")
        candidate_critical = _count_disposition(
            candidate,
            "exact",
            evaluation_split="acceptance",
            risk_class="critical",
        )
        incumbent_critical = _count_disposition(
            incumbent,
            "exact",
            evaluation_split="acceptance",
            risk_class="critical",
        )
        if candidate_critical < incumbent_critical:
            failures.append("candidate is inferior to incumbent on critical acceptance")
        return tuple(failures)

    if _count_disposition(candidate, "closed_failure"):
        failures.append("candidate produced a closed failure")
    if _count_disposition(candidate, "unsafe_executable_misroute"):
        failures.append("candidate produced an unsafe executable misroute")
    acceptance = [result for result in candidate if result.evaluation_split == "acceptance"]
    critical = [result for result in acceptance if result.risk_class == "critical"]
    if _count_disposition(critical, "exact") != len(critical):
        failures.append("not every critical acceptance case was exact")
    budget = _semantic_acceptance_budget(acceptance)
    exact = _count_disposition(acceptance, "exact")
    if exact < budget["required_exact_count"]:
        failures.append(
            "acceptance exact count "
            f"{exact} was below required {budget['required_exact_count']} "
            f"of {budget['cases']}"
        )
    return tuple(failures)


def _semantic_acceptance_budget(
    results: Sequence[SemanticRouteCaseResult],
) -> dict[str, int]:
    cases = len(results)
    if cases == 0:
        raise ValueError("semantic-route qualification requires acceptance cases")
    required = math.ceil(cases * _SEMANTIC_ROUTE_MIN_ACCEPTANCE_EXACT_RATE)
    return {
        "cases": cases,
        "required_exact_count": required,
        "allowed_nonexact_count": cases - required,
    }


def _semantic_series_gate_failures(
    candidate_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    incumbent_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    *,
    projection: ProjectedSemanticRouteCaseResult,
    gate: Literal["shadow", "cutover"],
) -> tuple[str, ...]:
    if not candidate_runs or len(candidate_runs) != len(incumbent_runs):
        raise ValueError("semantic-route qualification requires paired repetitions")
    failures: list[str] = []
    for repetition, (candidate, incumbent) in enumerate(
        zip(candidate_runs, incumbent_runs, strict=True),
        start=1,
    ):
        for failure in _semantic_gate_failures(
            candidate,
            incumbent,
            projection=projection,
            gate=gate,
        ):
            failures.append(f"repetition {repetition}: {failure}")
    pooled_candidate = tuple(result for run in candidate_runs for result in run)
    pooled_incumbent = tuple(result for run in incumbent_runs for result in run)
    for failure in _semantic_gate_failures(
        pooled_candidate,
        pooled_incumbent,
        projection=projection,
        gate=gate,
    ):
        failures.append(f"pooled: {failure}")
    return tuple(failures)


def _semantic_route_report(
    candidate_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    incumbent_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    *,
    corpus: SemanticRouteEvalCorpus,
    projection: ProjectedSemanticRouteCaseResult,
    gate: SemanticRouteGate,
) -> dict[str, object]:
    if not candidate_runs or len(candidate_runs) != len(incumbent_runs):
        raise ValueError("semantic-route report requires paired repetitions")
    if gate == "diagnostic" and len(candidate_runs) != 1:
        raise ValueError("semantic-route diagnostic requires one repetition")
    if gate != "diagnostic" and len(candidate_runs) != _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS:
        raise ValueError(
            "semantic-route qualification requires exactly "
            f"{_SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS} repetitions"
        )
    first_candidate = candidate_runs[0]
    first_incumbent = incumbent_runs[0]
    for candidate, incumbent in zip(candidate_runs, incumbent_runs, strict=True):
        _assert_semantic_results_are_comparable(candidate, incumbent)
        _assert_semantic_results_are_comparable(first_candidate, candidate)
        _assert_semantic_results_are_comparable(first_incumbent, incumbent)
    pooled_candidate = tuple(result for run in candidate_runs for result in run)
    pooled_incumbent = tuple(result for run in incumbent_runs for result in run)
    candidate_repetitions = tuple(
        repetition for repetition, run in enumerate(candidate_runs, start=1) for _ in run
    )
    incumbent_repetitions = tuple(
        repetition for repetition, run in enumerate(incumbent_runs, start=1) for _ in run
    )
    gate_failures: tuple[str, ...] = ()
    if gate != "diagnostic":
        gate_failures = _semantic_series_gate_failures(
            candidate_runs,
            incumbent_runs,
            projection=projection,
            gate=gate,
        )
    acceptance = [result for result in first_candidate if result.evaluation_split == "acceptance"]
    run_reports: list[dict[str, object]] = []
    for repetition, (candidate, incumbent) in enumerate(
        zip(candidate_runs, incumbent_runs, strict=True),
        start=1,
    ):
        run_failures: tuple[str, ...] = ()
        if gate != "diagnostic":
            run_failures = _semantic_gate_failures(
                candidate,
                incumbent,
                projection=projection,
                gate=gate,
            )
        run_reports.append(
            {
                "repetition": repetition,
                "gate": {
                    "mode": gate,
                    "passed": None if gate == "diagnostic" else not run_failures,
                    "failures": list(run_failures),
                },
                "models": {
                    "candidate": _semantic_model_report(
                        candidate,
                        repetitions=(repetition,) * len(candidate),
                    ),
                    "incumbent": _semantic_model_report(
                        incumbent,
                        repetitions=(repetition,) * len(incumbent),
                    ),
                },
            }
        )
    return {
        "schema_version": _SEMANTIC_ROUTE_REPORT_SCHEMA_VERSION,
        "corpus_schema_version": corpus.schema_version,
        "run_at": datetime.now(tz=UTC).isoformat(),
        "corpus_fingerprint": _corpus_fingerprint(corpus),
        "repetitions": len(candidate_runs),
        "acceptance_budget": _semantic_acceptance_budget(acceptance),
        "route_schema_bytes": len(
            json.dumps(RouteProposal.model_json_schema(), separators=(",", ":")).encode("utf-8")
        ),
        "projection": {
            "case_id": projection.case_id,
            "scenario_class": projection.scenario_class,
            "risk_class": projection.risk_class,
            "risk_domain": projection.risk_domain,
            "evaluation_split": projection.evaluation_split,
            "exact": projection.passed,
            "actual": (
                "context"
                if isinstance(projection.actual_context, RoutingContext)
                else projection.actual_context.reason
            ),
        },
        "gate": {
            "mode": gate,
            "passed": None if gate == "diagnostic" else not gate_failures,
            "failures": list(gate_failures),
        },
        "models": {
            "candidate": _semantic_model_report(
                pooled_candidate,
                repetitions=candidate_repetitions,
            ),
            "incumbent": _semantic_model_report(
                pooled_incumbent,
                repetitions=incumbent_repetitions,
            ),
        },
        "runs": run_reports,
    }


def _write_semantic_route_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def _run_semantic_route_eval(
    report_path: Path,
    gate: SemanticRouteGate = "cutover",
    *,
    repetitions: int = _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS,
    diagnostic_timeout_seconds: float | None = None,
) -> int:
    selection = _load_routing_eval_selection()
    config = selection.config
    if gate == "diagnostic":
        if repetitions != 1:
            raise ValueError("semantic-route diagnostic requires one repetition")
        if (
            diagnostic_timeout_seconds is None
            or not math.isfinite(diagnostic_timeout_seconds)
            or diagnostic_timeout_seconds <= 0
        ):
            raise ValueError("semantic-route diagnostic requires a positive finite timeout")
        timeout_seconds = diagnostic_timeout_seconds
    else:
        if repetitions != _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS:
            raise ValueError(
                "semantic-route qualification requires exactly "
                f"{_SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS} repetitions"
            )
        if diagnostic_timeout_seconds is not None:
            raise ValueError("diagnostic timeout is valid only in diagnostic mode")
        timeout_seconds = config.runtime.semantic_router_timeout_seconds
    runtime = _build_eval_runtime(
        config,
        selection.response_model,
        selection.gateway.chat_model(config.llm.reasoning),
        thread_id=f"semantic-route-eval-{uuid.uuid4().hex}",
        structured_output_method=selection.response_structured_output_method,
    )
    try:
        candidate_router = SemanticRouter(
            selection.routing_model,
            selection=config.llm.routing,
            structured_output_method=selection.routing_structured_output_method,
            timeout_seconds=timeout_seconds,
            input_max_chars=config.runtime.semantic_router_input_max_chars,
            registry=runtime.capability_registry,
        )
        incumbent_router = SemanticRouter(
            selection.response_model,
            selection=config.llm.response,
            structured_output_method=selection.response_structured_output_method,
            timeout_seconds=timeout_seconds,
            input_max_chars=config.runtime.semantic_router_input_max_chars,
            registry=runtime.capability_registry,
        )
        corpus = _load_semantic_route_corpus()
        config_key = {"configurable": {"thread_id": runtime.engine.thread_id}}
        state = ReasoningState.model_validate(runtime.graph.get_state(config_key).values)
        projected = project_routing_context(
            corpus.projected_case.turn,
            state,
            identity_store=runtime.identity_store,
            cart_store=runtime.cart_store,
            recent_orders=runtime.recent_orders,
            registry=runtime.capability_registry,
        )
        projection = ProjectedSemanticRouteCaseResult(
            case_id=corpus.projected_case.case_id,
            scenario_class=corpus.projected_case.scenario_class,
            risk_class=corpus.projected_case.risk_class,
            risk_domain=corpus.projected_case.risk_domain,
            evaluation_split=corpus.projected_case.evaluation_split,
            expected_context=corpus.projected_case.expected_context,
            actual_context=projected,
        )
        candidate_runs: list[tuple[SemanticRouteCaseResult, ...]] = []
        incumbent_runs: list[tuple[SemanticRouteCaseResult, ...]] = []
        for _ in range(repetitions):
            candidate_results = list(await _run_semantic_route_cases(candidate_router, corpus))
            incumbent_results = list(await _run_semantic_route_cases(incumbent_router, corpus))
            if projection.passed:
                assert isinstance(projected, RoutingContext)
                candidate_results.insert(
                    0,
                    SemanticRouteCaseResult(
                        case_id=corpus.projected_case.case_id,
                        scenario_class=corpus.projected_case.scenario_class,
                        risk_class=corpus.projected_case.risk_class,
                        risk_domain=corpus.projected_case.risk_domain,
                        evaluation_split=corpus.projected_case.evaluation_split,
                        expected=corpus.projected_case.expected,
                        attempt=await candidate_router.route(projected),
                    ),
                )
                incumbent_results.insert(
                    0,
                    SemanticRouteCaseResult(
                        case_id=corpus.projected_case.case_id,
                        scenario_class=corpus.projected_case.scenario_class,
                        risk_class=corpus.projected_case.risk_class,
                        risk_domain=corpus.projected_case.risk_domain,
                        evaluation_split=corpus.projected_case.evaluation_split,
                        expected=corpus.projected_case.expected,
                        attempt=await incumbent_router.route(projected),
                    ),
                )
            candidate_runs.append(tuple(candidate_results))
            incumbent_runs.append(tuple(incumbent_results))
        gate_failures: tuple[str, ...] = ()
        if gate != "diagnostic":
            gate_failures = _semantic_series_gate_failures(
                candidate_runs,
                incumbent_runs,
                projection=projection,
                gate=gate,
            )
        report = _semantic_route_report(
            candidate_runs,
            incumbent_runs,
            corpus=corpus,
            projection=projection,
            gate=gate,
        )
        await asyncio.to_thread(_write_semantic_route_report, report_path, report)
        for repetition, (candidate_results, incumbent_results) in enumerate(
            zip(candidate_runs, incumbent_runs, strict=True),
            start=1,
        ):
            candidate_exact = _count_disposition(candidate_results, "exact")
            incumbent_exact = _count_disposition(incumbent_results, "exact")
            route_cases = len(candidate_results)
            print(
                f"[semantic_routes] repetition {repetition} candidate exact: "
                f"{candidate_exact}/{route_cases}"
            )
            print(
                f"[semantic_routes] repetition {repetition} incumbent exact: "
                f"{incumbent_exact}/{route_cases}"
            )
            for result in candidate_results:
                if result.disposition != "exact":
                    print(
                        f"    repetition {repetition} {result.case_id}: "
                        f"{result.disposition}; expected "
                        f"{_route_signature(result.expected)}, got "
                        f"{_route_signature(result.attempt.resolution)}"
                    )
        print(f"[semantic_routes] projector: {int(projection.passed)}/1")
        if not projection.passed:
            print(f"    {projection.case_id}: production context projection did not match fixture")
        print(f"    sanitized report: {report_path}")
        if gate == "diagnostic":
            print("\nSEMANTIC ROUTING DIAGNOSTIC COMPLETED. [NO QUALIFICATION]")
            return 0
        if gate_failures:
            for failure in gate_failures:
                print(f"    gate failure: {failure}")
            print(f"\nSEMANTIC ROUTING {gate.upper()} GATE FAILED. [FAIL]")
            return 1
        print(f"\nSEMANTIC ROUTING {gate.upper()} GATE PASSED. [SEMANTIC-QUALITY PASS]")
        print(
            "Coverage limit: offline text/provider routing only; no live dispatch, speech, "
            "checkpoint, tool, or effect ran."
        )
        return 0
    finally:
        runtime.caller_context.close_session()


def _score_read_owner_output(
    case: ReadOwnerEvalCase,
    *,
    actual_disposition: Literal["answer", "clarify", "unsupported"],
    spoken_text: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    normalized = spoken_text.casefold()
    if actual_disposition != case.expected_disposition:
        failures.append(f"expected {case.expected_disposition}, got {actual_disposition}")
    for fact in case.required_facts:
        if fact.casefold() not in normalized:
            failures.append(f"missing required fact {fact!r}")
    for fact in case.forbidden_facts:
        if fact.casefold() in normalized:
            failures.append(f"included forbidden fact {fact!r}")
    return tuple(failures)


def _read_owner_disposition(
    case: ReadOwnerEvalCase,
    response_nodes: Sequence[str],
) -> Literal["answer", "clarify", "unsupported"] | None:
    node_dispositions: dict[str, Literal["answer", "clarify", "unsupported"]]
    if case.owner == "catalog":
        node_dispositions = {CATALOG_RESPONSE_NODE: "answer"}
    else:
        node_dispositions = {
            ANSWER_RESPONSE_NODE: "answer",
            ANSWER_CLARIFY_NODE: "clarify",
            ANSWER_UNSUPPORTED_NODE: "unsupported",
        }
    if len(response_nodes) != 1:
        return None
    return node_dispositions.get(response_nodes[0])


async def _run_read_owner_cases(
    graph: CompiledStateGraph,
    corpus: ReadOwnerEvalCorpus,
) -> dict[str, tuple[str, ...]]:
    failures: dict[str, tuple[str, ...]] = {}
    for case in corpus.cases:
        request: IntentRequest
        if case.owner == "catalog":
            request = SearchCatalog(query=case.query)
        else:
            request = AnswerQuestion(topic=case.topic)
        turn_id = f"read-owner:{case.case_id}:{uuid.uuid4().hex}"
        spoken: list[str] = []
        response_nodes: list[str] = []
        async for update in graph.astream(
            {
                "messages": [HumanMessage(content=case.utterance, id=turn_id)],
                "consumed_turn_ids": (turn_id,),
                "active_invocation": open_active_invocation(
                    request,
                    consumed_turn_ids=(turn_id,),
                ),
            },
            {"configurable": {"thread_id": turn_id}},
            stream_mode="updates",
        ):
            for node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                messages = delta.get("messages")
                if not isinstance(messages, (list, tuple)):
                    continue
                audible = [
                    message.text
                    for message in messages
                    if isinstance(message, AIMessage) and message.text.strip()
                ]
                if audible:
                    response_nodes.append(node)
                    spoken.extend(audible)
        if not spoken:
            failures[case.case_id] = ("owner produced no caller-audible text",)
            continue
        line = spoken[-1]
        disposition = _read_owner_disposition(case, response_nodes)
        if disposition is None:
            failures[case.case_id] = (
                "owner did not execute exactly one expected terminal response node; "
                f"got {tuple(response_nodes)!r}",
            )
            continue
        case_failures = _score_read_owner_output(
            case,
            actual_disposition=disposition,
            spoken_text=line,
        )
        if case_failures:
            failures[case.case_id] = case_failures
    return failures


_TRANSPORT_OWNER_SCENARIOS = (
    TransportOwnerScenario(
        scenario_id="frontline_browse",
        owner="frontline",
        origin_node="model",
        utterance="Tell me about trail shoes.",
        offline_faults=(FaultKind.PRE_RESPONSE_DISCONNECT, FaultKind.INTERRUPTED_BODY),
        live_faults=(FaultKind.PRE_RESPONSE_DISCONNECT,),
    ),
    TransportOwnerScenario(
        scenario_id="cart_checkout",
        owner="cart",
        origin_node="cart_assemble",
        utterance="Checkout now please.",
        offline_faults=(FaultKind.PRE_RESPONSE_DISCONNECT, FaultKind.INTERRUPTED_BODY),
    ),
    TransportOwnerScenario(
        scenario_id="identity_account_orders",
        owner="identity",
        origin_node="identity_assemble",
        utterance="What orders are on my account?",
    ),
    TransportOwnerScenario(
        scenario_id="support_cancel",
        owner="support",
        origin_node="support_assemble",
        utterance="Cancel order ORD-1002.",
    ),
    TransportOwnerScenario(
        scenario_id="catalog_grounded_response",
        owner="frontline",
        origin_node=CATALOG_RESPONSE_NODE,
        utterance="Tell me about trail shoes.",
        initial_request=SearchCatalog(query="trail shoes"),
    ),
    TransportOwnerScenario(
        scenario_id="answer_bounded_response",
        owner="frontline",
        origin_node=ANSWER_RESPONSE_NODE,
        utterance="What is your return policy?",
        initial_request=AnswerQuestion(topic="policy"),
    ),
    TransportOwnerScenario(
        scenario_id="order_status_target_proposal",
        owner="frontline",
        origin_node=ORDER_STATUS_TARGET_PROPOSE_NODE,
        utterance="Order 1002.",
        initial_request=VerifyOrderStatus(),
    ),
)
_EMPTY_RECOVERY_STATE = GraphObservation(
    active_flow=None,
    automation_channels=(),
    handover_destination=None,
    interrupted=False,
    unfinished=False,
    automation_terminal=False,
)


class _OfflineSecretResolver:
    def resolve(self, _ref: str) -> str:
        return "offline-not-a-secret"


def _transport_targets() -> tuple[dict[str, ProviderModel], int]:
    targets_config = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml")
    by_provider: dict[str, ProviderModel] = {}
    for target in targets_config.targets:
        by_provider.setdefault(target.provider, target)
    missing_contracts = sorted(set(by_provider) - set(PROVIDER_TRANSPORT_CONTRACTS))
    if missing_contracts:
        raise RuntimeError(
            f"transport certification has no endpoint contract for providers: {missing_contracts!r}"
        )
    return by_provider, targets_config.max_retries


def _transport_case_matrix(
    *,
    tier: Literal["offline", "live"],
    providers: Sequence[str],
) -> tuple[tuple[TransportOwnerScenario, str, FaultKind], ...]:
    if not providers:
        raise RuntimeError("transport certification has no configured providers")
    cases: list[tuple[TransportOwnerScenario, str, FaultKind]] = []
    provider_offsets: dict[FaultKind, int] = {}
    for scenario in _TRANSPORT_OWNER_SCENARIOS:
        fault_kinds = scenario.live_faults if tier == "live" else scenario.offline_faults
        if tier == "live":
            cases.extend(
                (scenario, provider, fault_kind)
                for fault_kind in fault_kinds
                for provider in providers
            )
            continue
        for fault_kind in fault_kinds:
            provider_offset = provider_offsets.get(fault_kind, 0)
            provider = providers[provider_offset % len(providers)]
            provider_offsets[fault_kind] = provider_offset + 1
            cases.append((scenario, provider, fault_kind))
    return tuple(cases)


def _seed_transport_request(runtime: EvalRuntime, scenario: TransportOwnerScenario) -> None:
    if scenario.initial_request is None:
        return
    opening_turn_ids = (f"transport:{scenario.scenario_id}:opening",)
    config = {"configurable": {"thread_id": runtime.engine.thread_id}}
    runtime.graph.update_state(
        config,
        {
            "consumed_turn_ids": opening_turn_ids,
            "active_invocation": open_active_invocation(
                scenario.initial_request,
                consumed_turn_ids=opening_turn_ids,
            ),
        },
        # Seed as a completed bootstrap node so the observed utterance enters as a real fresh
        # HumanMessage. START seeding creates a principal-continuation task that intentionally
        # consumes the next message id without appending caller text.
        as_node=runtime.graph.principal_seed_complete_node,
    )
    snapshot = runtime.graph.get_state(config)
    state = ReasoningState.model_validate(snapshot.values)
    if (
        snapshot.next
        or snapshot.interrupts
        or state.messages
        or state.consumed_turn_ids != opening_turn_ids
        or state.active_invocation is None
        or state.active_invocation.request != scenario.initial_request
    ):
        raise RuntimeError("transport initial request did not seed as a completed checkpoint")


def _read_telemetry(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        return ()
    return tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _attempt_dict(attempt: ProxyAttempt) -> dict[str, object]:
    return {
        "sequence": attempt.sequence,
        "method": attempt.method,
        "path": attempt.path,
        "outcome": attempt.outcome,
        "status_code": attempt.status_code,
    }


def _write_transport_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _score_transport_recovery(
    *,
    tier: Literal["offline", "live"],
    scenario: TransportOwnerScenario,
    observation: ScenarioObservation,
    attempts: tuple[ProxyAttempt, ...],
    turn_failed: tuple[dict[str, object], ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    final = observation.final
    if final.effects != observation.before:
        failures.append("authoritative commerce/identity state changed")
    if final.state != _EMPTY_RECOVERY_STATE:
        failures.append("graph did not finish in a clean non-terminal state")
    if final.admitted_user_messages != (scenario.utterance,):
        failures.append("committed caller turn was not admitted exactly once")

    outcomes = tuple(attempt.outcome for attempt in attempts)
    forbidden_outcomes = {"unexpected_provider_endpoint", "safety_ceiling", "upstream_error"}
    if set(outcomes) & forbidden_outcomes:
        failures.append("proxy reported an endpoint, safety, or upstream failure")
    if not attempts:
        failures.append("provider adapter made no request")
    elif tier == "offline":
        if set(outcomes) != {"faulted"}:
            failures.append("offline exhaustion forwarded or accepted a provider request")
        expected_action = "cart_review" if scenario.owner == "cart" else "safe_abort"
        expected_failure = {
            "event": "turn_failed",
            "reason": "node_exception",
            "node": scenario.origin_node,
            "action": expected_action,
        }
        if turn_failed != (expected_failure,):
            failures.append("typed recovery telemetry did not identify the owning model node")
        if final.completed_tool_calls != 0:
            failures.append("an incomplete provider response produced a completed tool call")
        if len(final.audible) != 1 or final.audible[0].node != RECOVERY_NODE_NAME:
            failures.append("recovery did not produce exactly one code-authored audible line")
        elif scenario.owner != "cart" and final.audible[0].text != TURN_FALLBACK_LINE:
            failures.append("safe-abort recovery line differed from the fixed contract")
    else:
        successful_retry = any(
            attempt.outcome == "passed_upstream"
            and attempt.status_code is not None
            and HTTPStatus.OK <= attempt.status_code < HTTPStatus.MULTIPLE_CHOICES
            for attempt in attempts[1:]
        )
        if outcomes[0:1] != ("faulted",) or not successful_retry:
            failures.append("retry masking did not prove fault-then-upstream-success")
        if turn_failed:
            failures.append("a retry-masked provider call entered failure recovery")
        if not final.audible:
            failures.append("successful retry produced no caller-audible response")
    return tuple(failures)


def _failed_transport_case(
    *,
    tier: Literal["offline", "live"],
    scenario: TransportOwnerScenario,
    provider: str,
    target: ProviderModel,
    fault_kind: FaultKind,
    failure: str,
    attempts: tuple[ProxyAttempt, ...],
    telemetry_path: Path,
) -> TransportCaseResult:
    turn_failed = tuple(
        record for record in _read_telemetry(telemetry_path) if record.get("event") == "turn_failed"
    )
    return TransportCaseResult(
        case_id=f"{tier}:{provider}:{scenario.scenario_id}:{fault_kind}",
        tier=tier,
        provider=provider,
        model=target.model,
        owner=scenario.owner,
        fault_kind=fault_kind,
        passed=False,
        failures=(failure,),
        attempts=tuple(_attempt_dict(attempt) for attempt in attempts),
        audible=(),
        before={},
        after={},
        state={},
        turn_failed=turn_failed,
    )


async def _run_transport_case(
    *,
    tier: Literal["offline", "live"],
    scenario: TransportOwnerScenario,
    provider: str,
    target: ProviderModel,
    fault_kind: FaultKind,
    max_retries: int,
    config: MerchantConfig,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    request_ceiling: int,
    fault_window_seconds: float,
    upstream_timeout_seconds: float,
    scenario_timeout_seconds: float,
) -> TransportCaseResult:
    mode = FaultMode.UNTIL_EXHAUSTED if tier == "offline" else FaultMode.RETRY_MASKED_ONCE
    runtime: EvalRuntime | None = None
    old_telemetry_path = telemetry._TELEMETRY_PATH
    with tempfile.TemporaryDirectory(prefix="transport-cert-") as temp_dir:
        telemetry_path = Path(temp_dir) / "telemetry.jsonl"
        telemetry._TELEMETRY_PATH = telemetry_path
        proxy = TransportFaultProxy(
            PROVIDER_TRANSPORT_CONTRACTS[provider],
            mode=mode,
            fault_kind=fault_kind,
            request_ceiling=request_ceiling,
            fault_window_seconds=fault_window_seconds,
            upstream_timeout_seconds=upstream_timeout_seconds,
        )
        try:
            with proxy:
                gateway = LLMGateway(credentials, secrets)
                model = gateway.chat_model(
                    target,
                    base_url=proxy.model_base_url,
                    max_retries=max_retries,
                    timeout=upstream_timeout_seconds,
                )
                runtime = _build_eval_runtime(
                    config,
                    model,
                    model,
                    thread_id=f"transport-{uuid.uuid4().hex}",
                    structured_output_method=gateway.structured_output_method(target),
                )
                _seed_transport_request(runtime, scenario)
                observation = await asyncio.wait_for(
                    _observe_scenario(
                        runtime.engine,
                        (scenario.utterance,),
                        scenario_key=f"transport-{tier}-{provider}-{scenario.scenario_id}",
                        store=runtime.store,
                        profile_store=runtime.profile_store,
                        otp=runtime.otp,
                        verification=runtime.verification,
                        identity_store=runtime.identity_store,
                        model_call_count=lambda: len(proxy.attempts),
                    ),
                    timeout=scenario_timeout_seconds,
                )
        except TimeoutError:
            return _failed_transport_case(
                tier=tier,
                scenario=scenario,
                provider=provider,
                target=target,
                fault_kind=fault_kind,
                failure="scenario timeout ceiling exceeded",
                attempts=proxy.attempts,
                telemetry_path=telemetry_path,
            )
        else:
            records = _read_telemetry(telemetry_path)
            turn_failed = tuple(
                record for record in records if record.get("event") == "turn_failed"
            )
            failures = _score_transport_recovery(
                tier=tier,
                scenario=scenario,
                observation=observation,
                attempts=proxy.attempts,
                turn_failed=turn_failed,
            )
            final = observation.final
            return TransportCaseResult(
                case_id=f"{tier}:{provider}:{scenario.scenario_id}:{fault_kind}",
                tier=tier,
                provider=provider,
                model=target.model,
                owner=scenario.owner,
                fault_kind=fault_kind,
                passed=not failures,
                failures=failures,
                attempts=tuple(_attempt_dict(attempt) for attempt in proxy.attempts),
                audible=tuple(asdict(item) for item in final.audible),
                before=asdict(observation.before),
                after=asdict(final.effects),
                state=asdict(final.state),
                turn_failed=turn_failed,
            )
        finally:
            try:
                if runtime is not None:
                    runtime.caller_context.close_session()
            finally:
                telemetry._TELEMETRY_PATH = old_telemetry_path


async def _run_transport_certification(
    *,
    tier: Literal["offline", "live"],
    report_path: Path,
    request_ceiling: int,
    fault_window_seconds: float,
    upstream_timeout_seconds: float,
    scenario_timeout_seconds: float,
) -> int:
    targets, max_retries = _transport_targets()
    providers = tuple(sorted(targets))
    credentials = load_provider_credentials(_CONFIG_ROOT / "base" / "providers.yaml")
    if tier == "live":
        load_dotenv()
        secrets: SecretResolver = EnvSecretResolver()
    else:
        secrets = _OfflineSecretResolver()
    config = ConfigRegistry(_CONFIG_ROOT).load().get(_MERCHANT_ID).config
    cases = _transport_case_matrix(tier=tier, providers=providers)
    results: list[TransportCaseResult] = []
    for scenario, provider, fault_kind in cases:
        result = await _run_transport_case(
            tier=tier,
            scenario=scenario,
            provider=provider,
            target=targets[provider],
            fault_kind=fault_kind,
            max_retries=max_retries,
            config=config,
            credentials=credentials,
            secrets=secrets,
            request_ceiling=request_ceiling,
            fault_window_seconds=fault_window_seconds,
            upstream_timeout_seconds=upstream_timeout_seconds,
            scenario_timeout_seconds=scenario_timeout_seconds,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        for failure in result.failures:
            print(f"    {failure}")

    report = {
        "transport_contract_version": TRANSPORT_CONTRACT_VERSION,
        "tier": tier,
        "run_at": datetime.now(tz=UTC).isoformat(),
        "passed": all(result.passed for result in results),
        "cases": [asdict(result) for result in results],
    }
    await asyncio.to_thread(_write_transport_report, report_path, report)
    print(f"transport recovery report: {report_path}")
    return 0 if report["passed"] else 1


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
    turn_id = uuid.uuid4().hex
    out = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=utterance, id=turn_id)],
            "consumed_turn_ids": (turn_id,),
        },
        config,
    )
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
    speech_failures = _speech_authority_failures()
    order_failures = _order_reference_failures(safety_data)
    print("[coverage] structural speech authority")
    for failure in speech_failures:
        print(f"    STRUCTURAL FAILURE: {failure}")
    if not speech_failures:
        print("[speech_authority] all registered contracts passed. [PASS]")
    print("[coverage] strong-labelled STT order references")
    for failure in order_failures:
        print(f"    STRUCTURAL FAILURE: {failure}")
    if not order_failures:
        print("[order_reference] all registered contracts passed. [PASS]")
    if speech_failures or order_failures:
        print("\nFRONTLINE EVAL BLOCKED - structural safety preflight failed. [FAIL]")
        return 1
    print("[structural_preflight] all registered contracts passed. [PASS]")
    if preflight_only:
        return 0

    # Eval runs must not pollute the LIVE telemetry dataset (classifier data): redirect
    # this process's sink to a sibling eval file (same local-only, gitignored dir).
    telemetry._TELEMETRY_PATH = telemetry._TELEMETRY_PATH.with_name("frontline_eval.jsonl")
    selection = _load_routing_eval_selection()
    config = selection.config
    chat_model = selection.response_model
    runtime = _build_eval_runtime(
        config,
        chat_model,
        selection.gateway.chat_model(config.llm.reasoning),
        thread_id=f"frontline-eval-{uuid.uuid4().hex}",
        structured_output_method=selection.response_structured_output_method,
    )
    graph = runtime.graph
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

    read_owner_corpus = _load_read_owner_corpus()
    read_owner_failures = await _run_read_owner_cases(graph, read_owner_corpus)
    print(
        f"[read_owners] semantic cases: "
        f"{len(read_owner_corpus.cases) - len(read_owner_failures)}/"
        f"{len(read_owner_corpus.cases)}"
    )
    for case_id, failures in read_owner_failures.items():
        for failure in failures:
            print(f"    {case_id}: {failure}")

    order_target_failures = await _evaluate_order_targets(
        chat_model,
        selection.response_structured_output_method,
    )

    if recall < _RECALL_BAR:
        print(
            f"\nT1 ROUTING EVAL FAILED - escalation recall {recall:.0%} < {_RECALL_BAR:.0%}. [FAIL]"
        )
        return 1
    if read_owner_failures:
        print("\nFRONTLINE READ-OWNER EVAL FAILED. [FAIL]")
        return 1
    if order_target_failures:
        print("\nFRONTLINE ORDER-TARGET EVAL FAILED. [FAIL]")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument(
        "--order-target-eval",
        action="store_true",
        help="run the text-only credentialed order-target semantic corpus",
    )
    mode.add_argument(
        "--semantic-routing-eval",
        action="store_true",
        help="run the typed semantic-router corpus without dispatching the live graph",
    )
    mode.add_argument(
        "--recovery-certification",
        choices=("offline", "live"),
        help=(
            "run Milestone 6E provider-transport recovery certification instead of the "
            "routing-quality eval"
        ),
    )
    parser.add_argument(
        "--transport-report",
        type=Path,
        help="report destination (defaults to a tier-specific transport report)",
    )
    parser.add_argument(
        "--semantic-routing-report",
        type=Path,
        help="sanitized semantic-router report destination",
    )
    parser.add_argument(
        "--semantic-routing-gate",
        choices=("diagnostic", "shadow", "cutover"),
        help="semantic-router qualification gate (defaults to cutover)",
    )
    parser.add_argument(
        "--semantic-routing-repetitions",
        type=int,
        help="one diagnostic repetition or exactly three qualification repetitions",
    )
    parser.add_argument(
        "--semantic-routing-diagnostic-timeout-seconds",
        type=float,
        help="evaluation-only router timeout; valid only with the diagnostic gate",
    )
    parser.add_argument(
        "--transport-request-ceiling",
        type=int,
        default=_TRANSPORT_REQUEST_CEILING,
    )
    parser.add_argument(
        "--transport-fault-window-seconds",
        type=float,
        default=_TRANSPORT_FAULT_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--transport-upstream-timeout-seconds",
        type=float,
        default=_TRANSPORT_UPSTREAM_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--transport-scenario-timeout-seconds",
        type=float,
        default=_TRANSPORT_SCENARIO_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    custom_transport_options = (
        args.transport_report is not None
        or args.transport_request_ceiling != _TRANSPORT_REQUEST_CEILING
        or args.transport_fault_window_seconds != _TRANSPORT_FAULT_WINDOW_SECONDS
        or args.transport_upstream_timeout_seconds != _TRANSPORT_UPSTREAM_TIMEOUT_SECONDS
        or args.transport_scenario_timeout_seconds != _TRANSPORT_SCENARIO_TIMEOUT_SECONDS
    )
    if args.recovery_certification is None and custom_transport_options:
        parser.error("transport options require --recovery-certification")
    if args.semantic_routing_report is not None and not args.semantic_routing_eval:
        parser.error("--semantic-routing-report requires --semantic-routing-eval")
    if args.semantic_routing_gate is not None and not args.semantic_routing_eval:
        parser.error("--semantic-routing-gate requires --semantic-routing-eval")
    if args.semantic_routing_repetitions is not None and not args.semantic_routing_eval:
        parser.error("--semantic-routing-repetitions requires --semantic-routing-eval")
    if (
        args.semantic_routing_diagnostic_timeout_seconds is not None
        and not args.semantic_routing_eval
    ):
        parser.error(
            "--semantic-routing-diagnostic-timeout-seconds requires --semantic-routing-eval"
        )
    if args.order_target_eval:
        return asyncio.run(_run_order_target_eval())
    if args.semantic_routing_eval:
        semantic_gate = args.semantic_routing_gate or "cutover"
        if semantic_gate == "diagnostic":
            repetitions = (
                1
                if args.semantic_routing_repetitions is None
                else args.semantic_routing_repetitions
            )
            if repetitions != 1:
                parser.error("semantic-routing diagnostic requires one repetition")
            diagnostic_timeout_seconds = args.semantic_routing_diagnostic_timeout_seconds
            if (
                diagnostic_timeout_seconds is None
                or not math.isfinite(diagnostic_timeout_seconds)
                or diagnostic_timeout_seconds <= 0
            ):
                parser.error(
                    "semantic-routing diagnostic requires a positive finite "
                    "--semantic-routing-diagnostic-timeout-seconds"
                )
        else:
            repetitions = (
                _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS
                if args.semantic_routing_repetitions is None
                else args.semantic_routing_repetitions
            )
            if repetitions != _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS:
                parser.error(
                    "semantic-routing qualification requires exactly "
                    f"{_SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS} repetitions"
                )
            if args.semantic_routing_diagnostic_timeout_seconds is not None:
                parser.error(
                    "--semantic-routing-diagnostic-timeout-seconds requires the diagnostic gate"
                )
            diagnostic_timeout_seconds = None
        return asyncio.run(
            _run_semantic_route_eval(
                args.semantic_routing_report or _SEMANTIC_ROUTE_REPORT_PATH,
                semantic_gate,
                repetitions=repetitions,
                diagnostic_timeout_seconds=diagnostic_timeout_seconds,
            )
        )
    if args.recovery_certification is not None:
        return asyncio.run(
            _run_transport_certification(
                tier=args.recovery_certification,
                report_path=(
                    args.transport_report or _TRANSPORT_REPORT_PATHS[args.recovery_certification]
                ),
                request_ceiling=args.transport_request_ceiling,
                fault_window_seconds=args.transport_fault_window_seconds,
                upstream_timeout_seconds=args.transport_upstream_timeout_seconds,
                scenario_timeout_seconds=args.transport_scenario_timeout_seconds,
            )
        )
    return asyncio.run(_run(preflight_only=args.preflight_only))


if __name__ == "__main__":
    sys.exit(main())
