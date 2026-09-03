"""Explicit evaluation gates for the active typed-routing architecture.

Run one mode at a time:
  uv run python scripts/frontline_eval.py --preflight-only
  uv run python scripts/frontline_eval.py --order-target-eval
  uv run python scripts/frontline_eval.py --semantic-routing-eval
  uv run python scripts/frontline_eval.py --semantic-routing-structural-coverage
  uv run python scripts/frontline_eval.py --semantic-routing-data-audit
  uv run python scripts/frontline_eval.py --recovery-certification offline
  uv run python scripts/frontline_eval.py --recovery-certification live

Credentialed modes use the configured provider gateway. Offline structural and recovery modes make
no upstream calls. The historical gate/model escalation corpus is preserved as source data but is
not an executable rubric.
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.engine import ReasoningEngine, _TurnSpeech
from agnostic_market.agents.frontline import (
    FRONTLINE_SPEAKABLE_NODES,
    MODEL_SPEECH_NODES,
    NON_SPEAKING_MODEL_NODES,
)
from agnostic_market.agents.frontline.graph import _build_frontline_capability_registry
from agnostic_market.agents.frontline.read_flow import (
    ANSWER_CLARIFY_NODE,
    ANSWER_RESPONSE_NODE,
    ANSWER_UNSUPPORTED_NODE,
    CATALOG_RESPONSE_NODE,
    ORDER_STATUS_TARGET_PROPOSE_NODE,
)
from agnostic_market.agents.frontline.typed_prompt import ORDER_TARGET_PROPOSAL_PROMPT
from agnostic_market.agents.recovery import RECOVERY_NODE_NAME, TURN_FALLBACK_LINE
from agnostic_market.agents.routing import (
    CONTEXT_PROJECTOR_VERSION,
    ROUTE_SCHEMA_FINGERPRINT,
    ProviderCallOutcome,
    RoutingAttempt,
    RoutingRecognizer,
    SemanticRouter,
    materialize_route,
    project_routing_context,
    registry_fingerprint,
    resolve_route,
)
from agnostic_market.agents.routing_activation import (
    SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION,
)
from agnostic_market.agents.telemetry import (
    DisabledTelemetrySink,
    JsonlTelemetrySink,
    TelemetrySink,
    TenantTelemetry,
)
from agnostic_market.application import (
    ApplicationModels,
    ApplicationSession,
    ApplicationSettings,
    build_application_session,
    build_fixture_tenant_services,
    build_in_memory_session_state,
)
from agnostic_market.checkpoints import SchemaValidatedCheckpointSaver, build_checkpointer
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import CallerIdentityStore
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext
from agnostic_market.commerce.profile import ProfileStore
from agnostic_market.commerce.spoken import caller_stated_order_ids
from agnostic_market.commerce.verification import OtpProvider, VerificationStore
from agnostic_market.config.loader import ConfigError, config_version, load_yaml_layer
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
    CancelOrders,
    CapabilityId,
    ChangeProfile,
    FocusedOrderSet,
    IntentRequest,
    ListOrders,
    ModifyCart,
    OrderContextOperation,
    OrderTargetProposal,
    RecentOrderSet,
    RouteDecision,
    RouteProposal,
    RouteResolution,
    RoutingContext,
    RoutingFailure,
    SearchCatalog,
    VerifyOrderStatus,
)
from agnostic_market.dtos.state import (
    CHECKPOINT_SCHEMA_VERSION,
    ReasoningState,
    open_active_invocation,
)
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import load_conformance_targets
from agnostic_market.secrets.base import SecretResolver
from agnostic_market.secrets.env_resolver import EnvSecretResolver
from agnostic_market.session import CallerContext
from agnostic_market.tenancy.context import TenantContext

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
_SAFETY_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_safety.yaml"
_READ_OWNER_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_read_owners.yaml"
_READ_OWNER_EVAL_SCHEMA_VERSION = "1"
_EVAL_DEPLOYMENT_ID = "frontline-eval"
_ORDER_TARGET_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_order_targets.yaml"
_ORDER_TARGET_EVAL_SCHEMA_VERSION = "1"
_SEMANTIC_ROUTE_EVAL_PATH = _CONFIG_ROOT / "eval" / "frontline_semantic_routes.yaml"
_SEMANTIC_ROUTE_STRUCTURAL_SUPPLEMENT_PATH = (
    _CONFIG_ROOT / "eval" / "frontline_semantic_route_structural.yaml"
)
_SEMANTIC_ROUTE_CORPUS_SCHEMA_VERSION = "5"
_SEMANTIC_ROUTE_STRUCTURAL_SCHEMA_VERSION = "1"
_SEMANTIC_ROUTE_STRUCTURAL_REPORT_SCHEMA_VERSION = "1"
_SEMANTIC_ROUTE_CANONICAL_LEAF_COUNT = 29
_SEMANTIC_ROUTE_REPORT_SCHEMA_VERSION = SEMANTIC_ROUTING_QUALIFICATION_SCHEMA_VERSION
_SEMANTIC_ROUTE_REPORT_PATH = _CONFIG_ROOT / "telemetry" / "semantic_routing_report.json"
_SEMANTIC_ROUTE_STRUCTURAL_REPORT_PATH = (
    _CONFIG_ROOT / "telemetry" / "semantic_routing_structural_coverage.json"
)
_ROUTING_DATA_MANIFEST_SCHEMA_VERSION = "1"
_ROUTING_DATA_ACQUISITION_PLAN_SCHEMA_VERSION = "1"
_ROUTING_DATA_AUTHORIZATION_SCHEMA_VERSION = "2"
_ROUTING_DATA_REPORT_SCHEMA_VERSION = "2"
_ROUTING_DATA_REPORT_PATH = _CONFIG_ROOT / "telemetry" / "semantic_routing_data_readiness.json"
_SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS = 3
_SEMANTIC_ROUTE_MIN_ACCEPTANCE_EXACT_RATE = 0.95
_SEMANTIC_ROUTE_P50_QUANTILE = 0.50
_SEMANTIC_ROUTE_P95_QUANTILE = 0.95
_MERCHANT_ID = "acme_store"
_TRANSPORT_REPORT_PATHS = {
    "offline": _CONFIG_ROOT / "telemetry" / "transport_recovery_offline_report.json",
    "live": _CONFIG_ROOT / "telemetry" / "transport_recovery_live_report.json",
}
_TRANSPORT_REQUEST_CEILING = 8
_TRANSPORT_FAULT_WINDOW_SECONDS = 30.0
_TRANSPORT_UPSTREAM_TIMEOUT_SECONDS = 60.0
_TRANSPORT_SCENARIO_TIMEOUT_SECONDS = 90.0

_AUTOMATION_CHANNELS = (
    "pending_capability_dispatch",
    "pending_router_no_action",
    "pending_cart_mutation",
    "pending_placement",
    "pending_refund",
    "pending_cancel",
    "pending_return",
    "pending_profile_change",
    "pending_identity",
    "active_invocation",
    "pending_ack",
    "pending_clarification",
    "clarification_liveness",
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
SemanticRouteMismatchKind = Literal[
    "exact",
    "closed_failure",
    "conservative_clarification",
    "false_handoff",
    "missed_handoff",
    "continuation_mismatch",
    "capability_mismatch",
    "discriminator_mismatch",
    "clarification_reason_mismatch",
    "decision_mismatch",
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
        risk_domains = {
            self.projected_case.risk_domain,
            *(case.risk_domain for case in self.cases),
        }
        required_risk_domains = set(get_args(SemanticRouteRiskDomain))
        if risk_domains != required_risk_domains:
            missing = ", ".join(sorted(required_risk_domains - risk_domains))
            extra = ", ".join(sorted(risk_domains - required_risk_domains))
            detail = "; ".join(
                value
                for value in (
                    f"missing: {missing}" if missing else "",
                    f"unexpected: {extra}" if extra else "",
                )
                if value
            )
            raise ValueError(f"semantic-route eval must cover every risk domain ({detail})")
        return self


StructuralRouteLeaf = Literal[
    "modify_cart:remove",
    "modify_cart:set_quantity",
    "change_profile:contact",
    "clarify:missing_value",
]
StructuralContextDimension = Literal[
    "bound_customer",
    "active_capability",
    "cart_state",
    "recent_order",
]


def _structural_route_leaf(decision: RouteDecision) -> StructuralRouteLeaf | None:
    if decision.decision == "clarify" and decision.clarification_reason == "missing_value":
        return "clarify:missing_value"
    request = decision.request
    if isinstance(request, ModifyCart) and request.operation in {"remove", "set_quantity"}:
        return f"modify_cart:{request.operation}"
    if isinstance(request, ChangeProfile) and request.field == "contact":
        return "change_profile:contact"
    return None


def _structural_checklist_cell(context: RoutingContext) -> str:
    return "|".join(
        (
            f"bound={int(context.bound_customer)}",
            f"active={int(context.active_capability is not None)}",
            f"cart={int(context.cart_state == 'nonempty')}",
            f"recent={int(context.recent_order_count > 0)}",
        )
    )


class SemanticRouteStructuralCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    case_id: str
    scenario_class: SemanticRouteScenarioClass
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    context: RoutingContext
    proposal: RouteProposal
    expected: RouteDecision
    checklist_cell: str
    required_route_leaf: StructuralRouteLeaf | None = None
    counterfactual_of: str | None = None
    changed_dimension: StructuralContextDimension | None = None

    @model_validator(mode="after")
    def route_and_obligations_are_exact(self) -> SemanticRouteStructuralCase:
        if not self.case_id.strip() or self.case_id != self.case_id.strip():
            raise ValueError("structural route case id must be normalized and non-empty")
        materialized = materialize_route(self.context, self.proposal)
        if materialized != self.expected:
            raise ValueError("structural proposal must materialize to the expected route")
        if resolve_route(self.context, self.expected) != self.expected:
            raise ValueError("structural expected route must resolve as executable")
        if self.checklist_cell != _structural_checklist_cell(self.context):
            raise ValueError("structural checklist-cell declaration does not match context")
        if self.required_route_leaf is not None and (
            _structural_route_leaf(self.expected) != self.required_route_leaf
        ):
            raise ValueError("structural route-leaf declaration does not match expected route")
        if (self.counterfactual_of is None) != (self.changed_dimension is None):
            raise ValueError("counterfactual reference and changed dimension must appear together")
        return self


class SemanticRouteStructuralSupplement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: str
    purpose: Literal["development_only"]
    frozen_corpus_fingerprint: str
    route_schema_fingerprint: str
    registry_fingerprint: str
    context_projector_version: str
    cases: tuple[SemanticRouteStructuralCase, ...]

    @model_validator(mode="after")
    def supplement_is_closed_and_counterfactuals_are_exact(
        self,
    ) -> SemanticRouteStructuralSupplement:
        if self.schema_version != _SEMANTIC_ROUTE_STRUCTURAL_SCHEMA_VERSION:
            raise ValueError("unsupported semantic-route structural schema version")
        fingerprints = (
            self.frozen_corpus_fingerprint,
            self.route_schema_fingerprint,
            self.registry_fingerprint,
            self.context_projector_version,
        )
        if any(not value.strip() or value != value.strip() for value in fingerprints):
            raise ValueError("structural supplement fingerprints must be normalized")
        by_id = {case.case_id: case for case in self.cases}
        if not self.cases or len(by_id) != len(self.cases):
            raise ValueError("structural supplement requires non-empty unique case ids")
        for case in self.cases:
            if case.counterfactual_of is None:
                continue
            baseline = by_id.get(case.counterfactual_of)
            if baseline is None or baseline.case_id == case.case_id:
                raise ValueError("counterfactual must reference another supplement case")
            if (
                baseline.context.utterance != case.context.utterance
                or baseline.proposal != case.proposal
                or baseline.expected != case.expected
            ):
                raise ValueError("counterfactual must preserve utterance, proposal, and label")
            before = baseline.context.model_dump()
            after = case.context.model_dump()
            changed = {key for key in before if before[key] != after[key]}
            expected_changed = {
                "bound_customer": {"bound_customer"},
                "active_capability": {"active_capability"},
                "cart_state": {"cart_state"},
                "recent_order": {"recent_order_operation", "recent_order_count"},
            }[case.changed_dimension]
            if changed != expected_changed:
                raise ValueError("counterfactual must change exactly its declared dimension")
        leaves = tuple(
            case.required_route_leaf for case in self.cases if case.required_route_leaf is not None
        )
        if set(leaves) != set(get_args(StructuralRouteLeaf)) or len(leaves) != len(set(leaves)):
            raise ValueError("structural supplement must declare each missing route leaf once")
        return self


RoutingDataPartition = Literal["training", "calibration", "acceptance", "ood"]
RoutingDataSourceKind = Literal["fixture", "public", "synthetic", "real_caller"]
RoutingDataContextOrigin = Literal["authored_counterfactual", "projected_fixture"]
RoutingDataAvailabilityOrigin = Literal["registry_cohort", "rejection_counterfactual"]
RoutingDataProtectedFamily = Literal[
    "ambiguous",
    "compound",
    "negated",
    "quoted",
    "hypothetical",
    "corrective",
    "adversarial_instruction",
]


class RoutingDataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: RoutingDataSourceKind
    source_reference: str
    lineage_review_reference: str
    training_use_authority: Literal["license", "compliance_approval"]
    training_use_reference: str
    access_control_reference: str
    deletion_policy_reference: str
    retention_authority_reference: str
    storage_scope: Literal["repository_sanitized", "external_restricted"]
    sanitized: bool
    legacy_diagnostic_lineage: bool = False

    @model_validator(mode="after")
    def authority_is_complete_and_coherent(self) -> RoutingDataProvenance:
        references = (
            self.source_reference,
            self.lineage_review_reference,
            self.training_use_reference,
            self.access_control_reference,
            self.deletion_policy_reference,
            self.retention_authority_reference,
        )
        if any(not value.strip() or value != value.strip() for value in references):
            raise ValueError("routing-data provenance references must be normalized and non-empty")
        if not self.sanitized:
            raise ValueError("routing-data input must complete sanitization before eligibility")
        if self.source_kind == "public" and self.training_use_authority != "license":
            raise ValueError("public routing data requires licence-backed training authority")
        if self.source_kind == "real_caller":
            if self.training_use_authority != "compliance_approval":
                raise ValueError("real-caller data requires compliance-approved training use")
            if self.storage_scope != "external_restricted":
                raise ValueError("real-caller data must remain in external restricted storage")
        return self


class RoutingDataCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    example_id: str
    source_id: str
    source_cluster_id: str
    provenance: RoutingDataProvenance
    locale: str
    generator_version: str | None = None
    annotation_policy_version: str
    context: RoutingContext
    expected: RouteDecision
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    scenario_class: SemanticRouteScenarioClass
    context_origin: RoutingDataContextOrigin
    availability_origin: RoutingDataAvailabilityOrigin
    context_evidence_reference: str | None = None
    protected_families: tuple[RoutingDataProtectedFamily, ...] = ()
    review_status: Literal["approved", "adjudicated"]
    reviewer_id: str
    second_reviewer_id: str | None = None
    adjudicator_id: str | None = None
    adjudication_reference: str | None = None

    @model_validator(mode="after")
    def annotation_is_complete_and_executable(self) -> RoutingDataCase:
        identifiers = (
            self.example_id,
            self.source_id,
            self.source_cluster_id,
            self.locale,
            self.annotation_policy_version,
            self.reviewer_id,
        )
        if any(not value.strip() or value != value.strip() for value in identifiers):
            raise ValueError("routing-data identifiers must be normalized and non-empty")
        if self.generator_version is not None and (
            not self.generator_version.strip()
            or self.generator_version != self.generator_version.strip()
        ):
            raise ValueError("routing-data generator version must be normalized and non-empty")
        if len(set(self.protected_families)) != len(self.protected_families):
            raise ValueError("routing-data protected-family labels must be unique")
        if self.context_origin == "projected_fixture":
            if (
                self.context_evidence_reference is None
                or not self.context_evidence_reference.strip()
            ):
                raise ValueError("projected fixture context requires an evidence reference")
        elif self.context_evidence_reference is not None:
            raise ValueError("authored counterfactual cannot claim projected-fixture evidence")
        if self.protected_families:
            if (
                self.second_reviewer_id is None
                or not self.second_reviewer_id.strip()
                or self.second_reviewer_id == self.reviewer_id
            ):
                raise ValueError("protected routing data requires an independent second reviewer")
        elif self.second_reviewer_id is not None:
            raise ValueError("unprotected routing data must not claim a second-review contract")
        if self.review_status == "adjudicated":
            if (
                self.adjudicator_id is None
                or not self.adjudicator_id.strip()
                or self.adjudicator_id in {self.reviewer_id, self.second_reviewer_id}
                or self.adjudication_reference is None
                or not self.adjudication_reference.strip()
            ):
                raise ValueError(
                    "adjudicated routing data requires an independent adjudicator and record"
                )
        elif self.adjudicator_id is not None or self.adjudication_reference is not None:
            raise ValueError("approved routing data must not carry adjudication fields")
        if isinstance(resolve_route(self.context, self.expected), RoutingFailure):
            raise ValueError("routing-data ground truth must be executable in its context")
        return self


class RoutingDataPartitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: str
    partition: RoutingDataPartition
    route_schema_fingerprint: str
    registry_fingerprint: str
    context_projector_version: str
    available_capabilities: tuple[CapabilityId, ...]
    label_access: Literal["builder_visible", "steward_only"]
    steward_id: str | None = None
    cases: tuple[RoutingDataCase, ...]

    @model_validator(mode="after")
    def partition_contract_is_coherent(self) -> RoutingDataPartitionManifest:
        if self.schema_version != _ROUTING_DATA_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported routing-data manifest schema version")
        fingerprints = (
            self.route_schema_fingerprint,
            self.registry_fingerprint,
            self.context_projector_version,
        )
        if any(not value.strip() or value != value.strip() for value in fingerprints):
            raise ValueError("routing-data contract identities must be normalized and non-empty")
        if not self.available_capabilities or len(set(self.available_capabilities)) != len(
            self.available_capabilities
        ):
            raise ValueError("routing-data manifest requires unique available capabilities")
        if not self.cases:
            raise ValueError("routing-data partition must contain at least one case")
        ids = [case.example_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("routing-data example ids must be unique within a partition")
        sealed = self.partition in {"acceptance", "ood"}
        if sealed:
            if self.label_access != "steward_only":
                raise ValueError("acceptance and OOD labels must remain steward-only")
            if self.steward_id is None or not self.steward_id.strip():
                raise ValueError("acceptance and OOD partitions require a steward")
        elif self.label_access != "builder_visible" or self.steward_id is not None:
            raise ValueError("training and calibration partitions are builder-visible")
        for case in self.cases:
            if case.provenance.legacy_diagnostic_lineage:
                raise ValueError("legacy semantic diagnostic lineage is ineligible")
            if case.provenance.source_kind == "synthetic":
                if self.partition != "training" or case.generator_version is None:
                    raise ValueError(
                        "synthetic routing data is training-only and requires a generator version"
                    )
            elif case.generator_version is not None:
                raise ValueError("non-synthetic routing data must not carry a generator version")
            if case.availability_origin == "registry_cohort":
                if case.context.available_capabilities != self.available_capabilities:
                    raise ValueError(
                        "registry-cohort context must use the manifest capability set exactly"
                    )
            elif (
                self.partition != "ood"
                or case.context_origin != "authored_counterfactual"
                or case.expected != RouteDecision.clarify("unsupported_capability")
            ):
                raise ValueError(
                    "availability rejection counterfactuals are OOD unsupported-capability cases"
                )
        return self


RoutingDataReviewerRole = Literal["primary", "secondary", "adjudicator"]


class RoutingDataSourcePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_kind: RoutingDataSourceKind
    source_reference: str
    training_use_authority: Literal["license", "compliance_approval"]
    training_use_reference: str
    access_control_reference: str
    deletion_policy_reference: str
    retention_authority_reference: str
    sanitization_control_reference: str
    storage_scope: Literal["repository_sanitized", "external_restricted"]
    locales: tuple[str, ...]
    max_examples: int = Field(ge=1)
    estimated_cost_minor_units: int = Field(ge=0)

    @model_validator(mode="after")
    def source_is_lawful_and_bounded(self) -> RoutingDataSourcePlan:
        text = (
            self.source_id,
            self.source_reference,
            self.training_use_reference,
            self.access_control_reference,
            self.deletion_policy_reference,
            self.retention_authority_reference,
            self.sanitization_control_reference,
            *self.locales,
        )
        if any(not value.strip() or value != value.strip() for value in text):
            raise ValueError("routing-data source-plan fields must be normalized and non-empty")
        if not self.locales or len(set(self.locales)) != len(self.locales):
            raise ValueError("routing-data source plan requires unique locales")
        if self.source_kind == "public" and self.training_use_authority != "license":
            raise ValueError("public routing-data source plan requires licence authority")
        if self.source_kind == "real_caller" and (
            self.training_use_authority != "compliance_approval"
            or self.storage_scope != "external_restricted"
        ):
            raise ValueError(
                "real-caller source plan requires compliance approval and restricted storage"
            )
        return self


class RoutingDataReviewerCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: str
    role: RoutingDataReviewerRole
    review_capacity: int = Field(ge=1)

    @model_validator(mode="after")
    def reviewer_is_named(self) -> RoutingDataReviewerCapacity:
        if not self.reviewer_id.strip() or self.reviewer_id != self.reviewer_id.strip():
            raise ValueError("routing-data reviewer id must be normalized and non-empty")
        return self


class RoutingDataStopBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_examples: int = Field(ge=1)
    max_cost_minor_units: int = Field(ge=0)
    currency: str
    max_calendar_days: int = Field(ge=1)

    @model_validator(mode="after")
    def currency_is_closed_for_one_plan(self) -> RoutingDataStopBudget:
        if len(self.currency) != 3 or not self.currency.isascii() or not self.currency.isupper():
            raise ValueError("routing-data stop-budget currency must be an ISO-style code")
        return self


class RoutingDataCoverageSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    planned_examples: int = Field(ge=1)

    @model_validator(mode="after")
    def source_is_named(self) -> RoutingDataCoverageSource:
        if not self.source_id.strip() or self.source_id != self.source_id.strip():
            raise ValueError("routing-data coverage source id must be normalized and non-empty")
        return self


class RoutingDataCoverageCell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    cell_id: str
    partition: RoutingDataPartition
    expected: RouteDecision
    risk_class: SemanticRouteRiskClass
    risk_domain: SemanticRouteRiskDomain
    scenario_class: SemanticRouteScenarioClass
    locale: str
    source_allocations: tuple[RoutingDataCoverageSource, ...] = Field(min_length=1)
    context_origin: RoutingDataContextOrigin
    availability_origin: RoutingDataAvailabilityOrigin
    bound_customer: bool
    active_capability: CapabilityId | None = None
    recent_order_operation: OrderContextOperation | None = None
    recent_order_count: int = Field(default=0, ge=0)
    cart_state: Literal["empty", "nonempty"]
    available_capabilities: tuple[CapabilityId, ...] = Field(min_length=1)
    protected_families: tuple[RoutingDataProtectedFamily, ...] = ()
    planned_examples: int = Field(ge=1)
    min_independent_clusters: int = Field(ge=1)
    primary_reviewer_id: str
    secondary_reviewer_id: str | None = None

    @model_validator(mode="after")
    def coverage_cell_is_executable_and_reviewable(self) -> RoutingDataCoverageCell:
        text = (self.cell_id, self.locale, self.primary_reviewer_id)
        if any(not value.strip() or value != value.strip() for value in text):
            raise ValueError("routing-data coverage-cell ids must be normalized and non-empty")
        source_ids = [allocation.source_id for allocation in self.source_allocations]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("coverage-cell source allocations must be unique")
        if sum(allocation.planned_examples for allocation in self.source_allocations) != (
            self.planned_examples
        ):
            raise ValueError("coverage-cell source allocations must equal planned examples")
        if self.min_independent_clusters > len(self.source_allocations):
            raise ValueError(
                "independent-cluster requirement cannot exceed planned source allocations"
            )
        if len(set(self.available_capabilities)) != len(self.available_capabilities):
            raise ValueError("coverage-cell capabilities must be unique")
        if len(set(self.protected_families)) != len(self.protected_families):
            raise ValueError("coverage-cell protected families must be unique")
        if self.protected_families:
            if (
                self.risk_class != "critical"
                or self.secondary_reviewer_id is None
                or not self.secondary_reviewer_id.strip()
                or self.secondary_reviewer_id == self.primary_reviewer_id
            ):
                raise ValueError(
                    "protected coverage requires critical risk and an independent second reviewer"
                )
        elif self.secondary_reviewer_id is not None:
            raise ValueError("unprotected coverage must not reserve a second reviewer")
        context = RoutingContext(
            utterance=self.cell_id,
            bound_customer=self.bound_customer,
            active_capability=self.active_capability,
            recent_order_operation=self.recent_order_operation,
            recent_order_count=self.recent_order_count,
            cart_state=self.cart_state,
            available_capabilities=self.available_capabilities,
        )
        if isinstance(resolve_route(context, self.expected), RoutingFailure):
            raise ValueError("coverage-cell expected route must be executable in its context")
        if self.availability_origin == "rejection_counterfactual" and (
            self.partition != "ood"
            or self.context_origin != "authored_counterfactual"
            or self.expected != RouteDecision.clarify("unsupported_capability")
        ):
            raise ValueError(
                "availability rejection coverage must be an authored OOD no-action cell"
            )
        return self


class RoutingDataAcquisitionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: str
    route_schema_fingerprint: str
    registry_fingerprint: str
    context_projector_version: str
    available_capabilities: tuple[CapabilityId, ...]
    activation_risk_domains: tuple[SemanticRouteRiskDomain, ...]
    activation_locales: tuple[str, ...]
    required_decisions: tuple[Literal["clarify", "direct", "continue"], ...]
    required_protected_families: tuple[RoutingDataProtectedFamily, ...] = ()
    coverage_rationale: str
    annotation_pilot_reference: str
    annotation_guide_version: str
    approved_by: str
    approved_at: datetime
    steward_control_reference: str
    adjudication_owner_id: str
    adjudication_reserve: int = Field(ge=0)
    sources: tuple[RoutingDataSourcePlan, ...]
    reviewer_capacity: tuple[RoutingDataReviewerCapacity, ...]
    stop_budget: RoutingDataStopBudget
    coverage_cells: tuple[RoutingDataCoverageCell, ...]

    @model_validator(mode="after")
    def acquisition_is_complete_and_within_capacity(self) -> RoutingDataAcquisitionPlan:
        if self.schema_version != _ROUTING_DATA_ACQUISITION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported routing-data acquisition-plan schema version")
        text = (
            self.route_schema_fingerprint,
            self.registry_fingerprint,
            self.context_projector_version,
            self.coverage_rationale,
            self.annotation_pilot_reference,
            self.annotation_guide_version,
            self.approved_by,
            self.steward_control_reference,
            self.adjudication_owner_id,
            *self.activation_locales,
        )
        if any(not value.strip() or value != value.strip() for value in text):
            raise ValueError(
                "routing-data acquisition-plan fields must be normalized and non-empty"
            )
        if self.approved_at.tzinfo is None:
            raise ValueError("routing-data acquisition-plan timestamp must be timezone-aware")
        unique_fields = (
            (self.available_capabilities, "capabilities"),
            (self.activation_risk_domains, "risk domains"),
            (self.activation_locales, "locales"),
            (self.required_decisions, "decisions"),
            (self.required_protected_families, "protected families"),
        )
        if any(not values or len(set(values)) != len(values) for values, _ in unique_fields[:4]):
            raise ValueError("routing-data acquisition-plan activation dimensions must be unique")
        if len(set(self.required_protected_families)) != len(self.required_protected_families):
            raise ValueError("required protected families must be unique")
        if not self.sources or not self.reviewer_capacity or not self.coverage_cells:
            raise ValueError("routing-data acquisition plan requires sources, reviewers, and cells")
        source_by_id = {source.source_id: source for source in self.sources}
        reviewer_by_id = {reviewer.reviewer_id: reviewer for reviewer in self.reviewer_capacity}
        if len(source_by_id) != len(self.sources):
            raise ValueError("routing-data acquisition source ids must be unique")
        if len(reviewer_by_id) != len(self.reviewer_capacity):
            raise ValueError("routing-data reviewer ids must be unique across roles")
        if len({cell.cell_id for cell in self.coverage_cells}) != len(self.coverage_cells):
            raise ValueError("routing-data coverage-cell ids must be unique")
        coverage_contracts = [
            json.dumps(
                cell.model_dump(
                    mode="json",
                    exclude={
                        "cell_id",
                        "planned_examples",
                        "min_independent_clusters",
                        "primary_reviewer_id",
                        "secondary_reviewer_id",
                    },
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
            for cell in self.coverage_cells
        ]
        if len(set(coverage_contracts)) != len(coverage_contracts):
            raise ValueError("routing-data coverage obligations must not overlap exactly")
        adjudicator = reviewer_by_id.get(self.adjudication_owner_id)
        if adjudicator is None or adjudicator.role != "adjudicator":
            raise ValueError("adjudication owner must have adjudicator capacity")
        if adjudicator.review_capacity < self.adjudication_reserve:
            raise ValueError("adjudication reserve exceeds owner capacity")

        planned_by_source: dict[str, int] = {}
        assigned_reviews: dict[str, int] = {}
        for cell in self.coverage_cells:
            for allocation in cell.source_allocations:
                source = source_by_id.get(allocation.source_id)
                if source is None or cell.locale not in source.locales:
                    raise ValueError("coverage cell must use a declared source and locale")
                if cell.context_origin == "projected_fixture" and source.source_kind != "fixture":
                    raise ValueError("projected-fixture coverage requires a fixture source")
                planned_by_source[allocation.source_id] = (
                    planned_by_source.get(allocation.source_id, 0) + allocation.planned_examples
                )
            if (
                cell.availability_origin == "registry_cohort"
                and cell.available_capabilities != self.available_capabilities
            ):
                raise ValueError("registry-cohort cell must use the activation capability set")
            if cell.risk_domain not in self.activation_risk_domains:
                raise ValueError("coverage cell risk domain is outside the activation cohort")
            if cell.locale not in self.activation_locales:
                raise ValueError("coverage cell locale is outside the activation cohort")
            primary = reviewer_by_id.get(cell.primary_reviewer_id)
            if primary is None or primary.role != "primary":
                raise ValueError("coverage cell primary reviewer lacks primary capacity")
            assigned_reviews[cell.primary_reviewer_id] = (
                assigned_reviews.get(cell.primary_reviewer_id, 0) + cell.planned_examples
            )
            if cell.secondary_reviewer_id is not None:
                secondary = reviewer_by_id.get(cell.secondary_reviewer_id)
                if secondary is None or secondary.role != "secondary":
                    raise ValueError("protected coverage reviewer lacks secondary capacity")
                assigned_reviews[cell.secondary_reviewer_id] = (
                    assigned_reviews.get(cell.secondary_reviewer_id, 0) + cell.planned_examples
                )
        for reviewer_id, assigned in assigned_reviews.items():
            if assigned > reviewer_by_id[reviewer_id].review_capacity:
                raise ValueError("planned review load exceeds named reviewer capacity")
        for source_id, planned in planned_by_source.items():
            if planned > source_by_id[source_id].max_examples:
                raise ValueError("planned examples exceed source ceiling")
        if set(planned_by_source) != set(source_by_id):
            raise ValueError("every planned source must own at least one coverage cell")

        planned_examples = sum(cell.planned_examples for cell in self.coverage_cells)
        if planned_examples > self.stop_budget.max_examples:
            raise ValueError("planned examples exceed the stop budget")
        planned_cost = sum(source.estimated_cost_minor_units for source in self.sources)
        if planned_cost > self.stop_budget.max_cost_minor_units:
            raise ValueError("planned source cost exceeds the stop budget")

        partitions = set(get_args(RoutingDataPartition))
        for partition in partitions:
            for domain in self.activation_risk_domains:
                for locale in self.activation_locales:
                    if not any(
                        cell.partition == partition
                        and cell.risk_domain == domain
                        and cell.locale == locale
                        for cell in self.coverage_cells
                    ):
                        raise ValueError(
                            "every partition must cover each activation risk-domain/locale pair"
                        )
        for partition in ("training", "calibration", "acceptance"):
            for decision in self.required_decisions:
                if not any(
                    cell.partition == partition and cell.expected.decision == decision
                    for cell in self.coverage_cells
                ):
                    raise ValueError("required decisions must cover fitting and evaluation sets")
        for partition in ("acceptance", "ood"):
            for family in self.required_protected_families:
                if not any(
                    cell.partition == partition and family in cell.protected_families
                    for cell in self.coverage_cells
                ):
                    raise ValueError("protected families must cover acceptance and OOD")
        return self


class RoutingDataPilotAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    manifest_set_fingerprint: str
    acquisition_plan_fingerprint: str
    reviewer_id: str
    reviewed_at: datetime
    coverage_rationale: str
    legal_basis_rationale: str
    steward_control_reference: str

    @model_validator(mode="after")
    def authorization_is_reviewable(self) -> RoutingDataPilotAuthorization:
        if self.schema_version != _ROUTING_DATA_AUTHORIZATION_SCHEMA_VERSION:
            raise ValueError("unsupported routing-data authorization schema version")
        text = (
            self.manifest_set_fingerprint,
            self.acquisition_plan_fingerprint,
            self.reviewer_id,
            self.coverage_rationale,
            self.legal_basis_rationale,
            self.steward_control_reference,
        )
        if any(not value.strip() or value != value.strip() for value in text):
            raise ValueError("routing-data authorization fields must be normalized and non-empty")
        if self.reviewed_at.tzinfo is None:
            raise ValueError("routing-data authorization timestamp must be timezone-aware")
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
    execution_owner: str | None
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
    application: ApplicationSession
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
    routing_scope: Literal["ordinary", "confirmation_escape"] = "ordinary"

    @property
    def disposition(self) -> SemanticRouteDisposition:
        return _semantic_route_disposition(self.expected, self.attempt.resolution)

    @property
    def mismatch_kind(self) -> SemanticRouteMismatchKind:
        return _semantic_route_mismatch_kind(self.expected, self.attempt.resolution)


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


@dataclass(frozen=True)
class _SemanticSeriesVerdict:
    gate: SemanticRouteGate
    run_failures: tuple[tuple[str, ...], ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool | None:
        return None if self.gate == "diagnostic" else not self.failures


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


async def _checkpoint_observation(
    engine: ReasoningEngine,
) -> tuple[GraphObservation, int, tuple[str, ...]]:
    # Evaluator-only checkpoint inspection. Adding a production introspection API solely for
    # tests would widen ReasoningEngine's application contract.
    snapshot = await engine._graph.aget_state(engine._config)
    values = snapshot.values
    handover = values.get("handover")
    return (
        GraphObservation(
            execution_owner=values.get("execution_owner"),
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


async def _build_eval_runtime(
    config: MerchantConfig,
    response_model: BaseChatModel,
    reasoning_model: BaseChatModel,
    *,
    fixture_config_root: Path,
    routing_model: BaseChatModel,
    routing_recognizer: RoutingRecognizer | None = None,
    thread_id: str,
    structured_output_method: StructuredOutputMethod,
    routing_structured_output_method: StructuredOutputMethod,
    operational_telemetry_sink: TelemetrySink,
    routing_evidence_sink: TelemetrySink,
) -> EvalRuntime:
    tenant = TenantContext(
        tenant_id=config.merchant_id,
        config_version=config_version(config.model_dump(mode="json")),
        policy=config.policies.to_policy_context(),
    )
    services = build_fixture_tenant_services(
        fixture_config_root,
        tenant,
        telemetry=TenantTelemetry(
            config.merchant_id,
            operational_telemetry_sink,
            routing_evidence_sink,
        ),
        checkpointer=build_checkpointer(),
    )

    def routing_factory(registry: CapabilityRegistry) -> RoutingRecognizer:
        return routing_recognizer or SemanticRouter(
            routing_model,
            selection=config.llm.routing,
            structured_output_method=routing_structured_output_method,
            timeout_seconds=config.runtime.semantic_router_timeout_seconds,
            input_max_chars=config.runtime.semantic_router_input_max_chars,
            registry=registry,
        )

    async def session_state_factory(tenant_context, tenant_services):
        return await build_in_memory_session_state(
            tenant_context,
            tenant_services,
            thread_id=thread_id,
        )

    application = await build_application_session(
        tenant,
        ApplicationSettings.from_merchant_config(config),
        ApplicationModels(
            response=response_model,
            reasoning=reasoning_model,
            response_structured_output_method=structured_output_method,
        ),
        services,
        deployment_id=_EVAL_DEPLOYMENT_ID,
        routing_factory=routing_factory,
        session_state_factory=session_state_factory,
    )
    state = application.state
    return EvalRuntime(
        application=application,
        graph=application.assembly.graph,
        capability_registry=application.assembly.capability_registry,
        engine=application.engine,
        store=services.order_store,
        cart_store=state.cart_store,
        profile_store=services.profile_store,
        otp=services.otp,
        verification=state.verification_store,
        identity_store=state.identity_store,
        recent_orders=state.recent_orders,
        caller_context=state.caller_context,
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
        state, completed_tool_calls, admitted_user_messages = await _checkpoint_observation(engine)
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

    code_node = "cart_view_render"
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
    for case in order_reference["accepted_labelled_stt"]:
        expected = tuple(case["expected_order_refs"])
        actual = caller_stated_order_ids(case["utterance"])
        if actual != expected:
            failures.append(f"labelled STT order-reference set was rejected: {case['utterance']!r}")
    for utterance in order_reference["rejected_stt"]:
        if caller_stated_order_ids(utterance):
            failures.append(
                f"weak or ambiguous STT order-reference set was accepted: {utterance!r}"
            )
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


def _load_semantic_route_structural_supplement(
    path: Path = _SEMANTIC_ROUTE_STRUCTURAL_SUPPLEMENT_PATH,
) -> SemanticRouteStructuralSupplement:
    supplement = SemanticRouteStructuralSupplement.model_validate(load_yaml_layer(path))
    _validate_semantic_route_structural_supplement(
        _load_semantic_route_corpus(),
        supplement,
    )
    return supplement


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


def _parse_provider_model(value: str) -> ProviderModel:
    provider, separator, model = value.partition(":")
    provider = provider.strip()
    model = model.strip()
    if not separator or not provider or not model:
        raise argparse.ArgumentTypeError("model selection must use provider:model")
    try:
        return ProviderModel(provider=provider, model=model)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError("model selection must use provider:model") from exc


def _resolve_conformance_target(selection: ProviderModel) -> ProviderModel:
    targets = load_conformance_targets(_CONFIG_ROOT / "conformance" / "targets.yaml").targets
    for target in targets:
        if (target.provider, target.model) == (selection.provider, selection.model):
            return target
    raise ValueError(
        f"semantic routing candidate {selection.provider}:{selection.model} is not a "
        "configured conformance target"
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
                routing_scope=case.context.routing_scope,
            )
        )
    return tuple(results)


def _ordered_capabilities(
    groups: Sequence[Sequence[CapabilityId]],
) -> tuple[CapabilityId, ...]:
    return tuple(dict.fromkeys(capability for group in groups for capability in group))


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


def _semantic_route_mismatch_kind(
    expected: RouteDecision,
    actual: RouteDecision | RoutingFailure,
) -> SemanticRouteMismatchKind:
    if actual == expected:
        return "exact"
    if isinstance(actual, RoutingFailure):
        return "closed_failure"
    if _is_request_person(actual) and not _is_request_person(expected):
        return "false_handoff"
    if _is_request_person(expected) and not _is_request_person(actual):
        return "missed_handoff"
    if expected.decision == "continue" or actual.decision == "continue":
        return "continuation_mismatch"
    if actual.decision == "clarify" and expected.decision != "clarify":
        return "conservative_clarification"
    if expected.decision == "clarify" and actual.decision == "clarify":
        return "clarification_reason_mismatch"
    if expected.decision == "direct" and actual.decision == "direct":
        assert expected.request is not None and actual.request is not None
        if expected.request.kind != actual.request.kind:
            return "capability_mismatch"
        return "discriminator_mismatch"
    return "decision_mismatch"


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
    mismatch_kinds = {
        mismatch_kind: sum(result.mismatch_kind == mismatch_kind for result in results)
        for mismatch_kind in get_args(SemanticRouteMismatchKind)
    }
    return {
        "cases": len(results),
        "dispositions": dispositions,
        "mismatch_kinds": mismatch_kinds,
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
        "routing_scope",
    ],
) -> dict[str, dict[str, object]]:
    values = dict.fromkeys(getattr(result, attribute) for result in results)
    return {
        value: _semantic_route_group(
            [result for result in results if getattr(result, attribute) == value]
        )
        for value in values
    }


def _is_request_person(resolution: RouteDecision | RoutingFailure) -> bool:
    return (
        isinstance(resolution, RouteDecision)
        and resolution.decision == "direct"
        and resolution.request is not None
        and resolution.request.kind == CapabilityId.REQUEST_PERSON
    )


def _confirmation_escape_metrics(
    results: Sequence[SemanticRouteCaseResult],
) -> dict[str, int]:
    scoped = [result for result in results if result.routing_scope == "confirmation_escape"]
    expected_escapes = [result for result in scoped if _is_request_person(result.expected)]
    false_escapes = [
        result
        for result in scoped
        if not _is_request_person(result.expected) and _is_request_person(result.attempt.resolution)
    ]
    return {
        "cases": len(scoped),
        "expected_escapes": len(expected_escapes),
        "exact_escapes": sum(
            _is_request_person(result.attempt.resolution) for result in expected_escapes
        ),
        "false_escapes": len(false_escapes),
    }


def _semantic_model_report(
    results: Sequence[SemanticRouteCaseResult],
    *,
    repetition: int | None,
) -> dict[str, object]:
    if not results:
        raise ValueError("semantic-route model report requires at least one result")
    first = results[0].attempt
    expected_identity = (
        first.provider,
        first.model,
        first.reasoning_effort,
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
            attempt.reasoning_effort,
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
    report: dict[str, object] = {
        "provider": first.provider,
        "model": first.model,
        "reasoning_effort": first.reasoning_effort,
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
        "by_routing_scope": _group_semantic_route_results(results, "routing_scope"),
        "confirmation_escape": _confirmation_escape_metrics(results),
    }
    if repetition is not None:
        report["cases"] = [
            {
                "repetition": repetition,
                "case_id": result.case_id,
                "scenario_class": result.scenario_class,
                "risk_class": result.risk_class,
                "risk_domain": result.risk_domain,
                "evaluation_split": result.evaluation_split,
                "routing_scope": result.routing_scope,
                "disposition": result.disposition,
                "mismatch_kind": result.mismatch_kind,
                "expected": _route_signature(result.expected),
                "actual": _route_signature(result.attempt.resolution),
                "latency_ms": result.attempt.elapsed_ms,
                "provider_call_outcome": result.attempt.provider_call_outcome,
                "input_tokens": result.attempt.input_tokens,
                "cache_read_tokens": result.attempt.cache_read_tokens,
                "output_tokens": result.attempt.output_tokens,
            }
            for result in results
        ]
    return report


def _json_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_fingerprint(corpus: SemanticRouteEvalCorpus) -> str:
    return _json_fingerprint(corpus.model_dump(mode="json"))


def _semantic_route_corpus_contexts(
    corpus: SemanticRouteEvalCorpus,
) -> tuple[RoutingContext, ...]:
    return (
        *(case.context for case in corpus.cases),
        corpus.projected_case.expected_context,
    )


def _semantic_route_corpus_decisions(
    corpus: SemanticRouteEvalCorpus,
) -> tuple[RouteDecision, ...]:
    return (
        *(case.expected for case in corpus.cases),
        corpus.projected_case.expected,
    )


def _all_structural_checklist_cells() -> frozenset[str]:
    return frozenset(
        "|".join(
            (
                f"bound={bound}",
                f"active={active}",
                f"cart={cart}",
                f"recent={recent}",
            )
        )
        for bound in (0, 1)
        for active in (0, 1)
        for cart in (0, 1)
        for recent in (0, 1)
    )


def _canonical_route_signatures(
    decisions: Sequence[RouteDecision],
) -> tuple[dict[str, str | None], ...]:
    by_key = {
        json.dumps(_route_signature(decision), sort_keys=True): _route_signature(decision)
        for decision in decisions
    }
    return tuple(by_key[key] for key in sorted(by_key))


def _validate_semantic_route_structural_supplement(
    corpus: SemanticRouteEvalCorpus,
    supplement: SemanticRouteStructuralSupplement,
) -> None:
    registry = _build_frontline_capability_registry()
    expected_contract = {
        "frozen_corpus_fingerprint": _corpus_fingerprint(corpus),
        "route_schema_fingerprint": ROUTE_SCHEMA_FINGERPRINT,
        "registry_fingerprint": registry_fingerprint(registry),
        "context_projector_version": CONTEXT_PROJECTOR_VERSION,
    }
    actual_contract = {
        "frozen_corpus_fingerprint": supplement.frozen_corpus_fingerprint,
        "route_schema_fingerprint": supplement.route_schema_fingerprint,
        "registry_fingerprint": supplement.registry_fingerprint,
        "context_projector_version": supplement.context_projector_version,
    }
    if actual_contract != expected_contract:
        raise ValueError("structural supplement contract fingerprint is stale")

    frozen_cells = {
        _structural_checklist_cell(context) for context in _semantic_route_corpus_contexts(corpus)
    }
    missing_cells = _all_structural_checklist_cells() - frozen_cells
    supplement_cells = [case.checklist_cell for case in supplement.cases]
    if {cell: supplement_cells.count(cell) for cell in missing_cells} != dict.fromkeys(
        missing_cells, 1
    ):
        raise ValueError("structural supplement must cover each missing checklist cell once")

    frozen_leaves = {
        leaf
        for decision in _semantic_route_corpus_decisions(corpus)
        if (leaf := _structural_route_leaf(decision)) is not None
    }
    if frozen_leaves:
        raise ValueError("frozen corpus unexpectedly contains a declared missing route leaf")
    union_signatures = _canonical_route_signatures(
        (
            *_semantic_route_corpus_decisions(corpus),
            *(case.expected for case in supplement.cases),
        )
    )
    if len(union_signatures) != _SEMANTIC_ROUTE_CANONICAL_LEAF_COUNT:
        raise ValueError("structural route-signature union has the wrong canonical leaf count")


def _semantic_route_structural_report(
    corpus: SemanticRouteEvalCorpus,
    supplement: SemanticRouteStructuralSupplement,
) -> dict[str, object]:
    _validate_semantic_route_structural_supplement(corpus, supplement)
    frozen_cells = frozenset(
        _structural_checklist_cell(context) for context in _semantic_route_corpus_contexts(corpus)
    )
    supplement_cells = frozenset(case.checklist_cell for case in supplement.cases)
    union_cells = frozen_cells | supplement_cells
    signatures = _canonical_route_signatures(
        (
            *_semantic_route_corpus_decisions(corpus),
            *(case.expected for case in supplement.cases),
        )
    )
    return {
        "schema_version": _SEMANTIC_ROUTE_STRUCTURAL_REPORT_SCHEMA_VERSION,
        "purpose": supplement.purpose,
        "qualification": None,
        "frozen_corpus_fingerprint": _corpus_fingerprint(corpus),
        "supplement_fingerprint": _json_fingerprint(supplement.model_dump(mode="json")),
        "route_schema_fingerprint": supplement.route_schema_fingerprint,
        "registry_fingerprint": supplement.registry_fingerprint,
        "context_projector_version": supplement.context_projector_version,
        "canonical_route_leaf_count": len(signatures),
        "canonical_route_leaves": signatures,
        "checklist": {
            "dimensions": (
                "bound_customer",
                "active_present",
                "cart_nonempty",
                "recent_present",
            ),
            "total_cells": len(_all_structural_checklist_cells()),
            "frozen_cells": tuple(sorted(frozen_cells)),
            "supplement_cells": tuple(sorted(supplement_cells - frozen_cells)),
            "union_cells": tuple(sorted(union_cells)),
            "uncovered_cells": tuple(sorted(_all_structural_checklist_cells() - union_cells)),
        },
    }


def _run_semantic_route_structural_coverage(report_path: Path) -> int:
    corpus = _load_semantic_route_corpus()
    supplement = _load_semantic_route_structural_supplement()
    report = _semantic_route_structural_report(corpus, supplement)
    _write_json_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _routing_data_contract() -> dict[str, object]:
    registry = _build_frontline_capability_registry()
    return {
        "route_schema_fingerprint": ROUTE_SCHEMA_FINGERPRINT,
        "registry_fingerprint": registry_fingerprint(registry),
        "context_projector_version": CONTEXT_PROJECTOR_VERSION,
        "available_capabilities": [capability.value for capability in registry.capability_ids],
    }


def _load_routing_data_manifests(
    paths: Sequence[Path],
) -> tuple[tuple[Path, RoutingDataPartitionManifest], ...]:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("routing-data partition paths must be unique")
    return tuple(
        (path, RoutingDataPartitionManifest.model_validate(load_yaml_layer(path)))
        for path in resolved
    )


def _load_routing_data_authorization(path: Path) -> RoutingDataPilotAuthorization:
    return RoutingDataPilotAuthorization.model_validate(load_yaml_layer(path))


def _load_routing_data_acquisition_plan(path: Path) -> RoutingDataAcquisitionPlan:
    return RoutingDataAcquisitionPlan.model_validate(load_yaml_layer(path))


def _routing_data_manifest_set_fingerprint(
    manifests: Sequence[tuple[Path, RoutingDataPartitionManifest]],
) -> str:
    return _json_fingerprint(
        [
            manifest.model_dump(mode="json")
            for _, manifest in sorted(manifests, key=lambda item: item[1].partition)
        ]
    )


def _routing_data_acquisition_plan_fingerprint(plan: RoutingDataAcquisitionPlan) -> str:
    return _json_fingerprint(plan.model_dump(mode="json"))


def _routing_data_case_matches_cell(
    case: RoutingDataCase,
    cell: RoutingDataCoverageCell,
) -> bool:
    context = case.context
    return (
        _route_signature(case.expected) == _route_signature(cell.expected)
        and case.risk_class == cell.risk_class
        and case.risk_domain == cell.risk_domain
        and case.scenario_class == cell.scenario_class
        and case.locale == cell.locale
        and case.source_id in {allocation.source_id for allocation in cell.source_allocations}
        and case.context_origin == cell.context_origin
        and case.availability_origin == cell.availability_origin
        and context.bound_customer == cell.bound_customer
        and context.active_capability == cell.active_capability
        and context.recent_order_operation == cell.recent_order_operation
        and context.recent_order_count == cell.recent_order_count
        and context.cart_state == cell.cart_state
        and context.available_capabilities == cell.available_capabilities
        and set(cell.protected_families).issubset(case.protected_families)
        and case.reviewer_id == cell.primary_reviewer_id
        and case.second_reviewer_id == cell.secondary_reviewer_id
    )


def _routing_data_acquisition_blockers(
    plan: RoutingDataAcquisitionPlan,
    by_partition: dict[str, RoutingDataPartitionManifest],
) -> list[str]:
    blockers: list[str] = []
    source_by_id = {source.source_id: source for source in plan.sources}
    reviewer_by_id = {reviewer.reviewer_id: reviewer for reviewer in plan.reviewer_capacity}
    matched_examples: set[str] = set()
    matched_cells: dict[str, set[str]] = {}
    source_examples: dict[str, int] = {}
    reviewer_examples: dict[str, int] = {}
    adjudicated_examples = 0

    for partition, manifest in by_partition.items():
        for case in manifest.cases:
            source = source_by_id.get(case.source_id)
            if source is None:
                blockers.append(f"unplanned_source:{case.source_id}")
                continue
            provenance = case.provenance
            source_contract = (
                provenance.source_kind == source.source_kind
                and provenance.source_reference == source.source_reference
                and provenance.training_use_authority == source.training_use_authority
                and provenance.training_use_reference == source.training_use_reference
                and provenance.access_control_reference == source.access_control_reference
                and provenance.deletion_policy_reference == source.deletion_policy_reference
                and provenance.retention_authority_reference == source.retention_authority_reference
                and provenance.storage_scope == source.storage_scope
                and case.locale in source.locales
            )
            if not source_contract:
                blockers.append(f"source_contract_mismatch:{case.source_id}")
            source_examples[case.source_id] = source_examples.get(case.source_id, 0) + 1
            reviewer_examples[case.reviewer_id] = reviewer_examples.get(case.reviewer_id, 0) + 1
            if case.second_reviewer_id is not None:
                reviewer_examples[case.second_reviewer_id] = (
                    reviewer_examples.get(case.second_reviewer_id, 0) + 1
                )
            if case.review_status == "adjudicated" and (
                case.adjudicator_id != plan.adjudication_owner_id
            ):
                blockers.append(f"adjudication_owner_mismatch:{case.example_id}")
            if case.review_status == "adjudicated":
                adjudicated_examples += 1

        cells = [cell for cell in plan.coverage_cells if cell.partition == partition]
        for cell in cells:
            matching = [
                case for case in manifest.cases if _routing_data_case_matches_cell(case, cell)
            ]
            if len(matching) < cell.planned_examples:
                blockers.append(f"coverage_examples_unmet:{cell.cell_id}")
            clusters = {case.source_cluster_id for case in matching}
            if len(clusters) < cell.min_independent_clusters:
                blockers.append(f"coverage_clusters_unmet:{cell.cell_id}")
            for allocation in cell.source_allocations:
                source_matches = [
                    case for case in matching if case.source_id == allocation.source_id
                ]
                if len(source_matches) < allocation.planned_examples:
                    blockers.append(
                        f"coverage_source_examples_unmet:{cell.cell_id}:{allocation.source_id}"
                    )
            for case in matching:
                matched_cells.setdefault(case.example_id, set()).add(cell.cell_id)
            matched_examples.update(case.example_id for case in matching)

    for source_id, count in source_examples.items():
        source = source_by_id.get(source_id)
        if source is not None and count > source.max_examples:
            blockers.append(f"source_ceiling_exceeded:{source_id}")
    for reviewer_id, count in reviewer_examples.items():
        reviewer = reviewer_by_id.get(reviewer_id)
        if reviewer is None:
            blockers.append(f"unplanned_reviewer:{reviewer_id}")
        elif count > reviewer.review_capacity:
            blockers.append(f"reviewer_capacity_exceeded:{reviewer_id}")
    if adjudicated_examples > plan.adjudication_reserve:
        blockers.append("adjudication_reserve_exceeded")
    for example_id, cells in matched_cells.items():
        if len(cells) > 1:
            blockers.append(f"overlapping_coverage_cells:{example_id}")
    all_cases = [case for manifest in by_partition.values() for case in manifest.cases]
    if len(all_cases) > plan.stop_budget.max_examples:
        blockers.append("acquired_examples_exceed_stop_budget")
    for case in all_cases:
        if case.example_id not in matched_examples:
            blockers.append(f"unplanned_example:{case.example_id}")
    return blockers


def _routing_data_acquisition_summary(plan: RoutingDataAcquisitionPlan) -> dict[str, object]:
    return {
        "status": "authorized",
        "artifact_fingerprint": _routing_data_acquisition_plan_fingerprint(plan),
        "approved_by": plan.approved_by,
        "approved_at": plan.approved_at.isoformat(),
        "annotation_guide_version": plan.annotation_guide_version,
        "coverage_cells": len(plan.coverage_cells),
        "planned_examples": sum(cell.planned_examples for cell in plan.coverage_cells),
        "required_independent_clusters": sum(
            cell.min_independent_clusters for cell in plan.coverage_cells
        ),
        "protected_second_reviews": sum(
            cell.planned_examples for cell in plan.coverage_cells if cell.protected_families
        ),
        "adjudication_reserve": plan.adjudication_reserve,
        "planned_source_cost_minor_units": sum(
            source.estimated_cost_minor_units for source in plan.sources
        ),
        "stop_budget": plan.stop_budget.model_dump(mode="json"),
    }


def _routing_data_partition_summary(
    manifest: RoutingDataPartitionManifest,
) -> dict[str, object]:
    cases = manifest.cases
    summary: dict[str, object] = {
        "artifact_fingerprint": _json_fingerprint(manifest.model_dump(mode="json")),
        "examples": len(cases),
        "independent_clusters": len({case.source_cluster_id for case in cases}),
        "independent_sources": len({case.source_id for case in cases}),
        "locales": sorted({case.locale for case in cases}),
        "source_kinds": sorted({case.provenance.source_kind for case in cases}),
        "training_use_authorities": sorted(
            {case.provenance.training_use_authority for case in cases}
        ),
        "context_origins": sorted({case.context_origin for case in cases}),
        "availability_origins": sorted({case.availability_origin for case in cases}),
        "context_coverage": {
            "bound_customer": sorted({case.context.bound_customer for case in cases}),
            "active_capability": sorted(
                {
                    (
                        case.context.active_capability.value
                        if case.context.active_capability is not None
                        else "none"
                    )
                    for case in cases
                }
            ),
            "recent_order_operation": sorted(
                {case.context.recent_order_operation or "none" for case in cases}
            ),
            "recent_order_count": sorted({case.context.recent_order_count for case in cases}),
            "cart_state": sorted({case.context.cart_state for case in cases}),
        },
    }
    if manifest.label_access == "builder_visible":
        signatures = {
            json.dumps(_route_signature(case.expected), sort_keys=True, separators=(",", ":"))
            for case in cases
        }
        summary.update(
            {
                "route_shapes": [json.loads(signature) for signature in sorted(signatures)],
                "risk_classes": sorted({case.risk_class for case in cases}),
                "risk_domains": sorted({case.risk_domain for case in cases}),
                "scenario_classes": sorted({case.scenario_class for case in cases}),
                "protected_families": sorted(
                    {family for case in cases for family in case.protected_families}
                ),
            }
        )
    return summary


def _routing_data_readiness_report(
    manifests: Sequence[tuple[Path, RoutingDataPartitionManifest]],
    *,
    acquisition_plan: RoutingDataAcquisitionPlan | None = None,
    acquisition_plan_path: Path | None = None,
    authorization: RoutingDataPilotAuthorization | None = None,
    authorization_path: Path | None = None,
) -> dict[str, object]:
    contract = _routing_data_contract()
    expected_capabilities = tuple(
        CapabilityId(value) for value in contract["available_capabilities"]
    )
    required_partitions = set(get_args(RoutingDataPartition))
    by_partition: dict[str, RoutingDataPartitionManifest] = {}
    blockers: list[str] = []
    example_partitions: dict[str, str] = {}
    source_clusters: dict[str, str] = {}
    cluster_partitions: dict[str, str] = {}
    project_root = _CONFIG_ROOT.parent.resolve()
    acquisition_plan_fingerprint: str | None = None
    acquisition_plan_valid = acquisition_plan is not None

    if acquisition_plan is None:
        blockers.append("acquisition_plan_missing")
    else:
        acquisition_plan_fingerprint = _routing_data_acquisition_plan_fingerprint(acquisition_plan)
        if acquisition_plan_path is None or acquisition_plan_path.resolve().is_relative_to(
            project_root
        ):
            blockers.append("acquisition_plan_not_path_isolated")
            acquisition_plan_valid = False
        if acquisition_plan.route_schema_fingerprint != contract["route_schema_fingerprint"]:
            blockers.append("acquisition_plan_route_schema_mismatch")
            acquisition_plan_valid = False
        if acquisition_plan.registry_fingerprint != contract["registry_fingerprint"]:
            blockers.append("acquisition_plan_registry_mismatch")
            acquisition_plan_valid = False
        if acquisition_plan.context_projector_version != contract["context_projector_version"]:
            blockers.append("acquisition_plan_projector_mismatch")
            acquisition_plan_valid = False
        if acquisition_plan.available_capabilities != expected_capabilities:
            blockers.append("acquisition_plan_capability_cohort_mismatch")
            acquisition_plan_valid = False

    for path, manifest in manifests:
        if manifest.partition in by_partition:
            blockers.append(f"duplicate_partition:{manifest.partition}")
            continue
        by_partition[manifest.partition] = manifest
        if manifest.route_schema_fingerprint != contract["route_schema_fingerprint"]:
            blockers.append(f"route_schema_mismatch:{manifest.partition}")
        if manifest.registry_fingerprint != contract["registry_fingerprint"]:
            blockers.append(f"registry_mismatch:{manifest.partition}")
        if manifest.context_projector_version != contract["context_projector_version"]:
            blockers.append(f"context_projector_mismatch:{manifest.partition}")
        if manifest.available_capabilities != expected_capabilities:
            blockers.append(f"capability_cohort_mismatch:{manifest.partition}")
        if path.is_relative_to(project_root) and any(
            case.provenance.storage_scope == "external_restricted" for case in manifest.cases
        ):
            blockers.append(f"restricted_data_inside_repository:{manifest.partition}")
        if manifest.partition in {"acceptance", "ood"} and path.is_relative_to(project_root):
            blockers.append(f"sealed_labels_inside_repository:{manifest.partition}")
        for case in manifest.cases:
            prior_partition = example_partitions.setdefault(case.example_id, manifest.partition)
            if prior_partition != manifest.partition:
                blockers.append(f"example_partition_leak:{case.example_id}")
            prior_cluster = source_clusters.setdefault(
                case.source_id,
                case.source_cluster_id,
            )
            if prior_cluster != case.source_cluster_id:
                blockers.append(f"source_cluster_conflict:{case.source_id}")
            cluster_partition = cluster_partitions.setdefault(
                case.source_cluster_id,
                manifest.partition,
            )
            if cluster_partition != manifest.partition:
                blockers.append(f"cluster_partition_leak:{case.source_cluster_id}")

    for partition in sorted(required_partitions - set(by_partition)):
        blockers.append(f"missing_partition:{partition}")

    if acquisition_plan is not None and acquisition_plan_valid:
        blockers.extend(_routing_data_acquisition_blockers(acquisition_plan, by_partition))

    manifest_set_fingerprint = _routing_data_manifest_set_fingerprint(manifests)
    if authorization is None:
        blockers.append("pilot_authorization_missing")
    else:
        if authorization.manifest_set_fingerprint != manifest_set_fingerprint:
            blockers.append("pilot_authorization_manifest_mismatch")
        if authorization.acquisition_plan_fingerprint != acquisition_plan_fingerprint:
            blockers.append("pilot_authorization_plan_mismatch")
        if authorization_path is None or authorization_path.resolve().is_relative_to(project_root):
            blockers.append("pilot_authorization_not_path_isolated")

    unique_blockers = list(dict.fromkeys(blockers))
    disposition = "classifier_pilot_authorized" if not unique_blockers else "insufficient_evidence"
    return {
        "schema_version": _ROUTING_DATA_REPORT_SCHEMA_VERSION,
        "run_at": datetime.now(tz=UTC).isoformat(),
        "disposition": disposition,
        "contract": contract,
        "manifest_set_fingerprint": manifest_set_fingerprint,
        "acquisition_plan": (
            None
            if acquisition_plan is None
            else {
                **_routing_data_acquisition_summary(acquisition_plan),
                "status": "authorized" if acquisition_plan_valid else "rejected",
            }
        ),
        "partitions": {
            partition: _routing_data_partition_summary(manifest)
            for partition, manifest in sorted(by_partition.items())
        },
        "blockers": unique_blockers,
        "authorization": (
            None
            if authorization is None
            else {
                "reviewer_id": authorization.reviewer_id,
                "reviewed_at": authorization.reviewed_at.isoformat(),
            }
        ),
    }


def _routing_data_invalid_report(reason: str) -> dict[str, object]:
    return {
        "schema_version": _ROUTING_DATA_REPORT_SCHEMA_VERSION,
        "run_at": datetime.now(tz=UTC).isoformat(),
        "disposition": "insufficient_evidence",
        "contract": _routing_data_contract(),
        "manifest_set_fingerprint": None,
        "acquisition_plan": None,
        "partitions": {},
        "blockers": [reason],
        "authorization": None,
    }


def _run_routing_data_audit(
    manifest_paths: Sequence[Path],
    *,
    acquisition_plan_path: Path | None,
    authorization_path: Path | None,
    report_path: Path,
) -> int:
    if acquisition_plan_path is None:
        acquisition_plan = None
    else:
        try:
            acquisition_plan = _load_routing_data_acquisition_plan(acquisition_plan_path)
        except ConfigError:
            report = _routing_data_invalid_report("acquisition_plan_file_invalid")
            _write_json_report(report_path, report)
            print("[routing_data] insufficient_evidence: acquisition_plan_file_invalid")
            return 1
        except ValidationError:
            report = _routing_data_invalid_report("acquisition_plan_schema_invalid")
            _write_json_report(report_path, report)
            print("[routing_data] insufficient_evidence: acquisition_plan_schema_invalid")
            return 1
    try:
        manifests = _load_routing_data_manifests(manifest_paths)
    except ConfigError:
        report = _routing_data_invalid_report("manifest_file_invalid")
    except ValidationError:
        report = _routing_data_invalid_report("manifest_schema_invalid")
    except ValueError:
        report = _routing_data_invalid_report("manifest_set_invalid")
    else:
        if authorization_path is None:
            authorization = None
        else:
            try:
                authorization = _load_routing_data_authorization(authorization_path)
            except ConfigError:
                report = _routing_data_invalid_report("authorization_file_invalid")
                _write_json_report(report_path, report)
                print("[routing_data] insufficient_evidence: authorization_file_invalid")
                return 1
            except ValidationError:
                report = _routing_data_invalid_report("authorization_schema_invalid")
                _write_json_report(report_path, report)
                print("[routing_data] insufficient_evidence: authorization_schema_invalid")
                return 1
        report = _routing_data_readiness_report(
            manifests,
            acquisition_plan=acquisition_plan,
            acquisition_plan_path=acquisition_plan_path,
            authorization=authorization,
            authorization_path=authorization_path,
        )
    _write_json_report(report_path, report)
    print(f"[routing_data] {report['disposition']}")
    for blocker in report["blockers"]:
        print(f"    {blocker}")
    print(f"    sanitized report: {report_path}")
    return 0 if report["disposition"] == "classifier_pilot_authorized" else 1


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
            result.routing_scope,
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
    confirmation = _confirmation_escape_metrics(candidate)
    if confirmation["false_escapes"]:
        failures.append("candidate produced a false person escape during confirmation")
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
    required_domains = {result.risk_domain for result in candidate}
    acceptance_domains = {result.risk_domain for result in acceptance}
    missing_domains = required_domains - acceptance_domains
    if missing_domains:
        failures.append("acceptance omitted risk domains: " + ", ".join(sorted(missing_domains)))

    required_capabilities = set(projection.expected_context.available_capabilities)
    accepted_capabilities = {
        result.expected.request.kind
        for result in acceptance
        if result.expected.decision == "direct" and result.expected.request is not None
    }
    missing_capabilities = required_capabilities - accepted_capabilities
    if missing_capabilities:
        failures.append(
            "acceptance omitted capabilities: "
            + ", ".join(sorted(capability.value for capability in missing_capabilities))
        )
    exact_capabilities = {
        result.expected.request.kind
        for result in acceptance
        if result.expected.decision == "direct"
        and result.expected.request is not None
        and result.disposition == "exact"
    }
    capabilities_without_exact = (
        required_capabilities & accepted_capabilities
    ) - exact_capabilities
    if capabilities_without_exact:
        failures.append(
            "acceptance had no exact result for capabilities: "
            + ", ".join(sorted(capability.value for capability in capabilities_without_exact))
        )

    critical = [result for result in acceptance if result.risk_class == "critical"]
    if _count_disposition(critical, "exact") != len(critical):
        failures.append("not every critical acceptance case was exact")
    for risk_domain in sorted(acceptance_domains):
        domain_results = [result for result in acceptance if result.risk_domain == risk_domain]
        domain_budget = _semantic_acceptance_budget(domain_results)
        domain_exact = _count_disposition(domain_results, "exact")
        if domain_exact < domain_budget["required_exact_count"]:
            failures.append(
                f"{risk_domain} acceptance exact count {domain_exact} was below required "
                f"{domain_budget['required_exact_count']} of {domain_budget['cases']}"
            )
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


def _semantic_series_verdict(
    candidate_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    incumbent_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    *,
    projection: ProjectedSemanticRouteCaseResult,
    gate: SemanticRouteGate,
) -> _SemanticSeriesVerdict:
    if not candidate_runs or len(candidate_runs) != len(incumbent_runs):
        raise ValueError("semantic-route qualification requires paired repetitions")
    if not projection.passed:
        raise ValueError("semantic-route projection must pass before provider evaluation")
    if gate == "diagnostic":
        return _SemanticSeriesVerdict(
            gate=gate,
            run_failures=tuple(() for _ in candidate_runs),
            failures=(),
        )
    run_failures: list[tuple[str, ...]] = []
    failures: list[str] = []
    for repetition, (candidate, incumbent) in enumerate(
        zip(candidate_runs, incumbent_runs, strict=True),
        start=1,
    ):
        current = _semantic_gate_failures(
            candidate,
            incumbent,
            projection=projection,
            gate=gate,
        )
        run_failures.append(current)
        for failure in current:
            failures.append(f"repetition {repetition}: {failure}")
    return _SemanticSeriesVerdict(
        gate=gate,
        run_failures=tuple(run_failures),
        failures=tuple(failures),
    )


def _semantic_route_report(
    candidate_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    incumbent_runs: Sequence[Sequence[SemanticRouteCaseResult]],
    *,
    corpus: SemanticRouteEvalCorpus,
    projection: ProjectedSemanticRouteCaseResult,
    verdict: _SemanticSeriesVerdict,
) -> dict[str, object]:
    gate = verdict.gate
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
    if len(verdict.run_failures) != len(candidate_runs):
        raise ValueError("semantic-route verdict must align with repetitions")
    acceptance = [result for result in first_candidate if result.evaluation_split == "acceptance"]
    run_reports: list[dict[str, object]] = []
    for repetition, (candidate, incumbent) in enumerate(
        zip(candidate_runs, incumbent_runs, strict=True),
        start=1,
    ):
        run_failures = verdict.run_failures[repetition - 1]
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
                        repetition=repetition,
                    ),
                    "incumbent": _semantic_model_report(
                        incumbent,
                        repetition=repetition,
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
        "acceptance_budget_per_repetition": _semantic_acceptance_budget(acceptance),
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
            "passed": verdict.passed,
            "failures": list(verdict.failures),
        },
        "models": {
            "candidate": _semantic_model_report(
                pooled_candidate,
                repetition=None,
            ),
            "incumbent": _semantic_model_report(
                pooled_incumbent,
                repetition=None,
            ),
        },
        "runs": run_reports,
    }


def _semantic_invalid_projection_report(
    corpus: SemanticRouteEvalCorpus,
    projection: ProjectedSemanticRouteCaseResult,
    *,
    gate: SemanticRouteGate,
    expected_repetitions: int,
) -> dict[str, object]:
    acceptance = [case for case in corpus.cases if case.evaluation_split == "acceptance"]
    return {
        "schema_version": _SEMANTIC_ROUTE_REPORT_SCHEMA_VERSION,
        "corpus_schema_version": corpus.schema_version,
        "run_at": datetime.now(tz=UTC).isoformat(),
        "corpus_fingerprint": _corpus_fingerprint(corpus),
        "repetitions": 0,
        "acceptance_budget_per_repetition": _semantic_acceptance_budget(acceptance),
        "route_schema_bytes": len(
            json.dumps(RouteProposal.model_json_schema(), separators=(",", ":")).encode("utf-8")
        ),
        "projection": {
            "case_id": projection.case_id,
            "scenario_class": projection.scenario_class,
            "risk_class": projection.risk_class,
            "risk_domain": projection.risk_domain,
            "evaluation_split": projection.evaluation_split,
            "exact": False,
            "actual": (
                "context"
                if isinstance(projection.actual_context, RoutingContext)
                else projection.actual_context.reason
            ),
        },
        "invalid_run": {
            "reason": "projection_mismatch",
            "expected_repetitions": expected_repetitions,
            "observed_repetitions": 0,
            "expected_cases_per_model_per_repetition": len(corpus.cases) + 1,
            "observed_cases_per_model": 0,
        },
        "gate": {
            "mode": gate,
            "passed": None,
            "failures": [],
        },
        "models": {},
        "runs": [],
    }


def _write_json_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def _run_semantic_route_eval(
    report_path: Path,
    gate: SemanticRouteGate = "cutover",
    *,
    fixture_config_root: Path,
    diagnostic_timeout_seconds: float | None = None,
    candidate_selection: ProviderModel | None = None,
) -> int:
    selection = _load_routing_eval_selection()
    config = selection.config
    effective_candidate = candidate_selection or config.llm.routing
    if candidate_selection is None:
        candidate_model = selection.routing_model
        candidate_structured_output_method = selection.routing_structured_output_method
    else:
        candidate_model = selection.gateway.chat_model(candidate_selection)
        candidate_structured_output_method = selection.gateway.structured_output_method(
            candidate_selection
        )
    repetitions = 1 if gate == "diagnostic" else _SEMANTIC_ROUTE_QUALIFICATION_REPETITIONS
    if gate == "diagnostic":
        if (
            diagnostic_timeout_seconds is None
            or not math.isfinite(diagnostic_timeout_seconds)
            or diagnostic_timeout_seconds <= 0
        ):
            raise ValueError("semantic-route diagnostic requires a positive finite timeout")
        timeout_seconds = diagnostic_timeout_seconds
    else:
        if diagnostic_timeout_seconds is not None:
            raise ValueError("diagnostic timeout is valid only in diagnostic mode")
        timeout_seconds = config.runtime.semantic_router_timeout_seconds
    runtime = await _build_eval_runtime(
        config,
        selection.response_model,
        selection.gateway.chat_model(config.llm.reasoning),
        fixture_config_root=fixture_config_root,
        routing_model=candidate_model,
        thread_id=f"semantic-route-eval-{uuid.uuid4().hex}",
        structured_output_method=selection.response_structured_output_method,
        routing_structured_output_method=candidate_structured_output_method,
        operational_telemetry_sink=DisabledTelemetrySink(),
        routing_evidence_sink=DisabledTelemetrySink(),
    )
    try:
        corpus = _load_semantic_route_corpus()
        state = ReasoningState.from_checkpoint(
            (await runtime.graph.aget_state(runtime.engine._config)).values
        )
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
        if not projection.passed:
            report = _semantic_invalid_projection_report(
                corpus,
                projection,
                gate=gate,
                expected_repetitions=repetitions,
            )
            await asyncio.to_thread(_write_json_report, report_path, report)
            print("[semantic_routes] projector: 0/1")
            print(f"    {projection.case_id}: production context projection did not match fixture")
            print(f"    sanitized report: {report_path}")
            print("\nSEMANTIC ROUTING EVALUATION INVALID. [FAIL]")
            return 1

        assert isinstance(projected, RoutingContext)
        candidate_router = SemanticRouter(
            candidate_model,
            selection=effective_candidate,
            structured_output_method=candidate_structured_output_method,
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
        candidate_runs: list[tuple[SemanticRouteCaseResult, ...]] = []
        incumbent_runs: list[tuple[SemanticRouteCaseResult, ...]] = []
        for _ in range(repetitions):
            candidate_results = [
                SemanticRouteCaseResult(
                    case_id=corpus.projected_case.case_id,
                    scenario_class=corpus.projected_case.scenario_class,
                    risk_class=corpus.projected_case.risk_class,
                    risk_domain=corpus.projected_case.risk_domain,
                    evaluation_split=corpus.projected_case.evaluation_split,
                    expected=corpus.projected_case.expected,
                    attempt=await candidate_router.route(projected),
                ),
                *await _run_semantic_route_cases(candidate_router, corpus),
            ]
            incumbent_results = [
                SemanticRouteCaseResult(
                    case_id=corpus.projected_case.case_id,
                    scenario_class=corpus.projected_case.scenario_class,
                    risk_class=corpus.projected_case.risk_class,
                    risk_domain=corpus.projected_case.risk_domain,
                    evaluation_split=corpus.projected_case.evaluation_split,
                    expected=corpus.projected_case.expected,
                    attempt=await incumbent_router.route(projected),
                ),
                *await _run_semantic_route_cases(incumbent_router, corpus),
            ]
            candidate_runs.append(tuple(candidate_results))
            incumbent_runs.append(tuple(incumbent_results))
        verdict = _semantic_series_verdict(
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
            verdict=verdict,
        )
        await asyncio.to_thread(_write_json_report, report_path, report)
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
        if verdict.failures:
            for failure in verdict.failures:
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
        await runtime.caller_context.aclose_session()


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
    engine: ReasoningEngine,
    corpus: ReadOwnerEvalCorpus,
) -> dict[str, tuple[str, ...]]:
    graph = engine._graph
    checkpointer = graph.checkpointer
    if not isinstance(checkpointer, SchemaValidatedCheckpointSaver):
        raise RuntimeError("read-owner evaluation requires the production checkpoint boundary")
    failures: dict[str, tuple[str, ...]] = {}
    for case in corpus.cases:
        request: IntentRequest
        if case.owner == "catalog":
            request = SearchCatalog(query=case.query)
        else:
            request = AnswerQuestion(topic=case.topic)
        turn_id = f"read-owner:{case.case_id}:{uuid.uuid4().hex}"
        binding = engine._checkpoint_binding.rotate(turn_id)
        checkpointer.bind_checkpoint_contract(
            graph.channels,
            binding=binding,
            io_timeout_seconds=engine._checkpoint_io_timeout_seconds,
        )
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
            binding.config,
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
        origin_node=CATALOG_RESPONSE_NODE,
        utterance="Tell me about trail shoes.",
        initial_request=SearchCatalog(query="trail shoes"),
        offline_faults=(FaultKind.PRE_RESPONSE_DISCONNECT, FaultKind.INTERRUPTED_BODY),
        live_faults=(FaultKind.PRE_RESPONSE_DISCONNECT,),
    ),
    TransportOwnerScenario(
        scenario_id="cart_typed_slot",
        owner="cart",
        origin_node="cart_capability_entry",
        utterance="Add a trail shoe to my cart.",
        initial_request=ModifyCart(operation="add"),
        offline_faults=(FaultKind.PRE_RESPONSE_DISCONNECT, FaultKind.INTERRUPTED_BODY),
    ),
    TransportOwnerScenario(
        scenario_id="identity_account_orders",
        owner="identity",
        origin_node="identity_assemble",
        utterance="What orders are on my account?",
        initial_request=ListOrders(scope="account"),
    ),
    TransportOwnerScenario(
        scenario_id="support_cancel",
        owner="support",
        origin_node="support_capability_entry",
        utterance="Cancel order ORD-1002.",
        initial_request=CancelOrders(),
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
    execution_owner=None,
    automation_channels=(),
    handover_destination=None,
    interrupted=False,
    unfinished=False,
    automation_terminal=False,
)


class _OfflineSecretResolver:
    def resolve(self, _ref: str) -> str:
        return "offline-not-a-secret"


class _TransportContinuationRecognizer:
    """Reach a seeded owner without including routing in the injected owner fault."""

    async def route(self, context: RoutingContext) -> RoutingAttempt:
        resolution: RouteResolution = (
            RouteDecision.continue_current()
            if context.active_capability is not None
            else RoutingFailure(reason="context_invalid")
        )
        return RoutingAttempt(
            resolution=resolution,
            provider="transport-fixture",
            model="seeded-continuation",
            structured_output_method="function_calling",
            elapsed_ms=0.0,
            input_tokens=None,
            cache_read_tokens=None,
            output_tokens=None,
            route_schema_fingerprint=ROUTE_SCHEMA_FINGERPRINT,
            prompt_fingerprint="transport-seeded-continuation",
            registry_fingerprint="transport-runtime-registry",
            input_max_chars=2048,
            timeout_seconds=1.0,
            provider_call_outcome="not_attempted",
        )


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


async def _seed_transport_request(runtime: EvalRuntime, scenario: TransportOwnerScenario) -> None:
    if scenario.initial_request is None:
        return
    opening_turn_ids = (f"transport:{scenario.scenario_id}:opening",)
    await runtime.graph.aupdate_state(
        runtime.engine._config,
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
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
    snapshot = await runtime.graph.aget_state(runtime.engine._config)
    state = ReasoningState.from_checkpoint(snapshot.values)
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
        observed_failures = tuple(_telemetry_event_payload(record) for record in turn_failed)
        if observed_failures != (expected_failure,):
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


def _telemetry_event_payload(record: dict[str, object]) -> dict[str, object] | None:
    metadata = {
        "schema_version": record.get("schema_version"),
        "purpose": record.get("purpose"),
        "tenant_id": record.get("tenant_id"),
        "session_id": record.get("session_id"),
    }
    if (
        metadata["schema_version"] != 1
        or metadata["purpose"] != "operational"
        or not isinstance(metadata["tenant_id"], str)
        or not metadata["tenant_id"].strip()
        or not isinstance(metadata["session_id"], str)
        or not metadata["session_id"].strip()
    ):
        return None
    return {
        key: value
        for key, value in record.items()
        if key not in {"schema_version", "purpose", "tenant_id", "session_id"}
    }


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
    fixture_config_root: Path,
    credentials: ProviderCredentialsConfig,
    secrets: SecretResolver,
    request_ceiling: int,
    fault_window_seconds: float,
    upstream_timeout_seconds: float,
    scenario_timeout_seconds: float,
) -> TransportCaseResult:
    mode = FaultMode.UNTIL_EXHAUSTED if tier == "offline" else FaultMode.RETRY_MASKED_ONCE
    runtime: EvalRuntime | None = None
    with tempfile.TemporaryDirectory(prefix="transport-cert-") as temp_dir:
        telemetry_path = Path(temp_dir) / "telemetry.jsonl"
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
                runtime = await _build_eval_runtime(
                    config,
                    model,
                    model,
                    fixture_config_root=fixture_config_root,
                    routing_model=model,
                    routing_recognizer=_TransportContinuationRecognizer(),
                    thread_id=f"transport-{uuid.uuid4().hex}",
                    structured_output_method=gateway.structured_output_method(target),
                    routing_structured_output_method=gateway.structured_output_method(target),
                    operational_telemetry_sink=JsonlTelemetrySink(telemetry_path),
                    routing_evidence_sink=DisabledTelemetrySink(),
                )
                await _seed_transport_request(runtime, scenario)
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
            if runtime is not None:
                await runtime.caller_context.aclose_session()


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
            fixture_config_root=_CONFIG_ROOT,
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


def _run_preflight() -> int:
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
        print("\\nFRONTLINE EVAL BLOCKED: structural safety preflight failed. [FAIL]")
        return 1
    print("[structural_preflight] all registered contracts passed. [PASS]")
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
        "--semantic-routing-structural-coverage",
        action="store_true",
        help="validate development-only route/checklist coverage without loading a model",
    )
    mode.add_argument(
        "--semantic-routing-data-audit",
        action="store_true",
        help="validate source-clustered routing-data partitions without loading a model",
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
        "--semantic-routing-structural-report",
        type=Path,
        help="development-only structural coverage report destination",
    )
    parser.add_argument(
        "--semantic-routing-gate",
        choices=("diagnostic", "shadow", "cutover"),
        help="semantic-router qualification gate (defaults to cutover)",
    )
    parser.add_argument(
        "--semantic-routing-diagnostic-timeout-seconds",
        type=float,
        help="evaluation-only router timeout; valid only with the diagnostic gate",
    )
    parser.add_argument(
        "--semantic-routing-candidate",
        type=_parse_provider_model,
        help=(
            "evaluation-only provider:model candidate override; requires an explicit report "
            "path and diagnostic or shadow gate"
        ),
    )
    parser.add_argument(
        "--semantic-routing-data-partition",
        action="append",
        type=Path,
        default=[],
        help="routing-data partition manifest; repeat once per available partition",
    )
    parser.add_argument(
        "--semantic-routing-acquisition-plan",
        type=Path,
        help="external steward-approved routing-data acquisition plan",
    )
    parser.add_argument(
        "--semantic-routing-data-authorization",
        type=Path,
        help="steward authorization bound to the complete manifest-set fingerprint",
    )
    parser.add_argument(
        "--semantic-routing-data-report",
        type=Path,
        help="sanitized routing-data readiness report destination",
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
    if (
        args.semantic_routing_structural_report is not None
        and not args.semantic_routing_structural_coverage
    ):
        parser.error(
            "--semantic-routing-structural-report requires --semantic-routing-structural-coverage"
        )
    if args.semantic_routing_gate is not None and not args.semantic_routing_eval:
        parser.error("--semantic-routing-gate requires --semantic-routing-eval")
    if (
        args.semantic_routing_diagnostic_timeout_seconds is not None
        and not args.semantic_routing_eval
    ):
        parser.error(
            "--semantic-routing-diagnostic-timeout-seconds requires --semantic-routing-eval"
        )
    if args.semantic_routing_candidate is not None and not args.semantic_routing_eval:
        parser.error("--semantic-routing-candidate requires --semantic-routing-eval")
    if args.semantic_routing_candidate is not None and args.semantic_routing_report is None:
        parser.error("--semantic-routing-candidate requires --semantic-routing-report")
    if args.semantic_routing_candidate is not None:
        try:
            args.semantic_routing_candidate = _resolve_conformance_target(
                args.semantic_routing_candidate
            )
        except ValueError as exc:
            parser.error(str(exc))
    custom_routing_data_options = (
        bool(args.semantic_routing_data_partition)
        or args.semantic_routing_acquisition_plan is not None
        or args.semantic_routing_data_authorization is not None
        or args.semantic_routing_data_report is not None
    )
    if custom_routing_data_options and not args.semantic_routing_data_audit:
        parser.error("semantic-routing data options require --semantic-routing-data-audit")
    if args.order_target_eval:
        return asyncio.run(_run_order_target_eval())
    if args.semantic_routing_structural_coverage:
        return _run_semantic_route_structural_coverage(
            args.semantic_routing_structural_report or _SEMANTIC_ROUTE_STRUCTURAL_REPORT_PATH
        )
    if args.semantic_routing_data_audit:
        return _run_routing_data_audit(
            args.semantic_routing_data_partition,
            acquisition_plan_path=args.semantic_routing_acquisition_plan,
            authorization_path=args.semantic_routing_data_authorization,
            report_path=args.semantic_routing_data_report or _ROUTING_DATA_REPORT_PATH,
        )
    if args.semantic_routing_eval:
        semantic_gate = args.semantic_routing_gate or "cutover"
        if args.semantic_routing_candidate is not None and semantic_gate == "cutover":
            parser.error(
                "--semantic-routing-candidate is evaluation-only; use diagnostic or shadow"
            )
        if semantic_gate == "diagnostic":
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
            if args.semantic_routing_diagnostic_timeout_seconds is not None:
                parser.error(
                    "--semantic-routing-diagnostic-timeout-seconds requires the diagnostic gate"
                )
            diagnostic_timeout_seconds = None
        return asyncio.run(
            _run_semantic_route_eval(
                args.semantic_routing_report or _SEMANTIC_ROUTE_REPORT_PATH,
                semantic_gate,
                fixture_config_root=_CONFIG_ROOT,
                diagnostic_timeout_seconds=diagnostic_timeout_seconds,
                candidate_selection=args.semantic_routing_candidate,
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
    if args.preflight_only:
        return _run_preflight()
    parser.error("no evaluation mode selected; choose an explicit current evaluator")


if __name__ == "__main__":
    sys.exit(main())
