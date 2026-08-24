"""Provider conformance suite + the fail-closed commerce gate (AGENTS §A11).

A model may serve commerce turns ONLY after passing the tool-calling + structured-output +
streaming checks. Verdicts persist in config/conformance/reports.json and are read
FAIL-CLOSED three ways: no report -> not ready; report older than `max_report_age_days` ->
not ready (target IDs are aliases — behavior can shift under the same ID); `suite_version`
mismatch -> not ready (a deepened suite invalidates every old certificate, forcing
re-certification — e.g. when Phase 3's real tool schemas are added).

Three-outcome result model (transport != verdict): a shape-valid-but-wrong response FAILS a
check (-> chat-only); a transport/auth error raises `ConformanceRunError` and NO verdict is
recorded — infrastructure failure is never converted into a capability verdict.

Checks use async APIs (`ainvoke`/`astream`) to certify async tool, structured-output, and
incremental-delivery capabilities. Production graph model nodes currently use synchronous
`.invoke()`; exact production transport-path failure behavior is a separate certification concern.
The structured-output check uses the real provider wire contracts and the gateway-selected
native transport. Routing conformance covers the flat coarse `RouteProposal`; deterministic
materialization into the nested internal `RouteDecision` is a separate local contract.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessageChunk
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.config import LLMConfig, MerchantConfig, ProviderModel
from agnostic_market.dtos.llm import (
    DETAIL_MAX_LENGTH,
    ConformanceCheck,
    ConformanceReport,
    ConformanceTargetsConfig,
    StructuredOutputMethod,
)
from agnostic_market.dtos.orchestration import (
    AnswerResponse,
    OrderTargetProposal,
    RouteProposal,
)

# After the first report for this suite version is accepted, bump when checks or
# transmitted schemas change so accepted reports fail closed. Before that, the version
# is unreleased and its checks may still be completed without a meaningless bump.
SUITE_VERSION = "5"

# Prompts name the required tool explicitly: the check verifies protocol conformance
# (parse/emit/select), not semantic routing quality (that is Phase 6's behavioral eval).
_TOOL_CALL_PROMPT = (
    "Use the get_order_status tool to look up order 'ORD-1001'. Do not answer directly."
)
_EXPECTED_TOOL = "get_order_status"
_STREAMING_PROMPT = "In two short sentences, describe a running shoe."


class ConformanceRunError(RuntimeError):
    """Transport/auth failure during a run — NO verdict may be recorded."""


class ChatOnlyModelError(RuntimeError):
    """A commerce turn was about to be routed to a non-certified model (gate refusal)."""


@dataclass(frozen=True, slots=True)
class _StructuredOutputCase:
    name: str
    schema: type[BaseModel]
    prompt: str
    accepts: Callable[[BaseModel], bool]


def _accepts_route_proposal(expected: RouteProposal) -> Callable[[BaseModel], bool]:
    return lambda result: isinstance(result, RouteProposal) and result == expected


def _accepts_answer(decision: str) -> Callable[[BaseModel], bool]:
    return lambda result: (
        isinstance(result, AnswerResponse)
        and result.decision == decision
        and ((decision == "answer") == (result.answer is not None))
    )


def _accepts_order_target(expected: OrderTargetProposal) -> Callable[[BaseModel], bool]:
    return lambda result: (
        isinstance(result, OrderTargetProposal)
        and result.relationship == expected.relationship
        and Counter(result.order_refs) == Counter(expected.order_refs)
    )


def _order_target_case(expected: OrderTargetProposal) -> _StructuredOutputCase:
    return _StructuredOutputCase(
        f"order_target_{expected.relationship}",
        OrderTargetProposal,
        "Return this structured target proposal; order_refs array order is insignificant: "
        f"{expected.model_dump_json()}.",
        _accepts_order_target(expected),
    )


def _route_case(name: str, prompt: str, expected: RouteProposal) -> _StructuredOutputCase:
    return _StructuredOutputCase(name, RouteProposal, prompt, _accepts_route_proposal(expected))


_STRUCTURED_OUTPUT_CASES = (
    _route_case(
        "route_direct",
        "Return decision=direct and capability=search_catalog. Leave clarification_reason and "
        "every coarse field null.",
        RouteProposal(decision="direct", capability="search_catalog"),
    ),
    _route_case(
        "route_clarify",
        "Return decision=clarify with clarification_reason=ambiguous_intent. Leave capability "
        "and every coarse field null.",
        RouteProposal(decision="clarify", clarification_reason="ambiguous_intent"),
    ),
    _route_case(
        "route_continue",
        "Return decision=continue. Leave capability, clarification_reason, and every coarse field "
        "null.",
        RouteProposal(decision="continue"),
    ),
    _route_case(
        "route_answer_topic",
        "Return decision=direct, capability=answer_question, and answer_topic=policy. Leave every "
        "other field null.",
        RouteProposal(decision="direct", capability="answer_question", answer_topic="policy"),
    ),
    _route_case(
        "route_list_scope",
        "Return decision=direct, capability=list_orders, and list_scope=account. Leave every other "
        "field null.",
        RouteProposal(decision="direct", capability="list_orders", list_scope="account"),
    ),
    _route_case(
        "route_cart_operation",
        "Return decision=direct, capability=modify_cart, and cart_operation=add. Leave every other "
        "field null.",
        RouteProposal(decision="direct", capability="modify_cart", cart_operation="add"),
    ),
    _route_case(
        "route_profile_field",
        "Return decision=direct, capability=change_profile, and profile_field=address. Leave every "
        "other field null.",
        RouteProposal(decision="direct", capability="change_profile", profile_field="address"),
    ),
    _route_case(
        "route_order_status_selector",
        "Return decision=direct, capability=verify_order_status, and "
        "order_status_selector=explicit. Leave every other field null.",
        RouteProposal(
            decision="direct",
            capability="verify_order_status",
            order_status_selector="explicit",
        ),
    ),
    _StructuredOutputCase(
        "answer",
        AnswerResponse,
        "Return decision=answer with a short non-empty answer.",
        _accepts_answer("answer"),
    ),
    _StructuredOutputCase(
        "clarify",
        AnswerResponse,
        "Return decision=clarify with no answer.",
        _accepts_answer("clarify"),
    ),
    _StructuredOutputCase(
        "unsupported",
        AnswerResponse,
        "Return decision=unsupported with no answer.",
        _accepts_answer("unsupported"),
    ),
    _order_target_case(
        OrderTargetProposal(relationship="single", order_refs=("ORD-1002",)),
    ),
    _order_target_case(
        OrderTargetProposal(
            relationship="plural",
            order_refs=("ORD-1001", "ORD-1002"),
        ),
    ),
    _order_target_case(
        OrderTargetProposal(
            relationship="alternative",
            order_refs=("ORD-1001", "ORD-1002"),
        ),
    ),
    _order_target_case(
        OrderTargetProposal(relationship="ambiguous", order_refs=()),
    ),
)


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of an order by its order id."""
    raise NotImplementedError("conformance probe - never executed")


@tool
def search_catalog(query: str) -> str:
    """Search the product catalog for items matching a text query."""
    raise NotImplementedError("conformance probe - never executed")


def _clip(text: str) -> str:
    return text if len(text) <= DETAIL_MAX_LENGTH else text[: DETAIL_MAX_LENGTH - 3] + "..."


async def _check_tool_call(chat_model: BaseChatModel) -> ConformanceCheck:
    """Two tools bound; the model must emit a call to the named one with valid args."""
    name = "tool_call"
    bound = chat_model.bind_tools([get_order_status, search_catalog])
    response = await bound.ainvoke(_TOOL_CALL_PROMPT)
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        return ConformanceCheck(name=name, passed=False, detail="no tool_calls in response")
    call = calls[0]
    if call["name"] != _EXPECTED_TOOL:
        return ConformanceCheck(
            name=name, passed=False, detail=_clip(f"wrong tool selected: {call['name'][:60]!r}")
        )
    order_id = (call.get("args") or {}).get("order_id")
    if not isinstance(order_id, str) or not order_id:
        return ConformanceCheck(
            name=name, passed=False, detail="tool args missing a valid 'order_id' string"
        )
    return ConformanceCheck(name=name, passed=True, detail="")


async def _check_structured_output(
    chat_model: BaseChatModel, *, method: StructuredOutputMethod
) -> ConformanceCheck:
    """The configured transport must validate every structured schema used in production."""
    name = "structured_output"
    for case in _STRUCTURED_OUTPUT_CASES:
        structured = chat_model.with_structured_output(case.schema, method=method)
        try:
            result = await structured.ainvoke(case.prompt)
        except (OutputParserException, ValidationError) as exc:
            return ConformanceCheck(
                name=name,
                passed=False,
                detail=_clip(f"{case.name}: output failed schema ({type(exc).__name__})"),
            )
        if not isinstance(result, BaseModel) or not case.accepts(result):
            return ConformanceCheck(
                name=name,
                passed=False,
                detail=_clip(f"{case.name}: wrong validated result ({type(result).__name__})"),
            )
    return ConformanceCheck(name=name, passed=True, detail="")


def _chunk_text(chunk: BaseMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


async def _check_streaming(chat_model: BaseChatModel) -> ConformanceCheck:
    """`astream` must yield >= 2 TEXT-BEARING chunks (real incremental delivery).

    Counting raw chunks would be wrong: langchain-core appends a trailing empty chunk
    after the stream closes, so a non-streaming model (all text in one chunk) still
    emits 2 wire chunks. Only chunks carrying text discriminate streaming.
    """
    name = "streaming"
    text_chunks = 0
    parts: list[str] = []
    async for chunk in chat_model.astream(_STREAMING_PROMPT):
        piece = _chunk_text(chunk)
        if piece:
            text_chunks += 1
            parts.append(piece)
    text = "".join(parts).strip()
    if text_chunks < 2 or not text:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=(
                f"{text_chunks} text-bearing chunk(s), {len(text)} chars (need >=2 text chunks)"
            ),
        )
    return ConformanceCheck(name=name, passed=True, detail="")


async def run_conformance(
    chat_model: BaseChatModel,
    *,
    selection: ProviderModel,
    structured_output_method: StructuredOutputMethod,
) -> ConformanceReport:
    """Run all three checks; verdict is commerce-ready iff ALL pass.

    Raises ConformanceRunError on any transport/auth exception (after the client's own
    retries) — callers must NOT record a verdict in that case. Shape failures never raise;
    they come back as failed checks.
    """
    checks: list[ConformanceCheck] = []
    try:
        checks.append(await _check_tool_call(chat_model))
        checks.append(await _check_structured_output(chat_model, method=structured_output_method))
        checks.append(await _check_streaming(chat_model))
    except Exception as exc:
        # Deliberately broad: provider clients raise provider-specific transport/auth
        # errors we cannot enumerate in an agnostic layer. Shape failures are already
        # handled inside the checks, so anything reaching here is infrastructure — it is
        # re-raised with cause, never swallowed and never turned into a verdict.
        raise ConformanceRunError(
            f"conformance run for {selection.provider}:{selection.model} hit a "
            f"transport/auth error "
            f"({type(exc).__name__}) - no verdict recorded; fix and re-run"
        ) from exc
    verdict = "commerce-ready" if all(check.passed for check in checks) else "chat-only"
    return ConformanceReport(
        provider=selection.provider,
        model=selection.model,
        reasoning_effort=selection.reasoning_effort,
        suite_version=SUITE_VERSION,
        run_at=datetime.now(tz=UTC),
        checks=checks,
        verdict=verdict,
    )


def load_conformance_targets(path: Path) -> ConformanceTargetsConfig:
    """Load + validate config/conformance/targets.yaml. Fails loudly."""
    return ConformanceTargetsConfig.model_validate(load_yaml_layer(path))


class ConformanceRegistry:
    """Persisted verdicts (config/conformance/reports.json) + the fail-closed gate."""

    def __init__(self, report_path: Path, *, max_report_age_days: int) -> None:
        self._path = report_path
        self._max_age = timedelta(days=max_report_age_days)
        self._reports: dict[str, ConformanceReport] = {}
        if report_path.is_file():
            try:
                raw = json.loads(report_path.read_text(encoding="utf-8"))
                self._reports = {
                    key: ConformanceReport.model_validate(value) for key, value in raw.items()
                }
            except (json.JSONDecodeError, ValidationError, AttributeError) as exc:
                raise ConfigError(
                    f"conformance reports file {report_path} is corrupt or has an invalid "
                    f"shape - delete it and re-run scripts/conformance_live.py: {exc}"
                ) from exc

    @staticmethod
    def _key(provider: str, model: str) -> str:
        return f"{provider}:{model}"

    def record(self, report: ConformanceReport) -> None:
        self._reports[self._key(report.provider, report.model)] = report

    def save(self) -> None:
        payload = {
            key: report.model_dump(mode="json") for key, report in sorted(self._reports.items())
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def is_commerce_ready(
        self,
        selection: ProviderModel,
        *,
        now: datetime | None = None,
    ) -> bool:
        """FAIL-CLOSED: True only for a current-suite, unexpired, commerce-ready report."""
        report = self._reports.get(self._key(selection.provider, selection.model))
        if report is None or report.suite_version != SUITE_VERSION:
            return False
        if report.reasoning_effort != selection.reasoning_effort:
            return False
        if report.verdict != "commerce-ready":
            return False
        return (now or datetime.now(tz=UTC)) - report.run_at <= self._max_age

    def require_commerce_ready(self, selection: ProviderModel) -> None:
        """The gate the Phase-3 router calls before routing a commerce turn."""
        if not self.is_commerce_ready(selection):
            raise ChatOnlyModelError(
                f"model '{selection.provider}:{selection.model}' is not certified "
                f"commerce-ready (fail-closed: report missing, expired, from an older "
                f"suite, for a different runtime recipe, or chat-only) - it may not "
                f"serve a commerce turn (AGENTS A11)"
            )


def check_llm_certification(config: MerchantConfig, registry: ConformanceRegistry) -> list[str]:
    """Config-time certification check — WARN-ONLY in Phase 1.

    Flags a merchant whose response, routing, or reasoning role selects a non-certified model, so
    it is caught at config time rather than mid-call at checkout. Phase 4's go-live gate
    makes this blocking. Lives here so the config plane stays free of llm-plane imports.
    """
    warnings: list[str] = []
    # Iterate the DTO's own fields so a future LLMConfig node is checked automatically.
    for node in LLMConfig.model_fields:
        selection: ProviderModel = getattr(config.llm, node)
        if not registry.is_commerce_ready(selection):
            warnings.append(
                f"merchant {config.merchant_id!r}: llm.{node} = "
                f"'{selection.provider}:{selection.model}' is not certified commerce-ready"
            )
    return warnings


def require_llm_certification(config: MerchantConfig, registry: ConformanceRegistry) -> None:
    """Reject startup when any configured LLM role lacks current conformance evidence."""

    failures = check_llm_certification(config, registry)
    if failures:
        raise ChatOnlyModelError("; ".join(failures))
