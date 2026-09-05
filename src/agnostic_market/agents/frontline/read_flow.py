"""Typed read/answer owners that do not belong to a transactional domain flow."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agnostic_market.agents._consent import classify_confirmation
from agnostic_market.agents._copy import ACCOUNT_CONTACT_QUESTION, ORDER_NUMBER_QUESTION, warm_close
from agnostic_market.agents.frontline.typed_prompt import (
    ORDER_TARGET_PROPOSAL_PROMPT,
    compose_answer_response_prompt,
    compose_catalog_response_prompt,
)
from agnostic_market.agents.model_speech import CallerAudibleModelTextPolicy
from agnostic_market.agents.telemetry import TelemetryRecorder, record_capability_answered
from agnostic_market.commerce.catalog import CatalogPort
from agnostic_market.commerce.identity import (
    CallerIdentityStore,
    CustomerDirectoryPort,
    classify_contact_claims,
    order_read_allowed,
    try_grant_orders_by_contact,
)
from agnostic_market.commerce.orders import (
    BOUND_ORDER_READ_UNAVAILABLE_LINE,
    ORDER_CONTACT_NOT_FOUND_LINE,
    GuestOrderScope,
    OrderPort,
    RecentOrderContext,
    render_order_status_line,
)
from agnostic_market.commerce.spoken import caller_stated_order_ids
from agnostic_market.dtos.llm import StructuredOutputMethod
from agnostic_market.dtos.orchestration import (
    AnswerQuestion,
    AnswerResponse,
    CapabilityId,
    ExplicitOrderSet,
    FocusedOrderSet,
    OrderTargetProposal,
    RecentOrderSet,
    SearchCatalog,
    VerifyOrderStatus,
)
from agnostic_market.dtos.state import HandoffRequest, PolicyContext, ReasoningState
from agnostic_market.durability.session_state import SessionStateCoordinator

CATALOG_ENTRY_NODE = "catalog_entry"
CATALOG_QUERY_REJECT_NODE = "catalog_query_reject"
CATALOG_RESPONSE_NODE = "catalog_response"
ANSWER_RESPONSE_NODE = "answer_response"
ANSWER_CLARIFY_NODE = "answer_clarify"
ANSWER_UNSUPPORTED_NODE = "answer_unsupported"
ORDER_STATUS_ENTRY_NODE = "order_status_entry"
ORDER_STATUS_TARGET_ASK_NODE = "order_status_target_ask"
ORDER_STATUS_TARGET_PROPOSE_NODE = "order_status_target_propose"
ORDER_STATUS_TARGET_CONFIRM_NODE = "order_status_target_confirm"
ORDER_STATUS_TARGET_REJECT_NODE = "order_status_target_reject"
ORDER_STATUS_FULFILL_NODE = "order_status_fulfill"
READ_FLOW_SPEAKABLE_NODES = frozenset(
    {
        CATALOG_QUERY_REJECT_NODE,
        ANSWER_CLARIFY_NODE,
        ANSWER_UNSUPPORTED_NODE,
        ORDER_STATUS_TARGET_ASK_NODE,
        ORDER_STATUS_TARGET_REJECT_NODE,
        ORDER_STATUS_FULFILL_NODE,
    }
)
READ_FLOW_MODEL_SPEECH_NODES = frozenset({CATALOG_RESPONSE_NODE, ANSWER_RESPONSE_NODE})

_CATALOG_QUERY_REJECTED = "I didn't get a product to look for. What else can I help with?"
_ANSWER_CONTEXT_QUESTION = "What would you like me to explain?"
_ANSWER_UNSUPPORTED_RETRY = (
    "Please restate that as a specific store request, including the order, product, account, "
    "or cart detail I should use."
)


@dataclass(frozen=True)
class ReadFlowNodes:
    catalog_entry: Callable[[ReasoningState], Command]
    catalog_query_reject: Callable[[ReasoningState], dict[str, object]]
    catalog_response: Callable[[ReasoningState], Awaitable[Command]]
    answer_response: Callable[[ReasoningState], Awaitable[Command]]
    answer_clarify: Callable[[ReasoningState], dict[str, object]]
    answer_unsupported: Callable[[ReasoningState], dict[str, object]]
    order_status_entry: Callable[[ReasoningState], Command]
    order_status_target_ask: Callable[[ReasoningState], dict[str, object]]
    order_status_target_propose: Callable[[ReasoningState], Awaitable[Command]]
    order_status_target_confirm: Callable[[ReasoningState], Command]
    order_status_target_reject: Callable[[ReasoningState], dict[str, object]]
    order_status_fulfill: Callable[[ReasoningState], Awaitable[Command]]
    speakable_nodes: frozenset[str]
    model_speech_nodes: frozenset[str]


def build_read_flow_nodes(
    response_model: BaseChatModel,
    order_store: OrderPort,
    catalog: CatalogPort,
    guest_orders: GuestOrderScope,
    policy: PolicyContext,
    *,
    display_name: str,
    structured_output_method: StructuredOutputMethod,
    model_text_policy: CallerAudibleModelTextPolicy,
    recent_orders: RecentOrderContext,
    session_state: SessionStateCoordinator,
    identity_store: CallerIdentityStore,
    customers: CustomerDirectoryPort,
    telemetry: TelemetryRecorder,
    routing_telemetry: TelemetryRecorder,
) -> ReadFlowNodes:
    """Build typed read owners over their shared session-bound dependencies."""

    answer_model = response_model.with_structured_output(
        AnswerResponse,
        method=structured_output_method,
    )
    order_target_model = response_model.with_structured_output(
        OrderTargetProposal,
        method=structured_output_method,
    )

    def resolve_order_target(request: VerifyOrderStatus) -> tuple[str, ...] | None:
        target = request.target
        if isinstance(target, ExplicitOrderSet):
            refs = tuple(dict.fromkeys(ref.strip().upper() for ref in target.order_refs))
        elif isinstance(target, FocusedOrderSet):
            focused = recent_orders.snapshot().focused_order_ref
            refs = (focused,) if focused is not None else ()
        elif isinstance(target, RecentOrderSet):
            recent = recent_orders.snapshot()
            refs = recent.order_refs if recent.complete else ()
        else:
            refs = ()
        if not refs or len(refs) > policy.cancel_batch_max:
            return None
        if any(re.fullmatch(r"ORD-\d+", ref) is None for ref in refs):
            return None
        return refs

    def order_status_entry_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyOrderStatus):
            raise TypeError("order-status entry requires a verify-order-status invocation")
        if invocation.request.target is not None:
            return Command(goto=ORDER_STATUS_FULFILL_NODE)
        return Command(goto=ORDER_STATUS_TARGET_PROPOSE_NODE)

    def order_status_target_ask_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyOrderStatus):
            raise TypeError("order-status target question requires an active invocation")
        line = (
            ORDER_NUMBER_QUESTION if invocation.request.target is None else ACCOUNT_CONTACT_QUESTION
        )
        return {"messages": [AIMessage(line)]}

    async def order_status_target_propose_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyOrderStatus):
            raise TypeError("order-status proposer requires a verify-order-status invocation")
        if invocation.request.target is not None:
            raise ValueError("order-status proposer requires a missing target")
        current = state.current_committed_user_message()
        if current is None or not isinstance(current.content, str):
            raise ValueError("order-status proposer requires committed caller text")
        proposal = await order_target_model.ainvoke(
            [SystemMessage(ORDER_TARGET_PROPOSAL_PROMPT), current]
        )
        if not isinstance(proposal, OrderTargetProposal):
            raise TypeError("order-status proposer returned an incompatible result")
        refs = tuple(dict.fromkeys(ref.strip().upper() for ref in proposal.order_refs))
        accepted = (
            proposal.relationship in {"single", "plural"}
            and len(refs) == len(proposal.order_refs)
            and 0 < len(refs) <= policy.cancel_batch_max
            and all(re.fullmatch(r"ORD-\d+", ref) is not None for ref in refs)
        )
        if not accepted:
            return Command(goto=ORDER_STATUS_TARGET_ASK_NODE)
        request = VerifyOrderStatus(target=ExplicitOrderSet(order_refs=refs))
        return Command(
            goto=ORDER_STATUS_FULFILL_NODE,
            update={"active_invocation": invocation.with_request(request)},
        )

    def order_status_target_confirm_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyOrderStatus):
            raise TypeError("order-status target confirmation requires an active invocation")
        request = invocation.request
        if (
            not isinstance(request.target, ExplicitOrderSet)
            or request.explicit_target_turn_id is None
            or request.explicit_target_confirmed
        ):
            raise ValueError("order-status target confirmation requires an unconfirmed target")
        order_ids = resolve_order_target(request)
        if order_ids is None:
            return Command(
                goto=ORDER_STATUS_TARGET_REJECT_NODE,
                update={"active_invocation": None},
            )
        listed = (
            order_ids[0]
            if len(order_ids) == 1
            else f"{', '.join(order_ids[:-1])} and {order_ids[-1]}"
        )
        subject = "that order" if len(order_ids) == 1 else "those orders"
        answer = interrupt(f"I heard {listed}. Did you mean {subject}?")
        decision = classify_confirmation(answer)
        if answer.get("readback_interrupted") or decision.verdict == "unclear":
            answer = interrupt(f"To confirm, should I check {listed}? Please say yes or no.")
            decision = classify_confirmation(answer)
        if decision.verdict == "yes":
            confirmed = request.with_confirmed_explicit_target()
            return Command(
                goto=ORDER_STATUS_FULFILL_NODE,
                update={"active_invocation": invocation.with_request(confirmed)},
            )
        if decision.verdict == "human":
            assert decision.handoff_source is not None
            return Command(
                goto="handover",
                update={
                    "active_invocation": None,
                    "handover": HandoffRequest(
                        destination="human",
                        reason_code="other",
                        source=decision.handoff_source,
                    ),
                },
            )
        return Command(
            goto=ORDER_STATUS_TARGET_REJECT_NODE,
            update={"active_invocation": None},
        )

    def order_status_target_reject_node(state: ReasoningState) -> dict[str, object]:
        if state.active_invocation is not None:
            raise TypeError("order-status rejection requires the invocation to be cleared")
        return {"messages": [AIMessage(ORDER_NUMBER_QUESTION)]}

    def _order_read_denied(order_ids: tuple[str, ...]) -> None:
        for order_id in order_ids:
            telemetry.record(
                {
                    "event": "order_read_denied",
                    "order_id_known": order_store.order_owner(order_id) is not None,
                }
            )

    async def order_status_fulfill_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, VerifyOrderStatus):
            raise TypeError("order-status fulfilment requires a verify-order-status invocation")
        order_ids = resolve_order_target(invocation.request)
        if order_ids is None:
            return Command(
                goto=ORDER_STATUS_TARGET_REJECT_NODE,
                update={"active_invocation": None},
            )
        current = state.current_committed_user_message()
        request = invocation.request
        authorization_message = current
        if isinstance(request.target, ExplicitOrderSet):
            if request.explicit_target_turn_id is None:
                if current is None or not isinstance(current.id, str):
                    raise ValueError(
                        "explicit order-status target requires an admitted source turn"
                    )
                request = request.with_explicit_target_turn(current.id)
                invocation = invocation.with_request(request)
            target_message = state.committed_user_message(request.explicit_target_turn_id)
            if target_message is None or not isinstance(target_message.content, str):
                raise ValueError("order-status target source is not an admitted caller message")
            if authorization_message is None:
                authorization_message = target_message
            if not request.explicit_target_confirmed:
                stated_order_ids = caller_stated_order_ids(target_message.content)
                if stated_order_ids and set(stated_order_ids) == set(order_ids):
                    confirmed = request.with_confirmed_explicit_target()
                    return Command(
                        goto=ORDER_STATUS_FULFILL_NODE,
                        update={"active_invocation": invocation.with_request(confirmed)},
                    )
                return Command(
                    goto=ORDER_STATUS_TARGET_CONFIRM_NODE,
                    update={"active_invocation": invocation},
                )
        if authorization_message is None or not isinstance(authorization_message.content, str):
            raise ValueError("order-status fulfilment requires committed caller text")

        unresolved = tuple(
            order_id
            for order_id in order_ids
            if not order_read_allowed(
                order_id,
                store=order_store,
                guest_orders=guest_orders,
                identity=identity_store,
            )
        )
        if unresolved:
            if identity_store.current() is not None:
                _order_read_denied(unresolved)
                return Command(
                    goto=END,
                    update={
                        "active_invocation": None,
                        "messages": [AIMessage(BOUND_ORDER_READ_UNAVAILABLE_LINE)],
                    },
                )
            selection = classify_contact_claims(authorization_message.content)
            if selection.disposition != "single":
                return Command(goto=ORDER_STATUS_TARGET_ASK_NODE)
            if selection.claim is None:
                raise RuntimeError("single contact selection omitted its claim")
            if (
                try_grant_orders_by_contact(
                    selection.claim,
                    *unresolved,
                    store=order_store,
                    customers=customers,
                    identity=identity_store,
                )
                == "mismatch"
            ):
                _order_read_denied(unresolved)
                return Command(
                    goto=END,
                    update={
                        "active_invocation": None,
                        "messages": [AIMessage(ORDER_CONTACT_NOT_FOUND_LINE)],
                    },
                )
            for order_id in unresolved:
                telemetry.record(
                    {
                        "event": "order_read_granted",
                        "order_id": order_id,
                        "method": "contact_match",
                    }
                )

        if any(
            not order_read_allowed(
                order_id,
                store=order_store,
                guest_orders=guest_orders,
                identity=identity_store,
            )
            for order_id in order_ids
        ):
            raise RuntimeError("order-status authorization did not survive grant resolution")
        line = " ".join(
            render_order_status_line(
                order_id=order_id,
                status=order_store.order_status(order_id) or "unknown",
                items=order_store.order_item_summary(order_id),
                eta=order_store.order_eta(order_id),
                today=datetime.now().date(),
            )
            for order_id in order_ids
        )
        line = f"{line} {warm_close()}"
        committed = await session_state.record_recent_orders(
            f"recent-orders:read:{invocation.invocation_id}",
            order_ids,
            operation="read",
        )
        record_capability_answered(
            routing_telemetry,
            authorization_message.content,
            CapabilityId.VERIFY_ORDER_STATUS.value,
            answer_source="code_authored_read",
        )
        return Command(
            goto=END,
            update={
                "active_invocation": None,
                "session_revision": committed.session_revision,
                "messages": [AIMessage(line)],
            },
        )

    def catalog_entry_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, SearchCatalog):
            raise TypeError("catalog entry requires a search-catalog invocation")
        request = invocation.request
        if request.query is not None:
            return Command(goto=CATALOG_RESPONSE_NODE)

        current = state.current_committed_user_message()
        if current is None:
            raise ValueError("catalog entry requires the current committed caller message")
        if not isinstance(current.content, str):
            return Command(
                goto=CATALOG_QUERY_REJECT_NODE,
                update={"active_invocation": None},
            )
        query = current.content.strip()
        if not query:
            return Command(
                goto=CATALOG_QUERY_REJECT_NODE,
                update={"active_invocation": None},
            )
        return Command(
            goto=CATALOG_RESPONSE_NODE,
            update={"active_invocation": invocation.with_request(SearchCatalog(query=query))},
        )

    def catalog_query_reject_node(state: ReasoningState) -> dict[str, object]:
        if state.active_invocation is not None:
            raise TypeError("catalog query rejection requires the invocation to be cleared")
        return {"messages": [AIMessage(_CATALOG_QUERY_REJECTED)]}

    async def catalog_response_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, SearchCatalog):
            raise TypeError("catalog response requires a search-catalog invocation")
        request = invocation.request
        if request.query is None:
            raise ValueError("catalog response requires a complete query")
        current = state.current_committed_user_message()
        if current is None:
            raise ValueError("catalog response requires the current committed caller message")
        if not isinstance(current.content, str):
            raise TypeError("catalog response requires plain committed caller text")

        result = catalog.search(request.query)
        response = await response_model.ainvoke(
            [
                SystemMessage(compose_catalog_response_prompt(display_name, policy, result)),
                current,
            ]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("catalog response model returned an incompatible message")
        if response.tool_calls:
            raise ValueError("catalog response model returned an unexpected tool call")
        model_text_policy.validate(response.text)

        record_capability_answered(
            routing_telemetry,
            current.content,
            CapabilityId.SEARCH_CATALOG.value,
            answer_source="grounded_model_response",
        )
        return Command(
            goto=END,
            update={"active_invocation": None, "messages": [response]},
        )

    async def answer_response_node(state: ReasoningState) -> Command:
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, AnswerQuestion):
            raise TypeError("answer response requires an answer-question invocation")
        current = state.current_committed_user_message()
        if current is None:
            raise ValueError("answer response requires the current committed caller message")
        if not isinstance(current.content, str):
            raise TypeError("answer response requires plain committed caller text")

        result = await answer_model.ainvoke(
            [
                SystemMessage(
                    compose_answer_response_prompt(
                        display_name,
                        policy,
                        invocation.request,
                    )
                ),
                current,
            ]
        )
        if not isinstance(result, AnswerResponse):
            raise TypeError("answer response model returned an incompatible result")
        if result.decision == "clarify":
            return Command(goto=ANSWER_CLARIFY_NODE, update={"active_invocation": None})
        if result.decision == "unsupported":
            return Command(goto=ANSWER_UNSUPPORTED_NODE, update={"active_invocation": None})

        answer = result.answer
        if answer is None:
            raise RuntimeError("validated answer response omitted its answer")
        model_text_policy.validate(answer)
        record_capability_answered(
            routing_telemetry,
            current.content,
            CapabilityId.ANSWER_QUESTION.value,
            answer_source=(
                "grounded_model_response"
                if invocation.request.topic == "policy"
                else "general_model_response"
            ),
        )
        return Command(
            goto=END,
            update={"active_invocation": None, "messages": [AIMessage(answer)]},
        )

    def answer_clarify_node(state: ReasoningState) -> dict[str, object]:
        if state.active_invocation is not None:
            raise TypeError("answer clarification requires the invocation to be cleared")
        return {"messages": [AIMessage(_ANSWER_CONTEXT_QUESTION)]}

    def answer_unsupported_node(state: ReasoningState) -> dict[str, object]:
        if state.active_invocation is not None:
            raise TypeError("unsupported answer requires the invocation to be cleared")
        return {"messages": [AIMessage(_ANSWER_UNSUPPORTED_RETRY)]}

    return ReadFlowNodes(
        catalog_entry=catalog_entry_node,
        catalog_query_reject=catalog_query_reject_node,
        catalog_response=catalog_response_node,
        answer_response=answer_response_node,
        answer_clarify=answer_clarify_node,
        answer_unsupported=answer_unsupported_node,
        order_status_entry=order_status_entry_node,
        order_status_target_ask=order_status_target_ask_node,
        order_status_target_propose=order_status_target_propose_node,
        order_status_target_confirm=order_status_target_confirm_node,
        order_status_target_reject=order_status_target_reject_node,
        order_status_fulfill=order_status_fulfill_node,
        speakable_nodes=READ_FLOW_SPEAKABLE_NODES,
        model_speech_nodes=READ_FLOW_MODEL_SPEECH_NODES,
    )
