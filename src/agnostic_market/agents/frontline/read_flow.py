"""Typed read/answer owners that do not belong to a transactional domain flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from agnostic_market.agents.frontline.prompt import compose_catalog_response_prompt
from agnostic_market.agents.telemetry import write_capability_answered
from agnostic_market.commerce.orders import OrderStore, lookup_catalog
from agnostic_market.dtos.orchestration import CapabilityId, SearchCatalog
from agnostic_market.dtos.state import PolicyContext, ReasoningState

CATALOG_ENTRY_NODE = "catalog_entry"
CATALOG_QUERY_CLARIFY_NODE = "catalog_query_clarify"
CATALOG_QUERY_REJECT_NODE = "catalog_query_reject"
CATALOG_RESPONSE_NODE = "catalog_response"
READ_FLOW_SPEAKABLE_NODES = frozenset({CATALOG_QUERY_CLARIFY_NODE, CATALOG_QUERY_REJECT_NODE})
READ_FLOW_MODEL_SPEECH_NODES = frozenset({CATALOG_RESPONSE_NODE})

_CATALOG_QUERY_QUESTION = "What product would you like me to look for?"
_CATALOG_QUERY_REJECTED = "I didn't get a product to look for. What else can I help with?"


@dataclass(frozen=True)
class ReadFlowNodes:
    catalog_entry: Callable[[ReasoningState], Command]
    catalog_query_clarify: Callable[[ReasoningState], dict[str, object]]
    catalog_query_reject: Callable[[ReasoningState], dict[str, object]]
    catalog_response: Callable[[ReasoningState], Command]
    speakable_nodes: frozenset[str]
    model_speech_nodes: frozenset[str]


def build_read_flow_nodes(
    response_model: BaseChatModel,
    order_store: OrderStore,
    policy: PolicyContext,
    *,
    display_name: str,
) -> ReadFlowNodes:
    """Build the typed catalog owner over session-bound dependencies."""

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
        if invocation.opened_turn_id == state.consumed_turn_ids[-1]:
            return Command(goto=CATALOG_QUERY_CLARIFY_NODE)

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

    def catalog_query_clarify_node(state: ReasoningState) -> dict[str, object]:
        invocation = state.active_invocation
        if (
            invocation is None
            or not isinstance(invocation.request, SearchCatalog)
            or invocation.request.query is not None
        ):
            raise TypeError("catalog query clarification requires an incomplete catalog request")
        return {"messages": [AIMessage(_CATALOG_QUERY_QUESTION)]}

    def catalog_query_reject_node(state: ReasoningState) -> dict[str, object]:
        if state.active_invocation is not None:
            raise TypeError("catalog query rejection requires the invocation to be cleared")
        return {"messages": [AIMessage(_CATALOG_QUERY_REJECTED)]}

    def catalog_response_node(state: ReasoningState) -> Command:
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

        result = lookup_catalog(order_store.fixture, request.query)
        response = response_model.invoke(
            [
                SystemMessage(compose_catalog_response_prompt(display_name, policy, result)),
                current,
            ]
        )
        if not isinstance(response, AIMessage):
            raise TypeError("catalog response model returned an incompatible message")
        if response.tool_calls:
            raise ValueError("catalog response model returned an unexpected tool call")
        if not response.text.strip():
            raise ValueError("catalog response model returned blank text")

        write_capability_answered(
            current.content,
            CapabilityId.SEARCH_CATALOG.value,
            answer_source="grounded_model_response",
        )
        return Command(
            goto=END,
            update={"active_invocation": None, "messages": [response]},
        )

    return ReadFlowNodes(
        catalog_entry=catalog_entry_node,
        catalog_query_clarify=catalog_query_clarify_node,
        catalog_query_reject=catalog_query_reject_node,
        catalog_response=catalog_response_node,
        speakable_nodes=READ_FLOW_SPEAKABLE_NODES,
        model_speech_nodes=READ_FLOW_MODEL_SPEECH_NODES,
    )
