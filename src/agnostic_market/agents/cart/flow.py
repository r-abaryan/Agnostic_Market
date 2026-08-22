"""Cart mutation, review, and whole-cart placement behavior.

Typed writes freeze code-resolved proposals before separate confirmation and idempotent effect
nodes. Models may select bounded candidates but never author catalog identity, consent, effect
keys, arithmetic, or caller-visible outcomes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from agnostic_market.agents._consent import classify_consent
from agnostic_market.agents._copy import warm_close
from agnostic_market.agents._toolcalls import ack_extra_tool_calls, unknown_tool_result
from agnostic_market.agents.cart.prompt import compose_cart_capability_prompt, compose_cart_prompt
from agnostic_market.agents.clarification import (
    advance_clarification,
    with_clarification_lifecycle,
)
from agnostic_market.agents.legacy_observation import (
    observe_legacy_capabilities,
    observe_legacy_model_tool_calls,
)
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartMutationRecord, CartStore
from agnostic_market.commerce.orders import (
    Candidate,
    OrderStore,
    PlacedOrder,
    RecentOrderContext,
    match_named_items,
    number_candidates,
    resolve_candidates,
    speak_lines,
)
from agnostic_market.dtos.confirmation import (
    ToolConfirmationPolicy,
    validate_confirmation_rendering,
)
from agnostic_market.dtos.orchestration import (
    CartItemChoices,
    CartItemQuery,
    ModifyCart,
    PlaceOrder,
    ResolvedCartItemRef,
)
from agnostic_market.dtos.state import (
    CartClarification,
    CartClarificationDetail,
    CartLine,
    HandoffRequest,
    PendingCartMutation,
    PendingPlacement,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.cart")

# place_order's declared confirmation contract (VOICE_PIPELINE §7a): the readback MUST speak
# these critical fields at explicit_yes strength. `line_items` (not `quantity`) — a single
# scalar can't honestly describe N line quantities; the readback interpolates the rendered
# `line_items` string VERBATIM, so the loud-fail check guarantees it is actually spoken (the
# anti-theater lesson from the removed `return_id`). Enforced in
# `_placement_confirmation_phrase`.
PLACE_ORDER_POLICY = ToolConfirmationPolicy(
    tool="place_order",
    confirm_fields=frozenset({"line_items", "total_amount"}),
    strength="explicit_yes",
)

_CART_CLARIFICATION_LINES: dict[CartClarificationDetail, str] = {
    "action": "What would you like to do with your cart?",
    "item": "Which item would you like?",
    "quantity": "How many would you like?",
}
_EMPTY_CART_CHECKOUT_LINE = "Your cart's empty - what would you like to add?"
_EMPTY_CART_REVIEW_LINE = "Your cart's empty right now - what would you like to add?"


def _placement_confirmation_phrase(
    pending: PendingPlacement, policy: ToolConfirmationPolicy
) -> str:
    """The canonical, policy-validated description used by both placement prompts.

    Renders every field the policy declares AND interpolates each rendered value verbatim,
    so the loud-fail check guarantees the spoken line literally contains them (no declared-
    but-not-spoken theater). A flagged duplicate flips to the explicit "SECOND order" form.
    """
    rendered: dict[str, str] = {
        "line_items": speak_lines(pending.lines),
        "total_amount": f"${pending.total_usd:.2f}",
    }
    if pending.duplicate_of is not None:
        phrase = (
            f"a SECOND order of {rendered['line_items']} for another "
            f"{rendered['total_amount']}; your earlier order ({pending.duplicate_of}) "
            "is already placed"
        )
    else:
        phrase = f"an order for {rendered['line_items']}, {rendered['total_amount']} total"
    return validate_confirmation_rendering(policy, rendered, phrase)


class _ProposeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_key: str
    quantity: int = Field(strict=True)


class _ProposeKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_key: str


class _ProposeQuantity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: int = Field(strict=True, ge=0)


class _RequestCartClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: CartClarificationDetail


# Reversible cart mutations BATCH within one model response (live call #9 P3: "one from
# each" emitted N calls and N-1 were silently dropped). Control/terminal tools (review,
# checkout, buy_now, leave) stay one-per-turn and must LEAD the response to act.
_MutationOperation = Literal["add", "remove", "set_quantity"]
_MUTATION_OPERATIONS: dict[str, _MutationOperation] = {
    "add_to_cart": "add",
    "remove_from_cart": "remove",
    "set_quantity": "set_quantity",
}
_MUTATION_TOOLS = frozenset(_MUTATION_OPERATIONS)


@dataclass(frozen=True)
class _AddedMutationOutcome:
    line: CartLine


@dataclass(frozen=True)
class _DescribedMutationOutcome:
    fragment: str


@overload
def _apply_resolved_mutation(
    cart_store: CartStore,
    operation: Literal["add"],
    item: Candidate,
    quantity: int,
) -> _AddedMutationOutcome: ...


@overload
def _apply_resolved_mutation(
    cart_store: CartStore,
    operation: Literal["remove", "set_quantity"],
    item: Candidate | CartLine,
    quantity: int | None,
) -> _DescribedMutationOutcome: ...


def _apply_resolved_mutation(
    cart_store: CartStore,
    operation: _MutationOperation,
    item: Candidate | CartLine,
    quantity: int | None,
) -> _AddedMutationOutcome | _DescribedMutationOutcome:
    """Apply one already-resolved reversible mutation and record its factual outcome."""
    if operation == "add":
        if not isinstance(item, Candidate):
            raise TypeError("resolved add requires a catalog candidate")
        if quantity is None or quantity < 1:
            raise ValueError("resolved add requires a positive quantity")
        line = cart_store.add_item(
            sku=item.sku,
            name=item.name,
            price_usd=item.price_usd,
            quantity=quantity,
        )
        write_event({"event": "cart_item_added", "sku": item.sku})
        return _AddedMutationOutcome(line=line)
    if operation == "remove":
        if quantity is not None:
            raise ValueError("resolved remove forbids a quantity")
        removed = cart_store.remove_item(item.sku)
        if removed:
            write_event({"event": "cart_item_removed", "sku": item.sku})
        return _DescribedMutationOutcome(
            fragment=(
                f"removed the {item.name} from your cart"
                if removed
                else f"the {item.name} wasn't in your cart"
            )
        )
    if operation == "set_quantity":
        if quantity is None or quantity < 0:
            raise ValueError("resolved set-quantity requires a non-negative quantity")
        if quantity == 0:
            removed = cart_store.remove_item(item.sku)
            if removed:
                write_event({"event": "cart_quantity_set", "sku": item.sku})
            return _DescribedMutationOutcome(
                fragment=(
                    f"removed the {item.name} from your cart"
                    if removed
                    else f"the {item.name} wasn't in your cart"
                )
            )
        line = cart_store.set_quantity(item.sku, quantity)
        if line is not None:
            write_event({"event": "cart_quantity_set", "sku": item.sku})
        return _DescribedMutationOutcome(
            fragment=(
                f"updated to {speak_lines([line])}"
                if line is not None
                else f"the {item.name} wasn't in your cart"
            )
        )
    raise ValueError(f"unsupported resolved mutation operation: {operation!r}")


@dataclass(frozen=True)
class CartNodes:
    """The flow's node callables + its caller-facing (speakable) node names.

    Wiring (add_node/add_edge) is the graph builder's job (frontline/graph.py); this module
    owns only behavior. Routers that branch purely on state live in graph.py (like checkout);
    `route_after_assemble` is exposed here because it maps the assemble outcome to a node.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    capability_entry: Callable[[ReasoningState], dict[str, object]]
    ack: Callable[[ReasoningState], dict[str, object]]
    clarify: Callable[[ReasoningState], dict[str, object]]
    mutation_confirm: Callable[[ReasoningState], dict[str, object]]
    mutation_apply: Callable[[ReasoningState], dict[str, object]]
    reconcile_mutation: Callable[[CartMutationRecord], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    confirm: Callable[[ReasoningState], dict[str, object]]
    place: Callable[[ReasoningState], dict[str, object]]
    finish_placement: Callable[[PlacedOrder], dict[str, object]]
    abort: Callable[[ReasoningState], dict[str, object]]
    escape_human: Callable[[ReasoningState], dict[str, object]]
    # After assemble: "place" (guardrail) | "ack" | "leave" (gate) | "clarify".
    route_after_assemble: Callable[[ReasoningState], str]
    speakable_nodes: frozenset[str]


def build_cart_nodes(
    reasoning_model: BaseChatModel,
    order_store: OrderStore,
    cart_store: CartStore,
    policy: PolicyContext,
    recent_orders: RecentOrderContext,
    *,
    display_name: str,
) -> CartNodes:
    """Build the cart flow's nodes, closed over the session's stores + policy (§A5:
    tenant/policy bound in code at build time, never carried in conversation state)."""

    @tool
    def add_to_cart(candidate_key: str, quantity: int) -> str:
        """Add an item to the cart: the option number the caller chose and how many."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def remove_from_cart(candidate_key: str) -> str:
        """Remove an item from the cart (the option number of a line already in it)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def set_quantity(candidate_key: str, quantity: int) -> str:
        """Set an item's quantity outright (0 removes it)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def review_cart() -> str:
        """Read the cart contents back to the caller."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def buy_now(candidate_key: str, quantity: int) -> str:
        """Add one item and go straight to placing the order (an explicit immediate buy)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def go_to_checkout() -> str:
        """Place the whole cart as an order (the caller is done shopping)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def request_cart_clarification(detail: CartClarificationDetail) -> str:
        """Ask one platform-authored Cart question for the selected missing detail."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_cart() -> str:
        """Leave the cart flow (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def provide_cart_item(candidate_key: str) -> str:
        """Supply only the missing item using its current code-bounded option number."""
        raise NotImplementedError("intercepted by the capability entry; never executed")

    @tool
    def provide_cart_quantity(quantity: int) -> str:
        """Supply only the missing whole-number quantity for the active cart request."""
        raise NotImplementedError("intercepted by the capability entry; never executed")

    cart_tools = (
        add_to_cart,
        remove_from_cart,
        set_quantity,
        review_cart,
        buy_now,
        go_to_checkout,
        request_cart_clarification,
        leave_cart,
    )
    bound_tool_names = frozenset(tool.name for tool in cart_tools)
    model = reasoning_model.bind_tools(cart_tools)

    def _resolve(candidate_key: str, candidates: list[Candidate]) -> Candidate | None:
        by_key = {c.key: c for c in candidates}
        return by_key.get(candidate_key)

    def _valid_keys(candidates: list[Candidate]) -> str:
        return ", ".join(sorted(c.key for c in candidates))

    def _mutation_ack(added: list[CartLine], fragments: list[str], *, invalid: int = 0) -> str:
        parts = [f"added {speak_lines(added)} to your cart"] if added else []
        parts.extend(fragments)
        ack = ", ".join(parts)
        ack = ack[0].upper() + ack[1:]
        if invalid:
            what = "one of those" if invalid == 1 else "some of those"
            return f"{ack} - {what} didn't go through, could you say it again?"
        return f"{ack}. {warm_close()}"

    def _mint_placement() -> dict[str, object]:
        """Freeze the current cart into a PendingPlacement (A10a rule 2: key exists in state
        before any interrupt/effect). Assumes the cart is non-empty (callers guard).

        KEY SCOPE (decided, per the 3c per-attempt-vs-per-intent distinction): the fresh uuid
        dedups MECHANICAL REPLAY only — the same PendingPlacement resumed/replayed places
        exactly once (place_cart is the arbiter). It does NOT block a second legitimate MINT
        of the same basket (mint K1 → interrupt abandoned → later mint K2). That duplicate-
        INTENT window is covered by two other guards: Clock-A TTL (a stale pending expires)
        and `identical_cart_order` (the guardrail reshapes the readback to name a SECOND
        order — it warns, so a caller can't over-buy silently). For single-turn voice HITL the
        window is narrow, and `place` clears the cart, closing it further. A durable
        per-intent key belongs with the Phase-4 real SoR, not here."""
        return {
            "pending_placement": PendingPlacement(
                lines=cart_store.snapshot(),
                total_usd=cart_store.cart_total(),
                idempotency_key=uuid.uuid4().hex,
                created_at=time.time(),
            )
        }

    def _mint_mutation(
        operation: _MutationOperation,
        item: Candidate | CartLine,
        quantity: int | None,
    ) -> PendingCartMutation:
        current = next((line for line in cart_store.view() if line.sku == item.sku), None)
        return PendingCartMutation(
            operation=operation,
            sku=item.sku,
            name=item.name,
            price_usd=item.price_usd,
            quantity=quantity,
            pre_confirm_quantity=current.quantity if current is not None else 0,
            idempotency_key=uuid.uuid4().hex,
            created_at=time.time(),
        )

    def _apply_candidate_key_mutation(
        call: dict,
        candidates: list[Candidate],
        new_messages: list,
        added: list[CartLine],
        fragments: list[str],
    ) -> bool:
        """Apply ONE reversible cart mutation and append its ToolMessage. Records its ack
        piece: adds collect into `added` (merged into one spoken phrase), remove/set into
        `fragments`. False = invalid proposal (corrective feedback appended instead)."""
        name = call["name"]
        if name == "remove_from_cart":
            try:
                proposal_key = _ProposeKey.model_validate(call["args"])
            except ValueError:
                proposal_key = None
            chosen = _resolve(proposal_key.candidate_key, candidates) if proposal_key else None
            if chosen is None:
                fb = f"Invalid proposal. Valid option numbers: {_valid_keys(candidates)}."
                new_messages.append(ToolMessage(fb, tool_call_id=call["id"]))
                return False
            outcome = _apply_resolved_mutation(cart_store, "remove", chosen, None)
            new_messages.append(ToolMessage(outcome.fragment, tool_call_id=call["id"]))
            fragments.append(outcome.fragment)
            return True
        # add_to_cart / set_quantity
        try:
            proposal = _ProposeItem.model_validate(call["args"])
        except ValueError:
            proposal = None
        chosen = _resolve(proposal.candidate_key, candidates) if proposal else None
        if (
            chosen is None
            or proposal.quantity < 0
            or (name == "add_to_cart" and proposal.quantity < 1)
        ):
            fb = (
                f"Invalid proposal. Valid option numbers: {_valid_keys(candidates)}; "
                "quantity must be a whole number "
                f"{'>= 0' if name == 'set_quantity' else '>= 1'}."
            )
            new_messages.append(ToolMessage(fb, tool_call_id=call["id"]))
            return False
        operation = _MUTATION_OPERATIONS[name]
        outcome = _apply_resolved_mutation(cart_store, operation, chosen, proposal.quantity)
        if operation == "add":
            new_messages.append(
                ToolMessage(f"added {proposal.quantity} of {chosen.key}", tool_call_id=call["id"])
            )
            added.append(outcome.line)
            return True
        new_messages.append(ToolMessage(outcome.fragment, tool_call_id=call["id"]))
        fragments.append(outcome.fragment)
        return True

    def _clarification_result(
        state: ReasoningState,
        new_messages: list,
        detail: CartClarificationDetail,
    ) -> dict[str, object]:
        step = advance_clarification(
            state,
            flow="cart",
            max_reasks=policy.cart_clarification_reask_max,
        )
        if step.exhausted:
            return {
                "messages": new_messages,
                "active_flow": "left_cart",
                "active_invocation": None,
                "pending_clarification": None,
            }
        return {
            "messages": new_messages,
            "active_flow": "cart",
            "pending_clarification": CartClarification(detail=detail),
            "clarification_progress": step.progress,
        }

    def _leave_result(new_messages: list, call_id: str) -> dict[str, object]:
        new_messages.append(ToolMessage("left cart", tool_call_id=call_id))
        write_event({"event": "cart_left", "reason": "left_flow"})
        return {
            "messages": new_messages,
            "active_flow": "left_cart",
            "active_invocation": None,
            "pending_cart_mutation": None,
            "pending_placement": None,
            "pending_clarification": None,
        }

    def capability_entry_node(state: ReasoningState) -> dict[str, object]:
        """Prepare one typed Cart request; code owns live state and every effect boundary."""
        invocation = state.active_invocation
        if invocation is None or not isinstance(invocation.request, ModifyCart | PlaceOrder):
            raise TypeError("cart capability entry requires a typed Cart invocation")
        observe_legacy_capabilities((invocation.request.kind,), source="typed_owner")
        if isinstance(invocation.request, PlaceOrder):
            if cart_store.is_empty():
                return {
                    "active_invocation": None,
                    "active_flow": None,
                    "pending_ack": _EMPTY_CART_CHECKOUT_LINE,
                }
            return {
                "active_invocation": None,
                "active_flow": "cart",
                **_mint_placement(),
            }
        request = invocation.request
        new_messages: list = []

        def domain():
            items = (
                list(order_store.fixture.products)
                if request.operation == "add"
                else list(cart_store.view())
            )
            candidates = number_candidates(items)
            return (
                items,
                candidates,
                {candidate.key: item for candidate, item in zip(candidates, items, strict=True)},
            )

        def retain(updated: ModifyCart) -> None:
            nonlocal invocation, request
            invocation = invocation.with_request(updated)
            request = updated

        def clarify(detail: CartClarificationDetail) -> dict[str, object]:
            result = _clarification_result(state, new_messages, detail)
            if result.get("active_flow") == "cart":
                result["active_invocation"] = invocation
            return result

        def empty_cart() -> dict[str, object]:
            return {
                "messages": new_messages,
                "active_invocation": None,
                "active_flow": None,
                "pending_ack": _EMPTY_CART_REVIEW_LINE,
            }

        items, candidates, by_key = domain()
        if request.operation != "add" and not items:
            return empty_cart()

        if isinstance(request.item, CartItemQuery):
            matches = match_named_items(items, request.item.query)
            if not matches:
                retain(
                    ModifyCart(
                        operation=request.operation,
                        quantity=request.quantity,
                    )
                )
                return clarify("item")
            if len(matches) > 1:
                retain(
                    ModifyCart(
                        operation=request.operation,
                        item=CartItemChoices(skus=tuple(item.sku for item in matches)),
                        quantity=request.quantity,
                    )
                )
                return clarify("item")
            retain(
                ModifyCart(
                    operation=request.operation,
                    item=ResolvedCartItemRef(sku=matches[0].sku),
                    quantity=request.quantity,
                )
            )

        selecting_item = request.item is None or isinstance(request.item, CartItemChoices)
        if isinstance(request.item, CartItemChoices):
            item_by_sku = {item.sku: item for item in items}
            choice_items = [item_by_sku.get(sku) for sku in request.item.skus]
            if any(item is None for item in choice_items):
                retain(ModifyCart(operation=request.operation, quantity=request.quantity))
                return clarify("item")
            narrowed_items = [item for item in choice_items if item is not None]
            candidates = number_candidates(narrowed_items)
            by_key = {
                candidate.key: item
                for candidate, item in zip(candidates, narrowed_items, strict=True)
            }

        if isinstance(request.item, ResolvedCartItemRef):
            live_item = next((item for item in items if item.sku == request.item.sku), None)
            if live_item is None:
                retain(
                    ModifyCart(
                        operation=request.operation,
                        quantity=request.quantity,
                    )
                )
                return clarify("item")
        else:
            live_item = None

        if selecting_item or not request.is_slot_complete():
            proposal_tool = provide_cart_item if selecting_item else provide_cart_quantity
            prompt_candidates = candidates if selecting_item else number_candidates([live_item])
            capability_model = reasoning_model.bind_tools(
                (proposal_tool, request_cart_clarification, leave_cart)
            )
            prompt = SystemMessage(
                compose_cart_capability_prompt(
                    display_name,
                    prompt_candidates,
                    policy,
                    request,
                    proposal_tool.name,
                )
            )
            current_user_message = state.current_committed_user_message()
            if current_user_message is None:
                return clarify("item" if selecting_item else "quantity")
            messages: list = [prompt, current_user_message]
            expected_tool = proposal_tool.name
            for _attempt in range(2):
                response = capability_model.invoke(messages)
                if not response.tool_calls:
                    return clarify("item" if selecting_item else "quantity")
                new_messages.append(response)
                ack_extra_tool_calls(response, new_messages)
                call = response.tool_calls[0]
                if call["name"] == leave_cart.name:
                    return _leave_result(new_messages, call["id"])
                if call["name"] == request_cart_clarification.name:
                    new_messages.append(
                        ToolMessage("cart clarification requested", tool_call_id=call["id"])
                    )
                    return clarify("item" if selecting_item else "quantity")
                if call["name"] == expected_tool:
                    try:
                        if selecting_item:
                            proposal = _ProposeKey.model_validate(call["args"])
                            chosen = by_key.get(proposal.candidate_key)
                            if chosen is None:
                                raise ValueError("unknown candidate key")
                            updated = ModifyCart(
                                operation=request.operation,
                                item=ResolvedCartItemRef(sku=chosen.sku),
                                quantity=request.quantity,
                            )
                        else:
                            proposal = _ProposeQuantity.model_validate(call["args"])
                            if request.operation == "add" and proposal.quantity < 1:
                                raise ValueError("add quantity must be positive")
                            updated = ModifyCart(
                                operation=request.operation,
                                item=request.item,
                                quantity=proposal.quantity,
                            )
                    except (TypeError, ValueError):
                        new_messages.append(
                            ToolMessage(
                                "The proposal was invalid. Fill only the missing field.",
                                tool_call_id=call["id"],
                            )
                        )
                        messages = [prompt, current_user_message, *new_messages]
                        continue
                    new_messages.append(
                        ToolMessage("cart request updated", tool_call_id=call["id"])
                    )
                    retain(updated)
                    if not request.is_slot_complete():
                        return clarify("quantity")
                    break
                if call["name"] not in {
                    expected_tool,
                    request_cart_clarification.name,
                    leave_cart.name,
                }:
                    new_messages.append(unknown_tool_result(call["id"], leave_tool=leave_cart.name))
                    messages = [prompt, current_user_message, *new_messages]
                    continue
                raise ValueError(
                    f"cart capability entry: bound tool has no handler: {call['name']!r}"
                )
            else:
                return clarify("item" if selecting_item else "quantity")

        if not isinstance(request.item, ResolvedCartItemRef):
            raise TypeError("complete cart mutation requires a resolved item")
        items, _candidates, _by_key = domain()
        live_item = next((item for item in items if item.sku == request.item.sku), None)
        if live_item is None:
            if request.operation != "add" and not items:
                return empty_cart()
            retain(ModifyCart(operation=request.operation, quantity=request.quantity))
            return clarify("item")
        return {
            "messages": new_messages,
            "active_invocation": None,
            "active_flow": "cart",
            "pending_cart_mutation": _mint_mutation(
                request.operation,
                live_item,
                request.quantity,
            ),
        }

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE the cart flow: mutate, review, place, clarify, or leave.
        It is not caller-speakable. Code-authored acks use `pending_ack`; clarification uses
        the typed selector rendered by `cart_clarify`.

        Reversible mutations BATCH (live call #9 P3: "one from each" emitted N adds and
        N-1 were silently dropped): when the response LEADS with a mutation, every mutation
        call in it applies in order under ONE combined ack; control calls mixed in behind
        mutations are answered but not acted on (control intent must lead its own turn).
        Control/terminal tools stay one-per-turn as before."""
        candidates = resolve_candidates(order_store.fixture, state.last_user_text())
        prompt = SystemMessage(compose_cart_prompt(display_name, candidates, cart_store, policy))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal/batch gets ONE corrective re-prompt
            response = model.invoke(messages)
            if not response.tool_calls:
                return _clarification_result(state, new_messages, "action")
            observe_legacy_model_tool_calls(response.tool_calls)
            new_messages.append(response)

            if response.tool_calls[0]["name"] in _MUTATION_TOOLS:
                added: list[CartLine] = []
                fragments: list[str] = []
                invalid = 0
                for call in response.tool_calls:
                    if call["name"] not in _MUTATION_TOOLS:
                        # F-4 invariant: every tool_use gets a tool_result, acted on or not.
                        new_messages.append(
                            ToolMessage(
                                "ignored - call this leading its own turn",
                                tool_call_id=call["id"],
                            )
                        )
                    elif not _apply_candidate_key_mutation(
                        call, candidates, new_messages, added, fragments
                    ):
                        invalid += 1
                if not added and not fragments:  # whole batch invalid -> corrective re-prompt
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                return {
                    "messages": new_messages,
                    "pending_ack": _mutation_ack(added, fragments, invalid=invalid),
                }

            ack_extra_tool_calls(response, new_messages)
            call = response.tool_calls[0]
            name = call["name"]

            if name == "request_cart_clarification":
                try:
                    clarification = _RequestCartClarification.model_validate(call["args"])
                except ValueError:
                    clarification = None
                if clarification is None:
                    new_messages.append(
                        ToolMessage("Invalid clarification request.", tool_call_id=call["id"])
                    )
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                new_messages.append(
                    ToolMessage("cart clarification requested", tool_call_id=call["id"])
                )
                return _clarification_result(state, new_messages, clarification.detail)

            if name == "leave_cart":
                return _leave_result(new_messages, call["id"])

            if name == "review_cart":
                new_messages.append(ToolMessage("reviewed cart", tool_call_id=call["id"]))
                if cart_store.is_empty():
                    ack = _EMPTY_CART_REVIEW_LINE
                else:
                    ack = (
                        f"You've got {speak_lines(cart_store.view())} - "
                        f"that's ${cart_store.cart_total():.2f}. {warm_close()}"
                    )
                return {"messages": new_messages, "pending_ack": ack}

            if name == "go_to_checkout":
                new_messages.append(ToolMessage("go to checkout", tool_call_id=call["id"]))
                if cart_store.is_empty():
                    return {
                        "messages": new_messages,
                        "pending_ack": _EMPTY_CART_CHECKOUT_LINE,
                    }
                return {"messages": new_messages, **_mint_placement()}

            # buy_now: direct-buy resolves a candidate key, then goes straight to the
            # placement tail (add_to_cart/set_quantity/remove_from_cart batch above).
            if name == "buy_now":
                try:
                    proposal = _ProposeItem.model_validate(call["args"])
                except ValueError:
                    proposal = None
                chosen = _resolve(proposal.candidate_key, candidates) if proposal else None
                if chosen is None or proposal.quantity < 1:
                    fb = (
                        f"Invalid proposal. Valid option numbers: {_valid_keys(candidates)}; "
                        "quantity must be a whole number >= 1."
                    )
                    new_messages.append(ToolMessage(fb, tool_call_id=call["id"]))
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                cart_store.add_item(
                    sku=chosen.sku,
                    name=chosen.name,
                    price_usd=chosen.price_usd,
                    quantity=proposal.quantity,
                )
                new_messages.append(
                    ToolMessage(
                        f"buy now {proposal.quantity} of {chosen.key}", tool_call_id=call["id"]
                    )
                )
                write_event({"event": "cart_item_added", "sku": chosen.sku, "reason": "buy_now"})
                return {"messages": new_messages, **_mint_placement()}

            if name not in bound_tool_names:
                new_messages.append(unknown_tool_result(call["id"], leave_tool=leave_cart.name))
                messages = [prompt, *state.messages, *new_messages]
                continue
            # A bound name reaching here is code drift: preserve the loud developer failure.
            raise ValueError(f"cart assemble: bound tool has no handler: {name!r}")

        logger.warning("cart assemble: two invalid tool calls; asking for action in code")
        return _clarification_result(state, new_messages, "action")

    def ack_node(state: ReasoningState) -> dict[str, object]:
        """Speak the code-authored in-flow line (mutation ack / review listing / empty-cart)
        and clear it. This node preserves `active_flow`: broad Cart stays sticky, while a completed
        typed mutation has already cleared it. Clear-before-speak: `pending_ack` goes None in the
        same update as the spoken message."""
        ack = state.pending_ack or warm_close()
        return {"pending_ack": None, "messages": [AIMessage(ack)]}

    def _mutation_action(pending: PendingCartMutation) -> str:
        if pending.operation == "add":
            return f"add {pending.quantity} of {pending.name} to your cart"
        if pending.operation == "remove":
            return f"remove {pending.name} from your cart"
        return f"set {pending.name} to {pending.quantity} in your cart"

    def mutation_confirm_node(state: ReasoningState) -> dict[str, object]:
        """Obtain explicit consent for one frozen cart mutation."""
        pending = state.pending_cart_mutation
        if not isinstance(pending, PendingCartMutation):
            raise TypeError("cart mutation confirmation requires a pending mutation")
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "cart_mutation_expired", "reason": "pending_ttl"})
            return {
                "pending_cart_mutation": None,
                "active_flow": None,
                "messages": [
                    AIMessage("That confirmation expired, so I haven't changed your cart.")
                ],
            }
        action = _mutation_action(pending)
        answer = interrupt(f"Just to confirm: {action}?")
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(f"Sorry - just to be clear: {action}. Please say yes or no.")
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "cart_mutation_cancelled", "reason": "human_requested"})
            return {
                "pending_cart_mutation": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="other", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "cart_mutation_cancelled", "reason": "declined"})
            return {
                "pending_cart_mutation": None,
                "active_flow": None,
                "messages": [AIMessage("Okay, I won't change your cart.")],
            }
        return {}

    def _mutation_result_ack(record: CartMutationRecord) -> str:
        if record.operation == "add":
            if record.quantity is None:
                raise ValueError("authoritative add result is missing its applied delta")
            line = CartLine(
                sku=record.sku,
                name=record.name,
                price_usd=record.price_usd,
                quantity=record.quantity,
            )
            return _mutation_ack([line], [])
        if record.operation == "remove":
            fragment = (
                f"removed the {record.name} from your cart"
                if record.outcome == "applied"
                else f"the {record.name} wasn't in your cart"
            )
            return _mutation_ack([], [fragment])
        if record.final_quantity == 0:
            fragment = (
                f"removed the {record.name} from your cart"
                if record.outcome == "applied"
                else f"the {record.name} wasn't in your cart"
            )
        else:
            line = CartLine(
                sku=record.sku,
                name=record.name,
                price_usd=record.price_usd,
                quantity=record.final_quantity,
            )
            fragment = f"updated to {speak_lines([line])}"
        return _mutation_ack([], [fragment])

    def finish_mutation(
        record: CartMutationRecord,
        *,
        reconciled: bool = False,
        speak_now: bool = False,
    ) -> dict[str, object]:
        """Project one authoritative mutation result into state and speech."""
        ack = _mutation_result_ack(record)
        update: dict[str, object] = {
            "pending_cart_mutation": None,
            "active_flow": None,
        }
        if speak_now:
            update["messages"] = [AIMessage(ack)]
        else:
            update["pending_ack"] = ack
        if not reconciled and record.outcome == "applied":
            event = {
                "add": "cart_item_added",
                "remove": "cart_item_removed",
                "set_quantity": "cart_quantity_set",
            }[record.operation]
            write_event({"event": event, "sku": record.sku})
        return update

    def mutation_apply_node(state: ReasoningState) -> dict[str, object]:
        """Apply the confirmed mutation once through the authoritative store."""
        pending = state.pending_cart_mutation
        if not isinstance(pending, PendingCartMutation):
            raise TypeError("cart mutation apply requires a pending mutation")
        record = cart_store.apply_confirmed_mutation(
            pending.idempotency_key,
            operation=pending.operation,
            sku=pending.sku,
            name=pending.name,
            price_usd=pending.price_usd,
            quantity=pending.quantity,
            pre_confirm_quantity=pending.pre_confirm_quantity,
        )
        return finish_mutation(record)

    def reconcile_mutation(record: CartMutationRecord) -> dict[str, object]:
        """Project a committed receipt without replaying effect telemetry."""
        return finish_mutation(record, reconciled=True, speak_now=True)

    def clarify_node(state: ReasoningState) -> dict[str, object]:
        clarification = state.pending_clarification
        if not isinstance(clarification, CartClarification):
            raise TypeError("cart clarify node requires CartClarification")
        line = _CART_CLARIFICATION_LINES[clarification.detail]
        invocation = state.active_invocation
        if (
            clarification.detail == "item"
            and invocation is not None
            and isinstance(invocation.request, ModifyCart)
            and isinstance(invocation.request.item, CartItemChoices)
        ):
            request = invocation.request
            items = (
                order_store.fixture.products if request.operation == "add" else cart_store.view()
            )
            item_by_sku = {item.sku: item for item in items}
            choices = [item_by_sku.get(sku) for sku in request.item.skus]
            if all(item is not None for item in choices):
                rendered = "; ".join(
                    f"option {index}, {item.name} at ${item.price_usd:.2f}"
                    for index, item in enumerate(choices, start=1)
                    if item is not None
                )
                line = f"I found multiple matches: {rendered}. Which item would you like?"
        return {
            "pending_clarification": None,
            "active_flow": "cart",
            "messages": [AIMessage(line)],
        }

    def guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced policy on the WHOLE cart: order-value cap on the cart total +
        duplicate-order disambiguation (an identical live order this session flips the
        readback to the explicit "SECOND order" form)."""
        pending = state.pending_placement
        assert pending is not None
        if pending.total_usd > policy.max_order_value_usd:
            write_event(
                {
                    "event": "checkout_denied",
                    "reason": "order_value_cap",
                    "total": pending.total_usd,
                }
            )
            return {
                "pending_placement": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"I'm sorry - that order comes to ${pending.total_usd:.2f}, which is more "
                        "than I'm able to place on this call. Your cart's still saved if you'd "
                        "like to change it."
                    )
                ],
            }
        dup = order_store.identical_cart_order(pending.lines)
        if dup is not None:
            write_event({"event": "checkout_duplicate_flagged", "existing": dup.order_id})
            return {"pending_placement": pending.model_copy(update={"duplicate_of": dup.order_id})}
        return {}

    def confirm_node(state: ReasoningState) -> dict[str, object]:
        """The HITL gate: readback -> interrupt -> deterministic consent. NO side effects; at
        most TWO interrupt sites; Clock-A TTL checked FIRST (clear-before-speak) so a stale
        resume cancels without consuming the answer."""
        pending = state.pending_placement
        assert pending is not None
        if time.time() - pending.created_at > policy.pending_ttl_seconds:
            write_event({"event": "checkout_expired", "reason": "pending_ttl"})
            return {
                "pending_placement": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That confirmation sat for a while, so I haven't placed anything. Your "
                        "cart's still saved - just tell me when you're ready."
                    )
                ],
            }
        phrase = _placement_confirmation_phrase(pending, PLACE_ORDER_POLICY)
        answer = interrupt(f"Just to confirm: shall I place {phrase}?")
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(f"Sorry - just to be clear: shall I place {phrase}? Yes or no?")
            verdict = classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "checkout_cancelled", "reason": "human_requested"})
            return {
                "pending_placement": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="other", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "checkout_cancelled", "reason": "declined"})
            return {
                "pending_placement": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "Okay, I won't place it - your cart's still saved if you want to change "
                        "anything."
                    )
                ],
            }
        return {}  # yes: pending survives; the router sends us to place

    def finish_placement(placed: PlacedOrder) -> dict[str, object]:
        """Finish one authoritative placement; safe to re-run after receipt reconciliation."""
        update = {
            "pending_placement": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - your order for {speak_lines(placed.lines)} is placed. Your order "
                    f"number is {placed.order_id}."
                )
            ],
        }
        cart_store.clear()
        recent_orders.record([placed.order_id], operation="place")
        write_event(
            {"event": "checkout_confirmed", "order_id": placed.order_id, "total": placed.total_usd}
        )
        return update

    def place_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Idempotent by the store's
        key dedup: a replay returns the ORIGINAL order. Clears the cart on success."""
        pending = state.pending_placement
        assert pending is not None
        placed = order_store.place_cart(
            pending.idempotency_key, lines=pending.lines, total_usd=pending.total_usd
        )
        return finish_placement(placed)

    def abort_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: explicit abort mid-flow. Drops the pending PLACEMENT but
        KEEPS the cart (D6: abandoning checkout doesn't empty the basket)."""
        write_event({"event": "checkout_cancelled", "reason": "aborted"})
        kept = "" if cart_store.is_empty() else " Your cart's still saved."
        return {
            "pending_cart_mutation": None,
            "pending_placement": None,
            "pending_clarification": None,
            "clarification_progress": None,
            "active_invocation": None,
            "active_flow": None,
            "messages": [AIMessage(f"No problem - I've dropped that.{kept}")],
        }

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-flow (§A9 no-trap)."""
        write_event({"event": "checkout_cancelled", "reason": "human_requested"})
        return {
            "pending_cart_mutation": None,
            "pending_placement": None,
            "pending_clarification": None,
            "clarification_progress": None,
            "active_invocation": None,
            "active_flow": None,
            "handover": HandoffRequest(destination="human", reason_code="other", source="gate"),
        }

    def route_after_assemble(state: ReasoningState) -> str:
        if state.active_flow == "left_cart":
            return "leave"  # explicit leave -> normal pipeline answers
        if state.pending_cart_mutation is not None:
            return "mutation_confirm"
        if state.pending_placement is not None:
            return "place"  # buy_now / go_to_checkout minted a placement
        if state.pending_ack is not None:
            return "ack"  # a mutation / review / empty-cart line to speak
        if state.pending_clarification is not None:
            if not isinstance(state.pending_clarification, CartClarification):
                raise TypeError("cart assemble produced a non-cart clarification")
            return "clarify"
        raise RuntimeError("cart assemble produced no outcome")

    return CartNodes(
        assemble=with_clarification_lifecycle(assemble_node),
        capability_entry=with_clarification_lifecycle(capability_entry_node),
        ack=ack_node,
        clarify=clarify_node,
        mutation_confirm=mutation_confirm_node,
        mutation_apply=mutation_apply_node,
        reconcile_mutation=reconcile_mutation,
        guardrail=guardrail_node,
        confirm=confirm_node,
        place=place_node,
        finish_placement=finish_placement,
        abort=abort_node,
        escape_human=escape_human_node,
        route_after_assemble=route_after_assemble,
        speakable_nodes=frozenset(
            {
                "cart_ack",
                "cart_clarify",
                "cart_mutation_confirm",
                "cart_guardrail",
                "cart_confirm",
                "cart_place",
                "cart_abort",
            }
        ),
    )
