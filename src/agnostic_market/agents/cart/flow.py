"""The cart gated flow (Tier 3, AGENTS §A10/§A10a) — Group B.

ONE flow owns the whole purchase journey, with the cart/placement distinction kept legible:

  MUTATION (reversible)          PLACEMENT (irreversible — the hardened tail from checkout 3b)
  assemble ─add/remove/qty──▶ ack        assemble ─buy_now/checkout──▶ guardrail
     │  review_cart ──────────▶ ack           │                            │
     │  empty checkout ────────▶ ack           ▼                            ▼
     └─ leave_cart                        confirm[HITL interrupt] ──yes──▶ place (place_cart)

Cart mutations are reversible → a lightweight spoken ack (`cart_ack`), NO HITL. The single
irreversible effect (placing the whole cart as one order) keeps every checkout-3b invariant:
the `idempotency_key`+`created_at` are minted into `PendingPlacement` (a FROZEN whole-cart
snapshot) BEFORE the interrupt (A10a rule 2); `confirm` holds the interrupt(s) and NO side
effects; the §4a barged-readback re-confirm; the Clock-A TTL checked first; `place` is the
sole post-interrupt effect, idempotent by the OrderStore key dedup.

`cart_assemble` never speaks. Dynamic mutation/review/empty-cart results travel through the
turn-scoped `pending_ack` channel to `cart_ack`; closed clarification selectors travel through
`pending_clarification` to `cart_clarify`. Each code-owned node speaks and clears its own channel.

SKU discipline (unchanged from checkout): the model NEVER emits a SKU — it picks a candidate
KEY; code resolves key → sku → name → price and does all arithmetic.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict

from agnostic_market.agents._consent import classify_consent
from agnostic_market.agents._copy import warm_close
from agnostic_market.agents._toolcalls import ack_extra_tool_calls, unknown_tool_result
from agnostic_market.agents.cart.prompt import compose_cart_prompt
from agnostic_market.agents.clarification import (
    advance_clarification,
    with_clarification_lifecycle,
)
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import (
    OrderStore,
    PlacedOrder,
    RecentOrderContext,
    resolve_candidates,
    speak_lines,
)
from agnostic_market.dtos.confirmation import (
    ToolConfirmationPolicy,
    validate_confirmation_rendering,
)
from agnostic_market.dtos.state import (
    CartClarification,
    CartClarificationDetail,
    CartLine,
    HandoffRequest,
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


def _last_user_text(state: ReasoningState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


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
    quantity: int


class _ProposeKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_key: str


class _RequestCartClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: CartClarificationDetail


# Reversible cart mutations BATCH within one model response (live call #9 P3: "one from
# each" emitted N calls and N-1 were silently dropped). Control/terminal tools (review,
# checkout, buy_now, leave) stay one-per-turn and must LEAD the response to act.
_MUTATION_TOOLS = frozenset({"add_to_cart", "set_quantity", "remove_from_cart"})


@dataclass(frozen=True)
class CartNodes:
    """The flow's node callables + its caller-facing (speakable) node names.

    Wiring (add_node/add_edge) is the graph builder's job (frontline/graph.py); this module
    owns only behavior. Routers that branch purely on state live in graph.py (like checkout);
    `route_after_assemble` is exposed here because it maps the assemble outcome to a node.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    ack: Callable[[ReasoningState], dict[str, object]]
    clarify: Callable[[ReasoningState], dict[str, object]]
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

    def _resolve(candidate_key: str, candidates: list) -> object | None:
        by_key = {c.key: c for c in candidates}
        return by_key.get(candidate_key)

    def _valid_keys(candidates: list) -> str:
        return ", ".join(sorted(c.key for c in candidates))

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

    def _apply_mutation(
        call: dict,
        candidates: list,
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
            removed = cart_store.remove_item(chosen.sku)
            new_messages.append(ToolMessage(f"removed {chosen.key}", tool_call_id=call["id"]))
            write_event({"event": "cart_item_removed", "sku": chosen.sku})
            fragments.append(
                f"removed the {chosen.name} from your cart"
                if removed
                else f"the {chosen.name} wasn't in your cart"
            )
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
        if name == "add_to_cart":
            line = cart_store.add_item(
                sku=chosen.sku,
                name=chosen.name,
                price_usd=chosen.price_usd,
                quantity=proposal.quantity,
            )
            new_messages.append(
                ToolMessage(f"added {proposal.quantity} of {chosen.key}", tool_call_id=call["id"])
            )
            write_event({"event": "cart_item_added", "sku": chosen.sku})
            added.append(line)
            return True
        line = cart_store.set_quantity(chosen.sku, proposal.quantity)
        new_messages.append(
            ToolMessage(f"set {chosen.key} to {proposal.quantity}", tool_call_id=call["id"])
        )
        write_event({"event": "cart_quantity_set", "sku": chosen.sku})
        fragments.append(
            f"updated to {speak_lines([line])}"
            if line
            else f"removed the {chosen.name} from your cart"
        )
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
                "pending_clarification": None,
            }
        return {
            "messages": new_messages,
            "active_flow": "cart",
            "pending_clarification": CartClarification(detail=detail),
            "clarification_progress": step.progress,
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
        candidates = resolve_candidates(order_store.fixture, _last_user_text(state))
        prompt = SystemMessage(compose_cart_prompt(display_name, candidates, cart_store, policy))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        for _attempt in range(2):  # one invalid proposal/batch gets ONE corrective re-prompt
            response = model.invoke(messages)
            if not response.tool_calls:
                return _clarification_result(state, new_messages, "action")
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
                    elif not _apply_mutation(call, candidates, new_messages, added, fragments):
                        invalid += 1
                if not added and not fragments:  # whole batch invalid -> corrective re-prompt
                    messages = [prompt, *state.messages, *new_messages]
                    continue
                parts = [f"added {speak_lines(added)} to your cart"] if added else []
                parts.extend(fragments)
                ack = ", ".join(parts)
                ack = ack[0].upper() + ack[1:]
                if invalid:
                    what = "one of those" if invalid == 1 else "some of those"
                    ack += f" - {what} didn't go through, could you say it again?"
                else:
                    ack += f". {warm_close()}"  # its own sentence (proper case), factual
                return {"messages": new_messages, "pending_ack": ack}

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
                new_messages.append(ToolMessage("left cart", tool_call_id=call["id"]))
                write_event({"event": "cart_left", "reason": "left_flow"})
                return {
                    "messages": new_messages,
                    "active_flow": "left_cart",
                    "pending_placement": None,
                    "pending_clarification": None,
                }

            if name == "review_cart":
                new_messages.append(ToolMessage("reviewed cart", tool_call_id=call["id"]))
                if cart_store.is_empty():
                    ack = "Your cart's empty right now - what would you like to add?"
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
                        "pending_ack": "Your cart's empty - what would you like to add?",
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
        and clear it. The flow stays sticky (`active_flow` unchanged). Clear-before-speak:
        `pending_ack` goes None in the same update as the spoken message."""
        ack = state.pending_ack or warm_close()
        return {"pending_ack": None, "messages": [AIMessage(ack)]}

    def clarify_node(state: ReasoningState) -> dict[str, object]:
        clarification = state.pending_clarification
        if not isinstance(clarification, CartClarification):
            raise TypeError("cart clarify node requires CartClarification")
        return {
            "pending_clarification": None,
            "active_flow": "cart",
            "messages": [AIMessage(_CART_CLARIFICATION_LINES[clarification.detail])],
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
            "pending_placement": None,
            "pending_clarification": None,
            "clarification_progress": None,
            "active_flow": None,
            "messages": [AIMessage(f"No problem - I've dropped that.{kept}")],
        }

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-flow (§A9 no-trap)."""
        write_event({"event": "checkout_cancelled", "reason": "human_requested"})
        return {
            "pending_placement": None,
            "pending_clarification": None,
            "clarification_progress": None,
            "active_flow": None,
            "handover": HandoffRequest(destination="human", reason_code="other", source="gate"),
        }

    def route_after_assemble(state: ReasoningState) -> str:
        if state.active_flow == "left_cart":
            return "leave"  # explicit leave -> normal pipeline answers
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
        ack=ack_node,
        clarify=clarify_node,
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
                "cart_guardrail",
                "cart_confirm",
                "cart_place",
                "cart_abort",
            }
        ),
    )
