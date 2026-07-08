"""The checkout gated flow (Tier 3, AGENTS §A10/§A10a) — assemble → guardrail → confirm
(HITL interrupt) → place, with the A10a replay invariants encoded in the node structure:

- `assemble` mints the `idempotency_key` + `created_at` into `PendingAction` (AGENTS §A10
  rule 1 puts key-generation with assembly) and persists it in state BEFORE any interrupt —
  so the effect is dedupable however many times the resume path replays (rule 2).
- `confirm` contains the `interrupt()`s and NOTHING with side effects (rule 1); its two
  interrupt call sites (readback, then at most ONE re-confirm) are a fixed, deterministic
  sequence (rule 3) — bounded structurally, not by a counter. No try/except anywhere near
  an interrupt (rule 4).
- `place` is the ONLY effect node, strictly post-interrupt; the OrderStore is the dedup
  ARBITER, so a replayed `place` returns the original order (rule 5 analogue for 3b).

SKU discipline (industry-standard narrowed choice): the model NEVER emits a SKU. Code
resolves candidates from the fixture; `propose_order(candidate_key, quantity)` picks INTO
that list; code resolves key -> sku -> price and computes the total. Consent discipline:
the readback line is GRAPH-authored (the `interrupt()` payload — one-author rule), covers
the fields `place_order`'s ToolConfirmationPolicy declares, and a confirmation given over
a barged-over readback is not consent (§4a) — the node re-confirms.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel

from agnostic_market.agents.checkout.prompt import compose_checkout_prompt
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.orders import OrderStore, resolve_candidates
from agnostic_market.dtos.confirmation import ToolConfirmationPolicy
from agnostic_market.dtos.state import HandoffRequest, PendingAction, PolicyContext, ReasoningState

logger = logging.getLogger("agnostic_market.agents.checkout")

# Clock A: a pending confirmation older than this cannot be resumed — the confirm node
# cancels and re-confirms fresh. Platform floor (same authority model as _ALWAYS_LOCKED);
# merchant-configurable later via policy-within-bounds.
_PENDING_TTL_SECONDS = 120.0

# place_order's declared confirmation contract (VOICE_PIPELINE §7a): the readback MUST
# cover these fields at explicit_yes strength before the effect may run. Enforced below in
# _readback_line (build fails loudly if a declared field has no rendered value).
PLACE_ORDER_POLICY = ToolConfirmationPolicy(
    tool="place_order",
    confirm_fields=frozenset({"quantity", "total_amount"}),
    strength="explicit_yes",
    min_verification_level=0,
)

# Deterministic consent/escape classification — committed transcript only, checked in
# order: human -> no -> yes -> unclear. Negatives before positives ("no, don't do it"
# contains "do it").
_HUMAN_RE = re.compile(
    r"\b(?:human|person|agent|representative|operator|somebody real)\b", re.IGNORECASE
)
_NO_RE = re.compile(
    r"\b(?:no|nope|nah|don'?t|do not|cancel|stop|never ?mind|forget it|wrong)\b", re.IGNORECASE
)
_YES_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|correct|right|confirm|confirmed|go ahead|place it|do it|"
    r"sounds good|please do|ok(?:ay)?)\b",
    re.IGNORECASE,
)
_ABORT_RE = re.compile(
    r"\b(?:stop|never ?mind|forget it|cancel (?:that|it|this)|no thanks|don'?t bother)\b",
    re.IGNORECASE,
)


def wants_human(text: str) -> bool:
    """§A9 no-trap escape: the caller asked for a person."""
    return bool(_HUMAN_RE.search(text))


def is_abort(text: str) -> bool:
    """Explicit abort of the in-flight checkout (entry-router escape)."""
    return bool(_ABORT_RE.search(text))


def _classify_consent(text: str) -> str:
    """'human' | 'no' | 'yes' | 'unclear' — deterministic, order matters."""
    if wants_human(text):
        return "human"
    if _NO_RE.search(text):
        return "no"
    if _YES_RE.search(text):
        return "yes"
    return "unclear"


def _last_user_text(state: ReasoningState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def speak_quantity(quantity: int, name: str) -> str:
    """'one waterproof rain jacket' / 'two waterproof rain jackets' — not '1 x name'
    (TTS reads the 'x' separator literally as 'ex')."""
    plural = name if quantity == 1 else f"{name}s"
    return f"{quantity} {plural}"


def _readback_line(pending: PendingAction, policy: ToolConfirmationPolicy) -> str:
    """The GRAPH-authored confirmation readback — the `interrupt()` payload.

    Renders every field the tool's ToolConfirmationPolicy declares; a declared field with
    no rendered value fails loudly here (readback can't be silently forgotten, §7a).
    """
    rendered: dict[str, str] = {
        "quantity": str(pending.quantity),
        "total_amount": f"${pending.total_usd:.2f}",
    }
    missing = policy.confirm_fields - rendered.keys()
    if missing:
        raise ValueError(f"readback cannot render declared confirm_fields: {sorted(missing)}")
    return (
        f"Just to confirm: {speak_quantity(pending.quantity, pending.name)}, "
        f"{rendered['total_amount']} total. Shall I place the order?"
    )


class _ProposeOrder(BaseModel):
    candidate_key: str
    quantity: int


@dataclass(frozen=True)
class CheckoutNodes:
    """The flow's node callables + its caller-facing (speakable) node names.

    Wiring (add_node/add_edge) is the graph builder's job (frontline.py); this module owns
    only behavior.
    """

    assemble: Callable[[ReasoningState], dict[str, object]]
    guardrail: Callable[[ReasoningState], dict[str, object]]
    confirm: Callable[[ReasoningState], dict[str, object]]
    place: Callable[[ReasoningState], dict[str, object]]
    abort: Callable[[ReasoningState], dict[str, object]]
    escape_human: Callable[[ReasoningState], dict[str, object]]
    speakable_nodes: frozenset[str]


def build_checkout_nodes(
    reasoning_model: BaseChatModel,
    store: OrderStore,
    policy: PolicyContext,
    *,
    display_name: str,
) -> CheckoutNodes:
    """Build the checkout flow's nodes, closed over the session's store + policy (§A5:
    tenant/policy bound in code at build time — never carried in conversation state)."""

    @tool
    def propose_order(candidate_key: str, quantity: int) -> str:
        """Propose the order: the option number the caller chose and how many they want."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    @tool
    def leave_checkout() -> str:
        """Leave the purchase flow (caller changed their mind or asked something else)."""
        raise NotImplementedError("intercepted by the assemble node; never executed")

    model = reasoning_model.bind_tools([propose_order, leave_checkout])

    def assemble_node(state: ReasoningState) -> dict[str, object]:
        """Model turn INSIDE checkout: pick a candidate (by key) + quantity, or clarify,
        or leave. Mints PendingAction (key + created_at) on a valid proposal — A10a rule 2:
        the idempotency key exists in state before any interrupt/effect."""
        candidates = resolve_candidates(store.fixture, _last_user_text(state))
        by_key = {c.key: c for c in candidates}
        prompt = SystemMessage(compose_checkout_prompt(display_name, candidates))
        messages: list = [prompt, *state.messages]
        new_messages: list = []
        # Bounded proposal loop: one invalid proposal gets ONE corrective re-prompt.
        for _attempt in range(2):
            response = model.invoke(messages)
            new_messages.append(response)
            if not response.tool_calls:
                # Clarifying question (streamed tokens reach the caller) — stay in flow.
                return {"messages": new_messages}
            call = response.tool_calls[0]
            if call["name"] == "leave_checkout":
                new_messages.append(
                    ToolMessage("left checkout", tool_call_id=call["id"])
                )
                write_event({"event": "checkout_cancelled", "reason": "left_flow"})
                # "left_checkout" (not None): blocks a same-turn checkout RE-entry —
                # otherwise assemble->gate->handover->assemble could cycle. Reset at entry.
                return {
                    "messages": new_messages,
                    "active_flow": "left_checkout",
                    "pending_action": None,
                }
            try:
                proposal = _ProposeOrder.model_validate(call["args"])
            except ValueError:
                proposal = None
            chosen = by_key.get(proposal.candidate_key) if proposal else None
            if chosen is None or proposal.quantity < 1:
                feedback = (
                    f"Invalid proposal. Valid option numbers: "
                    f"{', '.join(sorted(by_key))}; quantity must be >= 1."
                )
                new_messages.append(ToolMessage(feedback, tool_call_id=call["id"]))
                messages = [prompt, *state.messages, *new_messages]
                continue
            new_messages.append(
                ToolMessage(
                    f"proposed option {chosen.key} x {proposal.quantity}",
                    tool_call_id=call["id"],
                )
            )
            pending = PendingAction(
                sku=chosen.sku,
                name=chosen.name,
                quantity=proposal.quantity,
                # Price and total come from the FIXTURE, never the model (code arithmetic).
                total_usd=round(chosen.price_usd * proposal.quantity, 2),
                idempotency_key=uuid.uuid4().hex,
                created_at=time.time(),
            )
            return {"messages": new_messages, "pending_action": pending}
        # Two invalid proposals: leave the flow; the normal path answers the turn.
        logger.warning("checkout assemble: two invalid proposals; leaving flow")
        write_event({"event": "checkout_cancelled", "reason": "invalid_proposals"})
        return {
            "messages": new_messages,
            "active_flow": "left_checkout",
            "pending_action": None,
        }

    def guardrail_node(state: ReasoningState) -> dict[str, object]:
        """CODE-enforced policy check (never the model): order value cap."""
        pending = state.pending_action
        assert pending is not None  # only reached with a proposal minted
        if pending.total_usd > policy.max_order_value_usd:
            write_event(
                {
                    "event": "checkout_denied",
                    "reason": "order_value_cap",
                    "total": pending.total_usd,
                }
            )
            return {
                "pending_action": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        f"I'm sorry - that order comes to ${pending.total_usd:.2f}, which is "
                        "more than I'm able to place on this call. Nothing has been ordered."
                    )
                ],
            }
        return {}

    def confirm_node(state: ReasoningState) -> dict[str, object]:
        """The HITL gate: readback -> interrupt -> deterministic consent classification.

        A10a shape: NO side effects here; at most TWO interrupt call sites in a fixed
        order; the wall-clock TTL check (Clock A) runs FIRST, so a resume arriving after
        expiry takes the cancel branch without consuming the stale answer.
        """
        pending = state.pending_action
        assert pending is not None  # only reached with a minted pending action
        if time.time() - pending.created_at > _PENDING_TTL_SECONDS:
            write_event({"event": "checkout_expired", "reason": "pending_ttl"})
            # Clear-before-speak: state says "no pending" in the same update as the line.
            return {
                "pending_action": None,
                "active_flow": None,
                "messages": [
                    AIMessage(
                        "That confirmation sat for a while, so I haven't placed anything. "
                        "If you'd still like it, just tell me again."
                    )
                ],
            }
        answer = interrupt(_readback_line(pending, PLACE_ORDER_POLICY))
        verdict = _classify_consent(str(answer.get("text", "")))
        # §4a: consent over a barged-over readback is not consent - re-confirm once.
        if answer.get("readback_interrupted") or verdict == "unclear":
            retry = interrupt(
                f"Sorry - just to be clear: {speak_quantity(pending.quantity, pending.name)}, "
                f"${pending.total_usd:.2f} total. Yes or no?"
            )
            verdict = _classify_consent(str(retry.get("text", "")))
            if verdict != "yes":
                verdict = "human" if verdict == "human" else "no"
        if verdict == "human":
            write_event({"event": "checkout_cancelled", "reason": "human_requested"})
            return {
                "pending_action": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human", reason_code="other", source="model"
                ),
            }
        if verdict == "no":
            write_event({"event": "checkout_cancelled", "reason": "declined"})
            return {
                "pending_action": None,
                "active_flow": None,
                "messages": [
                    AIMessage("Okay, I won't place it - nothing has been ordered.")
                ],
            }
        return {}  # yes: pending survives; the router sends us to place

    def place_node(state: ReasoningState) -> dict[str, object]:
        """The EFFECT node (post-interrupt, own node - A10a rule 1). Idempotent by the
        store's key dedup: a replay returns the ORIGINAL order, so the spoken line and the
        SoR record can never drift or double."""
        pending = state.pending_action
        assert pending is not None  # only reached on confirmed consent
        placed = store.place(
            pending.idempotency_key,
            sku=pending.sku,
            name=pending.name,
            quantity=pending.quantity,
            total_usd=pending.total_usd,
        )
        write_event(
            {"event": "checkout_confirmed", "order_id": placed.order_id, "total": placed.total_usd}
        )
        return {
            "pending_action": None,
            "active_flow": None,
            "messages": [
                AIMessage(
                    f"Done - your order for {speak_quantity(placed.quantity, placed.name)} "
                    f"is placed. Your order number is {placed.order_id}."
                )
            ],
        }

    def abort_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: explicit abort while checkout was in flight."""
        write_event({"event": "checkout_cancelled", "reason": "aborted"})
        return {
            "pending_action": None,
            "active_flow": None,
            "messages": [AIMessage("No problem - I've dropped that. Nothing has been ordered.")],
        }

    def escape_human_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: the caller asked for a person mid-checkout (§A9 no-trap)."""
        write_event({"event": "checkout_cancelled", "reason": "human_requested"})
        return {
            "pending_action": None,
            "active_flow": None,
            "handover": HandoffRequest(destination="human", reason_code="other", source="gate"),
        }

    return CheckoutNodes(
        assemble=assemble_node,
        guardrail=guardrail_node,
        confirm=confirm_node,
        place=place_node,
        abort=abort_node,
        escape_human=escape_human_node,
        speakable_nodes=frozenset(
            {"checkout_guardrail", "checkout_confirm", "checkout_place", "checkout_abort"}
        ),
    )
