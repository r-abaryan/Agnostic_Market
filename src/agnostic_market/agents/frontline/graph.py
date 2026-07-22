"""The reasoning graph: frontline agent (Tier 1) + the cart & support gated flows (Tier 3).

The load-bearing safety invariant: the frontline holds NO state-changing tools. It can
answer (read-only tools — incl. `view_cart`, Group B) or hand over — so a wrong judgment is
at worst "answered without acting" (recoverable), never a dangerous mutation. Handover fires
two ways:
  - the model's `request_handover` tool — the PRIMARY escalation decider (reads intent,
    including paraphrases; industry-consistent LLM-router stance);
  - the deterministic gate (agents/gate.py) — a slim pre-generation floor for high-certainty
    IRREVERSIBLE requests only (cancel/refund/place-order). NOT the router; a bonus fast-path.
Safety is STRUCTURAL (no sensitive tools), not dependent on either catcher hitting 100%.

Graph shape (Group B):
    entry -> (sticky in a flow? escapes: human -> abort -> CROSS-SWITCH -> assemble : gate)
    gate  -> (trip? handover : model) ; model -> tools -> (handover? handover : model)
    handover -> (cart/support(refund|cancel)? enter flow : spoken deferral -> END)
    cart: assemble -> {mutate: ack | place: guardrail -> confirm[INTERRUPT] -> place_cart}
    support:  assemble -> {refund: guardrail -> [step-up] -> confirm[INT] -> place |
                           cancel: guardrail -> confirm[INT] -> void}
The cart flow owns BOTH cart mutation (reversible, ack) AND the whole-cart placement tail
(the single irreversible effect). The `checkout` handover DESTINATION (legacy name — the
gate's cart_write and the model's cart_write) enters the cart flow; `_GATE_OWNER` maps that
destination to the "cart" flow so the cross-switch guard compares like with like.

The CROSS-SWITCH escape: while sticky, a gate-certain intent for a DIFFERENT flow abandons
the current one and hands over — closes the sticky-flow trap.

The ONLY state-changing tool in the whole graph is the cart flow's place_cart effect, and it
sits strictly behind the guardrail + HITL interrupt (agents/cart/flow.py, AGENTS §A10a).
`request_handover` is a REAL executed tool that returns a `Command` routing to the handover
node. The model-facing prompt + few-shot live in `prompt.py` (F1: eval and production share
one prompt path). Node-authored caller copy (the deferral map) stays here — behavior.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from agnostic_market.agents._consent import (
    classify_consent,
    is_abort,
    is_support_abort,
    wants_human,
)
from agnostic_market.agents._copy import guest_list_close, warm_close
from agnostic_market.agents.cart import build_cart_nodes
from agnostic_market.agents.frontline.prompt import compose_system_prompt, resolved_order_line
from agnostic_market.agents.gate import enumeration_check, gate_check, status_check
from agnostic_market.agents.identity import build_identity_nodes
from agnostic_market.agents.support import build_support_nodes
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import (
    BoundIdentity,
    CallerIdentityStore,
    CustomerDirectory,
    order_read_allowed,
)
from agnostic_market.commerce.orders import (
    OrderStore,
    RecentOrderContext,
    render_cart_line,
    render_order_list_line,
    render_order_status_line,
)
from agnostic_market.commerce.profile import ProfileStore
from agnostic_market.commerce.verification import OtpProvider, RiskProvider, VerificationStore
from agnostic_market.dtos.orchestration import (
    IntentRequest,
    ListOrders,
    PrincipalTransition,
    SwitchAccount,
    VerificationProof,
)
from agnostic_market.dtos.state import (
    ActiveFlow,
    HandoffDestination,
    HandoffReasonCode,
    HandoffRequest,
    PolicyContext,
    ReasoningState,
)

logger = logging.getLogger("agnostic_market.agents.frontline")

_HANDOVER_TOOL_NAME = "request_handover"

# Graph topology is the source of truth for model-authored speech provenance. Identity has
# completed its 3b code-authored clarification migration; Support and Cart retain compatibility
# speech permission until their own atomic migrations.
TRANSACTIONAL_MODEL_NODES = frozenset({"cart_assemble", "support_assemble", "identity_assemble"})
MODEL_SPEECH_NODES = frozenset({"model"}) | (TRANSACTIONAL_MODEL_NODES - {"identity_assemble"})
FRONTLINE_SPEAKABLE_NODES = frozenset(
    {
        "handover",
        "automation_terminal_response",
        "principal_warning",
        "read_render",
        "forced_status",
    }
)

# The gate speaks in DESTINATION names (its enum is the handover destination); the entry
# router reasons in FLOW names (active_flow). They differ for the cart flow: the gate's
# place-order rule maps to the legacy "checkout" destination, but the flow that serves it is
# "cart". This map normalizes destination -> owning flow so the cross-switch guard ("a
# gate-certain intent for a DIFFERENT flow") compares like with like — without it, a
# "checkout now" utterance while sticky in cart would spuriously cross-switch into itself.
_GATE_OWNER: dict[HandoffDestination, ActiveFlow] = {
    "checkout": "cart",
    "support": "support",
}

_TOOL_LIMIT_FALLBACK = (
    "I've reached the lookup limit for this turn. Please ask again with the specific item "
    "or order you want me to check."
)

# The read-only loop bound comes from `policy.max_tool_hops`. `model_node` switches to the
# unbound model at the limit, producing a final answer without another dangling tool call.

# Reads whose result a CODE renderer speaks directly, skipping the second model pass (L3
# latency + grounding win). A SINGLE such call with nothing else pending is deterministic —
# code renders it and ENDs. catalog_search is deliberately EXCLUDED: product discovery is
# fuzzy and needs the model to frame ("we don't carry X, but we do have Y"). list_orders
# renders only on a BOUND session (`_render_ready` gates it); its UNVERIFIED branch takes
# the deterministic enumeration divert instead (below) — either way, no second model pass.
_RENDERABLE_READS = frozenset({"order_status", "view_cart", "list_orders"})

# 3a deferral copy — destination-keyed and HONEST: nothing downstream exists yet, so we do
# NOT promise a live connection. Structure carries into 3b/3c where the promises come true.
# (Node-authored caller-facing copy, not a model prompt — stays here with the graph.)
_DEFERRAL: dict[HandoffDestination, str] = {
    "support": (
        "That's something I'm not able to do on this call yet, but I'll make sure it reaches "
        "our support team."
    ),
    "checkout": (
        "I can't complete that change to your order on this call yet, but I'll pass it along "
        "so it can be handled."
    ),
    "planner": "That's a bit more than I can handle here yet, but I'll make sure it's picked up.",
}

_AUTOMATION_TERMINAL_LINE = (
    "I can't continue with automated assistance on this call. "
    "Please contact the store directly for further help."
)


def _build_handover_tool() -> BaseTool:
    """The `request_handover` control tool: executes, returns a Command routing to the sink."""

    @tool(_HANDOVER_TOOL_NAME)
    def request_handover(
        destination: HandoffDestination,
        reason_code: HandoffReasonCode,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Hand this turn to a higher tier when the caller needs an action you cannot perform
        (changing an address/payment/cart/order, cancelling, refunding, or a multi-step task)."""
        # Sets `handover` + leaves a proper ToolMessage (clean tool_use/tool_result pairing).
        # Routing to the handover node is done by the graph's route_after_tools edge, which
        # reads this `handover` state — so ToolNode's normal batch handling stays intact.
        return Command(
            update={
                "handover": HandoffRequest(
                    destination=destination, reason_code=reason_code, source="model"
                ),
                "messages": [
                    ToolMessage("handover requested", tool_call_id=tool_call_id),
                ],
            }
        )

    return request_handover


def _model_spoke_this_turn(state: ReasoningState) -> bool:
    """True if the model produced spoken text since the last user turn.

    Scans back to the most recent HumanMessage; any AIMessage with non-empty string
    content in that window is the model's own narration (which reaches TTS as streamed
    tokens). Used to decide whether the handover node's canned deferral would double-speak.
    """
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return False
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return True
    return False


def _single_renderable_read(state: ReasoningState) -> tuple[str, dict] | None:
    """The ONE render-decision predicate, shared by `route_after_tools` (to divert) and
    `read_render_node` (to render) so they can never drift (a router that diverts and a node
    that then declines to render would strand the turn).

    Returns `(tool_name, tool_args)` when THIS turn is a pure single renderable read — the
    model's most recent AIMessage carried EXACTLY ONE tool call, it is in `_RENDERABLE_READS`,
    and no handover is pending — else None. A multi-intent turn ("status AND do you have
    socks?") makes the model emit ≥2 tool calls, so the `== 1` guard fails and the turn stays
    on the model narration path. request_handover sets `handover`, so a read+handover turn is
    excluded here and routed to the sink instead.
    """
    if state.handover is not None:
        return None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            return None
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if len(msg.tool_calls) != 1:
                return None
            call = msg.tool_calls[0]
            if call["name"] in _RENDERABLE_READS:
                return call["name"], call["args"]
            return None
    return None


def build_frontline_graph(
    chat_model: BaseChatModel,
    read_only_tools: Sequence[BaseTool],
    *,
    display_name: str,
    tenant_id: str,
    reasoning_model: BaseChatModel,
    store: OrderStore,
    policy: PolicyContext,
    cart_store: CartStore | None = None,
    otp: OtpProvider,
    verification_store: VerificationStore | None = None,
    risk: RiskProvider | None = None,
    profile_store: ProfileStore,
    recent_orders: RecentOrderContext | None = None,
    identity_store: CallerIdentityStore | None = None,
    customers: CustomerDirectory,
    transition_principal: Callable[
        [BoundIdentity, VerificationProof, IntentRequest | None], PrincipalTransition
    ],
    principal_state_will_be_discarded: Callable[[], bool],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the reasoning graph: frontline (routing tier) + cart + support flows.

    `read_only_tools` are already audit-wrapped (tooling.py). `checkpointer` is REQUIRED for
    the interrupt/resume paths (the engine passes one per session); None keeps the per-turn
    stateless mode the text eval uses. The support flow's step-up seams
    (`verification_store`/`risk`) default to fresh fakes when omitted. `otp` is injected:
    production loads the merchant verification fixture; tests declare their fake code.

    `cart_store` defaults to a fresh instance for frontline-only callers, BUT production and
    any cart+view_cart test MUST pass the SAME instance here that they passed to
    `build_voice_tools` — otherwise the frontline's `view_cart` reads a different cart than
    the flow mutates (split-brain). The default exists only so eval/tests that never touch
    the cart don't have to build one.
    """
    # A defaulted VerificationStore shares the SAME provider the dispatch node uses (else
    # dispatch and verify would talk to different fakes).
    verification_store = verification_store or VerificationStore(otp)
    risk = risk or RiskProvider()
    cart_store = cart_store or CartStore()
    # Session recent-order context. Production + any context test MUST pass the
    # SAME instance given to build_voice_tools (the order_status set-site) — the default
    # exists only for callers that never resolve an order reference (split-brain otherwise).
    recent_orders = recent_orders or RecentOrderContext(max_refs=policy.cancel_batch_max)
    # Session authorization store + customer directory (P7). Same split-brain rule: the
    # order_status tool GRANTS into `identity_store` and the render router READS it — pass
    # the ONE instance to both build_voice_tools and here.
    identity_store = identity_store or CallerIdentityStore()
    handover_tool = _build_handover_tool()
    all_tools = [*read_only_tools, handover_tool]
    model_with_tools = chat_model.bind_tools(all_tools)
    system_prompt = compose_system_prompt(display_name, policy)
    read_only_names = {t.name for t in read_only_tools}

    def _last_user_text(state: ReasoningState) -> str:
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def gate_node(state: ReasoningState) -> dict[str, object]:
        """Deterministic pre-generation check. Trip -> set handover, skip the model.

        EXCEPT toward a flow the model just LEFT this turn: the flow model held the full
        conversation when it chose leave (e.g. "complete the purchase" on an ALREADY
        placed order), so a re-trip pointing straight back would either cycle or end in
        the stale canned deferral (live 2026-07-10: the caller heard "I'll pass it along"
        about a purchase that was already done). The frontline model answers instead —
        safe by construction, it holds no write tools.
        """
        hit = gate_check(_last_user_text(state))
        if hit is None:
            return {}
        reason_code, destination = hit
        # Skip a re-trip toward the flow just LEFT this turn (compared by OWNING flow, since
        # the gate says "checkout" but the flow is "cart").
        if state.active_flow == f"left_{_GATE_OWNER[destination]}":
            return {}
        return {
            "handover": HandoffRequest(
                destination=destination, reason_code=reason_code, source="gate"
            )
        }

    def route_after_gate(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"
        scope = status_check(_last_user_text(state))
        if scope is not None:
            if _status_order_ids(state, scope):
                return "forced_status"
            if scope == "list" and state.active_flow != "left_identity":
                return "enumeration_gate"
        if enumeration_check(_last_user_text(state)) and state.active_flow != "left_identity":
            return "enumeration_gate"
        return "model"

    def _tool_hops_this_turn(state: ReasoningState) -> int:
        hops = 0
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                break
            if isinstance(msg, AIMessage) and msg.tool_calls:
                hops += 1
        return hops

    def model_node(state: ReasoningState) -> dict[str, object]:
        # Prompt lives inside the graph (single source; eval == production).
        prompt_text = system_prompt
        if (last_order := recent_orders.snapshot().focused_order_ref) is not None:
            # Per-turn suffix (the static prompt stays composed once): "that order"
            # resolves to the focused order reference — never a cached state claim.
            prompt_text = f"{system_prompt}\n{resolved_order_line(last_order)}"
        tool_hops = _tool_hops_this_turn(state)
        force_final = tool_hops >= policy.max_tool_hops
        if force_final:
            logger.warning(
                "frontline reached the %d-tool-hop turn limit; forcing a final answer",
                policy.max_tool_hops,
            )
            prompt_text += (
                "\nThe read-tool limit for this turn has been reached. Answer now from the "
                "completed tool results already in the conversation. Do not request another "
                "tool or promise a later check."
            )
            model = chat_model
        else:
            model = model_with_tools
        messages = [SystemMessage(prompt_text), *state.messages]
        response = model.invoke(messages)
        if force_final and response.tool_calls:
            logger.error("unbound frontline model returned a tool call at the turn limit")
            content = response.content if isinstance(response.content, str) else ""
            response = AIMessage(content=content.strip() or _TOOL_LIMIT_FALLBACK)
        return {"messages": [response]}

    def route_after_model(state: ReasoningState) -> str:
        last = state.messages[-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return "finalize"
        return "tools"

    def _order_status_line(order_id: str) -> str:
        """Render one live store status with the same per-turn date semantics as L3."""
        return render_order_status_line(
            order_id=order_id,
            status=store.order_status(order_id) or "unknown",
            items=store.order_item_summary(order_id),
            eta=store.order_eta(order_id),
            today=datetime.now().date(),
        )

    def _status_order_ids(state: ReasoningState, scope: str) -> list[str]:
        recent = recent_orders.snapshot()
        referenced = [
            order_id
            for order_id in reversed(recent.order_refs)
            if order_read_allowed(order_id, store=store, identity=identity_store)
        ]
        if scope == "one":
            focused = recent.focused_order_ref
            if focused is not None and order_read_allowed(
                focused, store=store, identity=identity_store
            ):
                return [focused]
            return []

        current = _last_user_text(state)
        bound = identity_store.current()
        if bound is not None and re.search(r"\b(?:all|orders|purchases)\b", current, re.I):
            return [candidate.order_id for candidate in store.owned_orders(bound.customer_ref)]
        if referenced and recent.complete:
            return referenced[:2] if re.search(r"\bboth\b", current, re.I) else referenced
        if bound is not None:
            return [candidate.order_id for candidate in store.owned_orders(bound.customer_ref)]
        # Unbound GUEST (Fix 3): a "list" state-check ("are they shipped?") reads the orders they
        # placed THIS session — the same no-verify session-placed read the enumeration path allows.
        # Still [] for a guest who placed nothing (no account history without a bind).
        return [candidate.order_id for candidate in store.session_placed_orders()]

    def forced_status_node(state: ReasoningState) -> dict[str, object]:
        """Answer a state-verification follow-up from live authorized store reads."""
        scope = status_check(_last_user_text(state))
        assert scope is not None
        order_ids = _status_order_ids(state, scope)
        assert order_ids
        line = " ".join(_order_status_line(order_id) for order_id in order_ids)
        line += f" {warm_close()}"
        write_event(
            {
                "utterance": _last_user_text(state),
                "outcome": "answered",
                "outcome_detail": "forced_status_read",
                "order_count": len(order_ids),
            }
        )
        return {"messages": [AIMessage(line)]}

    def finalize_node(state: ReasoningState) -> dict[str, object]:
        """Telemetry sink for ANSWERED turns — the classifier dataset needs negatives
        (turns that did NOT escalate) just as much as the handover positives."""
        write_event({"utterance": _last_user_text(state), "outcome": "answered"})
        return {}

    def _render_ready(state: ReasoningState) -> tuple[str, dict] | None:
        """`_single_renderable_read` PLUS the P7 authorization gates — the render decision
        the router and the render node share (one function; a drifted second computation
        would let the render path leak an order line around the tool's object-binding gate).

        An order_status the session may NOT read falls back to the model, which narrates
        the tool's ask-for-contact / combined-not-found instruction. The tool GRANTS during
        ToolNode (before route_after_tools runs), so a just-verified read passes here.
        list_orders renders only when an identity is BOUND (the tool returned the real
        list); its unverified branch is the enumeration divert's job, never a render.
        view_cart needs no authorization (the session's own cart).
        """
        renderable = _single_renderable_read(state)
        if renderable is None:
            return None
        name, args = renderable
        if name == "order_status" and not order_read_allowed(
            str(args.get("order_id", "")), store=store, identity=identity_store
        ):
            return None
        if name == "list_orders" and identity_store.current() is None:
            return None
        return renderable

    def _unverified_enumeration(state: ReasoningState) -> bool:
        """True when THIS turn is a single UNVERIFIED list_orders probe — the deterministic
        enumeration divert (live call #13 latency + the F-12.3 class closed structurally).

        The tool's unverified result dictates exactly one action (hand over to the identity
        flow); returning to the model just to relay that costs a full model pass (~1-1.5s
        live) AND re-opens the compliance risk the prompt patches over (call #12: the model
        asked for the email itself). Code sets the handover instead — the model never gets
        the turn back. EXCEPT while `left_identity` (the model deliberately left the flow
        THIS turn): diverting straight back in would cycle; the model answers honestly.
        """
        if state.handover is not None or state.active_flow == "left_identity":
            return False
        if identity_store.current() is not None:
            return False  # bound: the tool returned the real list (render or narrate)
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                return False
            if isinstance(msg, AIMessage) and msg.tool_calls:
                return len(msg.tool_calls) == 1 and msg.tool_calls[0]["name"] == "list_orders"
        return False

    def enumeration_gate_node(state: ReasoningState) -> dict[str, object]:
        """Set the list_orders handover DETERMINISTICALLY (source='gate' — code decided,
        like cross_switch). Nothing is spoken here; handover renders for a bound caller or
        enters Identity for an unbound caller."""
        return {
            "handover": HandoffRequest(
                destination="support", reason_code="list_orders", source="gate"
            )
        }

    def read_render_node(state: ReasoningState) -> dict[str, object]:
        """CODE-author the spoken line for a single renderable read (order_status/view_cart)
        and END — skipping the second model pass (the ~2.5s `tool_to_next_model` cost, live
        calls #9/#10/#11) AND the embellishment class it introduced (a code renderer cannot
        say "on the way" for a processing order — the phrase is derived from the store field).

        Only routed here when `_render_ready` held (single renderable read AND, for
        order_status, the session may read that order), so the read already executed
        (its ToolMessage is in history). The line is re-derived from the SAME store the tool
        read — same turn, single-threaded, no concurrent writer (the frontline holds no write
        tools), so it is deterministically identical to the audited tool result. `today` is
        read per-turn here (never build-time — a stale date regresses the past/future ETA
        framing across midnight, the #9 P6 failure)."""
        renderable = _render_ready(state)
        assert renderable is not None  # router only sends us here when it held
        name, args = renderable
        # A warm close on a resolved answer; NOT on a not-found (a "what else?" after "I
        # couldn't find it" is jarring — the caller needs to give the right number first).
        close = f" {warm_close()}"
        if name == "view_cart":
            line = (
                "Your cart's empty at the moment."
                if cart_store.is_empty()
                else render_cart_line(cart_store.view(), cart_store.cart_total()) + close
            )
        elif name == "list_orders":
            bound = identity_store.current()
            assert bound is not None  # _render_ready only passes list_orders when bound
            line = render_order_list_line(store.owned_orders(bound.customer_ref)) + close
        else:  # order_status
            order_id = str(args.get("order_id", ""))
            if store.order_summary(order_id) is None:
                line = (
                    f"I couldn't find an order with id {order_id} - could you double-check "
                    "the number?"
                )
            else:
                line = _order_status_line(order_id) + close
        # Same answered-turn telemetry as finalize_node (this path ENDs here, bypassing it),
        # tagged so the code-render count is measurable against the model-narration path.
        write_event(
            {
                "utterance": _last_user_text(state),
                "outcome": "answered",
                "outcome_detail": "code_render",
                "tool": name,
            }
        )
        return {"messages": [AIMessage(line)]}

    def handover_node(state: ReasoningState) -> dict[str, object]:
        assert state.handover is not None  # only reached with a handover set
        handover = state.handover
        # A clear enumeration ask is code-routed before the frontline model. Two READ scopes
        # answer here without re-entering Identity; a third case (unbound with nothing placed)
        # falls through to Identity below:
        #   account  — a BOUND session: the full scoped account view (list_orders-tool parity).
        #   session  — an UNBOUND guest who placed orders THIS call (Fix 3): reading back what
        #              they just placed needs no verification (the session-placed read rule), but
        #              the spoken line DISCLOSES it is this-call-only and offers verify-for-more.
        # The two share ONE render/telemetry/cleanup tail (no drift). `order_scope` is a closed
        # slug so the two authorization paths are auditable without PII.
        bound = identity_store.current()
        if handover.destination == "support" and handover.reason_code == "list_orders":
            candidates = None
            order_scope = ""
            if bound is not None:
                candidates = store.owned_orders(bound.customer_ref)
                order_scope = "account"
            elif store.session_placed_orders():
                candidates = store.session_placed_orders()
                order_scope = "session"
            if candidates is not None:
                write_event(
                    {
                        "utterance": _last_user_text(state),
                        "outcome": "answered",
                        "outcome_detail": "code_render",
                        "tool": "list_orders",
                        "order_scope": order_scope,
                    }
                )
                if order_scope == "session":
                    line = render_order_list_line(candidates, scope="session")
                    close = guest_list_close()  # discloses partial scope + verify-for-more
                else:
                    line = render_order_list_line(candidates)
                    close = warm_close()
                return {
                    "messages": [AIMessage(f"{line} {close}")],
                    "active_flow": None,
                    "handover": None,
                    "identity_claim_misses": 0,
                }
            # Unbound with nothing placed this call: nothing to read -> verify (fall through).
        write_event(
            {
                "utterance": _last_user_text(state),
                "outcome": "handover",
                "gate_or_model": handover.source,
                "destination": handover.destination,
                "reason_code": handover.reason_code,
            }
        )
        if handover.destination == "human":
            # The warm-transfer CONTEXT PACKAGE (Group C on-ramp; real SIP transfer =
            # Phase 5, which consumes this across a service boundary — hence the schema
            # version). Closed slugs only, NEVER free text or a caller value (PII): the
            # reason_code names the caller's actual intent (a profile step-up exit writes
            # address_change/contact_change, not a generic slug). One choke point — every
            # human path (escapes, consent "human" verdicts, risk flags, store refusals)
            # already converges on this node.
            write_event(
                {
                    "event": "human_onramp",
                    "schema_version": 1,
                    "tenant": tenant_id,
                    "verification_level": verification_store.current_level(),
                    "active_flow": state.active_flow,
                    "reason_code": handover.reason_code,
                    "source": handover.source,
                }
            )
            return {
                "automation_terminal": True,
                "active_flow": None,
                "handover": None,
                "pending_placement": None,
                "pending_refund": None,
                "pending_cancel": None,
                "pending_return": None,
                "pending_profile_change": None,
                "pending_identity": None,
                "pending_request": None,
                "identity_claim_misses": 0,
                "pending_ack": None,
                "pending_clarification": None,
            }
        # A cart/support handover ENTERS the flow (3b/3c/Group B) instead of speaking a
        # deferral: set the sticky flow marker, clear the handover signal, and let routing
        # carry us into the flow's assemble in this same turn. EXCEPT when that flow was
        # already left this turn ("left_*") — re-entering would cycle; fall through to the
        # deferral. The "checkout" DESTINATION (gate cart_write / model cart_write) enters
        # the CART flow — the single-line checkout flow it replaces is gone.
        if handover.destination == "checkout" and state.active_flow != "left_cart":
            return {"active_flow": "cart", "handover": None}
        if (
            handover.destination == "support"
            and handover.reason_code == "switch_account"
            and state.active_flow != "left_identity"
        ):
            return {
                "active_flow": "identity",
                "handover": None,
                "identity_claim_misses": 0,
                "pending_request": SwitchAccount(),
            }
        # Unbound LIST_ORDERS enters the IDENTITY flow (P7 rung 2: enumeration needs an
        # OTP-bound identity) — checked ABOVE the generic support branch since it shares
        # the support destination. The re-ask counter resets on ENTRY: each fresh engagement
        # gets its one bounded re-ask (a prior engagement's miss must not carry over).
        if (
            handover.destination == "support"
            and handover.reason_code == "list_orders"
            and state.active_flow != "left_identity"
        ):
            return {
                "active_flow": "identity",
                "handover": None,
                "identity_claim_misses": 0,
                "pending_request": ListOrders(scope="account"),
            }
        # Support handles REFUND, CANCEL_ORDER (Group A), and profile changes — address +
        # contact (Group C; contact = the OTP factor itself, stepped-up on the OLD factor).
        # PAYMENT_CHANGE stays deferred (payments = Phase 5): entering would run the
        # reasoning model, fail to propose, bail to left_support, re-trip the gate and
        # double-speak — the honest deferral is correct until its flow exists.
        if (
            handover.destination == "support"
            and handover.reason_code
            in ("refund", "cancel_order", "address_change", "contact_change")
            and state.active_flow != "left_support"
        ):
            return {"active_flow": "support", "handover": None}
        # The canned deferral is a FALLBACK, spoken only if nothing else will be. On the
        # gate path the model never ran → speak it. On the model path the model usually
        # narrates the handover in its own (streamed) tokens; appending the canned line
        # there double-speaks it (observed live 2026-07-08) — so speak it only if the
        # model produced NO spoken text this turn (empty tool-call turn → not silent).
        if handover.source == "gate" or not _model_spoke_this_turn(state):
            return {"messages": [AIMessage(_DEFERRAL[handover.destination])]}
        return {}

    def entry_node(state: ReasoningState) -> dict[str, object]:
        # Fresh-turn hygiene: with a checkpointer, LAST turn's handover signal, the
        # turn-scoped "left_*" marker, pending_ack, and pending_clarification persist in
        # thread state and must not affect THIS turn. (pending_placement needs no such reset:
        # while one exists the graph is paused at confirm and turns arrive as resumes,
        # never through here.)
        update: dict[str, object] = {
            "handover": None,
            "pending_ack": None,
            "pending_clarification": None,
        }
        if state.active_flow in ("left_cart", "left_support", "left_identity"):
            update["active_flow"] = None
        return update

    def automation_terminal_response_node(_state: ReasoningState) -> dict[str, object]:
        write_event({"event": "automation_terminal_response"})
        return {"messages": [AIMessage(_AUTOMATION_TERMINAL_LINE)]}

    def cross_switch_node(state: ReasoningState) -> dict[str, object]:
        """Entry-router escape: while sticky in one gated flow, the caller voiced a
        HIGH-CERTAINTY intent for a DIFFERENT one (the gate tripped cross-flow). The 3b
        escape design's "hard topic switch", now code-owned for gate-certain utterances —
        the model-owned switch (leave_checkout) proved an unreliable sole owner live
        (2026-07-09: a refund request while sticky in checkout was refused once and
        narrated-over once). Abandons the in-flight flow and hands the turn over; NOTHING
        is spoken here — the receiving flow owns the voice."""
        text = _last_user_text(state)
        hit = gate_check(text)
        if hit is not None:
            reason_code, destination = hit
        else:
            assert enumeration_check(text)  # only the two deterministic checks route here
            reason_code, destination = "list_orders", "support"
        write_event(
            {
                "event": "flow_cross_switch",
                "from": state.active_flow,
                "destination": destination,
                "reason_code": reason_code,
            }
        )
        return {
            "active_flow": None,
            "pending_placement": None,
            "pending_refund": None,
            "pending_cancel": None,
            "pending_return": None,
            "pending_profile_change": None,
            "pending_request": None,
            "pending_identity": None,
            "identity_claim_misses": 0,
            "pending_ack": None,
            "pending_clarification": None,
            "handover": HandoffRequest(
                destination=destination, reason_code=reason_code, source="gate"
            ),
        }

    def route_after_entry(state: ReasoningState) -> str:
        # Escape checks BEFORE the sticky flow re-engages (decision: no caller is ever
        # trapped in a gated flow — AGENTS §A9). Deterministic, committed transcript only.
        # ORDER MATTERS: human -> abort -> cross-switch -> assemble. Abort precedes the
        # gate so a mid-flow "cancel that/it" aborts the IN-FLIGHT thing locally; "cancel
        # my order" (not an abort phrasing) falls through to the gate and cross-switches
        # to support's cancel-order path. The cross-switch closes the sticky-flow trap:
        # without it, a gate-certain intent for ANOTHER flow (e.g. "refund me" while stuck
        # in checkout) reaches the checkout model, which cannot serve it.
        if state.automation_terminal:
            return "automation_terminal_response"
        if state.pending_request is not None and state.active_flow is None:
            return "support_continuation"
        text = _last_user_text(state)
        if state.active_flow == "cart":
            if wants_human(text):
                return "cart_escape_human"
            if is_abort(text):
                return "cart_abort"
            hit = gate_check(text)
            # Compare by OWNING flow: the gate says "checkout" for a place-order, which this
            # SAME cart flow serves, so that is NOT a cross-switch (O4/D4 — without the map
            # "checkout now" while sticky in cart would spuriously self-cross-switch).
            if hit is not None and _GATE_OWNER[hit[1]] != "cart":
                return "cross_switch"
            if enumeration_check(text):
                return "cross_switch"
            return "cart_assemble"
        if state.active_flow == "support":
            if wants_human(text):
                return "support_escape_human"
            # Support-scoped abort set: "cancel it" is NOT an abort here — inside support,
            # cancellation is the subject matter (live 2026-07-10: "yeah cancel it" hit
            # this escape and cancelled nothing while sounding like it had). It falls
            # through to the model, which has the conversation context to propose it.
            if is_support_abort(text):
                return "support_abort"
            hit = gate_check(text)
            if hit is not None and _GATE_OWNER[hit[1]] != "support":
                return "cross_switch"
            if enumeration_check(text):
                return "cross_switch"
            return "support_assemble"
        if state.active_flow == "identity":
            if wants_human(text):
                return "identity_escape_human"
            # PLAIN is_abort: cancellation is NOT identity's subject matter (unlike
            # support), so "cancel that" aborts locally — while "cancel my order" is no
            # abort phrasing, falls through to the gate, and cross-switches to support.
            if is_abort(text):
                return "identity_abort"
            # NO gate destination owns identity (the _GATE_OWNER comparison would be
            # vacuous) — ANY gate-certain intent while verifying is a cross-switch.
            if gate_check(text) is not None:
                return "cross_switch"
            return "identity_assemble"
        return "gate"

    def route_after_handover(state: ReasoningState) -> str:
        # Human transitions speak through the one terminal node. A cart/support/identity
        # destination entered its flow (handover cleared, flow set); other destinations end.
        if state.automation_terminal:
            return "automation_terminal_response"
        if state.handover is None:
            if state.active_flow == "identity" and state.pending_request is not None:
                return (
                    "principal_warning"
                    if principal_state_will_be_discarded()
                    else "identity_assemble"
                )
            if state.active_flow == "cart":
                return "cart_assemble"
            if state.active_flow == "support":
                return "support_assemble"
            if state.active_flow == "identity":
                return "identity_assemble"
        return END

    def principal_warning_node(state: ReasoningState) -> dict[str, object]:
        request = state.pending_request
        assert request is not None
        switching = isinstance(request, SwitchAccount)
        prompt = (
            "If the new details belong to a different account, switching will clear this "
            "call's cart and recent order context. Orders already placed will remain placed, "
            "but they won't stay available in this conversation. Would you like to continue?"
            if switching
            else "Verifying an account will clear this call's cart and recent order context. "
            "Orders already placed will remain placed, but they won't stay available in this "
            "conversation. Would you like to continue?"
        )
        answer = interrupt(prompt)
        verdict = classify_consent(str(answer.get("text", "")))
        if answer.get("readback_interrupted") or verdict == "unclear":
            action = "switch accounts" if switching else "verify the account"
            answer = interrupt(f"To {action} and clear this call's context, say yes or no.")
            verdict = classify_consent(str(answer.get("text", "")))
        if verdict == "yes":
            return {"active_flow": "identity"}
        if verdict == "human":
            return {
                "pending_request": None,
                "active_flow": None,
                "handover": HandoffRequest(
                    destination="human",
                    reason_code="switch_account" if switching else "verification_required",
                    source="gate",
                ),
            }
        declined = (
            "Okay, I'll keep you on the current account."
            if switching
            else "Okay, I won't start account verification."
        )
        return {
            "pending_request": None,
            "active_flow": None,
            "messages": [AIMessage(declined)],
        }

    def route_after_principal_warning(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"
        if state.pending_request is not None:
            return "identity_assemble"
        return END

    def route_after_cart_assemble(state: ReasoningState) -> str:
        # "place" | "ack" | "leave" | "clarify" — the flow decides (route_after_assemble is
        # closed over the stores; graph maps its decision to a node).
        decision = cart.route_after_assemble(state)
        return {
            "place": "cart_guardrail",
            "ack": "cart_ack",
            "leave": "gate",  # model left; normal pipeline answers this same turn
            "clarify": END,  # Cart keeps model-authored clarification until Milestone 3d.
        }[decision]

    def route_after_cart_guardrail(state: ReasoningState) -> str:
        return "cart_confirm" if state.pending_placement is not None else END

    def route_after_cart_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_placement is not None:
            return "cart_place"  # explicit committed yes
        return END  # declined / expired (node spoke its line, clear-before-speak)

    cart = build_cart_nodes(
        reasoning_model, store, cart_store, policy, recent_orders, display_name=display_name
    )
    support = build_support_nodes(
        reasoning_model,
        store,
        verification_store,
        otp,
        risk,
        policy,
        profile_store,
        recent_orders,
        identity_store=identity_store,
        display_name=display_name,
    )
    identity = build_identity_nodes(
        reasoning_model,
        store,
        verification_store,
        otp,
        risk,
        customers,
        identity_store,
        policy,
        transition_principal,
        display_name=display_name,
    )

    # --- identity flow routers (the flow owns the store-dependent decisions; the graph
    #     maps them to node names — same stance as the support routers below) ---
    def route_after_identity_assemble(state: ReasoningState) -> str:
        # leave | handover | guardrail | reask | clarify — from the assemble outcome.
        decision = identity.route_after_assemble(state)
        return {
            "leave": "gate",  # model left; normal pipeline answers this same turn
            "handover": "handover",  # terminal no-match -> the silent human path
            "guardrail": "identity_guardrail",
            "reask": "identity_reask",  # the ONE bounded re-ask (own speakable node)

            "clarify": "identity_ask_contact",
        }[decision]

    def route_after_identity_guardrail(state: ReasoningState) -> str:
        # "confirm" (already bound to THIS customer at level) goes straight to apply —
        # identity has no confirm interrupt (nothing irreversible happens at a re-list).
        return {
            "confirm": "identity_apply",
            "stepup": "identity_risk_check",
        }[identity.route_after_guardrail(state)]

    def route_after_identity_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "identity_dispatch"

    def route_after_identity_collect(state: ReasoningState) -> str:
        # The flow's WRAPPED factory decision (binding invariant: a stale cross-family L2
        # "confirm" with no new grant re-collects) mapped to THIS family's nodes.
        return {
            "confirm": "identity_apply",
            "dispatch": "identity_dispatch",
            "handover": "handover",
        }[identity.route_after_collect(state)]

    def route_after_identity_apply(state: ReasoningState) -> str:
        # Typed continuation routes to deterministic resolution; no transcript replay.
        if state.handover is not None:
            return "handover"
        if state.pending_request is not None:
            return "support_continuation"
        return END

    # --- support flow routers (state-only; the level/status-dependent branches live INSIDE
    #     the flow, closed over the store — support.route_after_* ) ---
    def route_after_support_assemble(state: ReasoningState) -> str:
        # refund | cancel | return | profile | resolve | needs_identity | handover | leave |
        # clarify.
        decision = support.route_after_assemble(state)
        if decision == "needs_identity":
            return (
                "principal_warning" if principal_state_will_be_discarded() else "identity_assemble"
            )
        return {
            "refund": "support_guardrail",
            "cancel": "support_cancel_guardrail",
            "return": "support_return_guardrail",
            "profile": "support_profile_guardrail",
            "resolve": "support_resolve",  # a bound caller's "cancel all" -> resolve now
            "handover": "handover",  # deterministic fail-closed path (for example no profile)
            "leave": "gate",  # model left; normal pipeline answers this same turn
            "clarify": END,  # a clarifying question was streamed already
        }[decision]

    def route_after_support_resolve(state: ReasoningState) -> str:
        # confirm (a batch was frozen -> the shared cancel guardrail) | clarify (none
        # cancellable / >2 for 'both' / over-cap: the resolve node spoke + ends).
        return {
            "confirm": "support_cancel_guardrail",
            "clarify": END,
        }[support.route_after_resolve(state)]

    def route_after_support_cancel_guardrail(state: ReasoningState) -> str:
        # confirm (eligible) | handover (risk-flagged) | declined (shipped: guardrail spoke + ends).
        return {
            "confirm": "support_cancel_confirm",
            "handover": "handover",
            "declined": END,
        }[support.route_after_cancel_guardrail(state)]

    def route_after_support_cancel_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_cancel is not None:
            return "support_cancel_void"  # committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_cancel_void(state: ReasoningState) -> str:
        # The batch void processes ONE target per completion: while the pending survives (more
        # targets, progress checkpointed) it loops back to itself; when it clears (last target
        # done, the whole result spoken) it ends. A per-target store refusal is recorded as an
        # outcome and the batch continues — no handover from here.
        return "support_cancel_void" if state.pending_cancel is not None else END

    def route_support_guardrail(state: ReasoningState) -> str:
        # "confirm" (level ok) | "stepup" (-> risk_check) | "declined" (over amount /
        # cancelled / open return: guardrail spoke its line + ends) | "cancel" (remedy
        # steer) | "return" (return-first steer, Group C).
        decision = support.route_after_guardrail(state)
        return {
            "confirm": "support_confirm",
            "stepup": "support_risk_check",
            "declined": END,
            "cancel": "support_cancel_guardrail",
            "return": "support_return_guardrail",
        }[decision]

    def route_after_support_return_guardrail(state: ReasoningState) -> str:
        # confirm (eligible) | cancel (unshipped steer) | declined (guardrail spoke + ends).
        return {
            "confirm": "support_return_confirm",
            "cancel": "support_cancel_guardrail",
            "declined": END,
        }[support.route_after_return_guardrail(state)]

    def route_after_support_return_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_return is not None:
            return "support_return_place"  # committed yes
        return END  # declined/expired (node spoke its line, clear-before-speak)

    def route_after_support_return_place(state: ReasoningState) -> str:
        # place may hand to a human on a store refusal; else it spoke its outcome + ends.
        return "handover" if state.handover is not None else END

    def route_after_support_profile_guardrail(state: ReasoningState) -> str:
        # "confirm" (already L2) | "stepup" (OTP on the OLD factor) — flow-owned decision
        # (the LIVE level lives in the store, which graph.py can't see).
        return {
            "confirm": "support_profile_confirm",
            "stepup": "support_profile_risk_check",
        }[support.route_after_profile_guardrail(state)]

    def route_after_support_profile_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "support_profile_dispatch"

    def route_after_support_profile_collect(state: ReasoningState) -> str:
        # The factory's decision ("confirm" | "dispatch" | "handover") mapped to THIS
        # family's nodes — the confirm target is per-family (R5), never shared.
        decision = support.route_after_profile_collect(state)
        return {
            "confirm": "support_profile_confirm",
            "dispatch": "support_profile_dispatch",
            "handover": "handover",
        }[decision]

    def route_after_support_profile_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the readback
        if state.pending_profile_change is not None:
            return "support_profile_place"  # committed yes
        return END  # declined/expired (node spoke its line, clear-before-speak)

    def route_after_support_profile_place(state: ReasoningState) -> str:
        # place may hand to a human on a lapsed level / store refusal; else it spoke + ends.
        return "handover" if state.handover is not None else END

    def route_after_support_risk(state: ReasoningState) -> str:
        # risk_check sets a handover on a SIM-swap flag; otherwise proceed to dispatch.
        return "handover" if state.handover is not None else "support_dispatch"

    def route_after_support_collect(state: ReasoningState) -> str:
        # "confirm" (raised to L2) | "dispatch" (re-collect) | "handover" (exhausted).
        decision = support.route_after_collect(state)
        return {
            "confirm": "support_confirm",
            "dispatch": "support_dispatch",
            "handover": "handover",
        }[decision]

    def route_after_support_confirm(state: ReasoningState) -> str:
        if state.handover is not None:
            return "handover"  # caller asked for a person at the confirmation
        if state.pending_refund is not None:
            return "support_place"  # explicit committed yes
        return END  # declined (node spoke its line, clear-before-speak)

    def route_after_support_place(state: ReasoningState) -> str:
        # place may hand to a human on a store refusal / lapsed level; else it spoke + ends.
        return "handover" if state.handover is not None else END

    tool_node = ToolNode(all_tools)

    graph = StateGraph(ReasoningState)
    graph.add_node("entry", entry_node)
    graph.add_node("cross_switch", cross_switch_node)
    graph.add_node("gate", gate_node)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("handover", handover_node)
    graph.add_node("automation_terminal_response", automation_terminal_response_node)
    graph.add_node("principal_warning", principal_warning_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("read_render", read_render_node)
    graph.add_node("forced_status", forced_status_node)
    graph.add_node("enumeration_gate", enumeration_gate_node)
    graph.add_node("cart_assemble", cart.assemble)
    graph.add_node("cart_ack", cart.ack)
    graph.add_node("cart_guardrail", cart.guardrail)
    graph.add_node("cart_confirm", cart.confirm)
    graph.add_node("cart_place", cart.place)
    graph.add_node("cart_abort", cart.abort)
    graph.add_node("cart_escape_human", cart.escape_human)
    graph.add_node("support_assemble", support.assemble)
    graph.add_node("support_continuation", support.continuation)
    graph.add_node("support_guardrail", support.guardrail)
    graph.add_node("support_risk_check", support.risk_check)
    graph.add_node("support_dispatch", support.dispatch)
    graph.add_node("support_collect", support.collect)
    graph.add_node("support_confirm", support.confirm)
    graph.add_node("support_place", support.place)
    graph.add_node("support_cancel_guardrail", support.cancel_guardrail)
    graph.add_node("support_cancel_confirm", support.cancel_confirm)
    graph.add_node("support_cancel_void", support.cancel_void)
    graph.add_node("support_resolve", support.resolve)
    graph.add_node("support_return_guardrail", support.return_guardrail)
    graph.add_node("support_return_confirm", support.return_confirm)
    graph.add_node("support_return_place", support.return_place)
    graph.add_node("support_profile_guardrail", support.profile_guardrail)
    graph.add_node("support_profile_risk_check", support.profile_risk_check)
    graph.add_node("support_profile_dispatch", support.profile_dispatch)
    graph.add_node("support_profile_collect", support.profile_collect)
    graph.add_node("support_profile_confirm", support.profile_confirm)
    graph.add_node("support_profile_place", support.profile_place)
    graph.add_node("support_abort", support.abort)
    graph.add_node("support_escape_human", support.escape_human)
    graph.add_node("identity_assemble", identity.assemble)
    graph.add_node("identity_ask_contact", identity.ask_contact)
    graph.add_node("identity_reask", identity.reask)
    graph.add_node("identity_guardrail", identity.guardrail)
    graph.add_node("identity_risk_check", identity.risk_check)
    graph.add_node("identity_dispatch", identity.dispatch)
    graph.add_node("identity_collect", identity.collect)
    graph.add_node("identity_apply", identity.apply)
    graph.add_node("identity_abort", identity.abort)
    graph.add_node("identity_escape_human", identity.escape_human)

    def route_after_tools(state: ReasoningState) -> str:
        # request_handover's Command sets `handover` AND targets the handover node; but the
        # static tools->next edge still evaluates, so guard it: if a handover was set, go
        # there. A pure single renderable read (order_status/view_cart, nothing else) that
        # the session is AUTHORIZED for is rendered in code and ENDs (L3 — skips the second
        # model pass); `_render_ready` carries both the structural predicate and the P7
        # order-read gate, so a declined read returns to the model to narrate the tool's
        # ask-for-contact instruction — the render path can never leak around the gate.
        if state.handover is not None:
            return "handover"
        if _render_ready(state) is not None:
            return "read_render"
        if _unverified_enumeration(state):
            return "enumeration_gate"
        return "model"

    graph.add_edge(START, "entry")
    graph.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "automation_terminal_response": "automation_terminal_response",
            "gate": "gate",
            "cross_switch": "cross_switch",
            "cart_assemble": "cart_assemble",
            "cart_abort": "cart_abort",
            "cart_escape_human": "cart_escape_human",
            "support_assemble": "support_assemble",
            "support_continuation": "support_continuation",
            "support_abort": "support_abort",
            "support_escape_human": "support_escape_human",
            "identity_assemble": "identity_assemble",
            "identity_abort": "identity_abort",
            "identity_escape_human": "identity_escape_human",
        },
    )
    graph.add_edge("cross_switch", "handover")
    graph.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "handover": "handover",
            "model": "model",
            "forced_status": "forced_status",
            "enumeration_gate": "enumeration_gate",
        },
    )
    graph.add_conditional_edges(
        "model", route_after_model, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "handover": "handover",
            "model": "model",
            "read_render": "read_render",
            "enumeration_gate": "enumeration_gate",
        },
    )
    graph.add_edge("enumeration_gate", "handover")
    graph.add_conditional_edges(
        "handover",
        route_after_handover,
        {
            "automation_terminal_response": "automation_terminal_response",
            "cart_assemble": "cart_assemble",
            "support_assemble": "support_assemble",
            "identity_assemble": "identity_assemble",
            "principal_warning": "principal_warning",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "principal_warning",
        route_after_principal_warning,
        {"identity_assemble": "identity_assemble", "handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "cart_assemble",
        route_after_cart_assemble,
        {"gate": "gate", "cart_ack": "cart_ack", "cart_guardrail": "cart_guardrail", END: END},
    )
    graph.add_edge("cart_ack", END)
    graph.add_conditional_edges(
        "cart_guardrail",
        route_after_cart_guardrail,
        {"cart_confirm": "cart_confirm", END: END},
    )
    graph.add_conditional_edges(
        "cart_confirm",
        route_after_cart_confirm,
        {"handover": "handover", "cart_place": "cart_place", END: END},
    )
    graph.add_edge("cart_place", END)
    graph.add_edge("cart_abort", END)
    graph.add_edge("cart_escape_human", "handover")
    graph.add_conditional_edges(
        "support_assemble",
        route_after_support_assemble,
        {
            "gate": "gate",
            "support_guardrail": "support_guardrail",
            "support_cancel_guardrail": "support_cancel_guardrail",
            "support_return_guardrail": "support_return_guardrail",
            "support_profile_guardrail": "support_profile_guardrail",
            "support_resolve": "support_resolve",  # a bound caller's scope resolves now
            "identity_assemble": "identity_assemble",  # unbound scope, no state to discard
            "principal_warning": "principal_warning",
            "handover": "handover",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_continuation",
        route_after_support_assemble,
        {
            "support_guardrail": "support_guardrail",
            "support_cancel_guardrail": "support_cancel_guardrail",
            "support_return_guardrail": "support_return_guardrail",
            "support_profile_guardrail": "support_profile_guardrail",
            "support_resolve": "support_resolve",
            "handover": "handover",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_guardrail",
        route_support_guardrail,
        {
            "support_confirm": "support_confirm",
            "support_risk_check": "support_risk_check",
            "support_cancel_guardrail": "support_cancel_guardrail",
            "support_return_guardrail": "support_return_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_risk_check",
        route_after_support_risk,
        {"support_dispatch": "support_dispatch", "handover": "handover"},
    )
    graph.add_edge("support_dispatch", "support_collect")
    graph.add_conditional_edges(
        "support_collect",
        route_after_support_collect,
        {
            "support_confirm": "support_confirm",
            "support_dispatch": "support_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "support_confirm",
        route_after_support_confirm,
        {"handover": "handover", "support_place": "support_place", END: END},
    )
    graph.add_conditional_edges(
        "support_place",
        route_after_support_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_guardrail",
        route_after_support_cancel_guardrail,
        {"support_cancel_confirm": "support_cancel_confirm", "handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_confirm",
        route_after_support_cancel_confirm,
        {"handover": "handover", "support_cancel_void": "support_cancel_void", END: END},
    )
    graph.add_conditional_edges(
        "support_cancel_void",
        route_after_support_cancel_void,
        {"support_cancel_void": "support_cancel_void", END: END},  # self-loop per target
    )
    graph.add_conditional_edges(
        "support_resolve",
        route_after_support_resolve,
        {"support_cancel_guardrail": "support_cancel_guardrail", END: END},
    )
    graph.add_conditional_edges(
        "support_return_guardrail",
        route_after_support_return_guardrail,
        {
            "support_return_confirm": "support_return_confirm",
            "support_cancel_guardrail": "support_cancel_guardrail",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "support_return_confirm",
        route_after_support_return_confirm,
        {"handover": "handover", "support_return_place": "support_return_place", END: END},
    )
    graph.add_conditional_edges(
        "support_return_place",
        route_after_support_return_place,
        {"handover": "handover", END: END},
    )
    graph.add_conditional_edges(
        "support_profile_guardrail",
        route_after_support_profile_guardrail,
        {
            "support_profile_confirm": "support_profile_confirm",
            "support_profile_risk_check": "support_profile_risk_check",
        },
    )
    graph.add_conditional_edges(
        "support_profile_risk_check",
        route_after_support_profile_risk,
        {"support_profile_dispatch": "support_profile_dispatch", "handover": "handover"},
    )
    graph.add_edge("support_profile_dispatch", "support_profile_collect")
    graph.add_conditional_edges(
        "support_profile_collect",
        route_after_support_profile_collect,
        {
            "support_profile_confirm": "support_profile_confirm",
            "support_profile_dispatch": "support_profile_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "support_profile_confirm",
        route_after_support_profile_confirm,
        {"handover": "handover", "support_profile_place": "support_profile_place", END: END},
    )
    graph.add_conditional_edges(
        "support_profile_place",
        route_after_support_profile_place,
        {"handover": "handover", END: END},
    )
    graph.add_edge("support_abort", END)
    graph.add_edge("support_escape_human", "handover")
    graph.add_conditional_edges(
        "identity_assemble",
        route_after_identity_assemble,
        {
            "gate": "gate",
            "handover": "handover",
            "identity_guardrail": "identity_guardrail",
            "identity_ask_contact": "identity_ask_contact",
            "identity_reask": "identity_reask",
            END: END,
        },
    )
    graph.add_edge("identity_ask_contact", END)
    graph.add_edge("identity_reask", END)
    graph.add_conditional_edges(
        "identity_guardrail",
        route_after_identity_guardrail,
        {
            "identity_apply": "identity_apply",
            "identity_risk_check": "identity_risk_check",
        },
    )
    graph.add_conditional_edges(
        "identity_risk_check",
        route_after_identity_risk,
        {"identity_dispatch": "identity_dispatch", "handover": "handover"},
    )
    graph.add_edge("identity_dispatch", "identity_collect")
    graph.add_conditional_edges(
        "identity_collect",
        route_after_identity_collect,
        {
            "identity_apply": "identity_apply",
            "identity_dispatch": "identity_dispatch",
            "handover": "handover",
        },
    )
    graph.add_conditional_edges(
        "identity_apply",
        route_after_identity_apply,
        {
            "handover": "handover",
            "support_continuation": "support_continuation",
            END: END,
        },
    )
    graph.add_edge("identity_abort", END)
    graph.add_edge("identity_escape_human", "handover")
    graph.add_edge("automation_terminal_response", END)
    graph.add_edge("finalize", END)
    graph.add_edge("forced_status", END)
    graph.add_edge("read_render", END)  # code-authored read line ENDs — skips the 2nd model pass

    compiled = graph.compile(checkpointer=checkpointer)
    # Stashed for tests/introspection + the engine (single source of truth for which
    # node-authored messages are caller-facing — the voice side never hard-codes names).
    compiled.frontline_read_only_tools = read_only_names  # type: ignore[attr-defined]
    compiled.speakable_nodes = (  # type: ignore[attr-defined]
        FRONTLINE_SPEAKABLE_NODES
        | cart.speakable_nodes
        | support.speakable_nodes
        | identity.speakable_nodes
    )
    compiled.model_speech_nodes = MODEL_SPEECH_NODES  # type: ignore[attr-defined]
    overlap = compiled.speakable_nodes & compiled.model_speech_nodes  # type: ignore[attr-defined]
    if overlap:
        raise RuntimeError(f"code/model speech source sets overlap: {sorted(overlap)!r}")
    return compiled
