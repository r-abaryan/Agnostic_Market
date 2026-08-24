"""Deterministic semantic-routing fixtures for graph lifecycle tests."""

from __future__ import annotations

from llm_fakes import TEST_STRUCTURED_OUTPUT_METHOD

from agnostic_market.agents.capabilities import CapabilityRegistry
from agnostic_market.agents.routing import RoutingAttempt, RoutingSession
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import CallerIdentityStore
from agnostic_market.commerce.orders import RecentOrderContext
from agnostic_market.dtos.orchestration import (
    AbortCurrent,
    AnswerQuestion,
    CancelOrders,
    ChangeProfile,
    ListOrders,
    ModifyCart,
    PlaceOrder,
    RefundOrder,
    RequestPerson,
    ReturnOrder,
    RouteDecision,
    RouteResolution,
    RoutingContext,
    SearchCatalog,
    SwitchAccount,
    VerifyIdentity,
    VerifyOrderStatus,
    ViewCart,
    ViewIdentityStatus,
)

_REQUEST_PERSON_FIXTURES = frozenset(
    {"i want a person", "get me a person", "just get me a real person"}
)
_ABORT_CURRENT_FIXTURES = frozenset(
    {
        "stop",
        "never mind",
        "never mind, forget it",
        "forget it",
        "cancel that",
        "cancel it",
        "cancel this",
        "no thanks",
        "don't bother",
    }
)


class StaticRoutingRecognizer:
    """Inject one reviewed decision without claiming semantic quality."""

    def __init__(self, resolution: RouteResolution) -> None:
        self.resolution = resolution
        self.contexts: list[RoutingContext] = []

    async def route(self, context: RoutingContext) -> RoutingAttempt:
        self.contexts.append(context)
        return RoutingAttempt(
            resolution=self.resolution,
            provider="fake",
            model="static-routing-fixture",
            structured_output_method=TEST_STRUCTURED_OUTPUT_METHOD,
            elapsed_ms=0.0,
            input_tokens=None,
            cache_read_tokens=None,
            output_tokens=None,
            route_schema_fingerprint="test-route-schema",
            prompt_fingerprint="test-router-prompt",
            registry_fingerprint="test-registry",
            input_max_chars=2048,
            timeout_seconds=1.0,
            provider_call_outcome="completed",
        )


class ArchitectureRoutingRecognizer(StaticRoutingRecognizer):
    """Route fixture utterances for lifecycle tests, not semantic-quality claims."""

    def __init__(self) -> None:
        super().__init__(RouteDecision.clarify("ambiguous_intent"))

    async def route(self, context: RoutingContext) -> RoutingAttempt:
        text = context.utterance.casefold()
        direct: RouteResolution | None = None
        if text in _ABORT_CURRENT_FIXTURES:
            direct = RouteDecision.direct(AbortCurrent())
        elif text in _REQUEST_PERSON_FIXTURES:
            direct = RouteDecision.direct(RequestPerson())
        elif "cancel" in text and "order" in text:
            direct = RouteDecision.direct(CancelOrders())
        elif "refund" in text:
            direct = RouteDecision.direct(RefundOrder())
        elif "return" in text and "policy" not in text:
            direct = RouteDecision.direct(ReturnOrder())
        elif "address" in text and any(word in text for word in ("change", "update", "new")):
            direct = RouteDecision.direct(ChangeProfile(field="address"))
        elif "contact" in text and "change" in text:
            direct = RouteDecision.direct(ChangeProfile(field="contact"))
        elif ("switch" in text or "another" in text) and "account" in text:
            direct = RouteDecision.direct(SwitchAccount())
        elif "verify" in text and "account" in text:
            direct = RouteDecision.direct(VerifyIdentity())
        elif "verified" in text or "identified" in text:
            direct = RouteDecision.direct(ViewIdentityStatus())
        elif "orders" in text:
            direct = RouteDecision.direct(ListOrders(scope="account"))
        elif "status" in text and "order" in text:
            direct = RouteDecision.direct(VerifyOrderStatus())
        elif "checkout" in text or ("place" in text and "order" in text):
            direct = RouteDecision.direct(PlaceOrder())
        elif "cart" in text and any(word in text for word in ("add", "remove", "set")):
            operation = "remove" if "remove" in text else "add"
            direct = RouteDecision.direct(ModifyCart(operation=operation))
        elif "cart" in text:
            direct = RouteDecision.direct(ViewCart())
        elif any(word in text for word in ("shoe", "jacket", "sock", "catalog")):
            direct = RouteDecision.direct(SearchCatalog(query=context.utterance))

        if context.active_capability is not None and (
            direct is None
            or direct.request is None
            or direct.request.kind == context.active_capability
        ):
            resolution = RouteDecision.continue_current()
        elif direct is not None:
            resolution = direct
        else:
            resolution = RouteDecision.direct(AnswerQuestion(topic="general"))
        self.resolution = resolution
        return await super().route(context)


class LifecycleRoutingRecognizer(StaticRoutingRecognizer):
    """Open one reviewed request, then retain its active invocation on follow-ups."""

    async def route(self, context: RoutingContext) -> RoutingAttempt:
        text = context.utterance.casefold()
        if text in _ABORT_CURRENT_FIXTURES:
            original = self.resolution
            self.resolution = RouteDecision.direct(AbortCurrent())
            try:
                return await super().route(context)
            finally:
                self.resolution = original
        if text in _REQUEST_PERSON_FIXTURES:
            original = self.resolution
            self.resolution = RouteDecision.direct(RequestPerson())
            try:
                return await super().route(context)
            finally:
                self.resolution = original
        if context.active_capability is not None:
            original = self.resolution
            self.resolution = RouteDecision.continue_current()
            try:
                return await super().route(context)
            finally:
                self.resolution = original
        return await super().route(context)


def make_routing_session(
    registry: CapabilityRegistry,
    *,
    identity_store: CallerIdentityStore,
    cart_store: CartStore,
    recent_orders: RecentOrderContext,
    resolution: RouteResolution | None = None,
    continue_active: bool = False,
) -> RoutingSession:
    if resolution is None:
        recognizer = ArchitectureRoutingRecognizer()
    elif continue_active:
        recognizer = LifecycleRoutingRecognizer(resolution)
    else:
        recognizer = StaticRoutingRecognizer(resolution)
    return RoutingSession(
        recognizer,
        identity_store=identity_store,
        cart_store=cart_store,
        recent_orders=recent_orders,
        registry=registry,
    )
