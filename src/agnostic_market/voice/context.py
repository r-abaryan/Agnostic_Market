"""Caller-context lifecycle owner (Fix 5) — the ONE place that tears down per-call caller state.

Two futures live here. Milestone A ships `close_session()` only: total teardown at
`AgentSession.close`, consolidating the four inline store clears the thread reaper used to do
by hand into a single, tested operation. Milestone D adds `transition_principal()` — retire the
OLD principal's context then install a NEWLY proven binding + its fresh proof — which is a
DIFFERENT postcondition (close destroys everything; a transition keeps the new principal). The
two share the private `_clear_ephemeral()` helper for the non-identity/non-verification ephemeral
state (cart, pointer, guest/session orders); each public op owns identity + verification per its
own postcondition. `transition_principal()` is deliberately absent until D gives it a real caller
(no unused public surface).

State ownership (the durable-vs-ephemeral line this module must respect so it survives the Phase-4
shared-SoR swap): caller-ephemeral state (identity binding + rung-1 grants, verification level +
grants, cart, pointer, the placed-order visibility set, the reasoning checkpoint) is cleared here;
merchant/durable-SoR state (fixture/account orders, committed placement/idempotency records,
cancel/refund/return records + status overlays, customer directory, policy) is NEVER cleared on a
caller boundary; call-level security state (abuse counters, OTP dedup, telemetry) is retained for
the call and is not this module's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import CallerIdentityStore
from agnostic_market.commerce.orders import LastOrderPointer, OrderStore
from agnostic_market.commerce.verification import VerificationStore


@dataclass(frozen=True)
class CallerContext:
    """Owns the per-call caller-ephemeral stores + the reasoning engine, and the operations that
    tear them down. Built once at session assembly (voice/pipeline.py), where all these
    instances already converge. Frozen: it holds references, never mutable lifecycle flags."""

    engine: ReasoningEngine
    verification_store: VerificationStore
    cart_store: CartStore
    pointer: LastOrderPointer
    identity_store: CallerIdentityStore
    order_store: OrderStore

    def _clear_ephemeral(self) -> None:
        """Clear the non-identity, non-verification caller-ephemeral state — the part a close
        AND a principal transition both drop identically (cart, session pointer, guest/session
        orders). Identity + verification are handled by each PUBLIC op per its postcondition
        (close clears both; a transition installs the new binding/proof instead), so they are
        NOT touched here."""
        self.cart_store.clear()
        self.pointer.clear()
        self.order_store.clear_session_placed()

    def close_session(self) -> None:
        """Total teardown at session end (Clock B, AGENTS §A10 rule 4): every caller-scoped
        store is cleared and the reasoning thread deleted, so a reattaching session can never
        inherit a stale level, cart, pointer, verified identity, or guest order. Idempotent —
        safe to call more than once (a double-fired close, or a close racing a future
        transition). Durable business outcomes (committed cancels/refunds/returns) are NOT
        cleared (see `OrderStore.clear_session_placed`)."""
        self._clear_ephemeral()
        self.verification_store.clear()
        self.identity_store.clear()
        self.engine.delete_thread()
        write_event({"event": "caller_context_closed"})
