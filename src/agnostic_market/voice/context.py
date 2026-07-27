"""Caller-context lifecycle owner (Fix 5) — the ONE place that tears down per-call caller state.

`close_session()` destroys the caller context. `transition_principal()` retires the old
principal, then installs only a newly proven binding and its fresh proof. Their postconditions
differ, but both clear the non-authority ephemeral state through `_clear_ephemeral()`.

State ownership (the durable-vs-ephemeral line this module must respect so it survives the Phase-4
shared-SoR swap): caller-ephemeral state (identity binding + rung-1 grants, verification level +
grants, cart, recent-order context, the placed-order visibility set, and the reasoning checkpoint)
is cleared here;
merchant/durable-SoR state (fixture/account orders, committed placement/idempotency records,
cancel/refund/return records + status overlays, customer directory, policy) is NEVER cleared on a
caller boundary; call-level security state (abuse counters, OTP dedup, telemetry) is retained for
the call and is not this module's job.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.lifecycle import ExecutionQuiescence
from agnostic_market.agents.telemetry import write_event
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import BoundIdentity, CallerIdentityStore
from agnostic_market.commerce.orders import OrderStore, RecentOrderContext
from agnostic_market.commerce.verification import VerificationStore
from agnostic_market.dtos.orchestration import (
    IntentRequest,
    PrincipalTransition,
    PrincipalTransitionInspection,
    VerificationProof,
)


@dataclass
class CallerContext:
    """Owns the per-call caller-ephemeral stores + the reasoning engine, and the operations that
    tear them down. Built once at session assembly (voice/pipeline.py), where all these
    instances already converge. The engine attaches after graph construction because identity
    nodes need this lifecycle callback while the graph itself must exist before the engine."""

    verification_store: VerificationStore
    cart_store: CartStore
    recent_orders: RecentOrderContext
    identity_store: CallerIdentityStore
    order_store: OrderStore
    engine: ReasoningEngine | None = None
    _pending_transition: PrincipalTransition | None = field(default=None, init=False)
    _execution_quiescence: ExecutionQuiescence | None = field(default=None, init=False, repr=False)
    _close_started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def attach_execution_quiescence(self, tracker: ExecutionQuiescence) -> None:
        with self._close_lock:
            if self._close_started:
                raise RuntimeError("cannot attach graph execution tracking after close starts")
            if self._execution_quiescence is not None and self._execution_quiescence is not tracker:
                raise RuntimeError("caller context already has a different execution tracker")
            self._execution_quiescence = tracker

    def attach_engine(self, engine: ReasoningEngine) -> None:
        if self.engine is not None and self.engine is not engine:
            raise RuntimeError("caller context already has a different reasoning engine")
        self.engine = engine

    def _clear_ephemeral(self) -> None:
        """Clear the non-identity, non-verification caller-ephemeral state — the part a close
        AND a principal transition both drop identically (cart, recent-order context, guest/session
        orders). Identity + verification are handled by each PUBLIC op per its postcondition
        (close clears both; a transition installs the new binding/proof instead), so they are
        NOT touched here."""
        self.cart_store.clear()
        self.recent_orders.clear()
        self.order_store.clear_session_placed()

    def has_discardable_state(self) -> bool:
        return bool(
            not self.cart_store.is_empty()
            or self.order_store.session_placed_orders()
            or self.recent_orders.snapshot().order_refs
        )

    def transition_principal(
        self,
        new_identity: BoundIdentity,
        fresh_proof: VerificationProof,
        continuation: IntentRequest | None,
    ) -> PrincipalTransition:
        """Retire old caller authority and install exactly one newly proven principal."""
        current = self.identity_store.current()
        if current is not None and current.customer_ref == new_identity.customer_ref:
            raise ValueError("same-customer authentication is not a principal transition")
        transition = PrincipalTransition(
            customer_ref=new_identity.customer_ref,
            masked_contact=new_identity.masked_contact,
            fresh_proof=fresh_proof,
            continuation=continuation,
        )
        if self._pending_transition is not None:
            raise RuntimeError("a principal transition is already pending context rotation")
        # Publish the fail-closed marker before old authority is retired. If downstream graph
        # execution is cancelled, the engine's finally path still sees and rotates it.
        self._pending_transition = transition
        self._clear_ephemeral()
        self.identity_store.clear()
        self.verification_store.retain_only(fresh_proof)
        self.identity_store.bind(new_identity)
        write_event(
            {
                "event": "principal_transitioned",
                "customer_ref": new_identity.customer_ref,
                "transition_id": transition.transition_id,
            }
        )
        return transition

    def pending_transition(self) -> PrincipalTransition | None:
        return self._pending_transition

    def inspect_principal_transition(self) -> PrincipalTransitionInspection:
        transition = self._pending_transition
        if transition is None:
            return PrincipalTransitionInspection(outcome="none")
        expected_identity = BoundIdentity(
            customer_ref=transition.customer_ref,
            masked_contact=transition.masked_contact,
        )
        coherent = bool(
            self.identity_store.current() == expected_identity
            and not self.identity_store.has_residual_order_authority()
            and self.verification_store.current_level() == transition.fresh_proof.raised_to
            and self.verification_store.grants == [transition.fresh_proof]
            and self.cart_store.is_empty()
            and not self.recent_orders.snapshot().order_refs
            and not self.order_store.session_placed_orders()
        )
        return PrincipalTransitionInspection(
            outcome="coherent" if coherent else "inconsistent",
            transition=transition,
        )

    def invalidate_principal_transition(self, expected_transition_id: str | None = None) -> bool:
        """Destroy all caller authority after an ambiguous or inconsistent transition."""
        pending = self._pending_transition
        matched = bool(
            expected_transition_id is None
            or (pending is not None and pending.transition_id == expected_transition_id)
        )
        self._clear_ephemeral()
        self.identity_store.clear()
        self.verification_store.clear()
        self._pending_transition = None
        return matched

    def complete_transition(self, transition_id: str) -> None:
        pending = self._pending_transition
        if pending is None or pending.transition_id != transition_id:
            raise RuntimeError("principal transition completion does not match pending marker")
        self._pending_transition = None

    def _complete_close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
        self._clear_ephemeral()
        self.verification_store.clear()
        self.identity_store.clear()
        self._pending_transition = None
        if self.engine is not None:
            self.engine.delete_thread()
        write_event({"event": "caller_context_closed"})
        with self._close_lock:
            self._closed = True

    def close_session(self) -> None:
        """Total teardown at session end (Clock B, AGENTS §A10 rule 4): every caller-scoped
        store is cleared and the reasoning thread deleted, so a reattaching session can never
        inherit a stale level, cart, recent-order context, verified identity, or guest order.
        Idempotent: a double-fired close schedules or runs the total teardown exactly once.
        Durable business outcomes (committed cancels/refunds/returns) are NOT cleared (see
        `OrderStore.clear_session_placed`)."""
        with self._close_lock:
            if self._close_started:
                return
            self._close_started = True
            tracker = self._execution_quiescence
        if tracker is None:
            self._complete_close()
            return
        tracker.defer_until_idle(self._complete_close)
