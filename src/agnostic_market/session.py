"""Application-session lifecycle owner for caller-scoped state.

`aclose_session()` destroys the caller context. `transition_principal()` retires the old
principal, then installs only a newly proven binding and its fresh proof. Their postconditions
differ, but both clear the non-authority ephemeral state through `_clear_ephemeral()`.

State ownership (the durable-vs-ephemeral line this module must respect so it survives the Phase-4
shared-SoR swap): caller-ephemeral state (identity binding + rung-1 grants, verification level +
grants, cart, recent-order context, the guest-order scope, and the reasoning checkpoint)
is cleared here;
merchant/durable-SoR state (fixture/account orders, committed placement/idempotency records,
cancel/refund/return records + status overlays, customer directory, policy) is NEVER cleared on a
caller boundary; call-level security state (abuse counters, OTP dedup, telemetry) is retained for
the call and is not this module's job.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from agnostic_market.agents.engine import ReasoningEngine
from agnostic_market.agents.lifecycle import ExecutionQuiescence
from agnostic_market.agents.telemetry import TelemetryRecorder
from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.identity import BoundIdentity, CallerIdentityStore
from agnostic_market.commerce.orders import GuestOrderScope, RecentOrderContext
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
    tear them down. Built once at application-session assembly, where all these
    instances already converge. The engine attaches after graph construction because identity
    nodes need this lifecycle callback while the graph itself must exist before the engine."""

    verification_store: VerificationStore
    cart_store: CartStore
    recent_orders: RecentOrderContext
    identity_store: CallerIdentityStore
    guest_orders: GuestOrderScope
    telemetry: TelemetryRecorder
    engine: ReasoningEngine | None = None
    _pending_transition: PrincipalTransition | None = field(default=None, init=False)
    _execution_quiescence: ExecutionQuiescence | None = field(default=None, init=False, repr=False)
    _close_quiescence_timeout_seconds: float | None = field(default=None, init=False, repr=False)
    _close_had_pending_interrupt: bool | None = field(default=None, init=False, repr=False)
    _close_started: bool = field(default=False, init=False, repr=False)
    _active_cancellation_takeovers: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _async_close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _takeover_idle_callbacks: list[Callable[[], None]] = field(
        default_factory=list, init=False, repr=False
    )

    def attach_execution_quiescence(
        self,
        tracker: ExecutionQuiescence,
        *,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("close quiescence timeout must be positive")
        with self._close_lock:
            if self._close_started:
                raise RuntimeError("cannot attach graph execution tracking after close starts")
            if self._execution_quiescence is not None and self._execution_quiescence is not tracker:
                raise RuntimeError("caller context already has a different execution tracker")
            self._execution_quiescence = tracker
            self._close_quiescence_timeout_seconds = timeout_seconds

    def attach_engine(self, engine: ReasoningEngine) -> None:
        if self.engine is not None and self.engine is not engine:
            raise RuntimeError("caller context already has a different reasoning engine")
        self.engine = engine

    @contextmanager
    def cancellation_takeover_lease(self) -> Iterator[bool]:
        with self._close_lock:
            acquired = not self._close_started
            if acquired:
                self._active_cancellation_takeovers += 1
        try:
            yield acquired
        finally:
            if acquired:
                callbacks: tuple[Callable[[], None], ...] = ()
                with self._close_lock:
                    self._active_cancellation_takeovers -= 1
                    if not self._active_cancellation_takeovers:
                        callbacks = tuple(self._takeover_idle_callbacks)
                        self._takeover_idle_callbacks.clear()
                for callback in callbacks:
                    callback()

    def _clear_ephemeral(self) -> None:
        """Clear the non-identity, non-verification caller-ephemeral state — the part a close
        AND a principal transition both drop identically (cart, recent-order context, guest/session
        orders). Identity + verification are handled by each PUBLIC op per its postcondition
        (close clears both; a transition installs the new binding/proof instead), so they are
        NOT touched here."""
        self.cart_store.clear()
        self.recent_orders.clear()
        self.guest_orders.clear()

    def has_discardable_state(self) -> bool:
        return bool(
            not self.cart_store.is_empty()
            or self.guest_orders.order_refs
            or self.recent_orders.snapshot().order_refs
        )

    async def transition_principal(
        self,
        new_identity: BoundIdentity,
        fresh_proof: VerificationProof,
        initiating_request: IntentRequest,
    ) -> PrincipalTransition:
        """Retire old caller authority and install exactly one newly proven principal."""
        current = self.identity_store.current()
        if current is not None and current.customer_ref == new_identity.customer_ref:
            raise ValueError("same-customer authentication is not a principal transition")
        transition = PrincipalTransition(
            customer_ref=new_identity.customer_ref,
            masked_contact=new_identity.masked_contact,
            fresh_proof=fresh_proof,
            initiating_request=initiating_request,
        )
        if self._pending_transition is not None:
            raise RuntimeError("a principal transition is already pending context rotation")
        # Publish the fail-closed marker before old authority is retired. If downstream graph
        # execution is cancelled, the engine's finally path still sees and rotates it.
        self._pending_transition = transition
        self._clear_ephemeral()
        self.identity_store.clear()
        await self.verification_store.retain_only(fresh_proof)
        self.identity_store.bind(new_identity)
        self.telemetry.record(
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
            and not self.guest_orders.order_refs
        )
        return PrincipalTransitionInspection(
            outcome="coherent" if coherent else "inconsistent",
            transition=transition,
        )

    async def invalidate_principal_transition(
        self,
        expected_transition_id: str | None = None,
    ) -> bool:
        """Destroy all caller authority after an ambiguous or inconsistent transition."""
        pending = self._pending_transition
        matched = bool(
            expected_transition_id is None
            or (pending is not None and pending.transition_id == expected_transition_id)
        )
        self._clear_ephemeral()
        self.identity_store.clear()
        await self.verification_store.clear()
        self._pending_transition = None
        return matched

    def complete_transition(self, transition_id: str) -> None:
        pending = self._pending_transition
        if pending is None or pending.transition_id != transition_id:
            raise RuntimeError("principal transition completion does not match pending marker")
        self._pending_transition = None

    async def _await_callback(self, register: Callable[[Callable[[], None]], None]) -> None:
        loop = asyncio.get_running_loop()
        idle = loop.create_future()

        def mark_idle() -> None:
            def complete() -> None:
                if not idle.done():
                    idle.set_result(None)

            loop.call_soon_threadsafe(complete)

        register(mark_idle)
        await idle

    async def _await_fully_idle(self, tracker: ExecutionQuiescence | None) -> None:
        if tracker is not None:
            await self._await_callback(tracker.defer_until_fully_idle)

    def _defer_until_takeovers_idle(self, callback: Callable[[], None]) -> None:
        with self._close_lock:
            if self._active_cancellation_takeovers:
                self._takeover_idle_callbacks.append(callback)
                return
        callback()

    async def _await_takeovers_idle(self) -> None:
        await self._await_callback(self._defer_until_takeovers_idle)

    @property
    def close_had_pending_interrupt(self) -> bool:
        return self._close_had_pending_interrupt is True

    async def _acomplete_close(self) -> None:
        self._clear_ephemeral()
        await self.verification_store.clear()
        self.identity_store.clear()
        self._pending_transition = None
        if self.engine is not None:
            await self.engine.adelete_thread()
        self.telemetry.record({"event": "caller_context_closed"})
        with self._close_lock:
            self._closed = True

    async def aclose_session(self) -> None:
        """Stop admission, await quiescence, and delete the checkpoint asynchronously."""
        with self._close_lock:
            if not self._close_started:
                self._close_started = True
                if self._execution_quiescence is not None:
                    self._execution_quiescence.stop_turn_admission()
            tracker = self._execution_quiescence
        async with self._async_close_lock:
            with self._close_lock:
                if self._closed:
                    return
            timeout_seconds = self._close_quiescence_timeout_seconds
            if timeout_seconds is None:
                await self._await_fully_idle(tracker)
                await self._await_takeovers_idle()
            else:
                async with asyncio.timeout(timeout_seconds):
                    await self._await_fully_idle(tracker)
                    await self._await_takeovers_idle()
            if self._close_had_pending_interrupt is None:
                try:
                    self._close_had_pending_interrupt = bool(
                        self.engine is not None
                        and await self.engine.acheckpoint_has_pending_interrupt()
                    )
                except Exception:
                    self._close_had_pending_interrupt = False
                    self.telemetry.record(
                        {
                            "event": "flow_abandonment_observation_failed",
                            "reason": "checkpoint_unavailable",
                        }
                    )
            await self._acomplete_close()
