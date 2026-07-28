"""Shared structural contract for principal-context lifecycle ownership."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from agnostic_market.commerce.identity import BoundIdentity
from agnostic_market.dtos.orchestration import (
    IntentRequest,
    PrincipalTransition,
    PrincipalTransitionInspection,
    VerificationProof,
)


class ExecutionQuiescence(Protocol):
    """Minimal lifecycle-facing view of graph turn admission and full quiescence."""

    def stop_turn_admission(self) -> None: ...

    def defer_until_fully_idle(self, callback: Callable[[], None]) -> None: ...


class PrincipalTransitionLifecycle(Protocol):
    def attach_execution_quiescence(self, tracker: ExecutionQuiescence) -> None: ...

    def cancellation_takeover_lease(self) -> AbstractContextManager[bool]: ...

    def transition_principal(
        self,
        new_identity: BoundIdentity,
        fresh_proof: VerificationProof,
        continuation: IntentRequest | None,
    ) -> PrincipalTransition: ...

    def has_discardable_state(self) -> bool: ...

    def inspect_principal_transition(self) -> PrincipalTransitionInspection: ...

    def invalidate_principal_transition(
        self,
        expected_transition_id: str | None = None,
    ) -> bool: ...

    def complete_transition(self, transition_id: str) -> None: ...
