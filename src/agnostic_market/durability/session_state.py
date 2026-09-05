"""Revisioned mutation boundary for session-owned restorable state."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from agnostic_market.commerce.cart import CartMutationRecord, CartStore
from agnostic_market.commerce.orders import GuestOrderScope, RecentOrderContext
from agnostic_market.dtos.money import UsdAmount
from agnostic_market.dtos.orchestration import CartOperation, OrderContextOperation
from agnostic_market.durability.session_payload import (
    CartMutationSessionOperationResult,
    DurableSessionPayload,
    EmptySessionOperationResult,
    PrincipalRetirementMarker,
    SessionOperationResult,
)
from agnostic_market.durability.session_registry import (
    PostgresSessionRegistry,
    RestoredSessionState,
    SessionLeaseAuthority,
    SessionRegistryDataError,
    SessionRestoreError,
    SessionRestoreReason,
    SessionStatePublication,
    SessionStateWriteError,
    SessionStateWriteReason,
)


@dataclass(frozen=True, slots=True)
class VersionedSessionResult[T]:
    value: T
    session_revision: int


class CheckpointRevisionDisposition(StrEnum):
    CURRENT = "current"
    SESSION_AHEAD = "session_ahead"


def classify_checkpoint_revision(
    checkpoint_revision: object,
    session_revision: int,
) -> CheckpointRevisionDisposition:
    if (
        isinstance(checkpoint_revision, bool)
        or not isinstance(checkpoint_revision, int)
        or checkpoint_revision < 0
    ):
        raise SessionRestoreError(SessionRestoreReason.CHECKPOINT_INVALID)
    if session_revision < 0:
        raise ValueError("session revision must not be negative")
    if checkpoint_revision > session_revision:
        raise SessionRestoreError(SessionRestoreReason.REVISION_MISMATCH)
    if checkpoint_revision < session_revision:
        return CheckpointRevisionDisposition.SESSION_AHEAD
    return CheckpointRevisionDisposition.CURRENT


class SessionStatePersistencePort(Protocol):
    authority: SessionLeaseAuthority

    async def publish(
        self,
        *,
        expected_revision: int,
        operation_id: str,
        request_fingerprint: str,
        payload: DurableSessionPayload,
        operation_result: SessionOperationResult,
    ) -> RestoredSessionState: ...


@dataclass(frozen=True, slots=True)
class BoundPostgresSessionStatePersistence(SessionStatePersistencePort):
    registry: PostgresSessionRegistry
    authority: SessionLeaseAuthority

    async def publish(
        self,
        *,
        expected_revision: int,
        operation_id: str,
        request_fingerprint: str,
        payload: DurableSessionPayload,
        operation_result: SessionOperationResult,
    ) -> RestoredSessionState:
        return await self.registry.publish(
            SessionStatePublication(
                **self.authority.model_dump(),
                expected_revision=expected_revision,
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
                payload=payload,
                operation_result=operation_result,
            )
        )


def _fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_operation_result(value: object) -> SessionOperationResult:
    if value is None:
        return EmptySessionOperationResult()
    if isinstance(value, CartMutationRecord):
        return CartMutationSessionOperationResult(record=value)
    raise TypeError(f"unsupported session operation result: {type(value).__name__}")


def _decode_operation_result(result: SessionOperationResult) -> object:
    if isinstance(result, EmptySessionOperationResult):
        return None
    return result.record


class SessionStateCoordinator:
    """Commit each mutation before replacing the current fenced local projection."""

    def __init__(
        self,
        cart: CartStore,
        recent_orders: RecentOrderContext,
        guest_orders: GuestOrderScope,
        *,
        session_revision: int = 0,
        principal_retirement: PrincipalRetirementMarker | None = None,
        persistence: SessionStatePersistencePort | None = None,
    ) -> None:
        if session_revision < 0:
            raise ValueError("session revision must not be negative")
        self.cart = cart
        self.recent_orders = recent_orders
        self.guest_orders = guest_orders
        self._revision = session_revision
        self._principal_retirement = principal_retirement
        self._persistence = persistence
        self._lock = asyncio.Lock()
        self._local_operations: dict[str, tuple[str, object]] = {}

    @classmethod
    def reconstruct(
        cls,
        payload: DurableSessionPayload,
        *,
        tenant_id: str,
        session_id: str,
        recent_order_max_refs: int,
        session_revision: int,
        persistence: SessionStatePersistencePort | None = None,
    ) -> SessionStateCoordinator:
        return cls(
            CartStore.from_state(payload.cart),
            RecentOrderContext.from_snapshot(
                payload.recent_orders,
                max_refs=recent_order_max_refs,
            ),
            GuestOrderScope.from_refs(
                tenant_id=tenant_id,
                session_id=session_id,
                order_refs=payload.guest_order_refs,
            ),
            session_revision=session_revision,
            principal_retirement=payload.principal_retirement,
            persistence=persistence,
        )

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def principal_retirement(self) -> PrincipalRetirementMarker | None:
        return self._principal_retirement

    def snapshot(self) -> DurableSessionPayload:
        return DurableSessionPayload(
            cart=self.cart.export_state(),
            recent_orders=self.recent_orders.snapshot(),
            guest_order_refs=self.guest_orders.order_refs,
            principal_retirement=self._principal_retirement,
        )

    def _install(self, payload: DurableSessionPayload, revision: int) -> None:
        self.cart.restore_state(payload.cart)
        self.recent_orders.restore_snapshot(payload.recent_orders)
        self.guest_orders.restore_refs(payload.guest_order_refs)
        self._principal_retirement = payload.principal_retirement
        self._revision = revision

    async def discard_local_projection(self) -> None:
        """Destroy this worker's session projection without claiming a durable commit."""
        async with self._lock:
            self._install(DurableSessionPayload(), self._revision)
            self._local_operations.clear()

    def _copy_projection(
        self,
        payload: DurableSessionPayload,
    ) -> tuple[CartStore, RecentOrderContext, GuestOrderScope]:
        return (
            CartStore.from_state(payload.cart),
            RecentOrderContext.from_snapshot(
                payload.recent_orders,
                max_refs=self.recent_orders.max_refs,
            ),
            GuestOrderScope.from_refs(
                tenant_id=self.guest_orders.tenant_id,
                session_id=self.guest_orders.session_id,
                order_refs=payload.guest_order_refs,
            ),
        )

    async def _commit[T](
        self,
        operation_id: str,
        request: Mapping[str, object],
        mutate: Callable[
            [CartStore, RecentOrderContext, GuestOrderScope],
            T,
        ],
        *,
        principal_retirement: PrincipalRetirementMarker | None | object = ...,
    ) -> VersionedSessionResult[T]:
        if not operation_id.strip():
            raise ValueError("session-state operation id must not be blank")
        request_fingerprint = _fingerprint(request)
        async with self._lock:
            prior = self._local_operations.get(operation_id)
            if prior is not None:
                if prior[0] != request_fingerprint:
                    raise SessionStateWriteError(SessionStateWriteReason.OPERATION_CONFLICT)
                return VersionedSessionResult(
                    cast("T", prior[1]),
                    self._revision,
                )
            candidate_cart, candidate_recent, candidate_guest = self._copy_projection(
                self.snapshot()
            )
            value = mutate(candidate_cart, candidate_recent, candidate_guest)
            marker = (
                self._principal_retirement if principal_retirement is ... else principal_retirement
            )
            assert marker is None or isinstance(marker, PrincipalRetirementMarker)
            candidate = DurableSessionPayload(
                cart=candidate_cart.export_state(),
                recent_orders=candidate_recent.snapshot(),
                guest_order_refs=candidate_guest.order_refs,
                principal_retirement=marker,
            )
            candidate_result = _encode_operation_result(value)
            if self._persistence is None:
                next_revision = self._revision + 1
                self._local_operations[operation_id] = (request_fingerprint, value)
                self._install(candidate, next_revision)
                return VersionedSessionResult(value, next_revision)
            committed = await self._persistence.publish(
                expected_revision=self._revision,
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
                payload=candidate,
                operation_result=candidate_result,
            )
            if committed.replayed:
                operation_result = committed.operation_result
                if operation_result is None or type(operation_result) is not type(candidate_result):
                    raise SessionRegistryDataError(
                        "stored session operation result does not match the operation"
                    )
                value = cast("T", _decode_operation_result(operation_result))
            self._install(committed.payload, committed.record.session_revision)
            self._local_operations[operation_id] = (request_fingerprint, value)
            return VersionedSessionResult(value, committed.record.session_revision)

    async def apply_cart_mutation(
        self,
        idempotency_key: str,
        *,
        operation: CartOperation,
        sku: str,
        name: str,
        price_usd: UsdAmount,
        quantity: int | None,
        pre_confirm_quantity: int,
    ) -> VersionedSessionResult[CartMutationRecord]:
        request = {
            "kind": "cart_mutation",
            "operation": operation,
            "sku": sku,
            "name": name,
            "price_usd": str(price_usd),
            "quantity": quantity,
            "pre_confirm_quantity": pre_confirm_quantity,
        }
        return await self._commit(
            idempotency_key,
            request,
            lambda cart, _recent, _guest: cart.apply_confirmed_mutation(
                idempotency_key,
                operation=operation,
                sku=sku,
                name=name,
                price_usd=price_usd,
                quantity=quantity,
                pre_confirm_quantity=pre_confirm_quantity,
            ),
        )

    async def complete_placement(
        self,
        idempotency_key: str,
        order_id: str,
    ) -> VersionedSessionResult[None]:
        normalized_order_id = order_id.strip().upper()

        def mutate(
            cart: CartStore,
            recent: RecentOrderContext,
            guest: GuestOrderScope,
        ) -> None:
            guest.record(normalized_order_id)
            cart.clear()
            recent.record((normalized_order_id,), operation="place")

        return await self._commit(
            idempotency_key,
            {"kind": "placement_projection", "order_id": normalized_order_id},
            mutate,
        )

    async def record_recent_orders(
        self,
        operation_id: str,
        order_refs: Iterable[str],
        *,
        operation: OrderContextOperation,
        focused_order_ref: str | None = None,
        outcomes: Iterable[tuple[str, str]] = (),
    ) -> VersionedSessionResult[None]:
        refs = tuple(order_refs)
        recorded_outcomes = tuple(outcomes)

        def mutate(
            _cart: CartStore,
            recent: RecentOrderContext,
            _guest: GuestOrderScope,
        ) -> None:
            recent.record(
                refs,
                operation=operation,
                focused_order_ref=focused_order_ref,
                outcomes=recorded_outcomes,
            )

        return await self._commit(
            operation_id,
            {
                "kind": "recent_orders",
                "order_refs": refs,
                "operation": operation,
                "focused_order_ref": focused_order_ref,
                "outcomes": recorded_outcomes,
            },
            mutate,
        )

    async def clear_recent_orders(self, operation_id: str) -> VersionedSessionResult[None]:
        return await self._commit(
            operation_id,
            {"kind": "clear_recent_orders"},
            lambda _cart, recent, _guest: recent.clear(),
        )

    async def clear_ephemeral(self, operation_id: str) -> VersionedSessionResult[None]:
        def mutate(
            cart: CartStore,
            recent: RecentOrderContext,
            guest: GuestOrderScope,
        ) -> None:
            cart.clear()
            recent.clear()
            guest.clear()

        return await self._commit(
            operation_id,
            {"kind": "clear_ephemeral"},
            mutate,
        )

    async def begin_principal_retirement(
        self,
        transition_id: str,
    ) -> VersionedSessionResult[None]:
        marker = PrincipalRetirementMarker(transition_id=transition_id)

        def mutate(
            cart: CartStore,
            recent: RecentOrderContext,
            guest: GuestOrderScope,
        ) -> None:
            cart.clear()
            recent.clear()
            guest.clear()

        return await self._commit(
            f"principal-retirement:{transition_id}",
            {"kind": "begin_principal_retirement", "transition_id": transition_id},
            mutate,
            principal_retirement=marker,
        )

    async def finish_principal_retirement(
        self,
        transition_id: str,
    ) -> VersionedSessionResult[None]:
        marker = self._principal_retirement
        if marker is None or marker.transition_id != transition_id:
            raise RuntimeError("principal retirement completion does not match its marker")
        return await self._commit(
            f"principal-retirement-complete:{transition_id}",
            {"kind": "finish_principal_retirement", "transition_id": transition_id},
            lambda _cart, _recent, _guest: None,
            principal_retirement=None,
        )
