"""Revision and reconstruction contracts for session-owned state."""

from __future__ import annotations

import pytest

from agnostic_market.commerce.cart import CartStore
from agnostic_market.commerce.orders import GuestOrderScope, RecentOrderContext
from agnostic_market.durability.session_payload import (
    DurableSessionPayload,
    PrincipalRetirementMarker,
    SessionOperationResult,
)
from agnostic_market.durability.session_registry import (
    RestoredSessionState,
    SessionLeaseAuthority,
    SessionRestoreError,
    SessionRestoreReason,
    SessionStateWriteError,
    SessionStateWriteReason,
)
from agnostic_market.durability.session_state import (
    CheckpointRevisionDisposition,
    SessionStateCoordinator,
    classify_checkpoint_revision,
)


class _RejectingPersistence:
    def __init__(self) -> None:
        self.authority: SessionLeaseAuthority

    async def publish(
        self,
        *,
        expected_revision: int,
        operation_id: str,
        request_fingerprint: str,
        payload: DurableSessionPayload,
        operation_result: SessionOperationResult,
    ) -> RestoredSessionState:
        raise SessionStateWriteError(SessionStateWriteReason.STALE_REVISION)


def _coordinator() -> SessionStateCoordinator:
    return SessionStateCoordinator(
        CartStore(),
        RecentOrderContext(max_refs=3),
        GuestOrderScope(tenant_id="acme_store", session_id="session-1"),
    )


async def test_cart_mutation_commits_one_revision_and_replays_without_advancing() -> None:
    state = _coordinator()

    first = await state.apply_cart_mutation(
        "mutation-1",
        operation="add",
        sku="SKU-1",
        name="Trail Jacket",
        price_usd="79.00",
        quantity=1,
        pre_confirm_quantity=0,
    )
    replayed = await state.apply_cart_mutation(
        "mutation-1",
        operation="add",
        sku="SKU-1",
        name="Trail Jacket",
        price_usd="79.00",
        quantity=1,
        pre_confirm_quantity=0,
    )

    assert first.session_revision == replayed.session_revision == state.revision == 1
    assert first.value == replayed.value
    assert state.cart.view()[0].quantity == 1


async def test_operation_identity_conflict_does_not_change_session_state() -> None:
    state = _coordinator()
    await state.apply_cart_mutation(
        "mutation-1",
        operation="add",
        sku="SKU-1",
        name="Trail Jacket",
        price_usd="79.00",
        quantity=1,
        pre_confirm_quantity=0,
    )

    with pytest.raises(SessionStateWriteError) as rejected:
        await state.apply_cart_mutation(
            "mutation-1",
            operation="add",
            sku="SKU-1",
            name="Trail Jacket",
            price_usd="79.00",
            quantity=2,
            pre_confirm_quantity=0,
        )

    assert rejected.value.reason is SessionStateWriteReason.OPERATION_CONFLICT
    assert state.revision == 1
    assert state.cart.view()[0].quantity == 1


async def test_failed_publication_does_not_install_the_candidate_projection() -> None:
    persistence = _RejectingPersistence()
    state = SessionStateCoordinator(
        CartStore(),
        RecentOrderContext(max_refs=3),
        GuestOrderScope(tenant_id="acme_store", session_id="session-1"),
        session_revision=4,
        persistence=persistence,
    )

    with pytest.raises(SessionStateWriteError) as rejected:
        await state.apply_cart_mutation(
            "mutation-1",
            operation="add",
            sku="SKU-1",
            name="Trail Jacket",
            price_usd="79.00",
            quantity=1,
            pre_confirm_quantity=0,
        )

    assert rejected.value.reason is SessionStateWriteReason.STALE_REVISION
    assert state.revision == 4
    assert state.cart.is_empty()


async def test_placement_projection_updates_all_session_stores_in_one_revision() -> None:
    state = _coordinator()
    state.cart.add_item(sku="SKU-1", name="Trail Jacket", price_usd="79.00", quantity=1)

    committed = await state.complete_placement("placement-1", "ord-9001")

    assert committed.session_revision == state.revision == 1
    assert state.cart.is_empty()
    assert state.guest_orders.order_refs == ("ORD-9001",)
    assert state.recent_orders.snapshot().order_refs == ("ORD-9001",)


async def test_old_placement_replay_cannot_clear_a_later_cart() -> None:
    state = _coordinator()
    state.cart.add_item(sku="SKU-1", name="Trail Jacket", price_usd="79.00", quantity=1)
    await state.complete_placement("placement-1", "ORD-9001")
    await state.apply_cart_mutation(
        "mutation-2",
        operation="add",
        sku="SKU-2",
        name="Day Pack",
        price_usd="49.00",
        quantity=1,
        pre_confirm_quantity=0,
    )

    replayed = await state.complete_placement("placement-1", "ORD-9001")

    assert replayed.session_revision == 2
    assert state.revision == 2
    assert [line.sku for line in state.cart.view()] == ["SKU-2"]


def test_reconstruction_preserves_state_but_not_unrepresented_authority() -> None:
    source = _coordinator()
    source.cart.add_item(sku="SKU-1", name="Trail Jacket", price_usd="79.00", quantity=1)
    payload = source.snapshot().model_copy(
        update={"principal_retirement": PrincipalRetirementMarker(transition_id="transition-1")}
    )

    restored = SessionStateCoordinator.reconstruct(
        DurableSessionPayload.model_validate(payload),
        tenant_id="acme_store",
        session_id="session-1",
        recent_order_max_refs=3,
        session_revision=7,
    )

    assert restored.revision == 7
    assert restored.cart.view() == source.cart.view()
    assert restored.principal_retirement == PrincipalRetirementMarker(transition_id="transition-1")


def test_checkpoint_revision_join_rejects_ahead_and_classifies_session_ahead() -> None:
    assert classify_checkpoint_revision(4, 4) is CheckpointRevisionDisposition.CURRENT
    assert classify_checkpoint_revision(3, 4) is CheckpointRevisionDisposition.SESSION_AHEAD
    with pytest.raises(SessionRestoreError) as ahead:
        classify_checkpoint_revision(5, 4)
    with pytest.raises(SessionRestoreError) as malformed:
        classify_checkpoint_revision("4", 4)

    assert ahead.value.reason is SessionRestoreReason.REVISION_MISMATCH
    assert malformed.value.reason is SessionRestoreReason.CHECKPOINT_INVALID
