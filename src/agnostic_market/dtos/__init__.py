"""Shared Pydantic v2 DTOs — the single source of truth for cross-module shapes.

Reasoning code, voice layer, and the future ingestion repo import these; they never
redefine shapes (BUILD_PLAN "Shared DTOs/contracts"). Phase 0 builds only the config
DTOs (config.py) and the minimal tenancy fields (state.py) actually consumed now; the
rest are added in their consuming phase.

Note: there is deliberately NO config/schemas.py — models live here only (see the
Phase-0 plan; BUILD_PLAN's repo-tree listing of config/schemas.py is a documented
duplication to drop).
"""
