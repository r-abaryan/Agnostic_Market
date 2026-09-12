"""Versioned, redacted evidence for the controlled durable crash matrix."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agnostic_market.config.loader import ConfigError, load_yaml_layer
from agnostic_market.dtos.platform import ConfigIdentifier
from agnostic_market.durability.evidence import ExceptionTypeName, write_immutable_evidence

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

# Schema 1 predates per-case execution surfaces and observed collection counts, so its
# artifacts cannot express whether a case executed. They are refused by identity rather
# than by a generic validation failure.
_SUPERSEDED_METHODOLOGY_SCHEMA_VERSIONS = frozenset({"1"})


class CrashBoundary(StrEnum):
    BACKEND_BASELINE = "backend_baseline"
    ADMISSION = "admission"
    LEASE_AUTHORITY = "lease_authority"
    SESSION_PUBLICATION = "session_publication"
    CHECKPOINT_PUBLICATION = "checkpoint_publication"
    PENDING_CONFIRMATION = "pending_confirmation"
    PRINCIPAL_RETIREMENT = "principal_retirement"
    CLOSE_AND_REAP = "close_and_reap"
    LOCAL_TRANSPORT_RETIREMENT = "local_transport_retirement"


class CrashExecutionSurface(StrEnum):
    """Where a case actually runs, so the backend claim covers only backend cases."""

    POSTGRES = "postgres"
    PROCESS_LOCAL = "process_local"


class CrashMatrixCase(BaseModel):
    model_config = _STRICT

    case_id: ConfigIdentifier
    boundary: CrashBoundary
    execution_surface: CrashExecutionSurface
    nodeid: str = Field(
        min_length=1,
        pattern=r"^tests/[A-Za-z0-9_./-]+\.py::test_[A-Za-z0-9_]+$",
    )
    expected_test_count: int = Field(ge=1)


class DurableCrashMethodology(BaseModel):
    """Frozen controlled test selection; never live-transport evidence."""

    model_config = _STRICT

    schema_version: Literal["2"]
    environment: Literal["controlled"]
    postgres_version: Literal["18.6"]
    per_case_timeout_seconds: float = Field(gt=0)
    cases: tuple[CrashMatrixCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_exact_coverage(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        nodeids = tuple(case.nodeid for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("crash matrix case ids must be unique")
        if len(set(nodeids)) != len(nodeids):
            raise ValueError("crash matrix test nodeids must be unique")
        covered = {case.boundary for case in self.cases}
        if covered != set(CrashBoundary):
            raise ValueError("crash methodology must cover every controlled boundary")
        if not any(case.execution_surface is CrashExecutionSurface.POSTGRES for case in self.cases):
            raise ValueError("crash methodology declaring a postgres version must exercise it")
        return self


class CrashCaseOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class CrashCaseExecution(BaseModel):
    """Observed pytest collection counts proving the case ran rather than was skipped."""

    model_config = _STRICT

    tests: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failures: int = Field(ge=0)
    errors: int = Field(ge=0)


class CrashCaseResult(BaseModel):
    model_config = _STRICT

    case_id: ConfigIdentifier
    boundary: CrashBoundary
    execution_surface: CrashExecutionSurface
    nodeid: str
    outcome: CrashCaseOutcome
    elapsed_seconds: float = Field(ge=0)
    execution: CrashCaseExecution | None = None


class DurableCrashCertificationReport(BaseModel):
    """One complete controlled matrix run with no transcript, payload, or DSN."""

    model_config = _STRICT

    schema_version: Literal["2"] = "2"
    run_at: datetime
    implementation_id: ConfigIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology: DurableCrashMethodology
    results: tuple[CrashCaseResult, ...]
    passed: bool
    live_transport_certified: Literal[False] = False
    activation_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_report_binding(self) -> Self:
        if self.run_at.tzinfo is None:
            raise ValueError("crash certification timestamp must be timezone-aware")
        if self.methodology_fingerprint != crash_methodology_fingerprint(self.methodology):
            raise ValueError("crash report methodology fingerprint is invalid")
        if len(self.results) != len(self.methodology.cases):
            raise ValueError("crash report does not exactly cover its methodology")
        for case, result in zip(self.methodology.cases, self.results, strict=True):
            _validate_case_result_binding(case, result)
        derived_passed = all(result.outcome is CrashCaseOutcome.PASSED for result in self.results)
        if self.passed is not derived_passed:
            raise ValueError("crash report pass result is inconsistent")
        return self


class CrashCertificationOutcome(StrEnum):
    COMPLETED = "completed"
    ABORTED = "aborted"


class CrashCertificationAbort(BaseModel):
    """Redacted evidence for a matrix that could not reach its final case."""

    model_config = _STRICT

    case_id: ConfigIdentifier
    boundary: CrashBoundary
    nodeid: str
    error_type: ExceptionTypeName
    completed_results: tuple[CrashCaseResult, ...]


class DurableCrashCertificationRun(BaseModel):
    """Versioned outcome envelope for completed and aborted certification runs."""

    model_config = _STRICT

    schema_version: Literal["1"] = "1"
    run_at: datetime
    implementation_id: ConfigIdentifier
    methodology_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology: DurableCrashMethodology
    outcome: CrashCertificationOutcome
    report: DurableCrashCertificationReport | None = None
    abort: CrashCertificationAbort | None = None

    @model_validator(mode="after")
    def validate_outcome_binding(self) -> Self:
        if self.run_at.tzinfo is None:
            raise ValueError("crash certification timestamp must be timezone-aware")
        if self.methodology_fingerprint != crash_methodology_fingerprint(self.methodology):
            raise ValueError("crash certification methodology fingerprint is invalid")
        if self.outcome is CrashCertificationOutcome.COMPLETED:
            if self.report is None or self.abort is not None:
                raise ValueError("completed crash certification requires only a report")
            if (
                self.report.run_at != self.run_at
                or self.report.implementation_id != self.implementation_id
                or self.report.methodology != self.methodology
                or self.report.methodology_fingerprint != self.methodology_fingerprint
            ):
                raise ValueError("completed crash report does not match its run envelope")
        else:
            if self.report is not None or self.abort is None:
                raise ValueError("aborted crash certification requires only abort evidence")
            abort = self.abort
            completed_count = len(abort.completed_results)
            if completed_count >= len(self.methodology.cases):
                raise ValueError(
                    "aborted crash certification must identify an unexecuted methodology case"
                )
            for case, result in zip(
                self.methodology.cases[:completed_count],
                abort.completed_results,
                strict=True,
            ):
                _validate_case_result_binding(case, result)
            failed_case = self.methodology.cases[completed_count]
            if (
                abort.case_id != failed_case.case_id
                or abort.boundary is not failed_case.boundary
                or abort.nodeid != failed_case.nodeid
            ):
                raise ValueError(
                    "crash certification abort does not identify the next methodology case"
                )
        return self


class CrashCertificationError(RuntimeError):
    """The controlled crash methodology or evidence artifact is invalid."""


class SupersededCrashEvidenceError(CrashCertificationError):
    """The artifact was produced by a superseded crash-evidence schema."""


type CrashCaseExecutor = Callable[[CrashMatrixCase, float], CrashCaseResult]


def _validate_case_evidence(case: CrashMatrixCase, result: CrashCaseResult) -> None:
    if result.outcome is not CrashCaseOutcome.PASSED:
        return
    if result.execution is None:
        raise ValueError("a passing crash case must record its observed execution")
    if result.execution.tests != case.expected_test_count:
        raise ValueError("a passing crash case must collect its expected test count")
    if result.execution.skipped or result.execution.failures or result.execution.errors:
        raise ValueError("a passing crash case must record no skipped, failed, or errored test")


def _validate_case_result_binding(case: CrashMatrixCase, result: CrashCaseResult) -> None:
    if (
        result.case_id != case.case_id
        or result.boundary is not case.boundary
        or result.execution_surface is not case.execution_surface
        or result.nodeid != case.nodeid
    ):
        raise ValueError("crash case result does not match its methodology case")
    _validate_case_evidence(case, result)


def crash_methodology_fingerprint(methodology: DurableCrashMethodology) -> str:
    canonical = json.dumps(
        methodology.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_crash_methodology(path: Path) -> DurableCrashMethodology:
    try:
        payload = json.dumps(load_yaml_layer(path))
        return DurableCrashMethodology.model_validate_json(payload)
    except (ConfigError, TypeError, ValueError) as exc:
        raise CrashCertificationError("durable crash methodology is missing or invalid") from exc


def _reject_superseded_evidence(raw: str) -> None:
    try:
        document = json.loads(raw)
    except ValueError:
        return
    if not isinstance(document, dict):
        return
    methodology = document.get("methodology")
    if not isinstance(methodology, dict):
        return
    if methodology.get("schema_version") in _SUPERSEDED_METHODOLOGY_SCHEMA_VERSIONS:
        raise SupersededCrashEvidenceError(
            "durable crash evidence uses a superseded methodology schema and cannot be "
            "compared against the registered matrix"
        )


def load_crash_certification_run(path: Path) -> DurableCrashCertificationRun:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CrashCertificationError("durable crash evidence is missing or unreadable") from exc
    _reject_superseded_evidence(raw)
    try:
        return DurableCrashCertificationRun.model_validate_json(raw)
    except ValidationError as exc:
        raise CrashCertificationError("durable crash evidence is invalid") from exc


def build_crash_certification_report(
    methodology: DurableCrashMethodology,
    results: Sequence[CrashCaseResult],
    *,
    run_at: datetime,
    implementation_id: str,
) -> DurableCrashCertificationReport:
    evidence = tuple(results)
    return DurableCrashCertificationReport(
        run_at=run_at,
        implementation_id=implementation_id,
        methodology_fingerprint=crash_methodology_fingerprint(methodology),
        methodology=methodology,
        results=evidence,
        passed=all(result.outcome is CrashCaseOutcome.PASSED for result in evidence),
    )


def run_crash_certification(
    methodology: DurableCrashMethodology,
    *,
    run_at: datetime,
    implementation_id: str,
    execute_case: CrashCaseExecutor,
) -> DurableCrashCertificationRun:
    """Execute every case, retaining completed evidence even when a later case cannot run."""
    fingerprint = crash_methodology_fingerprint(methodology)
    completed: list[CrashCaseResult] = []
    for case in methodology.cases:
        try:
            completed.append(execute_case(case, methodology.per_case_timeout_seconds))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as failure:
            return DurableCrashCertificationRun(
                run_at=run_at,
                implementation_id=implementation_id,
                methodology_fingerprint=fingerprint,
                methodology=methodology,
                outcome=CrashCertificationOutcome.ABORTED,
                abort=CrashCertificationAbort(
                    case_id=case.case_id,
                    boundary=case.boundary,
                    nodeid=case.nodeid,
                    error_type=type(failure).__name__,
                    completed_results=tuple(completed),
                ),
            )
    return DurableCrashCertificationRun(
        run_at=run_at,
        implementation_id=implementation_id,
        methodology_fingerprint=fingerprint,
        methodology=methodology,
        outcome=CrashCertificationOutcome.COMPLETED,
        report=build_crash_certification_report(
            methodology,
            completed,
            run_at=run_at,
            implementation_id=implementation_id,
        ),
    )


def write_crash_certification_run(path: Path, run: DurableCrashCertificationRun) -> None:
    write_immutable_evidence(path, run)


def require_crash_certification_evidence(
    methodology_path: Path,
    report_path: Path,
    *,
    expected_implementation_id: str,
) -> DurableCrashCertificationReport:
    """Require one completed run matching the registered methodology exactly.

    The implementation identifier is operator-declared metadata, not build attestation;
    trusted build identity belongs to release assembly.
    """
    methodology = load_crash_methodology(methodology_path)
    run = load_crash_certification_run(report_path)
    if run.outcome is not CrashCertificationOutcome.COMPLETED or run.report is None:
        raise CrashCertificationError("durable crash evidence did not complete its matrix")
    if run.methodology != methodology:
        raise CrashCertificationError(
            "durable crash evidence does not match the registered methodology"
        )
    if run.methodology_fingerprint != crash_methodology_fingerprint(methodology):
        raise CrashCertificationError("durable crash evidence fingerprint is invalid")
    if run.implementation_id != expected_implementation_id:
        raise CrashCertificationError("durable crash evidence implementation identifier mismatch")
    report = run.report
    try:
        derived = build_crash_certification_report(
            methodology,
            report.results,
            run_at=report.run_at,
            implementation_id=report.implementation_id,
        )
    except (TypeError, ValueError) as exc:
        raise CrashCertificationError("durable crash evidence results are invalid") from exc
    if report != derived:
        raise CrashCertificationError("durable crash evidence derived results are inconsistent")
    if not report.passed:
        raise CrashCertificationError("durable crash matrix did not pass")
    return report
