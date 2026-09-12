"""Controlled durable crash-certification evidence contracts."""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agnostic_market.durability.crash_certification import (
    CrashBoundary,
    CrashCaseExecution,
    CrashCaseOutcome,
    CrashCaseResult,
    CrashCertificationError,
    CrashCertificationOutcome,
    CrashExecutionSurface,
    CrashMatrixCase,
    DurableCrashCertificationReport,
    DurableCrashCertificationRun,
    DurableCrashMethodology,
    SupersededCrashEvidenceError,
    build_crash_certification_report,
    crash_methodology_fingerprint,
    load_crash_certification_run,
    load_crash_methodology,
    require_crash_certification_evidence,
    run_crash_certification,
    write_crash_certification_run,
)
from scripts import durable_crash_certification as runner

_ROOT = Path(__file__).parents[1]
_METHODOLOGY_PATH = _ROOT / "config" / "eval" / "durable_crash_matrix.yaml"
_RUN_AT = datetime(2026, 9, 11, tzinfo=UTC)


def _result(
    methodology: DurableCrashMethodology,
    index: int,
    outcome: CrashCaseOutcome = CrashCaseOutcome.PASSED,
) -> CrashCaseResult:
    case = methodology.cases[index]
    execution = (
        CrashCaseExecution(tests=case.expected_test_count, skipped=0, failures=0, errors=0)
        if outcome is CrashCaseOutcome.PASSED
        else None
    )
    return CrashCaseResult(
        case_id=case.case_id,
        boundary=case.boundary,
        execution_surface=case.execution_surface,
        nodeid=case.nodeid,
        outcome=outcome,
        elapsed_seconds=0.1,
        execution=execution,
    )


def _passing_report(methodology: DurableCrashMethodology) -> DurableCrashCertificationReport:
    return build_crash_certification_report(
        methodology,
        tuple(_result(methodology, index) for index in range(len(methodology.cases))),
        run_at=_RUN_AT,
        implementation_id="build-a",
    )


def _completed_run(methodology: DurableCrashMethodology) -> DurableCrashCertificationRun:
    return DurableCrashCertificationRun(
        run_at=_RUN_AT,
        implementation_id="build-a",
        methodology_fingerprint=crash_methodology_fingerprint(methodology),
        methodology=methodology,
        outcome=CrashCertificationOutcome.COMPLETED,
        report=_passing_report(methodology),
    )


def _fabricated_methodology() -> DurableCrashMethodology:
    return DurableCrashMethodology(
        schema_version="2",
        environment="controlled",
        postgres_version="18.6",
        per_case_timeout_seconds=1.0,
        cases=tuple(
            CrashMatrixCase(
                case_id=f"trivial-{index}",
                boundary=boundary,
                execution_surface=CrashExecutionSurface.POSTGRES,
                nodeid=f"tests/test_crash_certification.py::test_trivial_{index}",
                expected_test_count=1,
            )
            for index, boundary in enumerate(CrashBoundary)
        ),
    )


def _declared_test(path: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    module = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _declares_postgres_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any("postgres" in ast.unparse(decorator) for decorator in node.decorator_list)


def _static_parametrized_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    total = 1
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if not (isinstance(target, ast.Attribute) and target.attr == "parametrize"):
            continue
        if len(decorator.args) < 2 or not isinstance(decorator.args[1], ast.List | ast.Tuple):
            return None
        total *= len(decorator.args[1].elts)
    return total


def test_repository_crash_methodology_is_closed_and_declares_its_execution_surfaces() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    assert crash_methodology_fingerprint(methodology) == (
        "efd64c262478a4108800001c950705d4056ae579bbd1720d964fb28849dab121"
    )
    assert {case.boundary for case in methodology.cases} == set(CrashBoundary)
    for case in methodology.cases:
        test_path, test_name = case.nodeid.split("::", maxsplit=1)
        node = _declared_test(test_path, test_name)
        assert node is not None, case.case_id
        declares_backend = case.execution_surface is CrashExecutionSurface.POSTGRES
        assert _declares_postgres_marker(node) is declares_backend, case.case_id
        static_count = _static_parametrized_count(node)
        if static_count is not None:
            assert case.expected_test_count == static_count, case.case_id


def test_crash_methodology_rejects_duplicates_incomplete_coverage_and_unused_backends() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    duplicate = methodology.cases[0].model_copy(update={"case_id": "duplicate-case"})

    with pytest.raises(ValidationError, match="nodeids must be unique"):
        DurableCrashMethodology.model_validate(
            methodology.model_dump() | {"cases": (*methodology.cases, duplicate)}
        )
    with pytest.raises(ValidationError, match="cover every controlled boundary"):
        DurableCrashMethodology.model_validate(
            methodology.model_dump()
            | {
                "cases": tuple(
                    case
                    for case in methodology.cases
                    if case.boundary is not CrashBoundary.ADMISSION
                )
            }
        )
    with pytest.raises(ValidationError, match="must exercise it"):
        DurableCrashMethodology.model_validate(
            methodology.model_dump()
            | {
                "cases": tuple(
                    case.model_dump() | {"execution_surface": CrashExecutionSurface.PROCESS_LOCAL}
                    for case in methodology.cases
                )
            }
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"case_id": "changed-case"},
        {"execution_surface": CrashExecutionSurface.PROCESS_LOCAL},
        {"expected_test_count": 7},
    ],
)
def test_fingerprint_changes_with_the_exact_case_contract(mutation: dict[str, object]) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    changed_case = methodology.cases[0].model_copy(update=mutation)
    changed = methodology.model_copy(update={"cases": (changed_case, *methodology.cases[1:])})

    assert crash_methodology_fingerprint(changed) != crash_methodology_fingerprint(methodology)


def test_report_requires_exact_ordered_coverage_and_derives_its_gate() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    report = _passing_report(methodology)

    assert report.passed
    assert report.live_transport_certified is False
    assert report.activation_authorized is False
    with pytest.raises(ValidationError, match="does not match its methodology case"):
        DurableCrashCertificationReport.model_validate(
            report.model_dump() | {"results": tuple(reversed(report.results))}
        )
    with pytest.raises(ValidationError, match="pass result is inconsistent"):
        DurableCrashCertificationReport.model_validate(report.model_dump() | {"passed": False})


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        (None, "must record its observed execution"),
        (
            {"tests": 0, "skipped": 0, "failures": 0, "errors": 0},
            "must collect its expected test count",
        ),
        (
            {"tests": 1, "skipped": 1, "failures": 0, "errors": 0},
            "no skipped, failed, or errored test",
        ),
    ],
)
def test_a_passing_case_cannot_claim_an_unexecuted_test(
    execution: dict[str, int] | None,
    message: str,
) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    report = _passing_report(methodology)
    first = report.results[0].model_dump() | {"execution": execution}

    with pytest.raises(ValidationError, match=message):
        DurableCrashCertificationReport.model_validate(
            report.model_dump() | {"results": (first, *report.results[1:])}
        )


def test_runner_retains_a_failed_case_without_stopping_the_matrix() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    visited: list[str] = []

    def execute(case: CrashMatrixCase, _timeout: float) -> CrashCaseResult:
        visited.append(case.case_id)
        index = methodology.cases.index(case)
        outcome = CrashCaseOutcome.TIMED_OUT if index == 1 else CrashCaseOutcome.PASSED
        return _result(methodology, index, outcome)

    run = run_crash_certification(
        methodology,
        run_at=_RUN_AT,
        implementation_id="build-a",
        execute_case=execute,
    )

    assert visited == [case.case_id for case in methodology.cases]
    assert run.outcome is CrashCertificationOutcome.COMPLETED
    assert run.report is not None
    assert run.report.passed is False
    assert run.report.results[1].outcome is CrashCaseOutcome.TIMED_OUT


def test_runner_retains_completed_evidence_when_a_later_case_cannot_run() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    def execute(case: CrashMatrixCase, _timeout: float) -> CrashCaseResult:
        index = methodology.cases.index(case)
        if index == 4:
            raise OSError("subprocess launch failed")
        return _result(methodology, index)

    run = run_crash_certification(
        methodology,
        run_at=_RUN_AT,
        implementation_id="build-a",
        execute_case=execute,
    )

    assert run.outcome is CrashCertificationOutcome.ABORTED
    assert run.report is None
    assert run.abort is not None
    assert run.abort.case_id == methodology.cases[4].case_id
    assert run.abort.boundary is methodology.cases[4].boundary
    assert run.abort.error_type == "OSError"
    assert len(run.abort.completed_results) == 4
    assert "subprocess launch failed" not in run.model_dump_json()


def test_abort_must_identify_the_case_after_its_completed_prefix() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    first_result = _result(methodology, 0)
    last_case = methodology.cases[-1]

    with pytest.raises(ValidationError, match="next methodology case"):
        DurableCrashCertificationRun(
            run_at=_RUN_AT,
            implementation_id="build-a",
            methodology_fingerprint=crash_methodology_fingerprint(methodology),
            methodology=methodology,
            outcome=CrashCertificationOutcome.ABORTED,
            abort={
                "case_id": last_case.case_id,
                "boundary": last_case.boundary,
                "nodeid": last_case.nodeid,
                "error_type": "OSError",
                "completed_results": (first_result,),
            },
        )


def test_abort_completed_results_must_be_the_exact_methodology_prefix() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    next_case = methodology.cases[1]

    with pytest.raises(ValidationError, match="does not match its methodology case"):
        DurableCrashCertificationRun(
            run_at=_RUN_AT,
            implementation_id="build-a",
            methodology_fingerprint=crash_methodology_fingerprint(methodology),
            methodology=methodology,
            outcome=CrashCertificationOutcome.ABORTED,
            abort={
                "case_id": next_case.case_id,
                "boundary": next_case.boundary,
                "nodeid": next_case.nodeid,
                "error_type": "OSError",
                "completed_results": (_result(methodology, 1),),
            },
        )


def test_abort_completed_prefix_enforces_passing_case_execution_evidence() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    first_result = _result(methodology, 0).model_copy(update={"execution": None})
    next_case = methodology.cases[1]

    with pytest.raises(ValidationError, match="must record its observed execution"):
        DurableCrashCertificationRun(
            run_at=_RUN_AT,
            implementation_id="build-a",
            methodology_fingerprint=crash_methodology_fingerprint(methodology),
            methodology=methodology,
            outcome=CrashCertificationOutcome.ABORTED,
            abort={
                "case_id": next_case.case_id,
                "boundary": next_case.boundary,
                "nodeid": next_case.nodeid,
                "error_type": "OSError",
                "completed_results": (first_result,),
            },
        )


def test_abort_cannot_follow_a_fully_completed_matrix() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    results = tuple(_result(methodology, index) for index in range(len(methodology.cases)))
    last_case = methodology.cases[-1]

    with pytest.raises(ValidationError, match="unexecuted methodology case"):
        DurableCrashCertificationRun(
            run_at=_RUN_AT,
            implementation_id="build-a",
            methodology_fingerprint=crash_methodology_fingerprint(methodology),
            methodology=methodology,
            outcome=CrashCertificationOutcome.ABORTED,
            abort={
                "case_id": last_case.case_id,
                "boundary": last_case.boundary,
                "nodeid": last_case.nodeid,
                "error_type": "OSError",
                "completed_results": results,
            },
        )


def test_loader_rejects_an_abort_with_unbound_prefix_and_missing_execution(
    tmp_path: Path,
) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    def execute(case: CrashMatrixCase, _timeout: float) -> CrashCaseResult:
        index = methodology.cases.index(case)
        if index == 1:
            raise OSError("subprocess launch failed")
        return _result(methodology, index)

    valid = run_crash_certification(
        methodology,
        run_at=_RUN_AT,
        implementation_id="build-a",
        execute_case=execute,
    )
    payload = json.loads(valid.model_dump_json())
    abort = payload["abort"]
    last_case = methodology.cases[-1]
    abort["case_id"] = last_case.case_id
    abort["boundary"] = last_case.boundary
    abort["nodeid"] = last_case.nodeid
    abort["completed_results"][0]["execution"] = None
    path = tmp_path / "unbound-abort.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CrashCertificationError, match="evidence is invalid"):
        load_crash_certification_run(path)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_runner_propagates_interrupts_instead_of_recording_evidence(
    interrupt: type[BaseException],
) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    def execute(case: CrashMatrixCase, _timeout: float) -> CrashCaseResult:
        raise interrupt

    with pytest.raises(interrupt):
        run_crash_certification(
            methodology,
            run_at=_RUN_AT,
            implementation_id="build-a",
            execute_case=execute,
        )


def test_run_envelope_rejects_mixed_or_mismatched_outcomes() -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    run = _completed_run(methodology)

    with pytest.raises(ValidationError, match="requires only a report"):
        DurableCrashCertificationRun.model_validate(run.model_dump() | {"report": None})
    with pytest.raises(ValidationError, match="does not match its run envelope"):
        DurableCrashCertificationRun.model_validate(
            run.model_dump() | {"implementation_id": "build-b"}
        )
    with pytest.raises(ValidationError, match="requires only abort evidence"):
        DurableCrashCertificationRun.model_validate(
            run.model_dump() | {"outcome": CrashCertificationOutcome.ABORTED}
        )


def test_evidence_consumer_binds_the_run_to_the_registered_methodology(tmp_path: Path) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    path = tmp_path / "crash.json"
    write_crash_certification_run(path, _completed_run(methodology))

    report = require_crash_certification_evidence(
        _METHODOLOGY_PATH,
        path,
        expected_implementation_id="build-a",
    )

    assert report.passed
    with pytest.raises(CrashCertificationError, match="implementation identifier mismatch"):
        require_crash_certification_evidence(
            _METHODOLOGY_PATH,
            path,
            expected_implementation_id="build-b",
        )


def test_evidence_consumer_rejects_a_fabricated_methodology(tmp_path: Path) -> None:
    fabricated = _fabricated_methodology()
    path = tmp_path / "fabricated.json"
    write_crash_certification_run(path, _completed_run(fabricated))

    assert load_crash_certification_run(path).report is not None
    with pytest.raises(CrashCertificationError, match="does not match the registered methodology"):
        require_crash_certification_evidence(
            _METHODOLOGY_PATH,
            path,
            expected_implementation_id="build-a",
        )


def test_evidence_consumer_rejects_aborted_and_failed_runs(tmp_path: Path) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    aborted = DurableCrashCertificationRun(
        run_at=_RUN_AT,
        implementation_id="build-a",
        methodology_fingerprint=crash_methodology_fingerprint(methodology),
        methodology=methodology,
        outcome=CrashCertificationOutcome.ABORTED,
        abort={
            "case_id": methodology.cases[0].case_id,
            "boundary": methodology.cases[0].boundary,
            "nodeid": methodology.cases[0].nodeid,
            "error_type": "OSError",
            "completed_results": (),
        },
    )
    aborted_path = tmp_path / "aborted.json"
    write_crash_certification_run(aborted_path, aborted)
    with pytest.raises(CrashCertificationError, match="did not complete its matrix"):
        require_crash_certification_evidence(
            _METHODOLOGY_PATH, aborted_path, expected_implementation_id="build-a"
        )

    failed_results = (
        _result(methodology, 0, CrashCaseOutcome.FAILED),
        *(_result(methodology, index) for index in range(1, len(methodology.cases))),
    )
    failed = DurableCrashCertificationRun(
        run_at=_RUN_AT,
        implementation_id="build-a",
        methodology_fingerprint=crash_methodology_fingerprint(methodology),
        methodology=methodology,
        outcome=CrashCertificationOutcome.COMPLETED,
        report=build_crash_certification_report(
            methodology, failed_results, run_at=_RUN_AT, implementation_id="build-a"
        ),
    )
    failed_path = tmp_path / "failed.json"
    write_crash_certification_run(failed_path, failed)
    with pytest.raises(CrashCertificationError, match="did not pass"):
        require_crash_certification_evidence(
            _METHODOLOGY_PATH, failed_path, expected_implementation_id="build-a"
        )


def test_superseded_schema_artifacts_are_refused_by_identity(tmp_path: Path) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    superseded = json.loads(_completed_run(methodology).model_dump_json())
    superseded["methodology"]["schema_version"] = "1"
    path = tmp_path / "superseded.json"
    path.write_text(json.dumps(superseded), encoding="utf-8")

    with pytest.raises(SupersededCrashEvidenceError, match="superseded methodology schema"):
        load_crash_certification_run(path)


def test_crash_evidence_writer_is_atomic_and_does_not_overwrite(tmp_path: Path) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    run = _completed_run(methodology)
    path = tmp_path / "crash-report.json"

    write_crash_certification_run(path, run)

    assert load_crash_certification_run(path) == run
    with pytest.raises(FileExistsError):
        write_crash_certification_run(path, run)


def test_concurrent_evidence_writers_publish_one_complete_artifact(tmp_path: Path) -> None:
    methodology = load_crash_methodology(_METHODOLOGY_PATH)
    run = _completed_run(methodology)
    path = tmp_path / "concurrent-crash-report.json"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(write_crash_certification_run, path, run) for _ in range(2))
    failures = tuple(future.exception() for future in futures if future.exception() is not None)

    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert load_crash_certification_run(path) == run


def _write_junit(path: Path, *, tests: int, skipped: int, failures: int, errors: int) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        f'tests="{tests}" skipped="{skipped}" failures="{failures}" errors="{errors}">'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("tests", "skipped", "failures", "errors", "return_code", "expected"),
    [
        (1, 0, 0, 0, 0, CrashCaseOutcome.PASSED),
        (1, 1, 0, 0, 0, CrashCaseOutcome.FAILED),
        (1, 0, 1, 0, 1, CrashCaseOutcome.FAILED),
        (1, 0, 0, 1, 4, CrashCaseOutcome.FAILED),
        (0, 0, 0, 0, 5, CrashCaseOutcome.FAILED),
        (2, 0, 0, 0, 0, CrashCaseOutcome.FAILED),
    ],
)
def test_junit_evidence_decides_the_case_outcome(
    tmp_path: Path,
    tests: int,
    skipped: int,
    failures: int,
    errors: int,
    return_code: int,
    expected: CrashCaseOutcome,
) -> None:
    case = load_crash_methodology(_METHODOLOGY_PATH).cases[0]
    path = tmp_path / "report.xml"
    _write_junit(path, tests=tests, skipped=skipped, failures=failures, errors=errors)

    execution = runner._read_execution(path)

    assert execution is not None
    assert runner._case_outcome(case, return_code, execution) is expected


def test_a_skipped_or_xfailed_report_is_never_a_passing_case(tmp_path: Path) -> None:
    """pytest exits zero for skip and xfail, so the XML is the only usable signal."""
    case = load_crash_methodology(_METHODOLOGY_PATH).cases[0]
    path = tmp_path / "skipped.xml"
    path.write_text(
        '<testsuite name="pytest" tests="1" skipped="1" failures="0" errors="0">'
        '<testcase name="t"><skipped type="pytest.xfail"/></testcase></testsuite>',
        encoding="utf-8",
    )

    execution = runner._read_execution(path)

    assert execution == CrashCaseExecution(tests=1, skipped=1, failures=0, errors=0)
    assert runner._case_outcome(case, 0, execution) is CrashCaseOutcome.FAILED


@pytest.mark.parametrize(
    "content",
    [None, "", "not xml", "<testsuites></testsuites>", '<testsuite tests="x"/>'],
)
def test_missing_or_unusable_junit_evidence_is_an_infrastructure_error(
    tmp_path: Path,
    content: str | None,
) -> None:
    case = load_crash_methodology(_METHODOLOGY_PATH).cases[0]
    path = tmp_path / "unusable.xml"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    execution = runner._read_execution(path)

    assert execution is None
    assert runner._case_outcome(case, 0, execution) is CrashCaseOutcome.INFRASTRUCTURE_ERROR


def test_abort_evidence_records_a_private_exception_class_name() -> None:
    """A failure the envelope exists to record must not be rejected by its own field type."""
    methodology = load_crash_methodology(_METHODOLOGY_PATH)

    class _InternalError(Exception):
        pass

    def execute(case: CrashMatrixCase, _timeout: float) -> CrashCaseResult:
        raise _InternalError

    run = run_crash_certification(
        methodology,
        run_at=_RUN_AT,
        implementation_id="build-a",
        execute_case=execute,
    )

    assert run.outcome is CrashCertificationOutcome.ABORTED
    assert run.abort is not None
    assert run.abort.error_type == "_InternalError"
