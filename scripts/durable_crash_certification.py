"""Run the pre-registered controlled durable crash matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from dotenv import load_dotenv

from agnostic_market.durability.crash_certification import (
    CrashCaseExecution,
    CrashCaseOutcome,
    CrashCaseResult,
    CrashCertificationError,
    CrashCertificationOutcome,
    CrashMatrixCase,
    load_crash_certification_run,
    load_crash_methodology,
    require_crash_certification_evidence,
    run_crash_certification,
    write_crash_certification_run,
)

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_METHODOLOGY = _ROOT / "config" / "eval" / "durable_crash_matrix.yaml"
_POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methodology",
        type=Path,
        default=_DEFAULT_METHODOLOGY,
        help="pre-registered controlled crash methodology YAML",
    )
    parser.add_argument("--report", type=Path, required=True, help="new immutable result path")
    parser.add_argument(
        "--implementation-id",
        required=True,
        help="operator-declared commit or build identifier under test",
    )
    arguments = parser.parse_args()
    # Cases run with cwd set to the repository root, so bind operator paths before then.
    arguments.methodology = arguments.methodology.resolve()
    arguments.report = arguments.report.resolve()
    return arguments


def _read_execution(path: Path) -> CrashCaseExecution | None:
    """Read observed collection counts, treating any unusable report as no evidence."""
    try:
        root = ElementTree.parse(path).getroot()  # noqa: S314 - report written by this runner
    except (OSError, ElementTree.ParseError):
        return None
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return None
    try:
        return CrashCaseExecution(
            tests=int(suite.attrib["tests"]),
            skipped=int(suite.attrib["skipped"]),
            failures=int(suite.attrib["failures"]),
            errors=int(suite.attrib["errors"]),
        )
    except (KeyError, ValueError):
        return None


def _case_outcome(
    case: CrashMatrixCase,
    return_code: int,
    execution: CrashCaseExecution | None,
) -> CrashCaseOutcome:
    if execution is None:
        return CrashCaseOutcome.INFRASTRUCTURE_ERROR
    executed_exactly = (
        return_code == 0
        and execution.tests == case.expected_test_count
        and execution.skipped == 0
        and execution.failures == 0
        and execution.errors == 0
    )
    return CrashCaseOutcome.PASSED if executed_exactly else CrashCaseOutcome.FAILED


def _execute_case(case: CrashMatrixCase, timeout_seconds: float) -> CrashCaseResult:
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    # A transient lock on the report file must not discard a completed matrix case.
    with tempfile.TemporaryDirectory(
        prefix="durable-crash-", ignore_cleanup_errors=True
    ) as directory:
        evidence_path = Path(directory) / f"{case.case_id}.xml"
        started = time.perf_counter()
        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter; nodeid is schema-restricted
                (
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    case.nodeid,
                    f"--junit-xml={evidence_path}",
                ),
                cwd=_ROOT,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _reported(case, CrashCaseOutcome.TIMED_OUT, time.perf_counter() - started)
        elapsed = time.perf_counter() - started
        execution = _read_execution(evidence_path)
    return _reported(case, _case_outcome(case, completed.returncode, execution), elapsed, execution)


def _reported(
    case: CrashMatrixCase,
    outcome: CrashCaseOutcome,
    elapsed_seconds: float,
    execution: CrashCaseExecution | None = None,
) -> CrashCaseResult:
    observed = "" if execution is None else f" tests={execution.tests} skipped={execution.skipped}"
    print(f"[{outcome.value}] {case.case_id} ({elapsed_seconds:.3f}s){observed}")
    if outcome is not CrashCaseOutcome.PASSED:
        print(
            f"Crash certification case did not execute cleanly: {case.case_id}; "
            f"rerun its exact nodeid for diagnostics",
            file=sys.stderr,
        )
    return _case_result(case, outcome, elapsed_seconds, execution)


def _case_result(
    case: CrashMatrixCase,
    outcome: CrashCaseOutcome,
    elapsed_seconds: float,
    execution: CrashCaseExecution | None = None,
) -> CrashCaseResult:
    return CrashCaseResult(
        case_id=case.case_id,
        boundary=case.boundary,
        execution_surface=case.execution_surface,
        nodeid=case.nodeid,
        outcome=outcome,
        elapsed_seconds=elapsed_seconds,
        execution=execution,
    )


def main() -> int:
    load_dotenv(_ROOT / ".env")
    arguments = _arguments()
    if not os.environ.get(_POSTGRES_DSN_ENV, "").strip():
        raise RuntimeError(
            "PHASE4C_POSTGRES_DSN is required; use postgres_checkpoint_harness.py "
            "for an isolated backend"
        )
    methodology = load_crash_methodology(arguments.methodology)
    run = run_crash_certification(
        methodology,
        run_at=datetime.now(tz=UTC),
        implementation_id=arguments.implementation_id,
        execute_case=_execute_case,
    )
    write_crash_certification_run(arguments.report, run)
    print(f"Crash certification evidence written to {arguments.report}")

    if load_crash_certification_run(arguments.report) != run:
        print("Written crash evidence does not reload identically", file=sys.stderr)
        return 1
    if run.outcome is CrashCertificationOutcome.ABORTED:
        assert run.abort is not None
        retained = len(run.abort.completed_results)
        print(
            f"Crash certification aborted at {run.abort.case_id} "
            f"({run.abort.error_type}); {retained} completed cases retained",
            file=sys.stderr,
        )
        return 1
    assert run.report is not None
    if not run.report.passed:
        print("Crash certification matrix did not pass; evidence retained", file=sys.stderr)
        return 1
    try:
        require_crash_certification_evidence(
            arguments.methodology,
            arguments.report,
            expected_implementation_id=arguments.implementation_id,
        )
    except CrashCertificationError as failure:
        print(f"Crash evidence failed its own consumer contract: {failure}", file=sys.stderr)
        return 1
    print("Crash certification evidence verified against the registered methodology")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
