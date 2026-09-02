"""Lifecycle contracts for PostgreSQL checkpoint harness provisioning."""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from psycopg.conninfo import conninfo_to_dict

from scripts import postgres_checkpoint_harness as harness


@pytest.fixture(autouse=True)
def isolated_provisioning_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "load_dotenv", lambda _path: None)
    monkeypatch.delenv(harness.POSTGRES_DSN_ENV, raising=False)
    monkeypatch.delenv(harness.POSTGRES_BIN_ENV, raising=False)


def test_remote_schema_dsn_preserves_options_and_owns_search_path() -> None:
    dsn = "postgresql://user:secret@localhost/database?sslmode=require&options=-cstatement_timeout%3D5s"

    isolated = harness._schema_dsn(dsn, "phase4c_0123456789abcdef")

    parameters = conninfo_to_dict(isolated)
    assert parameters["sslmode"] == "require"
    assert parameters["options"] == (
        "-cstatement_timeout=5s -csearch_path=phase4c_0123456789abcdef"
    )


def test_remote_schema_is_removed_when_contract_execution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[tuple[str, str]] = []
    monkeypatch.setattr(
        harness,
        "_create_schema",
        lambda _dsn, schema: lifecycle.append(("create", schema)),
    )
    monkeypatch.setattr(
        harness,
        "_drop_schema",
        lambda _dsn, schema: lifecycle.append(("drop", schema)),
    )
    monkeypatch.setattr(harness.uuid, "uuid4", lambda: type("Uuid", (), {"hex": "a" * 32})())

    with (
        pytest.raises(RuntimeError, match="contract failure"),
        harness._isolated_remote_postgres("postgresql://localhost/database"),
    ):
        raise RuntimeError("contract failure")

    assert lifecycle == [
        ("create", "phase4c_aaaaaaaaaaaaaaaaaaaaaaaa"),
        ("drop", "phase4c_aaaaaaaaaaaaaaaaaaaaaaaa"),
    ]


def test_native_postgres_requires_the_exact_pinned_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = Path("postgres.exe" if os.name == "nt" else "postgres")
    monkeypatch.setattr(
        harness,
        "_run_process",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=(str(executable), "--version"),
            returncode=0,
            stdout="postgres (PostgreSQL) 18.5\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match=r"requires PostgreSQL 18\.6"):
        harness._verify_native_version(executable)


def test_native_initialization_removes_password_file_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executables = harness._NativePostgresExecutables(
        initdb=Path("initdb"),
        pg_ctl=Path("pg_ctl"),
        postgres=Path("postgres"),
    )
    password_file = Mock(spec=Path)

    def fail_init(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "initdb")

    monkeypatch.setattr(harness, "_run_process", fail_init)

    with pytest.raises(subprocess.CalledProcessError):
        harness._initialize_native_postgres(
            executables,
            Path("data"),
            password_file,
            "synthetic-password",
        )

    password_file.write_text.assert_called_once_with("synthetic-password\n", encoding="utf-8")
    password_file.chmod.assert_called_once_with(0o600)
    password_file.unlink.assert_called_once_with(missing_ok=True)


def test_native_shutdown_falls_back_to_immediate_after_fast_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes: list[str] = []

    def run_process(*args: str | Path, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        mode = str(args[args.index("-m") + 1])
        modes.append(mode)
        return subprocess.CompletedProcess(
            args=tuple(str(argument) for argument in args),
            returncode=1 if mode == "fast" else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(harness, "_run_process", run_process)

    harness._stop_native_postgres(Path("pg_ctl"), Path("data"))

    assert modes == ["fast", "immediate"]


def test_native_postgres_owns_initialization_start_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[object] = []
    executables = harness._NativePostgresExecutables(
        initdb=Path("initdb"),
        pg_ctl=Path("pg_ctl"),
        postgres=Path("postgres"),
    )

    @contextmanager
    def temporary_directory(*, prefix: str):
        lifecycle.append(("temporary-enter", prefix))
        try:
            yield "owned-temporary-directory"
        finally:
            lifecycle.append("temporary-exit")

    monkeypatch.setattr(harness.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(harness, "_native_executables", lambda _directory: executables)
    monkeypatch.setattr(
        harness,
        "_verify_native_version",
        lambda executable: lifecycle.append(("version", executable)),
    )
    monkeypatch.setattr(harness.secrets, "token_urlsafe", lambda _size: "synthetic-password")
    monkeypatch.setattr(harness, "_available_loopback_port", lambda: 55432)
    monkeypatch.setattr(
        harness,
        "_initialize_native_postgres",
        lambda *_args: lifecycle.append("initialize"),
    )
    monkeypatch.setattr(
        harness,
        "_start_native_postgres",
        lambda *_args: lifecycle.append("start"),
    )
    monkeypatch.setattr(
        harness,
        "_stop_native_postgres",
        lambda *_args: lifecycle.append("stop"),
    )

    with harness._native_postgres("configured-bin") as dsn:
        assert dsn == (
            "postgresql://postgres:synthetic-password@127.0.0.1:55432/postgres?sslmode=disable"
        )
        lifecycle.append("contracts")

    assert lifecycle == [
        ("version", Path("postgres")),
        ("temporary-enter", "agnostic-market-phase4c-"),
        "initialize",
        "start",
        "contracts",
        "stop",
        "temporary-exit",
    ]


def test_native_start_failure_stops_a_partially_started_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    executables = harness._NativePostgresExecutables(
        initdb=Path("initdb"),
        pg_ctl=Path("pg_ctl"),
        postgres=Path("postgres"),
    )

    @contextmanager
    def temporary_directory(*, prefix: str):
        assert prefix == "agnostic-market-phase4c-"
        yield "owned-temporary-directory"

    def fail_start(*_args: object) -> None:
        lifecycle.append("start")
        raise TimeoutError("startup timed out")

    monkeypatch.setattr(harness.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(harness, "_native_executables", lambda _directory: executables)
    monkeypatch.setattr(harness, "_verify_native_version", lambda _executable: None)
    monkeypatch.setattr(harness, "_available_loopback_port", lambda: 55432)
    monkeypatch.setattr(harness, "_initialize_native_postgres", lambda *_args: None)
    monkeypatch.setattr(harness, "_start_native_postgres", fail_start)
    monkeypatch.setattr(harness, "_native_server_running", lambda _directory: True)
    monkeypatch.setattr(harness, "_report_native_log", lambda _path: lifecycle.append("log"))
    monkeypatch.setattr(
        harness,
        "_stop_native_postgres",
        lambda *_args: lifecycle.append("stop"),
    )

    with (
        pytest.raises(TimeoutError, match="startup timed out"),
        harness._native_postgres("configured-bin"),
    ):
        pytest.fail("startup failure must not yield a DSN")

    assert lifecycle == ["start", "log", "stop"]


def test_main_rejects_ambiguous_provisioning_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(harness.POSTGRES_DSN_ENV, "postgresql://localhost/database")
    monkeypatch.setenv(harness.POSTGRES_BIN_ENV, "C:/postgres/bin")

    with pytest.raises(RuntimeError, match="Configure only one"):
        harness.main()


def test_main_runs_contracts_through_the_remote_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = "postgresql://localhost/database"
    isolated = "host=localhost dbname=database options=-csearch_path=phase4c_owned"
    observed: list[str] = []
    monkeypatch.setenv(harness.POSTGRES_DSN_ENV, supplied)

    @contextmanager
    def isolated_remote(dsn: str):
        assert dsn == supplied
        yield isolated

    monkeypatch.setattr(harness, "_isolated_remote_postgres", isolated_remote)
    monkeypatch.setattr(harness, "_run_contracts", observed.append)

    harness.main()

    assert observed == [isolated]


def test_main_runs_contracts_through_native_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    monkeypatch.setenv(harness.POSTGRES_BIN_ENV, "C:/postgres/bin")

    @contextmanager
    def native_postgres(binary_directory: str):
        assert binary_directory == "C:/postgres/bin"
        yield "postgresql://native/isolated"

    monkeypatch.setattr(harness, "_native_postgres", native_postgres)
    monkeypatch.setattr(harness, "_run_contracts", observed.append)

    harness.main()

    assert observed == ["postgresql://native/isolated"]


def test_main_uses_the_container_when_no_alternative_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    @contextmanager
    def container_postgres():
        yield "postgresql://container/isolated"

    monkeypatch.setattr(harness, "_container_postgres", container_postgres)
    monkeypatch.setattr(harness, "_run_contracts", observed.append)

    harness.main()

    assert observed == ["postgresql://container/isolated"]
