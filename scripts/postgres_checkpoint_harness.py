"""Run the Phase 4C checkpoint contracts against isolated PostgreSQL 18.6."""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

POSTGRES_VERSION = "18.6"
POSTGRES_IMAGE = (
    "docker.io/library/postgres:18.6@"
    "sha256:4ef4dbc939d61acea57712655ddb4b4ab27419c913f94cca0cd57cb3ea3c2280"
)
POSTGRES_DSN_ENV = "PHASE4C_POSTGRES_DSN"
POSTGRES_BIN_ENV = "PHASE4C_POSTGRES_BIN"
_DATABASE = "agnostic_market_phase4c"
_USER = "postgres"
_ROOT = Path(__file__).resolve().parents[1]
_START_TIMEOUT_SECONDS = 60
_STOP_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class _NativePostgresExecutables:
    initdb: Path
    pg_ctl: Path
    postgres: Path


def _run_process(
    *args: str | os.PathLike[str],
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - executables and arguments are validated or internal
        args,
        cwd=_ROOT,
        check=check,
        text=True,
        capture_output=capture,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def _docker(
    *args: str,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("docker")
    if executable is None:
        raise RuntimeError(
            "Docker is required unless PHASE4C_POSTGRES_DSN or PHASE4C_POSTGRES_BIN is supplied"
        )
    return _run_process(
        executable,
        *args,
        check=check,
        capture=capture,
        quiet=quiet,
    )


def _schema_dsn(dsn: str, schema: str) -> str:
    parameters = conninfo_to_dict(dsn)
    existing_options = parameters.get("options")
    if existing_options is not None and not isinstance(existing_options, str):
        raise RuntimeError("PostgreSQL connection options must be text")
    options = " ".join(option for option in (existing_options, f"-csearch_path={schema}") if option)
    return make_conninfo(dsn, options=options)


def _create_schema(dsn: str, schema: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))


def _drop_schema(dsn: str, schema: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@contextmanager
def _isolated_remote_postgres(dsn: str) -> Iterator[str]:
    schema = f"phase4c_{uuid.uuid4().hex[:24]}"
    _create_schema(dsn, schema)
    try:
        yield _schema_dsn(dsn, schema)
    finally:
        _drop_schema(dsn, schema)


def _published_port(container_name: str) -> str:
    result = _docker("port", container_name, "5432/tcp", capture=True)
    endpoint = result.stdout.strip().rsplit(":", maxsplit=1)
    if len(endpoint) != 2 or not endpoint[1].isdigit():
        raise RuntimeError("Docker did not publish the PostgreSQL harness port")
    return endpoint[1]


def _await_healthy(container_name: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _docker(
            "inspect",
            "--format",
            "{{.State.Health.Status}}",
            container_name,
            check=False,
            capture=True,
        )
        status = result.stdout.strip()
        if result.returncode == 0 and status == "healthy":
            return
        if status == "unhealthy":
            raise RuntimeError("disposable PostgreSQL failed its health check")
        time.sleep(1)
    raise TimeoutError("disposable PostgreSQL did not become healthy")


@contextmanager
def _container_postgres() -> Iterator[str]:
    container_name = f"agnostic-market-phase4c-{uuid.uuid4().hex[:12]}"
    password = secrets.token_urlsafe(24)
    _docker(
        "run",
        "--detach",
        "--rm",
        "--name",
        container_name,
        "--env",
        f"POSTGRES_PASSWORD={password}",
        "--env",
        f"POSTGRES_DB={_DATABASE}",
        "--health-cmd",
        f"pg_isready -U {_USER} -d {_DATABASE}",
        "--health-interval",
        "1s",
        "--health-timeout",
        "5s",
        "--health-retries",
        "30",
        "--publish",
        "127.0.0.1::5432",
        POSTGRES_IMAGE,
    )
    try:
        _await_healthy(container_name)
        port = _published_port(container_name)
        yield f"postgresql://{_USER}:{quote(password, safe='')}@127.0.0.1:{port}/{_DATABASE}"
    finally:
        _docker("rm", "--force", container_name, check=False, quiet=True)


def _native_executables(binary_directory: str) -> _NativePostgresExecutables:
    directory = Path(binary_directory).expanduser().resolve()
    if not directory.is_dir():
        raise RuntimeError(f"{POSTGRES_BIN_ENV} must name a PostgreSQL binary directory")
    suffix = ".exe" if os.name == "nt" else ""

    def required(name: str) -> Path:
        executable = directory / f"{name}{suffix}"
        if not executable.is_file():
            raise RuntimeError(f"PostgreSQL binary is missing: {executable}")
        return executable

    return _NativePostgresExecutables(
        initdb=required("initdb"),
        pg_ctl=required("pg_ctl"),
        postgres=required("postgres"),
    )


def _verify_native_version(postgres: Path) -> None:
    result = _run_process(postgres, "--version", capture=True)
    expected = f"postgres (PostgreSQL) {POSTGRES_VERSION}"
    if result.stdout.strip() != expected:
        raise RuntimeError(
            f"{POSTGRES_BIN_ENV} requires PostgreSQL {POSTGRES_VERSION}; "
            f"found {result.stdout.strip() or 'unknown'}"
        )


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _initialize_native_postgres(
    executables: _NativePostgresExecutables,
    data_directory: Path,
    password_file: Path,
    password: str,
) -> None:
    password_file.write_text(f"{password}\n", encoding="utf-8")
    password_file.chmod(0o600)
    try:
        _run_process(
            executables.initdb,
            "-D",
            data_directory,
            "-U",
            _USER,
            "--auth-host=scram-sha-256",
            "--auth-local=scram-sha-256",
            f"--pwfile={password_file}",
            "--encoding=UTF8",
            "--no-instructions",
        )
    finally:
        password_file.unlink(missing_ok=True)


def _start_native_postgres(
    pg_ctl: Path,
    data_directory: Path,
    log_file: Path,
    port: int,
) -> None:
    _run_process(
        pg_ctl,
        "-D",
        data_directory,
        "-l",
        log_file,
        "-w",
        "-t",
        str(_START_TIMEOUT_SECONDS),
        "-o",
        f"-h 127.0.0.1 -p {port}",
        "start",
    )


def _stop_native_postgres(pg_ctl: Path, data_directory: Path) -> None:
    common = (pg_ctl, "-D", data_directory, "stop", "-w", "-t", str(_STOP_TIMEOUT_SECONDS))
    fast = _run_process(*common, "-m", "fast", check=False, capture=True)
    if fast.returncode == 0:
        return
    immediate = _run_process(*common, "-m", "immediate", check=False, capture=True)
    if immediate.returncode != 0:
        raise RuntimeError("temporary PostgreSQL did not stop after fast or immediate shutdown")


def _report_native_log(log_file: Path) -> None:
    if not log_file.is_file():
        return
    contents = log_file.read_text(encoding="utf-8", errors="replace").strip()
    if contents:
        print(f"Temporary PostgreSQL log:\n{contents}", file=sys.stderr)


def _native_server_running(data_directory: Path) -> bool:
    return (data_directory / "postmaster.pid").is_file()


@contextmanager
def _native_postgres(binary_directory: str) -> Iterator[str]:
    executables = _native_executables(binary_directory)
    _verify_native_version(executables.postgres)
    password = secrets.token_urlsafe(24)
    port = _available_loopback_port()
    body_failed = False

    with tempfile.TemporaryDirectory(prefix="agnostic-market-phase4c-") as temporary:
        root = Path(temporary)
        data_directory = root / "data"
        password_file = root / "initdb-password.txt"
        log_file = root / "postgres.log"
        _initialize_native_postgres(executables, data_directory, password_file, password)
        started = False
        try:
            _start_native_postgres(executables.pg_ctl, data_directory, log_file, port)
            started = True
            yield (
                f"postgresql://{_USER}:{quote(password, safe='')}@127.0.0.1:{port}/postgres"
                "?sslmode=disable"
            )
        except BaseException:
            body_failed = True
            _report_native_log(log_file)
            raise
        finally:
            if started or _native_server_running(data_directory):
                try:
                    _stop_native_postgres(executables.pg_ctl, data_directory)
                except Exception as error:
                    _report_native_log(log_file)
                    if not body_failed:
                        raise
                    print(f"Temporary PostgreSQL cleanup failed: {error}", file=sys.stderr)


def _run_contracts(dsn: str) -> None:
    environment = {**os.environ, POSTGRES_DSN_ENV: dsn}
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "postgres",
            "tests/test_postgres_checkpoints.py",
        ),
        cwd=_ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    load_dotenv(_ROOT / ".env")
    supplied_dsn = os.environ.get(POSTGRES_DSN_ENV, "").strip()
    binary_directory = os.environ.get(POSTGRES_BIN_ENV, "").strip()
    if supplied_dsn and binary_directory:
        raise RuntimeError(f"Configure only one of {POSTGRES_DSN_ENV} and {POSTGRES_BIN_ENV}")

    if supplied_dsn:
        provisioner = _isolated_remote_postgres(supplied_dsn)
    elif binary_directory:
        provisioner = _native_postgres(binary_directory)
    else:
        provisioner = _container_postgres()

    with provisioner as dsn:
        _run_contracts(dsn)


if __name__ == "__main__":
    main()
