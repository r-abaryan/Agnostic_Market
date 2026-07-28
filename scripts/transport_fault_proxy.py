"""Loopback-only provider transport fault harness for Milestone 6E certification."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

TRANSPORT_CONTRACT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ProviderTransportContract:
    provider: str
    method: Literal["POST"]
    path: str
    client_base_path: str
    upstream_base_url: str


# EXCEPTION: these literals are the deliberately versioned endpoint assertions under test.
PROVIDER_TRANSPORT_CONTRACTS: Mapping[str, ProviderTransportContract] = MappingProxyType(
    {
        "openai": ProviderTransportContract(
            provider="openai",
            method="POST",
            path="/v1/chat/completions",
            client_base_path="/v1",
            upstream_base_url="https://api.openai.com",
        ),
        "anthropic": ProviderTransportContract(
            provider="anthropic",
            method="POST",
            path="/v1/messages",
            client_base_path="",
            upstream_base_url="https://api.anthropic.com",
        ),
    }
)


class FaultMode(StrEnum):
    RETRY_MASKED_ONCE = "retry_masked_once"
    UNTIL_EXHAUSTED = "until_exhausted"


class FaultKind(StrEnum):
    PRE_RESPONSE_DISCONNECT = "pre_response_disconnect"
    INTERRUPTED_BODY = "interrupted_body"


AttemptOutcome = Literal[
    "faulted",
    "passed_upstream",
    "unexpected_provider_endpoint",
    "safety_ceiling",
    "upstream_error",
]


@dataclass(frozen=True, slots=True)
class ProxyAttempt:
    sequence: int
    method: str
    path: str
    outcome: AttemptOutcome
    status_code: int | None = None


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class TransportFaultProxy:
    """One scenario-exclusive proxy; request/response content is never recorded."""

    def __init__(
        self,
        contract: ProviderTransportContract,
        *,
        mode: FaultMode,
        fault_kind: FaultKind,
        request_ceiling: int,
        fault_window_seconds: float,
        upstream_timeout_seconds: float,
        upstream_base_url: str | None = None,
    ) -> None:
        if request_ceiling < 1:
            raise ValueError("request_ceiling must be positive")
        if fault_window_seconds <= 0 or upstream_timeout_seconds <= 0:
            raise ValueError("proxy timeouts must be positive")
        self._contract = contract
        self._mode = mode
        self._fault_kind = fault_kind
        self._request_ceiling = request_ceiling
        self._fault_window_seconds = fault_window_seconds
        self._upstream_timeout_seconds = upstream_timeout_seconds
        self._upstream_base_url = (
            upstream_base_url
            if upstream_base_url is not None
            else contract.upstream_base_url
            if mode == FaultMode.RETRY_MASKED_ONCE
            else None
        )
        self._lock = threading.Lock()
        self._attempts: list[ProxyAttempt] = []
        self._started_at = 0.0
        self._fault_count = 0
        self._server: _LoopbackServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def attempts(self) -> tuple[ProxyAttempt, ...]:
        with self._lock:
            return tuple(self._attempts)

    @property
    def model_base_url(self) -> str:
        server = self._server
        if server is None:
            raise RuntimeError("transport fault proxy is not running")
        host, port = server.server_address
        return f"http://{host}:{port}{self._contract.client_base_path}"

    def __enter__(self) -> TransportFaultProxy:
        if self._server is not None:
            raise RuntimeError("transport fault proxy is already running")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                owner._handle(self)

            def do_GET(self) -> None:
                owner._reject_endpoint(self)

            def do_PUT(self) -> None:
                owner._reject_endpoint(self)

            def do_PATCH(self) -> None:
                owner._reject_endpoint(self)

            def do_DELETE(self) -> None:
                owner._reject_endpoint(self)

            def do_CONNECT(self) -> None:
                owner._reject_endpoint(self)

            def do_HEAD(self) -> None:
                owner._reject_endpoint(self)

            def do_OPTIONS(self) -> None:
                owner._reject_endpoint(self)

            def do_TRACE(self) -> None:
                owner._reject_endpoint(self)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = _LoopbackServer(("127.0.0.1", 0), Handler)
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"transport-fault-proxy-{self._contract.provider}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=self._upstream_timeout_seconds)

    def _record(
        self,
        handler: BaseHTTPRequestHandler,
        outcome: AttemptOutcome,
        *,
        status_code: int | None = None,
    ) -> None:
        with self._lock:
            self._attempts.append(
                ProxyAttempt(
                    sequence=len(self._attempts) + 1,
                    method=handler.command,
                    path=handler.path,
                    outcome=outcome,
                    status_code=status_code,
                )
            )

    def _contract_matches(self, handler: BaseHTTPRequestHandler) -> bool:
        return handler.command == self._contract.method and handler.path == self._contract.path

    def _safety_exceeded_locked(self) -> bool:
        return bool(
            len(self._attempts) + 1 > self._request_ceiling
            or time.monotonic() - self._started_at > self._fault_window_seconds
        )

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        if not self._contract_matches(handler):
            self._reject_endpoint(handler)
            return
        with self._lock:
            over_safety_ceiling = self._safety_exceeded_locked()
            should_fault = bool(
                not over_safety_ceiling
                and (self._mode == FaultMode.UNTIL_EXHAUSTED or self._fault_count == 0)
            )
            if should_fault:
                self._fault_count += 1
            outcome: AttemptOutcome = (
                "safety_ceiling"
                if over_safety_ceiling
                else "faulted"
                if should_fault
                else "passed_upstream"
            )
            if outcome != "passed_upstream":
                self._attempts.append(
                    ProxyAttempt(
                        sequence=len(self._attempts) + 1,
                        method=handler.command,
                        path=handler.path,
                        outcome=outcome,
                    )
                )
        if over_safety_ceiling:
            self._disconnect(handler)
            return
        if should_fault:
            if self._fault_kind == FaultKind.PRE_RESPONSE_DISCONNECT:
                self._disconnect(handler)
            else:
                self._interrupt_body(handler)
            return
        self._forward(handler)

    def _reject_endpoint(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            over_safety_ceiling = self._safety_exceeded_locked()
            self._attempts.append(
                ProxyAttempt(
                    sequence=len(self._attempts) + 1,
                    method=handler.command,
                    path=handler.path,
                    outcome=(
                        "safety_ceiling" if over_safety_ceiling else "unexpected_provider_endpoint"
                    ),
                    status_code=None if over_safety_ceiling else 502,
                )
            )
        if over_safety_ceiling:
            self._disconnect(handler)
            return
        body = json.dumps({"error": "unexpected_provider_endpoint"}).encode()
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.close_connection = True

    @staticmethod
    def _disconnect(handler: BaseHTTPRequestHandler) -> None:
        handler.close_connection = True
        with suppress(OSError):
            handler.connection.shutdown(socket.SHUT_RDWR)
        handler.connection.close()

    def _interrupt_body(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            if content_length < 0:
                raise ValueError
        except ValueError:
            self._disconnect(handler)
            return
        handler.connection.settimeout(self._upstream_timeout_seconds)
        try:
            handler.rfile.read(content_length)
        except OSError:
            self._disconnect(handler)
            return

        partial = b'{"id":"transport-interrupted"'
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(partial) + 128))
        handler.send_header("Connection", "close")
        handler.end_headers()
        with suppress(OSError):
            handler.wfile.write(partial)
            handler.wfile.flush()
        handler.close_connection = True
        with suppress(OSError):
            handler.connection.shutdown(socket.SHUT_WR)

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        if self._upstream_base_url is None:
            self._record(handler, "upstream_error")
            self._disconnect(handler)
            return
        upstream = urlsplit(self._upstream_base_url)
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            self._record(handler, "upstream_error")
            self._disconnect(handler)
            return
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            self._record(handler, "upstream_error")
            self._disconnect(handler)
            return
        body = handler.rfile.read(content_length)
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.casefold() not in _HOP_BY_HOP_HEADERS and name.casefold() != "host"
        }
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            upstream.hostname,
            upstream.port,
            timeout=self._upstream_timeout_seconds,
        )
        try:
            connection.request(
                self._contract.method,
                self._contract.path,
                body=body,
                headers={**headers, "Host": upstream.netloc},
            )
            response = connection.getresponse()
            response_body = response.read()
            handler.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if (
                    name.casefold() not in _HOP_BY_HOP_HEADERS
                    and name.casefold() != "content-length"
                ):
                    handler.send_header(name, value)
            handler.send_header("Content-Length", str(len(response_body)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(response_body)
            handler.close_connection = True
            self._record(handler, "passed_upstream", status_code=response.status)
        except (OSError, http.client.HTTPException):
            self._record(handler, "upstream_error")
            self._disconnect(handler)
        finally:
            connection.close()
