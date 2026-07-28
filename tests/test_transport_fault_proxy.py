from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.llm.providers import load_conformance_targets
from scripts.transport_fault_proxy import (
    PROVIDER_TRANSPORT_CONTRACTS,
    TRANSPORT_CONTRACT_VERSION,
    FaultKind,
    FaultMode,
    TransportFaultProxy,
)

_REQUEST_CEILING = 6
_FAULT_WINDOW_SECONDS = 10.0
_UPSTREAM_TIMEOUT_SECONDS = 5.0
_DISCONNECT_ERRORS = (http.client.RemoteDisconnected, OSError)
_INTERRUPTED_BODY_ERRORS = (http.client.IncompleteRead, *_DISCONNECT_ERRORS)


class _DummySecretResolver:
    def resolve(self, _ref: str) -> str:
        return "offline-not-a-secret"


class _UpstreamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextmanager
def _provider_upstream():
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            paths.append(self.path)
            if self.path == "/v1/chat/completions":
                body = {
                    "id": "chatcmpl-transport-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "transport-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            elif self.path == "/v1/messages":
                body = {
                    "id": "msg_transport_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "transport-test",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            else:
                self.send_error(404)
                return
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection = True

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = _UpstreamServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=_UPSTREAM_TIMEOUT_SECONDS)


def _proxy(
    provider: str,
    *,
    mode: FaultMode,
    kind: FaultKind,
    upstream_base_url: str,
    request_ceiling: int = _REQUEST_CEILING,
) -> TransportFaultProxy:
    return TransportFaultProxy(
        PROVIDER_TRANSPORT_CONTRACTS[provider],
        mode=mode,
        fault_kind=kind,
        request_ceiling=request_ceiling,
        fault_window_seconds=_FAULT_WINDOW_SECONDS,
        upstream_timeout_seconds=_UPSTREAM_TIMEOUT_SECONDS,
        upstream_base_url=upstream_base_url,
    )


def _request(base_url: str, path: str, *, method: str = "POST") -> tuple[int, bytes]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port,
        timeout=_UPSTREAM_TIMEOUT_SECONDS,
    )
    try:
        connection.request(method, path, body=b"{}")
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_transport_contract_is_versioned_and_provider_paths_are_explicit() -> None:
    assert TRANSPORT_CONTRACT_VERSION == "1"
    assert {
        provider: (contract.method, contract.path, contract.client_base_path)
        for provider, contract in PROVIDER_TRANSPORT_CONTRACTS.items()
    } == {
        "openai": ("POST", "/v1/chat/completions", "/v1"),
        "anthropic": ("POST", "/v1/messages", ""),
    }


def test_unexpected_provider_endpoint_fails_without_reaching_upstream() -> None:
    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            "openai",
            mode=FaultMode.RETRY_MASKED_ONCE,
            kind=FaultKind.PRE_RESPONSE_DISCONNECT,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        status, body = _request(proxy.model_base_url, "/v1/responses")

    assert status == 502
    assert json.loads(body) == {"error": "unexpected_provider_endpoint"}
    assert paths == []
    assert [attempt.outcome for attempt in proxy.attempts] == ["unexpected_provider_endpoint"]


def test_unexpected_provider_method_fails_without_reaching_upstream() -> None:
    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            "anthropic",
            mode=FaultMode.RETRY_MASKED_ONCE,
            kind=FaultKind.PRE_RESPONSE_DISCONNECT,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        status, body = _request(
            proxy.model_base_url,
            "/v1/messages",
            method="OPTIONS",
        )

    assert status == 502
    assert json.loads(body) == {"error": "unexpected_provider_endpoint"}
    assert paths == []
    assert [attempt.outcome for attempt in proxy.attempts] == ["unexpected_provider_endpoint"]


def test_interrupted_body_closes_before_the_declared_content_length() -> None:
    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            "openai",
            mode=FaultMode.UNTIL_EXHAUSTED,
            kind=FaultKind.INTERRUPTED_BODY,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        parsed = urlsplit(proxy.model_base_url)
        connection = http.client.HTTPConnection(
            parsed.hostname,
            parsed.port,
            timeout=_UPSTREAM_TIMEOUT_SECONDS,
        )
        connection.request("POST", "/v1/chat/completions", body=b"{}")
        response = connection.getresponse()
        with pytest.raises(_INTERRUPTED_BODY_ERRORS):
            response.read()
        response.close()
        connection.close()

    assert paths == []
    assert [attempt.outcome for attempt in proxy.attempts] == ["faulted"]


def test_retry_masked_once_faults_then_passes_to_upstream() -> None:
    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            "openai",
            mode=FaultMode.RETRY_MASKED_ONCE,
            kind=FaultKind.PRE_RESPONSE_DISCONNECT,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        with pytest.raises(_DISCONNECT_ERRORS):
            _request(proxy.model_base_url, "/v1/chat/completions")
        status, _body = _request(proxy.model_base_url, "/v1/chat/completions")

    assert status == 200
    assert paths == ["/v1/chat/completions"]
    assert [attempt.outcome for attempt in proxy.attempts] == [
        "faulted",
        "passed_upstream",
    ]


def test_request_ceiling_stays_fail_closed() -> None:
    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            "openai",
            mode=FaultMode.UNTIL_EXHAUSTED,
            kind=FaultKind.PRE_RESPONSE_DISCONNECT,
            upstream_base_url=upstream,
            request_ceiling=1,
        ) as proxy,
    ):
        for _attempt in range(2):
            with pytest.raises(_DISCONNECT_ERRORS):
                _request(proxy.model_base_url, "/v1/chat/completions")

    assert paths == []
    assert [attempt.outcome for attempt in proxy.attempts] == ["faulted", "safety_ceiling"]


@pytest.mark.parametrize("provider", ("openai", "anthropic"))
def test_installed_sync_adapter_retries_once_then_succeeds(
    config_root: Path,
    provider: str,
) -> None:
    targets = load_conformance_targets(config_root / "conformance" / "targets.yaml")
    target = next(target for target in targets.targets if target.provider == provider)
    gateway = LLMGateway(
        load_provider_credentials(config_root / "base" / "providers.yaml"),
        _DummySecretResolver(),
    )

    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            provider,
            mode=FaultMode.RETRY_MASKED_ONCE,
            kind=FaultKind.PRE_RESPONSE_DISCONNECT,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        model = gateway.chat_model(
            target,
            base_url=proxy.model_base_url,
            max_retries=1,
            timeout=_UPSTREAM_TIMEOUT_SECONDS,
        )
        response = model.invoke("Say ok.")

    assert response.content == "ok"
    assert paths == [PROVIDER_TRANSPORT_CONTRACTS[provider].path]
    assert [attempt.outcome for attempt in proxy.attempts] == [
        "faulted",
        "passed_upstream",
    ]


@pytest.mark.parametrize("provider", ("openai", "anthropic"))
@pytest.mark.parametrize(
    "kind",
    (FaultKind.PRE_RESPONSE_DISCONNECT, FaultKind.INTERRUPTED_BODY),
)
def test_installed_sync_adapter_exhausts_retries_without_upstream(
    config_root: Path,
    provider: str,
    kind: FaultKind,
) -> None:
    targets = load_conformance_targets(config_root / "conformance" / "targets.yaml")
    target = next(target for target in targets.targets if target.provider == provider)
    gateway = LLMGateway(
        load_provider_credentials(config_root / "base" / "providers.yaml"),
        _DummySecretResolver(),
    )

    with (
        _provider_upstream() as (upstream, paths),
        _proxy(
            provider,
            mode=FaultMode.UNTIL_EXHAUSTED,
            kind=kind,
            upstream_base_url=upstream,
        ) as proxy,
    ):
        model = gateway.chat_model(
            target,
            base_url=proxy.model_base_url,
            max_retries=1,
            timeout=_UPSTREAM_TIMEOUT_SECONDS,
        )
        with pytest.raises(Exception) as raised:
            model.invoke("Say ok.")

    assert type(raised.value).__name__ == "APIConnectionError"
    assert paths == []
    assert len(proxy.attempts) == 2
    assert {attempt.outcome for attempt in proxy.attempts} == {"faulted"}
