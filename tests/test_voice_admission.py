"""Trusted tenant admission for console, SIP, and explicit-dispatch voice jobs."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import warnings
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest
from livekit import rtc

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.config.resolver import ConfigResolutionError
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver
from agnostic_market.voice.admission import (
    ConsoleVoiceTenantAdmission,
    NetworkVoiceTenantAdmission,
    VoiceJobAdmission,
    VoiceTenantAdmission,
)

_ADMISSION_TIMEOUT_SECONDS = 1.0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows worker loop contract")
def test_worker_prewarm_selects_psycopg_loop_before_livekit_runs() -> None:
    probe = r"""
import asyncio
import os
import runpy
import socket
import sys
from types import SimpleNamespace
from livekit import agents
from livekit.agents.ipc.proc_client import _ProcClient
from livekit.agents.ipc.channel import send_message, recv_message
from livekit.agents.ipc.proto import InitializeRequest, InitializeResponse, IPC_MESSAGES
from livekit.agents.utils.aio.duplex_unix import _Duplex, _AsyncDuplex
from psycopg import AsyncConnection

class ConnectionParametersReached(Exception):
    pass

def connection_parameters(cls, *args, **kwargs):
    raise ConnectionParametersReached

AsyncConnection._get_connection_params = classmethod(connection_parameters)

class ProbeClient(_ProcClient):
    async def _monitor_task(self):
        assert isinstance(asyncio.get_running_loop(), asyncio.SelectorEventLoop)
        try:
            await AsyncConnection.connect()
        except ConnectionParametersReached:
            pass
        else:
            raise AssertionError("psycopg connection guard was not exercised")
        left, right = socket.socketpair()
        first, second = await asyncio.gather(_AsyncDuplex.open(left), _AsyncDuplex.open(right))
        try:
            await first.send_bytes(b"worker-ipc")
            assert await asyncio.wait_for(second.recv_bytes(), 2) == b"worker-ipc"
        finally:
            await first.aclose()
            await second.aclose()

def run_job(options):
    parent_socket, child_socket = socket.socketpair()
    parent = _Duplex.open(parent_socket)
    client = ProbeClient(child_socket, None,
        lambda *_: options.prewarm_fnc(SimpleNamespace()), None)
    try:
        send_message(parent, InitializeRequest())
        client.initialize()
        response = recv_message(parent, IPC_MESSAGES)
        assert isinstance(response, InitializeResponse) and not response.error
        client.run()
        client._task.result()
    finally:
        parent.close()
        child_socket.close()
        asyncio.get_event_loop().close()

def run_app(options):
    supervisor = asyncio.new_event_loop()
    async def run_worker():
        await asyncio.to_thread(run_job, options)
        assert asyncio.get_running_loop() is supervisor
    try:
        supervisor.run_until_complete(run_worker())
        supervisor.run_until_complete(supervisor.shutdown_default_executor())
    finally:
        supervisor.close()

agents.cli.run_app = run_app
sys.argv = ["scripts/voice_agent.py", "dev"]
os.environ["VOICE_AGENT_NAME"] = "agnostic-market"
sys.path.insert(0, "scripts")
runpy.run_path("scripts/voice_agent.py", run_name="__main__")
"""
    result = subprocess.run(  # noqa: S603 - fixed probe in a fresh interpreter, no shell
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows worker loop contract")
def test_console_worker_does_not_replace_the_process_event_loop_policy() -> None:
    from scripts import voice_agent

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prewarm = voice_agent._prewarm_for_arguments(("console",))
        prewarm(SimpleNamespace())
    assert not caught
    loop = asyncio.new_event_loop()
    try:
        assert isinstance(loop, asyncio.ProactorEventLoop)
    finally:
        loop.close()


async def _run_admission(
    job_context: _JobContext,
    registry: ConfigRegistry,
    *,
    development_merchant_id: str | None,
) -> VoiceTenantAdmission:
    boundary = VoiceJobAdmission(
        registry,
        development_merchant_id=development_merchant_id,
    )
    preflight = boundary.preflight(job_context)
    return await boundary.complete(
        job_context,
        preflight,
        timeout_seconds=_ADMISSION_TIMEOUT_SECONDS,
    )


class _JobContext:
    def __init__(
        self,
        *,
        fake: bool,
        metadata: str = "",
        job_id: str = "AJ_test_job",
        dispatch_id: str = "AD_test_dispatch",
        room_id: str = "RM_test_room",
        worker_id: str = "AW_test_worker",
        participant_kind: int = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        participant_attributes: dict[str, str] | None = None,
        participant_identity: str = "caller-primary",
        additional_participants: tuple[SimpleNamespace, ...] = (),
        connect_event: asyncio.Event | None = None,
        wait_event: asyncio.Event | None = None,
    ) -> None:
        self._fake = fake
        self.job = SimpleNamespace(
            metadata=metadata,
            id=job_id,
            dispatch_id=dispatch_id,
            room=SimpleNamespace(sid=room_id),
        )
        self.worker_id = worker_id
        self.room = object()
        self.participants = (
            SimpleNamespace(
                identity=participant_identity,
                kind=participant_kind,
                attributes=participant_attributes or {},
            ),
            *additional_participants,
        )
        self.connect_count = 0
        self.wait_count = 0
        self.wait_identity: str | None = None
        self.wait_kind: int | None = None
        self.connect_event = connect_event
        self.wait_event = wait_event
        self.shutdown_callbacks = []

    def is_fake_job(self) -> bool:
        return self._fake

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_event is not None:
            await self.connect_event.wait()

    async def wait_for_participant(
        self,
        *,
        identity: str | None = None,
        kind: int | None = None,
    ):
        self.wait_count += 1
        self.wait_identity = identity
        self.wait_kind = kind
        if self.wait_event is not None:
            await self.wait_event.wait()
        for participant in self.participants:
            if identity is not None and participant.identity != identity:
                continue
            if kind is not None and participant.kind != kind:
                continue
            return participant
        raise AssertionError("no participant matched the requested admission binding")

    def add_shutdown_callback(self, callback) -> None:
        self.shutdown_callbacks.append(callback)


async def test_console_admission_requires_an_explicit_known_merchant(
    registry: ConfigRegistry,
) -> None:
    missing = _JobContext(fake=True)
    with pytest.raises(TenantResolutionError, match="requires VOICE_AGENT_MERCHANT_ID"):
        await _run_admission(  # type: ignore[arg-type]
            missing,
            registry,
            development_merchant_id=None,
        )
    assert missing.connect_count == 0

    admitted = await _run_admission(  # type: ignore[arg-type]
        job := _JobContext(fake=True),
        registry,
        development_merchant_id="acme_store",
    )
    assert isinstance(admitted, ConsoleVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "acme_store"
    assert admitted.tenant.config_version == admitted.resolved.config_version
    assert job.connect_count == 1


async def test_sip_admission_uses_the_called_trunk_and_not_caller_ani(
    registry: ConfigRegistry,
) -> None:
    job = _JobContext(
        fake=False,
        metadata=('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip"}'),
        participant_kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        participant_attributes={
            "sip.ruleID": "rule-acme",
            "sip.trunkPhoneNumber": "+15551230001",
            "sip.phoneNumber": "+15551230002",
        },
    )

    admitted = await _run_admission(  # type: ignore[arg-type]
        job,
        registry,
        development_merchant_id="demo_shop",
    )

    assert isinstance(admitted, NetworkVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "acme_store"
    assert admitted.participant_identity == "caller-primary"
    assert admitted.session_authority.logical_session_id == "AD_test_dispatch"
    assert admitted.session_authority.transport.room_id == "RM_test_room"
    assert admitted.session_authority.transport.assignment_id == "AJ_test_job"
    assert admitted.session_authority.transport.worker_id == "AW_test_worker"
    assert job.connect_count == 1
    assert job.wait_count == 1
    assert job.wait_kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP


async def test_non_sip_admission_requires_strict_explicit_dispatch_metadata(
    registry: ConfigRegistry,
) -> None:
    admitted = await _run_admission(  # type: ignore[arg-type]
        _JobContext(
            fake=False,
            metadata=(
                '{"schema_version":1,"merchant_id":"demo_shop",'
                '"participant_kind":"standard","participant_identity":"caller-primary"}'
            ),
        ),
        registry,
        development_merchant_id="acme_store",
    )
    assert admitted.tenant.tenant_id == "demo_shop"

    missing = _JobContext(fake=False)
    with pytest.raises(TenantResolutionError, match="approved dispatch metadata"):
        await _run_admission(  # type: ignore[arg-type]
            missing,
            registry,
            development_merchant_id="acme_store",
        )
    assert missing.connect_count == 0


@pytest.mark.parametrize(
    "field_name, value",
    (
        ("job_id", ""),
        ("job_id", " AJ_test_job"),
        ("dispatch_id", ""),
        ("dispatch_id", "AD_test_dispatch "),
        ("room_id", ""),
        ("room_id", " RM_test_room"),
        ("worker_id", ""),
        ("worker_id", " AW_test_worker"),
    ),
)
async def test_network_admission_requires_canonical_server_authority_before_connecting(
    registry: ConfigRegistry,
    field_name: str,
    value: str,
) -> None:
    arguments = {
        "fake": False,
        "metadata": (
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
        field_name: value,
    }
    job = _JobContext(**arguments)

    with pytest.raises(TenantResolutionError, match="transport authority"):
        await _run_admission(  # type: ignore[arg-type]
            job,
            registry,
            development_merchant_id=None,
        )

    assert job.connect_count == 0


async def test_admission_preserves_shared_dispatch_and_room_across_assignments(
    registry: ConfigRegistry,
) -> None:
    metadata = (
        '{"schema_version":1,"merchant_id":"demo_shop",'
        '"participant_kind":"standard","participant_identity":"caller-primary"}'
    )
    first = await _run_admission(  # type: ignore[arg-type]
        _JobContext(
            fake=False,
            metadata=metadata,
            job_id="AJ_first",
            dispatch_id="AD_shared",
            room_id="RM_shared",
            worker_id="AW_first",
        ),
        registry,
        development_merchant_id=None,
    )
    second = await _run_admission(  # type: ignore[arg-type]
        _JobContext(
            fake=False,
            metadata=metadata,
            job_id="AJ_second",
            dispatch_id="AD_shared",
            room_id="RM_shared",
            worker_id="AW_second",
        ),
        registry,
        development_merchant_id=None,
    )

    assert isinstance(first, NetworkVoiceTenantAdmission)
    assert isinstance(second, NetworkVoiceTenantAdmission)
    assert first.session_authority.logical_session_id == second.session_authority.logical_session_id
    assert first.session_authority.transport.room_id == second.session_authority.transport.room_id
    assert first.session_authority.transport != second.session_authority.transport


async def test_unknown_dispatch_merchant_fails_before_connecting(
    registry: ConfigRegistry,
) -> None:
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"missing_store",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    with pytest.raises(TenantResolutionError, match="unknown merchant_id"):
        await _run_admission(  # type: ignore[arg-type]
            job,
            registry,
            development_merchant_id=None,
        )

    assert job.connect_count == 0


@pytest.mark.parametrize(
    "metadata",
    (
        "not-json",
        '{"merchant_id":"acme_store","participant_kind":"sip"}',
        '{"schema_version":"1","merchant_id":"acme_store","participant_kind":"sip"}',
        ('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip","extra":true}'),
        '{"schema_version":1,"merchant_id":"acme_store","participant_kind":"standard"}',
        '{"schema_version":1,"merchant_id":"acme_store","participant_kind":"connector"}',
        '{"schema_version":1,"merchant_id":" acme_store","participant_kind":"sip"}',
        (
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":" caller-primary"}'
        ),
    ),
)
async def test_invalid_dispatch_metadata_fails_before_connecting(
    registry: ConfigRegistry,
    metadata: str,
) -> None:
    job = _JobContext(fake=False, metadata=metadata)

    with pytest.raises(TenantResolutionError, match="metadata is invalid"):
        await _run_admission(  # type: ignore[arg-type]
            job,
            registry,
            development_merchant_id=None,
        )

    assert job.connect_count == 0


async def test_sip_and_dispatch_authorities_must_agree(registry: ConfigRegistry) -> None:
    job = _JobContext(
        fake=False,
        metadata=('{"schema_version":1,"merchant_id":"demo_shop","participant_kind":"sip"}'),
        participant_kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        participant_attributes={
            "sip.ruleID": "rule-acme",
            "sip.trunkPhoneNumber": "+15551230001",
        },
    )

    with pytest.raises(TenantResolutionError, match="different tenants"):
        await _run_admission(  # type: ignore[arg-type]
            job,
            registry,
            development_merchant_id=None,
        )


async def test_matching_sip_and_dispatch_authorities_are_accepted(
    registry: ConfigRegistry,
) -> None:
    admitted = await _run_admission(  # type: ignore[arg-type]
        _JobContext(
            fake=False,
            metadata=('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip"}'),
            participant_kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            participant_attributes={
                "sip.ruleID": "rule-acme",
                "sip.trunkPhoneNumber": "+15551230001",
            },
        ),
        registry,
        development_merchant_id=None,
    )

    assert isinstance(admitted, NetworkVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "acme_store"


async def test_sip_admission_normalizes_carrier_attribute_padding(
    registry: ConfigRegistry,
) -> None:
    admitted = await _run_admission(  # type: ignore[arg-type]
        _JobContext(
            fake=False,
            metadata=('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip"}'),
            participant_kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
            participant_attributes={
                "sip.ruleID": "  rule-acme  ",
                "sip.trunkPhoneNumber": "  +15551230001  ",
            },
        ),
        registry,
        development_merchant_id=None,
    )

    assert isinstance(admitted, NetworkVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "acme_store"


@pytest.mark.parametrize(
    "attributes, expected_error",
    (
        ({"sip.trunkPhoneNumber": "+15551230001"}, "dispatch rule"),
        ({"sip.ruleID": "rule-acme"}, "inbound trunk number"),
    ),
)
async def test_sip_admission_requires_server_dispatch_attributes(
    registry: ConfigRegistry,
    attributes: dict[str, str],
    expected_error: str,
) -> None:
    job = _JobContext(
        fake=False,
        metadata=('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip"}'),
        participant_kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        participant_attributes=attributes,
    )

    with pytest.raises(TenantResolutionError, match=expected_error):
        await _run_admission(  # type: ignore[arg-type]
            job,
            registry,
            development_merchant_id=None,
        )


async def test_sip_admission_ignores_an_unrelated_standard_participant(
    registry: ConfigRegistry,
) -> None:
    sip_participant = SimpleNamespace(
        identity="sip-caller",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        attributes={
            "sip.ruleID": "rule-acme",
            "sip.trunkPhoneNumber": "+15551230001",
        },
    )
    job = _JobContext(
        fake=False,
        metadata=('{"schema_version":1,"merchant_id":"acme_store","participant_kind":"sip"}'),
        participant_identity="unrelated-standard",
        additional_participants=(sip_participant,),
    )

    admitted = await _run_admission(  # type: ignore[arg-type]
        job,
        registry,
        development_merchant_id=None,
    )

    assert isinstance(admitted, NetworkVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "acme_store"
    assert admitted.participant_identity == "sip-caller"
    assert job.wait_kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP


async def test_standard_admission_binds_the_declared_participant_identity(
    registry: ConfigRegistry,
) -> None:
    intended_participant = SimpleNamespace(
        identity="caller-intended",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        attributes={},
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-intended"}'
        ),
        participant_identity="unrelated-standard",
        additional_participants=(intended_participant,),
    )

    admitted = await _run_admission(  # type: ignore[arg-type]
        job,
        registry,
        development_merchant_id=None,
    )

    assert isinstance(admitted, NetworkVoiceTenantAdmission)
    assert admitted.tenant.tenant_id == "demo_shop"
    assert admitted.participant_identity == "caller-intended"
    assert job.wait_identity == "caller-intended"
    assert job.wait_kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD


@pytest.mark.parametrize("blocked_stage", ("connect", "participant"))
async def test_voice_admission_times_out_within_the_configured_budget(
    registry: ConfigRegistry,
    blocked_stage: str,
) -> None:
    blocked = asyncio.Event()
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
        connect_event=blocked if blocked_stage == "connect" else None,
        wait_event=blocked if blocked_stage == "participant" else None,
    )
    boundary = VoiceJobAdmission(registry, development_merchant_id=None)
    preflight = boundary.preflight(job)  # type: ignore[arg-type]
    started_at = asyncio.get_running_loop().time()

    with pytest.raises(TenantResolutionError, match="voice admission timed out"):
        await asyncio.wait_for(
            boundary.complete(  # type: ignore[arg-type]
                job,
                preflight,
                timeout_seconds=0.01,
            ),
            timeout=0.25,
        )

    assert asyncio.get_running_loop().time() - started_at < 0.25
    assert job.connect_count == 1
    assert job.wait_count == (blocked_stage == "participant")


async def test_close_certification_participant_wait_is_bounded(
    registry: ConfigRegistry,
) -> None:
    from scripts import voice_agent

    job = _JobContext(fake=True, wait_event=asyncio.Event())
    admission = await _run_admission(
        job,
        registry,
        development_merchant_id="acme_store",
    )

    with pytest.raises(TenantResolutionError, match="close certification timed out"):
        await voice_agent._certification_participant_identity(  # type: ignore[arg-type]
            job,
            admission,
            timeout_seconds=0.01,
        )


async def test_worker_session_uses_the_admitted_participant_binding(
    registry: ConfigRegistry,
) -> None:
    from scripts import voice_agent

    intended_participant = SimpleNamespace(
        identity="caller-intended",
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        attributes={},
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-intended"}'
        ),
        participant_identity="unrelated-standard",
        additional_participants=(intended_participant,),
    )
    admission = await _run_admission(  # type: ignore[arg-type]
        job,
        registry,
        development_merchant_id=None,
    )

    class RecordingSession:
        participant_identity: str | None = None

        async def start(self, _agent, *, room, room_options) -> None:
            assert room is job.room
            self.participant_identity = room_options.participant_identity

    session = RecordingSession()
    loop = SimpleNamespace(session=session, agent=object())

    await voice_agent._start_admitted_session(  # type: ignore[arg-type]
        job,
        loop,
        admission,
    )

    assert session.participant_identity == "caller-intended"


def test_duplicate_inbound_did_is_rejected_when_the_admission_index_is_built(
    config_root: Path,
    tmp_path: Path,
) -> None:
    conflicting_root = tmp_path / "config"
    copytree(config_root, conflicting_root)
    demo_config_path = conflicting_root / "merchants" / "demo_shop.yaml"
    demo_config_path.write_text(
        demo_config_path.read_text(encoding="utf-8").replace(
            "+15551230002",
            "+15551230001",
        ),
        encoding="utf-8",
    )
    registry = ConfigRegistry(conflicting_root).load()

    with pytest.raises(TenantResolutionError, match="claimed by both"):
        TenantResolver(registry)


def test_registry_rejects_a_noncanonical_inbound_did_for_any_merchant(
    config_root: Path,
    tmp_path: Path,
) -> None:
    invalid_root = tmp_path / "config"
    copytree(config_root, invalid_root)
    demo_config_path = invalid_root / "merchants" / "demo_shop.yaml"
    demo_config_path.write_text(
        demo_config_path.read_text(encoding="utf-8").replace(
            'inbound_number: "+15551230002"',
            'inbound_number: " +15551230002"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigResolutionError, match="inbound_number"):
        ConfigRegistry(invalid_root).load()


def test_worker_requires_an_explicit_dispatch_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import voice_agent

    monkeypatch.delenv("VOICE_AGENT_NAME", raising=False)
    assert voice_agent._agent_name(()) == ""
    assert voice_agent._agent_name(("--help",)) == ""
    assert voice_agent._agent_name(("dev", "--help")) == ""
    assert voice_agent._agent_name(("console",)) == ""
    assert voice_agent._agent_name(("download-files",)) == ""
    with pytest.raises(RuntimeError, match="VOICE_AGENT_NAME"):
        voice_agent._agent_name(("dev",))

    monkeypatch.setenv("VOICE_AGENT_NAME", "  agnostic-market  ")
    assert voice_agent._agent_name(("dev",)) == "agnostic-market"


def test_worker_connection_is_owned_by_admission() -> None:
    import ast
    from inspect import getsource
    from textwrap import dedent

    from scripts import voice_agent

    tree = ast.parse(dedent(getsource(voice_agent.entrypoint)))
    direct_connects = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "connect"
    ]
    assert direct_connects == []


class _WorkerCloser:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def begin_close(self) -> None:
        self._events.append("durable_begin_close")

    async def finalize_close(self) -> None:
        self._events.append("durable_finalize_close")


class _WorkerDurableSession:
    def __init__(self, events: list[str]) -> None:
        self.checkpointer = object()
        self.closer = _WorkerCloser(events)


class _WorkerPlatformResources:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.config = SimpleNamespace(
            sessions=SimpleNamespace(transport_retirement_timeout_seconds=3.0)
        )
        self.session = _WorkerDurableSession(events)

    async def acquire_fresh_session(self, **kwargs):
        assert kwargs["tenant_id"] == "demo_shop"
        assert kwargs["deployment_id"] == "deployment-test"
        assert kwargs["admitted_authority"].logical_session_id == "AD_test_dispatch"
        self._events.append("lease_acquired")
        return self.session

    async def aclose(self) -> None:
        self._events.append("platform_closed")


class _WorkerLoop:
    def __init__(
        self,
        events: list[str],
        *,
        session_start_failure: BaseException | None = None,
    ) -> None:
        self._events = events
        self._session_start_failure = session_start_failure
        self.application = SimpleNamespace(state=SimpleNamespace(caller_context=SimpleNamespace()))
        self.agent = object()
        self.session = SimpleNamespace(start=self._start_session)
        self.background_audio = SimpleNamespace(
            start=self._start_background_audio,
            aclose=self._close_background_audio,
        )

    async def _start_session(self, _agent, *, room, room_options) -> None:
        assert room is not None
        assert room_options.participant_identity == "caller-primary"
        self._events.append("session_started")
        if self._session_start_failure is not None:
            raise self._session_start_failure

    async def _start_background_audio(self, *, room, agent_session) -> None:
        assert room is not None
        assert agent_session is self.session
        self._events.append("background_audio_started")

    async def _close_background_audio(self) -> None:
        self._events.append("background_audio_closed")

    def register_shutdown(self, job_context, *, release_job_resources=None) -> None:
        self._events.append("shutdown_registered")

        async def shutdown() -> None:
            await self.aclose(release_job_resources=release_job_resources)

        job_context.add_shutdown_callback(shutdown)

    def start_lease_supervision(
        self,
        room,
        *,
        transport_retirement_timeout_seconds: float,
    ) -> None:
        assert room is not None
        assert transport_retirement_timeout_seconds == 3.0
        self._events.append("lease_supervision_started")

    async def aclose(self, *, release_job_resources=None) -> None:
        self._events.append("loop_closed")
        await self.background_audio.aclose()
        if release_job_resources is not None:
            await release_job_resources()


def _patch_durable_worker_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    config_root: Path,
    platform_config_path: Path,
    events: list[str],
    resources: _WorkerPlatformResources,
    *,
    composition_failure: BaseException | None = None,
    session_start_failure: BaseException | None = None,
) -> None:
    from scripts import voice_agent

    prepared_routing = object()
    monkeypatch.setattr(voice_agent, "_CONFIG_ROOT", config_root)
    monkeypatch.setattr(voice_agent, "load_close_certification_request", lambda _root: None)
    monkeypatch.setattr(voice_agent, "require_llm_certification", lambda *_args: None)
    monkeypatch.setattr(
        voice_agent,
        "prepare_application_routing",
        lambda _factory: events.append("routing_prepared") or prepared_routing,
    )
    journey_corpus = object()

    def require_latency_evidence(
        methodology_path: Path,
        report_path: Path,
        *,
        expected_deployment_id: str,
        expected_journey_corpus: object,
        expected_runtime_contract_fingerprint: str,
        required_measurement_surface: voice_agent.LatencyMeasurementSurface,
    ) -> None:
        assert methodology_path == platform_config_path.with_name("latency-methodology.yaml")
        assert report_path == platform_config_path.with_name("latency-report.json")
        assert expected_deployment_id == "deployment-test"
        assert expected_journey_corpus is journey_corpus
        assert expected_runtime_contract_fingerprint == "a" * 64
        assert (
            required_measurement_surface is voice_agent.LatencyMeasurementSurface.VOICE_PROCESSING
        )
        events.append("latency_authorized")

    monkeypatch.setattr(
        voice_agent,
        "require_deployment_latency_evidence",
        require_latency_evidence,
        raising=False,
    )
    platform_config = SimpleNamespace(
        database=SimpleNamespace(
            application_dsn_ref=SimpleNamespace(uri="env://PLATFORM_POSTGRES_DSN")
        )
    )
    monkeypatch.setattr(
        voice_agent,
        "load_platform_runtime_config",
        lambda path: events.append("platform_config_loaded") or platform_config,
    )
    monkeypatch.setattr(
        voice_agent,
        "load_latency_journey_corpus",
        lambda _path: journey_corpus,
    )
    monkeypatch.setattr(
        voice_agent,
        "deployment_runtime_contract_fingerprint",
        lambda *_args, **_kwargs: "a" * 64,
    )

    async def open_resources(config, _secrets, *, application_dsn: str):
        assert config is platform_config
        assert application_dsn == "postgresql://runtime.invalid/platform"
        events.append("platform_opened")
        return resources

    monkeypatch.setattr(
        voice_agent.DurablePlatformResources,
        "open",
        staticmethod(open_resources),
    )

    def build_services(_root, tenant, *, telemetry, checkpointer):
        assert tenant.tenant_id == "demo_shop"
        assert telemetry.tenant_id == "demo_shop"
        assert checkpointer is resources.session.checkpointer
        events.append("tenant_services_built")
        return object()

    monkeypatch.setattr(voice_agent, "build_fixture_tenant_services", build_services)

    async def build_loop(*_args, **kwargs):
        assert kwargs["durable_session"] is resources.session
        assert kwargs["routing_recognizer_factory"] is prepared_routing
        events.append("voice_loop_build_started")
        if composition_failure is not None:
            raise composition_failure
        return _WorkerLoop(events, session_start_failure=session_start_failure)

    monkeypatch.setattr(voice_agent, "build_voice_loop", build_loop)
    monkeypatch.setenv("VOICE_AGENT_DEPLOYMENT_ID", "deployment-test")
    monkeypatch.setenv("PLATFORM_POSTGRES_DSN", "postgresql://runtime.invalid/platform")
    monkeypatch.setenv("VOICE_AGENT_PLATFORM_CONFIG", str(platform_config_path))
    monkeypatch.setenv(
        "VOICE_AGENT_LATENCY_METHODOLOGY",
        str(platform_config_path.with_name("latency-methodology.yaml")),
    )
    monkeypatch.setenv(
        "VOICE_AGENT_LATENCY_REPORT",
        str(platform_config_path.with_name("latency-report.json")),
    )


def test_network_worker_requires_an_absolute_platform_config_path(tmp_path: Path) -> None:
    from scripts import voice_agent

    with pytest.raises(RuntimeError, match="must identify"):
        voice_agent._platform_config_path({})
    with pytest.raises(RuntimeError, match="absolute path"):
        voice_agent._platform_config_path({"VOICE_AGENT_PLATFORM_CONFIG": "runtime.yaml"})

    platform_config_path = tmp_path / "platform.yaml"
    assert (
        voice_agent._platform_config_path(
            {"VOICE_AGENT_PLATFORM_CONFIG": f"  {platform_config_path}  "}
        )
        == platform_config_path
    )

    with pytest.raises(RuntimeError, match="VOICE_AGENT_LATENCY_METHODOLOGY"):
        voice_agent._latency_evidence_paths({})
    with pytest.raises(RuntimeError, match="must be an absolute path"):
        voice_agent._latency_evidence_paths(
            {
                "VOICE_AGENT_LATENCY_METHODOLOGY": "methodology.yaml",
                "VOICE_AGENT_LATENCY_REPORT": "report.json",
            }
        )
    methodology_path = tmp_path / "methodology.yaml"
    report_path = tmp_path / "report.json"
    assert voice_agent._latency_evidence_paths(
        {
            "VOICE_AGENT_LATENCY_METHODOLOGY": str(methodology_path),
            "VOICE_AGENT_LATENCY_REPORT": str(report_path),
        }
    ) == (methodology_path, report_path)


async def test_network_worker_acquires_durable_authority_before_runtime_composition(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    resources = _WorkerPlatformResources(events)
    _patch_durable_worker_dependencies(
        monkeypatch,
        config_root,
        tmp_path / "platform.yaml",
        events,
        resources,
    )

    class OrderedJob(_JobContext):
        async def connect(self) -> None:
            events.append("room_connected")
            await super().connect()

    job = OrderedJob(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    from scripts import voice_agent

    await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert events == [
        "routing_prepared",
        "platform_config_loaded",
        "latency_authorized",
        "platform_opened",
        "room_connected",
        "lease_acquired",
        "tenant_services_built",
        "voice_loop_build_started",
        "shutdown_registered",
        "lease_supervision_started",
        "session_started",
        "background_audio_started",
    ]
    assert len(job.shutdown_callbacks) == 1
    await job.shutdown_callbacks[0]()
    assert events[-3:] == ["loop_closed", "background_audio_closed", "platform_closed"]


async def test_network_worker_closes_durable_session_when_composition_fails(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    resources = _WorkerPlatformResources(events)
    _patch_durable_worker_dependencies(
        monkeypatch,
        config_root,
        tmp_path / "platform.yaml",
        events,
        resources,
        composition_failure=RuntimeError("voice composition failed"),
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    from scripts import voice_agent

    with pytest.raises(RuntimeError, match="voice composition failed"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert events[-3:] == [
        "durable_begin_close",
        "durable_finalize_close",
        "platform_closed",
    ]
    assert job.shutdown_callbacks == []


async def test_network_worker_rolls_back_resources_when_voice_start_fails(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    resources = _WorkerPlatformResources(events)
    _patch_durable_worker_dependencies(
        monkeypatch,
        config_root,
        tmp_path / "platform.yaml",
        events,
        resources,
        session_start_failure=RuntimeError("voice start failed"),
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    from scripts import voice_agent

    with pytest.raises(RuntimeError, match="voice start failed"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert events[-3:] == ["loop_closed", "background_audio_closed", "platform_closed"]
    assert len(job.shutdown_callbacks) == 1


async def test_worker_completes_startup_gates_before_connecting(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import voice_agent

    def reject_certification(*_args, **_kwargs) -> None:
        raise RuntimeError("certification rejected")

    monkeypatch.setattr(voice_agent, "_CONFIG_ROOT", config_root)
    monkeypatch.setattr(voice_agent, "require_llm_certification", reject_certification)
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    with pytest.raises(RuntimeError, match="certification rejected"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert job.connect_count == 0


async def test_worker_rejects_routing_qualification_before_platform_or_connect(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agnostic_market.agents.routing_activation import RoutingActivationError
    from scripts import voice_agent

    monkeypatch.setattr(voice_agent, "_CONFIG_ROOT", config_root)
    monkeypatch.setattr(voice_agent, "load_close_certification_request", lambda _root: None)
    monkeypatch.setattr(voice_agent, "require_llm_certification", lambda *_args: None)

    def reject_routing(_factory):
        raise RoutingActivationError("routing qualification rejected")

    monkeypatch.setattr(voice_agent, "prepare_application_routing", reject_routing)
    monkeypatch.setattr(
        voice_agent,
        "load_platform_runtime_config",
        lambda _path: pytest.fail("platform startup preceded routing qualification"),
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    with pytest.raises(RoutingActivationError, match="routing qualification rejected"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert job.connect_count == 0


async def test_worker_rejects_latency_evidence_before_platform_or_connect(
    config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import voice_agent

    events: list[str] = []
    resources = _WorkerPlatformResources(events)
    platform_config_path = tmp_path / "platform.yaml"
    _patch_durable_worker_dependencies(
        monkeypatch,
        config_root,
        platform_config_path,
        events,
        resources,
    )

    def reject_latency_evidence(*_args, **_kwargs) -> None:
        raise RuntimeError("deployment latency rejected")

    monkeypatch.setattr(
        voice_agent,
        "require_deployment_latency_evidence",
        reject_latency_evidence,
        raising=False,
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"demo_shop",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    with pytest.raises(RuntimeError, match="deployment latency rejected"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]

    assert "platform_opened" not in events
    assert job.connect_count == 0


async def test_worker_rejects_unknown_metadata_tenant_before_runtime_composition(
    config_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import voice_agent

    def runtime_composition_started(*_args, **_kwargs):
        raise AssertionError("runtime composition started before tenant admission")

    monkeypatch.setattr(voice_agent, "_CONFIG_ROOT", config_root)
    monkeypatch.setattr(
        voice_agent,
        "load_close_certification_request",
        runtime_composition_started,
    )
    job = _JobContext(
        fake=False,
        metadata=(
            '{"schema_version":1,"merchant_id":"missing_store",'
            '"participant_kind":"standard","participant_identity":"caller-primary"}'
        ),
    )

    with pytest.raises(TenantResolutionError, match="unknown merchant_id"):
        await voice_agent.entrypoint(job)  # type: ignore[arg-type]
