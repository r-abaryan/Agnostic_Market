"""Trusted tenant admission for console, SIP, and explicit-dispatch voice jobs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest
from livekit import rtc

from agnostic_market.config.registry import ConfigRegistry
from agnostic_market.config.resolver import ConfigResolutionError
from agnostic_market.tenancy.resolver import TenantResolutionError, TenantResolver
from agnostic_market.voice.admission import VoiceJobAdmission, VoiceTenantAdmission

_ADMISSION_TIMEOUT_SECONDS = 1.0


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
        participant_kind: int = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
        participant_attributes: dict[str, str] | None = None,
        participant_identity: str = "caller-primary",
        additional_participants: tuple[SimpleNamespace, ...] = (),
        connect_event: asyncio.Event | None = None,
        wait_event: asyncio.Event | None = None,
    ) -> None:
        self._fake = fake
        self.job = SimpleNamespace(metadata=metadata)
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
    assert admitted.tenant.tenant_id == "acme_store"
    assert admitted.tenant.config_version == admitted.resolved.config_version
    assert admitted.participant_identity is None
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

    assert admitted.tenant.tenant_id == "acme_store"
    assert admitted.participant_identity == "caller-primary"
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
