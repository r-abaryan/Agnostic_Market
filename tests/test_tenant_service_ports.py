"""Tenant-service replacement contracts at application and graph boundaries."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

from agnostic_market.agents.cart.flow import build_cart_nodes
from agnostic_market.agents.frontline.graph import build_frontline_graph
from agnostic_market.agents.frontline.read_flow import build_read_flow_nodes
from agnostic_market.agents.identity.flow import build_identity_nodes
from agnostic_market.agents.recovery import build_recovery_node
from agnostic_market.agents.support._stepup import build_stepup_nodes
from agnostic_market.agents.support.flow import build_support_nodes
from agnostic_market.commerce.catalog import CatalogPort
from agnostic_market.commerce.identity import CustomerDirectoryPort
from agnostic_market.commerce.orders import OrderPort
from agnostic_market.commerce.payment_instruments import PaymentInstrumentPort
from agnostic_market.commerce.profile import ProfilePort
from agnostic_market.commerce.verification import OtpPort, RiskPort, VerificationStore


def _consumer_contracts() -> dict[object, dict[str, object]]:
    return {
        build_frontline_graph: {
            "store": OrderPort,
            "catalog": CatalogPort,
            "customers": CustomerDirectoryPort,
            "payment_instruments": PaymentInstrumentPort,
            "profile_store": ProfilePort,
            "risk": RiskPort,
        },
        build_cart_nodes: {"order_store": OrderPort, "catalog": CatalogPort},
        build_read_flow_nodes: {
            "order_store": OrderPort,
            "catalog": CatalogPort,
            "customers": CustomerDirectoryPort,
        },
        build_support_nodes: {
            "order_store": OrderPort,
            "payment_instruments": PaymentInstrumentPort,
            "profile_store": ProfilePort,
            "risk": RiskPort,
        },
        build_identity_nodes: {
            "customers": CustomerDirectoryPort,
            "risk": RiskPort,
        },
        build_stepup_nodes: {"risk": RiskPort},
        build_recovery_node: {"order_store": OrderPort, "profile_store": ProfilePort},
        VerificationStore.__init__: {"otp": OtpPort},
    }


def test_graph_consumers_depend_on_tenant_service_ports() -> None:
    for consumer, expected in _consumer_contracts().items():
        annotations = get_type_hints(consumer)
        assert {name: annotations[name] for name in expected} == expected


def test_remote_verification_ports_are_native_async() -> None:
    for method in (OtpPort.dispatch, OtpPort.verify, OtpPort.retain_only, RiskPort.assess):
        assert inspect.iscoroutinefunction(method), method.__qualname__


def test_port_contract_typecheck_covers_declared_consumers() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_path = repository_root / "pyright.port-contracts.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    included = {Path(path).as_posix() for path in config["include"]}
    consumer_sources = {
        Path(source).resolve().relative_to(repository_root).as_posix()
        for consumer in _consumer_contracts()
        if (source := inspect.getsourcefile(consumer)) is not None
    }

    assert consumer_sources <= included


def test_port_consumers_pass_static_protocol_check() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pyright",
            "--project",
            str(repository_root / "pyright.port-contracts.json"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
