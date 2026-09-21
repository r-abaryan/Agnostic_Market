"""Run the local merchant administration API on loopback."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from agnostic_market.agents.routing_activation import ConfiguredSemanticRouterFactory
from agnostic_market.application import ApplicationModels, RoutingFactory
from agnostic_market.dtos.config import MerchantConfig
from agnostic_market.llm.gateway import LLMGateway, load_provider_credentials
from agnostic_market.management.api import create_management_app
from agnostic_market.management.repository import SqliteMerchantConfigurationRepository
from agnostic_market.management.service import MerchantManagementService
from agnostic_market.management.simulation import PublishedMerchantSimulator
from agnostic_market.secrets.env_resolver import EnvSecretResolver

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=_CONFIG_ROOT,
        help="base configuration used to resolve and validate merchant drafts",
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="local SQLite management database",
    )
    parser.add_argument(
        "--actor-id",
        required=True,
        help="development operator identity recorded in management audit events",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help="loopback TCP port",
    )
    return parser.parse_args()


def serve(
    *,
    config_root: Path,
    database: Path,
    actor_id: str,
    port: int,
) -> None:
    """Build the local service and bind it only to IPv4 loopback."""

    repository = SqliteMerchantConfigurationRepository(
        database,
        active_config_root=config_root,
    )
    service = MerchantManagementService(config_root, repository)
    credentials = load_provider_credentials(config_root / "base" / "providers.yaml")
    secrets = EnvSecretResolver()
    gateway = LLMGateway(credentials, secrets)

    def models_factory(config: MerchantConfig) -> ApplicationModels:
        return ApplicationModels(
            response=gateway.chat_model(config.llm.response),
            reasoning=gateway.chat_model(config.llm.reasoning),
            response_structured_output_method=gateway.structured_output_method(config.llm.response),
        )

    def routing_factory(config: MerchantConfig) -> RoutingFactory:
        return ConfiguredSemanticRouterFactory(
            selection=config.llm.routing,
            credentials=credentials,
            secrets=secrets,
            structured_output_method=gateway.structured_output_method(config.llm.routing),
            timeout_seconds=config.runtime.semantic_router_timeout_seconds,
            input_max_chars=config.runtime.semantic_router_input_max_chars,
        )

    simulator = PublishedMerchantSimulator(
        service,
        models_factory=models_factory,
        routing_factory=routing_factory,
    )
    app = create_management_app(
        service,
        development_actor_id=actor_id,
        simulator=simulator,
    )
    uvicorn.run(app, host="127.0.0.1", port=port)


def main() -> None:
    load_dotenv()
    arguments = _arguments()
    serve(
        config_root=arguments.config_root,
        database=arguments.database,
        actor_id=arguments.actor_id,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
