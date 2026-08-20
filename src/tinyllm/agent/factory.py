"""Composition root for the M8 Agent API and local MCP authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter

from tinyllm.agent.api import AgentExecutionService, create_agent_router
from tinyllm.agent.config import load_agent_config
from tinyllm.agent.mcp_client import MCPPolicyClient
from tinyllm.agent.model import GatewayAgentModel
from tinyllm.agent.runtime import AgentRuntime
from tinyllm.agent.store import AgentRunStore


@dataclass(frozen=True, slots=True)
class AgentAPIComponents:
    """Objects whose lifetimes are owned by the Model Gateway."""

    router: APIRouter
    service: AgentExecutionService


def build_agent_api(
    *,
    config_path: Path,
    artifact_root: Path,
    project_root: Path,
    evidence_index: Path,
    gateway_base_url: str,
    bearer_token: str,
    model: str = "production",
) -> AgentAPIComponents:
    """Build one bounded Agent Runtime from administrator-owned paths and YAML."""

    config = load_agent_config(config_path)
    clients = {
        server.server_id: MCPPolicyClient(
            server,
            artifact_root=artifact_root,
            project_root=project_root,
            evidence_index=evidence_index,
            retry_delays_ms=config.read_retry_delays_ms,
        )
        for server in config.mcp_servers
    }
    store = AgentRunStore(artifact_root)
    agent_model = GatewayAgentModel(
        base_url=gateway_base_url,
        bearer_token=bearer_token,
        model=model,
        clients=clients,
        timeout_seconds=config.run_timeout_seconds,
    )
    runtime = AgentRuntime(
        config=config,
        store=store,
        model=agent_model,
        clients=clients,
    )
    service = AgentExecutionService(
        store=store,
        runtime=runtime,
        run_timeout_seconds=config.run_timeout_seconds,
    )
    router = create_agent_router(
        store=store,
        service=service,
        bearer_token=bearer_token,
        expected_model=model,
    )
    return AgentAPIComponents(router=router, service=service)
