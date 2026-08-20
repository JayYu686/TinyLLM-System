from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tinyllm.agent import load_agent_config, rebuild_evidence_index
from tinyllm.agent.mcp_client import MCPPolicyClient


def test_reference_stdio_mcp_server_is_policy_checked(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact = tmp_path / "artifacts"
    (project / "docs").mkdir(parents=True)
    (artifact / "runs" / "run-1").mkdir(parents=True)
    (artifact / "registry").mkdir()
    (project / "README.md").write_text("# TinyLLM\n", encoding="utf-8")
    (project / "docs" / "failure.md").write_text(
        "# Failure\n\nReadiness recovers after a backend failure.\n", encoding="utf-8"
    )
    (artifact / "runs" / "run-1" / "run.json").write_text(
        json.dumps({"run_id": "run-1", "status": "succeeded"}), encoding="utf-8"
    )
    index = tmp_path / "index"
    rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=index)
    server = load_agent_config(Path("configs/agent/m8_devops.yaml")).mcp_servers[0]
    client = MCPPolicyClient(
        server,
        artifact_root=artifact,
        project_root=project,
        evidence_index=index,
        retry_delays_ms=(1, 1),
    )

    result = asyncio.run(client.call("search_evidence", {"query": "backend failure", "top_k": 3}))
    assert result["schema_version"] == "1.0"
    assert result["results"]
    definitions = asyncio.run(client.discover_tools())
    patch = next(item for item in definitions if item.tool_name == "apply_sandbox_config_patch")
    properties = patch.input_schema["properties"]
    assert "run_id" not in properties
    assert "approval_id" not in properties
    assert "call_id" not in properties
    assert {"source_relative_path", "updates"}.issubset(properties)
