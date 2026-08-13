"""stdio reference MCP Server exposing the bounded TinyLLM DevOps tool set."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from tinyllm.agent.devops_tools import DevOpsTools


def create_server(tools: DevOpsTools) -> FastMCP:
    """Create the reference server; MCP annotations remain informational only."""

    server = FastMCP("tinyllm-devops", instructions="Return evidence-grounded DevOps facts.")

    @server.tool(structured_output=True)
    def search_evidence(query: str, top_k: int = 8) -> dict[str, Any]:
        """Search the immutable FTS5 evidence index."""

        return tools.search_evidence(query, top_k)

    @server.tool(structured_output=True)
    def list_runs(limit: int = 20, status: str | None = None) -> dict[str, Any]:
        """List content-minimized training Run records."""

        return tools.list_runs(limit, status)

    @server.tool(structured_output=True)
    def get_run(run_id: str) -> dict[str, Any]:
        """Read an allowlisted training Run record."""

        return tools.get_run(run_id)

    @server.tool(structured_output=True)
    def read_log_excerpt(
        relative_path: str, start_line: int = 1, end_line: int = 100
    ) -> dict[str, Any]:
        """Read a bounded excerpt from an Artifact Store text log."""

        return tools.read_log_excerpt(relative_path, start_line, end_line)

    @server.tool(structured_output=True)
    def query_metrics(
        relative_path: str, metric_names: list[str] | None = None, limit: int = 50
    ) -> dict[str, Any]:
        """Read selected metrics from a known metrics artifact."""

        return tools.query_metrics(relative_path, metric_names, limit)

    @server.tool(structured_output=True)
    def inspect_config(relative_path: str) -> dict[str, Any]:
        """Inspect and redact a project or Run configuration."""

        return tools.inspect_config(relative_path)

    @server.tool(structured_output=True)
    def apply_sandbox_config_patch(
        run_id: str,
        approval_id: str,
        source_relative_path: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply an approved patch to an Agent-owned sandbox copy."""

        return tools.apply_sandbox_config_patch(run_id, approval_id, source_relative_path, updates)

    return server


def main() -> None:
    artifact = os.environ.get("TINYLLM_ARTIFACT_ROOT", "")
    project = os.environ.get("TINYLLM_PROJECT_ROOT", "")
    index = os.environ.get("TINYLLM_EVIDENCE_INDEX", "")
    if not artifact or not project or not index:
        raise RuntimeError("reference MCP Server roots must be provided through the environment")
    tools = DevOpsTools(
        project_root=Path(project), artifact_root=Path(artifact), index_dir=Path(index)
    )
    create_server(tools).run(transport="stdio")


if __name__ == "__main__":
    main()
