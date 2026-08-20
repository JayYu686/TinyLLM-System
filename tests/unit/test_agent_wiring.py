from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tinyllm.agent import DevOpsTools, load_agent_config
from tinyllm.agent import devops_server as server_module
from tinyllm.agent import factory as factory_module
from tinyllm.agent.devops_server import create_server
from tinyllm.agent.factory import build_agent_api


class _Tools:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: {"name": name, "args": args, "kwargs": kwargs}


def _call(server: Any, name: str, **arguments: object) -> dict[str, object]:
    tool = server._tool_manager.get_tool(name)
    assert tool is not None
    return cast(dict[str, object], tool.fn(**arguments))


def test_reference_mcp_server_delegates_every_bounded_tool() -> None:
    server = create_server(cast(DevOpsTools, _Tools()))

    assert _call(server, "search_evidence", query="failure")["name"] == "search_evidence"
    assert _call(server, "list_runs")["name"] == "list_runs"
    assert _call(server, "get_run", run_id="run-1")["name"] == "get_run"
    assert (
        _call(server, "read_log_excerpt", relative_path="runs/run-1/a.log")["name"]
        == "read_log_excerpt"
    )
    assert (
        _call(server, "query_metrics", relative_path="runs/run-1/metrics.jsonl")["name"]
        == "query_metrics"
    )
    assert _call(server, "inspect_config", relative_path="configs/a.yaml")["name"] == (
        "inspect_config"
    )
    assert (
        _call(
            server,
            "apply_sandbox_config_patch",
            run_id="run",
            approval_id="approval",
            call_id="call",
            source_relative_path="configs/a.yaml",
            updates={"seed": 42},
        )["name"]
        == "apply_sandbox_config_patch"
    )


def test_reference_server_main_requires_admin_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TINYLLM_ARTIFACT_ROOT",
        "TINYLLM_PROJECT_ROOT",
        "TINYLLM_EVIDENCE_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="roots"):
        server_module.main()


def test_reference_server_main_constructs_and_runs_stdio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = {name: tmp_path / name.lower() for name in ("ARTIFACT", "PROJECT", "INDEX")}
    for path in roots.values():
        path.mkdir()
    monkeypatch.setenv("TINYLLM_ARTIFACT_ROOT", str(roots["ARTIFACT"]))
    monkeypatch.setenv("TINYLLM_PROJECT_ROOT", str(roots["PROJECT"]))
    monkeypatch.setenv("TINYLLM_EVIDENCE_INDEX", str(roots["INDEX"]))
    called: list[str] = []

    class _Server:
        def run(self, *, transport: str) -> None:
            called.append(transport)

    monkeypatch.setattr(server_module, "create_server", lambda _tools: _Server())
    server_module.main()
    assert called == ["stdio"]


def test_factory_composes_allowlisted_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    artifact = tmp_path / "artifact"
    index = tmp_path / "index"
    for path in (project, artifact, index):
        path.mkdir()
    config = load_agent_config(Path("configs/agent/m8_devops.yaml"))
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, server: object, **kwargs: object) -> None:
            self.server = server
            captured["client"] = kwargs

    class _Model:
        def __init__(self, **kwargs: object) -> None:
            captured["model"] = kwargs

    class _Runtime:
        def __init__(self, **kwargs: object) -> None:
            self.model = kwargs["model"]
            captured["runtime"] = kwargs

    class _Service:
        def __init__(self, **kwargs: object) -> None:
            captured["service"] = self
            captured["service_options"] = kwargs

    sentinel_router = cast(Any, object())
    monkeypatch.setattr(factory_module, "load_agent_config", lambda _path: config)
    monkeypatch.setattr(factory_module, "MCPPolicyClient", _Client)
    monkeypatch.setattr(factory_module, "GatewayAgentModel", _Model)
    monkeypatch.setattr(factory_module, "AgentRuntime", _Runtime)
    monkeypatch.setattr(factory_module, "AgentExecutionService", _Service)
    monkeypatch.setattr(factory_module, "create_agent_router", lambda **_kwargs: sentinel_router)

    components = build_agent_api(
        config_path=Path("agent.yaml"),
        artifact_root=artifact,
        project_root=project,
        evidence_index=index,
        gateway_base_url="http://127.0.0.1:8000",
        bearer_token="x" * 32,
    )

    assert components.router is sentinel_router
    assert components.service is captured["service"]
    runtime_options = cast(dict[str, Any], captured["runtime"])
    assert set(cast(dict[str, object], runtime_options["clients"])) == {"tinyllm-devops"}
