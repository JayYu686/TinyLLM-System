from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentRunRequest,
    AgentRunStore,
    AgentToolCall,
    DevOpsToolError,
    DevOpsTools,
    rebuild_evidence_index,
)

NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _tools(tmp_path: Path) -> tuple[DevOpsTools, Path, Path]:
    project = tmp_path / "project"
    artifact = tmp_path / "artifacts"
    (project / "configs").mkdir(parents=True)
    (project / "docs").mkdir()
    (artifact / "runs" / "run-1").mkdir(parents=True)
    (artifact / "evaluations" / "eval-1").mkdir(parents=True)
    (artifact / "registry").mkdir()
    (project / "README.md").write_text("# TinyLLM\n", encoding="utf-8")
    (project / "docs" / "recovery.md").write_text(
        "# Recovery\n\nBackend crash recovered.\n", encoding="utf-8"
    )
    (project / "configs" / "train.yaml").write_text(
        yaml.safe_dump({"learning_rate": 0.001, "api_token": "do-not-return"}),
        encoding="utf-8",
    )
    (artifact / "runs" / "run-1" / "run.json").write_text(
        json.dumps({"run_id": "run-1", "status": "failed", "prompt": "private"}),
        encoding="utf-8",
    )
    (artifact / "runs" / "run-1" / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "loss": 2.0, "token": "secret"}) + "\n",
        encoding="utf-8",
    )
    (artifact / "evaluations" / "eval-1" / "worker.log").write_text(
        "line one\nline two\nline three\n", encoding="utf-8"
    )
    index = tmp_path / "index"
    rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=index)
    tools = DevOpsTools(project_root=project, artifact_root=artifact, index_dir=index)
    return tools, project, artifact


def test_devops_read_tools_return_bounded_grounded_data(tmp_path: Path) -> None:
    tools, _project, _artifact = _tools(tmp_path)

    assert tools.search_evidence("backend crash")["results"]
    assert tools.list_runs(status="failed")["runs"] == [{"run_id": "run-1", "status": "failed"}]
    assert "prompt" not in tools.get_run("run-1")["run"]
    excerpt = tools.read_log_excerpt("evaluations/eval-1/worker.log", 2, 3)
    assert excerpt["excerpt"] == "line two\nline three"
    metrics = tools.query_metrics("runs/run-1/metrics.jsonl", ["step", "loss"])
    assert metrics["records"] == [{"step": 1, "loss": 2.0}]
    config = tools.inspect_config("configs/train.yaml")
    assert config["config"]["api_token"] == "[REDACTED]"


def test_devops_read_tools_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    tools, _project, artifact = _tools(tmp_path)
    outside = tmp_path / "outside.log"
    outside.write_text("secret", encoding="utf-8")
    (artifact / "evaluations" / "escape.log").symlink_to(outside)

    with pytest.raises(DevOpsToolError, match="allowed roots"):
        tools.read_log_excerpt("../outside.log")
    with pytest.raises(DevOpsToolError, match="symlink"):
        tools.read_log_excerpt("evaluations/escape.log")
    with pytest.raises(DevOpsToolError, match="metrics path"):
        tools.query_metrics("evaluations/eval-1/worker.log")


def test_sandbox_patch_requires_approval_and_is_single_use(tmp_path: Path) -> None:
    tools, _project, artifact = _tools(tmp_path)
    store = AgentRunStore(artifact)
    request = AgentRunRequest.model_validate({"messages": [{"role": "user", "content": "x"}]})
    record, _ = store.create(request, idempotency_key="agent-tool-create-0001", now=NOW)
    approval_id = "approval-123456abcdef"
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id=approval_id,
        pending_tool_call=AgentToolCall(
            call_id="call_patch_1",
            server_id="tinyllm-devops",
            tool_name="apply_sandbox_config_patch",
            arguments={},
        ),
        now=NOW + timedelta(seconds=1),
    )

    with pytest.raises(DevOpsToolError, match="approval"):
        tools.apply_sandbox_config_patch(
            record.run_id, approval_id, "configs/train.yaml", {"learning_rate": 0.002}
        )
    store.decide_approval(
        record.run_id,
        AgentApprovalDecision(
            approval_id=approval_id,
            decision="approved",
            idempotency_key="agent-tool-approval-0001",
            decided_at=NOW + timedelta(seconds=2),
        ),
    )
    result = tools.apply_sandbox_config_patch(
        record.run_id, approval_id, "configs/train.yaml", {"learning_rate": 0.002}
    )
    target = artifact / str(result["relative_path"])
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["learning_rate"] == 0.002
    with pytest.raises(DevOpsToolError, match="already been consumed"):
        tools.apply_sandbox_config_patch(
            record.run_id, approval_id, "configs/train.yaml", {"learning_rate": 0.003}
        )
