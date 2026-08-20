from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentRunRequest,
    AgentRunStore,
    AgentToolCall,
    DevOpsToolError,
    DevOpsTools,
    agent_tool_call_sha256,
    rebuild_evidence_index,
)
from tinyllm.agent.devops_tools import MAX_DOCUMENT_BYTES, _json_size_and_depth

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
    run = cast(dict[str, Any], tools.get_run("run-1")["run"])
    assert "prompt" not in run
    excerpt = tools.read_log_excerpt("evaluations/eval-1/worker.log", 2, 3)
    assert excerpt["excerpt"] == "line two\nline three"
    metrics = tools.query_metrics("runs/run-1/metrics.jsonl", ["step", "loss"])
    assert metrics["records"] == [{"step": 1, "loss": 2.0}]
    config = tools.inspect_config("configs/train.yaml")
    config_value = cast(dict[str, Any], config["config"])
    assert config_value["api_token"] == "[REDACTED]"


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
    pending_call = AgentToolCall(
        call_id="call_patch_1",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={
            "source_relative_path": "configs/train.yaml",
            "updates": {"learning_rate": 0.002},
        },
    )
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id=approval_id,
        pending_tool_call=pending_call,
        now=NOW + timedelta(seconds=1),
    )

    with pytest.raises(DevOpsToolError, match="approval"):
        tools.apply_sandbox_config_patch(
            record.run_id,
            approval_id,
            "call_patch_1",
            "configs/train.yaml",
            {"learning_rate": 0.002},
        )
    store.decide_approval(
        record.run_id,
        AgentApprovalDecision(
            approval_id=approval_id,
            tool_call_sha256=agent_tool_call_sha256(pending_call),
            decision="approved",
            idempotency_key="agent-tool-approval-0001",
            decided_at=NOW + timedelta(seconds=2),
        ),
    )
    result = tools.apply_sandbox_config_patch(
        record.run_id,
        approval_id,
        "call_patch_1",
        "configs/train.yaml",
        {"learning_rate": 0.002},
    )
    target = artifact / str(result["relative_path"])
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["learning_rate"] == 0.002
    repeated = tools.apply_sandbox_config_patch(
        record.run_id,
        approval_id,
        "call_patch_1",
        "configs/train.yaml",
        {"learning_rate": 0.002},
    )
    assert repeated == result
    with pytest.raises(DevOpsToolError, match="differs from the approved"):
        tools.apply_sandbox_config_patch(
            record.run_id,
            approval_id,
            "call_patch_1",
            "configs/train.yaml",
            {"learning_rate": 0.003},
        )


def test_devops_tool_roots_must_be_absolute_real_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact = tmp_path / "artifact"
    index = tmp_path / "index"
    project.mkdir()
    artifact.mkdir()
    index.mkdir()
    with pytest.raises(DevOpsToolError, match="absolute"):
        DevOpsTools(project_root=Path("relative"), artifact_root=artifact, index_dir=index)
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)
    with pytest.raises(DevOpsToolError, match="non-symlink"):
        DevOpsTools(project_root=alias, artifact_root=artifact, index_dir=index)


def test_devops_read_limits_and_invalid_artifacts_are_rejected(tmp_path: Path) -> None:
    tools, project, artifact = _tools(tmp_path)
    with pytest.raises(DevOpsToolError, match="list limit"):
        tools.list_runs(limit=0)
    with pytest.raises(DevOpsToolError, match="Run ID"):
        tools.get_run("../run")
    with pytest.raises(DevOpsToolError, match="missing or ambiguous"):
        tools.get_run("missing-run")
    with pytest.raises(DevOpsToolError, match="line range"):
        tools.read_log_excerpt("evaluations/eval-1/worker.log", 3, 2)

    (artifact / "evaluations" / "eval-1" / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DevOpsToolError, match="text log"):
        tools.read_log_excerpt("evaluations/eval-1/summary.json")
    with pytest.raises(DevOpsToolError, match="record limit"):
        tools.query_metrics("runs/run-1/metrics.jsonl", limit=0)
    with pytest.raises(DevOpsToolError, match="metric name"):
        tools.query_metrics("runs/run-1/metrics.jsonl", ["../loss"])

    invalid = artifact / "runs" / "run-1" / "summary.json"
    invalid.write_text("{broken", encoding="utf-8")
    with pytest.raises(DevOpsToolError, match="invalid JSON"):
        tools.query_metrics("runs/run-1/summary.json")

    oversized = artifact / "evaluations" / "eval-1" / "large.log"
    oversized.write_text("x" * (MAX_DOCUMENT_BYTES + 1), encoding="utf-8")
    with pytest.raises(DevOpsToolError, match="inspection limit"):
        tools.read_log_excerpt("evaluations/eval-1/large.log")

    binary = artifact / "evaluations" / "eval-1" / "binary.log"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(DevOpsToolError, match="cannot be read"):
        tools.read_log_excerpt("evaluations/eval-1/binary.log")

    (project / "configs" / "broken.yaml").write_text("a: [", encoding="utf-8")
    with pytest.raises(DevOpsToolError, match="invalid"):
        tools.inspect_config("configs/broken.yaml")
    with pytest.raises(DevOpsToolError, match="outside"):
        tools.inspect_config("docs/recovery.md")


def test_devops_metrics_scrub_secrets_and_bound_result_size(tmp_path: Path) -> None:
    tools, _project, artifact = _tools(tmp_path)
    summary = artifact / "runs" / "run-1" / "summary.json"
    summary.write_text(
        json.dumps({"loss": 1.0, "credentials": [{"api_key": "hidden"}]}),
        encoding="utf-8",
    )
    records = cast(list[dict[str, Any]], tools.query_metrics("runs/run-1/summary.json")["records"])
    assert records[0]["credentials"] == [{"api_key": "[REDACTED]"}]

    summary.write_text(json.dumps({"value": "x" * 40_000}), encoding="utf-8")
    with pytest.raises(DevOpsToolError, match="output limit"):
        tools.query_metrics("runs/run-1/summary.json")


def test_devops_structural_complexity_limit() -> None:
    with pytest.raises(DevOpsToolError, match="too complex"):
        _json_size_and_depth([{} for _ in range(4097)])


@pytest.mark.parametrize(
    ("run_id", "source", "updates", "pattern"),
    [
        ("bad-run", "configs/train.yaml", {"learning_rate": 2e-4}, "Run ID"),
        (
            "agent-20260813T120000Z-1234abcd-beef",
            "configs/train.json",
            {"learning_rate": 2e-4},
            "YAML",
        ),
        (
            "agent-20260813T120000Z-1234abcd-beef",
            "configs/train.yaml",
            {},
            "keys",
        ),
        (
            "agent-20260813T120000Z-1234abcd-beef",
            "configs/train.yaml",
            {"api_token": "secret"},
            "secrets",
        ),
    ],
)
def test_sandbox_patch_rejects_invalid_identity_source_or_updates(
    tmp_path: Path,
    run_id: str,
    source: str,
    updates: dict[str, Any],
    pattern: str,
) -> None:
    tools, project, _artifact = _tools(tmp_path)
    (project / "configs" / "train.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DevOpsToolError, match=pattern):
        tools.apply_sandbox_config_patch(
            run_id,
            "approval-123456abcdef",
            "call_patch_1",
            source,
            updates,
        )


def test_sandbox_patch_rejects_rejected_approval(tmp_path: Path) -> None:
    tools, _project, artifact = _tools(tmp_path)
    store = AgentRunStore(artifact)
    request = AgentRunRequest.model_validate({"messages": [{"role": "user", "content": "x"}]})
    record, _ = store.create(request, idempotency_key="agent-tool-create-0002", now=NOW)
    call = AgentToolCall(
        call_id="call_patch_rejected",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={"source_relative_path": "configs/train.yaml", "updates": {"x": 1}},
    )
    approval_id = "approval-abcdef123456"
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id=approval_id,
        pending_tool_call=call,
        now=NOW + timedelta(seconds=1),
    )
    store.decide_approval(
        record.run_id,
        AgentApprovalDecision(
            approval_id=approval_id,
            tool_call_sha256=agent_tool_call_sha256(call),
            decision="rejected",
            idempotency_key="agent-tool-approval-0002",
            decided_at=NOW + timedelta(seconds=2),
        ),
    )
    with pytest.raises(DevOpsToolError, match="was not approved"):
        tools.apply_sandbox_config_patch(
            record.run_id,
            approval_id,
            call.call_id,
            "configs/train.yaml",
            {"x": 1},
        )
