from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_m10_lora_stage_evaluation import NOW, _record
from tinyllm.agent_eval.finalize import AgentEvalFinalizeError, finalize_agent_evaluation
from tinyllm.agent_eval.schema import (
    AgentEvalItemResult,
    AgentEvalRunConfig,
    canonical_json_sha256,
)
from tinyllm.agent_eval.suite import load_suite
from tinyllm.deployment import publish_m10_lora_stage_evaluation_subject


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: object) -> bytes:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def test_finalize_recovers_complete_persisted_evaluation_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    record = _record(root, stage_tokens=5_000_000)
    _, record_sha256 = publish_m10_lora_stage_evaluation_subject(root, record)
    suite_root = Path("evals/agent/dev/v1")
    manifest, tasks = load_suite(suite_root)
    output = (tmp_path / "evaluation").resolve()
    output.mkdir()

    config = AgentEvalRunConfig(
        config_id="m10-agent-eval-finalize-unit",
        scoring_protocol="m10-agent-scoring-v3",
        gateway_base_url="http://127.0.0.1:18080",
        bearer_token_env="TINYLLM_FINALIZE_TOKEN",
        model=record.subject_id,
        max_concurrency=2,
        physical_gpu_index=4,
        seed=20260901,
    )
    environment = {
        "tinyllm": "1.0.0rc1",
        "gateway": {"version": "1.0.0rc1"},
    }
    hardware = {
        "name": "NVIDIA GeForce RTX 3090",
        "driver_version": "535.261.03",
    }
    config_payload = _write_json(output / "config.resolved.json", config.to_dict())
    environment_payload = _write_json(output / "environment.json", environment)
    hardware_payload = _write_json(output / "hardware.json", hardware)
    _write_json(output / "suite.manifest.json", manifest.to_dict())
    metadata = {
        "suite_version": manifest.suite_version,
        "suite_content_sha256": manifest.content_sha256,
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "environment_sha256": hashlib.sha256(environment_payload).hexdigest(),
        "hardware_sha256": hashlib.sha256(hardware_payload).hexdigest(),
        "model_id": record.subject_id,
        "model_artifact_sha256": record.effective_artifact_sha256,
        "evaluation_subject_sha256": record_sha256,
        "git_commit": "f" * 40,
    }
    _write_json(output / "evaluation.metadata.json", metadata)
    items_root = output / "items"
    items_root.mkdir()
    for index, task in enumerate(tasks):
        result = AgentEvalItemResult(
            scoring_protocol=config.scoring_protocol,
            task_id=task.task_id,
            cluster_id=task.cluster_id,
            category=task.category,
            language=task.language,
            run_id=f"finalize-unit-{index:03d}",
            status="succeeded",
            final_answer="evidence-grounded answer",
            duration_milliseconds=10,
            input_tokens=10,
            output_tokens=5,
            tool_selection_correct=True,
            argument_correct=True,
            schema_valid=True,
            task_success=True,
            tool_hallucination=False,
        )
        _write_json(items_root / f"{task.task_id}.json", result.to_dict())

    monkeypatch.setattr(
        "tinyllm.agent_eval.finalize.read_git_identity",
        lambda _root: ("e" * 40, False),
    )
    finalized_at = datetime(2026, 9, 1, 5, 0, tzinfo=UTC)
    summary = finalize_agent_evaluation(
        suite_directory=suite_root,
        output_directory=output,
        artifact_root=root,
        project_root=tmp_path.resolve(),
        finalized_at=finalized_at,
    )

    assert config_payload
    assert summary.completed is True
    assert summary.evaluated_at == finalized_at
    assert summary.model_id == record.subject_id
    assert summary.evaluation_subject_sha256 == record_sha256
    assert summary.metrics.item_count == manifest.item_count == 80
    assert (output / "items.jsonl").is_file()
    assert (output / "aggregation-recovery.json").is_file()
    assert (output / "summary.json").is_file()
    recovery = json.loads((output / "aggregation-recovery.json").read_text(encoding="utf-8"))
    assert recovery["model_generation_repeated"] is False
    assert recovery["persisted_item_count"] == 80
    with pytest.raises(AgentEvalFinalizeError, match="terminal aggregate evidence"):
        finalize_agent_evaluation(
            suite_directory=suite_root,
            output_directory=output,
            artifact_root=root,
            project_root=tmp_path.resolve(),
            finalized_at=NOW,
        )
