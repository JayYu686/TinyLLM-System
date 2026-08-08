from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m5_config import load_m5_sft_config


def _evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m5/raw/m5_full_sft_formal.json").read_text(encoding="utf-8")),
    )


def test_m5_full_sft_formal_evidence_is_complete_and_path_free() -> None:
    evidence = _evidence()
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_0_6b.yaml"))
    model = evidence["model"]
    data = evidence["data"]
    training = evidence["training"]
    checkpoints = evidence["checkpoints"]
    evaluation = evidence["evaluation"]
    export = evidence["export"]

    assert evidence["status"] == "pass"
    assert model["repository"] == config.model.repository
    assert model["revision"] == config.model.revision
    assert model["attention_architecture"] == "gqa"
    assert model["adaptation"] == "full_sft"
    assert data["dataset_version"] == config.data.dataset_version
    assert data["manifest_sha256"] == config.data.mix_manifest_sha256
    assert training["config_sha256"] == canonical_config_hash(config)
    assert training["git_dirty"] is False
    assert training["world_size"] == 4
    assert training["physical_gpu_indices"] == [5, 6, 8, 9]
    assert training["supervised_tokens"] == 50_000_000
    assert training["interruption_tokens"] == training["resumed_from_tokens"]
    assert training["peak_reserved_bytes_per_rank"] >= training["peak_allocated_bytes_per_rank"]
    assert training["thermal_pause_count"] == 25
    assert len(training["metrics_sha256"]) == 64

    assert [item.get("target_tokens") for item in checkpoints[1:]] == [
        10_000_000,
        20_000_000,
        30_000_000,
        40_000_000,
        50_000_000,
    ]
    assert [item["supervised_tokens"] for item in checkpoints[1:]] == [
        10_000_532,
        20_001_758,
        30_002_588,
        40_004_805,
        50_000_000,
    ]
    assert all(len(item["payload_sha256"]) == 64 for item in checkpoints)

    stages = evaluation["stages"]
    assert [stage["thinking_score_basis_points"] for stage in stages] == [
        9_500,
        9_250,
        9_200,
        9_100,
        9_150,
    ]
    assert [stage["nonthinking_score_basis_points"] for stage in stages] == [
        4_750,
        4_350,
        4_300,
        4_000,
        3_900,
    ]
    assert evaluation["selected_development_checkpoint"] == "checkpoint-tokens-0010000532"
    assert export["purpose"] == "deployment_export_not_training_checkpoint"
    assert len(export["tree_sha256"]) == 64
    assert len(export["model_safetensors_sha256"]) == 64
    assert all(value is False for value in evidence["limitations"].values())

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()


def test_m5_acceptance_preserves_m6_release_boundary() -> None:
    report = Path("reports/m5/m5_acceptance.md").read_text(encoding="utf-8")

    assert "M5 的代码、真实训练、恢复、失败路径、双模式评测、部署导出和血缘证据已经齐备" in report
    assert "均保持 `Development`" in report
    assert "M5 的完成不会自动授予 Candidate 或\nProduction 状态" in report
