from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m5_config import load_m5_sft_config


def _evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m5/raw/m5_lora_formal.json").read_text(encoding="utf-8")),
    )


def test_m5_lora_formal_evidence_is_complete_and_path_free() -> None:
    evidence = _evidence()
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_8b_lora.yaml"))
    model = evidence["model"]
    training = evidence["training"]
    export = evidence["export"]
    evaluation = evidence["evaluation"]

    assert evidence["status"] == "pass"
    assert model["revision"] == config.model.revision
    assert model["attention_architecture"] == "gqa"
    assert model["adaptation"] == "lora"
    assert model["trainable_parameters"] < model["total_parameters"]
    assert training["config_sha256"] == canonical_config_hash(config)
    assert training["git_dirty"] is False
    assert training["supervised_tokens"] == 10_000_000
    assert training["completed_dataset_epochs"] == 10.0
    assert training["interruption_tokens"] == training["resumed_from_tokens"]
    assert training["interruption_tokens"] >= 5_000_000
    assert len(training["evaluation_checkpoints"]) == 5
    assert training["evaluation_checkpoints"][-1] == "checkpoint-tokens-0010000000"
    assert training["peak_reserved_bytes"] >= training["peak_allocated_bytes"]
    assert training["thermal_pause_count"] == 16
    assert export["model_card_present"] is True
    assert export["base_weights_redistributed"] is False
    assert len(export["adapter_tree_sha256"]) == 64
    assert len(export["adapter_safetensors_sha256"]) == 64

    assert evaluation["protocol_version"] == "m5-thinking-budget-v2"
    assert evaluation["shared_gpu_evaluation"] is True
    assert evaluation["preflight_utilization_percent"] == 0
    assert evaluation["thinking"]["evaluated_items"] == 200
    assert evaluation["thinking"]["format_valid_basis_points"] == 10_000
    assert evaluation["thinking"]["final_answer_score_basis_points"] == 9_900
    assert evaluation["thinking"]["natural_close_basis_points"] == 9_950
    assert evaluation["thinking"]["forced_close_basis_points"] == 50
    assert evaluation["nonthinking"]["evaluated_items"] == 200
    assert evaluation["nonthinking"]["format_valid_basis_points"] == 10_000
    assert evaluation["nonthinking"]["final_answer_score_basis_points"] == 7_200
    assert evaluation["thinking"]["visible_reasoning_leakage_items"] == 0
    assert evaluation["nonthinking"]["visible_reasoning_leakage_items"] == 0
    assert len(evaluation["summary_sha256"]) == 64
    assert len(evaluation["raw_results_sha256"]) == 64

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()


def test_m5_lora_formal_report_preserves_release_boundary() -> None:
    report = Path("reports/m5/m5_lora_formal.md").read_text(encoding="utf-8")

    assert "M5 Dev 没有 Qwen3-8B 同协议 Base" in report
    assert "不宣称相对 Base 提升" in report
    assert "不授予\nCandidate 或 Production 状态" in report
