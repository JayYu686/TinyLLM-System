from __future__ import annotations

import json
from pathlib import Path

from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m4_qwen_config import load_m4_qwen_config


def test_m4_qwen_four_gpu_formal_evidence_is_strict_and_bounded() -> None:
    evidence = json.loads(
        Path("reports/m4/raw/fsdp2_qwen3_8b_formal.json").read_text(encoding="utf-8")
    )
    config = load_m4_qwen_config(Path(evidence["config"]["path"]))

    assert evidence["status"] == "pass"
    assert evidence["git"]["dirty"] is False
    assert evidence["config"]["sha256"] == canonical_config_hash(config)
    assert evidence["model"]["revision"] == config.model.revision
    assert evidence["model"]["parameter_count"] == 8_190_735_360
    assert evidence["data"]["view_sha256"] == (
        "5cef25622986f75fb882a3aa98decd710e3c3f6e22cd90021fecff22a4e0c2f9"
    )

    hardware = evidence["hardware"]
    assert hardware["physical_gpu_indices"] == [5, 6, 7, 8]
    assert hardware["numa_nodes"] == [1, 1, 1, 1]
    thresholds = hardware["preflight_thresholds"]
    assert all(
        row["memory_used_mib"] <= thresholds["memory_used_mib_lte"]
        and row["utilization_percent"] <= thresholds["utilization_percent_lte"]
        and row["temperature_c"] <= thresholds["temperature_c_lte"]
        for row in hardware["probe_preflight"]
    )

    probe = evidence["probe"]
    assert probe["status"] == "probe_succeeded"
    assert probe["global_step"] == 1
    assert probe["activation_checkpointed_layers"] == 36
    assert len(probe["peak_allocated_bytes_per_rank"]) == 4
    assert max(probe["peak_reserved_bytes_per_rank"]) < hardware["memory_total_bytes_per_gpu"]

    formal = evidence["formal_run"]
    assert formal["fresh_phase"]["status"] == "interrupted"
    assert formal["fresh_phase"]["global_step"] == 25
    assert formal["resume_phase"]["status"] == "succeeded"
    assert formal["resume_phase"]["resumed_from_step"] == 25
    assert formal["resume_phase"]["global_step"] == 50
    assert formal["metrics"]["global_steps"] == list(range(1, 51))
    assert formal["metrics"]["finite_loss"] is True
    assert formal["metrics"]["finite_gradient_norm"] is True
    assert formal["metrics"]["tokens_seen"] == 102_400
    assert formal["metrics"]["per_step_duration_collected"] is False

    checkpoints = evidence["checkpoints"]
    assert [item["pin_reason"] for item in checkpoints] == ["interruption", "final"]
    assert all(item["world_size"] == 4 for item in checkpoints)
    assert all(item["shard_count"] == 4 for item in checkpoints)
    assert all(item["rank_state_file_count"] == 4 for item in checkpoints)
    assert all(item["state_coverage_complete"] is True for item in checkpoints)

    export = evidence["export"]
    assert export["purpose"] == "deployment_export_not_training_checkpoint"
    assert export["tensor_count"] == 399
    assert export["independent_safetensors_validation"] is True
    assert export["independent_transformers_load"] is True
    assert export["loaded_parameter_count"] == evidence["model"]["parameter_count"]

    limitations = evidence["limitations"]
    assert all(value is False for value in limitations.values())
