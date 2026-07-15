from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.schemas import canonical_config_hash
from tinyllm.training import DDPCorrectnessSummary, load_training_config


def _load_evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m3/raw/ddp_correctness.json").read_text(encoding="utf-8")),
    )


def test_m3_ddp_correctness_evidence_is_real_complete_and_path_free() -> None:
    evidence = _load_evidence()
    one_gpu = evidence["runs"]["one_gpu"]
    two_gpu = evidence["runs"]["two_gpu"]
    one_summary = DDPCorrectnessSummary.model_validate_json(json.dumps(one_gpu["summary"]))
    two_summary = DDPCorrectnessSummary.model_validate_json(json.dumps(two_gpu["summary"]))

    one_config = load_training_config(
        Path("configs/pretrain/tinygpt_debug_ddp_1gpu_bf16_smoke.yaml")
    )
    two_config = load_training_config(
        Path("configs/pretrain/tinygpt_debug_ddp_2gpu_bf16_smoke.yaml")
    )

    assert evidence["status"] == "pass"
    assert evidence["code"]["git_dirty"] is False
    assert one_gpu["config_sha256"] == canonical_config_hash(one_config)
    assert two_gpu["config_sha256"] == canonical_config_hash(two_config)
    assert one_summary.world_size == 1
    assert two_summary.world_size == 2
    assert one_summary.global_batch_size == two_summary.global_batch_size == 8
    assert one_summary.durable_metric_records == one_summary.optimizer_steps == 8
    assert two_summary.durable_metric_records == two_summary.optimizer_steps == 8
    assert one_summary.initial_parameter_sha256 == two_summary.initial_parameter_sha256
    assert one_summary.sampler_union_samples == one_summary.sampler_num_samples == 256
    assert two_summary.sampler_union_samples == two_summary.sampler_num_samples == 256
    assert [item.sample_count for item in two_summary.partitions] == [128, 128]
    assert one_summary.max_loss_reduction_abs_diff == 0
    assert two_summary.max_loss_reduction_abs_diff == 0
    assert two_summary.max_gradient_norm_abs_diff == 0
    assert evidence["private_artifacts"]["stderr_bytes"] == {"one_gpu": 0, "two_gpu": 0}
    private_artifacts = evidence["private_artifacts"]
    artifact_hashes = [
        private_artifacts["doctor_report_sha256"],
        *private_artifacts["correctness_sha256"].values(),
        *private_artifacts["summary_sha256"].values(),
        *private_artifacts["stderr_sha256"].values(),
        *private_artifacts["stdout_sha256"].values(),
    ]
    assert all(len(value) == 64 for value in artifact_hashes)

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()
