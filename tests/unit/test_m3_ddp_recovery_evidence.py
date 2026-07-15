from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.schemas import canonical_config_hash
from tinyllm.training import load_training_config


def _load_evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m3/raw/ddp_recovery.json").read_text(encoding="utf-8")),
    )


def test_m3_ddp_recovery_evidence_is_real_complete_and_path_free() -> None:
    evidence = _load_evidence()
    config = load_training_config(
        Path("configs/pretrain/tinygpt_debug_ddp_recovery_2gpu_bf16_smoke.yaml")
    )
    runs = evidence["runs"]
    coordinated = runs["coordinated_recovery"]
    rank_failure = runs["rank_failure_recovery"]

    assert evidence["status"] == "pass"
    assert evidence["code"]["git_dirty"] is False
    assert evidence["config"]["config_sha256"] == canonical_config_hash(config)
    assert evidence["config"]["world_size"] == 2
    assert evidence["config"]["global_batch_size"] == 8
    assert evidence["config"]["optimizer_steps"] == 12
    assert evidence["tolerance"] == {
        "baseline_max_abs_loss_diff": 0.0,
        "final_parameter_hash_must_match": True,
        "frozen_before_recovery": True,
        "loss_atol": 1e-6,
        "rule": "max(1e-6, 2 * baseline_max_abs_loss_diff)",
    }
    expected_steps = list(range(1, 13))
    assert coordinated["resume_from_step"] == 6
    assert coordinated["canonical_metric_steps"] == expected_steps
    assert rank_failure["resume_from_step"] == rank_failure["failure_checkpoint_step"] == 8
    assert rank_failure["failure_rank"] == 1
    assert rank_failure["forced_exit_code"] == 17
    assert rank_failure["canonical_metric_steps"] == expected_steps
    hashes = {
        runs["baseline_a"]["final_parameter_sha256"],
        runs["baseline_b"]["final_parameter_sha256"],
        coordinated["final_parameter_sha256"],
        rank_failure["final_parameter_sha256"],
    }
    assert len(hashes) == 1
    assert coordinated["max_abs_loss_diff_from_baseline"] == 0
    assert rank_failure["max_abs_loss_diff_from_baseline"] == 0

    private = evidence["private_artifacts"]
    artifact_hashes = [
        private["doctor_report_sha256"],
        private["summary_sha256"],
        private["tolerance_sha256"],
        *private["checkpoint_manifest_sha256"].values(),
        *private["checkpoint_commit_marker_sha256"].values(),
    ]
    assert all(len(value) == 64 for value in artifact_hashes)
    assert private["stderr_bytes"]["rank_failure"] > 0
    assert private["stderr_bytes"]["rank_failure_resume"] == 0

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()
