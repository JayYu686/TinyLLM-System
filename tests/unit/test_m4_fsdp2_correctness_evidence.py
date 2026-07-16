from __future__ import annotations

import json
from pathlib import Path

from tinyllm.schemas import canonical_config_hash
from tinyllm.training.fsdp2_config import load_fsdp2_config
from tinyllm.training.fsdp2_schema import FSDP2CorrectnessSummary


def test_m4_fsdp2_cpu_correctness_evidence_is_bound_to_config_and_schema() -> None:
    evidence = json.loads(
        Path("reports/m4/raw/fsdp2_cpu_correctness.json").read_text(encoding="utf-8")
    )
    config = load_fsdp2_config(Path(evidence["run"]["config_path"]))
    summary = FSDP2CorrectnessSummary.model_validate_json(json.dumps(evidence["summary"]))

    assert evidence["status"] == "pass"
    assert evidence["run"]["config_sha256"] == canonical_config_hash(config)
    assert evidence["run"]["run_id"].split("-")[-2] == (evidence["run"]["config_sha256"][:8])
    assert evidence["run"]["git_dirty"] is False
    assert summary.backend == "gloo"
    assert summary.device_type == "cpu"
    assert summary.checkpoint_status == "not_evaluated_m4_1"
    assert summary.local_shard_parameter_sum == summary.logical_parameter_count
    assert evidence["environment"]["physical_gpu_indices"] == [None, None]

    mismatch = next(
        item for item in evidence["failure_paths"] if item["case"] == "torchrun_world_size_mismatch"
    )
    assert mismatch["status"] == "pass"
    assert mismatch["artifact_created"] is False
