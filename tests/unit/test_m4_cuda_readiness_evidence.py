from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinyllm.schemas import canonical_config_hash
from tinyllm.training import load_fsdp2_config
from tinyllm.training.fsdp2_schema import FSDP2CorrectnessSummary
from tinyllm.training.m4_dependencies import M4DependencySmokeResult


def test_m4_cuda_readiness_evidence_preserves_partial_boundary() -> None:
    evidence = json.loads(
        Path("reports/m4/raw/fsdp2_cuda_readiness.json").read_text(encoding="utf-8")
    )
    dependency = M4DependencySmokeResult.model_validate_json(
        json.dumps(evidence["dependency_gate"]["result"])
    )
    single = evidence["cuda_gate"]["single_gpu"]
    summary = FSDP2CorrectnessSummary.model_validate_json(json.dumps(single["summary"]))
    config_path = Path(single["config_path"])
    config = load_fsdp2_config(config_path)

    assert evidence["status"] == "partial_pass"
    assert dependency.fixed_qwen3_8b_revision_verified is False
    assert dependency.remote_model_assets_loaded is False
    assert summary.backend == "nccl"
    assert summary.device_type == "cuda"
    assert summary.world_size == 1
    assert single["config_sha256"] == canonical_config_hash(config)
    assert evidence["cuda_gate"]["two_gpu"]["status"] == "not_run"
    assert evidence["cuda_gate"]["two_gpu"]["training_process_started"] is False
    assert "two-or-more-GPU NCCL collectives in the M4 FSDP2 runtime" in evidence["not_evaluated"]

    constraints = Path("requirements/constraints/m4.txt").read_bytes()
    assert (
        evidence["dependency_gate"]["constraints_sha256"] == hashlib.sha256(constraints).hexdigest()
    )
