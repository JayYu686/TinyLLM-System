from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.evaluation import BaselineEvaluationResult, load_baseline_config
from tinyllm.schemas import canonical_config_hash


def _load_evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m2/raw/baseline_smoke.json").read_text(encoding="utf-8")),
    )


def test_baseline_smoke_evidence_is_real_bounded_and_path_free() -> None:
    evidence = _load_evidence()
    result = BaselineEvaluationResult.model_validate(evidence["result"])
    config = load_baseline_config(Path("configs/eval/m2_baseline_smoke.yaml"))

    assert evidence["status"] == "pass"
    assert evidence["scope"] == "m2-qwen3-0.6b-baseline-smoke"
    assert evidence["execution"]["exit_status"] == 0
    assert evidence["execution"]["lm_eval_validation_exit_status"] == 0
    assert evidence["execution"]["lm_eval_exit_status"] == 0
    assert evidence["execution"]["offline"] is True
    assert evidence["execution"]["wall_seconds"] > 0
    assert evidence["code"]["git_dirty"] is True
    assert evidence["environment"]["pip_freeze_entries"] > 0

    assert result.status == "succeeded"
    assert result.mode == "smoke"
    assert result.config_sha256 == canonical_config_hash(config)
    assert result.model_revision == config.model.revision
    assert result.domain.evaluated_items == config.domain.limit == 2
    assert result.domain.json_valid == result.domain.json_items == 2
    assert result.domain.objective_correct == 1
    assert all(task.samples == config.general.limit == 2 for task in result.general.tasks)
    assert result.general.model_parameters == 596_049_920
    assert evidence["environment"]["transformers"] == config.software.transformers
    assert evidence["environment"]["tokenizers"] == config.software.tokenizers

    assert evidence["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert evidence["hardware"]["physical_gpu_index"] == 5
    assert evidence["hardware"]["start_utilization_percent"] == 0
    assert evidence["artifacts"]["private_raw_outputs_retained"] is True
    assert evidence["artifacts"]["public_raw_outputs"] is False
    assert evidence["artifacts"]["domain_response_hashes_verified"] is True
    assert len(evidence["artifacts"]["domain_results_sha256"]) == 64
    assert len(evidence["artifacts"]["general_aggregate_sha256"]) == 64

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()
