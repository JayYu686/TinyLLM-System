from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.evaluation import BaselineEvaluationResult, load_baseline_config
from tinyllm.schemas import canonical_config_hash


def _load_evidence() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path("reports/m2/raw/baseline_formal.json").read_text(encoding="utf-8")),
    )


def test_formal_baseline_evidence_is_complete_real_and_path_free() -> None:
    evidence = _load_evidence()
    result = BaselineEvaluationResult.model_validate(evidence["result"])
    config = load_baseline_config(Path("configs/eval/m2_baseline.yaml"))

    assert evidence["status"] == "pass"
    assert evidence["scope"] == "m2-qwen3-0.6b-formal-pretraining-baseline"
    assert evidence["execution"]["exit_status"] == 0
    assert evidence["execution"]["lm_eval_validation_exit_status"] == 0
    assert evidence["execution"]["lm_eval_exit_status"] == 0
    assert evidence["execution"]["offline"] is True
    assert evidence["execution"]["model_evaluation_wall_seconds"] > 0
    assert evidence["code"]["git_dirty"] is False

    assert result.status == "succeeded"
    assert result.mode == "formal"
    assert result.config_sha256 == canonical_config_hash(config)
    assert result.model_revision == config.model.revision
    assert result.domain.status == "complete"
    assert result.domain.evaluated_items == config.domain.expected_items == 300
    assert result.domain.objective_items == 260
    assert result.domain.objective_correct == 16
    assert result.domain.json_items == 80
    assert result.domain.json_valid == 32
    assert result.domain.human_review_pending == 0
    assert result.domain.human_reviewed == 40
    assert result.domain.human_passed == 0

    tasks = {task.task: task for task in result.general.tasks}
    assert tasks["tinyllm_arc_easy"].samples == 2_376
    assert tasks["tinyllm_hellaswag"].samples == 10_042
    assert tasks["tinyllm_piqa"].samples == 1_838
    assert result.general.model_parameters == 596_049_920

    derived = evidence["derived_domain"]
    assert derived["overall_passed"] == 16
    assert derived["overall_items"] == 300
    assert derived["failed_items"] == 284
    assert len(derived["failed_item_ids"]) == len(set(derived["failed_item_ids"])) == 284

    artifacts = evidence["artifacts"]
    assert artifacts["private_raw_outputs_retained"] is True
    assert artifacts["public_raw_outputs"] is False
    assert artifacts["human_judgment_count"] == 40
    assert artifacts["general_sample_counts"] == {
        "tinyllm_arc_easy": 2_376,
        "tinyllm_hellaswag": 10_042,
        "tinyllm_piqa": 1_838,
    }
    assert all(
        len(value) == 64
        for value in (
            artifacts["domain_results_sha256"],
            artifacts["general_aggregate_sha256"],
            artifacts["baseline_summary_sha256"],
            artifacts["human_judgments_sha256"],
            *artifacts["general_sample_log_sha256"].values(),
        )
    )

    serialized = json.dumps(evidence, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "sitonholy" not in serialized.casefold()
