from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from tinyllm.evaluation import ContaminationReport, EvaluationSetManifest


def _load_json(path: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def test_formal_domain_contamination_evidence_matches_frozen_inputs() -> None:
    evidence = _load_json("reports/m2/raw/domain_eval_contamination.json")
    evaluation = EvaluationSetManifest.model_validate_json(
        Path("evals/domain/v1/manifest.json").read_text(encoding="utf-8")
    )
    dataset_evidence = _load_json("reports/m2/raw/full_dataset_build.json")["dataset"]
    report = ContaminationReport.model_validate(evidence["report"])

    assert evidence["status"] == "pass"
    assert evidence["scope"] == "formal-m2-domain-v1-exact-train-contamination"
    assert evidence["code"] == {
        "git_commit": "c944cdb633c4d13f2183c82b418b33e0c1364ef6",
        "git_dirty": False,
    }
    assert evidence["execution"]["exit_status"] == 0
    assert evidence["execution"]["wall_seconds"] > 0
    assert evidence["execution"]["max_rss_kib"] > 0

    assert report.status == "clean"
    assert report.evaluation_suite_version == evaluation.suite_version
    assert report.evaluation_content_sha256 == evaluation.content_sha256
    assert report.checked_evaluation_items == evaluation.item_count == 300
    assert report.dataset_version == dataset_evidence["dataset_version"]
    assert report.dataset_content_sha256 == dataset_evidence["content_sha256"]
    assert report.checked_training_samples == dataset_evidence["split_sample_counts"]["train"]
    assert report.contaminated_items == 0
    assert report.full_sequence_matches == 0
    assert report.prompt_prefix_matches == 0
    assert report.matches == ()
    assert report.near_dedup == "not_evaluated"
