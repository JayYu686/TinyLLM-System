from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tinyllm.data import M5TeacherSmokeResult


def _json(path: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(Path(path).read_text(encoding="utf-8")))


def test_m5_reasoning_cpu_evidence_is_explicitly_synthetic_and_clean() -> None:
    evidence = _json("reports/m5/raw/reasoning_data_smoke.json")
    assert evidence["evidence_kind"] == "synthetic_cpu_contract_smoke"
    assert evidence["model_generated"] is False
    assert evidence["quality_metric"] is False
    dev = cast(dict[str, object], evidence["dev_manifest"])
    assert dev["task_set_version"] == "m5-reasoning-dev-v1-3eb153c2"
    assert dev["task_count"] == 200
    contamination = cast(dict[str, object], evidence["contamination_report"])
    assert contamination["status"] == "pass"
    assert contamination["exact_prompt_matches"] == 0
    assert contamination["template_family_overlaps"] == 0
    assert contamination["matches"] == []


def test_m5_real_teacher_smoke_has_clean_lineage_and_retains_prior_failure() -> None:
    passed = M5TeacherSmokeResult.model_validate_json(
        Path("reports/m5/raw/teacher_offline_smoke.json").read_text(encoding="utf-8")
    )
    failed = M5TeacherSmokeResult.model_validate_json(
        Path("reports/m5/raw/teacher_offline_smoke_512_failure.json").read_text(encoding="utf-8")
    )

    assert passed.status == "pass"
    assert passed.git_dirty is False
    assert passed.git_commit == "5289e6e003360d06c962689d64f6c6606c75d311"
    assert passed.model.attention_architecture == "gqa"
    assert passed.accepted_samples == 1
    assert passed.dataset_version == "m5-reasoning-pilot-v1-f551031f"
    assert failed.status == "fail"
    assert failed.accepted_samples == 0
    assert failed.rejection_counts == {
        "no_candidate_passed": 1,
        "teacher_length_limit": 2,
    }


def test_m5_public_teacher_evidence_contains_no_raw_reasoning_text() -> None:
    for name in (
        "teacher_offline_smoke.json",
        "teacher_offline_smoke_512_failure.json",
        "teacher_offline_smoke_pre_contamination.json",
    ):
        text = (Path("reports/m5/raw") / name).read_text(encoding="utf-8")
        assert "raw_output" not in text
        assert "reasoning_content" not in text
        assert "/home/" not in text
        assert "/data/" not in text


def test_m5_reasoning_report_keeps_smoke_and_quality_claims_separate() -> None:
    report = Path("reports/m5/m5_reasoning_data.md").read_text(encoding="utf-8")
    assert "M5 整体仍为 `IN_PROGRESS`" in report
    assert "不声称模型质量提升" in report
    assert "CPU 合成 Fixture 当作模型输出" in report
    assert "M5.2" in report
