from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tinyllm.data import (
    M5FormatRepairMixtureManifest,
    M5TeacherPilotResult,
    M5TeacherSmokeResult,
)
from tinyllm.evaluation.m5_reasoning_schema import (
    M5AblationSelection,
    M5FormatFailureAnalysis,
    M5FormatRepairGateResult,
)


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


def test_m5_2_teacher_pilot_retains_failed_and_passing_protocols() -> None:
    failed = M5TeacherPilotResult.model_validate_json(
        Path("reports/m5/raw/teacher_pilot_100_placeholder_failure.json").read_text(
            encoding="utf-8"
        )
    )
    passed = M5TeacherPilotResult.model_validate_json(
        Path("reports/m5/raw/teacher_pilot_100.json").read_text(encoding="utf-8")
    )

    assert failed.status == "fail"
    assert failed.accepted_samples == 37
    assert set(failed.accepted_task_family_counts) == {"json", "python"}
    assert passed.status == "pass"
    assert passed.accepted_samples == 96
    assert set(passed.accepted_task_family_counts) == {
        "config",
        "json",
        "linux",
        "log_diagnosis",
        "python",
    }
    for name in ("teacher_pilot_100_placeholder_failure.json", "teacher_pilot_100.json"):
        text = (Path("reports/m5/raw") / name).read_text(encoding="utf-8")
        assert "raw_output" not in text
        assert "/home/" not in text
        assert "/data/" not in text


def test_m5_reasoning_report_keeps_smoke_and_quality_claims_separate() -> None:
    report = Path("reports/m5/m5_reasoning_data.md").read_text(encoding="utf-8")
    assert "M5 整体仍为 `IN_PROGRESS`" in report
    assert "不声称模型质量提升" in report
    assert "CPU 合成 Fixture 当作模型输出" in report
    assert "M5.2" in report


def test_m5_2_public_selection_retains_the_rejected_gate_result() -> None:
    path = Path("reports/m5/raw/m5_ablation_selection.json")
    selection = M5AblationSelection.model_validate_json(path.read_text(encoding="utf-8"))

    assert selection.status == "no_eligible_arm"
    assert selection.selected_thinking_fraction_basis_points is None
    assert selection.selection_reason == "no_arm_passed_preregistered_gates"
    assert selection.base_nonthinking_score_basis_points == 3700
    assert [arm.thinking_fraction_basis_points for arm in selection.arms] == [
        0,
        3000,
        5000,
    ]
    assert all(arm.nonthinking_regression_gate_passed for arm in selection.arms)
    assert all(not arm.thinking_format_gate_passed for arm in selection.arms)
    assert selection.arms[1].thinking_format_basis_points == (9550, 9700)
    assert selection.arms[1].mean_thinking_score_basis_points == 9425

    public_text = path.read_text(encoding="utf-8")
    assert "/home/" not in public_text
    assert "/data/" not in public_text


def test_m5_r1_public_failure_analysis_is_redacted_and_reproducible() -> None:
    path = Path("reports/m5/raw/m5_format_failure_analysis.json")
    analysis = M5FormatFailureAnalysis.model_validate_json(path.read_text(encoding="utf-8"))

    assert analysis.total_invalid_format_items == 38
    assert analysis.total_length_open_without_close_items == 35
    assert analysis.total_eos_open_without_close_items == 3
    assert analysis.total_open_without_close_items == 38
    assert sum(item.task_family_counts["config"] for item in analysis.slices) == 26
    public_text = path.read_text(encoding="utf-8")
    assert "response" not in public_text
    assert "item_id" not in public_text
    assert "/home/" not in public_text
    assert "/data/" not in public_text


def test_m5_r1_public_mixture_manifest_records_exact_three_strata() -> None:
    path = Path("reports/m5/raw/m5_format_repair_mixture.json")
    manifest = M5FormatRepairMixtureManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert manifest.mixture_version == "m5-format-repair-mixture-v1-1396b60b"
    assert manifest.nonthinking_supervised_tokens == 700_000
    assert manifest.general_thinking_supervised_tokens == 150_000
    assert manifest.repair_thinking_supervised_tokens == 150_000
    assert manifest.repair_source_family_counts == {
        "config": 8,
        "json": 8,
        "linux": 8,
        "log_diagnosis": 8,
        "python": 8,
    }
    public_text = path.read_text(encoding="utf-8")
    assert "/home/" not in public_text
    assert "/data/" not in public_text


def test_m5_r1_public_gate_retains_real_rejection() -> None:
    path = Path("reports/m5/raw/m5_format_repair_gate.json")
    gate = M5FormatRepairGateResult.model_validate_json(path.read_text(encoding="utf-8"))

    assert gate.status == "rejected"
    assert gate.gate_reason == "thinking_format_gate_failed"
    assert gate.training_seeds == (42, 20260727)
    assert gate.nonthinking_scores_basis_points == (6400, 6600)
    assert gate.thinking_format_basis_points == (9450, 9350)
    assert gate.thinking_scores_basis_points == (9300, 9300)
    assert gate.nonthinking_regression_gate_passed
    assert not gate.thinking_format_gate_passed
    public_text = path.read_text(encoding="utf-8")
    assert "/home/" not in public_text
    assert "/data/" not in public_text
