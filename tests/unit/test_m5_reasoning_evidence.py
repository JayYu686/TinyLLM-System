from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tinyllm.data import (
    M5FormatRepairMixtureManifest,
    M5R3SourceAudit,
    M5TeacherPilotResult,
    M5TeacherSmokeResult,
)
from tinyllm.evaluation.m5_r2_schema import M5R2DiagnosticDecision
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


def test_m5_r2_design_keeps_diagnostic_and_formal_evaluation_separate() -> None:
    design = Path("docs/m5_r2_diagnostic_design.md").read_text(encoding="utf-8")

    assert "R2 不训练新模型" in design
    assert "不降低 99% Thinking 格式门禁" in design
    assert "896 重放的 Response SHA256" in design
    assert "前 896 个生成 Token ID" in design
    assert "1024、1280、1536" in design
    assert "重新运行 Base、六个 M5.2 Candidate 和两个 R1 Candidate" in design
    assert "诊断不允许通过字符串补写 `</think>`" in design


def test_m5_r2_public_decision_retains_real_length_rejection() -> None:
    path = Path("reports/m5/raw/m5_r2_length_diagnostic.json")
    decision = M5R2DiagnosticDecision.model_validate_json(path.read_text(encoding="utf-8"))

    assert decision.status == "length_ceiling_insufficient"
    assert decision.selected_max_new_tokens is None
    assert decision.training_seeds == (42, 20260727)
    assert decision.formal_protocol_changed is False
    assert tuple(item.projected_format_basis_points for item in decision.projections) == (
        9800,
        9650,
    )
    assert tuple(item.unresolved_format_items for item in decision.projections) == (4, 7)
    public_text = path.read_text(encoding="utf-8")
    assert "response" not in public_text
    assert "item_id" not in public_text
    assert "/home/" not in public_text
    assert "/data/" not in public_text


def test_m5_r2_report_records_completed_gpu_replay_without_protocol_change() -> None:
    report = Path("reports/m5/m5_r2_diagnostic.md").read_text(encoding="utf-8")

    assert "`COMPLETED_DIAGNOSTIC_REJECTED`" in report
    assert "40 / 40" in report
    assert "36 / 36" in report
    assert "98.0%" in report
    assert "96.5%" in report
    assert "formal_protocol_changed: false" in report


def test_m5_r3_public_source_audit_requires_new_teacher_data() -> None:
    path = Path("reports/m5/raw/m5_r3_source_audit.json")
    result = M5R3SourceAudit.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.status == "insufficient_requires_new_source"
    assert result.eligible_source_items == 6
    assert result.required_source_items == 160
    assert result.new_teacher_source_required is True
    assert tuple(item.eligible_items for item in result.family_audits) == (2, 4)
    assert result.family_audits[0].eligible_language_counts == {"en": 2, "zh": 0}
    public = path.read_text(encoding="utf-8")
    assert "reasoning_content" not in public
    assert "prompt" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_design_keeps_one_variable_and_frozen_evaluation() -> None:
    design = Path("docs/m5_r3_targeted_repair_design.md").read_text(encoding="utf-8")
    report = Path("reports/m5/m5_r3_targeted_repair.md").read_text(encoding="utf-8")

    assert "700,000" in design
    assert "150,000" in design
    assert "最大 896 New Tokens" in design
    assert "可见推理不超过 192 Token" in design
    assert "同一来源最多出现四次" in design
    assert "P0 通过前不实现 240 条扩展，不启动 R3 训练" in design
    assert "`SOURCE_AUDIT_REJECTED_NEW_TEACHER_REQUIRED`" in report
    assert "CPU Fixture 为合成契约 Smoke" in report
    assert "R3-P0" in report


def test_m5_r3_p0_report_does_not_claim_gpu_results_before_execution() -> None:
    report = Path("reports/m5/m5_r3_p0.md").read_text(encoding="utf-8")

    assert "`IMPLEMENTED_AWAITING_REAL_TEACHER_PILOT`" in report
    assert "尚未产生真实 Qwen3-8B Teacher 结果" in report
    assert "`model_generated=false`" in report
    assert "`quality_metric=false`" in report
