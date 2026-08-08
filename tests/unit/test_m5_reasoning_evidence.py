from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from tinyllm.data import (
    M5FormatRepairMixtureManifest,
    M5R3ContentReviewResult,
    M5R3P0Result,
    M5R3P1CPUSmoke,
    M5R3P1Result,
    M5R3P2CPUSmoke,
    M5R3P2Result,
    M5R3SourceAudit,
    M5R3TeacherSourceStrategyReview,
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
    assert "该诊断通过前不实现" in design
    assert "240 条扩展，不启动 R3 训练" in design
    assert "`SOURCE_AUDIT_REJECTED_NEW_TEACHER_REQUIRED`" in report
    assert "CPU Fixture 为合成契约 Smoke" in report
    assert "R3-P0" in report


def test_m5_r3_p0_real_result_retains_rejected_gate() -> None:
    path = Path("reports/m5/raw/m5_r3_p0.json")
    result = M5R3P0Result.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.status == "fail"
    assert result.git_dirty is False
    assert result.accepted_samples == 10
    assert result.rejected_tasks == 30
    assert [item.accepted_items for item in result.family_results] == [5, 5]
    assert [item.gate_passed for item in result.family_results] == [False, False]
    assert result.rejection_counts == {
        "no_candidate_passed": 30,
        "reasoning_over_192_tokens": 52,
        "teacher_length_limit": 11,
    }
    public = path.read_text(encoding="utf-8")
    assert "raw_output" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_p0_report_records_real_rejection_and_next_boundary() -> None:
    report = Path("reports/m5/m5_r3_p0.md").read_text(encoding="utf-8")

    assert "`COMPLETED_GATE_REJECTED`" in report
    assert "10/40" in report
    assert "52" in report
    assert "11" in report
    assert "`model_generated=false`" in report
    assert "`quality_metric=false`" in report
    assert "不进入每类 120 条、合计 240 条的正式扩展" in report


def test_m5_r3_p0_r1_real_result_retains_rejected_gate() -> None:
    path = Path("reports/m5/raw/m5_r3_p0_r1.json")
    result = M5R3P0Result.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.pilot_version == "m5-r3-p0-r1-v1"
    assert result.status == "fail"
    assert result.git_dirty is False
    assert result.accepted_samples == 12
    assert result.rejected_tasks == 28
    assert result.generation_attempts == 80
    assert [item.accepted_items for item in result.family_results] == [4, 8]
    assert [item.accepted_language_counts for item in result.family_results] == [
        {"en": 3, "zh": 1},
        {"en": 7, "zh": 1},
    ]
    assert [item.gate_passed for item in result.family_results] == [False, False]
    assert result.rejection_counts == {
        "no_candidate_passed": 28,
        "reasoning_over_192_tokens": 46,
        "teacher_length_limit": 14,
    }
    assert result.contamination.status == "pass"
    assert result.contamination.dev_exact_prompt_matches == 0
    assert result.contamination.historical_exact_prompt_matches == 0
    public = path.read_text(encoding="utf-8")
    assert "raw_output" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_p0_r1_report_records_real_rejection_and_source_boundary() -> None:
    report = Path("reports/m5/m5_r3_p0_r1.md").read_text(encoding="utf-8")

    assert "`COMPLETED_GATE_REJECTED`" in report
    assert "接受 12/40 条" in report
    assert "推理超过 192 Token | 46" in report
    assert "Teacher 达到 384 Token 生成上限 | 14" in report
    assert "不实现 240 条扩展" in report
    assert "停止继续尝试同类 Prompt-only 变体" in report
    assert "Teacher 来源策略审查" in report


def test_m5_r3_teacher_source_review_authorizes_only_p1_contract() -> None:
    path = Path("reports/m5/raw/m5_r3_teacher_source_strategy_review.json")
    review = M5R3TeacherSourceStrategyReview.model_validate_json(path.read_text(encoding="utf-8"))

    assert review.status == "two_stage_contract_authorized"
    assert review.evidence_kind == "deterministic_review_of_real_public_results"
    assert review.quality_metric is False
    assert review.selected_strategy == "two_stage_solve_compress"
    assert review.controlled_baseline == "deterministic_rule_trace"
    assert review.p1_contract_implementation_authorized is True
    assert review.p1_gpu_pilot_authorized is False
    assert review.formal_source_expansion_authorized is False
    assert review.r3_mixture_authorized is False
    assert review.r3_training_authorized is False
    public = path.read_text(encoding="utf-8")
    assert "raw_output" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_teacher_source_report_keeps_control_and_teacher_separate() -> None:
    report = Path("reports/m5/m5_r3_teacher_source_strategy.md").read_text(encoding="utf-8")

    assert "`two_stage_contract_authorized`" in report
    assert "`two_stage_solve_compress`" in report
    assert "`deterministic_rule_trace`" in report
    assert "`training_source_authorized=false`" in report
    assert "`p1_gpu_pilot_authorized` 保持 `false`" in report


def test_m5_r3_p1_cpu_smoke_authorizes_only_real_gpu_execution() -> None:
    path = Path("reports/m5/raw/m5_r3_p1_cpu_smoke.json")
    smoke = M5R3P1CPUSmoke.model_validate_json(path.read_text(encoding="utf-8"))

    assert smoke.evidence_kind == "synthetic_cpu_contract_smoke"
    assert smoke.model_generated is False
    assert smoke.quality_metric is False
    assert smoke.accepted_samples == 40
    assert all(item.gate_passed for item in smoke.family_results)
    assert smoke.control.status == "pass"
    assert smoke.control.training_source_authorized is False
    assert smoke.p1_gpu_pilot_authorized is True
    assert smoke.formal_source_expansion_authorized is False
    assert smoke.r3_mixture_authorized is False
    assert smoke.r3_training_authorized is False


def test_m5_r3_p1_real_result_retains_rejected_gate() -> None:
    path = Path("reports/m5/raw/m5_r3_p1.json")
    result = M5R3P1Result.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.status == "fail"
    assert result.git_dirty is False
    assert result.accepted_samples == 11
    assert result.rejected_tasks == 29
    assert result.formal_source_expansion_authorized is False
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False
    assert [item.accepted_items for item in result.family_results] == [5, 6]
    assert [item.accepted_language_counts for item in result.family_results] == [
        {"en": 2, "zh": 3},
        {"en": 5, "zh": 1},
    ]
    assert result.rejection_counts == {
        "compressor_invalid_json": 3,
        "missing_evidence_anchor": 10,
        "other_label_mentioned": 10,
        "solver_length_limit": 6,
    }
    public = path.read_text(encoding="utf-8")
    assert "raw_output" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_p1_report_keeps_cpu_contract_and_quality_separate() -> None:
    report = Path("reports/m5/m5_r3_p1.md").read_text(encoding="utf-8")

    assert "`COMPLETED_GATE_REJECTED`" in report
    assert "`model_generated=false`" in report
    assert "`quality_metric=false`" in report
    assert "接受 11 条" in report
    assert "`formal_source_expansion_authorized=false`" in report
    assert "正式 240 条扩展、R3 Mixture 和训练继续阻断" in report


def test_m5_r3_p2_cpu_smoke_authorizes_only_real_gpu_execution() -> None:
    path = Path("reports/m5/raw/m5_r3_p2_cpu_smoke.json")
    smoke = M5R3P2CPUSmoke.model_validate_json(path.read_text(encoding="utf-8"))

    assert smoke.model_generated is False
    assert smoke.quality_metric is False
    assert smoke.fallback_solver_items == 6
    assert smoke.isolated_compressor_items == 40
    assert smoke.accepted_samples == 40
    assert smoke.p2_gpu_pilot_authorized is True
    assert smoke.formal_source_expansion_authorized is False
    assert smoke.r3_mixture_authorized is False
    assert smoke.r3_training_authorized is False


def test_m5_r3_p2_report_keeps_cpu_and_gpu_evidence_separate() -> None:
    report = Path("reports/m5/m5_r3_p2.md").read_text(encoding="utf-8")

    assert "`COMPLETED_GATE_PASSED`" in report
    assert "`model_generated=false`" in report
    assert "`quality_metric=false`" in report
    assert "接受 33/40 条" in report
    assert "Mixture 和训练" in report


def test_m5_r3_p2_real_result_authorizes_only_source_expansion() -> None:
    path = Path("reports/m5/raw/m5_r3_p2.json")
    result = M5R3P2Result.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert result.git_dirty is False
    assert result.git_commit == "f063947420d5c42eef883cba3d58cebe5baa79fb"
    assert result.accepted_samples == 33
    assert result.rejected_tasks == 7
    assert [item.accepted_items for item in result.family_results] == [17, 16]
    assert [item.accepted_language_counts for item in result.family_results] == [
        {"en": 13, "zh": 4},
        {"en": 11, "zh": 5},
    ]
    assert result.rejection_counts == {
        "compressor_invalid_json": 5,
        "solver_length_limit": 2,
    }
    assert result.contamination.status == "pass"
    assert result.formal_source_expansion_authorized is True
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False
    public = path.read_text(encoding="utf-8")
    assert "raw_output" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public


def test_m5_r3_p2_content_review_records_full_maintainer_approval() -> None:
    path = Path("reports/m5/raw/m5_r3_p2_content_review.json")
    result = M5R3ContentReviewResult.model_validate_json(path.read_text(encoding="utf-8"))

    assert result.status == "approved"
    assert result.reviewer_role == "maintainer"
    assert result.reviewed_items == result.passed_items == 33
    assert result.rejected_items == 0
    assert result.family_counts == {"config": 17, "log_diagnosis": 16}
    assert result.language_counts == {"en": 24, "zh": 9}
    assert (
        result.private_judgments_sha256
        == "30b669b2d4ec4aa86208e7dee44962283272a3aec05d17d4b1ef33f927adce7a"
    )
    assert result.formal_source_expansion_authorized is True
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False
    public = path.read_text(encoding="utf-8")
    assert "prompt" not in public
    assert "reasoning_content" not in public
    assert "/home/" not in public
    assert "/data/" not in public
