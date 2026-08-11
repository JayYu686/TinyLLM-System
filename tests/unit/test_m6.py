from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from tinyllm.cli import main
from tinyllm.evaluation import (
    M6ComparisonError,
    M6DomainItemScore,
    M6DomainModeResult,
    M6EvaluationResult,
    M6GeneralResult,
    M6GeneralTaskResult,
    M6ModelIdentity,
    M6PromotionError,
    compare_m6_evaluations,
    load_m6_release_config,
    promote_m6_candidate,
)
from tinyllm.schemas import canonical_config_hash


def _items(*, correct: bool) -> tuple[M6DomainItemScore, ...]:
    decoded = [
        json.loads(line)
        for line in Path("evals/domain/v1/items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results: list[M6DomainItemScore] = []
    for item in decoded:
        pair = next(
            (tag for tag in cast(list[str], item["tags"]) if tag.startswith("bilingual-pair-")),
            None,
        )
        category = cast(str, item["category"])
        item_id = cast(str, item["id"])
        cluster_id = (
            f"pair:{category}:{pair.rsplit('-', 1)[1]}"
            if pair is not None
            else f"singleton:{item_id}"
        )
        scorer_kind = cast(dict[str, Any], item["scorer"])["kind"]
        results.append(
            M6DomainItemScore(
                item_id=item_id,
                cluster_id=cluster_id,
                language=item["language"],
                category=cast(Any, category),
                scorer_kind=scorer_kind,
                correct=correct,
                json_valid=correct if scorer_kind == "json_object" else None,
                format_valid=True,
                visible_reasoning_leakage=False,
            )
        )
    return tuple(results)


def _mode(mode: str, *, correct: bool) -> M6DomainModeResult:
    items = _items(correct=correct)
    correct_items = sum(item.correct for item in items)
    json_valid = sum(item.json_valid is True for item in items)
    return M6DomainModeResult(
        mode=cast(Any, mode),
        items=items,
        evaluated_items=300,
        correct_items=correct_items,
        score_basis_points=round(correct_items * 10000 / 300),
        format_valid_items=300,
        format_valid_basis_points=10000,
        json_items=80,
        json_valid_items=json_valid,
        json_valid_basis_points=round(json_valid * 10000 / 80),
        visible_reasoning_leakage_items=0,
        visible_reasoning_leakage_basis_points=0,
        natural_thinking_closed_items=300 if mode == "thinking" else 0,
        budget_forced_close_items=0,
        forced_close_basis_points=0,
        generated_tokens=300,
        injected_tokens=0,
    )


def _general(*, delta: float = 0.0) -> M6GeneralResult:
    values = (
        ("tinyllm_arc_easy", 2376, 0.5374579125, 0.4726430976),
        ("tinyllm_hellaswag", 10042, 0.3613821948, 0.4207329217),
        ("tinyllm_piqa", 1838, 0.6643090316, 0.6605005441),
    )
    tasks = tuple(
        M6GeneralTaskResult(
            task=cast(Any, name),
            samples=samples,
            acc=acc,
            acc_stderr=0.01,
            acc_norm=acc_norm + delta,
            acc_norm_stderr=0.01,
        )
        for name, samples, acc, acc_norm in values
    )
    typed_tasks = cast(
        tuple[M6GeneralTaskResult, M6GeneralTaskResult, M6GeneralTaskResult],
        tasks,
    )
    return M6GeneralResult(
        harness_version="0.4.12",
        metric="acc_norm",
        aggregation="equal-task-mean",
        tasks=typed_tasks,
        aggregate_basis_points=round(sum(task.acc_norm for task in tasks) * 10000 / 3),
    )


def _model(role: str) -> M6ModelIdentity:
    common: dict[str, Any] = {
        "role": role,
        "repository": "Qwen/Qwen3-0.6B",
        "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "model_artifact_sha256": "a" * 64 if role == "base" else "b" * 64,
        "model_parameters": 596049920,
    }
    if role == "base":
        common["adaptation"] = "base"
    else:
        common.update(
            {
                "adaptation": "full_sft",
                "training_run_id": "20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
                "training_checkpoint_id": "checkpoint-tokens-0010000532",
                "training_tokens": 10000532,
                "training_config_sha256": "c" * 64,
                "dataset_version": "m5-dual-sft-v1-b5b9e839",
                "dataset_manifest_sha256": "d" * 64,
            }
        )
    return M6ModelIdentity.model_validate(common)


def _evaluation(role: str, *, correct: bool) -> M6EvaluationResult:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    return M6EvaluationResult(
        status="succeeded",
        evaluation_id=f"m6-{role}-evaluation",
        protocol_version="m6-release-v1",
        suite_version="tinyllm-domain-v1-83bdd8ef",
        config_sha256=canonical_config_hash(config),
        git_commit="e" * 40,
        git_dirty=False,
        model=_model(role),
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        thinking_template_sha256=(
            "7c37a1ab66f274f52208e50167e6cafbf00b6f5319207beca572e4b8cb1f8451"
        ),
        nonthinking_template_sha256=(
            "b9a510e2f016a112860e47056f770b04e5c93131cc4a8ecd47fcc950cfdb6273"
        ),
        general_chat_template_sha256=(
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
        ),
        software_environment_sha256="1" * 64,
        hardware_sha256="2" * 64,
        domain_modes=(
            _mode("thinking", correct=correct),
            _mode("nonthinking", correct=correct),
        ),
        general=_general(),
        human_review_complete=True,
        thinking_human_review_sha256="7" * 64,
        nonthinking_human_review_sha256="8" * 64,
        lineage_complete=True,
        raw_domain_results_sha256="3" * 64,
        raw_general_results_sha256="4" * 64,
    )


def test_m6_formal_config_and_cluster_contract_are_frozen() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))

    assert config.bootstrap.replicates == 10000
    assert config.gate.domain_min_delta_basis_points == 300
    assert config.gate.production_gate_enabled is False
    assert len({item.cluster_id for item in _mode("thinking", correct=True).items}) == 210

    payload = _mode("thinking", correct=True).to_dict()
    payload["items"][0]["cluster_id"] = "singleton:domain-config-001"
    with pytest.raises(ValidationError, match="cluster counts"):
        M6DomainModeResult.model_validate(payload)

    evaluation_payload = _evaluation("base", correct=False).to_dict()
    evaluation_payload["domain_modes"][1]["items"][0]["cluster_id"] = evaluation_payload[
        "domain_modes"
    ][1]["items"][1]["cluster_id"]
    with pytest.raises(ValidationError):
        M6EvaluationResult.model_validate(evaluation_payload)


def test_m6_v2_release_keeps_gate_and_binds_independent_suite() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release_v2.yaml"))

    assert config.protocol_version == "m6-release-v2"
    assert config.suite_version == "tinyllm-domain-holdout-v1-c0c948cc"
    assert config.bootstrap.seed == config.domain_execution.thinking.seed == 20260810
    assert config.gate == load_m6_release_config(Path("configs/eval/m6_release.yaml")).gate

    payload = config.to_dict()
    payload["suite_version"] = "tinyllm-domain-v1-83bdd8ef"
    with pytest.raises(ValidationError, match="protocol, suite identity"):
        type(config).model_validate(payload)


def test_m6_v3_release_keeps_gate_and_binds_second_holdout() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release_v3.yaml"))

    assert config.protocol_version == "m6-release-v3"
    assert config.suite_version == "tinyllm-domain-holdout-v1-2b167ce6"
    assert config.bootstrap.seed == config.domain_execution.thinking.seed == 20260811
    assert config.gate == load_m6_release_config(Path("configs/eval/m6_release.yaml")).gate


def test_m6_v4_release_keeps_gate_and_binds_sealed_final_audit() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release_v4.yaml"))

    assert config.protocol_version == "m6-release-v4"
    assert config.suite_version == "tinyllm-domain-final-audit-v1-bac25144"
    assert config.bootstrap.seed == config.domain_execution.thinking.seed == 20260812
    assert config.gate == load_m6_release_config(Path("configs/eval/m6_release.yaml")).gate
    assert config.domain_execution.output_control is not None
    assert config.domain_execution.output_control.json_repair_policy == "json-syntax-only-v2"
    assert config.domain_execution.thinking.final_answer_do_sample is False
    assert config.domain_execution.thinking.final_answer_batch_size == 4
    assert (
        config.domain_execution.output_control.thinking_final_separator_id
        == "qwen3-thinking-final-separator-v1"
    )
    assert (
        config.domain_execution.output_control.thinking_continuation_context_id
        == "qwen3-visible-text-retokenize-v1"
    )


def test_m6_comparison_accepts_only_the_complete_and_gate() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    result = compare_m6_evaluations(
        config,
        _evaluation("base", correct=False),
        _evaluation("candidate", correct=True),
        base_evaluation_sha256="5" * 64,
        candidate_evaluation_sha256="6" * 64,
    )

    assert result.status == "accepted"
    assert result.candidate_eligible is True
    assert result.production_eligible is False
    assert all(check.passed for check in result.checks)
    assert tuple(mode.bootstrap.lower_basis_points for mode in result.mode_comparisons) == (
        10000,
        10000,
    )


def test_m6_comparison_rejects_no_domain_gain() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    result = compare_m6_evaluations(
        config,
        _evaluation("base", correct=False),
        _evaluation("candidate", correct=False),
        base_evaluation_sha256="5" * 64,
        candidate_evaluation_sha256="6" * 64,
    )

    assert result.status == "rejected"
    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {
        "thinking_domain_delta",
        "thinking_domain_ci",
        "nonthinking_domain_delta",
        "nonthinking_domain_ci",
        "json_valid_rate",
    }


def test_m6_comparison_rejects_general_regression_and_incomplete_lineage() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    candidate_payload = _evaluation("candidate", correct=True).to_dict()
    candidate_payload["general"] = _general(delta=-0.021).to_dict()
    candidate_payload["lineage_complete"] = False
    candidate = M6EvaluationResult.model_validate(candidate_payload)

    result = compare_m6_evaluations(
        config,
        _evaluation("base", correct=False),
        candidate,
        base_evaluation_sha256="5" * 64,
        candidate_evaluation_sha256="6" * 64,
    )

    failed = {check.name for check in result.checks if not check.passed}
    assert failed == {"general_regression", "lineage"}


def test_m6_comparison_rejects_incompatible_model_pair() -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    candidate_payload = _evaluation("candidate", correct=True).to_dict()
    candidate_payload["thinking_template_sha256"] = "9" * 64

    with pytest.raises(M6ComparisonError, match="lineage differs"):
        compare_m6_evaluations(
            config,
            _evaluation("base", correct=False),
            M6EvaluationResult.model_validate(candidate_payload),
            base_evaluation_sha256="5" * 64,
            candidate_evaluation_sha256="6" * 64,
        )


def test_m6_promotion_is_atomic_idempotent_and_candidate_only(tmp_path: Path) -> None:
    config = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    comparison = compare_m6_evaluations(
        config,
        _evaluation("base", correct=False),
        _evaluation("candidate", correct=True),
        base_evaluation_sha256="5" * 64,
        candidate_evaluation_sha256="6" * 64,
    )
    now = datetime(2026, 8, 9, tzinfo=UTC)

    first = promote_m6_candidate(
        comparison,
        comparison_sha256="7" * 64,
        registry_root=tmp_path,
        now=now,
    )
    second = promote_m6_candidate(
        comparison,
        comparison_sha256="7" * 64,
        registry_root=tmp_path,
        now=datetime(2026, 8, 10, tzinfo=UTC),
    )

    assert first == second
    assert first.status == "Candidate"
    assert first.production_eligible is False
    assert (tmp_path / "candidates" / first.model_version / "model.json").is_file()

    rejected = compare_m6_evaluations(
        config,
        _evaluation("base", correct=False),
        _evaluation("candidate", correct=False),
        base_evaluation_sha256="5" * 64,
        candidate_evaluation_sha256="6" * 64,
    )
    with pytest.raises(M6PromotionError, match="rejected"):
        promote_m6_candidate(
            rejected,
            comparison_sha256="8" * 64,
            registry_root=tmp_path,
            now=now,
        )


def test_m6_compare_and_promote_cli_emit_stable_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"
    registry_root = tmp_path / "registry"
    base_path.write_text(
        json.dumps(_evaluation("base", correct=False).to_dict()),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_evaluation("candidate", correct=True).to_dict()),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "compare",
                "--config",
                "configs/eval/m6_release.yaml",
                "--baseline",
                str(base_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(comparison_path),
                "--json",
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["status"] == "accepted"
    assert comparison["production_eligible"] is False
    assert comparison_path.is_file()

    assert (
        main(
            [
                "promote",
                "--comparison",
                str(comparison_path),
                "--registry-root",
                str(registry_root),
                "--json",
            ]
        )
        == 0
    )
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "Candidate"
    assert record["production_eligible"] is False


def test_m6_cli_rejects_gate_failure_and_relative_artifact_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"
    base_path.write_text(
        json.dumps(_evaluation("base", correct=False).to_dict()),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_evaluation("candidate", correct=False).to_dict()),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "compare",
                "--config",
                "configs/eval/m6_release.yaml",
                "--baseline",
                str(base_path),
                "--candidate",
                str(candidate_path),
                "--output",
                str(comparison_path),
                "--json",
            ]
        )
        == 6
    )
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    assert comparison_path.is_file()

    assert (
        main(
            [
                "promote",
                "--comparison",
                str(comparison_path),
                "--registry-root",
                str(tmp_path / "registry"),
                "--json",
            ]
        )
        == 6
    )
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "M6_PROMOTION_REJECTED"

    assert (
        main(
            [
                "compare",
                "--config",
                "configs/eval/m6_release.yaml",
                "--baseline",
                str(base_path),
                "--candidate",
                str(candidate_path),
                "--output",
                "relative/comparison.json",
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["message"] == "M6 comparison output must be absolute"
