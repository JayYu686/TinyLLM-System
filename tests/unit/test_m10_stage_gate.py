from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from tinyllm.agent_eval import AgentEvalMetricSummary, AgentEvalSummary
from tinyllm.deployment import ResolvedEvaluationSubject
from tinyllm.evaluation import (
    M6GeneralPassSummary,
    M6GeneralResult,
    M6GeneralTaskResult,
    M6ModelIdentity,
)
from tinyllm.schemas.base import StrictSchema
from tinyllm.training.m10_sft_schema import M10FullSFTRunResult, M10StageExport
from tinyllm.training.m10_stage_gate import assemble_m10_stage_gate

NOW = datetime(2026, 8, 25, tzinfo=UTC)
RUN_ID = "20260824T011335Z-m10-agent-full-sft-qwen3-0-6b-seed42-1ac1cad4-7b63"
SUBJECT_ID = "qwen3-0-6b-m10-full-sft-5m-1234abcd"
SUBJECT_SHA = "1" * 64
STAGE_SHA = "2" * 64
PARENT_SHA: Literal["63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"] = (
    "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
)
PARENT_RECORD: Literal["a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"] = (
    "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
)
CONFIG_SHA = "3" * 64


def _write(path: Path, value: StrictSchema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_model() -> M6ModelIdentity:
    return M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=STAGE_SHA,
        model_parameters=596_049_920,
        training_run_id=RUN_ID,
        training_checkpoint_id="checkpoint-tokens-0005000000",
        training_tokens=5_000_000,
        training_config_sha256=CONFIG_SHA,
        dataset_version="m10-agent-sft-v1-4655d3e3",
        dataset_manifest_sha256=(
            "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
        ),
    )


def _parent_model() -> M6ModelIdentity:
    return M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=PARENT_SHA,
        model_parameters=596_049_920,
        training_run_id="parent-run",
        training_checkpoint_id="checkpoint-tokens-0001000000",
        training_tokens=1_000_000,
        training_config_sha256="4" * 64,
        dataset_version="parent-data",
        dataset_manifest_sha256="5" * 64,
    )


def _agent_summary(*, candidate: bool, task_success: int) -> AgentEvalSummary:
    return AgentEvalSummary(
        evaluation_id="m9-agent-eval-1234abcd" if candidate else "m9-agent-eval-5678abcd",
        evaluated_at=NOW,
        suite_version="tinyllm-devops-agent-dev-v1-f958bcc6",
        suite_content_sha256="6" * 64,
        model_id=SUBJECT_ID if candidate else "qwen3-0-6b-m7-fa678d92",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        model_artifact_sha256=STAGE_SHA if candidate else PARENT_SHA,
        parent_model_id="Qwen/Qwen3-0.6B@revision",
        deployment_record_sha256=None if candidate else PARENT_RECORD,
        evaluation_subject_sha256=SUBJECT_SHA if candidate else None,
        environment_sha256="7" * 64,
        hardware_sha256="8" * 64,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        driver_version="535.261.03",
        gateway_version="0.1.0",
        agent_runtime_version="0.1.0",
        git_commit="9" * 40,
        git_dirty=False,
        metrics=AgentEvalMetricSummary(
            item_count=80,
            tool_selection_accuracy_basis_points=5000,
            argument_accuracy_basis_points=5000,
            schema_valid_rate_basis_points=10000,
            no_tool_accuracy_basis_points=9000,
            multi_step_success_rate_basis_points=1000,
            task_success_rate_basis_points=task_success,
            tool_hallucination_rate_basis_points=100,
            error_recovery_rate_basis_points=1000,
            grounding_accuracy_basis_points=5000,
            approval_safety_basis_points=10000,
            average_tool_calls_milli=1000,
            average_tokens_per_task_milli=1000,
            p95_end_to_end_milliseconds=1000,
            unapproved_write_attempts=0,
            path_escape_attempts=0,
            arbitrary_command_attempts=0,
        ),
        item_results_sha256="a" * 64,
        completed=True,
    )


def _general_summary(*, candidate: bool, score: float) -> M6GeneralPassSummary:
    arc = M6GeneralTaskResult(
        task="tinyllm_arc_easy",
        samples=2376,
        acc=score,
        acc_stderr=0.01,
        acc_norm=score,
        acc_norm_stderr=0.01,
    )
    hellaswag = M6GeneralTaskResult(
        task="tinyllm_hellaswag",
        samples=10042,
        acc=score,
        acc_stderr=0.01,
        acc_norm=score,
        acc_norm_stderr=0.01,
    )
    piqa = M6GeneralTaskResult(
        task="tinyllm_piqa",
        samples=1838,
        acc=score,
        acc_stderr=0.01,
        acc_norm=score,
        acc_norm_stderr=0.01,
    )
    return M6GeneralPassSummary(
        status="succeeded",
        evaluation_id="m6-general-candidate-test",
        protocol_version="m6-release-v7",
        config_sha256="b" * 64,
        git_commit="c" * 40,
        git_dirty=False,
        model=_candidate_model() if candidate else _parent_model(),
        general=M6GeneralResult(
            harness_version="0.4.12",
            metric="acc_norm",
            aggregation="equal-task-mean",
            tasks=(arc, hellaswag, piqa),
            aggregate_basis_points=round(score * 10_000),
        ),
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=1.0,
        environment_sha256="d" * 64,
        hardware_sha256="e" * 64,
        raw_results_sha256="f" * 64,
    )


def test_assemble_m10_stage_gate_accepts_paired_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    source_run = root / "runs" / RUN_ID
    result = M10FullSFTRunResult(
        status="stage_completed",
        mode="exact_resume",
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        git_commit="1" * 40,
        git_dirty=False,
        dataset_version="m10-agent-sft-v1-4655d3e3",
        dataset_manifest_sha256=(
            "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
        ),
        parent_production_version="qwen3-0-6b-m7-fa678d92",
        parent_production_record_sha256=PARENT_RECORD,
        parent_model_artifact_sha256=PARENT_SHA,
        attention_architecture="gqa",
        seed=42,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        global_step=5040,
        completed_epochs=5,
        supervised_tokens=5_000_000,
        initial_loss=1.0,
        final_loss=0.2,
        duration_seconds=1.0,
        peak_allocated_bytes=2,
        peak_reserved_bytes=3,
        latest_checkpoint="checkpoint-tokens-0005000000",
        resumed_from_tokens=1_000_000,
        stage_export=M10StageExport(
            checkpoint_id="checkpoint-tokens-0005000000",
            supervised_tokens=5_000_000,
            export_sha256=STAGE_SHA,
        ),
    )
    _write(source_run / "result.json", result)
    parent_agent = root / "evidence" / "parent-agent.json"
    candidate_agent = root / "evidence" / "candidate-agent.json"
    parent_m6 = root / "evidence" / "parent-m6.json"
    candidate_m6 = root / "evidence" / "candidate-m6.json"
    _write(parent_agent, _agent_summary(candidate=False, task_success=2000))
    _write(candidate_agent, _agent_summary(candidate=True, task_success=2250))
    _write(parent_m6, _general_summary(candidate=False, score=0.5448))
    _write(candidate_m6, _general_summary(candidate=True, score=0.55))
    resolved = ResolvedEvaluationSubject(
        requested_ref=SUBJECT_ID,
        model_version=SUBJECT_ID,
        evaluation_subject_sha256=SUBJECT_SHA,
        model=_candidate_model(),
        model_dir=root / "model",
        model_artifact_sha256=STAGE_SHA,
        tokenizer_dir=root / "tokenizer",
        tokenizer_artifact_sha256="0" * 64,
        verified_at=NOW,
    )
    monkeypatch.setattr(
        "tinyllm.training.m10_stage_gate.resolve_m10_stage_evaluation_subject",
        lambda *_args: resolved,
    )

    m6_evidence, gate = assemble_m10_stage_gate(
        artifact_root=root,
        source_run=source_run,
        candidate_subject_id=SUBJECT_ID,
        parent_agent_summary_path=parent_agent,
        candidate_agent_summary_path=candidate_agent,
        parent_m6_summary_path=parent_m6,
        candidate_m6_summary_path=candidate_m6,
        output_directory=root / "gates" / "5m",
    )

    assert gate.decision == "accepted"
    assert gate.agent_dev_improvement_basis_points == 250
    assert m6_evidence.regression_basis_points == -52
    assert (root / "gates" / "5m" / "continuation-gate.json").is_file()
