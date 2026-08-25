from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tinyllm.deployment.m10_lora_stage as stage_module
import tinyllm.training.m10_lora_gate as gate_module
from tinyllm.agent_eval.schema import AgentEvalMetricSummary, AgentEvalSummary
from tinyllm.deployment.m10_lora_stage import (
    M10LoRAStageRegistrationError,
    build_m10_lora_stage_evaluation_subject,
    register_m10_lora_stage_evaluation_subject,
)
from tinyllm.evaluation.m6_schema import M6GeneralResult, M6GeneralTaskResult, M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_lora import (
    M10LoRACheckpointStore,
    M10LoRAProgress,
    export_m10_lora_stage,
    load_m10_lora_config,
    record_m10_lora_result,
)
from tinyllm.training.m10_lora_gate import (
    M10LoRAStageGateError,
    assemble_m10_lora_stage_gate,
)
from tinyllm.training.m10_lora_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10_LORA_PARENT_TOKENIZER_SHA256,
    M10LoRAGeneralPassSummary,
    M10LoRAMemoryProbeResult,
    M10LoRARunResult,
    M10LoRAStageExport,
)

CONFIG = Path("configs/sft/m10_agent_lora_qwen3_8b.yaml")
MODEL_FILES = (
    "config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
)


class _ExportableAdapter:
    peft_config: dict[str, Any] = {}

    def save_pretrained(self, root: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"m10-agent-lora-adapter")


def _prepare_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, M10LoRARunResult, SimpleNamespace]:
    root = (tmp_path / "artifacts").resolve()
    run = root / "runs" / "m10-lora-unit-run"
    run.mkdir(parents=True)
    shutil.copyfile(CONFIG, run / "config.original.yaml")
    config = load_m10_lora_config(run / "config.original.yaml")
    config_sha256 = canonical_config_hash(config)

    model_dir = root / "cache" / "qwen3-8b"
    model_dir.mkdir(parents=True)
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(name.encode())
    (model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    parent = SimpleNamespace(
        status="Evaluation",
        model_version=M10_LORA_PARENT_SUBJECT,
        evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
        model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
        tokenizer_artifact_sha256=M10_LORA_PARENT_TOKENIZER_SHA256,
        model_dir=model_dir,
        tokenizer_dir=model_dir,
    )
    monkeypatch.setattr(stage_module, "resolve_evaluation_subject", lambda *_: parent)

    probe_path = root / "memory-probes" / "m10" / "probe.json"
    probe_path.parent.mkdir(parents=True)
    probe_path.write_text(
        M10LoRAMemoryProbeResult(
            config_sha256=config_sha256,
            git_commit="a" * 40,
            git_dirty=False,
            dataset_version=M10_DATASET_VERSION,
            parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
            environment_sha256="b" * 64,
            hardware_compatibility_sha256="c" * 64,
            physical_gpu_index=4,
            gpu_name="NVIDIA GeForce RTX 3090",
            optimizer_steps=10,
            supervised_tokens=10_000,
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
            duration_seconds=1.0,
        ).model_dump_json(),
        encoding="utf-8",
    )
    probe_sha256 = hashlib.sha256(probe_path.read_bytes()).hexdigest()

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    store = M10LoRACheckpointStore(run / "checkpoints")
    checkpoint = store.save(
        adapter_state={"adapter.weight": torch.tensor([2.0])},
        optimizer=optimizer,
        progress=M10LoRAProgress(1008, 1, 0, 1_000_000, 2.0, 1.0),
        config=config,
        config_sha256=config_sha256,
        run_id=run.name,
        git_commit="a" * 40,
        environment_sha256="b" * 64,
        hardware_sha256="c" * 64,
        memory_probe_sha256=probe_sha256,
        pin_reason="stage",
    )
    stage_export = export_m10_lora_stage(
        _ExportableAdapter(), run / "exports", checkpoint.checkpoint_id
    )
    result = M10LoRARunResult(
        status="stage_completed",
        mode="fresh",
        run_id=run.name,
        config_sha256=config_sha256,
        git_commit="a" * 40,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
        parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
        parent_evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
        parent_model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
        attention_architecture="gqa",
        adaptation="lora",
        peft_version="0.19.1",
        seed=42,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        trainable_parameters=10,
        total_parameters=100,
        global_step=1008,
        completed_epochs=1,
        supervised_tokens=1_000_000,
        initial_loss=2.0,
        final_loss=1.0,
        duration_seconds=10.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        memory_probe_sha256=probe_sha256,
        latest_checkpoint=checkpoint.checkpoint_id,
        stage_export=stage_export,
    )
    record_m10_lora_result(run, result)
    return root, run, result, parent


def _agent_summary(
    *,
    model_id: str,
    model_artifact_sha256: str,
    evaluation_subject_sha256: str,
    task_success_basis_points: int,
) -> AgentEvalSummary:
    metrics = AgentEvalMetricSummary(
        item_count=80,
        tool_selection_accuracy_basis_points=8000,
        argument_accuracy_basis_points=8000,
        schema_valid_rate_basis_points=10_000,
        no_tool_accuracy_basis_points=9000,
        multi_step_success_rate_basis_points=5000,
        task_success_rate_basis_points=task_success_basis_points,
        tool_hallucination_rate_basis_points=0,
        error_recovery_rate_basis_points=7000,
        grounding_accuracy_basis_points=9000,
        approval_safety_basis_points=10_000,
        average_tool_calls_milli=1000,
        average_tokens_per_task_milli=100_000,
        p95_end_to_end_milliseconds=1000,
        unapproved_write_attempts=0,
        path_escape_attempts=0,
        arbitrary_command_attempts=0,
    )
    return AgentEvalSummary(
        evaluation_id="m9-agent-eval-12345678",
        evaluated_at=datetime(2026, 8, 25, tzinfo=UTC),
        suite_version="tinyllm-devops-agent-dev-v1-f958bcc6",
        suite_content_sha256="1" * 64,
        model_id=model_id,
        model_revision="b968826d9c46dd6066d109eabc6255188de91218",
        model_artifact_sha256=model_artifact_sha256,
        parent_model_id="Qwen/Qwen3-8B",
        deployment_record_sha256=None,
        evaluation_subject_sha256=evaluation_subject_sha256,
        environment_sha256="2" * 64,
        hardware_sha256="3" * 64,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        driver_version="535.261.03",
        gateway_version="0.9.0rc1",
        agent_runtime_version="0.9.0rc1",
        git_commit="4" * 40,
        git_dirty=False,
        metrics=metrics,
        item_results_sha256="5" * 64,
        completed=True,
    )


def _write_summary(path: Path, summary: AgentEvalSummary) -> None:
    path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _general_result(score: float) -> M6GeneralResult:
    tasks = tuple(
        M6GeneralTaskResult(
            task=task,  # type: ignore[arg-type]
            samples=samples,
            acc=score,
            acc_stderr=0.01,
            acc_norm=score,
            acc_norm_stderr=0.01,
        )
        for task, samples in (
            ("tinyllm_arc_easy", 2376),
            ("tinyllm_hellaswag", 10042),
            ("tinyllm_piqa", 1838),
        )
    )
    return M6GeneralResult(
        harness_version="0.4.12",
        metric="acc_norm",
        aggregation="equal-task-mean",
        tasks=tasks,  # type: ignore[arg-type]
        aggregate_basis_points=round(score * 10_000),
    )


def _m10_general_summary(
    *, subject_id: str, subject_sha256: str, model: M6ModelIdentity, score: float
) -> M10LoRAGeneralPassSummary:
    kind = "parent" if subject_id == M10_LORA_PARENT_SUBJECT else "candidate"
    return M10LoRAGeneralPassSummary(
        status="succeeded",
        evaluation_id=f"m10-lora-m6-general-{kind}-1234abcd",
        protocol_version="m6-release-v7",
        config_sha256="8" * 64,
        git_commit="9" * 40,
        git_dirty=False,
        evaluation_subject_id=subject_id,
        evaluation_subject_sha256=subject_sha256,
        model=model,
        general=_general_result(score),
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=1.0,
        environment_sha256="a" * 64,
        hardware_sha256="b" * 64,
        raw_results_sha256="c" * 64,
    )


def test_build_register_and_gate_one_million_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run, _, _ = _prepare_stage(tmp_path, monkeypatch)
    built = build_m10_lora_stage_evaluation_subject(
        artifact_root=root, source_run=run, stage_tokens=1_000_000
    )
    stored, record_sha256 = register_m10_lora_stage_evaluation_subject(
        artifact_root=root, source_run=run, stage_tokens=1_000_000
    )
    repeated, repeated_sha256 = register_m10_lora_stage_evaluation_subject(
        artifact_root=root, source_run=run, stage_tokens=1_000_000
    )
    assert built.subject_id == stored.subject_id == repeated.subject_id
    assert record_sha256 == repeated_sha256
    assert stored.production_eligible is False
    resolved_stage = SimpleNamespace(
        model=stored.model,
        model_artifact_sha256=stored.effective_artifact_sha256,
        evaluation_subject_sha256=record_sha256,
        adapter_artifact_sha256=stored.adapter_artifact_sha256,
    )
    monkeypatch.setattr(
        gate_module, "resolve_m10_lora_stage_evaluation_subject", lambda *_: resolved_stage
    )

    parent_path = tmp_path / "parent-agent.json"
    candidate_path = tmp_path / "candidate-agent.json"
    _write_summary(
        parent_path,
        _agent_summary(
            model_id=M10_LORA_PARENT_SUBJECT,
            model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
            evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            task_success_basis_points=4500,
        ),
    )
    _write_summary(
        candidate_path,
        _agent_summary(
            model_id=stored.subject_id,
            model_artifact_sha256=stored.effective_artifact_sha256,
            evaluation_subject_sha256=record_sha256,
            task_success_basis_points=4625,
        ),
    )
    output = tmp_path / "accepted-gate"
    m6_evidence, gate = assemble_m10_lora_stage_gate(
        artifact_root=root,
        source_run=run,
        candidate_subject_id=stored.subject_id,
        parent_agent_summary_path=parent_path,
        candidate_agent_summary_path=candidate_path,
        output_directory=output,
    )
    assert m6_evidence is None
    assert gate.decision == "accepted"
    assert gate.improvement_basis_points == 125
    assert (output / "continuation-gate.json").is_file()
    with pytest.raises(M10LoRAStageGateError, match="already exists"):
        assemble_m10_lora_stage_gate(
            artifact_root=root,
            source_run=run,
            candidate_subject_id=stored.subject_id,
            parent_agent_summary_path=parent_path,
            candidate_agent_summary_path=candidate_path,
            output_directory=output,
        )


def test_stage_registration_and_gate_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run, result, _ = _prepare_stage(tmp_path, monkeypatch)
    with pytest.raises(M10LoRAStageRegistrationError, match="1M, 5M, or 10M"):
        build_m10_lora_stage_evaluation_subject(
            artifact_root=root, source_run=run, stage_tokens=2_000_000
        )
    with pytest.raises(M10LoRAStageRegistrationError, match="absolute"):
        build_m10_lora_stage_evaluation_subject(
            artifact_root=Path("relative"), source_run=run, stage_tokens=1_000_000
        )

    attempt = run / "attempts" / "fresh-stage_completed-tokens-0001000000.json"
    drifted = result.model_copy(update={"run_id": "different-run"})
    attempt.write_text(drifted.model_dump_json(), encoding="utf-8")
    with pytest.raises(M10LoRAStageRegistrationError, match="incomplete or inconsistent"):
        build_m10_lora_stage_evaluation_subject(
            artifact_root=root, source_run=run, stage_tokens=1_000_000
        )
    attempt.write_text(result.model_dump_json(), encoding="utf-8")

    stored, record_sha256 = register_m10_lora_stage_evaluation_subject(
        artifact_root=root, source_run=run, stage_tokens=1_000_000
    )
    resolved_stage = SimpleNamespace(
        model=stored.model,
        model_artifact_sha256=stored.effective_artifact_sha256,
        evaluation_subject_sha256=record_sha256,
        adapter_artifact_sha256=stored.adapter_artifact_sha256,
    )
    monkeypatch.setattr(
        gate_module, "resolve_m10_lora_stage_evaluation_subject", lambda *_: resolved_stage
    )
    parent_path = tmp_path / "parent-agent.json"
    candidate_path = tmp_path / "candidate-agent.json"
    _write_summary(
        parent_path,
        _agent_summary(
            model_id=M10_LORA_PARENT_SUBJECT,
            model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
            evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            task_success_basis_points=4500,
        ),
    )
    _write_summary(
        candidate_path,
        _agent_summary(
            model_id=stored.subject_id,
            model_artifact_sha256=stored.effective_artifact_sha256,
            evaluation_subject_sha256=record_sha256,
            task_success_basis_points=4500,
        ),
    )
    _, rejected = assemble_m10_lora_stage_gate(
        artifact_root=root,
        source_run=run,
        candidate_subject_id=stored.subject_id,
        parent_agent_summary_path=parent_path,
        candidate_agent_summary_path=candidate_path,
        output_directory=tmp_path / "rejected-gate",
    )
    assert rejected.decision == "rejected"

    with pytest.raises(M10LoRAStageGateError, match="paths must be absolute"):
        assemble_m10_lora_stage_gate(
            artifact_root=root,
            source_run=run,
            candidate_subject_id=stored.subject_id,
            parent_agent_summary_path=parent_path,
            candidate_agent_summary_path=candidate_path,
            output_directory=Path("relative-gate"),
        )
    with pytest.raises(M10LoRAStageGateError, match="must not consume M6"):
        assemble_m10_lora_stage_gate(
            artifact_root=root,
            source_run=run,
            candidate_subject_id=stored.subject_id,
            parent_agent_summary_path=parent_path,
            candidate_agent_summary_path=candidate_path,
            output_directory=tmp_path / "invalid-m6-gate",
            parent_m6_summary_path=tmp_path / "unused-parent-m6.json",
        )

    candidate_path.write_text("{}", encoding="utf-8")
    with pytest.raises(M10LoRAStageGateError, match="evidence is invalid"):
        assemble_m10_lora_stage_gate(
            artifact_root=root,
            source_run=run,
            candidate_subject_id=stored.subject_id,
            parent_agent_summary_path=parent_path,
            candidate_agent_summary_path=candidate_path,
            output_directory=tmp_path / "invalid-summary-gate",
        )


def test_five_million_gate_accepts_paired_agent_and_m6_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    run = root / "runs" / "m10-lora-5m-unit-run"
    run.mkdir(parents=True)
    subject_id = "qwen3-8b-m10-agent-lora-5m-1234abcd"
    subject_sha256 = "d" * 64
    adapter_sha256 = "e" * 64
    effective_sha256 = "f" * 64
    config_sha256 = "1" * 64
    candidate_model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-8B",
        base_revision="b968826d9c46dd6066d109eabc6255188de91218",
        attention_architecture="gqa",
        adaptation="lora",
        model_artifact_sha256=effective_sha256,
        model_parameters=8_234_382_336,
        training_run_id=run.name,
        training_checkpoint_id="checkpoint-tokens-0005000000",
        training_tokens=5_000_000,
        training_config_sha256=config_sha256,
        dataset_version=M10_DATASET_VERSION,
        dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
        adapter_sha256=adapter_sha256,
    )
    result = M10LoRARunResult(
        status="stage_completed",
        mode="exact_resume",
        run_id=run.name,
        config_sha256=config_sha256,
        git_commit="2" * 40,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
        parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
        parent_evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
        parent_model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
        attention_architecture="gqa",
        adaptation="lora",
        peft_version="0.19.1",
        seed=42,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        trainable_parameters=10,
        total_parameters=100,
        global_step=5040,
        completed_epochs=5,
        supervised_tokens=5_000_000,
        initial_loss=2.0,
        final_loss=1.0,
        duration_seconds=10.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        memory_probe_sha256="3" * 64,
        latest_checkpoint="checkpoint-tokens-0005000000",
        resumed_from_tokens=4_000_000,
        continuation_gate_sha256="4" * 64,
        stage_export=M10LoRAStageExport(
            checkpoint_id="checkpoint-tokens-0005000000",
            supervised_tokens=5_000_000,
            adapter_artifact_sha256=adapter_sha256,
            adapter_files=("adapter_config.json", "adapter_model.safetensors"),
        ),
    )
    (run / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    resolved_stage = SimpleNamespace(
        model=candidate_model,
        model_artifact_sha256=effective_sha256,
        evaluation_subject_sha256=subject_sha256,
        adapter_artifact_sha256=adapter_sha256,
    )
    monkeypatch.setattr(
        gate_module, "resolve_m10_lora_stage_evaluation_subject", lambda *_: resolved_stage
    )

    parent_agent_path = root / "parent-agent.json"
    candidate_agent_path = root / "candidate-agent.json"
    _write_summary(
        parent_agent_path,
        _agent_summary(
            model_id=M10_LORA_PARENT_SUBJECT,
            model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
            evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            task_success_basis_points=4500,
        ),
    )
    _write_summary(
        candidate_agent_path,
        _agent_summary(
            model_id=subject_id,
            model_artifact_sha256=effective_sha256,
            evaluation_subject_sha256=subject_sha256,
            task_success_basis_points=4625,
        ),
    )
    parent_model = M6ModelIdentity(
        role="base",
        repository="Qwen/Qwen3-8B",
        base_revision="b968826d9c46dd6066d109eabc6255188de91218",
        attention_architecture="gqa",
        adaptation="base",
        model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
        model_parameters=8_234_382_336,
    )
    parent_m6_path = root / "parent-m6.json"
    candidate_m6_path = root / "candidate-m6.json"
    parent_m6_path.write_text(
        _m10_general_summary(
            subject_id=M10_LORA_PARENT_SUBJECT,
            subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            model=parent_model,
            score=0.5,
        ).model_dump_json(),
        encoding="utf-8",
    )
    candidate_m6_path.write_text(
        _m10_general_summary(
            subject_id=subject_id,
            subject_sha256=subject_sha256,
            model=candidate_model,
            score=0.485,
        ).model_dump_json(),
        encoding="utf-8",
    )

    evidence, gate = assemble_m10_lora_stage_gate(
        artifact_root=root,
        source_run=run,
        candidate_subject_id=subject_id,
        parent_agent_summary_path=parent_agent_path,
        candidate_agent_summary_path=candidate_agent_path,
        parent_m6_summary_path=parent_m6_path,
        candidate_m6_summary_path=candidate_m6_path,
        output_directory=root / "gate-5m",
    )
    assert evidence is not None
    assert evidence.regression_basis_points == 150
    assert gate.decision == "accepted"
    assert gate.target_stage_tokens == 10_000_000
