"""Validated registration of M10 Qwen3-8B Agent LoRA evaluation stages."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from tinyllm.deployment.evaluation_subject import (
    M10LoRAStageEvaluationSubjectRecord,
    effective_artifact_sha256,
    evaluation_artifact_sha256,
    m10_lora_stage_evaluation_subject_id,
    publish_m10_lora_stage_evaluation_subject,
    resolve_evaluation_subject,
)
from tinyllm.deployment.registry import DeploymentError
from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_lora import (
    M10LoRACheckpointStore,
    M10LoRAError,
    export_m10_lora_stage,
    load_m10_lora_config,
)
from tinyllm.training.m10_lora_schema import M10LoRARunResult

MODEL_FILES = (
    "config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
MODEL_PARAMETERS = 8_234_382_336
EVALUATION_STAGES = (1_000_000, 5_000_000, 10_000_000)


class M10LoRAStageRegistrationError(RuntimeError):
    """Raised when one LoRA stage has incomplete or drifting lineage."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_marker(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA commit marker is unavailable") from exc
    if not isinstance(value, dict):
        raise M10LoRAStageRegistrationError("M10 Agent LoRA commit marker is invalid")
    return value


def _find_probe(root: Path, expected_sha256: str) -> Path:
    candidates = []
    for path in sorted((root / "memory-probes" / "m10").glob("*.json")):
        if path.is_file() and not path.is_symlink() and _sha256_file(path) == expected_sha256:
            candidates.append(path)
    if len(candidates) != 1:
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA memory Probe evidence is missing or ambiguous"
        )
    return candidates[0]


def build_m10_lora_stage_evaluation_subject(
    *, artifact_root: Path, source_run: Path, stage_tokens: int
) -> M10LoRAStageEvaluationSubjectRecord:
    """Verify one durable LoRA stage and construct its Evaluation-only identity."""

    if stage_tokens not in EVALUATION_STAGES:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA stage must be 1M, 5M, or 10M")
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not source_run.is_absolute()
        or source_run.is_symlink()
    ):
        raise M10LoRAStageRegistrationError("M10 Agent LoRA paths must be absolute non-symlinks")
    try:
        root = artifact_root.resolve(strict=True)
        run = source_run.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA Artifact paths are unavailable"
        ) from exc
    if not run.is_relative_to(root):
        raise M10LoRAStageRegistrationError("M10 Agent LoRA Run escapes the Artifact Store")

    checkpoint_id = f"checkpoint-tokens-{stage_tokens:010d}"
    mode = "fresh" if stage_tokens == 1_000_000 else "exact_resume"
    status = "succeeded" if stage_tokens == 10_000_000 else "stage_completed"
    result_path = run / "attempts" / f"{mode}-{status}-tokens-{stage_tokens:010d}.json"
    config_path = run / "config.original.yaml"
    checkpoint_manifest_path = run / "checkpoints" / checkpoint_id / "manifest.json"
    export_dir = run / "exports" / checkpoint_id
    export_manifest_path = export_dir / "stage_export.json"
    adapter_dir = export_dir / "adapter"
    try:
        result = M10LoRARunResult.model_validate_json(result_path.read_bytes())
        config = load_m10_lora_config(config_path)
        checkpoint = M10LoRACheckpointStore(run / "checkpoints").validate(checkpoint_id)
        export_bytes = export_manifest_path.read_bytes()
        export = export_m10_lora_stage(None, run / "exports", checkpoint_id)
        parent = resolve_evaluation_subject(root, config.model.parent_evaluation_subject)
        export_marker = _load_marker(export_dir / "COMMITTED")
    except (
        OSError,
        ValidationError,
        ValueError,
        DeploymentError,
        M10LoRAError,
    ) as exc:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA stage metadata is invalid") from exc

    try:
        adapter_sha256 = evaluation_artifact_sha256(adapter_dir, ADAPTER_FILES)
    except DeploymentError as exc:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA Adapter Artifact is invalid") from exc
    effective_sha256 = effective_artifact_sha256(parent.model_artifact_sha256, adapter_sha256)
    resumed_is_valid = (
        result.resumed_from_tokens is None
        if stage_tokens == 1_000_000
        else (
            result.resumed_from_tokens is not None
            and (1_000_000 if stage_tokens == 5_000_000 else 5_000_000)
            <= result.resumed_from_tokens
            < stage_tokens
        )
    )
    identities = (
        (result.run_id, run.name),
        (result.mode, mode),
        (result.status, status),
        (result.supervised_tokens, stage_tokens),
        (resumed_is_valid, True),
        (result.latest_checkpoint, checkpoint_id),
        (result.stage_export.adapter_artifact_sha256, adapter_sha256),
        (export.adapter_artifact_sha256, adapter_sha256),
        (export.supervised_tokens, stage_tokens),
        (checkpoint.run_id, result.run_id),
        (checkpoint.supervised_tokens, stage_tokens),
        (checkpoint.config_sha256, result.config_sha256),
        (checkpoint.dataset_version, result.dataset_version),
        (checkpoint.dataset_manifest_sha256, result.dataset_manifest_sha256),
        (checkpoint.parent_evaluation_subject, result.parent_evaluation_subject),
        (
            checkpoint.parent_evaluation_subject_sha256,
            result.parent_evaluation_subject_sha256,
        ),
        (checkpoint.parent_model_artifact_sha256, result.parent_model_artifact_sha256),
        (checkpoint.git_commit, result.git_commit),
        (checkpoint.memory_probe_sha256, result.memory_probe_sha256),
        (checkpoint.pinned, True),
        (checkpoint.pin_reason, "final" if stage_tokens == 10_000_000 else "stage"),
        (canonical_config_hash(config), result.config_sha256),
        (parent.model_version, result.parent_evaluation_subject),
        (parent.evaluation_subject_sha256, result.parent_evaluation_subject_sha256),
        (parent.model_artifact_sha256, result.parent_model_artifact_sha256),
        (
            export_marker,
            {"manifest_sha256": hashlib.sha256(export_bytes).hexdigest()},
        ),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA stage lineage is incomplete or inconsistent"
        )
    _find_probe(root, result.memory_probe_sha256)

    model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-8B",
        base_revision=config.model.revision,
        attention_architecture="gqa",
        adaptation="lora",
        model_artifact_sha256=effective_sha256,
        model_parameters=MODEL_PARAMETERS,
        training_run_id=result.run_id,
        training_checkpoint_id=result.latest_checkpoint,
        training_tokens=result.supervised_tokens,
        training_config_sha256=result.config_sha256,
        dataset_version=result.dataset_version,
        dataset_manifest_sha256=result.dataset_manifest_sha256,
        adapter_sha256=adapter_sha256,
    )
    result_sha256 = _sha256_file(result_path)
    checkpoint_manifest_sha256 = _sha256_file(checkpoint_manifest_path)
    subject_id = m10_lora_stage_evaluation_subject_id(
        model=model,
        base_model_artifact_sha256=parent.model_artifact_sha256,
        tokenizer_artifact_sha256=parent.tokenizer_artifact_sha256,
        adapter_artifact_sha256=adapter_sha256,
        source_result_sha256=result_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        memory_probe_sha256=result.memory_probe_sha256,
    )
    stage_label = f"{stage_tokens // 1_000_000}m"
    return M10LoRAStageEvaluationSubjectRecord(
        subject_id=subject_id,
        kind=cast(
            Literal["m10_agent_lora_1m", "m10_agent_lora_5m", "m10_agent_lora_10m"],
            f"m10_agent_lora_{stage_label}",
        ),
        created_at=datetime.now(UTC),
        model=model,
        model_dir=parent.model_dir,
        model_files=MODEL_FILES,
        base_model_artifact_sha256=parent.model_artifact_sha256,
        tokenizer_dir=parent.tokenizer_dir,
        tokenizer_files=TOKENIZER_FILES,
        tokenizer_artifact_sha256=parent.tokenizer_artifact_sha256,
        adapter_dir=adapter_dir,
        adapter_files=ADAPTER_FILES,
        adapter_artifact_sha256=adapter_sha256,
        effective_artifact_sha256=effective_sha256,
        source_run_dir=run,
        source_result_sha256=result_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        checkpoint_payload_sha256=checkpoint.file.sha256,
        memory_probe_sha256=result.memory_probe_sha256,
        parent_evaluation_subject="qwen3-8b-m9-base-90587dd6",
        parent_evaluation_subject_sha256=result.parent_evaluation_subject_sha256,
    )


def register_m10_lora_stage_evaluation_subject(
    *, artifact_root: Path, source_run: Path, stage_tokens: int
) -> tuple[M10LoRAStageEvaluationSubjectRecord, str]:
    """Build and atomically publish one verified Agent LoRA stage record."""

    record = build_m10_lora_stage_evaluation_subject(
        artifact_root=artifact_root,
        source_run=source_run,
        stage_tokens=stage_tokens,
    )
    try:
        return publish_m10_lora_stage_evaluation_subject(artifact_root, record)
    except DeploymentError as exc:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA registration failed") from exc


__all__ = [
    "M10LoRAStageRegistrationError",
    "build_m10_lora_stage_evaluation_subject",
    "register_m10_lora_stage_evaluation_subject",
]
