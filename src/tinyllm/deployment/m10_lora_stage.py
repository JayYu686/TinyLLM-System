"""Validated registration of M10 Qwen3-8B Agent LoRA evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
from pydantic import Field, ValidationError, model_validator
from torch import Tensor

from tinyllm.deployment.evaluation_subject import (
    M9EvaluationSubjectRecord,
    M10LoRAStageEvaluationSubjectRecord,
    effective_artifact_sha256,
    evaluation_artifact_sha256,
    m10_lora_stage_evaluation_subject_id,
    publish_m10_lora_stage_evaluation_subject,
    resolve_evaluation_subject,
    resolve_m10_lora_stage_evaluation_subject,
)
from tinyllm.deployment.registry import DeploymentError
from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.schemas.base import StrictSchema
from tinyllm.training.m10_lora import (
    M10LoRACheckpointStore,
    M10LoRAError,
    export_m10_lora_stage,
    load_m10_lora_config,
)
from tinyllm.training.m10_lora_schema import M10LoRAMemoryProbeResult, M10LoRARunResult

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
INTERMEDIATE_CHECKPOINT_STAGES = (3_000_000, 4_000_000)


class M10LoRACheckpointExportEvidence(StrictSchema):
    """Content-free evidence for a PEFT Adapter exported from a saved checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["diagnostic_only"] = "diagnostic_only"
    adapter_serialization: Literal["peft_safetensors"] = "peft_safetensors"
    adapter_safetensors_metadata: dict[Literal["format"], Literal["pt"]]
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    checkpoint_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: str = Field(min_length=1, max_length=200)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1, max_length=200)
    supervised_tokens: Literal[3_000_000, 4_000_000]
    source_diagnostic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class M10LoRAAdapterCalibrationEvidence(StrictSchema):
    """Lineage for inference-only LoRA strength calibration with unchanged weights."""

    schema_version: Literal["1.0"] = "1.0"
    calibration_version: Literal["m10-lora-alpha-calibration-v1"] = "m10-lora-alpha-calibration-v1"
    source_subject_id: str = Field(pattern=r"^qwen3-8b-m10-agent-lora-5m-[0-9a-f]{8}$")
    source_evaluation_subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adapter_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adapter_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibrated_adapter_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibrated_adapter_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: Literal[16] = 16
    source_alpha: Literal[32] = 32
    calibrated_alpha: Literal[4, 8, 16]
    relative_scale_basis_points: Literal[1250, 2500, 5000]
    selection_inputs: Literal["public_dev_and_public_bfcl"] = "public_dev_and_public_bfcl"

    @property
    def expected_scale_basis_points(self) -> int:
        return self.calibrated_alpha * 10_000 // self.source_alpha

    @model_validator(mode="after")
    def validate_calibration(self) -> M10LoRAAdapterCalibrationEvidence:
        if self.relative_scale_basis_points != self.expected_scale_basis_points:
            raise ValueError("M10 LoRA calibrated scale differs from Alpha ratio")
        if self.source_adapter_weights_sha256 != self.calibrated_adapter_weights_sha256:
            raise ValueError("M10 LoRA calibration must not change Adapter weights")
        return self


class M10LoRAAdapterInterpolationEvidence(StrictSchema):
    """Lineage for one weight-space interpolation of same-Run LoRA checkpoints."""

    schema_version: Literal["1.0"] = "1.0"
    interpolation_version: Literal["m10-lora-checkpoint-interpolation-v1"] = (
        "m10-lora-checkpoint-interpolation-v1"
    )
    early_subject_id: str = Field(pattern=r"^qwen3-8b-m10-agent-lora-1m-[0-9a-f]{8}$")
    early_evaluation_subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    early_adapter_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    late_subject_id: str = Field(pattern=r"^qwen3-8b-m10-agent-lora-5m-[0-9a-f]{8}$")
    late_evaluation_subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    late_adapter_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_run_id: str = Field(min_length=1, max_length=200)
    early_weight_basis_points: Literal[2500, 5000, 7500]
    late_weight_basis_points: Literal[2500, 5000, 7500]
    interpolation_dtype: Literal["float32_accumulation_source_dtype_output"] = (
        "float32_accumulation_source_dtype_output"
    )
    interpolated_adapter_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_inputs: Literal["public_dev_and_public_bfcl"] = "public_dev_and_public_bfcl"

    @model_validator(mode="after")
    def validate_weights(self) -> M10LoRAAdapterInterpolationEvidence:
        if self.early_weight_basis_points + self.late_weight_basis_points != 10_000:
            raise ValueError("M10 LoRA interpolation weights must sum to 10000 basis points")
        return self


class M10LoRAStageRegistrationError(RuntimeError):
    """Raised when one LoRA stage has incomplete or drifting lineage."""


def _interpolate_adapter_states(
    early: dict[str, Tensor], late: dict[str, Tensor], *, late_weight_basis_points: int
) -> dict[str, Tensor]:
    """Interpolate identical Adapter state dictionaries with float32 accumulation."""

    if late_weight_basis_points not in {2500, 5000, 7500} or set(early) != set(late):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation inputs are incompatible")
    early_weight = 10_000 - late_weight_basis_points
    result: dict[str, Tensor] = {}
    for key in sorted(early):
        early_tensor = early[key]
        late_tensor = late[key]
        if early_tensor.shape != late_tensor.shape or early_tensor.dtype != late_tensor.dtype:
            raise M10LoRAStageRegistrationError("M10 LoRA interpolation tensor topology differs")
        result[key] = (
            early_tensor.float()
            .mul(early_weight / 10_000)
            .add(late_tensor.float(), alpha=late_weight_basis_points / 10_000)
        ).to(dtype=early_tensor.dtype)
    return result


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
    probe_path = _find_probe(root, result.memory_probe_sha256)
    try:
        probe = M10LoRAMemoryProbeResult.model_validate_json(probe_path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA memory Probe is invalid") from exc
    if (
        probe.config_sha256 != result.config_sha256
        or probe.git_commit != result.git_commit
        or probe.dataset_version != result.dataset_version
        or probe.parent_evaluation_subject != result.parent_evaluation_subject
        or probe.environment_sha256 != checkpoint.environment_sha256
        or probe.hardware_compatibility_sha256 != checkpoint.hardware_sha256
    ):
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA Probe and Checkpoint lineage is inconsistent"
        )

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


def build_m10_lora_checkpoint_evaluation_subject(
    *,
    artifact_root: Path,
    source_run: Path,
    checkpoint_export_directory: Path,
    historical_subject_id: str,
) -> M10LoRAStageEvaluationSubjectRecord:
    """Promote a Dev-selected saved checkpoint into an immutable evaluation subject."""

    paths = (artifact_root, source_run, checkpoint_export_directory)
    if any(not path.is_absolute() or path.is_symlink() for path in paths):
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint paths must be absolute non-symlinks"
        )
    try:
        root = artifact_root.resolve(strict=True)
        run = source_run.resolve(strict=True)
        export_root = checkpoint_export_directory.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint paths are unavailable"
        ) from exc
    if not run.is_relative_to(root) or not export_root.is_relative_to(root):
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint Artifact escapes the Artifact Store"
        )

    diagnostic_path = export_root / "diagnostic.json"
    adapter_dir = export_root / "adapter"
    historical_record_path = (
        root / "registry" / "evaluation-subjects" / historical_subject_id / "model.json"
    )
    result_path = run / "result.json"
    config_path = run / "config.original.yaml"
    try:
        diagnostic = M10LoRACheckpointExportEvidence.model_validate_json(
            diagnostic_path.read_bytes()
        )
        diagnostic_sha256 = _sha256_file(diagnostic_path)
        historical = M9EvaluationSubjectRecord.model_validate_json(
            historical_record_path.read_bytes()
        )
        result = M10LoRARunResult.model_validate_json(result_path.read_bytes())
        config = load_m10_lora_config(config_path)
        checkpoint = M10LoRACheckpointStore(run / "checkpoints").validate(diagnostic.checkpoint_id)
        parent = resolve_evaluation_subject(root, config.model.parent_evaluation_subject)
        adapter_sha256 = evaluation_artifact_sha256(adapter_dir, ADAPTER_FILES)
    except (
        OSError,
        ValidationError,
        ValueError,
        DeploymentError,
        M10LoRAError,
    ) as exc:
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint evidence is invalid"
        ) from exc

    stage_tokens = diagnostic.supervised_tokens
    if stage_tokens not in INTERMEDIATE_CHECKPOINT_STAGES:
        raise M10LoRAStageRegistrationError("M10 Agent LoRA checkpoint stage is unsupported")
    effective_sha256 = effective_artifact_sha256(parent.model_artifact_sha256, adapter_sha256)
    identities = (
        (historical.subject_id, historical_subject_id),
        (historical.kind, "historical_lora"),
        (historical.source_evidence_sha256, diagnostic_sha256),
        (historical.adapter_dir, adapter_dir),
        (historical.adapter_artifact_sha256, adapter_sha256),
        (historical.effective_artifact_sha256, effective_sha256),
        (historical.model.training_tokens, stage_tokens),
        (historical.model.training_checkpoint_id, diagnostic.checkpoint_id),
        (historical.model.training_run_id, run.name),
        (historical.model.training_config_sha256, diagnostic.config_sha256),
        (historical.model.dataset_version, diagnostic.dataset_version),
        (historical.model.dataset_manifest_sha256, diagnostic.dataset_manifest_sha256),
        (diagnostic.run_id, run.name),
        (
            diagnostic.checkpoint_manifest_sha256,
            _sha256_file(run / "checkpoints" / diagnostic.checkpoint_id / "manifest.json"),
        ),
        (diagnostic.checkpoint_payload_sha256, checkpoint.file.sha256),
        (checkpoint.supervised_tokens, stage_tokens),
        (checkpoint.config_sha256, diagnostic.config_sha256),
        (checkpoint.dataset_version, diagnostic.dataset_version),
        (checkpoint.dataset_manifest_sha256, diagnostic.dataset_manifest_sha256),
        (checkpoint.parent_evaluation_subject, result.parent_evaluation_subject),
        (checkpoint.parent_evaluation_subject_sha256, result.parent_evaluation_subject_sha256),
        (checkpoint.parent_model_artifact_sha256, result.parent_model_artifact_sha256),
        (checkpoint.memory_probe_sha256, result.memory_probe_sha256),
        (canonical_config_hash(config), diagnostic.config_sha256),
        (result.run_id, run.name),
        (result.config_sha256, diagnostic.config_sha256),
        (result.dataset_version, diagnostic.dataset_version),
        (result.dataset_manifest_sha256, diagnostic.dataset_manifest_sha256),
        (parent.model_version, result.parent_evaluation_subject),
        (parent.evaluation_subject_sha256, result.parent_evaluation_subject_sha256),
        (parent.model_artifact_sha256, result.parent_model_artifact_sha256),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint lineage is incomplete or inconsistent"
        )
    _find_probe(root, checkpoint.memory_probe_sha256)

    model = historical.model
    result_sha256 = _sha256_file(result_path)
    subject_id = m10_lora_stage_evaluation_subject_id(
        model=model,
        base_model_artifact_sha256=parent.model_artifact_sha256,
        tokenizer_artifact_sha256=parent.tokenizer_artifact_sha256,
        adapter_artifact_sha256=adapter_sha256,
        source_result_sha256=result_sha256,
        checkpoint_manifest_sha256=diagnostic.checkpoint_manifest_sha256,
        memory_probe_sha256=checkpoint.memory_probe_sha256,
        checkpoint_export_evidence_sha256=diagnostic_sha256,
    )
    stage_label = f"{stage_tokens // 1_000_000}m"
    return M10LoRAStageEvaluationSubjectRecord(
        subject_id=subject_id,
        kind=cast(
            Literal["m10_agent_lora_3m", "m10_agent_lora_4m"],
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
        checkpoint_manifest_sha256=diagnostic.checkpoint_manifest_sha256,
        checkpoint_payload_sha256=checkpoint.file.sha256,
        memory_probe_sha256=checkpoint.memory_probe_sha256,
        checkpoint_export_evidence_sha256=diagnostic_sha256,
        parent_evaluation_subject="qwen3-8b-m9-base-90587dd6",
        parent_evaluation_subject_sha256=result.parent_evaluation_subject_sha256,
    )


def register_m10_lora_checkpoint_evaluation_subject(
    *,
    artifact_root: Path,
    source_run: Path,
    checkpoint_export_directory: Path,
    historical_subject_id: str,
) -> tuple[M10LoRAStageEvaluationSubjectRecord, str]:
    """Validate and atomically publish one selected intermediate checkpoint."""

    record = build_m10_lora_checkpoint_evaluation_subject(
        artifact_root=artifact_root,
        source_run=source_run,
        checkpoint_export_directory=checkpoint_export_directory,
        historical_subject_id=historical_subject_id,
    )
    try:
        return publish_m10_lora_stage_evaluation_subject(artifact_root, record)
    except DeploymentError as exc:
        raise M10LoRAStageRegistrationError(
            "M10 Agent LoRA checkpoint registration failed"
        ) from exc


def create_m10_lora_interpolated_adapter(
    *,
    artifact_root: Path,
    early_subject_id: str,
    late_subject_id: str,
    output_directory: Path,
    late_weight_basis_points: Literal[2500, 5000, 7500],
) -> M10LoRAAdapterInterpolationEvidence:
    """Create one content-addressable same-Run checkpoint interpolation artifact."""

    from safetensors.torch import load_file, save_file  # type: ignore[import-not-found]

    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not output_directory.is_absolute()
        or output_directory.is_symlink()
        or output_directory.exists()
    ):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation paths are unsafe")
    try:
        root = artifact_root.resolve(strict=True)
        output_parent = output_directory.parent.resolve(strict=True)
        early_path = root / "registry" / "evaluation-subjects" / early_subject_id / "model.json"
        late_path = root / "registry" / "evaluation-subjects" / late_subject_id / "model.json"
        early_payload = early_path.read_bytes()
        late_payload = late_path.read_bytes()
        early = M10LoRAStageEvaluationSubjectRecord.model_validate_json(early_payload)
        late = M10LoRAStageEvaluationSubjectRecord.model_validate_json(late_payload)
        resolve_m10_lora_stage_evaluation_subject(root, early_subject_id)
        resolve_m10_lora_stage_evaluation_subject(root, late_subject_id)
    except (OSError, ValidationError, ValueError, DeploymentError) as exc:
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation sources are invalid") from exc
    if not output_parent.is_relative_to(root):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation escapes the Artifact Store")
    if (
        early.kind != "m10_agent_lora_1m"
        or late.kind != "m10_agent_lora_5m"
        or early.model.training_run_id is None
        or early.model.training_run_id != late.model.training_run_id
        or early.model.training_config_sha256 != late.model.training_config_sha256
        or early.model.dataset_version != late.model.dataset_version
        or early.model.dataset_manifest_sha256 != late.model.dataset_manifest_sha256
        or early.base_model_artifact_sha256 != late.base_model_artifact_sha256
        or early.tokenizer_artifact_sha256 != late.tokenizer_artifact_sha256
    ):
        raise M10LoRAStageRegistrationError(
            "M10 LoRA interpolation requires compatible 1M and 5M same-Run sources"
        )
    early_config = early.adapter_dir / "adapter_config.json"
    late_config = late.adapter_dir / "adapter_config.json"
    if early_config.read_bytes() != late_config.read_bytes():
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation Adapter configs differ")

    temporary = output_parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        shutil.copyfile(late_config, temporary / "adapter_config.json")
        interpolated = _interpolate_adapter_states(
            load_file(early.adapter_dir / "adapter_model.safetensors", device="cpu"),
            load_file(late.adapter_dir / "adapter_model.safetensors", device="cpu"),
            late_weight_basis_points=late_weight_basis_points,
        )
        save_file(
            interpolated,
            temporary / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        adapter_sha256 = evaluation_artifact_sha256(temporary, ADAPTER_FILES)
        evidence = M10LoRAAdapterInterpolationEvidence(
            early_subject_id=early.subject_id,
            early_evaluation_subject_sha256=hashlib.sha256(early_payload).hexdigest(),
            early_adapter_artifact_sha256=early.adapter_artifact_sha256,
            late_subject_id=late.subject_id,
            late_evaluation_subject_sha256=hashlib.sha256(late_payload).hexdigest(),
            late_adapter_artifact_sha256=late.adapter_artifact_sha256,
            training_run_id=late.model.training_run_id,
            early_weight_basis_points=cast(
                Literal[2500, 5000, 7500], 10_000 - late_weight_basis_points
            ),
            late_weight_basis_points=late_weight_basis_points,
            interpolated_adapter_artifact_sha256=adapter_sha256,
        )
        (temporary / "interpolation.json").write_text(
            json.dumps(
                evidence.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return evidence


def build_m10_lora_interpolated_evaluation_subject(
    *, artifact_root: Path, interpolated_adapter_dir: Path
) -> M10LoRAStageEvaluationSubjectRecord:
    """Validate and bind one same-Run checkpoint interpolation as an Evaluation subject."""

    from safetensors.torch import load_file

    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not interpolated_adapter_dir.is_absolute()
        or interpolated_adapter_dir.is_symlink()
    ):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation registration paths are unsafe")
    try:
        root = artifact_root.resolve(strict=True)
        adapter_dir = interpolated_adapter_dir.resolve(strict=True)
        evidence_path = adapter_dir / "interpolation.json"
        evidence = M10LoRAAdapterInterpolationEvidence.model_validate_json(
            evidence_path.read_bytes()
        )
        evidence_sha256 = _sha256_file(evidence_path)
        early_path = (
            root / "registry" / "evaluation-subjects" / evidence.early_subject_id / "model.json"
        )
        late_path = (
            root / "registry" / "evaluation-subjects" / evidence.late_subject_id / "model.json"
        )
        early_payload = early_path.read_bytes()
        late_payload = late_path.read_bytes()
        early = M10LoRAStageEvaluationSubjectRecord.model_validate_json(early_payload)
        late = M10LoRAStageEvaluationSubjectRecord.model_validate_json(late_payload)
        early_resolved = resolve_m10_lora_stage_evaluation_subject(root, early.subject_id)
        late_resolved = resolve_m10_lora_stage_evaluation_subject(root, late.subject_id)
        interpolated_sha256 = evaluation_artifact_sha256(adapter_dir, ADAPTER_FILES)
        expected_state = _interpolate_adapter_states(
            load_file(early.adapter_dir / "adapter_model.safetensors", device="cpu"),
            load_file(late.adapter_dir / "adapter_model.safetensors", device="cpu"),
            late_weight_basis_points=evidence.late_weight_basis_points,
        )
        observed_state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
        early_config = (early.adapter_dir / "adapter_config.json").read_bytes()
        late_config = (late.adapter_dir / "adapter_config.json").read_bytes()
        interpolated_config = (adapter_dir / "adapter_config.json").read_bytes()
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        DeploymentError,
        RuntimeError,
    ) as exc:
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation evidence is invalid") from exc
    if not adapter_dir.is_relative_to(root):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation escapes the Artifact Store")
    identities = (
        (early.subject_id, evidence.early_subject_id),
        (late.subject_id, evidence.late_subject_id),
        (hashlib.sha256(early_payload).hexdigest(), evidence.early_evaluation_subject_sha256),
        (hashlib.sha256(late_payload).hexdigest(), evidence.late_evaluation_subject_sha256),
        (early.adapter_artifact_sha256, evidence.early_adapter_artifact_sha256),
        (late.adapter_artifact_sha256, evidence.late_adapter_artifact_sha256),
        (late.model.training_run_id, evidence.training_run_id),
        (interpolated_sha256, evidence.interpolated_adapter_artifact_sha256),
        (early_resolved.model_artifact_sha256, early.effective_artifact_sha256),
        (late_resolved.model_artifact_sha256, late.effective_artifact_sha256),
        (early_config, late_config),
        (late_config, interpolated_config),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation lineage differs")
    if set(expected_state) != set(observed_state) or any(
        not torch.equal(expected_state[key], observed_state[key]) for key in expected_state
    ):
        raise M10LoRAStageRegistrationError("M10 LoRA interpolated weights differ from evidence")

    effective_sha256 = effective_artifact_sha256(
        late.base_model_artifact_sha256, interpolated_sha256
    )
    model = late.model.model_copy(
        update={
            "model_artifact_sha256": effective_sha256,
            "adapter_sha256": interpolated_sha256,
        }
    )
    subject_id = m10_lora_stage_evaluation_subject_id(
        model=model,
        base_model_artifact_sha256=late.base_model_artifact_sha256,
        tokenizer_artifact_sha256=late.tokenizer_artifact_sha256,
        adapter_artifact_sha256=interpolated_sha256,
        source_result_sha256=late.source_result_sha256,
        checkpoint_manifest_sha256=late.checkpoint_manifest_sha256,
        memory_probe_sha256=late.memory_probe_sha256,
        adapter_interpolation_evidence_sha256=evidence_sha256,
    )
    return M10LoRAStageEvaluationSubjectRecord(
        **late.model_dump(
            mode="python",
            exclude={
                "subject_id",
                "created_at",
                "model",
                "adapter_dir",
                "adapter_artifact_sha256",
                "effective_artifact_sha256",
                "adapter_calibration_evidence_sha256",
                "adapter_interpolation_evidence_sha256",
            },
        ),
        subject_id=subject_id,
        created_at=datetime.now(UTC),
        model=model,
        adapter_dir=adapter_dir,
        adapter_artifact_sha256=interpolated_sha256,
        effective_artifact_sha256=effective_sha256,
        adapter_interpolation_evidence_sha256=evidence_sha256,
    )


def register_m10_lora_interpolated_evaluation_subject(
    *, artifact_root: Path, interpolated_adapter_dir: Path
) -> tuple[M10LoRAStageEvaluationSubjectRecord, str]:
    """Validate and atomically publish one interpolated M10 Agent LoRA subject."""

    record = build_m10_lora_interpolated_evaluation_subject(
        artifact_root=artifact_root,
        interpolated_adapter_dir=interpolated_adapter_dir,
    )
    try:
        return publish_m10_lora_stage_evaluation_subject(artifact_root, record)
    except DeploymentError as exc:
        raise M10LoRAStageRegistrationError("M10 LoRA interpolation registration failed") from exc


def build_m10_lora_calibrated_evaluation_subject(
    *, artifact_root: Path, source_subject_id: str, calibrated_adapter_dir: Path
) -> M10LoRAStageEvaluationSubjectRecord:
    """Build an immutable subject whose only change is PEFT LoRA Alpha."""

    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not calibrated_adapter_dir.is_absolute()
        or calibrated_adapter_dir.is_symlink()
    ):
        raise M10LoRAStageRegistrationError(
            "M10 LoRA calibration paths must be absolute non-symlinks"
        )
    try:
        root = artifact_root.resolve(strict=True)
        calibrated_dir = calibrated_adapter_dir.resolve(strict=True)
        source_path = root / "registry" / "evaluation-subjects" / source_subject_id / "model.json"
        source_payload = source_path.read_bytes()
        source = M10LoRAStageEvaluationSubjectRecord.model_validate_json(source_payload)
        resolved = resolve_m10_lora_stage_evaluation_subject(root, source_subject_id)
        evidence_path = calibrated_dir / "calibration.json"
        evidence = M10LoRAAdapterCalibrationEvidence.model_validate_json(evidence_path.read_bytes())
        evidence_sha256 = _sha256_file(evidence_path)
        source_config: object = json.loads(
            (source.adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        calibrated_config: object = json.loads(
            (calibrated_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        calibrated_sha256 = evaluation_artifact_sha256(calibrated_dir, ADAPTER_FILES)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        DeploymentError,
    ) as exc:
        raise M10LoRAStageRegistrationError("M10 LoRA calibration evidence is invalid") from exc
    if not calibrated_dir.is_relative_to(root):
        raise M10LoRAStageRegistrationError(
            "M10 LoRA calibrated Adapter escapes the Artifact Store"
        )
    if source.kind != "m10_agent_lora_5m" or source.model.training_tokens != 5_000_000:
        raise M10LoRAStageRegistrationError("M10 LoRA calibration requires a 5M source")
    if not isinstance(source_config, dict) or not isinstance(calibrated_config, dict):
        raise M10LoRAStageRegistrationError("M10 LoRA Adapter configuration is invalid")
    expected_config = dict(source_config)
    expected_config["lora_alpha"] = evidence.calibrated_alpha
    identities = (
        (source.subject_id, evidence.source_subject_id),
        (hashlib.sha256(source_payload).hexdigest(), evidence.source_evaluation_subject_sha256),
        (source.adapter_artifact_sha256, evidence.source_adapter_artifact_sha256),
        (
            _sha256_file(source.adapter_dir / "adapter_model.safetensors"),
            evidence.source_adapter_weights_sha256,
        ),
        (
            _sha256_file(calibrated_dir / "adapter_model.safetensors"),
            evidence.calibrated_adapter_weights_sha256,
        ),
        (calibrated_sha256, evidence.calibrated_adapter_artifact_sha256),
        (source_config.get("r"), evidence.rank),
        (source_config.get("lora_alpha"), evidence.source_alpha),
        (calibrated_config, expected_config),
        (resolved.model_artifact_sha256, source.effective_artifact_sha256),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAStageRegistrationError(
            "M10 LoRA calibration changed weights, metadata, or source lineage"
        )

    effective_sha256 = effective_artifact_sha256(
        source.base_model_artifact_sha256, calibrated_sha256
    )
    model = source.model.model_copy(
        update={
            "model_artifact_sha256": effective_sha256,
            "adapter_sha256": calibrated_sha256,
        }
    )
    subject_id = m10_lora_stage_evaluation_subject_id(
        model=model,
        base_model_artifact_sha256=source.base_model_artifact_sha256,
        tokenizer_artifact_sha256=source.tokenizer_artifact_sha256,
        adapter_artifact_sha256=calibrated_sha256,
        source_result_sha256=source.source_result_sha256,
        checkpoint_manifest_sha256=source.checkpoint_manifest_sha256,
        memory_probe_sha256=source.memory_probe_sha256,
        adapter_calibration_evidence_sha256=evidence_sha256,
    )
    return M10LoRAStageEvaluationSubjectRecord(
        **source.model_dump(
            mode="python",
            exclude={
                "subject_id",
                "created_at",
                "model",
                "adapter_dir",
                "adapter_artifact_sha256",
                "effective_artifact_sha256",
                "adapter_calibration_evidence_sha256",
            },
        ),
        subject_id=subject_id,
        created_at=datetime.now(UTC),
        model=model,
        adapter_dir=calibrated_dir,
        adapter_artifact_sha256=calibrated_sha256,
        effective_artifact_sha256=effective_sha256,
        adapter_calibration_evidence_sha256=evidence_sha256,
    )


def register_m10_lora_calibrated_evaluation_subject(
    *, artifact_root: Path, source_subject_id: str, calibrated_adapter_dir: Path
) -> tuple[M10LoRAStageEvaluationSubjectRecord, str]:
    """Validate and publish one inference-calibrated M10 Agent LoRA subject."""

    record = build_m10_lora_calibrated_evaluation_subject(
        artifact_root=artifact_root,
        source_subject_id=source_subject_id,
        calibrated_adapter_dir=calibrated_adapter_dir,
    )
    try:
        return publish_m10_lora_stage_evaluation_subject(artifact_root, record)
    except DeploymentError as exc:
        raise M10LoRAStageRegistrationError("M10 LoRA calibration registration failed") from exc


__all__ = [
    "M10LoRACheckpointExportEvidence",
    "M10LoRAAdapterCalibrationEvidence",
    "M10LoRAAdapterInterpolationEvidence",
    "M10LoRAStageRegistrationError",
    "build_m10_lora_checkpoint_evaluation_subject",
    "build_m10_lora_calibrated_evaluation_subject",
    "build_m10_lora_interpolated_evaluation_subject",
    "create_m10_lora_interpolated_adapter",
    "build_m10_lora_stage_evaluation_subject",
    "register_m10_lora_checkpoint_evaluation_subject",
    "register_m10_lora_calibrated_evaluation_subject",
    "register_m10_lora_interpolated_evaluation_subject",
    "register_m10_lora_stage_evaluation_subject",
]
