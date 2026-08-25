"""Immutable evaluation-only model subjects for measured comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from tinyllm.deployment.registry import (
    DeploymentError,
    DeploymentErrorCode,
    _artifact_set_sha256,
    _validate_model_configuration,
)
from tinyllm.deployment.schema import ResolvedModel
from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas.base import StrictSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
M9_SUBJECT_PATTERN = r"^qwen3-8b-m9-(base|historical-lora)-[0-9a-f]{8}$"
M10_STAGE_SUBJECT_PATTERN = r"^qwen3-0-6b-m10-full-sft-(1m|5m)-[0-9a-f]{8}$"
M10_LORA_STAGE_SUBJECT_PATTERN = r"^qwen3-8b-m10-agent-lora-(1m|5m|10m)-[0-9a-f]{8}$"
SUBJECT_PATTERN = (
    r"^(qwen3-8b-m9-(base|historical-lora)|qwen3-0-6b-m10-full-sft-(1m|5m)|"
    r"qwen3-8b-m10-agent-lora-(1m|5m|10m))"
    r"-[0-9a-f]{8}$"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def effective_artifact_sha256(
    model_artifact_sha256: str, adapter_artifact_sha256: str | None
) -> str:
    """Identify the exact effective weights without merging or copying the base."""

    if adapter_artifact_sha256 is None:
        return model_artifact_sha256
    return _canonical_sha256(
        {
            "base_model_artifact_sha256": model_artifact_sha256,
            "adapter_artifact_sha256": adapter_artifact_sha256,
        }
    )


def evaluation_artifact_sha256(root: Path, names: tuple[str, ...]) -> str:
    """Hash an explicit top-level deployable Artifact set with M7 semantics."""

    return _artifact_set_sha256(root, names)


def evaluation_subject_id(
    *,
    kind: Literal["base", "historical_lora"],
    model: M6ModelIdentity,
    base_model_artifact_sha256: str,
    tokenizer_artifact_sha256: str,
    adapter_artifact_sha256: str | None,
    source_evidence_sha256: str,
) -> str:
    """Derive the immutable public-safe identity of one M9 evaluation subject."""

    identity = _canonical_sha256(
        {
            "kind": kind,
            "model": model.to_dict(),
            "base_model_artifact_sha256": base_model_artifact_sha256,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "adapter_artifact_sha256": adapter_artifact_sha256,
            "source_evidence_sha256": source_evidence_sha256,
        }
    )
    label = "base" if kind == "base" else "historical-lora"
    return f"qwen3-8b-m9-{label}-{identity[:8]}"


class M9EvaluationSubjectRecord(StrictSchema):
    """Private, immutable identity for a model that cannot enter Production Registry."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Evaluation"] = "Evaluation"
    subject_id: str = Field(pattern=M9_SUBJECT_PATTERN)
    kind: Literal["base", "historical_lora"]
    created_at: datetime
    model: M6ModelIdentity
    model_dir: Path
    model_files: tuple[str, ...] = Field(min_length=2, max_length=20)
    base_model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_dir: Path
    tokenizer_files: tuple[str, ...] = Field(min_length=2, max_length=8)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_dir: Path | None = None
    adapter_files: tuple[str, ...] = Field(default=(), max_length=8)
    adapter_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    effective_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    production_eligible: Literal[False] = False

    @field_validator("model_files", "tokenizer_files", "adapter_files", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("model_files", "tokenizer_files", "adapter_files")
    @classmethod
    def validate_file_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("evaluation subject file names must be unique and sorted")
        if any(Path(name).name != name or name in {".", ".."} for name in value):
            raise ValueError("evaluation subject accepts top-level file names only")
        return value

    @field_validator("model_dir", "tokenizer_dir", "adapter_dir")
    @classmethod
    def require_absolute_paths(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("evaluation subject paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> M9EvaluationSubjectRecord:
        if self.created_at.tzinfo is None:
            raise ValueError("evaluation subject timestamp must be timezone-aware")
        if self.model.repository != "Qwen/Qwen3-8B":
            raise ValueError("M9 evaluation subjects are frozen to Qwen3-8B")
        if "config.json" not in self.model_files or not any(
            name.endswith(".safetensors") for name in self.model_files
        ):
            raise ValueError("evaluation subject requires Config and Safetensors files")
        if set(self.tokenizer_files) != {"tokenizer.json", "tokenizer_config.json"}:
            raise ValueError("evaluation subject Tokenizer file set differs")
        expected_effective = effective_artifact_sha256(
            self.base_model_artifact_sha256, self.adapter_artifact_sha256
        )
        if self.effective_artifact_sha256 != expected_effective:
            raise ValueError("evaluation subject effective Artifact hash differs")
        if self.model.model_artifact_sha256 != expected_effective:
            raise ValueError("evaluation subject model identity hash differs")
        expected_id = evaluation_subject_id(
            kind=self.kind,
            model=self.model,
            base_model_artifact_sha256=self.base_model_artifact_sha256,
            tokenizer_artifact_sha256=self.tokenizer_artifact_sha256,
            adapter_artifact_sha256=self.adapter_artifact_sha256,
            source_evidence_sha256=self.source_evidence_sha256,
        )
        if self.subject_id != expected_id:
            raise ValueError("evaluation subject ID differs from immutable inputs")
        adapter_present = self.adapter_dir is not None
        if self.kind == "base":
            if self.model.role != "base" or self.model.adaptation != "base":
                raise ValueError("base evaluation subject requires Base model identity")
            if adapter_present or self.adapter_files or self.adapter_artifact_sha256 is not None:
                raise ValueError("base evaluation subject cannot contain an Adapter")
        else:
            if self.model.role != "candidate" or self.model.adaptation != "lora":
                raise ValueError("historical LoRA subject requires trained LoRA identity")
            if (
                not adapter_present
                or not self.adapter_files
                or self.adapter_artifact_sha256 is None
            ):
                raise ValueError("historical LoRA subject requires an Adapter Artifact")
            if self.model.adapter_sha256 != self.adapter_artifact_sha256:
                raise ValueError("historical LoRA identity and Adapter hash differ")
        return self


def m10_stage_evaluation_subject_id(
    *,
    model: M6ModelIdentity,
    tokenizer_artifact_sha256: str,
    source_result_sha256: str,
    checkpoint_manifest_sha256: str,
    environment_sha256: str,
) -> str:
    """Derive one immutable 1M/5M Full-SFT evaluation-only identity."""

    if model.training_tokens not in {1_000_000, 5_000_000}:
        raise ValueError("M10 evaluation subjects are limited to the 1M/5M stages")
    stage_label = f"{model.training_tokens // 1_000_000}m"
    kind = f"m10_full_sft_{stage_label}"

    identity = _canonical_sha256(
        {
            "kind": kind,
            "model": model.to_dict(),
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "source_result_sha256": source_result_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "environment_sha256": environment_sha256,
        }
    )
    return f"qwen3-0-6b-m10-full-sft-{stage_label}-{identity[:8]}"


class M10StageEvaluationSubjectRecord(StrictSchema):
    """Private, immutable 1M/5M Full-SFT stage exposed only to evaluation flows."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Evaluation"] = "Evaluation"
    subject_id: str = Field(pattern=M10_STAGE_SUBJECT_PATTERN)
    kind: Literal["m10_full_sft_1m", "m10_full_sft_5m"] = "m10_full_sft_5m"
    created_at: datetime
    model: M6ModelIdentity
    model_dir: Path
    model_files: tuple[str, ...] = Field(min_length=3, max_length=3)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_dir: Path
    tokenizer_files: tuple[str, ...] = Field(min_length=2, max_length=2)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_run_dir: Path
    source_result_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    production_eligible: Literal[False] = False

    @field_validator("model_files", "tokenizer_files", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("model_files", "tokenizer_files")
    @classmethod
    def validate_file_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("M10 evaluation subject file names must be unique and sorted")
        if any(Path(name).name != name or name in {".", ".."} for name in value):
            raise ValueError("M10 evaluation subject accepts top-level file names only")
        return value

    @field_validator("model_dir", "tokenizer_dir", "source_run_dir")
    @classmethod
    def require_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("M10 evaluation subject paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> M10StageEvaluationSubjectRecord:
        if self.created_at.tzinfo is None:
            raise ValueError("M10 evaluation subject timestamp must be timezone-aware")
        if set(self.model_files) != {
            "config.json",
            "generation_config.json",
            "model.safetensors",
        }:
            raise ValueError("M10 evaluation subject model file set differs")
        if set(self.tokenizer_files) != {"tokenizer.json", "tokenizer_config.json"}:
            raise ValueError("M10 evaluation subject Tokenizer file set differs")
        expected_tokens = 1_000_000 if self.kind == "m10_full_sft_1m" else 5_000_000
        expected_checkpoint = f"checkpoint-tokens-{expected_tokens:010d}"
        if (
            self.model.role != "candidate"
            or self.model.repository != "Qwen/Qwen3-0.6B"
            or self.model.adaptation != "full_sft"
            or self.model.training_checkpoint_id != expected_checkpoint
            or self.model.training_tokens != expected_tokens
        ):
            raise ValueError("M10 evaluation subject stage identity differs")
        if self.model.model_artifact_sha256 != self.model_artifact_sha256:
            raise ValueError("M10 evaluation subject model identity hash differs")
        expected_id = m10_stage_evaluation_subject_id(
            model=self.model,
            tokenizer_artifact_sha256=self.tokenizer_artifact_sha256,
            source_result_sha256=self.source_result_sha256,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            environment_sha256=self.environment_sha256,
        )
        if self.subject_id != expected_id:
            raise ValueError("M10 evaluation subject ID differs from immutable inputs")
        return self


def m10_lora_stage_evaluation_subject_id(
    *,
    model: M6ModelIdentity,
    base_model_artifact_sha256: str,
    tokenizer_artifact_sha256: str,
    adapter_artifact_sha256: str,
    source_result_sha256: str,
    checkpoint_manifest_sha256: str,
    memory_probe_sha256: str,
) -> str:
    """Derive one immutable M10 Agent LoRA stage identity."""

    if model.training_tokens not in {1_000_000, 5_000_000, 10_000_000}:
        raise ValueError("M10 Agent LoRA subjects are limited to 1M/5M/10M stages")
    stage_label = f"{model.training_tokens // 1_000_000}m"
    identity = _canonical_sha256(
        {
            "kind": f"m10_agent_lora_{stage_label}",
            "model": model.to_dict(),
            "base_model_artifact_sha256": base_model_artifact_sha256,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "adapter_artifact_sha256": adapter_artifact_sha256,
            "source_result_sha256": source_result_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "memory_probe_sha256": memory_probe_sha256,
        }
    )
    return f"qwen3-8b-m10-agent-lora-{stage_label}-{identity[:8]}"


class M10LoRAStageEvaluationSubjectRecord(StrictSchema):
    """Private, immutable M10 Agent LoRA stage for serving and evaluation only."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Evaluation"] = "Evaluation"
    subject_id: str = Field(pattern=M10_LORA_STAGE_SUBJECT_PATTERN)
    kind: Literal["m10_agent_lora_1m", "m10_agent_lora_5m", "m10_agent_lora_10m"]
    created_at: datetime
    model: M6ModelIdentity
    model_dir: Path
    model_files: tuple[str, ...] = Field(min_length=7, max_length=7)
    base_model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_dir: Path
    tokenizer_files: tuple[str, ...] = Field(min_length=2, max_length=2)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_dir: Path
    adapter_files: tuple[str, ...] = Field(min_length=2, max_length=2)
    adapter_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    effective_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_run_dir: Path
    source_result_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    memory_probe_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_evaluation_subject: Literal["qwen3-8b-m9-base-90587dd6"]
    parent_evaluation_subject_sha256: Literal[
        "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
    ]
    production_eligible: Literal[False] = False

    @field_validator("model_files", "tokenizer_files", "adapter_files", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("model_files", "tokenizer_files", "adapter_files")
    @classmethod
    def validate_file_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or tuple(sorted(value)) != value:
            raise ValueError("M10 Agent LoRA file names must be unique and sorted")
        if any(Path(name).name != name or name in {".", ".."} for name in value):
            raise ValueError("M10 Agent LoRA subjects accept top-level file names only")
        return value

    @field_validator("model_dir", "tokenizer_dir", "adapter_dir", "source_run_dir")
    @classmethod
    def require_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("M10 Agent LoRA subject paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> M10LoRAStageEvaluationSubjectRecord:
        if self.created_at.tzinfo is None:
            raise ValueError("M10 Agent LoRA subject timestamp must be timezone-aware")
        if set(self.tokenizer_files) != {"tokenizer.json", "tokenizer_config.json"}:
            raise ValueError("M10 Agent LoRA Tokenizer file set differs")
        if set(self.adapter_files) != {"adapter_config.json", "adapter_model.safetensors"}:
            raise ValueError("M10 Agent LoRA Adapter file set differs")
        expected_tokens = {
            "m10_agent_lora_1m": 1_000_000,
            "m10_agent_lora_5m": 5_000_000,
            "m10_agent_lora_10m": 10_000_000,
        }[self.kind]
        expected_checkpoint = f"checkpoint-tokens-{expected_tokens:010d}"
        expected_effective = effective_artifact_sha256(
            self.base_model_artifact_sha256, self.adapter_artifact_sha256
        )
        if (
            self.model.role != "candidate"
            or self.model.repository != "Qwen/Qwen3-8B"
            or self.model.adaptation != "lora"
            or self.model.adapter_sha256 != self.adapter_artifact_sha256
            or self.model.training_checkpoint_id != expected_checkpoint
            or self.model.training_tokens != expected_tokens
            or self.effective_artifact_sha256 != expected_effective
            or self.model.model_artifact_sha256 != expected_effective
        ):
            raise ValueError("M10 Agent LoRA subject stage identity differs")
        expected_id = m10_lora_stage_evaluation_subject_id(
            model=self.model,
            base_model_artifact_sha256=self.base_model_artifact_sha256,
            tokenizer_artifact_sha256=self.tokenizer_artifact_sha256,
            adapter_artifact_sha256=self.adapter_artifact_sha256,
            source_result_sha256=self.source_result_sha256,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            memory_probe_sha256=self.memory_probe_sha256,
        )
        if self.subject_id != expected_id:
            raise ValueError("M10 Agent LoRA subject ID differs from immutable inputs")
        return self


class ResolvedEvaluationSubject(StrictSchema):
    """Hash-verified path projection accepted only by serving and evaluation flows."""

    schema_version: Literal["1.0"] = "1.0"
    requested_ref: str = Field(pattern=SUBJECT_PATTERN)
    status: Literal["Evaluation"] = "Evaluation"
    model_version: str = Field(pattern=SUBJECT_PATTERN)
    evaluation_subject_sha256: str = Field(pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    model_dir: Path
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_dir: Path
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_dir: Path | None = None
    adapter_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    verified_at: datetime

    @field_validator("model_dir", "tokenizer_dir", "adapter_dir")
    @classmethod
    def require_absolute_paths(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("resolved evaluation subject paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> ResolvedEvaluationSubject:
        if self.requested_ref != self.model_version:
            raise ValueError("resolved evaluation subject identity differs")
        if self.verified_at.tzinfo is None:
            raise ValueError("evaluation subject verification timestamp must be timezone-aware")
        if self.model.model_artifact_sha256 != self.model_artifact_sha256:
            raise ValueError("resolved model hash differs from evaluation identity")
        if (self.adapter_dir is None) != (self.adapter_artifact_sha256 is None):
            raise ValueError("resolved Adapter path and hash must appear together")
        return self


ServingModel: TypeAlias = ResolvedModel | ResolvedEvaluationSubject


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _validate_contained_paths(artifact_root: Path, record: M9EvaluationSubjectRecord) -> None:
    try:
        root = artifact_root.resolve(strict=True)
        directories = (record.model_dir, record.tokenizer_dir, record.adapter_dir)
        for directory in directories:
            if directory is None:
                continue
            resolved = directory.resolve(strict=True)
            if directory.is_symlink() or not resolved.is_relative_to(root):
                raise DeploymentError(
                    DeploymentErrorCode.UNSAFE_ARTIFACT,
                    "Evaluation subject Artifact escapes the Artifact Store",
                )
    except (FileNotFoundError, OSError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND,
            "Evaluation subject Artifact directory is unavailable",
        ) from exc


def _validate_m10_contained_paths(
    artifact_root: Path, record: M10StageEvaluationSubjectRecord
) -> None:
    try:
        root = artifact_root.resolve(strict=True)
        for directory in (record.model_dir, record.tokenizer_dir, record.source_run_dir):
            resolved = directory.resolve(strict=True)
            if directory.is_symlink() or not resolved.is_relative_to(root):
                raise DeploymentError(
                    DeploymentErrorCode.UNSAFE_ARTIFACT,
                    "M10 evaluation subject Artifact escapes the Artifact Store",
                )
    except (FileNotFoundError, OSError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND,
            "M10 evaluation subject Artifact directory is unavailable",
        ) from exc


def _validate_m10_lora_contained_paths(
    artifact_root: Path, record: M10LoRAStageEvaluationSubjectRecord
) -> None:
    try:
        root = artifact_root.resolve(strict=True)
        for directory in (
            record.model_dir,
            record.tokenizer_dir,
            record.adapter_dir,
            record.source_run_dir,
        ):
            resolved = directory.resolve(strict=True)
            if directory.is_symlink() or not resolved.is_relative_to(root):
                raise DeploymentError(
                    DeploymentErrorCode.UNSAFE_ARTIFACT,
                    "M10 Agent LoRA Artifact escapes the Artifact Store",
                )
    except (FileNotFoundError, OSError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND,
            "M10 Agent LoRA Artifact directory is unavailable",
        ) from exc


def publish_evaluation_subject(
    artifact_root: Path, record: M9EvaluationSubjectRecord
) -> tuple[M9EvaluationSubjectRecord, str]:
    """Idempotently publish an Evaluation-only record outside Candidate/Production."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    try:
        record = M9EvaluationSubjectRecord.model_validate_json(record.model_dump_json())
    except ValueError as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Evaluation subject record is invalid"
        ) from exc
    _validate_contained_paths(artifact_root, record)
    target = artifact_root / "registry" / "evaluation-subjects" / record.subject_id / "model.json"
    if target.exists():
        try:
            payload = target.read_bytes()
            existing = M9EvaluationSubjectRecord.model_validate_json(payload)
        except (OSError, ValueError) as exc:
            raise DeploymentError(
                DeploymentErrorCode.INVALID_INPUT, "Evaluation subject record is invalid"
            ) from exc
        existing_identity = existing.model_dump(mode="json", exclude={"created_at"})
        requested_identity = record.model_dump(mode="json", exclude={"created_at"})
        if existing_identity != requested_identity:
            raise DeploymentError(
                DeploymentErrorCode.CONFLICT, "Evaluation subject already exists with drift"
            )
        return existing, hashlib.sha256(payload).hexdigest()
    _atomic_json(target, record.to_dict())
    payload = target.read_bytes()
    return record, hashlib.sha256(payload).hexdigest()


def publish_m10_stage_evaluation_subject(
    artifact_root: Path, record: M10StageEvaluationSubjectRecord
) -> tuple[M10StageEvaluationSubjectRecord, str]:
    """Idempotently publish one M10 stage outside Candidate/Production Registry."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    try:
        record = M10StageEvaluationSubjectRecord.model_validate_json(record.model_dump_json())
    except ValueError as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 evaluation subject record is invalid"
        ) from exc
    _validate_m10_contained_paths(artifact_root, record)
    target = artifact_root / "registry" / "evaluation-subjects" / record.subject_id / "model.json"
    if target.exists():
        try:
            payload = target.read_bytes()
            existing = M10StageEvaluationSubjectRecord.model_validate_json(payload)
        except (OSError, ValueError) as exc:
            raise DeploymentError(
                DeploymentErrorCode.INVALID_INPUT, "M10 evaluation subject record is invalid"
            ) from exc
        existing_identity = existing.model_dump(mode="json", exclude={"created_at"})
        requested_identity = record.model_dump(mode="json", exclude={"created_at"})
        if existing_identity != requested_identity:
            raise DeploymentError(
                DeploymentErrorCode.CONFLICT, "M10 evaluation subject already exists with drift"
            )
        return existing, hashlib.sha256(payload).hexdigest()
    _atomic_json(target, record.to_dict())
    payload = target.read_bytes()
    return record, hashlib.sha256(payload).hexdigest()


def publish_m10_lora_stage_evaluation_subject(
    artifact_root: Path, record: M10LoRAStageEvaluationSubjectRecord
) -> tuple[M10LoRAStageEvaluationSubjectRecord, str]:
    """Idempotently publish one Agent LoRA stage outside Candidate/Production."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    try:
        record = M10LoRAStageEvaluationSubjectRecord.model_validate_json(record.model_dump_json())
    except ValueError as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 Agent LoRA subject record is invalid"
        ) from exc
    _validate_m10_lora_contained_paths(artifact_root, record)
    target = artifact_root / "registry" / "evaluation-subjects" / record.subject_id / "model.json"
    if target.exists():
        try:
            payload = target.read_bytes()
            existing = M10LoRAStageEvaluationSubjectRecord.model_validate_json(payload)
        except (OSError, ValueError) as exc:
            raise DeploymentError(
                DeploymentErrorCode.INVALID_INPUT, "M10 Agent LoRA subject record is invalid"
            ) from exc
        existing_identity = existing.model_dump(mode="json", exclude={"created_at"})
        requested_identity = record.model_dump(mode="json", exclude={"created_at"})
        if existing_identity != requested_identity:
            raise DeploymentError(
                DeploymentErrorCode.CONFLICT, "M10 Agent LoRA subject already exists with drift"
            )
        return existing, hashlib.sha256(payload).hexdigest()
    _atomic_json(target, record.to_dict())
    payload = target.read_bytes()
    return record, hashlib.sha256(payload).hexdigest()


def resolve_evaluation_subject(
    artifact_root: Path, subject_id: str, *, now: datetime | None = None
) -> ResolvedEvaluationSubject:
    """Resolve one Evaluation subject and fail closed on any Artifact drift."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    if re.fullmatch(M9_SUBJECT_PATTERN, subject_id) is None:
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Evaluation subject ID is invalid")
    path = artifact_root / "registry" / "evaluation-subjects" / subject_id / "model.json"
    try:
        payload = path.read_bytes()
        record = M9EvaluationSubjectRecord.model_validate_json(payload)
    except FileNotFoundError as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND, "Evaluation subject is missing"
        ) from exc
    except (OSError, ValueError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Evaluation subject is invalid"
        ) from exc
    if record.subject_id != subject_id:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "Evaluation subject identity differs"
        )
    _validate_contained_paths(artifact_root, record)
    record_sha256 = hashlib.sha256(payload).hexdigest()
    actual_model = _artifact_set_sha256(record.model_dir, record.model_files)
    actual_tokenizer = _artifact_set_sha256(record.tokenizer_dir, record.tokenizer_files)
    actual_adapter = (
        _artifact_set_sha256(record.adapter_dir, record.adapter_files)
        if record.adapter_dir is not None
        else None
    )
    if (
        actual_model != record.base_model_artifact_sha256
        or actual_tokenizer != record.tokenizer_artifact_sha256
        or actual_adapter != record.adapter_artifact_sha256
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "Evaluation subject Artifact hash differs"
        )
    _validate_model_configuration(record.model_dir, record.tokenizer_dir)
    return ResolvedEvaluationSubject(
        requested_ref=subject_id,
        model_version=subject_id,
        evaluation_subject_sha256=record_sha256,
        model=record.model,
        model_dir=record.model_dir,
        model_artifact_sha256=record.effective_artifact_sha256,
        tokenizer_dir=record.tokenizer_dir,
        tokenizer_artifact_sha256=record.tokenizer_artifact_sha256,
        adapter_dir=record.adapter_dir,
        adapter_artifact_sha256=record.adapter_artifact_sha256,
        verified_at=now or datetime.now(UTC),
    )


def resolve_m10_stage_evaluation_subject(
    artifact_root: Path, subject_id: str, *, now: datetime | None = None
) -> ResolvedEvaluationSubject:
    """Resolve one M10 stage and fail closed on metadata or Artifact drift."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    if re.fullmatch(M10_STAGE_SUBJECT_PATTERN, subject_id) is None:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 evaluation subject ID is invalid"
        )
    path = artifact_root / "registry" / "evaluation-subjects" / subject_id / "model.json"
    try:
        payload = path.read_bytes()
        record = M10StageEvaluationSubjectRecord.model_validate_json(payload)
    except FileNotFoundError as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND, "M10 evaluation subject is missing"
        ) from exc
    except (OSError, ValueError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 evaluation subject is invalid"
        ) from exc
    if record.subject_id != subject_id:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "M10 evaluation subject identity differs"
        )
    _validate_m10_contained_paths(artifact_root, record)
    record_sha256 = hashlib.sha256(payload).hexdigest()
    actual_model = _artifact_set_sha256(record.model_dir, record.model_files)
    actual_tokenizer = _artifact_set_sha256(record.tokenizer_dir, record.tokenizer_files)
    if (
        actual_model != record.model_artifact_sha256
        or actual_tokenizer != record.tokenizer_artifact_sha256
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "M10 evaluation subject Artifact hash differs"
        )
    _validate_model_configuration(record.model_dir, record.tokenizer_dir)
    return ResolvedEvaluationSubject(
        requested_ref=subject_id,
        model_version=subject_id,
        evaluation_subject_sha256=record_sha256,
        model=record.model,
        model_dir=record.model_dir,
        model_artifact_sha256=record.model_artifact_sha256,
        tokenizer_dir=record.tokenizer_dir,
        tokenizer_artifact_sha256=record.tokenizer_artifact_sha256,
        verified_at=now or datetime.now(UTC),
    )


def resolve_m10_lora_stage_evaluation_subject(
    artifact_root: Path, subject_id: str, *, now: datetime | None = None
) -> ResolvedEvaluationSubject:
    """Resolve one M10 Agent LoRA stage and fail closed on Artifact drift."""

    if not artifact_root.is_absolute() or artifact_root.is_symlink():
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "Artifact root must be absolute")
    if re.fullmatch(M10_LORA_STAGE_SUBJECT_PATTERN, subject_id) is None:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 Agent LoRA subject ID is invalid"
        )
    path = artifact_root / "registry" / "evaluation-subjects" / subject_id / "model.json"
    try:
        payload = path.read_bytes()
        record = M10LoRAStageEvaluationSubjectRecord.model_validate_json(payload)
    except FileNotFoundError as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND, "M10 Agent LoRA subject is missing"
        ) from exc
    except (OSError, ValueError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M10 Agent LoRA subject is invalid"
        ) from exc
    if record.subject_id != subject_id:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "M10 Agent LoRA subject identity differs"
        )
    _validate_m10_lora_contained_paths(artifact_root, record)
    record_sha256 = hashlib.sha256(payload).hexdigest()
    actual_model = _artifact_set_sha256(record.model_dir, record.model_files)
    actual_tokenizer = _artifact_set_sha256(record.tokenizer_dir, record.tokenizer_files)
    actual_adapter = _artifact_set_sha256(record.adapter_dir, record.adapter_files)
    if (
        actual_model != record.base_model_artifact_sha256
        or actual_tokenizer != record.tokenizer_artifact_sha256
        or actual_adapter != record.adapter_artifact_sha256
        or effective_artifact_sha256(actual_model, actual_adapter)
        != record.effective_artifact_sha256
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "M10 Agent LoRA Artifact hash differs"
        )
    _validate_model_configuration(record.model_dir, record.tokenizer_dir)
    return ResolvedEvaluationSubject(
        requested_ref=subject_id,
        model_version=subject_id,
        evaluation_subject_sha256=record_sha256,
        model=record.model,
        model_dir=record.model_dir,
        model_artifact_sha256=record.effective_artifact_sha256,
        tokenizer_dir=record.tokenizer_dir,
        tokenizer_artifact_sha256=record.tokenizer_artifact_sha256,
        adapter_dir=record.adapter_dir,
        adapter_artifact_sha256=record.adapter_artifact_sha256,
        verified_at=now or datetime.now(UTC),
    )


def resolve_serving_model(
    artifact_root: Path, model_ref: str, *, now: datetime | None = None
) -> ServingModel:
    """Resolve a deployable M6/M7 model or an explicit evaluation-only subject."""

    if model_ref.startswith("qwen3-8b-m9-"):
        return resolve_evaluation_subject(artifact_root, model_ref, now=now)
    if model_ref.startswith("qwen3-0-6b-m10-full-sft-"):
        return resolve_m10_stage_evaluation_subject(artifact_root, model_ref, now=now)
    if model_ref.startswith("qwen3-8b-m10-agent-lora-"):
        return resolve_m10_lora_stage_evaluation_subject(artifact_root, model_ref, now=now)
    from tinyllm.deployment.registry import resolve_model

    return resolve_model(artifact_root, model_ref, now=now)
