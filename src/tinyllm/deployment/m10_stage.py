"""Validated registration of M10 1M/5M Full-SFT evaluation stages."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from tinyllm.deployment.evaluation_subject import (
    M10StageEvaluationSubjectRecord,
    evaluation_artifact_sha256,
    m10_stage_evaluation_subject_id,
    publish_m10_stage_evaluation_subject,
)
from tinyllm.deployment.registry import DeploymentError, resolve_model
from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_sft import (
    M10CheckpointStore,
    M10FullSFTError,
    load_m10_full_sft_config,
)
from tinyllm.training.m10_sft_schema import M10FullSFTRunResult, M10StageExport

MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MODEL_PARAMETERS = 596_049_920
M10_EVALUATION_STAGE_TOKENS = (1_000_000, 5_000_000)


class M10StageRegistrationError(RuntimeError):
    """Raised when a completed evaluation stage has incomplete or drifting lineage."""


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
        raise M10StageRegistrationError("M10 stage commit marker is unavailable") from exc
    if not isinstance(value, dict):
        raise M10StageRegistrationError("M10 stage commit marker is invalid")
    return value


def build_m10_stage_evaluation_subject(
    *,
    artifact_root: Path,
    source_run: Path,
    stage_tokens: int = 5_000_000,
) -> M10StageEvaluationSubjectRecord:
    """Verify one durable 1M/5M boundary and construct its evaluation-only record."""

    if stage_tokens not in M10_EVALUATION_STAGE_TOKENS:
        raise M10StageRegistrationError("M10 evaluation stage must be 1M or 5M Tokens")

    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not source_run.is_absolute()
        or source_run.is_symlink()
    ):
        raise M10StageRegistrationError("M10 stage paths must be absolute non-symlink paths")
    try:
        root = artifact_root.resolve(strict=True)
        run = source_run.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise M10StageRegistrationError("M10 stage Artifact paths are unavailable") from exc
    if not run.is_relative_to(root):
        raise M10StageRegistrationError("M10 stage Run escapes the Artifact Store")

    checkpoint_id = f"checkpoint-tokens-{stage_tokens:010d}"
    if stage_tokens == 1_000_000:
        result_path = run / "attempts" / "fresh-stage_completed-tokens-0001000000.json"
        expected_mode = "fresh"
        expected_resumed_from: int | None = None
    else:
        result_path = run / "result.json"
        expected_mode = "exact_resume"
        expected_resumed_from = 1_000_000
    config_path = run / "config.original.yaml"
    environment_path = run / "environment.json"
    checkpoint_dir = run / "checkpoints" / checkpoint_id
    checkpoint_manifest_path = checkpoint_dir / "manifest.json"
    export_dir = run / "exports" / checkpoint_id
    export_manifest_path = export_dir / "stage_export.json"
    model_dir = export_dir / "model"
    try:
        result = M10FullSFTRunResult.model_validate_json(result_path.read_bytes())
        config = load_m10_full_sft_config(config_path)
        checkpoint = M10CheckpointStore(run / "checkpoints").validate(checkpoint_id)
        export_bytes = export_manifest_path.read_bytes()
        export = M10StageExport.model_validate_json(export_bytes)
        production = resolve_model(root, config.model.parent_model_ref)
    except (OSError, ValidationError, ValueError, DeploymentError, M10FullSFTError) as exc:
        raise M10StageRegistrationError("M10 evaluation stage metadata is invalid") from exc

    try:
        model_sha256 = evaluation_artifact_sha256(model_dir, MODEL_FILES)
        tokenizer_sha256 = evaluation_artifact_sha256(production.tokenizer_dir, TOKENIZER_FILES)
    except DeploymentError as exc:
        raise M10StageRegistrationError("M10 evaluation stage Artifact set is invalid") from exc
    config_sha256 = canonical_config_hash(config)
    export_marker = _load_marker(export_dir / "COMMITTED")
    identities = (
        (result.status, "stage_completed"),
        (result.mode, expected_mode),
        (result.run_id, run.name),
        (result.config_sha256, config_sha256),
        (result.supervised_tokens, stage_tokens),
        (result.resumed_from_tokens, expected_resumed_from),
        (result.latest_checkpoint, checkpoint.checkpoint_id),
        (result.stage_export, export),
        (result.stage_export.export_sha256, model_sha256),
        (checkpoint.run_id, result.run_id),
        (checkpoint.config_sha256, result.config_sha256),
        (checkpoint.dataset_version, result.dataset_version),
        (checkpoint.dataset_manifest_sha256, result.dataset_manifest_sha256),
        (checkpoint.git_commit, result.git_commit),
        (checkpoint.supervised_tokens, result.supervised_tokens),
        (checkpoint.pinned, True),
        (checkpoint.pin_reason, "stage"),
        (production.model_version, result.parent_production_version),
        (production.production_record_sha256, result.parent_production_record_sha256),
        (production.model_artifact_sha256, result.parent_model_artifact_sha256),
        (
            export_marker,
            {"manifest_sha256": hashlib.sha256(export_bytes).hexdigest()},
        ),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10StageRegistrationError(
            "M10 evaluation stage lineage is incomplete or inconsistent"
        )

    model = M6ModelIdentity(
        role="candidate",
        repository=config.model.repository,
        base_revision=config.model.revision,
        attention_architecture=config.model.attention_architecture,
        adaptation="full_sft",
        model_artifact_sha256=model_sha256,
        model_parameters=MODEL_PARAMETERS,
        training_run_id=result.run_id,
        training_checkpoint_id=result.latest_checkpoint,
        training_tokens=result.supervised_tokens,
        training_config_sha256=result.config_sha256,
        dataset_version=result.dataset_version,
        dataset_manifest_sha256=result.dataset_manifest_sha256,
    )
    result_sha256 = _sha256_file(result_path)
    manifest_sha256 = _sha256_file(checkpoint_manifest_path)
    environment_sha256 = _sha256_file(environment_path)
    subject_id = m10_stage_evaluation_subject_id(
        model=model,
        tokenizer_artifact_sha256=tokenizer_sha256,
        source_result_sha256=result_sha256,
        checkpoint_manifest_sha256=manifest_sha256,
        environment_sha256=environment_sha256,
    )
    return M10StageEvaluationSubjectRecord(
        subject_id=subject_id,
        kind="m10_full_sft_1m" if stage_tokens == 1_000_000 else "m10_full_sft_5m",
        created_at=datetime.now(UTC),
        model=model,
        model_dir=model_dir,
        model_files=MODEL_FILES,
        model_artifact_sha256=model_sha256,
        tokenizer_dir=production.tokenizer_dir,
        tokenizer_files=TOKENIZER_FILES,
        tokenizer_artifact_sha256=tokenizer_sha256,
        source_run_dir=run,
        source_result_sha256=result_sha256,
        checkpoint_manifest_sha256=manifest_sha256,
        checkpoint_payload_sha256=checkpoint.file.sha256,
        environment_sha256=environment_sha256,
    )


def register_m10_stage_evaluation_subject(
    *,
    artifact_root: Path,
    source_run: Path,
    stage_tokens: int = 5_000_000,
) -> tuple[M10StageEvaluationSubjectRecord, str]:
    """Build and atomically publish one verified M10 1M/5M stage record."""

    record = build_m10_stage_evaluation_subject(
        artifact_root=artifact_root,
        source_run=source_run,
        stage_tokens=stage_tokens,
    )
    try:
        return publish_m10_stage_evaluation_subject(artifact_root, record)
    except DeploymentError as exc:
        raise M10StageRegistrationError("M10 stage registration failed") from exc
