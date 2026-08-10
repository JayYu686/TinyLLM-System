"""Strict M5 Candidate import for the frozen M6 release evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from pydantic import ValidationError

from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_base import sha256_file
from tinyllm.evaluation.m6_schema import M6CandidateImportResult, M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.m5_formal_schema import (
    M5FormalCheckpointManifest,
    M5FormalEvaluationSnapshot,
    M5FormalRunResult,
)

FROZEN_RUN_ID = "20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15"
FROZEN_CHECKPOINT_ID = "checkpoint-tokens-0010000532"
FROZEN_EXPORT_SHA256 = "b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c"
FROZEN_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FROZEN_MODEL_PARAMETERS = 596_049_920


class M6CandidateImportError(RuntimeError):
    """Raised when the frozen M5 Candidate lineage is incomplete or inconsistent."""


def model_export_sha256(root: Path) -> str:
    """Reproduce the stable M5 export digest over immediate regular files."""

    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise M6CandidateImportError("M6 Candidate model export is missing or unsafe")
    files = sorted(root.iterdir(), key=lambda item: item.name)
    if not files:
        raise M6CandidateImportError("M6 Candidate model export is empty")
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise M6CandidateImportError("M6 Candidate model export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def import_m5_candidate_evidence(
    *,
    release_config_path: Path,
    source_run: Path,
    model_dir: Path,
    output_path: Path | None = None,
) -> M6CandidateImportResult:
    """Validate the frozen M5 10M Full-SFT snapshot and emit its M6 identity."""

    if (
        not source_run.is_absolute()
        or not source_run.is_dir()
        or source_run.is_symlink()
        or source_run.name != FROZEN_RUN_ID
    ):
        raise M6CandidateImportError("M6 Candidate source Run differs from the frozen Run")
    snapshot_root = source_run / "evaluations" / FROZEN_CHECKPOINT_ID
    expected_model_dir = snapshot_root / "model"
    if model_dir.resolve() != expected_model_dir.resolve():
        raise M6CandidateImportError("M6 Candidate model path differs from the frozen snapshot")
    result_path = source_run / "result.json"
    snapshot_path = snapshot_root / "snapshot.json"
    manifest_path = source_run / "checkpoints" / FROZEN_CHECKPOINT_ID / "manifest.json"
    try:
        result = M5FormalRunResult.model_validate_json(result_path.read_bytes())
        snapshot = M5FormalEvaluationSnapshot.model_validate_json(snapshot_path.read_bytes())
        manifest = M5FormalCheckpointManifest.model_validate_json(manifest_path.read_bytes())
        config = load_m5_sft_config(source_run / "config.original.yaml")
    except (OSError, ValueError, ValidationError) as exc:
        raise M6CandidateImportError("M6 Candidate source metadata is invalid") from exc
    release = load_m6_release_config(release_config_path)
    checkpoint_index = result.evaluation_checkpoints.index(FROZEN_CHECKPOINT_ID)
    export_sha256 = model_export_sha256(model_dir)
    if (
        result.status != "succeeded"
        or result.run_id != source_run.name
        or result.git_dirty
        or canonical_config_hash(config) != result.config_sha256
        or snapshot.run_id != result.run_id
        or snapshot.checkpoint_id != FROZEN_CHECKPOINT_ID
        or snapshot.target_tokens != 10_000_000
        or snapshot.supervised_tokens != 10_000_532
        or snapshot.export_sha256 != FROZEN_EXPORT_SHA256
        or result.evaluation_export_sha256s[checkpoint_index] != snapshot.export_sha256
        or export_sha256 != snapshot.export_sha256
        or manifest.run_id != result.run_id
        or manifest.checkpoint_id != snapshot.checkpoint_id
        or manifest.supervised_tokens != snapshot.supervised_tokens
        or manifest.config_sha256 != result.config_sha256
        or manifest.dataset_version != result.dataset_version
        or manifest.dataset_manifest_sha256 != result.dataset_manifest_sha256
        or manifest.git_commit != result.git_commit
        or manifest.environment_sha256 != result.environment_sha256
        or manifest.hardware_sha256 != result.hardware_sha256
        or not manifest.pinned
        or manifest.pin_reason != "evaluation"
        or sha256_file(manifest_path) != snapshot.checkpoint_manifest_sha256
        or sha256_file(source_run / "environment.json") != result.environment_sha256
        or sha256_file(source_run / "hardware.json") != result.hardware_sha256
    ):
        raise M6CandidateImportError("M6 Candidate lineage is incomplete or inconsistent")
    imported = M6CandidateImportResult(
        status="succeeded",
        protocol_version=release.protocol_version,
        config_sha256=canonical_config_hash(release),
        source_run_id=result.run_id,
        source_result_sha256=sha256_file(result_path),
        source_git_commit=result.git_commit,
        source_environment_sha256=result.environment_sha256,
        source_hardware_sha256=result.hardware_sha256,
        checkpoint_manifest_sha256=snapshot.checkpoint_manifest_sha256,
        snapshot_sha256=sha256_file(snapshot_path),
        model=M6ModelIdentity(
            role="candidate",
            repository="Qwen/Qwen3-0.6B",
            base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            attention_architecture="gqa",
            adaptation="full_sft",
            model_artifact_sha256=export_sha256,
            model_parameters=FROZEN_MODEL_PARAMETERS,
            training_run_id=result.run_id,
            training_checkpoint_id=snapshot.checkpoint_id,
            training_tokens=snapshot.supervised_tokens,
            training_config_sha256=result.config_sha256,
            dataset_version=result.dataset_version,
            dataset_manifest_sha256=result.dataset_manifest_sha256,
        ),
    )
    if output_path is not None:
        if not output_path.is_absolute():
            raise M6CandidateImportError("M6 Candidate import output path must be absolute")
        _atomic_json(output_path, imported.to_dict())
    return imported


def load_m6_candidate_import(path: Path) -> M6CandidateImportResult:
    """Load one persisted M6 Candidate import."""

    try:
        return M6CandidateImportResult.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise M6CandidateImportError("M6 Candidate import result is invalid") from exc
