"""M7 model resolution and atomic Production Registry operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from tinyllm.deployment.schema import (
    CANDIDATE_VERSION_PATTERN,
    M7ProductionAlias,
    M7ProductionGate,
    M7ProductionRecord,
    ResolvedModel,
)
from tinyllm.evaluation.m6_schema import M6PromotionRecord

FORBIDDEN_MODEL_CONFIG_KEYS = frozenset(
    {
        "_attn_implementation_internal",
        "auto_map",
        "custom_pipelines",
        "sbert_ce_default_activation_function",
        "sentence_transformers",
        "trust_remote_code",
    }
)


class DeploymentErrorCode(StrEnum):
    """Stable deployment failure classes used by CLI and Gateway."""

    INVALID_INPUT = "DEPLOYMENT_INVALID_INPUT"
    NOT_FOUND = "DEPLOYMENT_NOT_FOUND"
    HASH_MISMATCH = "DEPLOYMENT_HASH_MISMATCH"
    UNSAFE_ARTIFACT = "DEPLOYMENT_UNSAFE_ARTIFACT"
    GATE_REJECTED = "DEPLOYMENT_GATE_REJECTED"
    CONFLICT = "DEPLOYMENT_CONFLICT"
    IO_ERROR = "DEPLOYMENT_IO_ERROR"


class DeploymentError(RuntimeError):
    """Raised when a model cannot be resolved or published safely."""

    def __init__(self, code: DeploymentErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_set_sha256(root: Path, names: tuple[str, ...] | None = None) -> str:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise DeploymentError(
            DeploymentErrorCode.UNSAFE_ARTIFACT,
            "artifact directory is missing, relative, or a symbolic link",
        )
    paths = sorted(root.iterdir(), key=lambda item: item.name)
    if names is not None:
        expected = set(names)
        paths = [path for path in paths if path.name in expected]
        if {path.name for path in paths} != expected:
            raise DeploymentError(
                DeploymentErrorCode.NOT_FOUND,
                "artifact directory is missing required files",
            )
    if not paths:
        raise DeploymentError(DeploymentErrorCode.NOT_FOUND, "artifact directory is empty")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise DeploymentError(
                DeploymentErrorCode.UNSAFE_ARTIFACT,
                "artifact set contains a non-regular file",
            )
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


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


def _load_json(path: Path, model: type[Any], label: str) -> tuple[Any, str]:
    try:
        payload = path.read_bytes()
        return model.model_validate_json(payload), hashlib.sha256(payload).hexdigest()
    except FileNotFoundError as exc:
        raise DeploymentError(DeploymentErrorCode.NOT_FOUND, f"{label} is missing") from exc
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, f"{label} is invalid") from exc


def _load_plain_json(path: Path, label: str) -> dict[str, Any]:
    try:
        decoded: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, f"{label} is invalid") from exc
    if not isinstance(decoded, dict):
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, f"{label} must be a JSON object")
    return cast(dict[str, Any], decoded)


def _reject_unsafe_config_keys(value: object, label: str) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            unsafe = FORBIDDEN_MODEL_CONFIG_KEYS.intersection(current)
            if unsafe:
                raise DeploymentError(
                    DeploymentErrorCode.UNSAFE_ARTIFACT,
                    f"{label} contains a forbidden dynamic configuration key",
                )
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _validate_model_configuration(model_dir: Path, tokenizer_dir: Path) -> None:
    model_config = _load_plain_json(model_dir / "config.json", "model config")
    _reject_unsafe_config_keys(model_config, "model config")
    if model_config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise DeploymentError(
            DeploymentErrorCode.UNSAFE_ARTIFACT,
            "model config does not declare the reviewed Qwen3ForCausalLM architecture",
        )
    if model_config.get("model_type") != "qwen3":
        raise DeploymentError(
            DeploymentErrorCode.UNSAFE_ARTIFACT,
            "model config does not declare the reviewed qwen3 model type",
        )
    tokenizer_config = _load_plain_json(tokenizer_dir / "tokenizer_config.json", "tokenizer config")
    _reject_unsafe_config_keys(tokenizer_config, "tokenizer config")


def _require_absolute_root(root: Path) -> None:
    if not root.is_absolute() or root.is_symlink():
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT,
            "Artifact Store root must be an absolute non-symlink path",
        )


def _load_candidate(artifact_root: Path, candidate_version: str) -> tuple[M6PromotionRecord, str]:
    if not re.fullmatch(CANDIDATE_VERSION_PATTERN, candidate_version):
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Candidate model version is invalid"
        )
    path = artifact_root / "registry" / "candidates" / candidate_version / "model.json"
    record, digest = _load_json(path, M6PromotionRecord, "M6 Candidate record")
    return record, digest


def _load_production_record(
    artifact_root: Path, production_version: str
) -> tuple[M7ProductionRecord, str]:
    path = artifact_root / "registry" / "production" / production_version / "model.json"
    record, digest = _load_json(path, M7ProductionRecord, "M7 Production record")
    return record, digest


def _load_alias(artifact_root: Path) -> M7ProductionAlias:
    path = artifact_root / "registry" / "aliases" / "production.json"
    alias, _ = _load_json(path, M7ProductionAlias, "Production Alias")
    return cast(M7ProductionAlias, alias)


def _resolve_run_directory(artifact_root: Path, run_id: str) -> Path:
    candidates = tuple(
        path
        for path in (artifact_root / "runs").glob(f"*/{run_id}")
        if path.is_dir() and not path.is_symlink()
    )
    if len(candidates) != 1:
        code = DeploymentErrorCode.NOT_FOUND if not candidates else DeploymentErrorCode.CONFLICT
        raise DeploymentError(code, "training Run directory did not resolve uniquely")
    root = artifact_root.resolve(strict=True)
    resolved = candidates[0].resolve(strict=True)
    if not resolved.is_relative_to(root / "runs"):
        raise DeploymentError(
            DeploymentErrorCode.UNSAFE_ARTIFACT, "training Run escapes the Artifact Store"
        )
    return resolved


def _resolve_candidate(
    artifact_root: Path,
    candidate_version: str,
    *,
    requested_ref: str,
    status: str,
    production_version: str | None = None,
    production_record_sha256: str | None = None,
    now: datetime | None = None,
) -> ResolvedModel:
    candidate, candidate_sha256 = _load_candidate(artifact_root, candidate_version)
    if candidate.model.training_run_id is None:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT,
            "M6 Candidate record is missing its training Run identity",
        )
    run_dir = _resolve_run_directory(artifact_root, candidate.model.training_run_id)
    model_dir = run_dir / "exports" / "model"
    actual_model_sha256 = _artifact_set_sha256(model_dir)
    if actual_model_sha256 != candidate.model.model_artifact_sha256:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "model export hash differs from the immutable M6 Candidate record",
        )
    repository_parts = candidate.model.repository.split("/")
    if len(repository_parts) != 2 or not all(
        re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in repository_parts
    ):
        raise DeploymentError(
            DeploymentErrorCode.UNSAFE_ARTIFACT, "model repository identity is unsafe"
        )
    tokenizer_dir = (
        artifact_root
        / "cache"
        / "models"
        / repository_parts[0]
        / repository_parts[1]
        / candidate.model.base_revision
    )
    tokenizer_sha256 = _artifact_set_sha256(
        tokenizer_dir, ("tokenizer.json", "tokenizer_config.json")
    )
    _validate_model_configuration(model_dir, tokenizer_dir)
    return ResolvedModel(
        requested_ref=requested_ref,
        status=cast(Literal["Candidate", "Production"], status),
        model_version=production_version or candidate_version,
        candidate_model_version=candidate_version,
        candidate_record_sha256=candidate_sha256,
        production_record_sha256=production_record_sha256,
        model=candidate.model,
        model_dir=model_dir,
        model_artifact_sha256=actual_model_sha256,
        tokenizer_dir=tokenizer_dir,
        tokenizer_artifact_sha256=tokenizer_sha256,
        verified_at=now or datetime.now(UTC),
    )


def resolve_model(
    artifact_root: Path,
    model_ref: str = "production",
    *,
    now: datetime | None = None,
) -> ResolvedModel:
    """Resolve Candidate, Production version, or Production Alias and verify hashes."""

    _require_absolute_root(artifact_root)
    if model_ref == "production":
        alias = _load_alias(artifact_root)
        production, production_sha256 = _load_production_record(
            artifact_root, alias.production_version
        )
        if production_sha256 != alias.production_record_sha256:
            raise DeploymentError(
                DeploymentErrorCode.HASH_MISMATCH,
                "Production Alias hash differs from its immutable record",
            )
        resolved = _resolve_candidate(
            artifact_root,
            production.source_candidate_version,
            requested_ref=model_ref,
            status="Production",
            production_version=production.production_version,
            production_record_sha256=production_sha256,
            now=now,
        )
        _validate_production_lineage(production, resolved)
        return resolved
    if model_ref.startswith("qwen3-") and "-m6-" in model_ref:
        return _resolve_candidate(
            artifact_root,
            model_ref,
            requested_ref=model_ref,
            status="Candidate",
            now=now,
        )
    if model_ref.startswith("qwen3-") and "-m7-" in model_ref:
        production, production_sha256 = _load_production_record(artifact_root, model_ref)
        resolved = _resolve_candidate(
            artifact_root,
            production.source_candidate_version,
            requested_ref=model_ref,
            status="Production",
            production_version=production.production_version,
            production_record_sha256=production_sha256,
            now=now,
        )
        _validate_production_lineage(production, resolved)
        return resolved
    raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, "model reference is invalid")


def _validate_production_lineage(production: M7ProductionRecord, resolved: ResolvedModel) -> None:
    identities = (
        (production.candidate_record_sha256, resolved.candidate_record_sha256),
        (production.model, resolved.model),
        (production.model_artifact_sha256, resolved.model_artifact_sha256),
        (production.tokenizer_artifact_sha256, resolved.tokenizer_artifact_sha256),
    )
    if any(expected != actual for expected, actual in identities):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "Production record differs from its current verified M6 Candidate evidence",
        )


def load_production_gate(path: Path) -> tuple[M7ProductionGate, str]:
    """Load one M7 Gate and return the exact persisted-file identity."""

    if not path.is_absolute():
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Production Gate path must be absolute"
        )
    gate, digest = _load_json(path, M7ProductionGate, "M7 Production Gate")
    return gate, digest


def promote_production(
    artifact_root: Path,
    gate_path: Path,
    *,
    now: datetime | None = None,
) -> M7ProductionRecord:
    """Publish a new immutable Production record and atomically update its Alias."""

    _require_absolute_root(artifact_root)
    gate, gate_sha256 = load_production_gate(gate_path)
    if not gate.production_eligible or gate.status != "accepted":
        raise DeploymentError(
            DeploymentErrorCode.GATE_REJECTED, "M7 Production Gate rejected this model"
        )
    resolved = resolve_model(artifact_root, gate.candidate_model_version, now=now)
    identities = (
        (gate.candidate_record_sha256, resolved.candidate_record_sha256),
        (gate.model_artifact_sha256, resolved.model_artifact_sha256),
        (gate.tokenizer_artifact_sha256, resolved.tokenizer_artifact_sha256),
    )
    if any(expected != actual for expected, actual in identities):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "M7 Gate evidence differs from the current verified Candidate",
        )
    size = "0-6b" if resolved.model.repository.endswith("0.6B") else "8b"
    production_version = f"qwen3-{size}-m7-{gate_sha256[:8]}"
    record = M7ProductionRecord(
        production_version=production_version,
        promoted_at=now or datetime.now(UTC),
        source_candidate_version=resolved.candidate_model_version,
        candidate_record_sha256=resolved.candidate_record_sha256,
        production_gate_id=gate.gate_id,
        production_gate_sha256=gate_sha256,
        model=resolved.model,
        model_artifact_sha256=resolved.model_artifact_sha256,
        tokenizer_artifact_sha256=resolved.tokenizer_artifact_sha256,
        serving_config_sha256=gate.serving_config_sha256,
        environment_sha256=gate.environment_sha256,
        benchmark_report_sha256=gate.benchmark_report_sha256,
        recovery_report_sha256=gate.recovery_report_sha256,
        rollback_report_sha256=gate.rollback_report_sha256,
        security_audit_sha256=gate.security_audit_sha256,
    )
    target = artifact_root / "registry" / "production" / production_version
    record_path = target / "model.json"
    if target.exists():
        existing, existing_sha256 = _load_production_record(artifact_root, production_version)
        if existing != record.model_copy(update={"promoted_at": existing.promoted_at}):
            raise DeploymentError(
                DeploymentErrorCode.CONFLICT,
                "Production version already exists with different evidence",
            )
        record = existing
        record_sha256 = existing_sha256
    else:
        temporary = target.with_name(f".{production_version}.tmp-{uuid.uuid4().hex}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.parent.chmod(0o700)
            temporary.mkdir(mode=0o700)
            _atomic_json(temporary / "model.json", record.to_dict())
            os.replace(temporary, target)
            target.chmod(0o700)
            record_sha256 = _sha256_file(record_path)
        except OSError as exc:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
            raise DeploymentError(
                DeploymentErrorCode.IO_ERROR,
                "cannot atomically publish Production record",
            ) from exc
    alias_path = artifact_root / "registry" / "aliases" / "production.json"
    previous: str | None = None
    if alias_path.exists():
        previous = _load_alias(artifact_root).production_version
        if previous == production_version:
            return record
    alias = M7ProductionAlias(
        production_version=production_version,
        production_record_sha256=record_sha256,
        previous_production_version=previous,
        updated_at=now or datetime.now(UTC),
    )
    try:
        _atomic_json(alias_path, alias.to_dict())
    except OSError as exc:
        raise DeploymentError(
            DeploymentErrorCode.IO_ERROR, "cannot atomically update Production Alias"
        ) from exc
    return record


def rollback_production(
    artifact_root: Path,
    target_version: str | None = None,
    *,
    now: datetime | None = None,
) -> M7ProductionAlias:
    """Atomically point the Production Alias at a previously accepted record."""

    _require_absolute_root(artifact_root)
    current = _load_alias(artifact_root)
    target_version = target_version or current.previous_production_version
    if target_version is None:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Production Alias has no previous target"
        )
    if target_version == current.production_version:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "rollback target is already active"
        )
    _, target_sha256 = _load_production_record(artifact_root, target_version)
    alias = M7ProductionAlias(
        production_version=target_version,
        production_record_sha256=target_sha256,
        previous_production_version=current.production_version,
        updated_at=now or datetime.now(UTC),
    )
    try:
        _atomic_json(artifact_root / "registry" / "aliases" / "production.json", alias.to_dict())
    except OSError as exc:
        raise DeploymentError(
            DeploymentErrorCode.IO_ERROR, "cannot atomically roll back Production Alias"
        ) from exc
    return alias


def show_deployment(artifact_root: Path, model_ref: str = "production") -> dict[str, object]:
    """Return a path-free Registry view for CLI inspection."""

    resolved = resolve_model(artifact_root, model_ref)
    return {
        "schema_version": "1.0",
        "status": resolved.status,
        "model_version": resolved.model_version,
        "candidate_model_version": resolved.candidate_model_version,
        "candidate_record_sha256": resolved.candidate_record_sha256,
        "production_record_sha256": resolved.production_record_sha256,
        "repository": resolved.model.repository,
        "base_revision": resolved.model.base_revision,
        "training_run_id": resolved.model.training_run_id,
        "model_artifact_sha256": resolved.model_artifact_sha256,
        "tokenizer_artifact_sha256": resolved.tokenizer_artifact_sha256,
        "verified_at": resolved.verified_at.isoformat(),
    }
