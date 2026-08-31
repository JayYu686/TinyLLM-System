"""Fail-closed aggregation recovery for fully persisted Agent evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tinyllm.agent_eval.schema import (
    AgentEvalItemResult,
    AgentEvalRunConfig,
    AgentEvalSummary,
    canonical_json_sha256,
)
from tinyllm.agent_eval.scoring import aggregate_results
from tinyllm.agent_eval.suite import load_suite
from tinyllm.deployment import resolve_evaluation_subject
from tinyllm.lineage.git import read_git_identity


class AgentEvalFinalizeError(ValueError):
    """Raised when persisted task evidence cannot be aggregated exactly."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AgentEvalFinalizeError(f"recovered output already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value: object = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentEvalFinalizeError(f"persisted {path.name} is invalid") from exc
    if not isinstance(value, dict):
        raise AgentEvalFinalizeError(f"persisted {path.name} is not an object")
    return payload, value


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise AgentEvalFinalizeError(f"persisted metadata lacks {key}")
    return item


def finalize_agent_evaluation(
    *,
    suite_directory: Path,
    output_directory: Path,
    artifact_root: Path,
    project_root: Path,
    finalized_at: datetime | None = None,
) -> AgentEvalSummary:
    """Aggregate an already complete item set after a terminal summary-only failure."""

    if (
        not output_directory.is_absolute()
        or output_directory.is_symlink()
        or not artifact_root.is_absolute()
    ):
        raise AgentEvalFinalizeError("recovery roots must be absolute non-symlink paths")
    for name in ("items.jsonl", "summary.json", "aggregation-recovery.json"):
        if (output_directory / name).exists() or (output_directory / name).is_symlink():
            raise AgentEvalFinalizeError("evaluation already has terminal aggregate evidence")

    manifest, tasks = load_suite(suite_directory)
    _, persisted_manifest = _object(output_directory / "suite.manifest.json")
    if persisted_manifest != manifest.to_dict():
        raise AgentEvalFinalizeError("persisted Suite Manifest differs from the sealed suite")

    config_payload, _ = _object(output_directory / "config.resolved.json")
    try:
        config = AgentEvalRunConfig.model_validate_json(config_payload)
    except ValidationError as exc:
        raise AgentEvalFinalizeError("persisted evaluation config is invalid") from exc
    environment_payload, environment = _object(output_directory / "environment.json")
    hardware_payload, hardware = _object(output_directory / "hardware.json")
    _, metadata = _object(output_directory / "evaluation.metadata.json")
    if (
        _string(metadata, "suite_version") != manifest.suite_version
        or _string(metadata, "suite_content_sha256") != manifest.content_sha256
        or _string(metadata, "config_sha256") != canonical_json_sha256(config.to_dict())
        or _string(metadata, "environment_sha256") != _sha256(environment_payload)
        or _string(metadata, "hardware_sha256") != _sha256(hardware_payload)
        or _string(metadata, "model_id") != config.model
    ):
        raise AgentEvalFinalizeError("persisted evaluation lineage hashes differ")

    resolved = resolve_evaluation_subject(artifact_root, config.model)
    if (
        resolved.model_version != config.model
        or resolved.model_artifact_sha256 != _string(metadata, "model_artifact_sha256")
        or resolved.evaluation_subject_sha256 != _string(metadata, "evaluation_subject_sha256")
    ):
        raise AgentEvalFinalizeError("resolved model differs from persisted evaluation lineage")

    items_root = output_directory / "items"
    expected_ids = {task.task_id for task in tasks}
    if not items_root.is_dir() or items_root.is_symlink():
        raise AgentEvalFinalizeError("persisted item directory is missing or unsafe")
    paths = tuple(sorted(items_root.glob("*.json")))
    if {path.stem for path in paths} != expected_ids or any(path.is_symlink() for path in paths):
        raise AgentEvalFinalizeError("persisted item identities differ from the sealed suite")
    task_by_id = {task.task_id: task for task in tasks}
    results_by_id: dict[str, AgentEvalItemResult] = {}
    try:
        for path in paths:
            result = AgentEvalItemResult.model_validate_json(path.read_bytes())
            task = task_by_id[result.task_id]
            if (
                result.cluster_id != task.cluster_id
                or result.category != task.category
                or result.language != task.language
                or result.scoring_protocol != config.scoring_protocol
            ):
                raise AgentEvalFinalizeError("persisted item content differs from its task")
            results_by_id[result.task_id] = result
    except (OSError, ValidationError, KeyError) as exc:
        raise AgentEvalFinalizeError("persisted item result is invalid") from exc
    results = tuple(results_by_id[task.task_id] for task in tasks)
    result_bytes = b"".join(
        json.dumps(
            item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
        for item in results
    )
    item_results_sha256 = _sha256(result_bytes)

    gateway = environment.get("gateway")
    if not isinstance(gateway, dict):
        raise AgentEvalFinalizeError("persisted Gateway environment is invalid")
    timestamp = datetime.now(UTC) if finalized_at is None else finalized_at
    if timestamp.tzinfo is None:
        raise AgentEvalFinalizeError("recovery timestamp must be timezone-aware")
    finalizer_commit, finalizer_dirty = read_git_identity(project_root)
    if finalizer_dirty:
        raise AgentEvalFinalizeError("aggregation recovery requires a clean Git worktree")
    source_commit = _string(metadata, "git_commit")
    recovery_identity = canonical_json_sha256(
        {
            "kind": "summary_schema_recovery",
            "source_git_commit": source_commit,
            "finalizer_git_commit": finalizer_commit,
            "suite_content_sha256": manifest.content_sha256,
            "item_results_sha256": item_results_sha256,
            "finalized_at": timestamp.isoformat(),
        }
    )
    summary = AgentEvalSummary(
        scoring_protocol=config.scoring_protocol,
        evaluation_id=f"m9-agent-eval-{recovery_identity[:8]}",
        evaluated_at=timestamp,
        suite_version=manifest.suite_version,
        suite_content_sha256=manifest.content_sha256,
        model_id=resolved.model_version,
        model_revision=resolved.model.base_revision,
        model_artifact_sha256=resolved.model_artifact_sha256,
        parent_model_id=f"{resolved.model.repository}@{resolved.model.base_revision}",
        deployment_record_sha256=None,
        evaluation_subject_sha256=resolved.evaluation_subject_sha256,
        environment_sha256=_sha256(environment_payload),
        hardware_sha256=_sha256(hardware_payload),
        physical_gpu_index=config.physical_gpu_index,
        gpu_name=_string(hardware, "name"),
        driver_version=_string(hardware, "driver_version"),
        gateway_version=_string(gateway, "version"),
        agent_runtime_version=_string(environment, "tinyllm"),
        git_commit=source_commit,
        git_dirty=False,
        metrics=aggregate_results(results),
        item_results_sha256=item_results_sha256,
        completed=len(results) == manifest.item_count,
    )
    summary_payload = _json_bytes(summary.to_dict())
    recovery = {
        "schema_version": "1.0",
        "kind": "agent_evaluation_aggregation_recovery",
        "reason": "summary_schema_rejected_release_v2_after_all_items_persisted",
        "source_git_commit": source_commit,
        "finalizer_git_commit": finalizer_commit,
        "suite_version": manifest.suite_version,
        "suite_content_sha256": manifest.content_sha256,
        "persisted_item_count": len(results),
        "item_results_sha256": item_results_sha256,
        "summary_sha256": _sha256(summary_payload),
        "model_generation_repeated": False,
        "finalized_at": timestamp.isoformat(),
    }
    _atomic_new(output_directory / "items.jsonl", result_bytes)
    _atomic_new(output_directory / "aggregation-recovery.json", _json_bytes(recovery))
    _atomic_new(output_directory / "summary.json", summary_payload)
    return summary
