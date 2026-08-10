"""Fail-closed assembly of complete M6 Base and Candidate evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_base import load_m6_base_import, sha256_file, sha256_tree
from tinyllm.evaluation.m6_candidate import load_m6_candidate_import
from tinyllm.evaluation.m6_schema import (
    M6DomainModeResult,
    M6DomainPassSummary,
    M6EvaluationResult,
    M6GeneralPassSummary,
    M6GeneralResult,
    M6ModelIdentity,
    M6ReleaseConfig,
)
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash


class M6AssemblyError(RuntimeError):
    """Raised when complete M6 evidence cannot be assembled safely."""


def _component_sha256(components: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"algorithm": "tinyllm-component-sha256-v1", "components": components},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_evaluation(path: Path, result: M6EvaluationResult) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise M6AssemblyError("M6 evaluation output must be an absolute regular path")
    payload = (
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise M6AssemblyError("existing M6 evaluation differs from assembled evidence")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise M6AssemblyError("cannot atomically persist M6 evaluation") from exc


def _regular_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise M6AssemblyError(f"{label} must be an existing absolute directory")


def _load_domain_component(
    directory: Path,
    judgments_path: Path,
    *,
    expected_mode: str,
    expected_model: M6ModelIdentity,
    expected_config_sha256: str,
) -> tuple[M6DomainPassSummary, M6DomainModeResult]:
    _regular_directory(directory, f"M6 {expected_mode} pass")
    if not judgments_path.is_absolute():
        raise M6AssemblyError("M6 approved judgments path must be absolute")
    try:
        summary = M6DomainPassSummary.model_validate_json(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        result = M6DomainModeResult.model_validate_json(
            (directory / "mode_result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise M6AssemblyError("M6 finalized domain component is invalid") from exc
    raw_path = directory / "results.jsonl"
    environment_path = directory / "environment.json"
    hardware_path = directory / "hardware.json"
    if (
        summary.status != "succeeded"
        or summary.mode != expected_mode
        or result.mode != expected_mode
        or summary.model != expected_model
        or summary.config_sha256 != expected_config_sha256
        or summary.git_dirty
        or summary.human_review_pending != 0
        or summary.human_reviewed != 40
        or summary.human_review_sha256 is None
        or sha256_file(judgments_path) != summary.human_review_sha256
        or sha256_file(raw_path) != summary.raw_results_sha256
        or sha256_file(environment_path) != summary.environment_sha256
        or sha256_file(hardware_path) != summary.hardware_sha256
    ):
        raise M6AssemblyError("M6 finalized domain lineage is incomplete or inconsistent")
    return summary, result


def _load_general_component(
    directory: Path,
    *,
    expected_model: M6ModelIdentity,
    expected_config_sha256: str,
) -> M6GeneralPassSummary:
    _regular_directory(directory, "M6 Candidate general pass")
    try:
        summary = M6GeneralPassSummary.model_validate_json(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise M6AssemblyError("M6 Candidate general summary is invalid") from exc
    if (
        summary.model != expected_model
        or summary.config_sha256 != expected_config_sha256
        or summary.git_dirty
        or sha256_file(directory / "environment.json") != summary.environment_sha256
        or sha256_file(directory / "hardware.json") != summary.hardware_sha256
        or sha256_tree(directory / "raw") != summary.raw_results_sha256
    ):
        raise M6AssemblyError("M6 Candidate general lineage is incomplete or inconsistent")
    return summary


def _build_evaluation(
    *,
    release: M6ReleaseConfig,
    git_commit: str,
    model: M6ModelIdentity,
    thinking: M6DomainModeResult,
    nonthinking: M6DomainModeResult,
    general: M6GeneralResult,
    thinking_review_sha256: str,
    nonthinking_review_sha256: str,
    software_environment_sha256: str,
    hardware_sha256: str,
    raw_domain_results_sha256: str,
    raw_general_results_sha256: str,
) -> M6EvaluationResult:
    role = model.role
    evaluation_id = (
        f"m6-evaluation-{role}-{model.model_artifact_sha256[:8]}-{raw_domain_results_sha256[:8]}"
    )
    return M6EvaluationResult(
        status="succeeded",
        evaluation_id=evaluation_id,
        protocol_version=release.protocol_version,
        suite_version=release.suite_version,
        config_sha256=canonical_config_hash(release),
        git_commit=git_commit,
        git_dirty=False,
        model=model,
        tokenizer_revision=model.base_revision,
        thinking_template_sha256=release.domain_execution.thinking.template_sha256,
        nonthinking_template_sha256=release.domain_execution.nonthinking.template_sha256,
        general_chat_template_sha256=(release.general_execution.tokenizer_chat_template_sha256),
        software_environment_sha256=software_environment_sha256,
        hardware_sha256=hardware_sha256,
        domain_modes=(thinking, nonthinking),
        general=general,
        human_review_complete=True,
        thinking_human_review_sha256=thinking_review_sha256,
        nonthinking_human_review_sha256=nonthinking_review_sha256,
        lineage_complete=True,
        raw_domain_results_sha256=raw_domain_results_sha256,
        raw_general_results_sha256=raw_general_results_sha256,
    )


def assemble_m6_base_evaluation(
    *,
    release_config_path: Path,
    base_import_path: Path,
    thinking_pass_directory: Path,
    thinking_judgments_path: Path,
    output_path: Path,
    project_root: Path,
) -> M6EvaluationResult:
    """Assemble complete Base evidence from verified M2 reuse and M6 Thinking."""

    release = load_m6_release_config(release_config_path)
    imported = load_m6_base_import(base_import_path)
    config_sha256 = canonical_config_hash(release)
    if imported.config_sha256 != config_sha256:
        raise M6AssemblyError("M6 Base import and Release config identities differ")
    thinking_summary, thinking = _load_domain_component(
        thinking_pass_directory,
        thinking_judgments_path,
        expected_mode="thinking",
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    git_commit, git_dirty = read_git_identity(project_root.resolve())
    if git_dirty:
        raise M6AssemblyError("formal M6 assembly requires a clean Git worktree")
    raw_domain = _component_sha256(
        {
            "base_import": sha256_file(base_import_path),
            "nonthinking_raw": imported.source_domain_results_sha256,
            "thinking_raw": thinking_summary.raw_results_sha256,
        }
    )
    result = _build_evaluation(
        release=release,
        git_commit=git_commit,
        model=imported.model,
        thinking=thinking,
        nonthinking=imported.nonthinking,
        general=imported.general,
        thinking_review_sha256=cast(str, thinking_summary.human_review_sha256),
        nonthinking_review_sha256=imported.source_human_review_sha256,
        software_environment_sha256=_component_sha256(
            {
                "nonthinking": imported.source_environment_sha256,
                "thinking": thinking_summary.environment_sha256,
            }
        ),
        hardware_sha256=_component_sha256(
            {
                "nonthinking": imported.source_hardware_sha256,
                "thinking": thinking_summary.hardware_sha256,
            }
        ),
        raw_domain_results_sha256=raw_domain,
        raw_general_results_sha256=imported.source_general_tree_sha256,
    )
    _atomic_evaluation(output_path, result)
    return result


def assemble_m6_base_v2_evaluation(
    *,
    release_config_path: Path,
    base_import_path: Path,
    thinking_pass_directory: Path,
    thinking_judgments_path: Path,
    nonthinking_pass_directory: Path,
    nonthinking_judgments_path: Path,
    output_path: Path,
    project_root: Path,
) -> M6EvaluationResult:
    """Assemble a holdout Base from two new domain passes and unchanged general evidence."""

    release = load_m6_release_config(release_config_path)
    imported = load_m6_base_import(base_import_path)
    if release.protocol_version == "m6-release-v1":
        raise M6AssemblyError("full Base assembly requires an M6 holdout protocol")
    config_sha256 = canonical_config_hash(release)
    thinking_summary, thinking = _load_domain_component(
        thinking_pass_directory,
        thinking_judgments_path,
        expected_mode="thinking",
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    nonthinking_summary, nonthinking = _load_domain_component(
        nonthinking_pass_directory,
        nonthinking_judgments_path,
        expected_mode="nonthinking",
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    git_commit, git_dirty = read_git_identity(project_root.resolve())
    if git_dirty:
        raise M6AssemblyError("formal M6 assembly requires a clean Git worktree")
    raw_domain = _component_sha256(
        {
            "base_import": sha256_file(base_import_path),
            "thinking_raw": thinking_summary.raw_results_sha256,
            "nonthinking_raw": nonthinking_summary.raw_results_sha256,
        }
    )
    result = _build_evaluation(
        release=release,
        git_commit=git_commit,
        model=imported.model,
        thinking=thinking,
        nonthinking=nonthinking,
        general=imported.general,
        thinking_review_sha256=cast(str, thinking_summary.human_review_sha256),
        nonthinking_review_sha256=cast(str, nonthinking_summary.human_review_sha256),
        software_environment_sha256=_component_sha256(
            {
                "thinking": thinking_summary.environment_sha256,
                "nonthinking": nonthinking_summary.environment_sha256,
                "general_reuse": imported.source_environment_sha256,
            }
        ),
        hardware_sha256=_component_sha256(
            {
                "thinking": thinking_summary.hardware_sha256,
                "nonthinking": nonthinking_summary.hardware_sha256,
                "general_reuse": imported.source_hardware_sha256,
            }
        ),
        raw_domain_results_sha256=raw_domain,
        raw_general_results_sha256=imported.source_general_tree_sha256,
    )
    _atomic_evaluation(output_path, result)
    return result


def assemble_m6_candidate_evaluation(
    *,
    release_config_path: Path,
    candidate_import_path: Path,
    thinking_pass_directory: Path,
    thinking_judgments_path: Path,
    nonthinking_pass_directory: Path,
    nonthinking_judgments_path: Path,
    general_pass_directory: Path,
    output_path: Path,
    project_root: Path,
) -> M6EvaluationResult:
    """Assemble the complete frozen Candidate evaluation without copying raw text."""

    release = load_m6_release_config(release_config_path)
    imported = load_m6_candidate_import(candidate_import_path)
    config_sha256 = canonical_config_hash(release)
    if imported.config_sha256 != config_sha256:
        raise M6AssemblyError("M6 Candidate import and Release config identities differ")
    thinking_summary, thinking = _load_domain_component(
        thinking_pass_directory,
        thinking_judgments_path,
        expected_mode="thinking",
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    nonthinking_summary, nonthinking = _load_domain_component(
        nonthinking_pass_directory,
        nonthinking_judgments_path,
        expected_mode="nonthinking",
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    general_summary = _load_general_component(
        general_pass_directory,
        expected_model=imported.model,
        expected_config_sha256=config_sha256,
    )
    git_commit, git_dirty = read_git_identity(project_root.resolve())
    if git_dirty:
        raise M6AssemblyError("formal M6 assembly requires a clean Git worktree")
    raw_domain = _component_sha256(
        {
            "candidate_import": sha256_file(candidate_import_path),
            "thinking_raw": thinking_summary.raw_results_sha256,
            "nonthinking_raw": nonthinking_summary.raw_results_sha256,
        }
    )
    result = _build_evaluation(
        release=release,
        git_commit=git_commit,
        model=imported.model,
        thinking=thinking,
        nonthinking=nonthinking,
        general=general_summary.general,
        thinking_review_sha256=cast(str, thinking_summary.human_review_sha256),
        nonthinking_review_sha256=cast(str, nonthinking_summary.human_review_sha256),
        software_environment_sha256=_component_sha256(
            {
                "thinking": thinking_summary.environment_sha256,
                "nonthinking": nonthinking_summary.environment_sha256,
                "general": general_summary.environment_sha256,
            }
        ),
        hardware_sha256=_component_sha256(
            {
                "thinking": thinking_summary.hardware_sha256,
                "nonthinking": nonthinking_summary.hardware_sha256,
                "general": general_summary.hardware_sha256,
            }
        ),
        raw_domain_results_sha256=raw_domain,
        raw_general_results_sha256=general_summary.raw_results_sha256,
    )
    _atomic_evaluation(output_path, result)
    return result
