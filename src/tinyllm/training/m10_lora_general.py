"""Paired M6-v7 general-regression execution for M10 8B LoRA subjects."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from tinyllm.deployment import (
    ResolvedEvaluationSubject,
    resolve_evaluation_subject,
    resolve_m10_lora_stage_evaluation_subject,
)
from tinyllm.evaluation import load_m6_release_config
from tinyllm.evaluation.baseline import load_baseline_config
from tinyllm.evaluation.baseline_run import run_general_evaluation
from tinyllm.evaluation.m6_base import sha256_file, sha256_tree
from tinyllm.evaluation.m6_general import (
    _atomic_json,
    _environment_payload,
    _general_result,
    _hardware_payload,
)
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_lora_schema import (
    M10_LORA_PARENT_SUBJECT,
    M10LoRAGeneralPassSummary,
)


class M10LoRAGeneralError(RuntimeError):
    """Raised when paired 8B general-regression evidence cannot be trusted."""


def _resolve_subject(artifact_root: Path, subject_id: str) -> ResolvedEvaluationSubject:
    if subject_id == M10_LORA_PARENT_SUBJECT:
        return resolve_evaluation_subject(artifact_root, subject_id)
    if subject_id.startswith(
        (
            "qwen3-8b-m10-agent-lora-3m-",
            "qwen3-8b-m10-agent-lora-4m-",
            "qwen3-8b-m10-agent-lora-5m-",
        )
    ):
        return resolve_m10_lora_stage_evaluation_subject(artifact_root, subject_id)
    raise M10LoRAGeneralError(
        "M10 LoRA M6 accepts only the frozen parent or a selected 3M/4M/5M stage"
    )


def _evaluation_id(
    *, kind: str, subject: ResolvedEvaluationSubject, raw_results_sha256: str
) -> str:
    identity = json.dumps(
        {
            "evaluation_subject_sha256": subject.evaluation_subject_sha256,
            "model_artifact_sha256": subject.model_artifact_sha256,
            "raw_results_sha256": raw_results_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"m10-lora-m6-general-{kind}-{hashlib.sha256(identity).hexdigest()[:8]}"


def run_m10_lora_general_pass(
    *,
    artifact_root: Path,
    subject_id: str,
    output_dir: Path,
    project_root: Path,
    release_config_path: Path,
    physical_gpu_index: int,
) -> M10LoRAGeneralPassSummary:
    """Evaluate one immutable 8B Base/LoRA subject on the frozen M6 general suite."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M10LoRAGeneralError("M10 LoRA M6 requires exactly one visible CUDA GPU")
    project_root = project_root.resolve()
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M10LoRAGeneralError("formal M10 LoRA M6 evaluation requires clean Git")
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not output_dir.is_absolute()
        or output_dir.exists()
    ):
        raise M10LoRAGeneralError("M10 LoRA M6 Artifact paths are unsafe or already exist")
    subject = _resolve_subject(artifact_root, subject_id)
    release = load_m6_release_config(release_config_path)
    if release.protocol_version != "m6-release-v7":
        raise M10LoRAGeneralError("M10 LoRA M6 requires the frozen v7 Release config")

    baseline_config = load_baseline_config(project_root / "configs/eval/m2_baseline.yaml")
    started = time.monotonic()
    general_summary = run_general_evaluation(
        baseline_config,
        project_root=project_root,
        artifact_root=artifact_root,
        model_path=subject.model_dir,
        tokenizer_path=subject.tokenizer_dir,
        adapter_path=subject.adapter_dir,
        output_path=output_dir,
        device="cuda",
        offline=True,
    )
    environment_path = output_dir / "environment.json"
    hardware_path = output_dir / "hardware.json"
    _atomic_json(
        environment_path,
        _environment_payload(adapter_enabled=subject.adapter_dir is not None),
    )
    _atomic_json(hardware_path, _hardware_payload(physical_gpu_index))
    raw_results_sha256 = sha256_tree(output_dir / "raw")
    kind = "parent" if subject_id == M10_LORA_PARENT_SUBJECT else "candidate"
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise M10LoRAGeneralError("M10 LoRA M6 formal evidence requires an RTX 3090")
    result = M10LoRAGeneralPassSummary(
        status="succeeded",
        evaluation_id=_evaluation_id(
            kind=kind,
            subject=subject,
            raw_results_sha256=raw_results_sha256,
        ),
        protocol_version="m6-release-v7",
        config_sha256=canonical_config_hash(release),
        git_commit=git_commit,
        git_dirty=False,
        evaluation_subject_id=subject_id,
        evaluation_subject_sha256=subject.evaluation_subject_sha256,
        model=subject.model,
        general=_general_result(general_summary),
        physical_gpu_index=physical_gpu_index,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=time.monotonic() - started,
        environment_sha256=sha256_file(environment_path),
        hardware_sha256=sha256_file(hardware_path),
        raw_results_sha256=raw_results_sha256,
    )
    _atomic_json(output_dir / "summary.json", result.to_dict())
    return result


__all__ = ["M10LoRAGeneralError", "run_m10_lora_general_pass"]
