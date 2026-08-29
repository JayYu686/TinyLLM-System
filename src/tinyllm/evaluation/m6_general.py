"""Formal M6 Candidate general-regression execution."""

from __future__ import annotations

import json
import os
import platform
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from tinyllm.deployment.evaluation_subject import (
    effective_artifact_sha256,
    evaluation_artifact_sha256,
)
from tinyllm.evaluation.baseline import load_baseline_config
from tinyllm.evaluation.baseline_run import run_general_evaluation
from tinyllm.evaluation.baseline_schema import GeneralBaselineSummary
from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_base import sha256_file, sha256_tree
from tinyllm.evaluation.m6_candidate import model_export_sha256
from tinyllm.evaluation.m6_schema import (
    M6GeneralPassSummary,
    M6GeneralResult,
    M6GeneralTaskResult,
    M6ModelIdentity,
)
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash


class M6GeneralError(RuntimeError):
    """Raised when formal M6 general-regression execution fails closed."""


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _general_result(summary: GeneralBaselineSummary) -> M6GeneralResult:
    tasks = tuple(
        M6GeneralTaskResult(
            task=item.task,
            samples=item.samples,
            acc=item.acc,
            acc_stderr=item.acc_stderr,
            acc_norm=item.acc_norm,
            acc_norm_stderr=item.acc_norm_stderr,
        )
        for item in summary.tasks
    )
    typed = cast(
        tuple[M6GeneralTaskResult, M6GeneralTaskResult, M6GeneralTaskResult],
        tasks,
    )
    return M6GeneralResult(
        harness_version=summary.harness_version,
        metric="acc_norm",
        aggregation="equal-task-mean",
        tasks=typed,
        aggregate_basis_points=round(sum(item.acc_norm for item in tasks) * 10_000 / 3),
    )


def _environment_payload(*, adapter_enabled: bool = False) -> dict[str, object]:
    import lm_eval  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "lm_eval": lm_eval.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if adapter_enabled:
        import peft  # type: ignore[import-not-found]

        payload["peft"] = peft.__version__
    return payload


def _hardware_payload(physical_gpu_index: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": "1.0",
        "physical_gpu_index": physical_gpu_index,
        "logical_gpu_index": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_total_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def run_m6_general_pass(
    *,
    release_config_path: Path,
    artifact_root: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    project_root: Path,
    physical_gpu_index: int,
    model_identity: M6ModelIdentity,
    expected_config_sha256: str,
    adapter_dir: Path | None = None,
    base_model_artifact_sha256: str | None = None,
) -> M6GeneralPassSummary:
    """Run the full frozen lm-eval suite for the M6 Candidate."""

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M6GeneralError("M6 general evaluation requires exactly one visible CUDA GPU")
    project_root = project_root.resolve()
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M6GeneralError("formal M6 general evaluation requires a clean Git worktree")
    if output_dir.exists() or not output_dir.is_absolute() or not artifact_root.is_absolute():
        raise M6GeneralError("M6 general artifact paths must be absolute and output absent")
    release = load_m6_release_config(release_config_path)
    if canonical_config_hash(release) != expected_config_sha256:
        raise M6GeneralError("M6 Candidate import and Release config identities differ")
    if model_identity.role != "candidate":
        raise M6GeneralError("M6 general model differs from the imported Candidate")
    if model_identity.adaptation == "full_sft":
        valid_artifact = (
            adapter_dir is None
            and base_model_artifact_sha256 is None
            and model_export_sha256(model_dir) == model_identity.model_artifact_sha256
        )
    elif model_identity.adaptation == "lora":
        if adapter_dir is None or base_model_artifact_sha256 is None:
            valid_artifact = False
        else:
            try:
                adapter_sha256 = evaluation_artifact_sha256(
                    adapter_dir,
                    ("adapter_config.json", "adapter_model.safetensors"),
                )
            except (OSError, RuntimeError, ValueError):
                valid_artifact = False
            else:
                valid_artifact = (
                    adapter_sha256 == model_identity.adapter_sha256
                    and effective_artifact_sha256(
                        base_model_artifact_sha256,
                        adapter_sha256,
                    )
                    == model_identity.model_artifact_sha256
                )
    else:
        valid_artifact = False
    if not valid_artifact:
        raise M6GeneralError("M6 general model differs from the imported Candidate")
    if not tokenizer_dir.is_absolute() or not tokenizer_dir.is_dir():
        raise M6GeneralError("M6 general tokenizer must be an absolute existing directory")
    baseline_config = load_baseline_config(project_root / "configs/eval/m2_baseline.yaml")
    started = time.monotonic()
    general_summary = run_general_evaluation(
        baseline_config,
        project_root=project_root,
        artifact_root=artifact_root,
        model_path=model_dir,
        tokenizer_path=tokenizer_dir,
        adapter_path=adapter_dir,
        output_path=output_dir,
        device="cuda",
        offline=True,
    )
    environment_path = output_dir / "environment.json"
    hardware_path = output_dir / "hardware.json"
    _atomic_json(
        environment_path,
        _environment_payload(adapter_enabled=model_identity.adaptation == "lora"),
    )
    _atomic_json(hardware_path, _hardware_payload(physical_gpu_index))
    result = M6GeneralPassSummary(
        status="succeeded",
        evaluation_id=(
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-m6-general-candidate-"
            f"{model_identity.model_artifact_sha256[:8]}"
        ),
        protocol_version=release.protocol_version,
        config_sha256=canonical_config_hash(release),
        git_commit=git_commit,
        git_dirty=False,
        model=model_identity,
        general=_general_result(general_summary),
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(0),
        duration_seconds=time.monotonic() - started,
        environment_sha256=sha256_file(environment_path),
        hardware_sha256=sha256_file(hardware_path),
        raw_results_sha256=sha256_tree(output_dir / "raw"),
    )
    _atomic_json(output_dir / "summary.json", result.to_dict())
    return result
