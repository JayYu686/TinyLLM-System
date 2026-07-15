"""Private Run creation and orchestration for the complete M2.4c Baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tinyllm.evaluation.baseline import (
    BaselineContractError,
    build_lm_eval_command,
    build_lm_eval_validation_command,
    load_baseline_config,
    load_human_rubric_judgments,
    validate_baseline_inputs,
)
from tinyllm.evaluation.baseline_inference import (
    TransformersDomainBackend,
    run_domain_generation,
    seed_baseline,
)
from tinyllm.evaluation.baseline_results import (
    build_domain_summary,
    load_general_summary,
)
from tinyllm.evaluation.baseline_runtime import (
    BaselinePreflightError,
    BaselineRuntime,
    BaselineRuntimeError,
    acquire_baseline_model,
    load_baseline_runtime,
)
from tinyllm.evaluation.baseline_schema import (
    BaselineEvaluationResult,
    BaselineRunConfig,
    DomainBaselineSummary,
    DomainItemResult,
    GeneralBaselineSummary,
    HumanReviewCommit,
)
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import ArtifactRoots, RunManifest, RunStatus
from tinyllm.schemas.run import canonical_config_hash, generate_run_id


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _atomic_jsonl(path: Path, values: tuple[Mapping[str, object], ...]) -> None:
    payload = _jsonl_bytes(values)
    _atomic_bytes(path, payload)


def _jsonl_bytes(values: tuple[Mapping[str, object], ...]) -> bytes:
    return "".join(json.dumps(value, sort_keys=True) + "\n" for value in values).encode()


def _append_event(path: Path, value: Mapping[str, object]) -> None:
    payload = {"timestamp": datetime.now(UTC).isoformat(), **value}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_branch(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineContractError("cannot determine Git branch for Baseline") from exc


def _formal_worktree_dirty(project_root: Path) -> bool:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineContractError("cannot determine Git worktree state") from exc


def _runtime_environment(
    config: BaselineRunConfig,
    runtime: BaselineRuntime,
    *,
    device: Literal["cpu", "cuda"],
    git_branch: str,
) -> dict[str, object]:
    try:
        packages = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselinePreflightError("cannot capture complete Python environment") from exc
    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(runtime.torch.__version__),
        "cuda_runtime": runtime.torch.version.cuda,
        "transformers": config.software.transformers,
        "tokenizers": config.software.tokenizers,
        "accelerate": config.software.accelerate,
        "datasets": config.software.datasets,
        "lm_eval": config.software.lm_eval,
        "safetensors": config.software.safetensors,
        "device": device,
        "git_branch": git_branch,
        "pip_freeze": sorted(line for line in packages if line),
    }


def _runtime_hardware(
    runtime: BaselineRuntime, *, device: Literal["cpu", "cuda"]
) -> dict[str, object]:
    hardware: dict[str, object] = {
        "schema_version": "1.0",
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device == "cuda":
        physical_index = os.environ.get("CUDA_VISIBLE_DEVICES")
        if physical_index is None or re.fullmatch(r"[0-9]+", physical_index) is None:
            raise BaselinePreflightError(
                "CUDA Baseline requires one numeric physical GPU visibility"
            )
        try:
            inventory = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    physical_index,
                    "--query-gpu=index,driver_version,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            fields = tuple(field.strip() for field in inventory.split(","))
            if len(fields) != 5 or fields[0] != physical_index:
                raise ValueError("selected GPU inventory does not match visibility")
            start_memory, start_utilization, start_temperature = (
                int(fields[2]),
                int(fields[3]),
                int(fields[4]),
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise BaselinePreflightError("cannot capture selected GPU hardware lineage") from exc
        properties = runtime.torch.cuda.get_device_properties(0)
        hardware.update(
            {
                "visible_device_count": runtime.torch.cuda.device_count(),
                "logical_device": 0,
                "physical_device": int(physical_index),
                "gpu_name": properties.name,
                "memory_total_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": bool(runtime.torch.cuda.is_bf16_supported()),
                "driver_version": fields[1],
                "start_memory_used_mib": start_memory,
                "start_utilization_percent": start_utilization,
                "start_temperature_c": start_temperature,
            }
        )
    return hardware


def _write_process_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    _atomic_bytes(
        path,
        (f"returncode={completed.returncode}\n{completed.stdout}\n{completed.stderr}").encode(),
    )


def _run_general_baseline(
    config: BaselineRunConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    model_path: Path,
    output_path: Path,
    device: Literal["cpu", "cuda"],
    offline: bool,
) -> GeneralBaselineSummary:
    output_path.mkdir(parents=True, exist_ok=False)
    raw_path = output_path / "raw"
    environment = os.environ.copy()
    hf_home = artifact_root / "cache/huggingface"
    environment.update(
        {
            "HF_HOME": str(hf_home),
            "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        }
    )
    if offline:
        environment.update(
            {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    validation_command = build_lm_eval_validation_command(config, project_root=project_root)
    _atomic_json(output_path / "validation.command.json", list(validation_command))
    validation = subprocess.run(
        validation_command,
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    _write_process_log(output_path / "validation.log", validation)
    if validation.returncode != 0:
        raise BaselineRuntimeError("lm-eval task validation failed")
    command = build_lm_eval_command(
        config,
        project_root=project_root,
        model_path=model_path,
        output_path=raw_path,
        device="cuda:0" if device == "cuda" else "cpu",
    )
    _atomic_json(output_path / "run.command.json", list(command))
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    _write_process_log(output_path / "run.log", completed)
    if completed.returncode != 0:
        raise BaselineRuntimeError(f"lm-eval execution failed with code {completed.returncode}")
    return load_general_summary(config, output_path=raw_path)


def _new_run_directory(
    *,
    config_path: Path,
    config: BaselineRunConfig,
    roots: ArtifactRoots,
    manifest: RunManifest,
    environment: Mapping[str, object],
    hardware: Mapping[str, object],
) -> Path:
    run_directory = roots.run_directory(manifest.run_id)
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "checkpoints").mkdir()
    (run_directory / "evaluations/domain").mkdir(parents=True)
    (run_directory / "exports").mkdir()
    _atomic_bytes(run_directory / "config.original.yaml", config_path.read_bytes())
    _atomic_json(run_directory / "config.resolved.json", config.to_dict())
    _atomic_json(run_directory / "environment.json", dict(environment))
    _atomic_json(run_directory / "hardware.json", dict(hardware))
    _atomic_json(run_directory / "run.json", manifest.to_dict())
    _atomic_bytes(run_directory / "events.jsonl", b"")
    _atomic_bytes(run_directory / "metrics.jsonl", b"")
    return run_directory


def run_baseline_evaluation(
    *,
    config_path: Path,
    project_root: Path,
    artifact_root: Path,
    device: Literal["cpu", "cuda"],
    offline: bool,
) -> BaselineEvaluationResult:
    """Execute Domain and general Baselines into one private, traceable Run directory."""

    project_root = project_root.resolve()
    config_path = config_path.resolve()
    if not project_root.is_dir() or not config_path.is_file():
        raise BaselineContractError("project root and Baseline config must exist")
    if not config_path.is_relative_to(project_root):
        raise BaselineContractError("Baseline config must be inside the project root")
    config = load_baseline_config(config_path)
    items = validate_baseline_inputs(config, project_root=project_root)
    git_commit, git_dirty = read_git_identity(project_root)
    branch = _git_branch(project_root)
    if config.mode == "formal":
        if device != "cuda":
            raise BaselineContractError("formal Baseline requires one CUDA device")
        if branch != "main" or git_dirty or _formal_worktree_dirty(project_root):
            raise BaselineContractError("formal Baseline requires a clean main worktree")
    roots = ArtifactRoots(root=artifact_root)
    model_path = acquire_baseline_model(config, cache_root=roots.cache, offline=offline)
    runtime = load_baseline_runtime(config)
    if device == "cuda" and not bool(runtime.torch.cuda.is_available()):
        raise BaselinePreflightError("CUDA is unavailable for the Baseline")
    config_hash = canonical_config_hash(config)
    now = datetime.now(UTC)
    run_id = generate_run_id(config.run_slug, config_hash, now=now)
    environment = _runtime_environment(config, runtime, device=device, git_branch=branch)
    hardware = _runtime_hardware(runtime, device=device)
    manifest = RunManifest(
        run_id=run_id,
        name=config.run_slug,
        status=RunStatus.RUNNING,
        created_at=now,
        updated_at=now,
        config_hash=config_hash,
        git_commit=git_commit,
        git_dirty=git_dirty,
        artifact_root=artifact_root,
        strategy="evaluation",
        world_size=1,
        dataset_version=config.domain.suite_version,
        tokenizer_revision=config.model.revision,
    )
    run_directory = _new_run_directory(
        config_path=config_path,
        config=config,
        roots=roots,
        manifest=manifest,
        environment=environment,
        hardware=hardware,
    )
    events_path = run_directory / "events.jsonl"
    _append_event(
        events_path,
        {
            "event": "baseline_started",
            "mode": config.mode,
            "run_id": run_id,
            "offline": offline,
        },
    )
    try:
        seed_baseline(config, runtime)
        backend = TransformersDomainBackend(
            runtime=runtime,
            config=config,
            model_path=model_path,
            device=device,
        )
        domain_results = run_domain_generation(config, items, backend=backend)
        domain_summary = build_domain_summary(config, domain_results)
        _atomic_jsonl(
            run_directory / "evaluations/domain/results.jsonl",
            tuple(result.to_dict() for result in domain_results),
        )
        _atomic_json(run_directory / "evaluations/domain/summary.json", domain_summary.to_dict())
        del backend
        gc.collect()
        if device == "cuda":
            runtime.torch.cuda.empty_cache()
        general_summary = _run_general_baseline(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            model_path=model_path,
            output_path=run_directory / "evaluations/general",
            device=device,
            offline=offline,
        )
        status: Literal["succeeded", "awaiting_human_review"] = (
            "awaiting_human_review"
            if domain_summary.status == "awaiting_human_review"
            else "succeeded"
        )
        result = BaselineEvaluationResult(
            status=status,
            mode=config.mode,
            run_id=run_id,
            config_sha256=config_hash,
            git_commit=git_commit,
            git_dirty=git_dirty,
            model_repository=config.model.repository,
            model_revision=config.model.revision,
            domain=domain_summary,
            general=general_summary,
        )
        _atomic_json(run_directory / "evaluations/summary.json", result.to_dict())
        metric_records: list[Mapping[str, object]] = [
            {"scope": "domain", **domain_summary.to_dict()}
        ]
        metric_records.extend(
            {"scope": "general", **task.to_dict()} for task in general_summary.tasks
        )
        _atomic_jsonl(run_directory / "metrics.jsonl", tuple(metric_records))
        completed_at = datetime.now(UTC)
        completed_manifest = manifest.model_copy(
            update={
                "status": (
                    RunStatus.EVALUATING
                    if status == "awaiting_human_review"
                    else RunStatus.SUCCEEDED
                ),
                "updated_at": completed_at,
            }
        )
        _atomic_json(run_directory / "run.json", completed_manifest.to_dict())
        _append_event(
            events_path,
            {"event": "baseline_completed", "run_id": run_id, "status": status},
        )
        return result
    except Exception as exc:
        failed_manifest = manifest.model_copy(
            update={"status": RunStatus.FAILED, "updated_at": datetime.now(UTC)}
        )
        _atomic_json(run_directory / "run.json", failed_manifest.to_dict())
        _append_event(
            events_path,
            {
                "event": "baseline_failed",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def _validate_committed_human_review(
    review_directory: Path,
    *,
    run_id: str,
    config_sha256: str,
    judgment_bytes: bytes,
    domain_summary: DomainBaselineSummary,
    completed_result: BaselineEvaluationResult,
) -> None:
    """Validate a published review directory before resuming outer Run finalization."""

    if not review_directory.is_dir() or review_directory.is_symlink():
        raise BaselineContractError("committed human-rubric review directory is unsafe")
    try:
        stored_judgments = (review_directory / "judgments.jsonl").read_bytes()
        stored_domain = DomainBaselineSummary.model_validate_json(
            (review_directory / "domain.summary.json").read_text(encoding="utf-8")
        )
        stored_result = BaselineEvaluationResult.model_validate_json(
            (review_directory / "baseline.summary.json").read_text(encoding="utf-8")
        )
        commit = HumanReviewCommit.model_validate_json(
            (review_directory / "commit.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise BaselineContractError("cannot validate committed human-rubric review") from exc
    expected_hash = hashlib.sha256(judgment_bytes).hexdigest()
    if (
        stored_judgments != judgment_bytes
        or stored_domain != domain_summary
        or stored_result != completed_result
        or commit.run_id != run_id
        or commit.config_sha256 != config_sha256
        or commit.judgment_count != domain_summary.human_reviewed
        or commit.judgments_sha256 != expected_hash
    ):
        raise BaselineContractError("committed human-rubric review does not match this request")


def _human_review_event_exists(events_path: Path, *, run_id: str) -> bool:
    """Return whether final review lineage already exists, rejecting malformed event logs."""

    try:
        with events_path.open(encoding="utf-8") as stream:
            events = tuple(json.loads(line) for line in stream if line.strip())
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineContractError("cannot validate Baseline event lineage") from exc
    return any(
        event.get("event") == "baseline_human_review_completed" and event.get("run_id") == run_id
        for event in events
        if isinstance(event, dict)
    )


def complete_baseline_human_review(
    *,
    run_id: str,
    artifact_root: Path,
    judgments_path: Path,
) -> BaselineEvaluationResult:
    """Commit all frozen human-rubric Judgments and finalize an awaiting Baseline Run."""

    roots = ArtifactRoots(root=artifact_root)
    run_directory = roots.run_directory(run_id)
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise BaselineContractError("Baseline Run does not exist or is unsafe")
    try:
        manifest = RunManifest.model_validate_json(
            (run_directory / "run.json").read_text(encoding="utf-8")
        )
        config = BaselineRunConfig.model_validate_json(
            (run_directory / "config.resolved.json").read_text(encoding="utf-8")
        )
        result = BaselineEvaluationResult.model_validate_json(
            (run_directory / "evaluations/summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise BaselineContractError("cannot load awaiting Baseline Run state") from exc
    if manifest.run_id != run_id or result.run_id != run_id:
        raise BaselineContractError("Baseline Run identity mismatch")
    state = (manifest.status, result.status)
    recoverable_states = {
        (RunStatus.EVALUATING, "awaiting_human_review"),
        (RunStatus.EVALUATING, "succeeded"),
        (RunStatus.SUCCEEDED, "succeeded"),
    }
    if state not in recoverable_states:
        raise BaselineContractError("Baseline Run is not reviewable or recoverable")
    if (
        canonical_config_hash(config) != result.config_sha256
        or manifest.config_hash != result.config_sha256
    ):
        raise BaselineContractError("Baseline Run config identity mismatch")

    raw_results_path = run_directory / "evaluations/domain/results.jsonl"
    domain_results: list[DomainItemResult] = []
    try:
        with raw_results_path.open(encoding="utf-8") as stream:
            for line in stream:
                domain_results.append(DomainItemResult.model_validate_json(line))
    except (OSError, ValueError) as exc:
        raise BaselineContractError("cannot validate private Domain results") from exc
    pending_summary = build_domain_summary(config, domain_results)
    if result.status == "awaiting_human_review" and pending_summary != result.domain:
        raise BaselineContractError("private Domain results do not match awaiting summary")
    expected_ids = tuple(
        domain_result.item_id
        for domain_result in domain_results
        if domain_result.human_review_required
    )
    judgments = load_human_rubric_judgments(judgments_path)
    if tuple(judgment.item_id for judgment in judgments) != expected_ids:
        raise BaselineContractError(
            "human-rubric Judgments must cover every pending item in Run order"
        )
    if len(judgments) != pending_summary.human_review_pending:
        raise BaselineContractError("human-rubric Judgment count does not match pending state")

    domain_summary = DomainBaselineSummary.model_validate(
        {
            **pending_summary.to_dict(),
            "status": "complete",
            "human_review_pending": 0,
            "human_reviewed": len(judgments),
            "human_passed": sum(judgment.passed for judgment in judgments),
        }
    )
    completed_result = BaselineEvaluationResult.model_validate(
        {**result.to_dict(), "status": "succeeded", "domain": domain_summary.to_dict()}
    )
    if result.status == "succeeded" and result != completed_result:
        raise BaselineContractError("completed Baseline summary does not match private results")
    review_directory = run_directory / "evaluations/domain/human_review"
    judgment_records = tuple(judgment.to_dict() for judgment in judgments)
    judgment_bytes = _jsonl_bytes(judgment_records)
    if review_directory.exists() or review_directory.is_symlink():
        _validate_committed_human_review(
            review_directory,
            run_id=run_id,
            config_sha256=result.config_sha256,
            judgment_bytes=judgment_bytes,
            domain_summary=domain_summary,
            completed_result=completed_result,
        )
    else:
        temporary = review_directory.with_name(f".human-review-partial-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            _atomic_bytes(temporary / "judgments.jsonl", judgment_bytes)
            _atomic_json(temporary / "domain.summary.json", domain_summary.to_dict())
            _atomic_json(temporary / "baseline.summary.json", completed_result.to_dict())
            review_commit = HumanReviewCommit(
                run_id=run_id,
                config_sha256=result.config_sha256,
                committed_at=datetime.now(UTC),
                judgment_count=len(judgments),
                judgments_sha256=hashlib.sha256(judgment_bytes).hexdigest(),
            )
            _atomic_json(temporary / "commit.json", review_commit.to_dict())
            os.replace(temporary, review_directory)
        except OSError:
            if temporary.is_dir():
                for path in temporary.iterdir():
                    path.unlink(missing_ok=True)
                temporary.rmdir()
            raise

    _atomic_json(run_directory / "evaluations/domain/summary.json", domain_summary.to_dict())
    _atomic_json(run_directory / "evaluations/summary.json", completed_result.to_dict())
    metric_records: list[Mapping[str, object]] = [{"scope": "domain", **domain_summary.to_dict()}]
    metric_records.extend({"scope": "general", **task.to_dict()} for task in result.general.tasks)
    _atomic_jsonl(run_directory / "metrics.jsonl", tuple(metric_records))
    completed_manifest = RunManifest.model_validate(
        {
            **manifest.model_dump(),
            "status": RunStatus.SUCCEEDED,
            "updated_at": datetime.now(UTC),
        }
    )
    _atomic_json(run_directory / "run.json", completed_manifest.to_dict())
    events_path = run_directory / "events.jsonl"
    if not _human_review_event_exists(events_path, run_id=run_id):
        _append_event(
            events_path,
            {
                "event": "baseline_human_review_completed",
                "run_id": run_id,
                "reviewed": len(judgments),
                "passed": domain_summary.human_passed,
            },
        )
    return completed_result
