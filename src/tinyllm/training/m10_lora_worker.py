"""CUDA worker for staged Qwen3-8B Agent LoRA and its memory Probe."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from tinyllm.data.m10_mixture import M10FrozenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m5_ablation import group_loss_scale, token_learning_rate
from tinyllm.training.m10_lora import (
    M10LoRACheckpointStore,
    M10LoRAError,
    M10LoRAProgress,
    collect_m10_lora_environment,
    collect_m10_lora_hardware,
    export_m10_lora_stage,
    load_m10_lora_continuation_gate,
    load_m10_lora_memory_probe,
    preflight_m10_lora,
    record_m10_lora_result,
)
from tinyllm.training.m10_lora_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10_LORA_TARGET_MODULES,
    M10LoRAConfig,
    M10LoRAMemoryProbeResult,
    M10LoRARunResult,
)
from tinyllm.training.m10_sft import _append_jsonl, _atomic_json, _batch, epoch_order
from tinyllm.training.m10_sft_schema import M10_STAGE_TOKENS
from tinyllm.training.seed import seed_everything


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M10LoRAError("M10 Agent LoRA worker requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise M10LoRAError("M10 Agent LoRA worker requires BF16 support")
    return torch.device("cuda", 0)


def _build_model(
    config: M10LoRAConfig, *, model_dir: Path, device: torch.device
) -> tuple[Any, tuple[nn.Parameter, ...], int, int]:
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()
    lora = config.model.lora
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(M10_LORA_TARGET_MODULES),
            bias=lora.bias,
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters <= 0 or trainable_parameters >= total_parameters:
        raise M10LoRAError("M10 Agent LoRA trainable-parameter topology is invalid")
    return model, trainable, trainable_parameters, total_parameters


def _optimizer_step(
    *,
    model: Any,
    trainable: tuple[nn.Parameter, ...],
    optimizer: Optimizer,
    dataset: Any,
    indices: tuple[int, ...],
    micro_batch: int,
    learning_rate: float,
    max_grad_norm: float,
    device: torch.device,
) -> tuple[float, float, int]:
    examples = tuple(dataset[index] for index in indices)
    group_tokens = sum(int(item["labels"][1:].ne(-100).sum()) for item in examples)
    if group_tokens <= 0:
        raise M10LoRAError("M10 Agent LoRA accumulation group has no supervised Tokens")
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for offset in range(0, len(indices), micro_batch):
        micro_indices = indices[offset : offset + micro_batch]
        batch = {key: value.to(device) for key, value in _batch(dataset, micro_indices).items()}
        valid_tokens = int(batch["labels"][:, 1:].ne(-100).sum())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch, use_cache=False)
            loss = getattr(output, "loss", None)
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise M10LoRAError("M10 Agent LoRA produced a non-finite scalar loss")
        scale = group_loss_scale(valid_tokens, group_tokens)
        torch.autograd.backward(loss * scale)
        weighted_loss += float(loss.detach().float()) * scale
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
    if not math.isfinite(gradient_norm):
        raise M10LoRAError("M10 Agent LoRA produced a non-finite gradient norm")
    optimizer.step()
    return weighted_loss, gradient_norm, group_tokens


def run_m10_lora_memory_probe(
    *,
    config_path: Path,
    mixture_root: Path,
    artifact_root: Path,
    output_path: Path,
    physical_gpu_index: int,
) -> M10LoRAMemoryProbeResult:
    """Run ten real optimizer steps without creating a formal training Run."""

    device = _require_cuda()
    config, parent, _ = preflight_m10_lora(
        config_path=config_path,
        mixture_root=mixture_root,
        artifact_root=artifact_root,
    )
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M10LoRAError("formal M10 Agent LoRA Probe requires a clean Git worktree")
    if not output_path.is_absolute() or output_path.is_symlink() or output_path.exists():
        raise M10LoRAError("M10 Agent LoRA Probe output must be a new absolute non-symlink path")
    try:
        root = artifact_root.resolve(strict=True)
        mixture = mixture_root.resolve(strict=True)
        output_parent = output_path.parent.resolve(strict=True)
    except (OSError, FileNotFoundError) as exc:
        raise M10LoRAError("M10 Agent LoRA Probe output parent is unavailable") from exc
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not mixture_root.is_absolute()
        or mixture_root.is_symlink()
        or not mixture.is_relative_to(root)
        or not output_parent.is_relative_to(root)
    ):
        raise M10LoRAError("M10 Agent LoRA Probe input/output escapes the Artifact Store")

    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    _, environment_sha256 = collect_m10_lora_environment()
    _, hardware_sha256 = collect_m10_lora_hardware(physical_gpu_index)
    seed_everything(config.run.seed, deterministic_algorithms=False)
    dataset = M10FrozenDataset(mixture_root)
    model, trainable, _, _ = _build_model(config, model_dir=parent.model_dir, device=device)
    optimizer = AdamW(
        trainable,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    order = epoch_order(len(dataset), seed=config.run.seed, epoch=0)
    group_size = (
        config.optimization.micro_batch_size * config.optimization.gradient_accumulation_steps
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    supervised_tokens = 0
    for step in range(config.memory_probe.optimizer_steps):
        indices = order[step * group_size : (step + 1) * group_size]
        if len(indices) != group_size:
            raise M10LoRAError("M10 Agent LoRA Probe Dataset is too small")
        _, _, tokens = _optimizer_step(
            model=model,
            trainable=trainable,
            optimizer=optimizer,
            dataset=dataset,
            indices=indices,
            micro_batch=config.optimization.micro_batch_size,
            learning_rate=config.optimization.learning_rate,
            max_grad_norm=config.optimization.max_grad_norm,
            device=device,
        )
        supervised_tokens += tokens
    torch.cuda.synchronize(device)
    result = M10LoRAMemoryProbeResult(
        config_sha256=canonical_config_hash(config),
        git_commit=git_commit,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
        environment_sha256=environment_sha256,
        hardware_compatibility_sha256=hardware_sha256,
        physical_gpu_index=physical_gpu_index,
        gpu_name=cast(Literal["NVIDIA GeForce RTX 3090"], torch.cuda.get_device_name(device)),
        optimizer_steps=10,
        supervised_tokens=supervised_tokens,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        duration_seconds=time.monotonic() - started,
    )
    _atomic_json(output_path, result.to_dict())
    return result


def run_m10_lora(
    *,
    config_path: Path,
    mixture_root: Path,
    artifact_root: Path,
    output_root: Path,
    physical_gpu_index: int,
    memory_probe_path: Path,
    resume_run: Path | None = None,
    stop_after_tokens: int | None = None,
    continuation_gate_path: Path | None = None,
) -> M10LoRARunResult:
    """Train or Exact-Resume one formal Qwen3-8B Agent LoRA stage."""

    from peft import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    device = _require_cuda()
    config, parent, manifest_sha256 = preflight_m10_lora(
        config_path=config_path,
        mixture_root=mixture_root,
        artifact_root=artifact_root,
    )
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M10LoRAError("formal M10 Agent LoRA training requires a clean Git worktree")
    config_sha256 = canonical_config_hash(config)
    try:
        root = artifact_root.resolve(strict=True)
        resolved_mixture = mixture_root.resolve(strict=True)
        resolved_output = output_root.resolve(strict=True)
        resolved_probe = memory_probe_path.resolve(strict=True)
        resolved_resume = resume_run.resolve(strict=True) if resume_run is not None else None
        resolved_gate = (
            continuation_gate_path.resolve(strict=True)
            if continuation_gate_path is not None
            else None
        )
    except (OSError, FileNotFoundError) as exc:
        raise M10LoRAError("M10 Agent LoRA Artifact path is unavailable") from exc
    checked_paths = (
        resolved_mixture,
        resolved_output,
        resolved_probe,
        resolved_resume,
        resolved_gate,
    )
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or not mixture_root.is_absolute()
        or mixture_root.is_symlink()
        or not output_root.is_absolute()
        or output_root.is_symlink()
        or not memory_probe_path.is_absolute()
        or memory_probe_path.is_symlink()
        or (resume_run is not None and (not resume_run.is_absolute() or resume_run.is_symlink()))
        or (
            continuation_gate_path is not None
            and (not continuation_gate_path.is_absolute() or continuation_gate_path.is_symlink())
        )
        or any(path is not None and not path.is_relative_to(root) for path in checked_paths)
    ):
        raise M10LoRAError("M10 Agent LoRA Run paths must stay inside the Artifact Store")
    memory_probe, memory_probe_sha256 = load_m10_lora_memory_probe(
        memory_probe_path,
        config_sha256=config_sha256,
        git_commit=git_commit,
    )
    target_tokens = (
        config.optimization.max_train_tokens if stop_after_tokens is None else stop_after_tokens
    )
    if target_tokens not in M10_STAGE_TOKENS:
        raise M10LoRAError("M10 Agent LoRA stop must be exactly 1M, 5M, or 10M Tokens")
    if resume_run is None and target_tokens != 1_000_000:
        raise M10LoRAError("fresh M10 Agent LoRA training must stop at 1M Tokens")
    if resume_run is None and continuation_gate_path is not None:
        raise M10LoRAError("fresh M10 Agent LoRA training cannot consume a Gate")

    environment, environment_sha256 = collect_m10_lora_environment()
    hardware, hardware_sha256 = collect_m10_lora_hardware(physical_gpu_index)
    if (
        memory_probe.environment_sha256 != environment_sha256
        or memory_probe.hardware_compatibility_sha256 != hardware_sha256
    ):
        raise M10LoRAError("M10 Agent LoRA Probe and training runtime differ")
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    seed_everything(config.run.seed, deterministic_algorithms=False)
    dataset = M10FrozenDataset(mixture_root)
    model, trainable, trainable_parameters, total_parameters = _build_model(
        config, model_dir=parent.model_dir, device=device
    )
    optimizer = AdamW(
        trainable,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    mode: Literal["fresh", "exact_resume"] = "fresh"
    resumed_from_tokens: int | None = None
    continuation_gate_sha256: str | None = None
    source_stage_tokens: int | None = None
    source_stage_adapter_sha256: str | None = None
    if resume_run is None:
        run_id = generate_run_id(config.run.name, config_sha256, now=datetime.now(UTC))
        artifact_dir = output_root / run_id
        artifact_dir.mkdir(parents=True)
        shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
        _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
        _atomic_json(artifact_dir / "dataset.json", dataset.manifest.to_dict())
        _atomic_json(artifact_dir / "parent.json", parent.to_dict())
        _atomic_json(artifact_dir / "environment.json", environment)
        _atomic_json(artifact_dir / "hardware.json", hardware)
        progress = M10LoRAProgress(0, 0, 0, 0, None, None)
    else:
        artifact_dir = resume_run
        try:
            run_payload = json.loads((artifact_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise M10LoRAError("M10 Agent LoRA Resume metadata is missing") from exc
        if (
            run_payload.get("config_sha256") != config_sha256
            or run_payload.get("memory_probe_sha256") != memory_probe_sha256
            or run_payload.get("parent_evaluation_subject") != M10_LORA_PARENT_SUBJECT
        ):
            raise M10LoRAError("M10 Agent LoRA Resume Run identity changed")
        run_id = str(run_payload["run_id"])
        mode = "exact_resume"
        progress = M10LoRAProgress(0, 0, 0, 0, None, None)
        source_stage_tokens = int(run_payload.get("last_evaluated_stage_tokens", 0))
        source_stage_adapter_sha256 = str(run_payload.get("stage_adapter_artifact_sha256", ""))
        _atomic_json(
            artifact_dir / "attempts" / f"hardware-target-{target_tokens:010d}.json",
            hardware,
        )

    store = M10LoRACheckpointStore(artifact_dir / "checkpoints")
    if mode == "exact_resume":
        latest = store.latest_valid()
        payload, progress = store.load_payload(
            latest,
            config=config,
            config_sha256=config_sha256,
            git_commit=git_commit,
            environment_sha256=environment_sha256,
            hardware_sha256=hardware_sha256,
            memory_probe_sha256=memory_probe_sha256,
            device=device,
        )
        set_peft_model_state_dict(model, cast(dict[str, Tensor], payload["adapter"]))
        optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
        resumed_from_tokens = progress.supervised_tokens
        expected_source = 1_000_000 if target_tokens == 5_000_000 else 5_000_000
        if (
            source_stage_tokens != expected_source
            or not source_stage_adapter_sha256
            or not expected_source <= progress.supervised_tokens < target_tokens
        ):
            raise M10LoRAError(
                "M10 Agent LoRA Resume point and evaluated source stage are incompatible"
            )
        if continuation_gate_path is None:
            raise M10LoRAError("M10 Agent LoRA Resume requires an accepted continuation Gate")
        gate, continuation_gate_sha256 = load_m10_lora_continuation_gate(
            continuation_gate_path,
            run_id=run_id,
            config_sha256=config_sha256,
            source_stage_tokens=source_stage_tokens,
            source_adapter_artifact_sha256=source_stage_adapter_sha256,
        )
        _atomic_json(
            artifact_dir
            / "continuation-gates"
            / f"{source_stage_tokens // 1_000_000}m-to-{target_tokens // 1_000_000}m.json",
            gate.to_dict(),
        )
        _append_jsonl(
            artifact_dir / "events.jsonl",
            {
                "event": "m10_lora_exact_resume_applied",
                "supervised_tokens": resumed_from_tokens,
                "continuation_gate_sha256": continuation_gate_sha256,
            },
        )

    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "git_dirty": False,
            "dataset_version": M10_DATASET_VERSION,
            "dataset_manifest_sha256": manifest_sha256,
            "parent_evaluation_subject": M10_LORA_PARENT_SUBJECT,
            "parent_evaluation_subject_sha256": M10_LORA_PARENT_RECORD_SHA256,
            "parent_model_artifact_sha256": M10_LORA_PARENT_MODEL_SHA256,
            "memory_probe_sha256": memory_probe_sha256,
            "physical_gpu_index": physical_gpu_index,
            "continuation_gate_sha256": continuation_gate_sha256,
            "last_evaluated_stage_tokens": source_stage_tokens,
            "stage_adapter_artifact_sha256": source_stage_adapter_sha256,
        },
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    initial_loss = progress.initial_loss
    final_loss = progress.final_loss
    latest_checkpoint = None
    micro_batch = config.optimization.micro_batch_size
    accumulation = config.optimization.gradient_accumulation_steps
    while progress.supervised_tokens < target_tokens:
        order = epoch_order(len(dataset), seed=config.run.seed, epoch=progress.completed_epochs)
        cursor = progress.sequence_cursor
        while cursor < len(order):
            group_end = min(cursor + micro_batch * accumulation, len(order))
            indices = order[cursor:group_end]
            learning_rate = token_learning_rate(
                base_learning_rate=config.optimization.learning_rate,
                tokens=progress.supervised_tokens,
                warmup_tokens=config.optimization.warmup_tokens,
                total_tokens=config.optimization.max_train_tokens,
            )
            step_started = time.monotonic()
            weighted_loss, gradient_norm, group_tokens = _optimizer_step(
                model=model,
                trainable=trainable,
                optimizer=optimizer,
                dataset=dataset,
                indices=indices,
                micro_batch=micro_batch,
                learning_rate=learning_rate,
                max_grad_norm=config.optimization.max_grad_norm,
                device=device,
            )
            initial_loss = weighted_loss if initial_loss is None else initial_loss
            final_loss = weighted_loss
            cursor += len(indices)
            progress = M10LoRAProgress(
                global_step=progress.global_step + 1,
                completed_epochs=progress.completed_epochs,
                sequence_cursor=cursor,
                supervised_tokens=progress.supervised_tokens + group_tokens,
                initial_loss=initial_loss,
                final_loss=final_loss,
            )
            duration = time.monotonic() - step_started
            _append_jsonl(
                artifact_dir / "metrics.jsonl",
                {
                    "global_step": progress.global_step,
                    "completed_epochs": progress.completed_epochs,
                    "sequence_cursor": progress.sequence_cursor,
                    "supervised_tokens": progress.supervised_tokens,
                    "loss": weighted_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "optimizer_step_duration_seconds": duration,
                    "supervised_tokens_per_second": group_tokens / duration,
                },
            )
        expected_tokens = (progress.completed_epochs + 1) * 1_000_000
        if progress.supervised_tokens != expected_tokens:
            raise M10LoRAError("M10 Agent LoRA epoch is not exactly 1M supervised Tokens")
        progress = M10LoRAProgress(
            global_step=progress.global_step,
            completed_epochs=progress.completed_epochs + 1,
            sequence_cursor=0,
            supervised_tokens=progress.supervised_tokens,
            initial_loss=initial_loss,
            final_loss=final_loss,
        )
        pin_reason: Literal["stage", "final"] | None
        if progress.supervised_tokens in M10_STAGE_TOKENS[:-1]:
            pin_reason = "stage"
        elif progress.supervised_tokens == M10_STAGE_TOKENS[-1]:
            pin_reason = "final"
        else:
            pin_reason = None
        latest_checkpoint = store.save(
            adapter_state=cast(dict[str, Tensor], get_peft_model_state_dict(model)),
            optimizer=optimizer,
            progress=progress,
            config=config,
            config_sha256=config_sha256,
            run_id=run_id,
            git_commit=git_commit,
            environment_sha256=environment_sha256,
            hardware_sha256=hardware_sha256,
            memory_probe_sha256=memory_probe_sha256,
            pin_reason=pin_reason,
        )
        if time.monotonic() - started > config.optimization.max_job_duration_seconds:
            raise M10LoRAError("M10 Agent LoRA exceeded the configured wall-clock limit")

    if initial_loss is None or final_loss is None or latest_checkpoint is None:
        raise M10LoRAError("M10 Agent LoRA stage ended without complete evidence")
    stage_export = export_m10_lora_stage(
        model, artifact_dir / "exports", latest_checkpoint.checkpoint_id
    )
    status: Literal["stage_completed", "succeeded"] = (
        "succeeded" if progress.supervised_tokens == 10_000_000 else "stage_completed"
    )
    result = M10LoRARunResult(
        status=status,
        mode=mode,
        run_id=run_id,
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
        parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
        parent_evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
        parent_model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
        attention_architecture="gqa",
        adaptation="lora",
        peft_version="0.19.1",
        seed=42,
        physical_gpu_index=physical_gpu_index,
        gpu_name=cast(Literal["NVIDIA GeForce RTX 3090"], torch.cuda.get_device_name(device)),
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        global_step=progress.global_step,
        completed_epochs=progress.completed_epochs,
        supervised_tokens=cast(
            Literal[1_000_000, 5_000_000, 10_000_000], progress.supervised_tokens
        ),
        initial_loss=initial_loss,
        final_loss=final_loss,
        duration_seconds=time.monotonic() - started,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        memory_probe_sha256=memory_probe_sha256,
        latest_checkpoint=latest_checkpoint.checkpoint_id,
        resumed_from_tokens=resumed_from_tokens,
        continuation_gate_sha256=continuation_gate_sha256,
        stage_export=stage_export,
    )
    record_m10_lora_result(artifact_dir, result)
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": status,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "git_dirty": False,
            "dataset_version": M10_DATASET_VERSION,
            "dataset_manifest_sha256": M10_DATASET_MANIFEST_SHA256,
            "parent_evaluation_subject": M10_LORA_PARENT_SUBJECT,
            "parent_evaluation_subject_sha256": M10_LORA_PARENT_RECORD_SHA256,
            "parent_model_artifact_sha256": M10_LORA_PARENT_MODEL_SHA256,
            "memory_probe_sha256": memory_probe_sha256,
            "physical_gpu_index": physical_gpu_index,
            "supervised_tokens": progress.supervised_tokens,
            "latest_checkpoint": latest_checkpoint.checkpoint_id,
            "stage_adapter_artifact_sha256": stage_export.adapter_artifact_sha256,
            "last_evaluated_stage_tokens": progress.supervised_tokens,
            "continuation_gate_sha256": continuation_gate_sha256,
        },
    )
    return result


__all__ = ["run_m10_lora", "run_m10_lora_memory_probe"]
