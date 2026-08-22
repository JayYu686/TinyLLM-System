"""CUDA worker for staged M10 Qwen3-0.6B Agent Full SFT."""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor
from torch.optim import AdamW

from tinyllm.data.m10_mixture import M10FrozenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m5_ablation import group_loss_scale, token_learning_rate
from tinyllm.training.m10_sft import (
    M10CheckpointStore,
    M10FullSFTError,
    M10Progress,
    _append_jsonl,
    _atomic_json,
    _batch,
    _export_stage,
    _record_result,
    epoch_order,
    load_m10_continuation_gate,
    preflight_m10_full_sft,
)
from tinyllm.training.m10_sft_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_PARENT_MODEL_SHA256,
    M10_PARENT_RECORD_SHA256,
    M10_PARENT_VERSION,
    M10_STAGE_TOKENS,
    M10CheckpointManifest,
    M10FullSFTRunResult,
)
from tinyllm.training.seed import seed_everything


def run_m10_full_sft(
    *,
    config_path: Path,
    mixture_root: Path,
    artifact_root: Path,
    output_root: Path,
    physical_gpu_index: int,
    resume_run: Path | None = None,
    stop_after_tokens: int | None = None,
    continuation_gate_path: Path | None = None,
) -> M10FullSFTRunResult:
    """Train or Exact-Resume one M10 stage on exactly one visible RTX 3090."""

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M10FullSFTError("M10.2 worker requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise M10FullSFTError("M10.2 worker requires BF16 support")
    config, resolved, manifest_sha256 = preflight_m10_full_sft(
        config_path=config_path,
        mixture_root=mixture_root,
        artifact_root=artifact_root,
    )
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M10FullSFTError("formal M10.2 training requires a clean Git worktree")
    target_tokens = (
        config.optimization.max_train_tokens if stop_after_tokens is None else stop_after_tokens
    )
    if target_tokens not in M10_STAGE_TOKENS:
        raise M10FullSFTError("M10 coordinated stop must be exactly 1M, 5M, or 10M Tokens")
    if resume_run is None and target_tokens != 1_000_000:
        raise M10FullSFTError("fresh M10 training must stop at the 1M evaluation stage")
    if resume_run is None and continuation_gate_path is not None:
        raise M10FullSFTError("fresh M10 training cannot consume a continuation gate")

    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    seed_everything(config.run.seed, deterministic_algorithms=False)
    dataset = M10FrozenDataset(mixture_root)
    model = AutoModelForCausalLM.from_pretrained(
        resolved.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    config_sha256 = canonical_config_hash(config)
    mode: Literal["fresh", "exact_resume"] = "fresh"
    resumed_from_tokens: int | None = None
    continuation_gate_sha256: str | None = None
    if resume_run is None:
        run_id = generate_run_id(config.run.name, config_sha256, now=datetime.now(UTC))
        artifact_dir = output_root / run_id
        artifact_dir.mkdir(parents=True)
        shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
        _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
        _atomic_json(artifact_dir / "dataset.json", dataset.manifest.to_dict())
        _atomic_json(artifact_dir / "parent.json", resolved.to_dict())
        _atomic_json(
            artifact_dir / "environment.json",
            {
                "schema_version": "1.0",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "physical_gpu_index": physical_gpu_index,
                "gpu_name": torch.cuda.get_device_name(device),
            },
        )
        progress = M10Progress(0, 0, 0, 0, None, None)
    else:
        artifact_dir = resume_run
        try:
            run_payload = json.loads((artifact_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise M10FullSFTError("M10 Resume Run metadata is missing or invalid") from exc
        if run_payload.get("config_sha256") != config_sha256:
            raise M10FullSFTError("M10 Resume Run config identity changed")
        run_id = str(run_payload["run_id"])
        mode = "exact_resume"
        progress = M10Progress(0, 0, 0, 0, None, None)
    store = M10CheckpointStore(artifact_dir / "checkpoints")
    if mode == "exact_resume":
        latest = store.latest_valid()
        progress = store.load(
            latest,
            model=model,
            optimizer=optimizer,
            config=config,
            config_sha256=config_sha256,
            git_commit=git_commit,
            device=device,
        )
        resumed_from_tokens = progress.supervised_tokens
        expected_transition = {1_000_000: 5_000_000, 5_000_000: 10_000_000}
        if expected_transition.get(progress.supervised_tokens) != target_tokens:
            raise M10FullSFTError("M10 Resume must follow the frozen 1M to 5M to 10M stages")
        if progress.supervised_tokens == 5_000_000:
            if continuation_gate_path is None:
                raise M10FullSFTError("M10 5M-to-10M Resume requires an accepted continuation gate")
            gate, continuation_gate_sha256 = load_m10_continuation_gate(
                continuation_gate_path,
                run_id=run_id,
                config_sha256=config_sha256,
                source_stage_export_sha256=str(run_payload.get("stage_export_sha256", "")),
            )
            _atomic_json(artifact_dir / "continuation-gates" / "5m-to-10m.json", gate.to_dict())
        elif continuation_gate_path is not None:
            raise M10FullSFTError("M10 1M-to-5M Resume does not accept a continuation gate")
        _append_jsonl(
            artifact_dir / "events.jsonl",
            {
                "event": "m10_exact_resume_applied",
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
            "dataset_version": config.data.dataset_version,
            "dataset_manifest_sha256": manifest_sha256,
            "parent_production_version": resolved.model_version,
            "parent_production_record_sha256": resolved.production_record_sha256,
            "parent_model_artifact_sha256": resolved.model_artifact_sha256,
            "physical_gpu_index": physical_gpu_index,
            "continuation_gate_sha256": continuation_gate_sha256,
        },
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    initial_loss = progress.initial_loss
    final_loss = progress.final_loss
    latest_checkpoint: M10CheckpointManifest | None = None
    micro_batch = config.optimization.micro_batch_size
    accumulation = config.optimization.gradient_accumulation_steps
    while progress.supervised_tokens < target_tokens:
        order = epoch_order(len(dataset), seed=config.run.seed, epoch=progress.completed_epochs)
        cursor = progress.sequence_cursor
        while cursor < len(order):
            group_end = min(cursor + micro_batch * accumulation, len(order))
            indices = order[cursor:group_end]
            examples = tuple(dataset[index] for index in indices)
            group_tokens = sum(int(item["labels"][1:].ne(-100).sum()) for item in examples)
            if group_tokens <= 0:
                raise M10FullSFTError("M10 accumulation group contains no supervised tokens")
            learning_rate = token_learning_rate(
                base_learning_rate=config.optimization.learning_rate,
                tokens=progress.supervised_tokens,
                warmup_tokens=config.optimization.warmup_tokens,
                total_tokens=config.optimization.max_train_tokens,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            consumed = 0
            for offset in range(0, len(indices), micro_batch):
                micro_indices = indices[offset : offset + micro_batch]
                batch = {
                    key: value.to(device) for key, value in _batch(dataset, micro_indices).items()
                }
                valid_tokens = int(batch["labels"][:, 1:].ne(-100).sum())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**batch, use_cache=False)
                    loss = getattr(output, "loss", None)
                if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
                    raise M10FullSFTError("M10 training produced a non-finite scalar loss")
                scale = group_loss_scale(valid_tokens, group_tokens)
                torch.autograd.backward(loss * scale)
                weighted_loss += float(loss.detach().float()) * scale
                consumed += len(micro_indices)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimization.max_grad_norm
                )
            )
            if not math.isfinite(gradient_norm):
                raise M10FullSFTError("M10 training produced a non-finite gradient norm")
            optimizer.step()
            initial_loss = weighted_loss if initial_loss is None else initial_loss
            final_loss = weighted_loss
            cursor += consumed
            progress = M10Progress(
                global_step=progress.global_step + 1,
                completed_epochs=progress.completed_epochs,
                sequence_cursor=cursor,
                supervised_tokens=progress.supervised_tokens + group_tokens,
                initial_loss=initial_loss,
                final_loss=final_loss,
            )
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
                },
            )
        completed_tokens = progress.supervised_tokens
        expected_tokens = (progress.completed_epochs + 1) * 1_000_000
        if completed_tokens != expected_tokens:
            raise M10FullSFTError("M10 logical epoch does not contain exactly 1M supervised Tokens")
        progress = M10Progress(
            global_step=progress.global_step,
            completed_epochs=progress.completed_epochs + 1,
            sequence_cursor=0,
            supervised_tokens=completed_tokens,
            initial_loss=initial_loss,
            final_loss=final_loss,
        )
        pin_reason: Literal["stage", "final"] | None = None
        if completed_tokens in M10_STAGE_TOKENS[:-1]:
            pin_reason = "stage"
        elif completed_tokens == M10_STAGE_TOKENS[-1]:
            pin_reason = "final"
        latest_checkpoint = store.save(
            model=model,
            optimizer=optimizer,
            progress=progress,
            config=config,
            config_sha256=config_sha256,
            run_id=run_id,
            git_commit=git_commit,
            pin_reason=pin_reason,
        )
        if time.monotonic() - started > config.optimization.max_job_duration_seconds:
            raise M10FullSFTError("M10 training exceeded the configured wall-clock limit")

    if initial_loss is None or final_loss is None or latest_checkpoint is None:
        raise M10FullSFTError("M10 stage ended without complete training evidence")
    stage_export = _export_stage(model, artifact_dir / "exports", latest_checkpoint.checkpoint_id)
    status: Literal["stage_completed", "succeeded"] = (
        "succeeded" if progress.supervised_tokens == 10_000_000 else "stage_completed"
    )
    result = M10FullSFTRunResult(
        status=status,
        mode=mode,
        run_id=run_id,
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
        parent_production_version=M10_PARENT_VERSION,
        parent_production_record_sha256=M10_PARENT_RECORD_SHA256,
        parent_model_artifact_sha256=M10_PARENT_MODEL_SHA256,
        attention_architecture="gqa",
        seed=42,
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
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
        latest_checkpoint=latest_checkpoint.checkpoint_id,
        resumed_from_tokens=resumed_from_tokens,
        continuation_gate_sha256=continuation_gate_sha256,
        stage_export=stage_export,
    )
    _record_result(artifact_dir, result)
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
            "parent_production_version": M10_PARENT_VERSION,
            "parent_production_record_sha256": M10_PARENT_RECORD_SHA256,
            "parent_model_artifact_sha256": M10_PARENT_MODEL_SHA256,
            "physical_gpu_index": physical_gpu_index,
            "supervised_tokens": progress.supervised_tokens,
            "latest_checkpoint": latest_checkpoint.checkpoint_id,
            "stage_export_sha256": stage_export.export_sha256,
            "continuation_gate_sha256": continuation_gate_sha256,
        },
    )
    return result
