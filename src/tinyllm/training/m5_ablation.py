"""Native single-GPU Qwen3-0.6B Full-SFT loop for the bounded M5.2 ablation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from tinyllm.data import M5AblationDataset, open_m5_ablation_mixture
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m5_ablation_schema import (
    M5AblationRunResult,
    M5CheckpointFile,
    M5CheckpointManifest,
)
from tinyllm.training.m5_config import M5SFTConfig, load_m5_sft_config
from tinyllm.training.seed import seed_everything

_TRAINING_STATE = "training_state.pt"
_MANIFEST = "manifest.json"
_COMMITTED = "COMMITTED"
_LATEST = "LATEST"


class M5AblationError(RuntimeError):
    """Raised when M5.2 training or Exact Resume fails closed."""


@dataclass(frozen=True, slots=True)
class M5Progress:
    """Durable optimizer and data-order progress."""

    global_step: int
    sequence_cursor: int
    supervised_tokens: int
    initial_loss: float | None
    final_loss: float | None


def token_learning_rate(
    *, base_learning_rate: float, tokens: int, warmup_tokens: int, total_tokens: int
) -> float:
    """Return warmup/cosine LR indexed by durable supervised-token progress."""

    if not 0 <= tokens <= total_tokens or total_tokens <= 0 or warmup_tokens < 0:
        raise ValueError("invalid token-indexed scheduler inputs")
    if warmup_tokens > 0 and tokens < warmup_tokens:
        return base_learning_rate * max(tokens, 1) / warmup_tokens
    denominator = max(total_tokens - warmup_tokens, 1)
    progress = min(max((tokens - warmup_tokens) / denominator, 0.0), 1.0)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def group_loss_scale(valid_tokens: int, group_tokens: int) -> float:
    """Scale a mean micro-batch loss into one token-weighted accumulation group."""

    if valid_tokens <= 0 or group_tokens < valid_tokens:
        raise ValueError("invalid supervised-token accumulation counts")
    return valid_tokens / group_tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_attempt_result(artifact_dir: Path, result: M5AblationRunResult) -> None:
    """Preserve every interrupted or resumed attempt before updating the latest result."""

    attempt_name = f"{result.mode}-{result.status}-tokens-{result.supervised_tokens:010d}.json"
    _atomic_json(artifact_dir / "attempts" / attempt_name, result.to_dict())
    _atomic_json(artifact_dir / "result.json", result.to_dict())
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": f"m5_run_{result.status}",
            "mode": result.mode,
            "supervised_tokens": result.supervised_tokens,
            "attempt_result": f"attempts/{attempt_name}",
        },
    )


def _capture_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: dict[str, object]) -> None:
    if set(value) != {"python", "numpy", "torch", "cuda"}:
        raise M5AblationError("M5 Checkpoint RNG state is incomplete")
    random.setstate(cast(tuple[Any, ...], value["python"]))
    np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
    torch.set_rng_state(cast(Tensor, value["torch"]))
    torch.cuda.set_rng_state_all(cast(list[Tensor], value["cuda"]))


class M5CheckpointStore:
    """Atomic, SHA256-validated M5 single-GPU Checkpoints with rolling retention."""

    def __init__(self, root: Path, *, keep_last: int) -> None:
        if keep_last != 2:
            raise ValueError("M5 Checkpoint retention is fixed to two")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        progress: M5Progress,
        order: tuple[int, ...],
        config: M5SFTConfig,
        config_sha256: str,
        mixture_version: str,
        mixture_manifest_sha256: str,
        run_id: str,
        git_commit: str,
        pin_reason: Literal["interruption", "final"] | None,
    ) -> M5CheckpointManifest:
        checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / checkpoint_id
        if destination.exists():
            return self.validate(checkpoint_id)
        temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            state_path = temporary / _TRAINING_STATE
            torch.save(
                {
                    "schema_version": "1.0",
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "progress": asdict(progress),
                    "order": order,
                    "rng": _capture_rng(),
                    "config": config.to_dict(),
                    "config_sha256": config_sha256,
                    "mixture_version": mixture_version,
                    "mixture_manifest_sha256": mixture_manifest_sha256,
                    "run_id": run_id,
                    "git_commit": git_commit,
                },
                state_path,
            )
            file = M5CheckpointFile(
                path="training_state.pt",
                size_bytes=state_path.stat().st_size,
                sha256=_sha256_file(state_path),
            )
            manifest = M5CheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                global_step=progress.global_step,
                sequence_cursor=progress.sequence_cursor,
                supervised_tokens=progress.supervised_tokens,
                config_sha256=config_sha256,
                mixture_version=mixture_version,
                mixture_manifest_sha256=mixture_manifest_sha256,
                model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
                git_commit=git_commit,
                file=file,
                pinned=pin_reason is not None,
                pin_reason=pin_reason,
            )
            manifest_bytes = (
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode()
            (temporary / _MANIFEST).write_bytes(manifest_bytes)
            (temporary / _COMMITTED).write_text(
                json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}) + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, destination)
            latest_tmp = self.root / f".{_LATEST}.tmp-{uuid.uuid4().hex}"
            latest_tmp.write_text(checkpoint_id + "\n", encoding="utf-8")
            os.replace(latest_tmp, self.root / _LATEST)
            self.validate(checkpoint_id)
            self._retain()
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def validate(self, checkpoint_id: str) -> M5CheckpointManifest:
        directory = self.root / checkpoint_id
        try:
            manifest_bytes = (directory / _MANIFEST).read_bytes()
            manifest = M5CheckpointManifest.model_validate_json(manifest_bytes)
            marker = cast(
                dict[str, str],
                json.loads((directory / _COMMITTED).read_text(encoding="utf-8")),
            )
        except Exception as exc:
            raise M5AblationError("M5 Checkpoint metadata is incomplete or corrupt") from exc
        if marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}:
            raise M5AblationError("M5 Checkpoint commit marker is invalid")
        state_path = directory / manifest.file.path
        if (
            manifest.checkpoint_id != checkpoint_id
            or not state_path.is_file()
            or state_path.stat().st_size != manifest.file.size_bytes
            or _sha256_file(state_path) != manifest.file.sha256
        ):
            raise M5AblationError("M5 Checkpoint payload failed integrity validation")
        return manifest

    def latest_valid(self) -> M5CheckpointManifest:
        candidates = sorted(
            (path.name for path in self.root.glob("checkpoint-tokens-*") if path.is_dir()),
            reverse=True,
        )
        for checkpoint_id in candidates:
            try:
                return self.validate(checkpoint_id)
            except M5AblationError:
                continue
        raise M5AblationError("M5 Checkpoint store contains no valid Exact Resume point")

    def load(
        self,
        manifest: M5CheckpointManifest,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        config: M5SFTConfig,
        config_sha256: str,
        mixture_version: str,
        mixture_manifest_sha256: str,
        git_commit: str,
        device: torch.device,
    ) -> tuple[M5Progress, tuple[int, ...]]:
        if (
            manifest.config_sha256 != config_sha256
            or manifest.mixture_version != mixture_version
            or manifest.mixture_manifest_sha256 != mixture_manifest_sha256
            or manifest.model_revision != config.model.revision
            or manifest.git_commit != git_commit
        ):
            raise M5AblationError("M5 Exact Resume lineage or configuration changed")
        try:
            payload = cast(
                dict[str, Any],
                torch.load(
                    self.root / manifest.checkpoint_id / _TRAINING_STATE,
                    map_location=device,
                    weights_only=False,
                ),
            )
            if payload["config"] != config.to_dict() or payload["config_sha256"] != config_sha256:
                raise M5AblationError("M5 Checkpoint config payload changed")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            progress_value = cast(dict[str, Any], payload["progress"])
            initial_value = progress_value["initial_loss"]
            final_value = progress_value["final_loss"]
            progress = M5Progress(
                global_step=int(progress_value["global_step"]),
                sequence_cursor=int(progress_value["sequence_cursor"]),
                supervised_tokens=int(progress_value["supervised_tokens"]),
                initial_loss=None if initial_value is None else float(initial_value),
                final_loss=None if final_value is None else float(final_value),
            )
            order = tuple(int(value) for value in cast(tuple[int, ...], payload["order"]))
            _restore_rng(cast(dict[str, object], payload["rng"]))
        except M5AblationError:
            raise
        except Exception as exc:
            raise M5AblationError("M5 Checkpoint training state cannot be restored") from exc
        if (
            progress.global_step != manifest.global_step
            or progress.sequence_cursor != manifest.sequence_cursor
            or progress.supervised_tokens != manifest.supervised_tokens
        ):
            raise M5AblationError("M5 Checkpoint progress differs from manifest")
        return progress, order

    def _retain(self) -> None:
        manifests: list[M5CheckpointManifest] = []
        for path in self.root.glob("checkpoint-tokens-*"):
            if path.is_dir():
                try:
                    manifests.append(self.validate(path.name))
                except M5AblationError:
                    continue
        unpinned = sorted(
            (item for item in manifests if not item.pinned),
            key=lambda item: item.supervised_tokens,
            reverse=True,
        )
        for item in unpinned[self.keep_last :]:
            shutil.rmtree(self.root / item.checkpoint_id)


def _build_order(length: int, seed: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return tuple(int(value) for value in torch.randperm(length, generator=generator).tolist())


def _batch(dataset: M5AblationDataset, indices: tuple[int, ...]) -> dict[str, Tensor]:
    examples = tuple(dataset[index] for index in indices)
    return {key: torch.stack([item[key] for item in examples]) for key in examples[0]}


def _export_model(model: nn.Module, export_root: Path) -> str:
    export_root.mkdir(parents=True, exist_ok=False)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise M5AblationError("Qwen model does not expose save_pretrained")
    save_pretrained(export_root, safe_serialization=True)
    digest = hashlib.sha256()
    for path in sorted(export_root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise M5AblationError("M5 export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _validate_runtime_identity(config: M5SFTConfig, model_dir: Path) -> None:
    try:
        decoded = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5AblationError("pinned Qwen3-0.6B config cannot be parsed") from exc
    expected = {
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
    }
    if (
        model_dir.name != config.model.revision
        or {key: decoded.get(key) for key in expected} != expected
    ):
        raise M5AblationError("local Qwen3-0.6B GQA identity differs from M5 config")


def run_m5_ablation(
    *,
    config_path: Path,
    mixture_root: Path,
    model_dir: Path,
    output_root: Path,
    physical_gpu_index: int,
    resume_run: Path | None = None,
    stop_after_tokens: int | None = None,
) -> M5AblationRunResult:
    """Train or Exact-Resume one bounded M5.2 arm on one visible RTX 3090."""

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5AblationError("M5.2 worker requires exactly one visible CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise M5AblationError("M5.2 worker requires BF16 support")
    config = load_m5_sft_config(config_path)
    if config.run.purpose != "ablation":
        raise M5AblationError("M5.2 worker accepts only ablation configs")
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5AblationError("formal M5.2 training requires a clean Git worktree")
    _validate_runtime_identity(config, model_dir)
    opened = open_m5_ablation_mixture(mixture_root)
    manifest_bytes = (mixture_root / "manifest.json").read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        opened.manifest.pilot_dataset_version != config.data.dataset_version
        or manifest_sha256 != config.data.mix_manifest_sha256
        or opened.manifest.thinking_fraction_basis_points
        != int(config.data.thinking_token_fraction * 10_000)
    ):
        raise M5AblationError("M5 config and mixture identity or ratio differ")
    if stop_after_tokens is not None and not 0 < stop_after_tokens < 1_000_000:
        raise M5AblationError("coordinated stop must be inside the 1M-token budget")

    config_sha256 = canonical_config_hash(config)
    device = torch.device("cuda", 0)
    seed_everything(config.run.seed, deterministic_algorithms=False)
    dataset = M5AblationDataset(mixture_root)
    order = _build_order(len(dataset), config.run.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
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
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    mode: Literal["fresh", "exact_resume"] = "fresh"
    resumed_from_tokens: int | None = None
    if resume_run is None:
        run_id = generate_run_id(config.run.name, config_sha256, now=datetime.now(UTC))
        artifact_dir = output_root / run_id
        artifact_dir.mkdir(parents=True)
        _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
        shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
        _atomic_json(artifact_dir / "mixture.json", opened.manifest.to_dict())
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
        progress = M5Progress(0, 0, 0, None, None)
    else:
        artifact_dir = resume_run
        run_payload = json.loads((artifact_dir / "run.json").read_text(encoding="utf-8"))
        if run_payload.get("config_sha256") != config_sha256:
            raise M5AblationError("M5 Resume Run config identity changed")
        run_id = str(run_payload["run_id"])
        mode = "exact_resume"
        progress = M5Progress(0, 0, 0, None, None)
    store = M5CheckpointStore(artifact_dir / "checkpoints", keep_last=2)
    if mode == "exact_resume":
        latest = store.latest_valid()
        progress, order = store.load(
            latest,
            model=model,
            optimizer=optimizer,
            config=config,
            config_sha256=config_sha256,
            mixture_version=opened.manifest.mixture_version,
            mixture_manifest_sha256=manifest_sha256,
            git_commit=git_commit,
            device=device,
        )
        resumed_from_tokens = progress.supervised_tokens
        _append_jsonl(
            artifact_dir / "events.jsonl",
            {"event": "m5_exact_resume_applied", "supervised_tokens": resumed_from_tokens},
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
            "mixture_version": opened.manifest.mixture_version,
            "mixture_manifest_sha256": manifest_sha256,
            "physical_gpu_index": physical_gpu_index,
        },
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    next_save = (
        (progress.supervised_tokens // config.checkpoint.save_interval_tokens) + 1
    ) * config.checkpoint.save_interval_tokens
    initial_loss = progress.initial_loss
    final_loss = progress.final_loss
    latest_checkpoint: str | None = None
    micro_batch = config.training.micro_batch_size
    accumulation = config.training.gradient_accumulation_steps
    while progress.sequence_cursor < len(order):
        group_start = progress.sequence_cursor
        group_end = min(group_start + micro_batch * accumulation, len(order))
        group_indices = order[group_start:group_end]
        group_examples = tuple(dataset[index] for index in group_indices)
        group_tokens = sum(int(item["labels"][1:].ne(-100).sum()) for item in group_examples)
        if group_tokens <= 0:
            raise M5AblationError("M5 accumulation group contains no supervised tokens")
        learning_rate = token_learning_rate(
            base_learning_rate=config.training.learning_rate,
            tokens=progress.supervised_tokens,
            warmup_tokens=config.training.warmup_tokens,
            total_tokens=config.training.max_train_tokens,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        consumed_sequences = 0
        for offset in range(0, len(group_indices), micro_batch):
            indices = group_indices[offset : offset + micro_batch]
            batch = {key: value.to(device) for key, value in _batch(dataset, indices).items()}
            valid_tokens = int(batch["labels"][:, 1:].ne(-100).sum())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
                loss = getattr(output, "loss", None)
            if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
                raise M5AblationError("M5 training produced a non-finite scalar loss")
            scale = group_loss_scale(valid_tokens, group_tokens)
            torch.autograd.backward(loss * scale)
            weighted_loss += float(loss.detach().float()) * scale
            consumed_sequences += len(indices)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        )
        if not math.isfinite(gradient_norm):
            raise M5AblationError("M5 training produced a non-finite gradient norm")
        optimizer.step()
        if initial_loss is None:
            initial_loss = weighted_loss
        final_loss = weighted_loss
        progress = M5Progress(
            global_step=progress.global_step + 1,
            sequence_cursor=progress.sequence_cursor + consumed_sequences,
            supervised_tokens=progress.supervised_tokens + group_tokens,
            initial_loss=initial_loss,
            final_loss=final_loss,
        )
        _append_jsonl(
            artifact_dir / "metrics.jsonl",
            {
                "global_step": progress.global_step,
                "supervised_tokens": progress.supervised_tokens,
                "loss": weighted_loss,
                "learning_rate": learning_rate,
                "gradient_norm": gradient_norm,
            },
        )
        coordinated_stop = (
            stop_after_tokens is not None and progress.supervised_tokens >= stop_after_tokens
        )
        final = progress.sequence_cursor == len(order)
        if progress.supervised_tokens >= next_save or coordinated_stop or final:
            checkpoint = store.save(
                model=model,
                optimizer=optimizer,
                progress=progress,
                order=order,
                config=config,
                config_sha256=config_sha256,
                mixture_version=opened.manifest.mixture_version,
                mixture_manifest_sha256=manifest_sha256,
                run_id=run_id,
                git_commit=git_commit,
                pin_reason="interruption" if coordinated_stop else "final" if final else None,
            )
            latest_checkpoint = checkpoint.checkpoint_id
            while next_save <= progress.supervised_tokens:
                next_save += config.checkpoint.save_interval_tokens
        if coordinated_stop:
            break
        if time.monotonic() - started > config.training.max_job_duration_seconds:
            raise M5AblationError("M5 training exceeded the configured wall-clock limit")

    if initial_loss is None or final_loss is None or latest_checkpoint is None:
        raise M5AblationError("M5 training ended without loss or Checkpoint evidence")
    status: Literal["succeeded", "interrupted"] = (
        "succeeded" if progress.supervised_tokens == 1_000_000 else "interrupted"
    )
    export_sha256: str | None = None
    if status == "succeeded":
        export_sha256 = _export_model(model, artifact_dir / "exports" / "model")
    duration = time.monotonic() - started
    result = M5AblationRunResult(
        status=status,
        mode=mode,
        run_id=run_id,
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=False,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture=config.model.attention_architecture,
        mixture_version=opened.manifest.mixture_version,
        mixture_manifest_sha256=manifest_sha256,
        thinking_fraction_basis_points=opened.manifest.thinking_fraction_basis_points,
        seed=config.run.seed,
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        global_step=progress.global_step,
        supervised_tokens=progress.supervised_tokens,
        sequence_cursor=progress.sequence_cursor,
        initial_loss=initial_loss,
        final_loss=final_loss,
        duration_seconds=duration,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        latest_checkpoint=latest_checkpoint,
        resumed_from_tokens=resumed_from_tokens,
        export_sha256=export_sha256,
    )
    _record_attempt_result(artifact_dir, result)
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": status,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "git_dirty": False,
            "mixture_version": opened.manifest.mixture_version,
            "mixture_manifest_sha256": manifest_sha256,
            "physical_gpu_index": physical_gpu_index,
            "supervised_tokens": progress.supervised_tokens,
            "latest_checkpoint": latest_checkpoint,
        },
    )
    return result
