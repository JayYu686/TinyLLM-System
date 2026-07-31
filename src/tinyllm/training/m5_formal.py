"""Native four-GPU Qwen3-0.6B Full SFT with exact DDP recovery."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer

from tinyllm.data import M5FormalDataset, open_m5_formal_dataset
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m5_ablation import token_learning_rate
from tinyllm.training.m5_config import M5SFTConfig, load_m5_sft_config
from tinyllm.training.m5_formal_schema import (
    M5FormalCheckpointFile,
    M5FormalCheckpointManifest,
    M5FormalRankMemory,
    M5FormalRunResult,
)
from tinyllm.training.seed import seed_everything

_STATE_FILE = "training_state.pt"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED"
_LATEST_FILE = "LATEST"


class M5FormalTrainingError(RuntimeError):
    """Raised when formal M5 training or Exact Resume fails closed."""


@dataclass(frozen=True, slots=True)
class M5FormalProgress:
    """Durable shared optimizer and local sampler progress."""

    global_step: int
    local_sequence_cursor: int
    supervised_tokens: int
    initial_loss: float | None
    final_loss: float | None
    evaluation_checkpoints: tuple[str, ...]


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


def _capture_rng(device: torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu(),
    }


def _restore_rng(value: object, *, device: torch.device) -> None:
    if not isinstance(value, dict) or set(value) != {"python", "numpy", "torch", "cuda"}:
        raise M5FormalTrainingError("formal M5 Rank RNG state is incomplete")
    try:
        random.setstate(cast(tuple[Any, ...], value["python"]))
        np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
        torch.set_rng_state(cast(Tensor, value["torch"]).cpu())
        torch.cuda.set_rng_state(cast(Tensor, value["cuda"]).cpu(), device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M5FormalTrainingError("formal M5 Rank RNG state cannot be restored") from exc


class M5FormalCheckpointStore:
    """Atomic full-state Checkpoints with per-Rank RNG and rolling retention."""

    def __init__(self, root: Path, *, keep_last: int) -> None:
        if keep_last != 2:
            raise ValueError("formal M5 Checkpoint retention is fixed to two")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        progress: M5FormalProgress,
        rank_rng: tuple[object, object, object, object],
        config: M5SFTConfig,
        config_sha256: str,
        dataset_manifest_sha256: str,
        run_id: str,
        git_commit: str,
        pin_reason: Literal["interruption", "evaluation", "final"] | None,
    ) -> M5FormalCheckpointManifest:
        """Publish one complete optimizer-boundary state from Rank zero."""

        checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / checkpoint_id
        if destination.exists():
            raise M5FormalTrainingError("formal M5 Checkpoint destination already exists")
        temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            state_path = temporary / _STATE_FILE
            torch.save(
                {
                    "schema_version": "1.0",
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": {
                        "kind": "token_warmup_cosine",
                        "tokens": progress.supervised_tokens,
                        "warmup_tokens": config.training.warmup_tokens,
                        "total_tokens": config.training.max_train_tokens,
                    },
                    "progress": asdict(progress),
                    "rank_rng": rank_rng,
                    "config": config.to_dict(),
                    "config_sha256": config_sha256,
                    "dataset_version": config.data.dataset_version,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "world_size": 4,
                },
                state_path,
            )
            with state_path.open("rb") as handle:
                os.fsync(handle.fileno())
            file = M5FormalCheckpointFile(
                path="training_state.pt",
                size_bytes=state_path.stat().st_size,
                sha256=_sha256_file(state_path),
            )
            manifest = M5FormalCheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                global_step=progress.global_step,
                local_sequence_cursor=progress.local_sequence_cursor,
                supervised_tokens=progress.supervised_tokens,
                config_sha256=config_sha256,
                dataset_version=cast(
                    Literal["m5-dual-sft-v1-b5b9e839"],
                    config.data.dataset_version,
                ),
                dataset_manifest_sha256=cast(
                    Literal["607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"],
                    dataset_manifest_sha256,
                ),
                model_revision=cast(
                    Literal["c1899de289a04d12100db370d81485cdf75e47ca"],
                    config.model.revision,
                ),
                git_commit=git_commit,
                file=file,
                pinned=pin_reason is not None,
                pin_reason=pin_reason,
            )
            manifest_bytes = (
                json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode()
            (temporary / _MANIFEST_FILE).write_bytes(manifest_bytes)
            (temporary / _COMMIT_FILE).write_text(
                json.dumps(
                    {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, destination)
            self.validate(checkpoint_id)
            latest = self.root / f".{_LATEST_FILE}.tmp-{uuid.uuid4().hex}"
            latest.write_text(checkpoint_id + "\n", encoding="utf-8")
            os.replace(latest, self.root / _LATEST_FILE)
            self._retain()
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def validate(self, checkpoint_id: str) -> M5FormalCheckpointManifest:
        """Validate manifest, commit marker, state size, and SHA256."""

        directory = self.root / checkpoint_id
        try:
            manifest_bytes = (directory / _MANIFEST_FILE).read_bytes()
            manifest = M5FormalCheckpointManifest.model_validate_json(manifest_bytes)
            marker = cast(
                dict[str, str],
                json.loads((directory / _COMMIT_FILE).read_text(encoding="utf-8")),
            )
        except Exception as exc:
            raise M5FormalTrainingError(
                "formal M5 Checkpoint metadata is incomplete or corrupt"
            ) from exc
        state_path = directory / manifest.file.path
        if (
            marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
            or manifest.checkpoint_id != checkpoint_id
            or not state_path.is_file()
            or state_path.stat().st_size != manifest.file.size_bytes
            or _sha256_file(state_path) != manifest.file.sha256
        ):
            raise M5FormalTrainingError("formal M5 Checkpoint integrity validation failed")
        return manifest

    def latest_valid(self) -> M5FormalCheckpointManifest:
        """Select the newest complete Checkpoint, skipping corrupt newer directories."""

        candidates = sorted(
            (path.name for path in self.root.glob("checkpoint-tokens-*") if path.is_dir()),
            reverse=True,
        )
        for checkpoint_id in candidates:
            try:
                return self.validate(checkpoint_id)
            except M5FormalTrainingError:
                continue
        raise M5FormalTrainingError("formal M5 Run has no valid Exact Resume point")

    def load_payload(
        self,
        manifest: M5FormalCheckpointManifest,
        *,
        map_location: torch.device,
    ) -> dict[str, Any]:
        """Load a previously Rank-zero-validated payload onto one Rank."""

        try:
            payload = torch.load(
                self.root / manifest.checkpoint_id / _STATE_FILE,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise M5FormalTrainingError("formal M5 training state cannot be loaded") from exc
        if not isinstance(payload, dict):
            raise M5FormalTrainingError("formal M5 training state is not an object")
        return cast(dict[str, Any], payload)

    def _retain(self) -> None:
        manifests: list[M5FormalCheckpointManifest] = []
        for path in self.root.glob("checkpoint-tokens-*"):
            if not path.is_dir():
                continue
            try:
                manifests.append(
                    M5FormalCheckpointManifest.model_validate_json(
                        (path / _MANIFEST_FILE).read_bytes()
                    )
                )
            except Exception:
                continue
        unpinned = sorted(
            (item for item in manifests if not item.pinned),
            key=lambda item: item.supervised_tokens,
            reverse=True,
        )
        for item in unpinned[self.keep_last :]:
            shutil.rmtree(self.root / item.checkpoint_id)


def _validate_model_identity(config: M5SFTConfig, model_dir: Path) -> None:
    try:
        decoded = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5FormalTrainingError("formal Qwen3-0.6B config cannot be parsed") from exc
    if {
        "model_type": decoded.get("model_type"),
        "num_attention_heads": decoded.get("num_attention_heads"),
        "num_key_value_heads": decoded.get("num_key_value_heads"),
    } != {"model_type": "qwen3", "num_attention_heads": 16, "num_key_value_heads": 8}:
        raise M5FormalTrainingError("formal model is not the frozen Qwen3-0.6B GQA route")
    if model_dir.name != config.model.revision:
        raise M5FormalTrainingError("formal model path does not match the pinned Revision")


def _export_model(model: nn.Module, root: Path) -> str:
    root.mkdir(parents=True, exist_ok=False)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise M5FormalTrainingError("formal Qwen model cannot export Safetensors")
    save_pretrained(root, safe_serialization=True)
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise M5FormalTrainingError("formal model export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _record_result(artifact_dir: Path, result: M5FormalRunResult) -> None:
    attempt = f"{result.mode}-{result.status}-tokens-{result.supervised_tokens:010d}.json"
    _atomic_json(artifact_dir / "attempts" / attempt, result.to_dict())
    _atomic_json(artifact_dir / "result.json", result.to_dict())
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": f"m5_formal_{result.status}",
            "mode": result.mode,
            "supervised_tokens": result.supervised_tokens,
            "attempt_result": f"attempts/{attempt}",
        },
    )


def _load_progress(payload: dict[str, Any]) -> M5FormalProgress:
    try:
        raw = cast(dict[str, object], payload["progress"])
        return M5FormalProgress(
            global_step=int(cast(int, raw["global_step"])),
            local_sequence_cursor=int(cast(int, raw["local_sequence_cursor"])),
            supervised_tokens=int(cast(int, raw["supervised_tokens"])),
            initial_loss=(
                None if raw["initial_loss"] is None else float(cast(float, raw["initial_loss"]))
            ),
            final_loss=(
                None if raw["final_loss"] is None else float(cast(float, raw["final_loss"]))
            ),
            evaluation_checkpoints=tuple(
                str(value) for value in cast(tuple[str, ...], raw["evaluation_checkpoints"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M5FormalTrainingError("formal M5 Checkpoint progress is invalid") from exc


def run_m5_formal_ddp(
    *,
    config_path: Path,
    dataset_root: Path,
    model_dir: Path,
    output_root: Path,
    physical_gpu_indices: tuple[int, int, int, int],
    resume_run: Path | None = None,
    stop_after_tokens: int | None = None,
) -> M5FormalRunResult | None:
    """Train or Exact-Resume the formal 50M-token four-GPU Full-SFT Run."""

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if world_size != 4 or rank not in range(4) or local_rank not in range(4):
        raise M5FormalTrainingError("formal M5 worker requires torchrun World Size 4")
    config = load_m5_sft_config(config_path)
    if (
        config.run.purpose != "formal"
        or config.model.adaptation != "full_sft"
        or config.parallel.strategy != "ddp"
    ):
        raise M5FormalTrainingError("formal M5 worker received the wrong training route")
    if stop_after_tokens is not None and not 0 < stop_after_tokens < 50_000_000:
        raise M5FormalTrainingError("formal coordinated stop must be inside the 50M budget")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise M5FormalTrainingError("formal M5 Full SFT requires BF16 support")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=config.parallel.timeout_seconds),
    )
    artifact_dir: Path | None = None
    try:
        project_root = Path(__file__).resolve().parents[3]
        git_commit, git_dirty = read_git_identity(project_root)
        if git_dirty:
            raise M5FormalTrainingError("formal M5 training requires a clean Git worktree")
        _validate_model_identity(config, model_dir)
        opened = open_m5_formal_dataset(dataset_root)
        dataset_manifest_sha256 = hashlib.sha256(
            (dataset_root / _MANIFEST_FILE).read_bytes()
        ).hexdigest()
        if (
            opened.manifest.dataset_version != config.data.dataset_version
            or dataset_manifest_sha256 != config.data.mix_manifest_sha256
            or opened.manifest.target_supervised_tokens != config.training.max_train_tokens
        ):
            raise M5FormalTrainingError("formal M5 config and Dataset identity differ")
        config_sha256 = canonical_config_hash(config)
        run_id_box: list[object] = [
            (
                generate_run_id(config.run.name, config_sha256, now=datetime.now(UTC))
                if rank == 0 and resume_run is None
                else (
                    json.loads((resume_run / "run.json").read_text(encoding="utf-8"))["run_id"]
                    if rank == 0 and resume_run is not None
                    else None
                )
            )
        ]
        dist.broadcast_object_list(run_id_box, src=0)
        run_id = str(run_id_box[0])
        if resume_run is None:
            artifact_dir = output_root / run_id
            if rank == 0:
                artifact_dir.mkdir(parents=True, exist_ok=False)
                for name in ("checkpoints", "evaluations", "exports", "attempts"):
                    (artifact_dir / name).mkdir()
                shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
                _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
                _atomic_json(artifact_dir / "dataset.json", opened.manifest.to_dict())
                _atomic_json(
                    artifact_dir / "environment.json",
                    {
                        "schema_version": "1.0",
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "world_size": 4,
                        "physical_gpu_indices": physical_gpu_indices,
                    },
                )
                _atomic_json(
                    artifact_dir / "run.json",
                    {
                        "schema_version": "1.0",
                        "run_id": run_id,
                        "status": "running",
                        "strategy": "ddp",
                        "world_size": 4,
                        "config_sha256": config_sha256,
                        "dataset_version": opened.manifest.dataset_version,
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "git_commit": git_commit,
                        "git_dirty": False,
                    },
                )
                (artifact_dir / "metrics.jsonl").touch()
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {"event": "m5_formal_started", "world_size": 4},
                )
            progress = M5FormalProgress(0, 0, 0, None, None, ())
            mode: Literal["fresh", "exact_resume"] = "fresh"
            resumed_from_tokens: int | None = None
        else:
            artifact_dir = resume_run
            mode = "exact_resume"
            resumed_from_tokens = None
            progress = M5FormalProgress(0, 0, 0, None, None, ())
        dist.barrier()

        seed_everything(config.run.seed, deterministic_algorithms=False)
        if config.precision.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        dataset = M5FormalDataset(dataset_root)
        if len(dataset) % 4 != 0:
            raise M5FormalTrainingError("formal M5 Dataset cannot partition evenly over four Ranks")
        local_positions = tuple(range(rank, len(dataset), 4))
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
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        store = M5FormalCheckpointStore(artifact_dir / "checkpoints", keep_last=2)
        if mode == "exact_resume":
            selection: list[object] = [store.latest_valid() if rank == 0 else None]
            dist.broadcast_object_list(selection, src=0)
            manifest = cast(M5FormalCheckpointManifest, selection[0])
            payload: dict[str, Any] | None = None
            for reader_rank in range(4):
                if rank == reader_rank:
                    payload = store.load_payload(manifest, map_location=device)
                dist.barrier()
            assert payload is not None
            if (
                payload.get("config") != config.to_dict()
                or payload.get("config_sha256") != config_sha256
                or payload.get("dataset_version") != config.data.dataset_version
                or payload.get("dataset_manifest_sha256") != dataset_manifest_sha256
                or payload.get("git_commit") != git_commit
                or payload.get("world_size") != 4
            ):
                raise M5FormalTrainingError(
                    "formal M5 Exact Resume lineage or configuration changed"
                )
            model.load_state_dict(cast(dict[str, Tensor], payload["model"]), strict=True)
            optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
            progress = _load_progress(payload)
            rng_states = cast(tuple[object, object, object, object], payload["rank_rng"])
            if len(rng_states) != 4:
                raise M5FormalTrainingError("formal M5 Checkpoint Rank RNG set is incomplete")
            _restore_rng(rng_states[rank], device=device)
            resumed_from_tokens = progress.supervised_tokens
            if (
                progress.supervised_tokens != manifest.supervised_tokens
                or progress.global_step != manifest.global_step
                or progress.local_sequence_cursor != manifest.local_sequence_cursor
            ):
                raise M5FormalTrainingError("formal M5 Checkpoint payload differs from Manifest")
            if rank == 0:
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {
                        "event": "m5_formal_exact_resume",
                        "checkpoint_id": manifest.checkpoint_id,
                        "supervised_tokens": manifest.supervised_tokens,
                    },
                )

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        nn.Module.train(ddp_model)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        initial_loss = progress.initial_loss
        final_loss = progress.final_loss
        latest_checkpoint: str | None = None
        next_save = (
            progress.supervised_tokens // config.checkpoint.save_interval_tokens + 1
        ) * config.checkpoint.save_interval_tokens
        next_evaluation = (
            progress.supervised_tokens // config.training.evaluation_interval_tokens + 1
        ) * config.training.evaluation_interval_tokens

        while progress.local_sequence_cursor < len(local_positions):
            remaining = len(local_positions) - progress.local_sequence_cursor
            group_sequence_count = min(
                (config.training.micro_batch_size * config.training.gradient_accumulation_steps),
                remaining,
            )
            group_positions = local_positions[
                progress.local_sequence_cursor : (
                    progress.local_sequence_cursor + group_sequence_count
                )
            ]
            micro_batches = tuple(
                group_positions[offset : offset + config.training.micro_batch_size]
                for offset in range(0, len(group_positions), config.training.micro_batch_size)
            )
            valid_counts = tuple(
                sum(int((dataset[position]["labels"][1:] != -100).sum()) for position in positions)
                for positions in micro_batches
            )
            group_micro_steps = len(micro_batches)
            local_group_tokens = sum(valid_counts)
            group_tensor = torch.tensor(local_group_tokens, dtype=torch.int64, device=device)
            dist.all_reduce(group_tensor, op=dist.ReduceOp.SUM)
            global_group_tokens = int(group_tensor.item())
            if global_group_tokens <= 0:
                raise M5FormalTrainingError("formal M5 optimizer group has no supervision")
            learning_rate = token_learning_rate(
                base_learning_rate=config.training.learning_rate,
                tokens=progress.supervised_tokens,
                warmup_tokens=config.training.warmup_tokens,
                total_tokens=config.training.max_train_tokens,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            local_loss_numerator = 0.0
            for micro_index, (positions, valid_tokens) in enumerate(
                zip(micro_batches, valid_counts, strict=True)
            ):
                samples = tuple(dataset[position] for position in positions)
                batch = {
                    key: torch.stack(tuple(sample[key] for sample in samples)).to(
                        device,
                        non_blocking=True,
                    )
                    for key in samples[0]
                }
                sync_context = (
                    contextlib.nullcontext()
                    if micro_index == group_micro_steps - 1
                    else ddp_model.no_sync()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = ddp_model(**batch)
                        loss = output.loss
                    finite = torch.tensor(
                        int(bool(torch.isfinite(loss).item())),
                        dtype=torch.int32,
                        device=device,
                    )
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not bool(finite.item()):
                        raise M5FormalTrainingError("formal M5 training produced non-finite loss")
                    scaled = loss * (4.0 * valid_tokens / global_group_tokens)
                    scaled.backward()
                local_loss_numerator += float(loss.detach()) * valid_tokens
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(),
                    config.training.max_grad_norm,
                )
            )
            finite_gradient = torch.tensor(
                int(math.isfinite(gradient_norm)),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(finite_gradient, op=dist.ReduceOp.MIN)
            if not bool(finite_gradient.item()):
                raise M5FormalTrainingError("formal M5 training produced non-finite gradient norm")
            optimizer.step()
            loss_tensor = torch.tensor(
                local_loss_numerator,
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            weighted_loss = float(loss_tensor.item() / global_group_tokens)
            if initial_loss is None:
                initial_loss = weighted_loss
            final_loss = weighted_loss
            progress = M5FormalProgress(
                global_step=progress.global_step + 1,
                local_sequence_cursor=(progress.local_sequence_cursor + group_sequence_count),
                supervised_tokens=(progress.supervised_tokens + global_group_tokens),
                initial_loss=initial_loss,
                final_loss=final_loss,
                evaluation_checkpoints=progress.evaluation_checkpoints,
            )
            if rank == 0:
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
            final = progress.local_sequence_cursor == len(local_positions)
            evaluation_boundary = progress.supervised_tokens >= next_evaluation
            save_boundary = progress.supervised_tokens >= next_save
            if save_boundary or coordinated_stop or final:
                pin_reason: Literal["interruption", "evaluation", "final"] | None
                if final:
                    pin_reason = "final"
                elif coordinated_stop:
                    pin_reason = "interruption"
                elif evaluation_boundary:
                    pin_reason = "evaluation"
                else:
                    pin_reason = None
                local_rng = _capture_rng(device)
                gathered_rng: list[object] = [None, None, None, None]
                dist.all_gather_object(gathered_rng, local_rng)
                checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
                evaluation_points = progress.evaluation_checkpoints
                if evaluation_boundary and checkpoint_id not in evaluation_points:
                    evaluation_points = evaluation_points + (checkpoint_id,)
                    progress = M5FormalProgress(
                        global_step=progress.global_step,
                        local_sequence_cursor=progress.local_sequence_cursor,
                        supervised_tokens=progress.supervised_tokens,
                        initial_loss=progress.initial_loss,
                        final_loss=progress.final_loss,
                        evaluation_checkpoints=evaluation_points,
                    )
                if rank == 0:
                    checkpoint = store.save(
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        rank_rng=cast(
                            tuple[object, object, object, object],
                            tuple(gathered_rng),
                        ),
                        config=config,
                        config_sha256=config_sha256,
                        dataset_manifest_sha256=dataset_manifest_sha256,
                        run_id=run_id,
                        git_commit=git_commit,
                        pin_reason=pin_reason,
                    )
                    latest_checkpoint = checkpoint.checkpoint_id
                    _append_jsonl(
                        artifact_dir / "events.jsonl",
                        {
                            "event": "m5_formal_checkpoint",
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "pin_reason": pin_reason,
                        },
                    )
                checkpoint_box: list[object] = [latest_checkpoint if rank == 0 else None]
                dist.broadcast_object_list(checkpoint_box, src=0)
                latest_checkpoint = str(checkpoint_box[0])
                dist.barrier()
                while next_save <= progress.supervised_tokens:
                    next_save += config.checkpoint.save_interval_tokens
                while next_evaluation <= progress.supervised_tokens:
                    next_evaluation += config.training.evaluation_interval_tokens
            if coordinated_stop:
                break
            if time.monotonic() - started > config.training.max_job_duration_seconds:
                raise M5FormalTrainingError(
                    "formal M5 training exceeded the configured wall-clock limit"
                )

        if initial_loss is None or final_loss is None or latest_checkpoint is None:
            raise M5FormalTrainingError(
                "formal M5 training ended without loss or Checkpoint evidence"
            )
        status: Literal["succeeded", "interrupted"] = (
            "succeeded"
            if progress.supervised_tokens == config.training.max_train_tokens
            else "interrupted"
        )
        if status == "succeeded" and progress.local_sequence_cursor != len(local_positions):
            raise M5FormalTrainingError(
                "formal M5 reached Token target before consuming the fixed Dataset"
            )
        if status == "interrupted" and stop_after_tokens is None:
            raise M5FormalTrainingError("formal M5 stopped without a coordinated boundary")
        export_sha256: str | None = None
        if rank == 0 and status == "succeeded":
            export_sha256 = _export_model(model, artifact_dir / "exports" / "model")
        export_box: list[object] = [export_sha256]
        dist.broadcast_object_list(export_box, src=0)
        export_sha256 = cast(str | None, export_box[0])
        torch.cuda.synchronize(device)
        memory = M5FormalRankMemory(
            rank=rank,
            physical_gpu_index=physical_gpu_indices[local_rank],
            gpu_name=torch.cuda.get_device_name(device),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
        gathered_memory: list[object] = [None, None, None, None]
        dist.all_gather_object(gathered_memory, memory)
        result: M5FormalRunResult | None = None
        if rank == 0:
            result = M5FormalRunResult(
                status=status,
                mode=mode,
                run_id=run_id,
                config_sha256=config_sha256,
                git_commit=git_commit,
                git_dirty=False,
                model_revision=cast(
                    Literal["c1899de289a04d12100db370d81485cdf75e47ca"],
                    config.model.revision,
                ),
                attention_architecture=config.model.attention_architecture,
                dataset_version=cast(
                    Literal["m5-dual-sft-v1-b5b9e839"],
                    config.data.dataset_version,
                ),
                dataset_manifest_sha256=cast(
                    Literal["607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"],
                    dataset_manifest_sha256,
                ),
                thinking_fraction_basis_points=3000,
                seed=config.run.seed,
                world_size=4,
                global_step=progress.global_step,
                local_sequence_cursor=progress.local_sequence_cursor,
                supervised_tokens=progress.supervised_tokens,
                initial_loss=initial_loss,
                final_loss=final_loss,
                duration_seconds=time.monotonic() - started,
                rank_memory=cast(
                    tuple[
                        M5FormalRankMemory,
                        M5FormalRankMemory,
                        M5FormalRankMemory,
                        M5FormalRankMemory,
                    ],
                    tuple(gathered_memory),
                ),
                latest_checkpoint=latest_checkpoint,
                evaluation_checkpoints=progress.evaluation_checkpoints,
                resumed_from_tokens=resumed_from_tokens,
                export_sha256=export_sha256,
            )
            _record_result(artifact_dir, result)
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": status,
                    "strategy": "ddp",
                    "world_size": 4,
                    "config_sha256": config_sha256,
                    "dataset_version": config.data.dataset_version,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "git_commit": git_commit,
                    "git_dirty": False,
                    "supervised_tokens": progress.supervised_tokens,
                    "global_step": progress.global_step,
                    "latest_checkpoint": latest_checkpoint,
                },
            )
            print(result.model_dump_json(), flush=True)
        dist.barrier()
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
