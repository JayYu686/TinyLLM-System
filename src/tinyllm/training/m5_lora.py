"""Native single-GPU Qwen3-8B BF16 LoRA with exact recovery."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
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

from tinyllm.data import M5FormalDataset, open_m5_formal_dataset
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m5_ablation import token_learning_rate
from tinyllm.training.m5_config import M5SFTConfig, load_m5_sft_config
from tinyllm.training.m5_formal import (
    _append_jsonl,
    _atomic_json,
    _fsync_directory,
    _json_bytes,
    _json_sha256,
    _sha256_file,
    _write_fsynced,
)
from tinyllm.training.m5_formal_schema import M5FormalPackage
from tinyllm.training.m5_lora_schema import (
    M5LoRACheckpointFile,
    M5LoRACheckpointManifest,
    M5LoRAEnvironment,
    M5LoRAGPU,
    M5LoRAHardware,
    M5LoRAMemory,
    M5LoRARunResult,
)
from tinyllm.training.seed import seed_everything

_STATE_FILE = "training_state.pt"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED"
_LATEST_FILE = "LATEST"
_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class M5LoRAError(RuntimeError):
    """Raised when the formal LoRA route fails closed."""


@dataclass(frozen=True, slots=True)
class M5LoRAProgress:
    """Durable optimizer and deterministic dataset progress."""

    global_step: int
    sequence_cursor: int
    supervised_tokens: int
    initial_loss: float | None
    final_loss: float | None
    evaluation_checkpoints: tuple[str, ...]


def _collect_environment() -> M5LoRAEnvironment:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata["Name"] or "").strip().lower().replace("_", "-")
        if name:
            packages[name] = distribution.version
    try:
        transformers_version = importlib.metadata.version("transformers")
        peft_version = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError as exc:
        raise M5LoRAError("M5 LoRA requires the reviewed Transformers and PEFT packages") from exc
    return M5LoRAEnvironment(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        python_executable=sys.executable,
        torch_version=cast(Literal["2.7.1+cu118"], torch.__version__),
        cuda_runtime=cast(Literal["11.8"], torch.version.cuda),
        transformers_version=cast(Literal["4.57.6"], transformers_version),
        peft_version=cast(Literal["0.19.1"], peft_version),
        packages=tuple(
            M5FormalPackage(name=name, version=version)
            for name, version in sorted(packages.items())
        ),
    )


def _collect_hardware(physical_gpu_index: int) -> M5LoRAHardware:
    query = "index,uuid,name,memory.total,pci.bus_id,driver_version"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise M5LoRAError("M5 LoRA stable hardware inventory failed") from exc
    inventory: dict[int, tuple[str, str, int, str, str]] = {}
    try:
        for line in completed.stdout.splitlines():
            index, uuid_value, name, memory, pci_bus_id, driver = (
                item.strip() for item in line.split(",", maxsplit=5)
            )
            inventory[int(index)] = (
                uuid_value,
                name,
                int(memory),
                pci_bus_id,
                driver,
            )
        selected = inventory[physical_gpu_index]
    except (KeyError, TypeError, ValueError) as exc:
        raise M5LoRAError("M5 LoRA GPU inventory cannot be parsed") from exc
    return M5LoRAHardware(
        hostname=platform.node(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        cuda_driver=selected[4],
        selected_gpu=M5LoRAGPU(
            physical_gpu_index=physical_gpu_index,
            uuid=selected[0],
            name=cast(Literal["NVIDIA GeForce RTX 3090"], selected[1]),
            memory_total_mib=selected[2],
            pci_bus_id=selected[3],
        ),
        gpu_topology=topology,
    )


def _capture_rng(device: torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu(),
    }


def _restore_rng(value: object, *, device: torch.device) -> None:
    if not isinstance(value, dict) or set(value) != {"python", "numpy", "torch", "cuda"}:
        raise M5LoRAError("M5 LoRA RNG state is incomplete")
    try:
        random.setstate(cast(tuple[Any, ...], value["python"]))
        np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
        torch.set_rng_state(cast(Tensor, value["torch"]).cpu())
        torch.cuda.set_rng_state(cast(Tensor, value["cuda"]).cpu(), device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M5LoRAError("M5 LoRA RNG state cannot be restored") from exc


class M5LoRACheckpointStore:
    """Atomic adapter-state Checkpoints with rolling retention."""

    def __init__(self, root: Path, *, keep_last: int) -> None:
        if keep_last != 2:
            raise ValueError("M5 LoRA Checkpoint retention is fixed to two")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        adapter_state: dict[str, Tensor],
        optimizer: Optimizer,
        progress: M5LoRAProgress,
        rng: dict[str, object],
        config: M5SFTConfig,
        config_sha256: str,
        dataset_manifest_sha256: str,
        source_sequence_count: int,
        environment_sha256: str,
        hardware_sha256: str,
        run_id: str,
        git_commit: str,
        pin_reason: Literal["interruption", "evaluation", "final"] | None,
    ) -> M5LoRACheckpointManifest:
        """Publish one complete optimizer-boundary LoRA state."""

        checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
        dataset_epoch = progress.sequence_cursor / source_sequence_count
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / checkpoint_id
        if destination.exists():
            raise M5LoRAError("M5 LoRA Checkpoint destination already exists")
        temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            state_path = temporary / _STATE_FILE
            torch.save(
                {
                    "schema_version": "1.0",
                    "adapter": adapter_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": {
                        "kind": "token_warmup_cosine",
                        "tokens": progress.supervised_tokens,
                        "warmup_tokens": config.training.warmup_tokens,
                        "total_tokens": config.training.max_train_tokens,
                    },
                    "grad_scaler": None,
                    "progress": asdict(progress),
                    "rng": rng,
                    "config": config.to_dict(),
                    "config_sha256": config_sha256,
                    "dataset_version": config.data.dataset_version,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "source_sequence_count": source_sequence_count,
                    "dataset_epoch": dataset_epoch,
                    "tokenizer_revision": config.model.revision,
                    "environment_sha256": environment_sha256,
                    "hardware_sha256": hardware_sha256,
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "world_size": 1,
                    "peft_version": "0.19.1",
                },
                state_path,
            )
            with state_path.open("rb") as handle:
                os.fsync(handle.fileno())
            file = M5LoRACheckpointFile(
                path="training_state.pt",
                size_bytes=state_path.stat().st_size,
                sha256=_sha256_file(state_path),
            )
            manifest = M5LoRACheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                global_step=progress.global_step,
                sequence_cursor=progress.sequence_cursor,
                supervised_tokens=progress.supervised_tokens,
                dataset_epoch=dataset_epoch,
                config_sha256=config_sha256,
                dataset_version=cast(
                    Literal["m5-dual-sft-v1-b5b9e839"],
                    config.data.dataset_version,
                ),
                dataset_manifest_sha256=cast(
                    Literal["607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"],
                    dataset_manifest_sha256,
                ),
                model_revision="b968826d9c46dd6066d109eabc6255188de91218",
                peft_version="0.19.1",
                git_commit=git_commit,
                environment_sha256=environment_sha256,
                hardware_sha256=hardware_sha256,
                file=file,
                pinned=pin_reason is not None,
                pin_reason=pin_reason,
            )
            manifest_bytes = _json_bytes(manifest.to_dict())
            _write_fsynced(temporary / _MANIFEST_FILE, manifest_bytes)
            _write_fsynced(
                temporary / _COMMIT_FILE,
                (
                    json.dumps(
                        {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode(),
            )
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(self.root)
            self.validate(checkpoint_id)
            self._retain()
            latest = self.root / f".{_LATEST_FILE}.tmp-{uuid.uuid4().hex}"
            _write_fsynced(latest, (checkpoint_id + "\n").encode())
            os.replace(latest, self.root / _LATEST_FILE)
            _fsync_directory(self.root)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def validate(self, checkpoint_id: str) -> M5LoRACheckpointManifest:
        """Validate metadata, commit marker, size, and SHA256."""

        directory = self.root / checkpoint_id
        try:
            manifest_bytes = (directory / _MANIFEST_FILE).read_bytes()
            manifest = M5LoRACheckpointManifest.model_validate_json(manifest_bytes)
            marker = cast(
                dict[str, str],
                json.loads((directory / _COMMIT_FILE).read_text(encoding="utf-8")),
            )
        except Exception as exc:
            raise M5LoRAError("M5 LoRA Checkpoint metadata is incomplete or corrupt") from exc
        state_path = directory / manifest.file.path
        if (
            marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
            or manifest.checkpoint_id != checkpoint_id
            or not state_path.is_file()
            or state_path.stat().st_size != manifest.file.size_bytes
            or _sha256_file(state_path) != manifest.file.sha256
        ):
            raise M5LoRAError("M5 LoRA Checkpoint integrity validation failed")
        return manifest

    def latest_valid(self) -> M5LoRACheckpointManifest:
        """Return the newest valid Checkpoint while skipping corrupt newer states."""

        candidates = sorted(
            (path.name for path in self.root.glob("checkpoint-tokens-*") if path.is_dir()),
            reverse=True,
        )
        for checkpoint_id in candidates:
            try:
                return self.validate(checkpoint_id)
            except M5LoRAError:
                continue
        raise M5LoRAError("M5 LoRA Run has no valid Exact Resume point")

    def load_payload(
        self,
        manifest: M5LoRACheckpointManifest,
        *,
        map_location: torch.device,
    ) -> dict[str, Any]:
        """Load one previously integrity-validated state."""

        try:
            payload = torch.load(
                self.root / manifest.checkpoint_id / _STATE_FILE,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise M5LoRAError("M5 LoRA training state cannot be loaded") from exc
        if not isinstance(payload, dict):
            raise M5LoRAError("M5 LoRA training state is not an object")
        return cast(dict[str, Any], payload)

    def _retain(self) -> None:
        manifests: list[M5LoRACheckpointManifest] = []
        for path in self.root.glob("checkpoint-tokens-*"):
            if not path.is_dir():
                continue
            try:
                manifests.append(
                    M5LoRACheckpointManifest.model_validate_json(
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
        raise M5LoRAError("M5 LoRA Qwen3-8B config cannot be parsed") from exc
    if {
        "model_type": decoded.get("model_type"),
        "num_attention_heads": decoded.get("num_attention_heads"),
        "num_key_value_heads": decoded.get("num_key_value_heads"),
    } != {"model_type": "qwen3", "num_attention_heads": 32, "num_key_value_heads": 8}:
        raise M5LoRAError("M5 LoRA model is not the frozen Qwen3-8B GQA route")
    if model_dir.name != config.model.revision:
        raise M5LoRAError("M5 LoRA model path does not match the pinned Revision")


def _load_progress(payload: dict[str, Any]) -> M5LoRAProgress:
    try:
        raw = cast(dict[str, object], payload["progress"])
        return M5LoRAProgress(
            global_step=int(cast(int, raw["global_step"])),
            sequence_cursor=int(cast(int, raw["sequence_cursor"])),
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
        raise M5LoRAError("M5 LoRA Checkpoint progress is invalid") from exc


def _export_adapter(
    model: Any,
    *,
    model_dir: Path,
    root: Path,
    run_id: str,
    dataset_version: str,
    git_commit: str,
) -> str:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    root.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(root, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(root)
    model_card = f"""---
base_model: Qwen/Qwen3-8B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - qwen3
  - tinyllm-system
---

# TinyLLM-System Qwen3-8B Dual-Mode LoRA

This adapter was trained with BF16 LoRA rank 16, alpha 32, and dropout 0.05 over
the Qwen3 Attention and MLP linear layers. It supports explicit Thinking and
Non-thinking prompts under the TinyLLM M5 dual-mode contract.

- Base revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Dataset version: `{dataset_version}`
- Training Run: `{run_id}`
- Training Git commit: `{git_commit}`
- Sequence length: 1024
- Supervised-token budget: 10,000,000

The adapter requires the pinned Base model and does not redistribute Base weights.
M6 evaluation determines Candidate eligibility; this card does not claim Production status.
"""
    _write_fsynced(root / "README.md", model_card.encode())
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise M5LoRAError("M5 LoRA Adapter export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _record_result(artifact_dir: Path, result: M5LoRARunResult) -> None:
    attempt = f"{result.mode}-{result.status}-tokens-{result.supervised_tokens:010d}.json"
    _atomic_json(artifact_dir / "attempts" / attempt, result.to_dict())
    _atomic_json(artifact_dir / "result.json", result.to_dict())
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": f"m5_lora_{result.status}",
            "mode": result.mode,
            "supervised_tokens": result.supervised_tokens,
            "attempt_result": f"attempts/{attempt}",
        },
    )


def run_m5_lora(
    *,
    config_path: Path,
    dataset_root: Path,
    model_dir: Path,
    output_root: Path,
    physical_gpu_index: int,
    resume_run: Path | None = None,
    stop_after_tokens: int | None = None,
) -> M5LoRARunResult:
    """Train or Exact-Resume the formal 10M-token Qwen3-8B LoRA Run."""

    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM

    config = load_m5_sft_config(config_path)
    if (
        config.run.purpose != "formal"
        or config.model.adaptation != "lora"
        or config.parallel.strategy != "single"
        or config.model.lora is None
    ):
        raise M5LoRAError("M5 LoRA worker received the wrong training route")
    if stop_after_tokens is not None and not 0 < stop_after_tokens < 10_000_000:
        raise M5LoRAError("M5 LoRA coordinated stop must be inside the 10M budget")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise M5LoRAError("M5 LoRA requires one BF16-capable CUDA device")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    artifact_dir: Path | None = None
    try:
        project_root = Path(__file__).resolve().parents[3]
        git_commit, git_dirty = read_git_identity(project_root)
        if git_dirty:
            raise M5LoRAError("M5 LoRA training requires a clean Git worktree")
        _validate_model_identity(config, model_dir)
        opened = open_m5_formal_dataset(dataset_root)
        dataset_manifest_sha256 = hashlib.sha256(
            (dataset_root / _MANIFEST_FILE).read_bytes()
        ).hexdigest()
        if (
            opened.manifest.dataset_version != config.data.dataset_version
            or dataset_manifest_sha256 != config.data.mix_manifest_sha256
            or opened.manifest.source_supervised_tokens != 1_000_000
        ):
            raise M5LoRAError("M5 LoRA config and Dataset identity differ")
        environment = _collect_environment()
        hardware = _collect_hardware(physical_gpu_index)
        environment_payload = environment.to_dict()
        hardware_payload = hardware.to_dict()
        environment_sha256 = _json_sha256(environment_payload)
        hardware_sha256 = _json_sha256(hardware_payload)
        config_sha256 = canonical_config_hash(config)
        if resume_run is None:
            run_id = generate_run_id(config.run.name, config_sha256, now=datetime.now(UTC))
            artifact_dir = output_root / run_id
            artifact_dir.mkdir(parents=True, exist_ok=False)
            for name in ("checkpoints", "evaluations", "exports", "attempts"):
                (artifact_dir / name).mkdir()
            shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
            _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
            _atomic_json(artifact_dir / "dataset.json", opened.manifest.to_dict())
            _atomic_json(artifact_dir / "environment.json", environment_payload)
            _atomic_json(artifact_dir / "hardware.json", hardware_payload)
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": "running",
                    "strategy": "single_gpu_bf16_lora",
                    "world_size": 1,
                    "config_sha256": config_sha256,
                    "dataset_version": opened.manifest.dataset_version,
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "git_commit": git_commit,
                    "git_dirty": False,
                    "environment_sha256": environment_sha256,
                    "hardware_sha256": hardware_sha256,
                },
            )
            (artifact_dir / "metrics.jsonl").touch()
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {"event": "m5_lora_started", "physical_gpu_index": physical_gpu_index},
            )
            progress = M5LoRAProgress(0, 0, 0, None, None, ())
            mode: Literal["fresh", "exact_resume"] = "fresh"
            resumed_from_tokens: int | None = None
        else:
            artifact_dir = resume_run
            try:
                stored_environment = M5LoRAEnvironment.model_validate_json(
                    (artifact_dir / "environment.json").read_bytes()
                )
                stored_hardware = M5LoRAHardware.model_validate_json(
                    (artifact_dir / "hardware.json").read_bytes()
                )
                run_id = str(
                    json.loads((artifact_dir / "run.json").read_text(encoding="utf-8"))["run_id"]
                )
            except Exception as exc:
                raise M5LoRAError("M5 LoRA Exact Resume lineage is incomplete") from exc
            if (
                stored_environment.to_dict() != environment_payload
                or stored_hardware.to_dict() != hardware_payload
            ):
                raise M5LoRAError("M5 LoRA Exact Resume software or hardware identity changed")
            progress = M5LoRAProgress(0, 0, 0, None, None, ())
            mode = "exact_resume"
            resumed_from_tokens = None

        seed_everything(config.run.seed, deterministic_algorithms=False)
        if config.precision.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        dataset = M5FormalDataset(dataset_root)
        sequence_limit = opened.manifest.source_sequence_count * 10
        torch.cuda.reset_peak_memory_stats(device)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=config.model.lora.rank,
            lora_alpha=config.model.lora.alpha,
            lora_dropout=config.model.lora.dropout,
            target_modules=list(_TARGET_MODULES),
            bias=config.model.lora.bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        nn.Module.train(model)
        trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        trainable_parameters = sum(parameter.numel() for parameter in trainable)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        if trainable_parameters <= 0 or trainable_parameters >= total_parameters:
            raise M5LoRAError("M5 LoRA trainable-parameter topology is invalid")
        optimizer = AdamW(
            trainable,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        store = M5LoRACheckpointStore(artifact_dir / "checkpoints", keep_last=2)
        if mode == "exact_resume":
            manifest = store.latest_valid()
            payload = store.load_payload(manifest, map_location=device)
            if (
                payload.get("config") != config.to_dict()
                or payload.get("config_sha256") != config_sha256
                or payload.get("dataset_version") != config.data.dataset_version
                or payload.get("dataset_manifest_sha256") != dataset_manifest_sha256
                or payload.get("source_sequence_count") != opened.manifest.source_sequence_count
                or payload.get("tokenizer_revision") != config.model.revision
                or payload.get("git_commit") != git_commit
                or payload.get("environment_sha256") != environment_sha256
                or payload.get("hardware_sha256") != hardware_sha256
                or payload.get("world_size") != 1
                or payload.get("peft_version") != "0.19.1"
                or "grad_scaler" not in payload
                or payload["grad_scaler"] is not None
            ):
                raise M5LoRAError("M5 LoRA Exact Resume lineage or configuration changed")
            set_peft_model_state_dict(
                model,
                cast(dict[str, Tensor], payload["adapter"]),
            )
            optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer"]))
            progress = _load_progress(payload)
            _restore_rng(payload["rng"], device=device)
            resumed_from_tokens = progress.supervised_tokens
            dataset_epoch = progress.sequence_cursor / opened.manifest.source_sequence_count
            if (
                progress.supervised_tokens != manifest.supervised_tokens
                or progress.global_step != manifest.global_step
                or progress.sequence_cursor != manifest.sequence_cursor
                or payload.get("dataset_epoch") != dataset_epoch
                or manifest.dataset_epoch != dataset_epoch
            ):
                raise M5LoRAError("M5 LoRA Checkpoint payload differs from Manifest")
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "m5_lora_exact_resume",
                    "checkpoint_id": manifest.checkpoint_id,
                    "supervised_tokens": manifest.supervised_tokens,
                },
            )

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
        while progress.sequence_cursor < sequence_limit:
            step_started = time.monotonic()
            remaining = sequence_limit - progress.sequence_cursor
            group_sequence_count = min(
                config.training.micro_batch_size * config.training.gradient_accumulation_steps,
                remaining,
            )
            group_positions = tuple(
                range(
                    progress.sequence_cursor,
                    progress.sequence_cursor + group_sequence_count,
                )
            )
            micro_batches = tuple(
                group_positions[offset : offset + config.training.micro_batch_size]
                for offset in range(0, len(group_positions), config.training.micro_batch_size)
            )
            valid_counts = tuple(
                sum(int((dataset[position]["labels"][1:] != -100).sum()) for position in positions)
                for positions in micro_batches
            )
            group_tokens = sum(valid_counts)
            if group_tokens <= 0:
                raise M5LoRAError("M5 LoRA optimizer group has no supervision")
            learning_rate = token_learning_rate(
                base_learning_rate=config.training.learning_rate,
                tokens=progress.supervised_tokens,
                warmup_tokens=config.training.warmup_tokens,
                total_tokens=config.training.max_train_tokens,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss_numerator = 0.0
            for positions, valid_tokens in zip(micro_batches, valid_counts, strict=True):
                samples = tuple(dataset[position] for position in positions)
                batch = {
                    key: torch.stack(tuple(sample[key] for sample in samples)).to(
                        device,
                        non_blocking=True,
                    )
                    for key in samples[0]
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**batch)
                    loss = output.loss
                if not bool(torch.isfinite(loss).item()):
                    raise M5LoRAError("M5 LoRA training produced non-finite loss")
                (loss * (valid_tokens / group_tokens)).backward()
                loss_numerator += float(loss.detach()) * valid_tokens
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    config.training.max_grad_norm,
                )
            )
            if not math.isfinite(gradient_norm):
                raise M5LoRAError("M5 LoRA training produced non-finite gradient norm")
            optimizer.step()
            weighted_loss = loss_numerator / group_tokens
            if initial_loss is None:
                initial_loss = weighted_loss
            final_loss = weighted_loss
            progress = M5LoRAProgress(
                global_step=progress.global_step + 1,
                sequence_cursor=progress.sequence_cursor + group_sequence_count,
                supervised_tokens=progress.supervised_tokens + group_tokens,
                initial_loss=initial_loss,
                final_loss=final_loss,
                evaluation_checkpoints=progress.evaluation_checkpoints,
            )
            step_duration = time.monotonic() - step_started
            _append_jsonl(
                artifact_dir / "metrics.jsonl",
                {
                    "global_step": progress.global_step,
                    "supervised_tokens": progress.supervised_tokens,
                    "loss": weighted_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "optimizer_step_duration_seconds": step_duration,
                    "supervised_tokens_per_second": group_tokens / step_duration,
                },
            )
            coordinated_stop = (
                stop_after_tokens is not None and progress.supervised_tokens >= stop_after_tokens
            )
            final = progress.sequence_cursor == sequence_limit
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
                checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
                evaluation_points = progress.evaluation_checkpoints
                if evaluation_boundary and checkpoint_id not in evaluation_points:
                    evaluation_points = evaluation_points + (checkpoint_id,)
                    progress = M5LoRAProgress(
                        global_step=progress.global_step,
                        sequence_cursor=progress.sequence_cursor,
                        supervised_tokens=progress.supervised_tokens,
                        initial_loss=progress.initial_loss,
                        final_loss=progress.final_loss,
                        evaluation_checkpoints=evaluation_points,
                    )
                checkpoint = store.save(
                    adapter_state=cast(
                        dict[str, Tensor],
                        get_peft_model_state_dict(model),
                    ),
                    optimizer=optimizer,
                    progress=progress,
                    rng=_capture_rng(device),
                    config=config,
                    config_sha256=config_sha256,
                    dataset_manifest_sha256=dataset_manifest_sha256,
                    source_sequence_count=opened.manifest.source_sequence_count,
                    environment_sha256=environment_sha256,
                    hardware_sha256=hardware_sha256,
                    run_id=run_id,
                    git_commit=git_commit,
                    pin_reason=pin_reason,
                )
                latest_checkpoint = checkpoint.checkpoint_id
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {
                        "event": "m5_lora_checkpoint",
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "pin_reason": pin_reason,
                    },
                )
                while next_save <= progress.supervised_tokens:
                    next_save += config.checkpoint.save_interval_tokens
                while next_evaluation <= progress.supervised_tokens:
                    next_evaluation += config.training.evaluation_interval_tokens
            if coordinated_stop:
                break
            if time.monotonic() - started > config.training.max_job_duration_seconds:
                raise M5LoRAError("M5 LoRA exceeded the configured wall-clock limit")

        if initial_loss is None or final_loss is None or latest_checkpoint is None:
            raise M5LoRAError("M5 LoRA ended without loss or Checkpoint evidence")
        status: Literal["succeeded", "interrupted"] = (
            "succeeded" if progress.supervised_tokens == 10_000_000 else "interrupted"
        )
        if status == "succeeded" and progress.sequence_cursor != sequence_limit:
            raise M5LoRAError("M5 LoRA reached Token target before fixed Dataset completion")
        if status == "interrupted" and stop_after_tokens is None:
            raise M5LoRAError("M5 LoRA stopped without a coordinated boundary")
        adapter_sha256: str | None = None
        if status == "succeeded":
            adapter_sha256 = _export_adapter(
                model,
                model_dir=model_dir,
                root=artifact_dir / "exports" / "adapter",
                run_id=run_id,
                dataset_version=config.data.dataset_version,
                git_commit=git_commit,
            )
        torch.cuda.synchronize(device)
        result = M5LoRARunResult(
            status=status,
            mode=mode,
            run_id=run_id,
            config_sha256=config_sha256,
            git_commit=git_commit,
            git_dirty=False,
            environment_sha256=environment_sha256,
            hardware_sha256=hardware_sha256,
            model_revision="b968826d9c46dd6066d109eabc6255188de91218",
            attention_architecture="gqa",
            adaptation="lora",
            peft_version="0.19.1",
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
            world_size=1,
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
            global_step=progress.global_step,
            sequence_cursor=progress.sequence_cursor,
            supervised_tokens=progress.supervised_tokens,
            completed_dataset_epochs=(
                progress.sequence_cursor / opened.manifest.source_sequence_count
            ),
            initial_loss=initial_loss,
            final_loss=final_loss,
            duration_seconds=time.monotonic() - started,
            memory=M5LoRAMemory(
                physical_gpu_index=physical_gpu_index,
                gpu_name="NVIDIA GeForce RTX 3090",
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            ),
            latest_checkpoint=latest_checkpoint,
            evaluation_checkpoints=progress.evaluation_checkpoints,
            resumed_from_tokens=resumed_from_tokens,
            adapter_sha256=adapter_sha256,
        )
        _record_result(artifact_dir, result)
        _atomic_json(
            artifact_dir / "run.json",
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "status": status,
                "strategy": "single_gpu_bf16_lora",
                "world_size": 1,
                "config_sha256": config_sha256,
                "dataset_version": config.data.dataset_version,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "git_commit": git_commit,
                "git_dirty": False,
                "environment_sha256": environment_sha256,
                "hardware_sha256": hardware_sha256,
                "supervised_tokens": progress.supervised_tokens,
                "global_step": progress.global_step,
                "latest_checkpoint": latest_checkpoint,
            },
        )
        print(result.model_dump_json(), flush=True)
        return result
    except Exception as exc:
        if artifact_dir is not None and artifact_dir.is_dir():
            with contextlib.suppress(OSError):
                _atomic_json(
                    artifact_dir / "run.json",
                    {
                        "schema_version": "1.0",
                        "run_id": artifact_dir.name,
                        "status": "failed",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    },
                )
        raise
