"""Lineage, Checkpoint, export, and preflight support for M10 Agent LoRA."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from pydantic import ValidationError
from torch import Tensor
from torch.optim import Optimizer

from tinyllm.data.m10_mixture import M10FrozenDataset, open_frozen_mixture
from tinyllm.deployment.evaluation_subject import (
    ResolvedEvaluationSubject,
    evaluation_artifact_sha256,
    resolve_evaluation_subject,
)
from tinyllm.deployment.registry import DeploymentError
from tinyllm.training.m10_lora_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10_LORA_PARENT_TOKENIZER_SHA256,
    M10LoRACheckpointFile,
    M10LoRACheckpointManifest,
    M10LoRAConfig,
    M10LoRAContinuationGate,
    M10LoRAMemoryProbeResult,
    M10LoRARunResult,
    M10LoRAStageExport,
)
from tinyllm.training.m10_sft import (
    _append_jsonl,
    _atomic_json,
    _json_bytes,
    _sha256_file,
)

_TRAINING_STATE: Literal["training_state.pt"] = "training_state.pt"
_MANIFEST = "manifest.json"
_COMMITTED = "COMMITTED"
_LATEST = "LATEST"
_EXPORT_MANIFEST = "stage_export.json"
_ADAPTER_FILES: tuple[Literal["adapter_config.json"], Literal["adapter_model.safetensors"]] = (
    "adapter_config.json",
    "adapter_model.safetensors",
)


class M10LoRAError(RuntimeError):
    """Raised when the M10 Agent LoRA route fails closed."""


@dataclass(frozen=True, slots=True)
class M10LoRAProgress:
    """Durable optimizer and logical-epoch progress."""

    global_step: int
    completed_epochs: int
    sequence_cursor: int
    supervised_tokens: int
    initial_loss: float | None
    final_loss: float | None


def load_m10_lora_config(path: Path) -> M10LoRAConfig:
    """Load the strict Qwen3-8B Agent LoRA YAML."""

    try:
        return M10LoRAConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise M10LoRAError("M10 Agent LoRA config is invalid") from exc


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def collect_m10_lora_environment() -> tuple[dict[str, object], str]:
    """Collect the exact reviewed runtime used by Checkpoint lineage."""

    try:
        transformers_version = importlib.metadata.version("transformers")
        peft_version = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError as exc:
        raise M10LoRAError("M10 Agent LoRA requires Transformers and PEFT") from exc
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
    }
    expected = ("2.7.1+cu118", "11.8", "4.57.6", "0.19.1")
    actual = (
        payload["torch_version"],
        payload["cuda_runtime"],
        payload["transformers_version"],
        payload["peft_version"],
    )
    if actual != expected:
        raise M10LoRAError("M10 Agent LoRA software identity differs from the reviewed runtime")
    return payload, _payload_sha256(payload)


def collect_m10_lora_hardware(physical_gpu_index: int) -> tuple[dict[str, object], str]:
    """Collect stable physical-GPU identity even inside CUDA isolation."""

    query = "index,uuid,name,memory.total,pci.bus_id,driver_version"
    try:
        inventory = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise M10LoRAError("M10 Agent LoRA hardware inventory failed") from exc
    selected: tuple[str, str, int, str, str] | None = None
    try:
        for line in inventory.splitlines():
            index, uuid_value, name, memory, bus, driver = (
                part.strip() for part in line.split(",", maxsplit=5)
            )
            if int(index) == physical_gpu_index:
                selected = (uuid_value, name, int(memory), bus, driver)
                break
    except (TypeError, ValueError) as exc:
        raise M10LoRAError("M10 Agent LoRA hardware inventory cannot be parsed") from exc
    if selected is None or selected[1] != "NVIDIA GeForce RTX 3090":
        raise M10LoRAError("M10 Agent LoRA requires the selected physical RTX 3090")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "cuda_driver": selected[4],
        "selected_gpu": {
            "physical_gpu_index": physical_gpu_index,
            "uuid": selected[0],
            "name": selected[1],
            "memory_total_mib": selected[2],
            "pci_bus_id": selected[3],
        },
        "gpu_topology": topology,
    }
    compatibility = {
        "schema_version": "1.0",
        "platform": payload["platform"],
        "machine": payload["machine"],
        "cuda_driver": payload["cuda_driver"],
        "gpu_name": selected[1],
        "memory_total_mib": selected[2],
    }
    payload["exact_resume_compatibility_sha256"] = _payload_sha256(compatibility)
    return payload, cast(str, payload["exact_resume_compatibility_sha256"])


def _capture_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"python", "numpy", "torch", "cuda"}:
        raise M10LoRAError("M10 Agent LoRA Checkpoint RNG state is incomplete")
    try:
        random.setstate(cast(tuple[Any, ...], value["python"]))
        np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
        torch.set_rng_state(cast(Tensor, value["torch"]).cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in cast(list[Tensor], value["cuda"])])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M10LoRAError("M10 Agent LoRA Checkpoint RNG state cannot be restored") from exc


class M10LoRACheckpointStore:
    """Atomic Adapter-only Exact Resume points with stage-aware retention."""

    def __init__(self, root: Path, *, keep_last: int = 2) -> None:
        if keep_last != 2:
            raise ValueError("M10 Agent LoRA Checkpoint retention is fixed to two")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        adapter_state: dict[str, Tensor],
        optimizer: Optimizer,
        progress: M10LoRAProgress,
        config: M10LoRAConfig,
        config_sha256: str,
        run_id: str,
        git_commit: str,
        environment_sha256: str,
        hardware_sha256: str,
        memory_probe_sha256: str,
        pin_reason: Literal["stage", "final"] | None,
    ) -> M10LoRACheckpointManifest:
        if progress.sequence_cursor != 0:
            raise M10LoRAError("M10 Agent LoRA Checkpoints require an epoch boundary")
        checkpoint_id = f"checkpoint-tokens-{progress.supervised_tokens:010d}"
        destination = self.root / checkpoint_id
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return self.validate(checkpoint_id)
        temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            state_path = temporary / _TRAINING_STATE
            torch.save(
                {
                    "schema_version": "1.0",
                    "adapter": adapter_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": {
                        "kind": "token_warmup_cosine",
                        "tokens": progress.supervised_tokens,
                        "warmup_tokens": config.optimization.warmup_tokens,
                        "total_tokens": config.optimization.max_train_tokens,
                    },
                    "grad_scaler": None,
                    "progress": asdict(progress),
                    "rng": _capture_rng(),
                    "config": config.to_dict(),
                    "config_sha256": config_sha256,
                    "dataset_version": M10_DATASET_VERSION,
                    "dataset_manifest_sha256": M10_DATASET_MANIFEST_SHA256,
                    "parent_evaluation_subject": M10_LORA_PARENT_SUBJECT,
                    "parent_evaluation_subject_sha256": M10_LORA_PARENT_RECORD_SHA256,
                    "parent_model_artifact_sha256": M10_LORA_PARENT_MODEL_SHA256,
                    "environment_sha256": environment_sha256,
                    "hardware_sha256": hardware_sha256,
                    "memory_probe_sha256": memory_probe_sha256,
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "world_size": 1,
                    "peft_version": "0.19.1",
                },
                state_path,
            )
            with state_path.open("rb") as handle:
                os.fsync(handle.fileno())
            file = M10LoRACheckpointFile(
                path=_TRAINING_STATE,
                size_bytes=state_path.stat().st_size,
                sha256=_sha256_file(state_path),
            )
            manifest = M10LoRACheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                global_step=progress.global_step,
                completed_epochs=progress.completed_epochs,
                sequence_cursor=0,
                supervised_tokens=progress.supervised_tokens,
                config_sha256=config_sha256,
                dataset_version=M10_DATASET_VERSION,
                dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
                parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
                parent_evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
                parent_model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
                peft_version="0.19.1",
                git_commit=git_commit,
                environment_sha256=environment_sha256,
                hardware_sha256=hardware_sha256,
                memory_probe_sha256=memory_probe_sha256,
                file=file,
                pinned=pin_reason is not None,
                pin_reason=pin_reason,
            )
            manifest_bytes = _json_bytes(manifest.to_dict())
            (temporary / _MANIFEST).write_bytes(manifest_bytes)
            (temporary / _COMMITTED).write_bytes(
                _json_bytes({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
            )
            os.rename(temporary, destination)
            latest = self.root / f".{_LATEST}.tmp-{uuid.uuid4().hex}"
            latest.write_text(checkpoint_id + "\n", encoding="utf-8")
            os.replace(latest, self.root / _LATEST)
            self.validate(checkpoint_id)
            self._retain()
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def validate(self, checkpoint_id: str) -> M10LoRACheckpointManifest:
        directory = self.root / checkpoint_id
        try:
            manifest_bytes = (directory / _MANIFEST).read_bytes()
            manifest = M10LoRACheckpointManifest.model_validate_json(manifest_bytes)
            marker = json.loads((directory / _COMMITTED).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise M10LoRAError("M10 Agent LoRA Checkpoint metadata is incomplete") from exc
        state_path = directory / manifest.file.path
        if (
            marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
            or manifest.checkpoint_id != checkpoint_id
            or not state_path.is_file()
            or state_path.is_symlink()
            or state_path.stat().st_size != manifest.file.size_bytes
            or _sha256_file(state_path) != manifest.file.sha256
        ):
            raise M10LoRAError("M10 Agent LoRA Checkpoint failed integrity validation")
        return manifest

    def latest_valid(self) -> M10LoRACheckpointManifest:
        candidates = sorted(
            (path.name for path in self.root.glob("checkpoint-tokens-*") if path.is_dir()),
            reverse=True,
        )
        for checkpoint_id in candidates:
            try:
                return self.validate(checkpoint_id)
            except M10LoRAError:
                continue
        raise M10LoRAError("M10 Agent LoRA Run has no valid Exact Resume point")

    def load_payload(
        self,
        manifest: M10LoRACheckpointManifest,
        *,
        config: M10LoRAConfig,
        config_sha256: str,
        git_commit: str,
        environment_sha256: str,
        hardware_sha256: str,
        memory_probe_sha256: str,
        device: torch.device,
    ) -> tuple[dict[str, Any], M10LoRAProgress]:
        identities = (
            (manifest.config_sha256, config_sha256),
            (manifest.git_commit, git_commit),
            (manifest.environment_sha256, environment_sha256),
            (manifest.hardware_sha256, hardware_sha256),
            (manifest.memory_probe_sha256, memory_probe_sha256),
        )
        if any(actual != expected for actual, expected in identities):
            raise M10LoRAError("M10 Agent LoRA Exact Resume lineage changed")
        try:
            payload = cast(
                dict[str, Any],
                torch.load(
                    self.root / manifest.checkpoint_id / _TRAINING_STATE,
                    map_location=device,
                    weights_only=False,
                ),
            )
            if (
                payload["config"] != config.to_dict()
                or payload["config_sha256"] != config_sha256
                or payload["dataset_version"] != M10_DATASET_VERSION
                or payload["dataset_manifest_sha256"] != M10_DATASET_MANIFEST_SHA256
                or payload["parent_evaluation_subject"] != M10_LORA_PARENT_SUBJECT
                or payload["parent_evaluation_subject_sha256"] != M10_LORA_PARENT_RECORD_SHA256
                or payload["parent_model_artifact_sha256"] != M10_LORA_PARENT_MODEL_SHA256
                or payload["peft_version"] != "0.19.1"
                or payload["world_size"] != 1
                or payload["grad_scaler"] is not None
            ):
                raise M10LoRAError("M10 Agent LoRA Checkpoint payload lineage changed")
            value = cast(dict[str, Any], payload["progress"])
            progress = M10LoRAProgress(
                global_step=int(value["global_step"]),
                completed_epochs=int(value["completed_epochs"]),
                sequence_cursor=int(value["sequence_cursor"]),
                supervised_tokens=int(value["supervised_tokens"]),
                initial_loss=float(value["initial_loss"]),
                final_loss=float(value["final_loss"]),
            )
            _restore_rng(payload["rng"])
        except M10LoRAError:
            raise
        except Exception as exc:
            raise M10LoRAError("M10 Agent LoRA Checkpoint cannot be restored") from exc
        if (
            progress.global_step != manifest.global_step
            or progress.completed_epochs != manifest.completed_epochs
            or progress.sequence_cursor != 0
            or progress.supervised_tokens != manifest.supervised_tokens
        ):
            raise M10LoRAError("M10 Agent LoRA Checkpoint progress differs from Manifest")
        return payload, progress

    def _retain(self) -> None:
        manifests: list[M10LoRACheckpointManifest] = []
        for path in self.root.glob("checkpoint-tokens-*"):
            if path.is_dir():
                try:
                    manifests.append(self.validate(path.name))
                except M10LoRAError:
                    continue
        unpinned = sorted(
            (item for item in manifests if not item.pinned),
            key=lambda item: item.supervised_tokens,
            reverse=True,
        )
        for item in unpinned[self.keep_last :]:
            shutil.rmtree(self.root / item.checkpoint_id)


def export_m10_lora_stage(model: Any, root: Path, checkpoint_id: str) -> M10LoRAStageExport:
    """Atomically export one Adapter without redistributing Base weights."""

    destination = root / checkpoint_id
    if destination.exists():
        try:
            manifest_bytes = (destination / _EXPORT_MANIFEST).read_bytes()
            export = M10LoRAStageExport.model_validate_json(manifest_bytes)
            marker = json.loads((destination / _COMMITTED).read_text(encoding="utf-8"))
            digest = evaluation_artifact_sha256(destination / "adapter", _ADAPTER_FILES)
        except (OSError, json.JSONDecodeError, ValidationError, DeploymentError) as exc:
            raise M10LoRAError("M10 Agent LoRA stage export is incomplete") from exc
        if (
            export.checkpoint_id != checkpoint_id
            or export.adapter_artifact_sha256 != digest
            or marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        ):
            raise M10LoRAError("M10 Agent LoRA stage export failed integrity validation")
        return export
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
    adapter_root = temporary / "adapter"
    adapter_root.mkdir(parents=True)
    try:
        peft_config = getattr(model, "peft_config", {}).get("default")
        if peft_config is not None:
            peft_config.base_model_name_or_path = "Qwen/Qwen3-8B"
            peft_config.revision = "b968826d9c46dd6066d109eabc6255188de91218"
        model.save_pretrained(
            adapter_root,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        if any(not (adapter_root / name).is_file() for name in _ADAPTER_FILES):
            raise M10LoRAError("M10 Agent LoRA Adapter export is incomplete")
        digest = evaluation_artifact_sha256(adapter_root, _ADAPTER_FILES)
        tokens = int(checkpoint_id.removeprefix("checkpoint-tokens-"))
        export = M10LoRAStageExport(
            checkpoint_id=checkpoint_id,
            supervised_tokens=cast(Literal[1_000_000, 5_000_000, 10_000_000], tokens),
            adapter_artifact_sha256=digest,
            adapter_files=_ADAPTER_FILES,
        )
        manifest_bytes = _json_bytes(export.to_dict())
        (temporary / _EXPORT_MANIFEST).write_bytes(manifest_bytes)
        (temporary / _COMMITTED).write_bytes(
            _json_bytes({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()})
        )
        os.rename(temporary, destination)
        return export
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def preflight_m10_lora(
    *, config_path: Path, mixture_root: Path, artifact_root: Path
) -> tuple[M10LoRAConfig, ResolvedEvaluationSubject, str]:
    """Verify frozen data and the exact Qwen3-8B Base evaluation subject."""

    config = load_m10_lora_config(config_path)
    manifest = open_frozen_mixture(mixture_root)
    manifest_sha256 = _sha256_file(mixture_root / _MANIFEST)
    if (
        manifest.dataset_version != config.data.dataset_version
        or manifest_sha256 != config.data.manifest_sha256
        or manifest.target_supervised_tokens != config.data.target_supervised_tokens_per_epoch
        or manifest.sequence_length != config.data.sequence_length
    ):
        raise M10LoRAError("M10 Agent LoRA config and frozen Dataset differ")
    try:
        parent = resolve_evaluation_subject(artifact_root, config.model.parent_evaluation_subject)
    except DeploymentError as exc:
        raise M10LoRAError("M10 Agent LoRA parent cannot be resolved") from exc
    identities = (
        (parent.status, "Evaluation"),
        (parent.model_version, M10_LORA_PARENT_SUBJECT),
        (parent.evaluation_subject_sha256, M10_LORA_PARENT_RECORD_SHA256),
        (parent.model_artifact_sha256, M10_LORA_PARENT_MODEL_SHA256),
        (parent.tokenizer_artifact_sha256, M10_LORA_PARENT_TOKENIZER_SHA256),
        (parent.adapter_dir, None),
        (parent.model.repository, config.model.repository),
        (parent.model.base_revision, config.model.revision),
        (parent.model.role, "base"),
        (parent.model.adaptation, "base"),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAError("M10 Agent LoRA parent differs from the frozen Qwen3-8B Base")
    return config, parent, manifest_sha256


def require_m10_lora_storage(path: Path, *, minimum_free_bytes: int = 16 * 1024**3) -> int:
    """Reject a formal Run when the target filesystem lacks safe headroom."""

    if minimum_free_bytes <= 0:
        raise ValueError("M10 Agent LoRA minimum free bytes must be positive")
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise M10LoRAError("M10 Agent LoRA storage path has no existing ancestor")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    free_bytes = shutil.disk_usage(candidate).free
    if free_bytes < minimum_free_bytes:
        raise M10LoRAError(
            "M10 Agent LoRA storage preflight failed: "
            f"requires {minimum_free_bytes} free bytes, observed {free_bytes}"
        )
    return free_bytes


def load_m10_lora_memory_probe(
    path: Path, *, config_sha256: str, git_commit: str
) -> tuple[M10LoRAMemoryProbeResult, str]:
    """Verify the real 10-step BF16 feasibility evidence."""

    try:
        payload = path.read_bytes()
        result = M10LoRAMemoryProbeResult.model_validate_json(payload)
    except (OSError, ValidationError, ValueError) as exc:
        raise M10LoRAError("M10 Agent LoRA memory Probe is missing or invalid") from exc
    if (
        result.config_sha256 != config_sha256
        or result.git_commit != git_commit
        or result.parent_evaluation_subject != M10_LORA_PARENT_SUBJECT
    ):
        raise M10LoRAError("M10 Agent LoRA memory Probe lineage differs")
    return result, hashlib.sha256(payload).hexdigest()


def load_m10_lora_continuation_gate(
    path: Path,
    *,
    run_id: str,
    config_sha256: str,
    source_stage_tokens: int,
    source_adapter_artifact_sha256: str,
) -> tuple[M10LoRAContinuationGate, str]:
    """Verify immutable evidence before allowing the next LoRA stage."""

    try:
        payload = path.read_bytes()
        gate = M10LoRAContinuationGate.model_validate_json(payload)
    except (OSError, ValidationError, ValueError) as exc:
        raise M10LoRAError("M10 Agent LoRA continuation Gate is missing or invalid") from exc
    if gate.decision != "accepted":
        raise M10LoRAError("M10 Agent LoRA continuation Gate was rejected")
    identities = (
        (gate.run_id, run_id),
        (gate.config_sha256, config_sha256),
        (gate.source_stage_tokens, source_stage_tokens),
        (gate.source_adapter_artifact_sha256, source_adapter_artifact_sha256),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10LoRAError("M10 Agent LoRA continuation Gate lineage differs")
    return gate, hashlib.sha256(payload).hexdigest()


def record_m10_lora_result(artifact_dir: Path, result: M10LoRARunResult) -> None:
    attempt = f"{result.mode}-{result.status}-tokens-{result.supervised_tokens:010d}.json"
    _atomic_json(artifact_dir / "attempts" / attempt, result.to_dict())
    _atomic_json(artifact_dir / "result.json", result.to_dict())
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": f"m10_lora_{result.status}",
            "mode": result.mode,
            "supervised_tokens": result.supervised_tokens,
            "attempt_result": f"attempts/{attempt}",
        },
    )


__all__ = [
    "M10FrozenDataset",
    "M10LoRACheckpointStore",
    "M10LoRAError",
    "M10LoRAProgress",
    "collect_m10_lora_environment",
    "collect_m10_lora_hardware",
    "export_m10_lora_stage",
    "load_m10_lora_config",
    "load_m10_lora_continuation_gate",
    "load_m10_lora_memory_probe",
    "preflight_m10_lora",
    "record_m10_lora_result",
    "require_m10_lora_storage",
]
