"""Native staged single-GPU Qwen3-0.6B Agent Full-SFT training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from pydantic import ValidationError
from torch import Tensor, nn
from torch.optim import Optimizer

from tinyllm.data.m10_mixture import M10FrozenDataset, open_frozen_mixture
from tinyllm.deployment.registry import DeploymentError, resolve_model
from tinyllm.deployment.schema import ResolvedModel
from tinyllm.training.m5_ablation import M5AblationError, model_export_sha256
from tinyllm.training.m10_sft_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_PARENT_MODEL_SHA256,
    M10_PARENT_RECORD_SHA256,
    M10_PARENT_VERSION,
    M10CheckpointFile,
    M10CheckpointManifest,
    M10ContinuationGate,
    M10FullSFTConfig,
    M10FullSFTRunResult,
    M10StageExport,
)

_TRAINING_STATE: Literal["training_state.pt"] = "training_state.pt"
_MANIFEST = "manifest.json"
_COMMITTED = "COMMITTED"
_LATEST = "LATEST"
_EXPORT_MANIFEST = "stage_export.json"


class M10FullSFTError(RuntimeError):
    """Raised when M10.2 preflight, training, or Exact Resume fails closed."""


@dataclass(frozen=True, slots=True)
class M10Progress:
    """Durable optimizer and logical-epoch progress."""

    global_step: int
    completed_epochs: int
    sequence_cursor: int
    supervised_tokens: int
    initial_loss: float | None
    final_loss: float | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_m10_full_sft_config(path: Path) -> M10FullSFTConfig:
    """Load the strict M10.2 YAML contract."""

    try:
        return M10FullSFTConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise M10FullSFTError("M10 Full-SFT config is invalid") from exc


def _capture_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: dict[str, object]) -> None:
    if set(value) != {"python", "numpy", "torch", "cuda"}:
        raise M10FullSFTError("M10 Checkpoint RNG state is incomplete")
    random.setstate(cast(tuple[Any, ...], value["python"]))
    np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
    torch.set_rng_state(cast(Tensor, value["torch"]).cpu())
    states = cast(list[Tensor], value["cuda"])
    torch.cuda.set_rng_state_all([state.cpu() for state in states])


def epoch_order(length: int, *, seed: int, epoch: int) -> tuple[int, ...]:
    """Return the deterministic permutation for one logical 1M-token epoch."""

    if length <= 0 or epoch < 0:
        raise ValueError("M10 epoch order inputs are invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return tuple(int(value) for value in torch.randperm(length, generator=generator).tolist())


def _batch(dataset: M10FrozenDataset, indices: tuple[int, ...]) -> dict[str, Tensor]:
    examples = tuple(dataset[index] for index in indices)
    if not examples:
        raise M10FullSFTError("M10 optimizer group cannot be empty")
    return {key: torch.stack([item[key] for item in examples]) for key in examples[0]}


def validate_m10_parent(config: M10FullSFTConfig, resolved: ResolvedModel) -> None:
    """Bind training to the current immutable M7 Production parent."""

    identities = (
        (resolved.status, "Production"),
        (resolved.model_version, config.model.parent_production_version),
        (resolved.production_record_sha256, config.model.parent_production_record_sha256),
        (resolved.model_artifact_sha256, config.model.parent_model_artifact_sha256),
        (resolved.model.repository, config.model.repository),
        (resolved.model.base_revision, config.model.revision),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10FullSFTError("M10 parent differs from the frozen M7 Production identity")


class M10CheckpointStore:
    """Atomic hash-verified M10 Checkpoints with stage-aware retention."""

    def __init__(self, root: Path, *, keep_last: int = 2) -> None:
        if keep_last != 2:
            raise ValueError("M10 Checkpoint retention is fixed to two")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        progress: M10Progress,
        config: M10FullSFTConfig,
        config_sha256: str,
        run_id: str,
        git_commit: str,
        pin_reason: Literal["stage", "final"] | None,
    ) -> M10CheckpointManifest:
        if progress.sequence_cursor != 0:
            raise M10FullSFTError("M10 Checkpoints may only commit at epoch boundaries")
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
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "progress": asdict(progress),
                    "rng": _capture_rng(),
                    "config": config.to_dict(),
                    "config_sha256": config_sha256,
                    "dataset_version": M10_DATASET_VERSION,
                    "dataset_manifest_sha256": M10_DATASET_MANIFEST_SHA256,
                    "parent_production_version": M10_PARENT_VERSION,
                    "parent_production_record_sha256": M10_PARENT_RECORD_SHA256,
                    "parent_model_artifact_sha256": M10_PARENT_MODEL_SHA256,
                    "run_id": run_id,
                    "git_commit": git_commit,
                },
                state_path,
            )
            file = M10CheckpointFile(
                path=_TRAINING_STATE,
                size_bytes=state_path.stat().st_size,
                sha256=_sha256_file(state_path),
            )
            manifest = M10CheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                global_step=progress.global_step,
                completed_epochs=progress.completed_epochs,
                sequence_cursor=0,
                supervised_tokens=progress.supervised_tokens,
                config_sha256=config_sha256,
                dataset_version=M10_DATASET_VERSION,
                dataset_manifest_sha256=M10_DATASET_MANIFEST_SHA256,
                parent_production_version=M10_PARENT_VERSION,
                parent_production_record_sha256=M10_PARENT_RECORD_SHA256,
                parent_model_artifact_sha256=M10_PARENT_MODEL_SHA256,
                git_commit=git_commit,
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

    def validate(self, checkpoint_id: str) -> M10CheckpointManifest:
        directory = self.root / checkpoint_id
        try:
            manifest_bytes = (directory / _MANIFEST).read_bytes()
            manifest = M10CheckpointManifest.model_validate_json(manifest_bytes)
            marker = json.loads((directory / _COMMITTED).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise M10FullSFTError("M10 Checkpoint metadata is incomplete or corrupt") from exc
        if marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}:
            raise M10FullSFTError("M10 Checkpoint commit marker is invalid")
        state_path = directory / manifest.file.path
        if (
            manifest.checkpoint_id != checkpoint_id
            or not state_path.is_file()
            or state_path.is_symlink()
            or state_path.stat().st_size != manifest.file.size_bytes
            or _sha256_file(state_path) != manifest.file.sha256
        ):
            raise M10FullSFTError("M10 Checkpoint payload failed integrity validation")
        return manifest

    def latest_valid(self) -> M10CheckpointManifest:
        candidates = sorted(
            (path.name for path in self.root.glob("checkpoint-tokens-*") if path.is_dir()),
            reverse=True,
        )
        for checkpoint_id in candidates:
            try:
                return self.validate(checkpoint_id)
            except M10FullSFTError:
                continue
        raise M10FullSFTError("M10 Checkpoint store contains no valid Exact Resume point")

    def load(
        self,
        manifest: M10CheckpointManifest,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        config: M10FullSFTConfig,
        config_sha256: str,
        git_commit: str,
        device: torch.device,
    ) -> M10Progress:
        identities = (
            (manifest.config_sha256, config_sha256),
            (manifest.dataset_version, config.data.dataset_version),
            (manifest.dataset_manifest_sha256, config.data.manifest_sha256),
            (manifest.parent_production_version, config.model.parent_production_version),
            (
                manifest.parent_production_record_sha256,
                config.model.parent_production_record_sha256,
            ),
            (manifest.parent_model_artifact_sha256, config.model.parent_model_artifact_sha256),
            (manifest.git_commit, git_commit),
        )
        if any(actual != expected for actual, expected in identities):
            raise M10FullSFTError("M10 Exact Resume lineage or configuration changed")
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
                raise M10FullSFTError("M10 Checkpoint config payload changed")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            value = cast(dict[str, Any], payload["progress"])
            progress = M10Progress(
                global_step=int(value["global_step"]),
                completed_epochs=int(value["completed_epochs"]),
                sequence_cursor=int(value["sequence_cursor"]),
                supervised_tokens=int(value["supervised_tokens"]),
                initial_loss=float(value["initial_loss"]),
                final_loss=float(value["final_loss"]),
            )
            _restore_rng(cast(dict[str, object], payload["rng"]))
        except M10FullSFTError:
            raise
        except Exception as exc:
            raise M10FullSFTError("M10 Checkpoint training state cannot be restored") from exc
        if (
            progress.global_step != manifest.global_step
            or progress.completed_epochs != manifest.completed_epochs
            or progress.sequence_cursor != 0
            or progress.supervised_tokens != manifest.supervised_tokens
        ):
            raise M10FullSFTError("M10 Checkpoint progress differs from manifest")
        return progress

    def _retain(self) -> None:
        manifests: list[M10CheckpointManifest] = []
        for path in self.root.glob("checkpoint-tokens-*"):
            if path.is_dir():
                try:
                    manifests.append(self.validate(path.name))
                except M10FullSFTError:
                    continue
        unpinned = sorted(
            (item for item in manifests if not item.pinned),
            key=lambda item: item.supervised_tokens,
            reverse=True,
        )
        for item in unpinned[self.keep_last :]:
            shutil.rmtree(self.root / item.checkpoint_id)


def _export_stage(model: nn.Module, root: Path, checkpoint_id: str) -> M10StageExport:
    destination = root / checkpoint_id
    if destination.exists():
        try:
            manifest_bytes = (destination / _EXPORT_MANIFEST).read_bytes()
            export = M10StageExport.model_validate_json(manifest_bytes)
            marker = json.loads((destination / _COMMITTED).read_text(encoding="utf-8"))
            digest = model_export_sha256(destination / "model")
        except (OSError, json.JSONDecodeError, ValidationError, M5AblationError) as exc:
            raise M10FullSFTError("M10 stage export is incomplete or corrupt") from exc
        if (
            export.checkpoint_id != checkpoint_id
            or export.export_sha256 != digest
            or marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        ):
            raise M10FullSFTError("M10 stage export failed integrity validation")
        return export
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
    export_root = temporary / "model"
    export_root.mkdir(parents=True)
    try:
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise M10FullSFTError("Qwen model does not expose save_pretrained")
        save_pretrained(export_root, safe_serialization=True)
        digest = model_export_sha256(export_root)
        tokens = int(checkpoint_id.removeprefix("checkpoint-tokens-"))
        export = M10StageExport(
            checkpoint_id=checkpoint_id,
            supervised_tokens=cast(Literal[1_000_000, 5_000_000, 10_000_000], tokens),
            export_sha256=digest,
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


def preflight_m10_full_sft(
    *, config_path: Path, mixture_root: Path, artifact_root: Path
) -> tuple[M10FullSFTConfig, ResolvedModel, str]:
    """Verify the config, frozen arrays, and M7 parent without loading CUDA weights."""

    config = load_m10_full_sft_config(config_path)
    manifest = open_frozen_mixture(mixture_root)
    manifest_sha256 = _sha256_file(mixture_root / _MANIFEST)
    if (
        manifest.dataset_version != config.data.dataset_version
        or manifest_sha256 != config.data.manifest_sha256
        or manifest.target_supervised_tokens != config.data.target_supervised_tokens_per_epoch
        or manifest.sequence_length != config.data.sequence_length
    ):
        raise M10FullSFTError("M10 config and frozen mixture identity differ")
    try:
        resolved = resolve_model(artifact_root, config.model.parent_model_ref)
    except DeploymentError as exc:
        raise M10FullSFTError("M10 Production parent cannot be resolved") from exc
    validate_m10_parent(config, resolved)
    return config, resolved, manifest_sha256


def load_m10_continuation_gate(
    path: Path,
    *,
    run_id: str,
    config_sha256: str,
    source_stage_export_sha256: str,
) -> tuple[M10ContinuationGate, str]:
    """Verify immutable evidence before allowing the 5M-to-10M transition."""

    try:
        payload = path.read_bytes()
        gate = M10ContinuationGate.model_validate_json(payload)
    except (OSError, ValidationError, ValueError) as exc:
        raise M10FullSFTError("M10 continuation gate is missing or invalid") from exc
    if gate.decision != "accepted":
        raise M10FullSFTError("M10 5M-to-10M continuation gate was rejected")
    identities = (
        (gate.run_id, run_id),
        (gate.config_sha256, config_sha256),
        (gate.source_stage_export_sha256, source_stage_export_sha256),
    )
    if any(actual != expected for actual, expected in identities):
        raise M10FullSFTError("M10 continuation gate lineage differs from the 5M stage")
    return gate, hashlib.sha256(_json_bytes(gate.to_dict())).hexdigest()


def _record_result(artifact_dir: Path, result: M10FullSFTRunResult) -> None:
    attempt = f"{result.mode}-{result.status}-tokens-{result.supervised_tokens:010d}.json"
    _atomic_json(artifact_dir / "attempts" / attempt, result.to_dict())
    _atomic_json(artifact_dir / "result.json", result.to_dict())
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": f"m10_{result.status}",
            "mode": result.mode,
            "supervised_tokens": result.supervised_tokens,
            "attempt_result": f"attempts/{attempt}",
        },
    )
