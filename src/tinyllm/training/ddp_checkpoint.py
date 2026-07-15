"""Atomic full-state DDP checkpoints with one integrity-checked state file per Rank."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tinyllm.data import DistributedSamplerState
from tinyllm.schemas import (
    CheckpointCommitMarker,
    CheckpointFile,
    CheckpointManifest,
    CheckpointStateCoverage,
    canonical_config_hash,
)
from tinyllm.training.checkpoint import (
    CHECKPOINT_ID_PATTERN,
    CHECKPOINT_ID_PREFIX,
    COMMIT_MARKER_FILENAME,
    CONFIG_FILENAME,
    ENVIRONMENT_FILENAME,
    LATEST_FILENAME,
    MANIFEST_FILENAME,
    TRAINING_STATE_FILENAME,
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointSelection,
    CheckpointStore,
)
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.metrics import TrainerState

RankState = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedDDPCheckpoint:
    """Validated shared and Rank-local state ready for Exact Resume preflight."""

    manifest: CheckpointManifest
    shared: dict[str, Any]
    rank: RankState


def _checkpoint_id(global_step: int) -> str:
    if not 0 <= global_step <= 99_999_999:
        raise ValueError("checkpoint global_step must fit eight decimal digits")
    return f"{CHECKPOINT_ID_PREFIX}{global_step:08d}"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_durable(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture_local_rng_state(device: torch.device) -> dict[str, object]:
    """Capture every RNG family plus only this Rank's selected CUDA device."""

    cuda_state: object
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    else:
        cuda_state = {"not_applicable": True}
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_local_rng_state(raw: object, *, device: torch.device) -> None:
    """Restore a validated local RNG payload after distributed synchronization."""

    python_state, numpy_state, torch_state, cuda_state = _validated_rng_state(
        raw,
        device=device,
    )
    random.setstate(python_state)
    numpy.random.set_state(numpy_state)
    torch.set_rng_state(torch_state.cpu())
    if device.type == "cuda":
        assert isinstance(cuda_state, torch.Tensor)
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


def validate_local_rng_state(raw: object, *, device: torch.device) -> None:
    """Validate a Rank-local RNG payload without changing process state."""

    python_state, numpy_state, torch_state, _ = _validated_rng_state(raw, device=device)
    try:
        local_python = random.Random()
        local_python.setstate(python_state)
        local_numpy = numpy.random.RandomState()
        local_numpy.set_state(numpy_state)
        torch.Generator().set_state(torch_state.cpu())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("DDP Rank RNG state cannot be restored") from exc


def _validated_rng_state(
    raw: object,
    *,
    device: torch.device | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], torch.Tensor, object]:
    if not isinstance(raw, dict) or set(raw) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("DDP Rank RNG state is incomplete")
    python_state = raw["python"]
    numpy_state = raw["numpy"]
    torch_state = raw["torch"]
    cuda_state = raw["cuda"]
    if not isinstance(python_state, tuple) or not isinstance(numpy_state, tuple):
        raise ValueError("DDP Rank Python or NumPy RNG state is invalid")
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("DDP Rank PyTorch RNG state is invalid")
    if not isinstance(cuda_state, torch.Tensor) and cuda_state != {"not_applicable": True}:
        raise ValueError("DDP Rank CUDA RNG state is invalid")
    if device is not None:
        if device.type == "cuda" and not isinstance(cuda_state, torch.Tensor):
            raise ValueError("CUDA DDP Rank is missing its local CUDA RNG state")
        if device.type != "cuda" and cuda_state != {"not_applicable": True}:
            raise ValueError("CPU DDP Rank cannot restore CUDA RNG state")
    return python_state, numpy_state, torch_state, cuda_state


def build_rank_state(
    *,
    rank: int,
    world_size: int,
    trainer_state: TrainerState,
    sampler_state: dict[str, Any],
    device: torch.device,
) -> RankState:
    """Capture one complete Rank-local Exact Resume state."""

    value: RankState = {
        "schema_version": "1.0",
        "rank": rank,
        "world_size": world_size,
        "trainer_state": trainer_state.to_dict(),
        "sampler": sampler_state,
        "rng": capture_local_rng_state(device),
    }
    _validate_rank_state(value, expected_rank=rank, expected_world_size=world_size)
    return value


def _validate_rank_state(
    raw: object,
    *,
    expected_rank: int,
    expected_world_size: int,
) -> RankState:
    required = {"schema_version", "rank", "world_size", "trainer_state", "sampler", "rng"}
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw.get("schema_version") != "1.0"
        or type(raw.get("rank")) is not int
        or type(raw.get("world_size")) is not int
        or raw.get("rank") != expected_rank
        or raw.get("world_size") != expected_world_size
    ):
        raise ValueError("DDP Rank state identity or schema is invalid")
    typed = cast(RankState, raw)
    try:
        trainer_state = TrainerState.model_validate(typed["trainer_state"])
        sampler_state = DistributedSamplerState.model_validate(typed["sampler"])
    except ValidationError as exc:
        raise ValueError("DDP Rank state contains invalid structured progress") from exc
    if sampler_state.rank != expected_rank or sampler_state.num_replicas != expected_world_size:
        raise ValueError("DDP Rank Sampler identity does not match the state file")
    if trainer_state.epoch != sampler_state.epoch:
        raise ValueError("DDP Rank Trainer and Sampler Epoch differ")
    rng = typed["rng"]
    _validated_rng_state(rng)
    return typed


class DDPCheckpointStore:
    """Publish and load full DDP state while keeping shared storage Rank-zero-only."""

    def __init__(self, root: Path, *, keep_last: int) -> None:
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.root = root
        self.keep_last = keep_last
        self._base = CheckpointStore(root, keep_last=keep_last)

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        trainer_state: TrainerState,
        config: M1TrainingConfig,
        context: CheckpointContext,
        rank_states: Sequence[RankState],
        pin_reason: Literal["interruption", "final"] | None = None,
        created_at: datetime | None = None,
    ) -> CheckpointManifest:
        """Atomically publish one complete shared state and all contiguous Rank states."""

        if context.strategy != "ddp" or context.world_size < 1:
            raise ValueError("DDP Checkpoint context must declare strategy=ddp and a World Size")
        if len(rank_states) != context.world_size:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
                "DDP Checkpoint is missing one or more Rank states",
                context={
                    "expected_world_size": context.world_size,
                    "rank_states": len(rank_states),
                },
            )
        validated_rank_states = tuple(
            _validate_rank_state(
                value,
                expected_rank=rank,
                expected_world_size=context.world_size,
            )
            for rank, value in enumerate(rank_states)
        )
        if any(
            TrainerState.model_validate(value["trainer_state"]) != trainer_state
            for value in validated_rank_states
        ):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "DDP Rank Trainer states differ at the Checkpoint boundary",
            )

        checkpoint_id = _checkpoint_id(trainer_state.global_step)
        destination = self.root / checkpoint_id
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_EXISTS,
                "checkpoint destination already exists",
                context={"checkpoint_id": checkpoint_id},
            )
        temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            config_hash = canonical_config_hash(config)
            shared = {
                "schema_version": "1.0",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
                "grad_scaler": {"not_applicable": True},
                "trainer_state": trainer_state.to_dict(),
                "config": config.to_dict(),
                "config_hash": config_hash,
                "dataset_version": context.dataset_version,
                "git_commit": context.git_commit,
                "environment": dict(context.environment),
                "strategy": "ddp",
                "world_size": context.world_size,
            }
            training_state_path = temporary / TRAINING_STATE_FILENAME
            torch.save(shared, training_state_path)
            _fsync_file(training_state_path)

            rank_paths: list[Path] = []
            for rank, rank_state in enumerate(validated_rank_states):
                rank_path = temporary / f"rank-{rank:05d}.pt"
                torch.save(rank_state, rank_path)
                _fsync_file(rank_path)
                rank_paths.append(rank_path)

            config_path = temporary / CONFIG_FILENAME
            _write_durable(config_path, _json_bytes(config.to_dict()))
            environment_path = temporary / ENVIRONMENT_FILENAME
            _write_durable(environment_path, _json_bytes(context.environment))

            files = (
                CheckpointFile(
                    path=training_state_path.name,
                    role="training_state",
                    size_bytes=training_state_path.stat().st_size,
                    sha256=_sha256_file(training_state_path),
                ),
                *(
                    CheckpointFile(
                        path=path.name,
                        role="rank_state",
                        size_bytes=path.stat().st_size,
                        sha256=_sha256_file(path),
                    )
                    for path in rank_paths
                ),
                CheckpointFile(
                    path=config_path.name,
                    role="metadata",
                    size_bytes=config_path.stat().st_size,
                    sha256=_sha256_file(config_path),
                ),
                CheckpointFile(
                    path=environment_path.name,
                    role="metadata",
                    size_bytes=environment_path.stat().st_size,
                    sha256=_sha256_file(environment_path),
                ),
            )
            manifest = CheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=context.run_id,
                created_at=created_at or datetime.now(UTC),
                strategy="ddp",
                resume_capability="exact",
                world_size=context.world_size,
                global_step=trainer_state.global_step,
                micro_step=trainer_state.micro_step,
                epoch=trainer_state.epoch,
                config_hash=config_hash,
                dataset_version=context.dataset_version,
                git_commit=context.git_commit,
                state=CheckpointStateCoverage(
                    model=True,
                    optimizer=True,
                    scheduler=True,
                    grad_scaler=True,
                    python_rng=True,
                    numpy_rng=True,
                    torch_rng=True,
                    cuda_rng=True,
                    sampler=True,
                    config_snapshot=True,
                    environment=True,
                ),
                files=files,
                pinned=pin_reason is not None,
                pin_reason=pin_reason,
            )
            manifest_bytes = _json_bytes(manifest.to_dict())
            _write_durable(temporary / MANIFEST_FILENAME, manifest_bytes)
            marker = CheckpointCommitMarker(
                checkpoint_id=checkpoint_id,
                manifest_sha256=_sha256_bytes(manifest_bytes),
            )
            _write_durable(temporary / COMMIT_MARKER_FILENAME, _json_bytes(marker.to_dict()))
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(self.root)
            self.validate(checkpoint_id, expected_world_size=context.world_size)
            self._update_latest(checkpoint_id)
            self._apply_retention()
            return manifest
        except CheckpointError:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination.exists() and not self._latest_points_to(checkpoint_id):
                shutil.rmtree(destination, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination.exists() and not self._latest_points_to(checkpoint_id):
                shutil.rmtree(destination, ignore_errors=True)
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_WRITE_FAILED,
                "failed to publish DDP Checkpoint",
                context={"checkpoint_id": checkpoint_id, "cause": type(exc).__name__},
            ) from exc

    def validate(
        self,
        checkpoint_id: str,
        *,
        expected_world_size: int | None = None,
    ) -> CheckpointManifest:
        """Validate generic integrity plus every shared and Rank-local DDP payload."""

        manifest = self._base.validate(checkpoint_id)
        if manifest.strategy != "ddp":
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint strategy is not DDP",
                context={"checkpoint_id": checkpoint_id},
            )
        if expected_world_size is not None and manifest.world_size != expected_world_size:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "DDP Exact Resume requires the original World Size",
                context={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_world_size": manifest.world_size,
                    "expected_world_size": expected_world_size,
                },
            )
        shared = self._load_shared(checkpoint_id, map_location="cpu")
        self._validate_shared(shared, manifest=manifest)
        try:
            resolved_environment = json.loads(
                (self.root / checkpoint_id / ENVIRONMENT_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP Checkpoint environment snapshot cannot be parsed",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        if shared["environment"] != resolved_environment:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared environment differs from its public snapshot",
                context={"checkpoint_id": checkpoint_id},
            )
        for rank in range(manifest.world_size):
            rank_state = self._load_rank(checkpoint_id, rank=rank, map_location="cpu")
            try:
                validated = _validate_rank_state(
                    rank_state,
                    expected_rank=rank,
                    expected_world_size=manifest.world_size,
                )
            except ValueError as exc:
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_CORRUPT,
                    "DDP Rank state payload is invalid",
                    context={"checkpoint_id": checkpoint_id, "rank": rank},
                ) from exc
            if TrainerState.model_validate(
                validated["trainer_state"]
            ) != TrainerState.model_validate(shared["trainer_state"]):
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_CORRUPT,
                    "DDP Rank progress differs from shared progress",
                    context={"checkpoint_id": checkpoint_id, "rank": rank},
                )
        return manifest

    def load(
        self,
        checkpoint_id: str,
        *,
        rank: int,
        expected_world_size: int,
        map_location: str | torch.device,
    ) -> LoadedDDPCheckpoint:
        """Load validated shared state and one Rank-local state onto the target device."""

        manifest = self.validate(
            checkpoint_id,
            expected_world_size=expected_world_size,
        )
        if not 0 <= rank < manifest.world_size:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "requested DDP Rank does not exist in the Checkpoint",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            )
        shared = self._load_shared(checkpoint_id, map_location=map_location)
        rank_state = self._load_rank(checkpoint_id, rank=rank, map_location=map_location)
        return LoadedDDPCheckpoint(manifest=manifest, shared=shared, rank=rank_state)

    def latest_valid(self, *, expected_world_size: int) -> CheckpointSelection:
        """Select the newest complete DDP point and report newer invalid candidates."""

        candidates = sorted(
            (
                path.name
                for path in self.root.glob(f"{CHECKPOINT_ID_PREFIX}*")
                if path.is_dir() and CHECKPOINT_ID_PATTERN.fullmatch(path.name) is not None
            ),
            reverse=True,
        )
        skipped: list[str] = []
        for checkpoint_id in candidates:
            try:
                self.validate(checkpoint_id, expected_world_size=expected_world_size)
            except CheckpointError:
                skipped.append(checkpoint_id)
                continue
            return CheckpointSelection(checkpoint_id, tuple(skipped))
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_NO_VALID,
            "DDP Checkpoint Store does not contain a valid compatible point",
            context={"candidate_count": len(candidates)},
        )

    def _load_shared(
        self,
        checkpoint_id: str,
        *,
        map_location: str | torch.device,
    ) -> dict[str, Any]:
        try:
            raw = torch.load(
                self.root / checkpoint_id / TRAINING_STATE_FILENAME,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared training state cannot be loaded",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        if not isinstance(raw, dict):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared training state is not an object",
                context={"checkpoint_id": checkpoint_id},
            )
        return cast(dict[str, Any], raw)

    def _load_rank(
        self,
        checkpoint_id: str,
        *,
        rank: int,
        map_location: str | torch.device,
    ) -> RankState:
        try:
            raw = torch.load(
                self.root / checkpoint_id / f"rank-{rank:05d}.pt",
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP Rank state cannot be loaded",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            ) from exc
        if not isinstance(raw, dict):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP Rank state is not an object",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            )
        return cast(RankState, raw)

    def _validate_shared(
        self,
        raw: dict[str, Any],
        *,
        manifest: CheckpointManifest,
    ) -> None:
        required = {
            "schema_version",
            "model",
            "optimizer",
            "scheduler",
            "grad_scaler",
            "trainer_state",
            "config",
            "config_hash",
            "dataset_version",
            "git_commit",
            "environment",
            "strategy",
            "world_size",
        }
        if raw.get("schema_version") != "1.0" or set(raw) != required:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared state has an unsupported schema",
                context={"checkpoint_id": manifest.checkpoint_id},
            )
        if (
            not isinstance(raw["model"], dict)
            or not raw["model"]
            or any(
                not isinstance(key, str) or not isinstance(value, torch.Tensor)
                for key, value in raw["model"].items()
            )
            or not isinstance(raw["optimizer"], dict)
            or set(raw["optimizer"]) != {"state", "param_groups"}
            or not isinstance(raw["scheduler"], dict)
            or not isinstance(raw["environment"], dict)
        ):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared state contains invalid runtime payloads",
                context={"checkpoint_id": manifest.checkpoint_id},
            )
        try:
            trainer_state = TrainerState.model_validate(raw["trainer_state"])
            payload_config = M1TrainingConfig.model_validate(raw["config"])
        except ValidationError as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared state contains invalid structured data",
                context={"checkpoint_id": manifest.checkpoint_id},
            ) from exc
        matches = (
            raw["grad_scaler"] == {"not_applicable": True}
            and raw["config_hash"] == manifest.config_hash
            and canonical_config_hash(payload_config) == manifest.config_hash
            and raw["dataset_version"] == manifest.dataset_version
            and raw["git_commit"] == manifest.git_commit
            and raw["strategy"] == manifest.strategy
            and raw["world_size"] == manifest.world_size
            and trainer_state.global_step == manifest.global_step
            and trainer_state.micro_step == manifest.micro_step
            and trainer_state.epoch == manifest.epoch
        )
        if not matches:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "DDP shared state lineage differs from its Manifest",
                context={"checkpoint_id": manifest.checkpoint_id},
            )

    def _update_latest(self, checkpoint_id: str) -> None:
        temporary = self.root / f".{LATEST_FILENAME}.tmp-{uuid.uuid4().hex}"
        try:
            _write_durable(temporary, f"{checkpoint_id}\n".encode())
            os.replace(temporary, self.root / LATEST_FILENAME)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def _latest_points_to(self, checkpoint_id: str) -> bool:
        try:
            return (self.root / LATEST_FILENAME).read_text(
                encoding="utf-8"
            ).strip() == checkpoint_id
        except OSError:
            return False

    def _apply_retention(self) -> None:
        unpinned: list[CheckpointManifest] = []
        for path in self.root.glob(f"{CHECKPOINT_ID_PREFIX}*"):
            if not path.is_dir():
                continue
            try:
                manifest = self.validate(path.name)
            except CheckpointError:
                continue
            if not manifest.pinned:
                unpinned.append(manifest)
        unpinned.sort(key=lambda manifest: manifest.global_step)
        try:
            for manifest in unpinned[: -self.keep_last]:
                shutil.rmtree(self.root / manifest.checkpoint_id)
            _fsync_directory(self.root)
        except OSError as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_RETENTION_FAILED,
                "failed to remove an expired DDP Checkpoint",
            ) from exc
