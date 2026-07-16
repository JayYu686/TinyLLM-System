"""Atomic full-state checkpoint storage for M1 single-device training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tinyllm.data import SamplerState, StatefulSampler, StatefulSequentialSampler
from tinyllm.schemas import (
    CheckpointCommitMarker,
    CheckpointFile,
    CheckpointManifest,
    CheckpointStateCoverage,
    canonical_config_hash,
)
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.metrics import TrainerState

CHECKPOINT_ID_PREFIX = "checkpoint-step-"
CHECKPOINT_ID_PATTERN = re.compile(r"^checkpoint-step-\d{8}$")
MANIFEST_FILENAME = "manifest.json"
COMMIT_MARKER_FILENAME = "COMMITTED"
LATEST_FILENAME = "LATEST"
TRAINING_STATE_FILENAME = "training_state.pt"
CONFIG_FILENAME = "config.resolved.json"
ENVIRONMENT_FILENAME = "environment.json"


class StatefulScaler(Protocol):
    """Minimal GradScaler interface needed by the checkpoint writer."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable scaler state."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore serializable scaler state."""


class CheckpointErrorCode(StrEnum):
    """Stable checkpoint storage failure categories mapped to CLI exit code 5."""

    CHECKPOINT_EXISTS = "CHECKPOINT_EXISTS"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_WRITE_FAILED = "CHECKPOINT_WRITE_FAILED"
    CHECKPOINT_INCOMPLETE = "CHECKPOINT_INCOMPLETE"
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    CHECKPOINT_NO_VALID = "CHECKPOINT_NO_VALID"
    CHECKPOINT_RETENTION_FAILED = "CHECKPOINT_RETENTION_FAILED"


class CheckpointError(RuntimeError):
    """Checkpoint failure with a stable code and sanitized context."""

    def __init__(
        self,
        code: CheckpointErrorCode,
        message: str,
        *,
        context: dict[str, bool | int | str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True, slots=True)
class CheckpointContext:
    """Lineage and environment identity attached to every saved checkpoint."""

    run_id: str
    dataset_version: str
    git_commit: str
    environment: dict[str, object]
    strategy: Literal["single", "ddp", "fsdp2", "zero3"] = "single"
    world_size: int = 1


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    """Latest structurally valid checkpoint and any newer invalid candidates."""

    checkpoint_id: str
    skipped_invalid_checkpoints: tuple[str, ...]


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


def _checkpoint_id(global_step: int) -> str:
    if not 0 <= global_step <= 99_999_999:
        raise ValueError("checkpoint global_step must fit eight decimal digits")
    return f"{CHECKPOINT_ID_PREFIX}{global_step:08d}"


def capture_rng_state() -> dict[str, object]:
    """Capture all RNG families required by the M1 Exact Resume contract."""

    cuda_state: object
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = {"not_applicable": True}
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


class CheckpointStore:
    """Publish, validate, and retain integrity-checked checkpoint directories."""

    def __init__(self, root: Path, *, keep_last: int) -> None:
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.root = root
        self.keep_last = keep_last

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: StatefulScaler | None,
        sampler: StatefulSampler,
        trainer_state: TrainerState,
        config: M1TrainingConfig,
        context: CheckpointContext,
        pin_reason: Literal["interruption", "best", "final"] | None = None,
        created_at: datetime | None = None,
    ) -> CheckpointManifest:
        """Write full state to a temporary directory and publish it atomically."""

        if not isinstance(sampler, StatefulSequentialSampler):
            raise ValueError("single-device Checkpoint requires a sequential stateful sampler")

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
            payload = {
                "schema_version": "1.0",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
                "grad_scaler": (
                    scaler.state_dict() if scaler is not None else {"not_applicable": True}
                ),
                "trainer_state": trainer_state.to_dict(),
                "rng": capture_rng_state(),
                "sampler": sampler.state_dict(),
                "config": config.to_dict(),
                "config_hash": config_hash,
                "dataset_version": context.dataset_version,
                "git_commit": context.git_commit,
                "environment": dict(context.environment),
                "strategy": context.strategy,
                "world_size": context.world_size,
            }
            training_state_path = temporary / TRAINING_STATE_FILENAME
            torch.save(payload, training_state_path)
            _fsync_file(training_state_path)

            config_path = temporary / CONFIG_FILENAME
            _write_durable(config_path, _json_bytes(config.to_dict()))
            environment_path = temporary / ENVIRONMENT_FILENAME
            _write_durable(environment_path, _json_bytes(context.environment))

            files = tuple(
                CheckpointFile(
                    path=path.name,
                    role="training_state" if path.name == TRAINING_STATE_FILENAME else "metadata",
                    size_bytes=path.stat().st_size,
                    sha256=_sha256_file(path),
                )
                for path in (training_state_path, config_path, environment_path)
            )
            state_coverage = CheckpointStateCoverage(
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
            )
            manifest = CheckpointManifest(
                checkpoint_id=checkpoint_id,
                run_id=context.run_id,
                created_at=created_at or datetime.now(UTC),
                strategy=context.strategy,
                resume_capability="exact",
                world_size=context.world_size,
                global_step=trainer_state.global_step,
                micro_step=trainer_state.micro_step,
                epoch=trainer_state.epoch,
                config_hash=config_hash,
                dataset_version=context.dataset_version,
                git_commit=context.git_commit,
                state=state_coverage,
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
            self.validate(checkpoint_id)
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
                "failed to publish checkpoint",
                context={"checkpoint_id": checkpoint_id, "cause": type(exc).__name__},
            ) from exc

    def validate(self, checkpoint_id: str) -> CheckpointManifest:
        """Validate structure, manifest identity, file sizes, and every SHA256."""

        if CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id) is None:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_NOT_FOUND,
                "invalid checkpoint identifier",
                context={"checkpoint_id": checkpoint_id},
            )
        directory = self.root / checkpoint_id
        if not directory.is_dir():
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_NOT_FOUND,
                "checkpoint directory does not exist",
                context={"checkpoint_id": checkpoint_id},
            )
        manifest_path = directory / MANIFEST_FILENAME
        marker_path = directory / COMMIT_MARKER_FILENAME
        if not manifest_path.is_file() or not marker_path.is_file():
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
                "checkpoint is missing manifest or completion marker",
                context={"checkpoint_id": checkpoint_id},
            )
        try:
            manifest_bytes = manifest_path.read_bytes()
            marker = CheckpointCommitMarker.model_validate_json(
                marker_path.read_text(encoding="utf-8")
            )
            manifest = CheckpointManifest.model_validate_json(manifest_bytes)
        except (OSError, ValidationError, ValueError) as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint metadata cannot be parsed",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        if marker.checkpoint_id != checkpoint_id or manifest.checkpoint_id != checkpoint_id:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint directory identity does not match metadata",
                context={"checkpoint_id": checkpoint_id},
            )
        if marker.manifest_sha256 != _sha256_bytes(manifest_bytes):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest hash does not match completion marker",
                context={"checkpoint_id": checkpoint_id},
            )

        expected = {MANIFEST_FILENAME, COMMIT_MARKER_FILENAME}
        expected.update(entry.path for entry in manifest.files)
        actual: set[str] = set()
        for path in directory.rglob("*"):
            if path.is_symlink() or not path.is_file():
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_CORRUPT,
                    "checkpoint contains a non-regular file",
                    context={"checkpoint_id": checkpoint_id},
                )
            actual.add(path.relative_to(directory).as_posix())
        if actual != expected:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint file set does not match manifest",
                context={"checkpoint_id": checkpoint_id},
            )
        for entry in manifest.files:
            path = directory / entry.path
            if path.stat().st_size != entry.size_bytes or _sha256_file(path) != entry.sha256:
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_CORRUPT,
                    "checkpoint payload failed size or SHA256 validation",
                    context={"checkpoint_id": checkpoint_id, "file": entry.path},
                )
        try:
            resolved_config = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
            json.loads((directory / ENVIRONMENT_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint metadata snapshot cannot be parsed",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        if canonical_config_hash(resolved_config) != manifest.config_hash:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint config snapshot does not match manifest hash",
                context={"checkpoint_id": checkpoint_id},
            )
        return manifest

    def load_training_state(
        self,
        checkpoint_id: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a previously validated project-created PyTorch training payload."""

        manifest = self.validate(checkpoint_id)
        try:
            payload = torch.load(
                self.root / checkpoint_id / TRAINING_STATE_FILENAME,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint training payload cannot be loaded",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        required_keys = {
            "schema_version",
            "model",
            "optimizer",
            "scheduler",
            "grad_scaler",
            "trainer_state",
            "rng",
            "sampler",
            "config",
            "config_hash",
            "dataset_version",
            "git_commit",
            "environment",
            "strategy",
            "world_size",
        }
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "1.0"
            or set(payload) != required_keys
        ):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint training payload has an unsupported schema",
                context={"checkpoint_id": checkpoint_id},
            )
        typed_payload = cast(dict[str, Any], payload)
        try:
            resolved_environment = json.loads(
                (self.root / checkpoint_id / ENVIRONMENT_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint environment snapshot cannot be parsed",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        try:
            trainer_state = TrainerState.model_validate(typed_payload["trainer_state"])
            SamplerState.model_validate(typed_payload["sampler"])
            payload_config = M1TrainingConfig.model_validate(typed_payload["config"])
        except ValidationError as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint payload contains invalid structured state",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        rng = typed_payload["rng"]
        lineage_matches = (
            isinstance(rng, dict)
            and set(rng) == {"python", "numpy", "torch", "cuda"}
            and typed_payload["config_hash"] == manifest.config_hash
            and canonical_config_hash(payload_config) == manifest.config_hash
            and typed_payload["dataset_version"] == manifest.dataset_version
            and typed_payload["git_commit"] == manifest.git_commit
            and typed_payload["strategy"] == manifest.strategy
            and typed_payload["world_size"] == manifest.world_size
            and typed_payload["environment"] == resolved_environment
            and trainer_state.global_step == manifest.global_step
            and trainer_state.micro_step == manifest.micro_step
            and trainer_state.epoch == manifest.epoch
        )
        if not lineage_matches:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "checkpoint payload lineage does not match manifest",
                context={"checkpoint_id": checkpoint_id},
            )
        return typed_payload

    def latest_valid(self) -> CheckpointSelection:
        """Select the highest-Step payload that passes full integrity validation."""

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
                self.load_training_state(checkpoint_id)
            except CheckpointError:
                skipped.append(checkpoint_id)
                continue
            return CheckpointSelection(
                checkpoint_id=checkpoint_id,
                skipped_invalid_checkpoints=tuple(skipped),
            )
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_NO_VALID,
            "checkpoint store does not contain a valid checkpoint",
            context={"candidate_count": len(candidates)},
        )

    def latest(self) -> str:
        """Return and validate the atomically published latest checkpoint ID."""

        path = self.root / LATEST_FILENAME
        try:
            checkpoint_id = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_NOT_FOUND,
                "LATEST checkpoint pointer does not exist",
            ) from exc
        self.validate(checkpoint_id)
        return checkpoint_id

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
                "failed to remove an expired checkpoint",
            ) from exc
