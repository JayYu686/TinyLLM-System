"""Atomic PyTorch DCP sharded checkpoints and same-World-Size FSDP2 restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from pydantic import ValidationError
from torch import distributed as dist
from torch import nn
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tinyllm.data import StatefulDistributedSampler
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
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointSelection,
    CheckpointStore,
)
from tinyllm.training.ddp_checkpoint import (
    _validate_rank_state,
    build_rank_state,
    restore_local_rng_state,
    validate_local_rng_state,
)
from tinyllm.training.metrics import TrainerState

RUNTIME_STATE_FILENAME = "runtime_state.pt"
DCP_METADATA_FILENAME = ".metadata"


class CheckpointableFSDP2Config(Protocol):
    """Structural config boundary shared by TinyGPT and formal Qwen FSDP2 runs."""

    def to_dict(self) -> dict[str, Any]:
        """Return the complete resolved config as canonical JSON values."""


@dataclass(frozen=True, slots=True)
class LoadedFSDP2Checkpoint:
    """Validated non-sharded and Rank-local state for one restore phase."""

    manifest: CheckpointManifest
    runtime: dict[str, Any]
    rank: dict[str, Any]


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


def _broadcast_rank_zero(value: object, *, rank: int) -> object:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


class FSDP2CheckpointStore:
    """Collectively save DCP shards and publish only a fully durable checkpoint."""

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
        sampler: StatefulDistributedSampler,
        device: torch.device,
        config: CheckpointableFSDP2Config,
        context: CheckpointContext,
        rank: int,
        pin_reason: Literal["interruption", "final"] | None = None,
        created_at: datetime | None = None,
    ) -> CheckpointManifest:
        """Collectively write DCP, Rank states, and atomically commit on Rank zero."""

        if context.strategy != "fsdp2" or context.world_size != dist.get_world_size():
            raise ValueError("FSDP2 Checkpoint context must match the process group")
        if rank != dist.get_rank() or sampler.rank != rank:
            raise ValueError("FSDP2 Checkpoint Rank identity is inconsistent")
        checkpoint_id = _checkpoint_id(trainer_state.global_step)
        destination = self.root / checkpoint_id

        setup: object = None
        if rank == 0:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise CheckpointError(
                        CheckpointErrorCode.CHECKPOINT_EXISTS,
                        "checkpoint destination already exists",
                        context={"checkpoint_id": checkpoint_id},
                    )
                temporary = self.root / f".{checkpoint_id}.tmp-{uuid.uuid4().hex}"
                temporary.mkdir()
                setup = {"ok": True, "temporary": str(temporary)}
            except Exception as exc:
                setup = {"ok": False, "error_type": type(exc).__name__}
        setup = _broadcast_rank_zero(setup, rank=rank)
        if not isinstance(setup, dict) or not setup.get("ok"):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_WRITE_FAILED,
                "Rank zero could not create the FSDP2 temporary Checkpoint",
                context={"checkpoint_id": checkpoint_id},
            )
        temporary = Path(str(setup["temporary"]))
        dist.barrier()

        try:
            model_state, optimizer_state = get_state_dict(model, optimizer)
            dcp_save(
                {"model": model_state, "optimizer": optimizer_state},
                storage_writer=FileSystemWriter(
                    temporary,
                    single_file_per_rank=True,
                    sync_files=True,
                    overwrite=True,
                ),
            )
            rank_state = build_rank_state(
                rank=rank,
                world_size=context.world_size,
                trainer_state=trainer_state,
                sampler_state=sampler.state_dict(),
                device=device,
            )
            rank_path = temporary / f"rank-{rank:05d}.pt"
            torch.save(rank_state, rank_path)
            _fsync_file(rank_path)
            if rank == 0:
                runtime = {
                    "schema_version": "1.0",
                    "scheduler": scheduler.state_dict(),  # type: ignore[no-untyped-call]
                    "grad_scaler": {"not_applicable": True},
                    "trainer_state": trainer_state.to_dict(),
                    "config": config.to_dict(),
                    "config_hash": canonical_config_hash(config),
                    "dataset_version": context.dataset_version,
                    "git_commit": context.git_commit,
                    "environment": dict(context.environment),
                    "strategy": "fsdp2",
                    "world_size": context.world_size,
                    "precision": cast(dict[str, object], config.to_dict()["precision"]),
                }
                runtime_path = temporary / RUNTIME_STATE_FILENAME
                torch.save(runtime, runtime_path)
                _fsync_file(runtime_path)
                _write_durable(temporary / CONFIG_FILENAME, _json_bytes(config.to_dict()))
                _write_durable(temporary / ENVIRONMENT_FILENAME, _json_bytes(context.environment))
            dist.barrier()

            publication: object = None
            if rank == 0:
                try:
                    manifest = self._publish(
                        temporary=temporary,
                        destination=destination,
                        checkpoint_id=checkpoint_id,
                        trainer_state=trainer_state,
                        config=config,
                        context=context,
                        pin_reason=pin_reason,
                        created_at=created_at,
                    )
                    publication = {"ok": True, "manifest_json": manifest.model_dump_json()}
                except Exception as exc:
                    shutil.rmtree(temporary, ignore_errors=True)
                    publication = {"ok": False, "error_type": type(exc).__name__}
            publication = _broadcast_rank_zero(publication, rank=rank)
            if not isinstance(publication, dict) or not publication.get("ok"):
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_WRITE_FAILED,
                    "Rank zero failed to atomically publish the FSDP2 Checkpoint",
                    context={"checkpoint_id": checkpoint_id},
                )
            return CheckpointManifest.model_validate_json(str(publication["manifest_json"]))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_WRITE_FAILED,
                "failed to write the collective FSDP2 Checkpoint",
                context={
                    "checkpoint_id": checkpoint_id,
                    "rank": rank,
                    "cause": type(exc).__name__,
                },
            ) from exc

    def _publish(
        self,
        *,
        temporary: Path,
        destination: Path,
        checkpoint_id: str,
        trainer_state: TrainerState,
        config: CheckpointableFSDP2Config,
        context: CheckpointContext,
        pin_reason: Literal["interruption", "final"] | None,
        created_at: datetime | None,
    ) -> CheckpointManifest:
        files: list[CheckpointFile] = []
        rank_paths = {f"rank-{rank:05d}.pt" for rank in range(context.world_size)}
        actual_rank_paths = {path.name for path in temporary.glob("rank-*.pt")}
        if actual_rank_paths != rank_paths:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
                "FSDP2 Checkpoint is missing one or more Rank states",
                context={"checkpoint_id": checkpoint_id},
            )
        if not (temporary / DCP_METADATA_FILENAME).is_file():
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
                "FSDP2 Checkpoint is missing DCP metadata",
                context={"checkpoint_id": checkpoint_id},
            )
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            if relative.endswith(".distcp"):
                role: Literal["training_state", "rank_state", "shard", "metadata"] = "shard"
            elif relative in rank_paths:
                role = "rank_state"
            elif relative == RUNTIME_STATE_FILENAME:
                role = "training_state"
            else:
                role = "metadata"
            _fsync_file(path)
            files.append(
                CheckpointFile(
                    path=relative,
                    role=role,
                    size_bytes=path.stat().st_size,
                    sha256=_sha256_file(path),
                )
            )
        if not any(item.role == "shard" for item in files):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
                "FSDP2 Checkpoint does not contain a DCP shard",
            )
        manifest = CheckpointManifest(
            checkpoint_id=checkpoint_id,
            run_id=context.run_id,
            created_at=created_at or datetime.now(UTC),
            strategy="fsdp2",
            resume_capability="exact",
            world_size=context.world_size,
            global_step=trainer_state.global_step,
            micro_step=trainer_state.micro_step,
            epoch=trainer_state.epoch,
            config_hash=canonical_config_hash(config),
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
            files=tuple(files),
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

    def validate(
        self,
        checkpoint_id: str,
        *,
        expected_world_size: int | None = None,
    ) -> CheckpointManifest:
        """Validate generic integrity, DCP metadata, lineage, and every Rank state."""

        manifest = self._base.validate(checkpoint_id)
        if manifest.strategy != "fsdp2":
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint strategy is not FSDP2",
                context={"checkpoint_id": checkpoint_id},
            )
        if expected_world_size is not None and manifest.world_size != expected_world_size:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "FSDP2 Exact Resume requires the original World Size",
                context={
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_world_size": manifest.world_size,
                    "expected_world_size": expected_world_size,
                },
            )
        directory = self.root / checkpoint_id
        try:
            FileSystemReader(directory).read_metadata()
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 DCP metadata cannot be parsed",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        runtime = self._load_runtime(checkpoint_id)
        self._validate_runtime(runtime, manifest=manifest)
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
                    "FSDP2 Rank state payload is invalid",
                    context={"checkpoint_id": checkpoint_id, "rank": rank},
                ) from exc
            rank_progress = TrainerState.model_validate(validated["trainer_state"])
            runtime_progress = TrainerState.model_validate(runtime["trainer_state"])
            if rank_progress != runtime_progress:
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_CORRUPT,
                    "FSDP2 Rank progress differs from shared runtime progress",
                    context={"checkpoint_id": checkpoint_id, "rank": rank},
                )
        return manifest

    def load_local_state(
        self,
        checkpoint_id: str,
        *,
        rank: int,
        expected_world_size: int,
        map_location: str | torch.device,
    ) -> LoadedFSDP2Checkpoint:
        """Load validated runtime and one Rank-local payload without loading DCP tensors."""

        manifest = self.validate(checkpoint_id, expected_world_size=expected_world_size)
        if not 0 <= rank < manifest.world_size:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "requested FSDP2 Rank is outside the Checkpoint World Size",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            )
        return LoadedFSDP2Checkpoint(
            manifest=manifest,
            runtime=self._load_runtime(checkpoint_id, map_location=map_location),
            rank=self._load_rank(checkpoint_id, rank=rank, map_location=map_location),
        )

    def restore(
        self,
        *,
        checkpoint_id: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        sampler: StatefulDistributedSampler,
        device: torch.device,
        config: CheckpointableFSDP2Config,
        context: CheckpointContext,
        rank: int,
    ) -> TrainerState:
        """Collectively restore a validated same-World-Size Exact Resume point."""

        if optimizer.state:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "FSDP2 Exact Resume requires a pristine optimizer",
                context={"checkpoint_id": checkpoint_id},
            )
        loaded = self.load_local_state(
            checkpoint_id,
            rank=rank,
            expected_world_size=context.world_size,
            map_location=device,
        )
        runtime = loaded.runtime
        comparisons = {
            "config": (canonical_config_hash(config), loaded.manifest.config_hash),
            "run_id": (context.run_id, loaded.manifest.run_id),
            "dataset_version": (context.dataset_version, loaded.manifest.dataset_version),
            "git_commit": (context.git_commit, loaded.manifest.git_commit),
            "environment": (context.environment, runtime["environment"]),
            "world_size": (context.world_size, loaded.manifest.world_size),
        }
        mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
        if mismatches:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "FSDP2 runtime lineage changed since the Checkpoint",
                context={"checkpoint_id": checkpoint_id, "reason": ",".join(mismatches)},
            )
        validate_local_rng_state(loaded.rank["rng"], device=device)
        try:
            model_state, optimizer_state = get_state_dict(model, optimizer)
            state: dict[str, Any] = {"model": model_state, "optimizer": optimizer_state}
            dcp_load(
                state,
                storage_reader=FileSystemReader(self.root / checkpoint_id),
            )
            set_state_dict(
                model,
                optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
            )
            scheduler.load_state_dict(runtime["scheduler"])
            sampler.load_state_dict(loaded.rank["sampler"])
            progress = TrainerState.model_validate(runtime["trainer_state"])
            if sampler.epoch != progress.epoch:
                raise ValueError("restored FSDP2 Sampler Epoch differs from Trainer progress")
            dist.barrier()
            restore_local_rng_state(loaded.rank["rng"], device=device)
            return progress
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "FSDP2 Checkpoint state could not be applied",
                context={
                    "checkpoint_id": checkpoint_id,
                    "rank": rank,
                    "cause": type(exc).__name__,
                },
            ) from exc

    def latest_valid(self, *, expected_world_size: int) -> CheckpointSelection:
        """Select the newest fully valid same-World-Size DCP checkpoint."""

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
            "FSDP2 Checkpoint Store does not contain a valid compatible point",
            context={"candidate_count": len(candidates)},
        )

    def _load_runtime(
        self,
        checkpoint_id: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        try:
            raw = torch.load(
                self.root / checkpoint_id / RUNTIME_STATE_FILENAME,
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 runtime state cannot be loaded",
                context={"checkpoint_id": checkpoint_id},
            ) from exc
        if not isinstance(raw, dict):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 runtime state is not an object",
                context={"checkpoint_id": checkpoint_id},
            )
        return cast(dict[str, Any], raw)

    def _load_rank(
        self,
        checkpoint_id: str,
        *,
        rank: int,
        map_location: str | torch.device,
    ) -> dict[str, Any]:
        try:
            raw = torch.load(
                self.root / checkpoint_id / f"rank-{rank:05d}.pt",
                map_location=map_location,
                weights_only=False,
            )
        except Exception as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 Rank state cannot be loaded",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            ) from exc
        if not isinstance(raw, dict):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 Rank state is not an object",
                context={"checkpoint_id": checkpoint_id, "rank": rank},
            )
        return cast(dict[str, Any], raw)

    def _validate_runtime(
        self,
        raw: dict[str, Any],
        *,
        manifest: CheckpointManifest,
    ) -> None:
        required = {
            "schema_version",
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
            "precision",
        }
        if raw.get("schema_version") != "1.0" or set(raw) != required:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 runtime state has an unsupported schema",
                context={"checkpoint_id": manifest.checkpoint_id},
            )
        try:
            progress = TrainerState.model_validate(raw["trainer_state"])
            if not isinstance(raw["config"], dict):
                raise ValueError("FSDP2 runtime config must be an object")
            payload_config = cast(dict[str, object], raw["config"])
            environment = json.loads(
                (self.root / manifest.checkpoint_id / ENVIRONMENT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 runtime state contains invalid structured data",
                context={"checkpoint_id": manifest.checkpoint_id},
            ) from exc
        matches = (
            isinstance(raw["scheduler"], dict)
            and raw["grad_scaler"] == {"not_applicable": True}
            and raw["config_hash"] == manifest.config_hash
            and canonical_config_hash(payload_config) == manifest.config_hash
            and raw["dataset_version"] == manifest.dataset_version
            and raw["git_commit"] == manifest.git_commit
            and raw["environment"] == environment
            and raw["strategy"] == "fsdp2"
            and raw["world_size"] == manifest.world_size
            and raw["precision"] == payload_config.get("precision")
            and progress.global_step == manifest.global_step
            and progress.micro_step == manifest.micro_step
            and progress.epoch == manifest.epoch
        )
        if not matches:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_CORRUPT,
                "FSDP2 runtime state lineage differs from its Manifest",
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
        unpinned.sort(key=lambda value: value.global_step)
        try:
            for manifest in unpinned[: -self.keep_last]:
                shutil.rmtree(self.root / manifest.checkpoint_id)
            _fsync_directory(self.root)
        except OSError as exc:
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_RETENTION_FAILED,
                "failed to remove an expired FSDP2 Checkpoint",
            ) from exc
