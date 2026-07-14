"""Unified checkpoint metadata shared by single, DDP, and FSDP2 strategies."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN


class CheckpointFile(StrictSchema):
    """One integrity-checked file in a committed checkpoint directory."""

    path: str
    role: Literal["training_state", "shard", "metadata", "rng", "sampler"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        """Keep manifest entries inside their checkpoint directory."""

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("checkpoint file path must be a safe relative POSIX path")
        return value


class CheckpointStateCoverage(StrictSchema):
    """Declare which runtime state categories are accounted for.

    A true value means the payload contains the state or an explicit not-applicable
    marker, such as GradScaler under BF16 or CUDA RNG in a CPU-only test.
    """

    model: bool
    optimizer: bool
    scheduler: bool
    grad_scaler: bool
    python_rng: bool
    numpy_rng: bool
    torch_rng: bool
    cuda_rng: bool
    sampler: bool
    config_snapshot: bool
    environment: bool

    @property
    def supports_exact_resume(self) -> bool:
        """Return whether every Exact Resume state category is present."""

        return all(self.model_dump().values())


class CheckpointManifest(StrictSchema):
    """Integrity and compatibility contract for a published checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-step-\d{8}$")
    run_id: str
    created_at: datetime
    strategy: Literal["single", "ddp", "fsdp2", "zero3"]
    resume_capability: Literal["exact", "warm", "transfer"]
    world_size: int = Field(ge=1)
    global_step: int = Field(ge=0)
    micro_step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    dataset_version: str
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    state: CheckpointStateCoverage
    files: tuple[CheckpointFile, ...] = Field(min_length=1)
    pinned: bool = False
    committed: Literal[True] = True

    @model_validator(mode="after")
    def validate_exact_resume_and_files(self) -> CheckpointManifest:
        """Reject incomplete Exact Resume claims and duplicate paths."""

        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:
            raise ValueError("invalid run_id")
        if match.group("config_hash") != self.config_hash[:8]:
            raise ValueError("run_id config hash does not match config_hash")
        if self.created_at.tzinfo is None:
            raise ValueError("checkpoint timestamp must be timezone-aware")
        if self.resume_capability == "exact" and not self.state.supports_exact_resume:
            raise ValueError("exact resume requires every state category")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("checkpoint manifest contains duplicate file paths")
        return self
