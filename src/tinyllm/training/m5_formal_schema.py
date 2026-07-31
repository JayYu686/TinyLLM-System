"""Strict Checkpoint and Run results for M5.3 four-GPU Full SFT."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5FormalCheckpointFile(StrictSchema):
    """One integrity-checked complete DDP training state."""

    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5FormalCheckpointManifest(StrictSchema):
    """Atomic exact-resume point at an optimizer boundary."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=180)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["ddp"] = "ddp"
    world_size: Literal[4] = 4
    global_step: int = Field(ge=0)
    local_sequence_cursor: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0, le=50_000_000)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: M5FormalCheckpointFile
    pinned: bool
    pin_reason: Literal["interruption", "evaluation", "final"] | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> M5FormalCheckpointManifest:
        """Bind checkpoint identity, progress, and retention pin."""

        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("formal M5 Checkpoint ID differs from Token progress")
        if self.pinned != (self.pin_reason is not None):
            raise ValueError("formal M5 Checkpoint pin flag and reason differ")
        if self.pin_reason == "final" and self.supervised_tokens != 50_000_000:
            raise ValueError("formal M5 final Checkpoint requires exactly 50M Tokens")
        return self


class M5FormalRankMemory(StrictSchema):
    """Peak CUDA memory from one training Rank."""

    rank: int = Field(ge=0, le=3)
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_memory(self) -> M5FormalRankMemory:
        """Reject inverted CUDA memory accounting."""

        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("formal M5 reserved memory cannot be below allocated memory")
        return self


class M5FormalRunResult(StrictSchema):
    """Path-free result for one fresh or Exact-Resume Full-SFT attempt."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "interrupted"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    thinking_fraction_basis_points: Literal[3000]
    seed: int = Field(ge=0, le=2**32 - 1)
    world_size: Literal[4]
    global_step: int = Field(gt=0)
    local_sequence_cursor: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0, le=50_000_000)
    initial_loss: float = Field(gt=0)
    final_loss: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    rank_memory: tuple[
        M5FormalRankMemory,
        M5FormalRankMemory,
        M5FormalRankMemory,
        M5FormalRankMemory,
    ]
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    evaluation_checkpoints: tuple[str, ...]
    resumed_from_tokens: int | None = Field(default=None, ge=0, lt=50_000_000)
    export_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> M5FormalRunResult:
        """Bind ordered Ranks, attempt mode, completion, and exported model."""

        if tuple(item.rank for item in self.rank_memory) != (0, 1, 2, 3):
            raise ValueError("formal M5 Rank memory must be ordered 0–3")
        if len({item.physical_gpu_index for item in self.rank_memory}) != 4:
            raise ValueError("formal M5 DDP requires four distinct physical GPUs")
        if self.mode == "fresh" and self.resumed_from_tokens is not None:
            raise ValueError("fresh formal M5 run cannot claim resumed progress")
        if self.mode == "exact_resume" and self.resumed_from_tokens is None:
            raise ValueError("formal M5 Exact Resume requires source progress")
        if self.status == "succeeded":
            if self.supervised_tokens != 50_000_000 or self.export_sha256 is None:
                raise ValueError("successful formal M5 run requires 50M Tokens and export")
        elif self.supervised_tokens >= 50_000_000 or self.export_sha256 is not None:
            raise ValueError("interrupted formal M5 run cannot claim completion or export")
        return self
