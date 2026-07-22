"""Strict result contracts for M5.2 Qwen3-0.6B ablation training."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5CheckpointFile(StrictSchema):
    """One integrity-checked file in an M5 single-GPU training Checkpoint."""

    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5CheckpointManifest(StrictSchema):
    """Atomic M5 exact-resume Checkpoint identity and progress."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=160)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["single"] = "single"
    world_size: Literal[1] = 1
    global_step: int = Field(ge=0)
    sequence_cursor: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0, le=1_000_000)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    mixture_version: str = Field(pattern=r"^m5-ablation-mixture-v1-[0-9a-f]{8}$")
    mixture_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: M5CheckpointFile
    pinned: bool
    pin_reason: Literal["interruption", "final"] | None = None

    @model_validator(mode="after")
    def validate_pin(self) -> M5CheckpointManifest:
        """Require explicit reasons for pinned interruption and final states."""

        if self.pinned != (self.pin_reason is not None):
            raise ValueError("M5 Checkpoint pin flag and reason differ")
        expected = f"checkpoint-tokens-{self.supervised_tokens:010d}"
        if self.checkpoint_id != expected:
            raise ValueError("M5 Checkpoint ID does not match supervised-token progress")
        return self


class M5AblationRunResult(StrictSchema):
    """Path-free summary of one real M5.2 Full-SFT ablation run."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "interrupted"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=160)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    mixture_version: str = Field(pattern=r"^m5-ablation-mixture-v1-[0-9a-f]{8}$")
    mixture_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    thinking_fraction_basis_points: Literal[0, 3000, 5000]
    seed: int = Field(ge=0, le=2**32 - 1)
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    global_step: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0, le=1_000_000)
    sequence_cursor: int = Field(gt=0)
    initial_loss: float = Field(gt=0.0)
    final_loss: float = Field(gt=0.0)
    duration_seconds: float = Field(gt=0.0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    resumed_from_tokens: int | None = Field(default=None, ge=0, lt=1_000_000)
    export_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_status(self) -> M5AblationRunResult:
        """Only a full 1M-token run may publish a candidate export."""

        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M5 reserved memory cannot be below allocated memory")
        if self.status == "succeeded":
            if self.supervised_tokens != 1_000_000 or self.export_sha256 is None:
                raise ValueError("successful M5.2 run requires 1M Tokens and an export")
        elif self.supervised_tokens >= 1_000_000 or self.export_sha256 is not None:
            raise ValueError("interrupted M5.2 run cannot claim completion or an export")
        if self.mode == "fresh" and self.resumed_from_tokens is not None:
            raise ValueError("fresh M5.2 run cannot claim resumed progress")
        if self.mode == "exact_resume" and self.resumed_from_tokens is None:
            raise ValueError("Exact Resume must record its source token count")
        return self
