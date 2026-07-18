"""Strict machine-readable results for the M4.3 Qwen3-8B four-GPU gate."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN


class M4QwenRankMemory(StrictSchema):
    """Peak CUDA memory and physical identity for one logical Rank."""

    rank: int = Field(ge=0, le=3)
    physical_gpu_index: int = Field(ge=0, le=9)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    final_allocated_bytes: int = Field(gt=0)
    final_reserved_bytes: int = Field(gt=0)


class M4QwenRunResult(StrictSchema):
    """Rank-zero result for a Probe, interruption, or completed M4.3 phase."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["probe_succeeded", "interrupted", "succeeded"]
    mode: Literal["probe", "fresh", "exact_resume"]
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    data_view_sha256: str = Field(pattern=SHA256_PATTERN)
    world_size: Literal[4]
    global_step: int = Field(ge=1, le=50)
    checkpoint_id: str | None = Field(
        default=None,
        pattern=r"^checkpoint-step-\d{8}$",
    )
    model_parameter_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    resumed_from_step: int | None = Field(default=None, ge=1, le=49)
    durable_metric_records: int = Field(ge=1, le=50)
    activation_checkpointed_layers: Literal[36]
    rank_memory: tuple[M4QwenRankMemory, M4QwenRankMemory, M4QwenRankMemory, M4QwenRankMemory]
    export_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_mode_and_artifacts(self) -> M4QwenRunResult:
        """Bind Probe and formal recovery claims to their required artifacts."""

        if not self.artifact_dir.is_absolute():
            raise ValueError("artifact_dir must be absolute")
        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:  # pragma: no cover
            raise ValueError("run_id is invalid")
        if match.group("config_hash") != self.config_sha256[:8]:
            raise ValueError("run_id config hash does not match config_sha256")
        if tuple(item.rank for item in self.rank_memory) != (0, 1, 2, 3):
            raise ValueError("rank_memory must contain contiguous Rank 0-3 evidence")
        if self.mode == "probe":
            if self.status != "probe_succeeded" or self.global_step != 1:
                raise ValueError("Probe must succeed after exactly one optimizer step")
            if self.checkpoint_id is not None or self.model_parameter_sha256 is not None:
                raise ValueError("Probe cannot claim a formal Checkpoint or final model hash")
            if self.resumed_from_step is not None or self.export_sha256 is not None:
                raise ValueError("Probe cannot claim Resume or deployment export")
        else:
            if self.checkpoint_id != f"checkpoint-step-{self.global_step:08d}":
                raise ValueError("formal phase Checkpoint must describe global_step")
            if self.model_parameter_sha256 is None:
                raise ValueError("formal phase must record a complete model hash")
            if self.durable_metric_records != self.global_step:
                raise ValueError("formal metrics must contain one row per optimizer step")
            if self.mode == "fresh" and self.resumed_from_step is not None:
                raise ValueError("fresh phase cannot declare resumed_from_step")
            if self.mode == "exact_resume" and (
                self.resumed_from_step is None or self.resumed_from_step >= self.global_step
            ):
                raise ValueError("Exact Resume must advance beyond its source Checkpoint")
        return self
