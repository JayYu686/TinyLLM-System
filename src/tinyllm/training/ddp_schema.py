"""Strict M3.1 DDP correctness evidence schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN

LOSS_REDUCTION_ATOL = 1e-12
GRADIENT_NORM_ATOL = 1e-6


class DDPPartitionEvidence(StrictSchema):
    """Privacy-safe identity of one DistributedSampler partition."""

    schema_version: Literal["1.0"] = "1.0"
    rank: int = Field(ge=0)
    sample_count: int = Field(gt=0)
    sample_ids_sha256: str = Field(pattern=SHA256_PATTERN)


class DDPCorrectnessSummary(StrictSchema):
    """Correctness facts shared by all ranks in one bounded DDP run."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"] = "pass"
    backend: Literal["gloo", "nccl"]
    world_size: int = Field(ge=1, le=10)
    global_batch_size: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    durable_metric_records: int = Field(gt=0)
    durable_writer_rank: Literal[0] = 0
    model_parameter_count: int = Field(gt=0)
    initial_parameter_sha256: str = Field(pattern=SHA256_PATTERN)
    final_parameter_sha256: str = Field(pattern=SHA256_PATTERN)
    sampler_num_samples: int = Field(gt=0)
    sampler_union_samples: int = Field(gt=0)
    sampler_no_overlap: Literal[True] = True
    partitions: tuple[DDPPartitionEvidence, ...]
    max_loss_reduction_abs_diff: float = Field(ge=0.0, allow_inf_nan=False)
    max_gradient_norm_abs_diff: float = Field(ge=0.0, allow_inf_nan=False)
    loss_reduction_atol: float = Field(
        default=LOSS_REDUCTION_ATOL,
        ge=0.0,
        allow_inf_nan=False,
    )
    gradient_norm_atol: float = Field(
        default=GRADIENT_NORM_ATOL,
        ge=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> DDPCorrectnessSummary:
        """Bind World Size, partitions, steps, and complete sampler coverage."""

        if len(self.partitions) != self.world_size:
            raise ValueError("partition count must equal world_size")
        if tuple(item.rank for item in self.partitions) != tuple(range(self.world_size)):
            raise ValueError("partition ranks must be contiguous and ordered")
        if self.sampler_union_samples != self.sampler_num_samples:
            raise ValueError("sampler partitions must cover the complete dataset")
        if sum(item.sample_count for item in self.partitions) != self.sampler_num_samples:
            raise ValueError("partition sample counts must equal sampler_num_samples")
        if self.durable_metric_records != self.optimizer_steps:
            raise ValueError("rank-zero durable metrics must contain exactly one row per step")
        if self.loss_reduction_atol != LOSS_REDUCTION_ATOL:
            raise ValueError("loss reduction tolerance is fixed by the M3.1 contract")
        if self.gradient_norm_atol != GRADIENT_NORM_ATOL:
            raise ValueError("gradient norm tolerance is fixed by the M3.1 contract")
        if self.max_loss_reduction_abs_diff > self.loss_reduction_atol:
            raise ValueError("loss reduction difference exceeds the fixed tolerance")
        if self.max_gradient_norm_abs_diff > self.gradient_norm_atol:
            raise ValueError("gradient norm difference exceeds the fixed tolerance")
        return self


class DDPTrainingResult(StrictSchema):
    """Rank-zero terminal result emitted by the M3.1 torchrun worker."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"] = "succeeded"
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    summary: DDPCorrectnessSummary

    @model_validator(mode="after")
    def validate_identity(self) -> DDPTrainingResult:
        """Keep the Run ID bound to the resolved configuration hash."""

        if not self.artifact_dir.is_absolute():
            raise ValueError("artifact_dir must be absolute")
        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:  # pragma: no cover - rejected by the field pattern first
            raise ValueError("run_id is invalid")
        if match.group("config_hash") != self.config_sha256[:8]:
            raise ValueError("run_id config hash does not match config_sha256")
        return self
