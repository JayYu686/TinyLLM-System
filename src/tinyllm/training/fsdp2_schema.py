"""Strict M4.1 FSDP2 correctness evidence schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN

FSDP2_LOSS_REDUCTION_ATOL = 1e-12
FSDP2_GRADIENT_NORM_ATOL = 1e-6


class FSDP2RankEvidence(StrictSchema):
    """Privacy-safe local-shard evidence for one FSDP2 Rank."""

    schema_version: Literal["1.0"] = "1.0"
    rank: int = Field(ge=0)
    device_type: Literal["cpu", "cuda"]
    parameters_are_dtensor: Literal[True] = True
    local_shard_numel: int = Field(gt=0)
    local_shard_sha256: str = Field(pattern=SHA256_PATTERN)


class FSDP2CorrectnessSummary(StrictSchema):
    """Correctness facts shared by all Ranks in one bounded FSDP2 run."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"] = "pass"
    backend: Literal["gloo", "nccl"]
    device_type: Literal["cpu", "cuda"]
    world_size: int = Field(ge=1, le=4)
    global_batch_size: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    durable_metric_records: int = Field(gt=0)
    durable_writer_rank: Literal[0] = 0
    logical_parameter_count: int = Field(gt=0)
    local_shard_parameter_sum: int = Field(gt=0)
    initial_full_parameter_sha256: str = Field(pattern=SHA256_PATTERN)
    final_full_parameter_sha256: str = Field(pattern=SHA256_PATTERN)
    reshard_after_forward: Literal[True] = True
    cpu_offload: Literal[False] = False
    activation_checkpointing: Literal[False] = False
    rank_evidence: tuple[FSDP2RankEvidence, ...]
    max_loss_reduction_abs_diff: float = Field(ge=0.0, allow_inf_nan=False)
    max_gradient_norm_abs_diff: float = Field(ge=0.0, allow_inf_nan=False)
    loss_reduction_atol: float = Field(
        default=FSDP2_LOSS_REDUCTION_ATOL,
        ge=0.0,
        allow_inf_nan=False,
    )
    gradient_norm_atol: float = Field(
        default=FSDP2_GRADIENT_NORM_ATOL,
        ge=0.0,
        allow_inf_nan=False,
    )
    checkpoint_status: Literal["not_evaluated_m4_1"] = "not_evaluated_m4_1"

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> FSDP2CorrectnessSummary:
        """Bind Rank order, shard coverage, metrics, and fixed tolerances."""

        if len(self.rank_evidence) != self.world_size:
            raise ValueError("rank evidence count must equal world_size")
        if tuple(item.rank for item in self.rank_evidence) != tuple(range(self.world_size)):
            raise ValueError("rank evidence must be contiguous and ordered")
        if any(item.device_type != self.device_type for item in self.rank_evidence):
            raise ValueError("rank evidence device type must match the summary")
        if sum(item.local_shard_numel for item in self.rank_evidence) != (
            self.local_shard_parameter_sum
        ):
            raise ValueError("rank shard counts must equal local_shard_parameter_sum")
        if self.local_shard_parameter_sum != self.logical_parameter_count:
            raise ValueError("local FSDP2 shards must cover each logical parameter exactly once")
        if self.durable_metric_records != self.optimizer_steps:
            raise ValueError("rank-zero durable metrics must contain exactly one row per step")
        if self.initial_full_parameter_sha256 == self.final_full_parameter_sha256:
            raise ValueError("optimizer steps must change the full model state")
        if self.loss_reduction_atol != FSDP2_LOSS_REDUCTION_ATOL:
            raise ValueError("loss reduction tolerance is fixed by the M4.1 contract")
        if self.gradient_norm_atol != FSDP2_GRADIENT_NORM_ATOL:
            raise ValueError("gradient norm tolerance is fixed by the M4.1 contract")
        if self.max_loss_reduction_abs_diff > self.loss_reduction_atol:
            raise ValueError("loss reduction difference exceeds the fixed tolerance")
        if self.max_gradient_norm_abs_diff > self.gradient_norm_atol:
            raise ValueError("gradient norm difference exceeds the fixed tolerance")
        return self


class FSDP2TrainingResult(StrictSchema):
    """Rank-zero terminal result emitted by the M4.1 torchrun worker."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"] = "succeeded"
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    summary: FSDP2CorrectnessSummary

    @model_validator(mode="after")
    def validate_identity(self) -> FSDP2TrainingResult:
        """Keep the Artifact path and Run ID bound to the resolved config."""

        if not self.artifact_dir.is_absolute():
            raise ValueError("artifact_dir must be absolute")
        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:  # pragma: no cover - rejected by the field pattern first
            raise ValueError("run_id is invalid")
        if match.group("config_hash") != self.config_sha256[:8]:
            raise ValueError("run_id config hash does not match config_sha256")
        return self
