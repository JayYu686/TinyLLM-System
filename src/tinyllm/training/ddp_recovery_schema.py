"""Strict result schema for one M3.2 DDP recovery worker invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN


class DDPRecoveryResult(StrictSchema):
    """Rank-zero result for one successful or intentionally interrupted phase."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "interrupted"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    backend: Literal["gloo", "nccl"]
    world_size: int = Field(ge=1, le=10)
    global_step: int = Field(ge=1)
    checkpoint_id: str = Field(pattern=r"^checkpoint-step-\d{8}$")
    model_parameter_sha256: str = Field(pattern=SHA256_PATTERN)
    resumed_from_step: int | None = Field(default=None, ge=1)
    durable_metric_records: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_identity_and_progress(self) -> DDPRecoveryResult:
        """Bind the Run ID, invocation mode, Checkpoint, and durable metrics."""

        if not self.artifact_dir.is_absolute():
            raise ValueError("artifact_dir must be absolute")
        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:  # pragma: no cover - field validation rejects this first
            raise ValueError("run_id is invalid")
        if match.group("config_hash") != self.config_sha256[:8]:
            raise ValueError("run_id config hash does not match config_sha256")
        if self.checkpoint_id != f"checkpoint-step-{self.global_step:08d}":
            raise ValueError("checkpoint_id must describe global_step")
        if self.durable_metric_records != self.global_step:
            raise ValueError("metrics must contain exactly one durable row per optimizer step")
        if self.mode == "fresh" and self.resumed_from_step is not None:
            raise ValueError("fresh phase cannot declare resumed_from_step")
        if self.mode == "exact_resume" and (
            self.resumed_from_step is None or self.resumed_from_step >= self.global_step
        ):
            raise ValueError("Exact Resume must advance beyond its source Checkpoint")
        return self
