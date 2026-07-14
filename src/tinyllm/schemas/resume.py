"""Versioned result schema for explicit checkpoint restore semantics."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import RUN_ID_PATTERN


class ResumeResult(StrictSchema):
    """Machine-readable account of state applied from one checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["exact", "warm", "transfer"]
    checkpoint_id: str = Field(pattern=r"^checkpoint-step-\d{8}$")
    source_run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    source_global_step: int = Field(ge=0)
    target_global_step: int = Field(ge=0)
    optimizer_restored: bool
    scheduler_restored: bool
    scaler_restored: bool
    sampler_restored: bool
    rng_restored: bool
    loaded_model_keys: tuple[str, ...]
    missing_model_keys: tuple[str, ...] = ()
    unexpected_checkpoint_keys: tuple[str, ...] = ()
    incompatible_checkpoint_keys: tuple[str, ...] = ()
    skipped_invalid_checkpoints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_mode_semantics(self) -> ResumeResult:
        """Prevent partial restore results from being labelled Exact or Warm."""

        if not self.loaded_model_keys:
            raise ValueError("resume must load at least one model key")
        partial_keys = (
            self.missing_model_keys
            or self.unexpected_checkpoint_keys
            or self.incompatible_checkpoint_keys
        )
        if self.mode in {"exact", "warm"} and partial_keys:
            raise ValueError("exact and warm resume require a complete model state")
        full_runtime = (
            self.optimizer_restored,
            self.scheduler_restored,
            self.scaler_restored,
            self.sampler_restored,
            self.rng_restored,
        )
        if self.mode == "exact":
            if not all(full_runtime) or self.target_global_step != self.source_global_step:
                raise ValueError("exact resume requires complete runtime state")
        elif any(full_runtime) or self.target_global_step != 0:
            raise ValueError("warm and transfer resume must reset runtime state")
        return self
