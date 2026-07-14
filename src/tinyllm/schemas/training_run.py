"""Stable public result emitted by ``tinyllm train``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import RUN_ID_PATTERN


class TrainingRunResult(StrictSchema):
    """Terminal state of one single-device CLI invocation."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "terminated"]
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    device: str
    global_step: int = Field(ge=0)
    checkpoint_id: str = Field(pattern=r"^checkpoint-step-\d{8}$")
    resume_mode: Literal["none", "exact", "warm", "transfer"]
    resumed_from_step: int | None = Field(default=None, ge=0)
    skipped_invalid_checkpoints: tuple[str, ...] = ()
