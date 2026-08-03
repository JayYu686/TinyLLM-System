"""Machine-readable evidence contract for the M5 failure-path acceptance."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema

M5_FAILURE_PATHS = (
    "cuda_oom",
    "non_finite",
    "corrupt_checkpoint",
    "disk_insufficient",
    "dataset_drift",
    "world_size_mismatch",
    "child_process_exit",
)
M5FailurePathName = Literal[
    "cuda_oom",
    "non_finite",
    "corrupt_checkpoint",
    "disk_insufficient",
    "dataset_drift",
    "world_size_mismatch",
    "child_process_exit",
]


class M5FailurePathCase(StrictSchema):
    """One deliberately injected, safely rejected failure."""

    name: M5FailurePathName
    status: Literal["rejected_as_expected"]
    injection_kind: Literal["safe_cpu_fault_injection"]
    observed_error: str = Field(min_length=1, max_length=500)


class M5FailurePathEvidence(StrictSchema):
    """Complete ordered M5 failure-path acceptance result."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed"]
    generated_at: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    model_generated: Literal[False]
    quality_metric: Literal[False]
    cases: tuple[
        M5FailurePathCase,
        M5FailurePathCase,
        M5FailurePathCase,
        M5FailurePathCase,
        M5FailurePathCase,
        M5FailurePathCase,
        M5FailurePathCase,
    ]

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> M5FailurePathEvidence:
        """Require exactly one passing case for every frozen failure path."""

        if tuple(item.name for item in self.cases) != M5_FAILURE_PATHS:
            raise ValueError("M5 failure-path evidence is incomplete or unordered")
        return self
