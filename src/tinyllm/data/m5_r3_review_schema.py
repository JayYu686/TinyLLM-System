"""Strict maintainer-review contracts for the passing M5.2-R3 P2 source pilot."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5_R3_P2_PUBLIC_RESULT_SHA256: Literal[
    "94b89e3a72af6edb0f8c50772f8b8d807bbc6fa8ccf70b44f610426817fb03f2"
] = "94b89e3a72af6edb0f8c50772f8b8d807bbc6fa8ccf70b44f610426817fb03f2"
M5_R3_P2_PRIVATE_RAW_SHA256: Literal[
    "2b693da607c880fa456b6701ad8bc2449ed1927d5b07aab33dc662ef2127a7b5"
] = "2b693da607c880fa456b6701ad8bc2449ed1927d5b07aab33dc662ef2127a7b5"

M5R3ReviewCriterion = Literal[
    "label_matches_direct_evidence",
    "rationale_supports_selected_label",
    "no_unsupported_or_alternative_claims",
]
M5R3ReviewerRole = Literal["codex_draft", "maintainer"]


class M5R3ContentCriterionJudgment(StrictSchema):
    """One fixed content-review criterion judgment."""

    criterion: M5R3ReviewCriterion
    passed: bool


class M5R3ContentReviewJudgment(StrictSchema):
    """Private item-level review record."""

    schema_version: Literal["1.0"]
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:r3p1-(config|log)-(en|zh)-\d{3}$")
    reviewer_role: M5R3ReviewerRole
    criteria: tuple[
        M5R3ContentCriterionJudgment,
        M5R3ContentCriterionJudgment,
        M5R3ContentCriterionJudgment,
    ]
    passed: bool
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("criteria", mode="before")
    @classmethod
    def normalize_criteria(cls, value: object) -> object:
        """Normalize JSON arrays before validating the frozen order."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_judgment(self) -> M5R3ContentReviewJudgment:
        """Require every fixed criterion exactly once and consistent accounting."""

        if tuple(item.criterion for item in self.criteria) != (
            "label_matches_direct_evidence",
            "rationale_supports_selected_label",
            "no_unsupported_or_alternative_claims",
        ) or self.passed != all(item.passed for item in self.criteria):
            raise ValueError("M5 R3 content-review judgment differs")
        return self


class M5R3ContentReviewResult(StrictSchema):
    """Path-free public summary of the private maintainer review."""

    schema_version: Literal["1.0"]
    review_version: Literal["m5-r3-p2-content-review-v1"]
    reviewed_at: datetime
    status: Literal["approved", "rejected"]
    reviewer_role: Literal["maintainer"]
    source_pilot_version: Literal["m5-r3-p2-fallback-isolated-v1"]
    source_public_result_sha256: Literal[
        "94b89e3a72af6edb0f8c50772f8b8d807bbc6fa8ccf70b44f610426817fb03f2"
    ]
    source_private_raw_sha256: Literal[
        "2b693da607c880fa456b6701ad8bc2449ed1927d5b07aab33dc662ef2127a7b5"
    ]
    source_samples_sha256: Literal[
        "2d73a4d62b657b98e9da2539d7cf10fdc8a2f3af369e4db4956348b5b79c3ea8"
    ]
    reviewed_items: Literal[33]
    passed_items: int = Field(ge=0, le=33)
    rejected_items: int = Field(ge=0, le=33)
    family_counts: dict[Literal["config", "log_diagnosis"], int]
    language_counts: dict[Literal["en", "zh"], int]
    private_judgments_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_source_expansion_authorized: bool
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False

    @field_validator("reviewed_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require an aware UTC review timestamp."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M5 R3 content-review timestamp must use UTC")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> M5R3ContentReviewResult:
        """Require full coverage and fail-closed expansion authorization."""

        approved = self.passed_items == self.reviewed_items
        if (
            self.passed_items + self.rejected_items != self.reviewed_items
            or self.family_counts != {"config": 17, "log_diagnosis": 16}
            or self.language_counts != {"en": 24, "zh": 9}
            or self.status != ("approved" if approved else "rejected")
            or self.formal_source_expansion_authorized != approved
        ):
            raise ValueError("M5 R3 content-review result accounting differs")
        return self
