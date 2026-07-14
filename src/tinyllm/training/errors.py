"""Stable training failures surfaced by the M1 runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

ErrorContextValue: TypeAlias = bool | float | int | str


class TrainingErrorCode(StrEnum):
    """Machine-readable M1 training failure categories."""

    TRAIN_OUTPUT_INVALID = "TRAIN_OUTPUT_INVALID"
    NON_FINITE_LOSS = "NON_FINITE_LOSS"
    NON_FINITE_GRADIENT = "NON_FINITE_GRADIENT"
    EMPTY_DATALOADER = "EMPTY_DATALOADER"
    UNSUPPORTED_PRECISION = "UNSUPPORTED_PRECISION"


class TrainingError(RuntimeError):
    """Training failure with a stable code and sanitized context."""

    def __init__(
        self,
        code: TrainingErrorCode,
        message: str,
        *,
        context: dict[str, ErrorContextValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})
