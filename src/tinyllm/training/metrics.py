"""Versioned metrics emitted by the M1 training loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field

from tinyllm.schemas.base import StrictSchema


class TrainerState(StrictSchema):
    """Serializable progress after the last successful operation."""

    schema_version: Literal["1.0"] = "1.0"
    global_step: int = Field(default=0, ge=0)
    micro_step: int = Field(default=0, ge=0)
    epoch: int = Field(default=0, ge=0)
    tokens_seen: int = Field(default=0, ge=0)


class TrainingStepMetrics(StrictSchema):
    """One successful optimizer-step event."""

    schema_version: Literal["1.0"] = "1.0"
    event: Literal["optimizer_step"] = "optimizer_step"
    global_step: int = Field(ge=1)
    micro_step: int = Field(ge=1)
    epoch: int = Field(ge=0)
    loss: float = Field(ge=0.0, allow_inf_nan=False)
    learning_rate: float = Field(ge=0.0, allow_inf_nan=False)
    gradient_norm: float = Field(ge=0.0, allow_inf_nan=False)
    gradient_clipped: bool
    tokens_seen: int = Field(ge=1)


class MetricSink(Protocol):
    """Consumer for validated training metrics."""

    def emit(self, metric: TrainingStepMetrics) -> None:
        """Consume one optimizer-step metric."""


class InMemoryMetricSink:
    """Collect validated metrics for correctness tests and short smoke runs."""

    def __init__(self) -> None:
        self._metrics: list[TrainingStepMetrics] = []

    def emit(self, metric: TrainingStepMetrics) -> None:
        """Append one immutable metric."""

        self._metrics.append(metric)

    @property
    def metrics(self) -> Sequence[TrainingStepMetrics]:
        """Return an immutable view of emitted metrics."""

        return tuple(self._metrics)
