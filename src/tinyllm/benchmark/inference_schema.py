"""Strict M7 inference-benchmark schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema


class InferenceBenchmarkConfig(StrictSchema):
    """Frozen Direct/Gateway inference matrix policy."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str = Field(pattern=r"^m7-inference-[a-z0-9-]{1,64}$")
    concurrency: tuple[Literal[1, 4, 8, 16, 32], ...]
    input_tokens: tuple[Literal[128, 512, 1024], ...]
    output_tokens: tuple[Literal[128, 256], ...]
    warmup_requests: Literal[20]
    measured_requests: Literal[100]
    repeats: Literal[3]
    request_timeout_seconds: float = Field(gt=0, le=600)

    @field_validator("concurrency", "input_tokens", "output_tokens", mode="before")
    @classmethod
    def freeze(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_matrix(self) -> InferenceBenchmarkConfig:
        if self.concurrency != (1, 4, 8, 16, 32):
            raise ValueError("M7 concurrency matrix differs from the frozen policy")
        if self.input_tokens != (128, 512, 1024) or self.output_tokens != (128, 256):
            raise ValueError("M7 token matrix differs from the frozen policy")
        return self


class InferenceBenchmarkConfigError(RuntimeError):
    """Raised when the formal M7 benchmark policy is invalid."""


def load_inference_benchmark_config(path: Path) -> InferenceBenchmarkConfig:
    """Load a strict M7 benchmark policy from YAML."""

    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return InferenceBenchmarkConfig.model_validate(decoded)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise InferenceBenchmarkConfigError("M7 inference benchmark config is invalid") from exc


class InferenceRequestResult(StrictSchema):
    """Content-free request-level inference evidence."""

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(pattern=r"^m7req-[0-9]{6}$")
    backend: Literal["direct", "gateway"]
    concurrency: Literal[1, 4, 8, 16, 32]
    input_tokens_target: Literal[128, 512, 1024]
    output_tokens_target: Literal[128, 256]
    repeat: Literal[1, 2, 3]
    success: bool
    status_code: int = Field(ge=0, le=599)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    queue_milliseconds: float | None = Field(default=None, ge=0)
    ttft_milliseconds: float | None = Field(default=None, ge=0)
    tpot_milliseconds: float | None = Field(default=None, ge=0)
    latency_milliseconds: float = Field(ge=0)
    error_code: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_result(self) -> InferenceRequestResult:
        if self.success and not 200 <= self.status_code < 300:
            raise ValueError("successful request requires a successful HTTP status")
        if self.success and self.error_code is not None:
            raise ValueError("successful request cannot contain an error code")
        if not self.success and self.error_code is None:
            raise ValueError("failed request requires an error code")
        return self


class InferenceBenchmarkSummary(StrictSchema):
    """Path-free formal inference benchmark summary."""

    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str = Field(pattern=r"^m7-inference-[a-z0-9-]{1,64}$")
    completed_at: datetime
    model_version: str = Field(pattern=r"^qwen3-(0-6b|8b)-m[67]-[0-9a-f]{8}$")
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gateway_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_requests: int = Field(ge=1)
    successful_requests: int = Field(ge=0)
    success_rate_basis_points: int = Field(ge=0, le=10_000)
    gateway_throughput_ratio_basis_points: int = Field(ge=0)
    p95_latency_overhead_median_basis_points: int
    oom_events: int = Field(ge=0)
    hung_processes: int = Field(ge=0)
    unexplained_5xx: int = Field(ge=0)
    status: Literal["succeeded", "failed"]

    @model_validator(mode="after")
    def validate_summary(self) -> InferenceBenchmarkSummary:
        if self.completed_at.tzinfo is None:
            raise ValueError("benchmark completion time must be timezone-aware")
        if self.successful_requests > self.total_requests:
            raise ValueError("successful requests cannot exceed total requests")
        expected = self.successful_requests * 10_000 // self.total_requests
        if self.success_rate_basis_points != expected:
            raise ValueError("success rate differs from request counts")
        passed = (
            self.success_rate_basis_points >= 9950
            and self.unexplained_5xx == 0
            and self.oom_events == 0
            and self.hung_processes == 0
        )
        if self.status != ("succeeded" if passed else "failed"):
            raise ValueError("benchmark status differs from the frozen reliability thresholds")
        return self
