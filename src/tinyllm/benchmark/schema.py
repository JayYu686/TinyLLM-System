"""Machine-readable schemas for DDP benchmark runs and matrix summaries."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN

BenchmarkProfile = Literal["strong", "weak"]
BenchmarkGroup = Literal["standard", "same_numa", "cross_numa"]


class BenchmarkTimingSummary(StrictSchema):
    """Distribution summary that always retains a bounded raw sample count."""

    count: int = Field(gt=0)
    total_ms: float = Field(gt=0, allow_inf_nan=False)
    min_ms: float = Field(gt=0, allow_inf_nan=False)
    median_ms: float = Field(gt=0, allow_inf_nan=False)
    p95_ms: float = Field(gt=0, allow_inf_nan=False)
    max_ms: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> BenchmarkTimingSummary:
        """Require monotonic finite timing statistics."""

        if not self.min_ms <= self.median_ms <= self.p95_ms <= self.max_ms:
            raise ValueError("timing percentiles must be monotonic")
        return self


class CommunicationMeasurement(StrictSchema):
    """Profiler-observed distributed communication for a bounded step window."""

    status: Literal["measured", "not_collected", "not_applicable", "unavailable"]
    profiled_steps: int = Field(ge=0)
    device_time_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    event_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> CommunicationMeasurement:
        """Prevent missing measurements from being represented as numeric zero."""

        if self.status == "measured":
            if self.profiled_steps <= 0 or self.device_time_ms is None or not self.event_keys:
                raise ValueError("measured communication requires steps, time, and event keys")
        elif self.device_time_ms is not None or self.event_keys:
            raise ValueError("non-measured communication cannot contain measured values")
        return self


class RankBenchmarkMetrics(StrictSchema):
    """Raw timing, memory, profiler, and hardware facts from one Rank."""

    rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    physical_gpu_index: int | None = Field(default=None, ge=0)
    gpu_name: str | None = None
    step_time_ms: tuple[float, ...]
    data_wait_ms: tuple[float, ...]
    peak_memory_allocated_bytes: int = Field(ge=0)
    communication: CommunicationMeasurement
    profiler_trace_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_raw_windows(self) -> RankBenchmarkMetrics:
        """Bind Rank-local step and data timing windows."""

        if not self.step_time_ms or len(self.step_time_ms) != len(self.data_wait_ms):
            raise ValueError("step and data timing windows must be non-empty and equal")
        if any(value <= 0 for value in self.step_time_ms):
            raise ValueError("step timings must be positive")
        if any(value < 0 for value in self.data_wait_ms):
            raise ValueError("data wait timings must be non-negative")
        if self.communication.profiled_steps and self.profiler_trace_sha256 is None:
            raise ValueError("profiled runs require a trace SHA256")
        return self


class DDPBenchmarkRunResult(StrictSchema):
    """Complete private result for one Profile, World Size, and independent repeat."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"] = "pass"
    run_id: str = Field(pattern=RUN_ID_PATTERN.pattern)
    artifact_dir: Path
    group: BenchmarkGroup
    profile: BenchmarkProfile
    world_size: int = Field(ge=1, le=10)
    repeat: int = Field(gt=0, le=10)
    seed: int = Field(ge=0)
    base_config_sha256: str = Field(pattern=SHA256_PATTERN)
    resolved_config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    started_at: datetime
    finished_at: datetime
    backend: Literal["gloo", "nccl"]
    precision: Literal["fp32", "bf16"]
    model_parameter_count: int = Field(gt=0)
    sequence_length: int = Field(ge=2)
    warmup_steps: int = Field(ge=0)
    measurement_steps: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    predicted_tokens_per_step: int = Field(gt=0)
    tokens_per_second: float = Field(gt=0, allow_inf_nan=False)
    samples_per_second: float = Field(gt=0, allow_inf_nan=False)
    effective_step_time: BenchmarkTimingSummary
    effective_data_wait: BenchmarkTimingSummary
    data_wait_percent: float = Field(ge=0, allow_inf_nan=False)
    peak_memory_allocated_bytes: int = Field(ge=0)
    rank_metrics: tuple[RankBenchmarkMetrics, ...]

    @model_validator(mode="after")
    def validate_complete_run(self) -> DDPBenchmarkRunResult:
        """Bind identity, Rank windows, batch semantics, and throughput arithmetic."""

        if not self.artifact_dir.is_absolute():
            raise ValueError("artifact_dir must be absolute")
        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None or match.group("config_hash") != self.resolved_config_sha256[:8]:
            raise ValueError("run_id must be bound to resolved_config_sha256")
        if self.finished_at <= self.started_at:
            raise ValueError("finished_at must follow started_at")
        if len(self.rank_metrics) != self.world_size:
            raise ValueError("rank_metrics count must equal world_size")
        if tuple(item.rank for item in self.rank_metrics) != tuple(range(self.world_size)):
            raise ValueError("rank_metrics must be rank ordered and contiguous")
        if any(len(item.step_time_ms) != self.measurement_steps for item in self.rank_metrics):
            raise ValueError("every Rank must retain all measured steps")
        expected_batch = self.micro_batch_size * self.gradient_accumulation_steps * self.world_size
        if self.global_batch_size != expected_batch:
            raise ValueError("global_batch_size arithmetic is invalid")
        if self.predicted_tokens_per_step != self.global_batch_size * (self.sequence_length - 1):
            raise ValueError("predicted_tokens_per_step arithmetic is invalid")
        if self.effective_step_time.count != self.measurement_steps:
            raise ValueError("effective step summary count must equal measurement_steps")
        if self.effective_data_wait.count != self.measurement_steps:
            raise ValueError("effective data summary count must equal measurement_steps")
        return self


class BenchmarkProfileAggregate(StrictSchema):
    """Three-repeat summary for one controlled matrix cell."""

    group: BenchmarkGroup
    profile: BenchmarkProfile
    world_size: int = Field(ge=1, le=10)
    gpu_indices: tuple[int, ...]
    repeats: tuple[Literal[1, 2, 3], ...]
    run_ids: tuple[str, ...]
    tokens_per_second_by_repeat: tuple[float, float, float]
    step_time_ms_by_repeat: tuple[float, float, float]
    peak_memory_bytes_by_repeat: tuple[int, int, int]
    data_wait_percent_by_repeat: tuple[float, float, float]
    tokens_per_second_median: float = Field(gt=0, allow_inf_nan=False)
    tokens_per_second_min: float = Field(gt=0, allow_inf_nan=False)
    tokens_per_second_max: float = Field(gt=0, allow_inf_nan=False)
    step_time_ms_median: float = Field(gt=0, allow_inf_nan=False)
    peak_memory_bytes_median: float = Field(ge=0, allow_inf_nan=False)
    data_wait_percent_median: float = Field(ge=0, allow_inf_nan=False)
    scaling_efficiency: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_cell(self) -> BenchmarkProfileAggregate:
        """Require exactly three ordered repeats on one fixed GPU set."""

        if self.repeats != (1, 2, 3) or len(self.run_ids) != 3:
            raise ValueError("formal aggregate requires repeats 1, 2, and 3")
        if any(
            not math.isfinite(value) or value <= 0 for value in self.tokens_per_second_by_repeat
        ):
            raise ValueError("repeat throughput must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.step_time_ms_by_repeat):
            raise ValueError("repeat step times must be positive")
        if any(value < 0 for value in self.peak_memory_bytes_by_repeat):
            raise ValueError("repeat peak memory must be non-negative")
        if any(not math.isfinite(value) or value < 0 for value in self.data_wait_percent_by_repeat):
            raise ValueError("repeat data wait percentages must be non-negative")
        if len(self.gpu_indices) != self.world_size:
            raise ValueError("GPU index count must equal world_size")
        if (
            not self.tokens_per_second_min
            <= self.tokens_per_second_median
            <= self.tokens_per_second_max
        ):
            raise ValueError("throughput range must contain the median")
        if self.tokens_per_second_min != min(self.tokens_per_second_by_repeat) or (
            self.tokens_per_second_max != max(self.tokens_per_second_by_repeat)
        ):
            raise ValueError("throughput range must match repeat measurements")
        expected_medians = (
            statistics.median(self.tokens_per_second_by_repeat),
            statistics.median(self.step_time_ms_by_repeat),
            statistics.median(self.peak_memory_bytes_by_repeat),
            statistics.median(self.data_wait_percent_by_repeat),
        )
        actual_medians = (
            self.tokens_per_second_median,
            self.step_time_ms_median,
            self.peak_memory_bytes_median,
            self.data_wait_percent_median,
        )
        if actual_medians != expected_medians:
            raise ValueError("aggregate medians must match repeat measurements")
        return self


class DDPBenchmarkMatrixSummary(StrictSchema):
    """Strict acceptance summary for the complete M3.3/M3.4 matrix."""

    schema_version: Literal["1.1"] = "1.1"
    status: Literal["pass"] = "pass"
    base_config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    model_parameter_count: int = Field(gt=0)
    acceptance_world_sizes: tuple[Literal[1, 2, 4], ...] = (1, 2, 4)
    eight_gpu_status: Literal["complete", "not_collected"]
    numa_comparison_status: Literal["complete", "partial", "not_collected"]
    standard: tuple[BenchmarkProfileAggregate, ...]
    numa: tuple[BenchmarkProfileAggregate, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> DDPBenchmarkMatrixSummary:
        """Require 1/2/4 Strong/Weak while validating optional eight-GPU/NUMA evidence."""

        if self.acceptance_world_sizes != (1, 2, 4):
            raise ValueError("M3 acceptance_world_sizes must be exactly 1/2/4")
        required_standard = {
            (profile, world_size) for profile in ("strong", "weak") for world_size in (1, 2, 4)
        }
        optional_eight = {(profile, 8) for profile in ("strong", "weak")}
        actual_standard = {(item.profile, item.world_size) for item in self.standard}
        if (
            not required_standard.issubset(actual_standard)
            or not actual_standard.issubset(required_standard | optional_eight)
            or len(actual_standard) != len(self.standard)
            or any(item.group != "standard" for item in self.standard)
        ):
            raise ValueError("standard matrix must contain complete 1/2/4 Strong/Weak cells")
        has_eight = bool(actual_standard & optional_eight)
        if has_eight and not optional_eight.issubset(actual_standard):
            raise ValueError("optional eight-GPU evidence must contain both Strong and Weak cells")
        expected_eight_status = "complete" if has_eight else "not_collected"
        if self.eight_gpu_status != expected_eight_status:
            raise ValueError("eight_gpu_status disagrees with standard evidence")
        expected_numa = {"same_numa", "cross_numa"}
        actual_numa = {item.group for item in self.numa}
        if (
            not actual_numa.issubset(expected_numa)
            or len(actual_numa) != len(self.numa)
            or any(item.profile != "weak" or item.world_size != 4 for item in self.numa)
        ):
            raise ValueError("NUMA cells must be four-GPU Weak Scaling")
        expected_numa_status = (
            "complete"
            if actual_numa == expected_numa
            else "partial"
            if actual_numa
            else "not_collected"
        )
        if self.numa_comparison_status != expected_numa_status:
            raise ValueError("numa_comparison_status disagrees with NUMA evidence")
        aggregates = {(item.profile, item.world_size): item for item in self.standard}
        strong_one = aggregates[("strong", 1)].tokens_per_second_median
        weak_one = aggregates[("weak", 1)].tokens_per_second_median
        for item in self.standard:
            baseline = strong_one if item.profile == "strong" else weak_one
            expected_efficiency = item.tokens_per_second_median / (item.world_size * baseline)
            if item.scaling_efficiency is None or not math.isclose(
                item.scaling_efficiency,
                expected_efficiency,
                rel_tol=1e-12,
            ):
                raise ValueError("scaling efficiency disagrees with throughput measurements")
        if any(item.scaling_efficiency is not None for item in self.numa):
            raise ValueError("unpaired NUMA cells cannot claim scaling efficiency")
        return self
