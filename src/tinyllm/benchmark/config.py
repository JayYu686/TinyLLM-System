"""Strict YAML contract for M3 DDP training benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from tinyllm.models.tinygpt import TinyGPTConfig
from tinyllm.schemas.base import StrictSchema

BenchmarkProfile = Literal["strong", "weak"]


class DDPBenchmarkConfigError(ValueError):
    """Raised when a DDP benchmark YAML violates its strict contract."""


class BenchmarkRunConfig(StrictSchema):
    """Stable benchmark identity and independent-repeat seed."""

    name: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 4)


class BenchmarkDataConfig(StrictSchema):
    """Deterministic synthetic data semantics used for systems measurement."""

    kind: Literal["toy"] = "toy"
    vocab_size: int = Field(ge=2)
    sequence_length: int = Field(ge=2)


class BenchmarkTrainingConfig(StrictSchema):
    """Optimizer and batch semantics shared by all benchmark profiles."""

    micro_batch_size: int = Field(gt=0)
    strong_global_batch_size: int = Field(gt=0)
    weak_per_rank_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    learning_rate_warmup_steps: int = Field(ge=0)


class BenchmarkWindowConfig(StrictSchema):
    """Warmup, measurement, repetition, and Profiler windows."""

    warmup_steps: int = Field(ge=0)
    measurement_steps: int = Field(gt=0)
    repetitions: int = Field(gt=0, le=10)
    profiler_steps: int = Field(ge=0)
    profiler_repeat: int = Field(gt=0)
    telemetry_interval_seconds: float = Field(gt=0, le=60)

    @model_validator(mode="after")
    def validate_windows(self) -> BenchmarkWindowConfig:
        """Keep bounded profile and repeat windows inside the experiment."""

        if self.profiler_steps > self.measurement_steps:
            raise ValueError("benchmark.profiler_steps cannot exceed measurement_steps")
        if self.profiler_repeat > self.repetitions:
            raise ValueError("benchmark.profiler_repeat cannot exceed repetitions")
        return self


class BenchmarkPrecisionConfig(StrictSchema):
    """Numerical policy for a benchmark host."""

    dtype: Literal["fp32", "bf16"]
    allow_tf32: bool
    use_grad_scaler: Literal[False] = False


class BenchmarkDistributedConfig(StrictSchema):
    """Single-node process-group policy."""

    backend: Literal["gloo", "nccl"]
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    broadcast_buffers: Literal[False] = False
    find_unused_parameters: Literal[False] = False


class DDPBenchmarkConfig(StrictSchema):
    """Complete, immutable DDP benchmark YAML."""

    schema_version: Literal["1.0"]
    run: BenchmarkRunConfig
    model: TinyGPTConfig
    data: BenchmarkDataConfig
    training: BenchmarkTrainingConfig
    benchmark: BenchmarkWindowConfig
    precision: BenchmarkPrecisionConfig
    distributed: BenchmarkDistributedConfig

    @model_validator(mode="after")
    def validate_cross_field_contract(self) -> DDPBenchmarkConfig:
        """Reject mismatched model, data, precision, and batch semantics."""

        if self.model.vocab_size != self.data.vocab_size:
            raise ValueError("model.vocab_size must equal data.vocab_size")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("data.sequence_length cannot exceed model.max_sequence_length")
        if self.training.weak_per_rank_batch_size % self.training.micro_batch_size != 0:
            raise ValueError("weak_per_rank_batch_size must be divisible by micro_batch_size")
        total_steps = self.benchmark.warmup_steps + self.benchmark.measurement_steps
        if self.training.learning_rate_warmup_steps > total_steps:
            raise ValueError("learning_rate_warmup_steps cannot exceed total benchmark steps")
        if self.distributed.backend == "gloo" and self.precision.dtype != "fp32":
            raise ValueError("Gloo benchmark requires fp32")
        if self.distributed.backend == "nccl" and self.precision.dtype != "bf16":
            raise ValueError("formal NCCL benchmark requires bf16")
        return self


class ResolvedBenchmarkProfile(StrictSchema):
    """World-size-specific batch and step values derived from the YAML."""

    schema_version: Literal["1.0"] = "1.0"
    profile: BenchmarkProfile
    world_size: int = Field(ge=1, le=10)
    repeat: int = Field(gt=0, le=10)
    seed: int = Field(ge=0, le=2**32 - 1)
    warmup_steps: int = Field(ge=0)
    measurement_steps: int = Field(gt=0)
    profiler_steps: int = Field(ge=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    samples_per_rank: int = Field(gt=0)
    dataset_samples: int = Field(gt=0)


def resolve_benchmark_profile(
    config: DDPBenchmarkConfig,
    *,
    profile: BenchmarkProfile,
    world_size: int,
    repeat: int,
) -> ResolvedBenchmarkProfile:
    """Derive one exact Strong or Weak Scaling execution."""

    if not 1 <= world_size <= 10:
        raise DDPBenchmarkConfigError("world_size must be between 1 and 10")
    if not 1 <= repeat <= config.benchmark.repetitions:
        raise DDPBenchmarkConfigError("repeat is outside the configured repetitions")
    micro_batch_size = config.training.micro_batch_size
    if profile == "strong":
        denominator = micro_batch_size * world_size
        if config.training.strong_global_batch_size % denominator != 0:
            raise DDPBenchmarkConfigError(
                "strong_global_batch_size must be divisible by micro_batch_size * world_size"
            )
        accumulation = config.training.strong_global_batch_size // denominator
        global_batch_size = config.training.strong_global_batch_size
    else:
        accumulation = config.training.weak_per_rank_batch_size // micro_batch_size
        global_batch_size = micro_batch_size * accumulation * world_size
    total_steps = config.benchmark.warmup_steps + config.benchmark.measurement_steps
    samples_per_rank = total_steps * micro_batch_size * accumulation
    profiler_steps = (
        config.benchmark.profiler_steps if repeat == config.benchmark.profiler_repeat else 0
    )
    return ResolvedBenchmarkProfile(
        profile=profile,
        world_size=world_size,
        repeat=repeat,
        seed=config.run.seed + repeat - 1,
        warmup_steps=config.benchmark.warmup_steps,
        measurement_steps=config.benchmark.measurement_steps,
        profiler_steps=profiler_steps,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation,
        global_batch_size=global_batch_size,
        samples_per_rank=samples_per_rank,
        dataset_samples=samples_per_rank * world_size,
    )


def validate_formal_m3_config(config: DDPBenchmarkConfig) -> None:
    """Require the frozen M3.3/M3.4 acceptance values."""

    expected_model = {
        "vocab_size": 32768,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 2304,
        "max_sequence_length": 1024,
        "tie_word_embeddings": True,
    }
    actual_model = config.model.to_dict()
    mismatches = {
        key: {"actual": actual_model[key], "expected": value}
        for key, value in expected_model.items()
        if actual_model[key] != value
    }
    expected_values: dict[str, object] = {
        "data.sequence_length": 1024,
        "training.micro_batch_size": 1,
        "training.strong_global_batch_size": 8,
        "training.weak_per_rank_batch_size": 1,
        "benchmark.warmup_steps": 20,
        "benchmark.measurement_steps": 100,
        "benchmark.repetitions": 3,
        "benchmark.profiler_steps": 5,
        "benchmark.profiler_repeat": 1,
        "precision.dtype": "bf16",
        "precision.allow_tf32": True,
        "distributed.backend": "nccl",
    }
    actual_values: dict[str, object] = {
        "data.sequence_length": config.data.sequence_length,
        "training.micro_batch_size": config.training.micro_batch_size,
        "training.strong_global_batch_size": config.training.strong_global_batch_size,
        "training.weak_per_rank_batch_size": config.training.weak_per_rank_batch_size,
        "benchmark.warmup_steps": config.benchmark.warmup_steps,
        "benchmark.measurement_steps": config.benchmark.measurement_steps,
        "benchmark.repetitions": config.benchmark.repetitions,
        "benchmark.profiler_steps": config.benchmark.profiler_steps,
        "benchmark.profiler_repeat": config.benchmark.profiler_repeat,
        "precision.dtype": config.precision.dtype,
        "precision.allow_tf32": config.precision.allow_tf32,
        "distributed.backend": config.distributed.backend,
    }
    mismatches.update(
        {
            key: {"actual": actual_values[key], "expected": value}
            for key, value in expected_values.items()
            if actual_values[key] != value
        }
    )
    if mismatches:
        raise DDPBenchmarkConfigError(f"formal M3 benchmark config mismatch: {mismatches}")
    for world_size in (1, 2, 4, 8):
        resolve_benchmark_profile(config, profile="strong", world_size=world_size, repeat=1)
        resolve_benchmark_profile(config, profile="weak", world_size=world_size, repeat=1)


def benchmark_config_from_mapping(raw: object) -> DDPBenchmarkConfig:
    """Validate a decoded YAML mapping with readable errors."""

    try:
        return DDPBenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
        raise DDPBenchmarkConfigError("; ".join(messages)) from exc


def load_ddp_benchmark_config(path: Path) -> DDPBenchmarkConfig:
    """Read one strict DDP benchmark YAML file."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise DDPBenchmarkConfigError("benchmark config must use .yaml or .yml")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DDPBenchmarkConfigError(f"cannot read benchmark config: {path}") from exc
    except yaml.YAMLError as exc:
        raise DDPBenchmarkConfigError(f"invalid benchmark YAML: {path}") from exc
    return benchmark_config_from_mapping(decoded)
