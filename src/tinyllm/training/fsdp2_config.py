"""Strict configuration for bounded M4.1 FSDP2 correctness runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from tinyllm.models.tinygpt.config import TinyGPTConfig
from tinyllm.schemas.base import StrictSchema
from tinyllm.training.config import PrecisionConfig, RunConfig, ToyDataConfig, TrainingLoopConfig


class FSDP2ConfigError(ValueError):
    """Raised when an M4.1 correctness configuration violates its public schema."""


class FSDP2PolicyConfig(StrictSchema):
    """Explicit FSDP2 sharding policy for the bounded correctness gate."""

    strategy: Literal["fsdp2"]
    backend: Literal["gloo", "nccl"]
    device_type: Literal["cpu", "cuda"]
    world_size: int = Field(ge=1, le=4)
    timeout_seconds: int = Field(default=120, ge=10, le=1800)
    reshard_after_forward: Literal[True] = True
    cpu_offload: Literal[False] = False
    activation_checkpointing: Literal[False] = False

    @model_validator(mode="after")
    def validate_backend_device(self) -> FSDP2PolicyConfig:
        """Bind Gloo to CPU and NCCL to CUDA without implicit device fallback."""

        if self.backend == "gloo" and self.device_type != "cpu":
            raise ValueError("gloo FSDP2 correctness requires device_type=cpu")
        if self.backend == "nccl" and self.device_type != "cuda":
            raise ValueError("nccl FSDP2 correctness requires device_type=cuda")
        return self


class FSDP2CorrectnessConfig(StrictSchema):
    """Complete M4.1 TinyGPT configuration used before Qwen model loading."""

    schema_version: Literal["1.0"]
    run: RunConfig
    model: TinyGPTConfig
    data: ToyDataConfig
    training: TrainingLoopConfig
    precision: PrecisionConfig
    distributed: FSDP2PolicyConfig

    @model_validator(mode="after")
    def validate_correctness_boundary(self) -> FSDP2CorrectnessConfig:
        """Keep this first gate deterministic, bounded, and distinct from formal Qwen runs."""

        if self.model.vocab_size != self.data.vocab_size:
            raise ValueError("model.vocab_size must equal data.vocab_size")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("data.sequence_length cannot exceed model.max_sequence_length")
        if self.training.max_steps > 10:
            raise ValueError("M4.1 correctness runs are bounded to at most 10 optimizer steps")
        if self.training.gradient_accumulation_steps != 1:
            raise ValueError("M4.1 correctness does not yet validate gradient accumulation")
        if self.precision.use_grad_scaler:
            raise ValueError("M4.1 FSDP2 correctness does not support GradScaler")
        if self.distributed.device_type == "cpu":
            if self.precision.dtype != "fp32" or self.precision.allow_tf32:
                raise ValueError("CPU/Gloo FSDP2 correctness requires fp32 with TF32 disabled")
        elif self.precision.dtype != "bf16":
            raise ValueError("CUDA/NCCL FSDP2 correctness requires bf16")
        if self.data.num_samples % self.distributed.world_size != 0:
            raise ValueError("FSDP2 data.num_samples must be divisible by world_size")
        samples_per_rank = self.data.num_samples // self.distributed.world_size
        if samples_per_rank % self.training.micro_batch_size != 0:
            raise ValueError("FSDP2 samples per rank must be divisible by micro_batch_size")
        required_per_rank = self.training.max_steps * self.training.micro_batch_size
        if required_per_rank > samples_per_rank:
            raise ValueError("bounded FSDP2 runs must fit within one sampler epoch")
        return self

    @property
    def global_batch_size(self) -> int:
        """Return micro batch × World Size for this no-accumulation correctness gate."""

        return self.training.micro_batch_size * self.distributed.world_size

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""

        return self.model_dump(mode="json")


def fsdp2_config_from_mapping(raw: object) -> FSDP2CorrectnessConfig:
    """Validate one decoded YAML object as an M4.1 correctness configuration."""

    try:
        return FSDP2CorrectnessConfig.model_validate(raw)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            if error["type"] == "extra_forbidden":
                messages.append(f"unknown config field: {location}")
            elif location:
                messages.append(f"{location}: {error['msg']}")
            else:
                messages.append(str(error["msg"]))
        raise FSDP2ConfigError("; ".join(messages)) from exc


def load_fsdp2_config(path: Path) -> FSDP2CorrectnessConfig:
    """Load and validate an M4.1 correctness YAML file."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise FSDP2ConfigError("FSDP2 config must use a .yaml or .yml extension")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FSDP2ConfigError(f"cannot read FSDP2 config: {path}") from exc
    try:
        decoded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise FSDP2ConfigError(f"invalid YAML in FSDP2 config: {path}") from exc
    return fsdp2_config_from_mapping(decoded)
