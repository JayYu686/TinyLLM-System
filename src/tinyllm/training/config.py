"""Strict Pydantic YAML configuration schema for M1 single-device training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from tinyllm.models.tinygpt.config import TinyGPTConfig
from tinyllm.schemas.base import StrictSchema


class TrainingConfigError(ValueError):
    """Raised when a training configuration violates the public M1 schema."""


class RunConfig(StrictSchema):
    """Human identity and random seed for one training run."""

    name: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 1)


class ToyDataConfig(StrictSchema):
    """Configuration for deterministic synthetic token data."""

    kind: Literal["toy"]
    vocab_size: int = Field(ge=2)
    sequence_length: int = Field(ge=2)
    num_samples: int = Field(gt=0)


class TrainingLoopConfig(StrictSchema):
    """Optimizer-step semantics for M1."""

    max_steps: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    warmup_steps: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_warmup(self) -> TrainingLoopConfig:
        """Keep warmup within the configured optimizer-step budget."""

        if self.warmup_steps > self.max_steps:
            raise ValueError("training.warmup_steps must be in [0, max_steps]")
        return self

    @property
    def global_batch_size(self) -> int:
        """Return the M1 world-size-one global batch size."""

        return self.micro_batch_size * self.gradient_accumulation_steps


class PrecisionConfig(StrictSchema):
    """Numerical precision policy for one hardware profile."""

    dtype: Literal["fp32", "bf16", "fp16"]
    allow_tf32: bool
    use_grad_scaler: bool

    @model_validator(mode="after")
    def validate_scaler(self) -> PrecisionConfig:
        """Restrict GradScaler to the FP16 compatibility profile."""

        if self.dtype in {"fp32", "bf16"} and self.use_grad_scaler:
            raise ValueError("GradScaler is only valid for the M1 fp16 profile")
        return self


class CheckpointConfig(StrictSchema):
    """Checkpoint retention and resume policy."""

    output_dir: str = Field(min_length=1)
    save_steps: int = Field(gt=0)
    keep_last: int = Field(gt=0)
    resume: Literal["none", "auto", "exact", "warm", "transfer"]


class M1TrainingConfig(StrictSchema):
    """Complete validated configuration for an M1 training run."""

    schema_version: Literal["1.0"]
    run: RunConfig
    model: TinyGPTConfig
    data: ToyDataConfig
    training: TrainingLoopConfig
    precision: PrecisionConfig
    checkpoint: CheckpointConfig

    @model_validator(mode="after")
    def validate_cross_field_contract(self) -> M1TrainingConfig:
        """Validate model/data compatibility after nested parsing."""

        if self.model.vocab_size != self.data.vocab_size:
            raise ValueError("model.vocab_size must equal data.vocab_size")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError("data.sequence_length cannot exceed model.max_sequence_length")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration snapshot."""

        return self.model_dump(mode="json")


def training_config_from_mapping(raw: object) -> M1TrainingConfig:
    """Validate a decoded YAML object as an M1 training configuration."""

    try:
        return M1TrainingConfig.model_validate(raw)
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
        raise TrainingConfigError("; ".join(messages)) from exc


def training_config_json_schema() -> dict[str, Any]:
    """Return the public JSON Schema for M1 YAML configuration."""

    return M1TrainingConfig.model_json_schema()


def load_training_config(path: Path) -> M1TrainingConfig:
    """Load and validate an M1 YAML configuration file."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise TrainingConfigError("training config must use a .yaml or .yml extension")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainingConfigError(f"cannot read training config: {path}") from exc
    try:
        decoded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"invalid YAML in training config: {path}") from exc
    return training_config_from_mapping(decoded)
