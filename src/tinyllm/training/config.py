"""Strict YAML configuration schema for M1 single-device training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from tinyllm.models.tinygpt.config import TinyGPTConfig, TinyGPTConfigError


class TrainingConfigError(ValueError):
    """Raised when a training configuration violates the M1 schema."""


def _mapping(raw: object, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise TrainingConfigError(f"{path} must be a string-keyed mapping")
    return cast(dict[str, Any], raw)


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise TrainingConfigError(f"unknown {path} field(s): {', '.join(unknown)}")


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise TrainingConfigError(f"missing required field: {path}.{key}")
    return mapping[key]


def _string(mapping: dict[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{path}.{key} must be a non-empty string")
    return value


def _integer(mapping: dict[str, Any], key: str, path: str, *, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(f"{path}.{key} must be an integer")
    return value


def _number(mapping: dict[str, Any], key: str, path: str, *, default: float | None = None) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{path}.{key} must be a number")
    return float(value)


def _boolean(mapping: dict[str, Any], key: str, path: str, *, default: bool | None = None) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise TrainingConfigError(f"{path}.{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Identity and random seed for one training run."""

    name: str
    seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise TrainingConfigError("run.name must be non-empty")
        if not 0 <= self.seed <= 2**32 - 1:
            raise TrainingConfigError("run.seed must be between 0 and 2**32 - 1")


@dataclass(frozen=True, slots=True)
class ToyDataConfig:
    """Configuration for deterministic synthetic token data."""

    kind: Literal["toy"]
    vocab_size: int
    sequence_length: int
    num_samples: int

    def __post_init__(self) -> None:
        if self.kind != "toy":
            raise TrainingConfigError("data.kind must be 'toy' in M1")
        if self.vocab_size < 2:
            raise TrainingConfigError("data.vocab_size must be at least 2")
        if self.sequence_length < 2:
            raise TrainingConfigError("data.sequence_length must be at least 2")
        if self.num_samples <= 0:
            raise TrainingConfigError("data.num_samples must be positive")


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    """Optimizer-step semantics for M1."""

    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise TrainingConfigError("training.max_steps must be positive")
        if self.micro_batch_size <= 0:
            raise TrainingConfigError("training.micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise TrainingConfigError("training.gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise TrainingConfigError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise TrainingConfigError("training.weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise TrainingConfigError("training.max_grad_norm must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise TrainingConfigError("training.warmup_steps must be in [0, max_steps]")

    @property
    def global_batch_size(self) -> int:
        """Return the M1 world-size-one global batch size."""

        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    """Numerical precision policy for one hardware profile."""

    dtype: Literal["fp32", "bf16", "fp16"]
    allow_tf32: bool
    use_grad_scaler: bool

    def __post_init__(self) -> None:
        if self.dtype not in {"fp32", "bf16", "fp16"}:
            raise TrainingConfigError("precision.dtype must be fp32, bf16, or fp16")
        if self.dtype in {"fp32", "bf16"} and self.use_grad_scaler:
            raise TrainingConfigError("GradScaler is only valid for the M1 fp16 profile")


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint retention and resume policy."""

    output_dir: str
    save_steps: int
    keep_last: int
    resume: Literal["none", "auto", "exact"]

    def __post_init__(self) -> None:
        if not self.output_dir.strip():
            raise TrainingConfigError("checkpoint.output_dir must be non-empty")
        if self.save_steps <= 0:
            raise TrainingConfigError("checkpoint.save_steps must be positive")
        if self.keep_last <= 0:
            raise TrainingConfigError("checkpoint.keep_last must be positive")
        if self.resume not in {"none", "auto", "exact"}:
            raise TrainingConfigError("checkpoint.resume must be none, auto, or exact")


@dataclass(frozen=True, slots=True)
class M1TrainingConfig:
    """Complete validated configuration for an M1 training run."""

    schema_version: Literal["1.0"]
    run: RunConfig
    model: TinyGPTConfig
    data: ToyDataConfig
    training: TrainingLoopConfig
    precision: PrecisionConfig
    checkpoint: CheckpointConfig

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise TrainingConfigError("schema_version must be '1.0'")
        if self.model.vocab_size != self.data.vocab_size:
            raise TrainingConfigError("model.vocab_size must equal data.vocab_size")
        if self.data.sequence_length > self.model.max_sequence_length:
            raise TrainingConfigError(
                "data.sequence_length cannot exceed model.max_sequence_length"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable resolved configuration snapshot."""

        return asdict(self)


def _parse_run(raw: object) -> RunConfig:
    mapping = _mapping(raw, "run")
    _reject_unknown(mapping, {"name", "seed"}, "run")
    return RunConfig(
        name=_string(mapping, "name", "run"),
        seed=_integer(mapping, "seed", "run"),
    )


def _parse_data(raw: object) -> ToyDataConfig:
    mapping = _mapping(raw, "data")
    _reject_unknown(mapping, {"kind", "vocab_size", "sequence_length", "num_samples"}, "data")
    kind = _string(mapping, "kind", "data")
    if kind != "toy":
        raise TrainingConfigError("data.kind must be 'toy' in M1")
    return ToyDataConfig(
        kind="toy",
        vocab_size=_integer(mapping, "vocab_size", "data"),
        sequence_length=_integer(mapping, "sequence_length", "data"),
        num_samples=_integer(mapping, "num_samples", "data"),
    )


def _parse_training(raw: object) -> TrainingLoopConfig:
    mapping = _mapping(raw, "training")
    allowed = {
        "max_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "warmup_steps",
    }
    _reject_unknown(mapping, allowed, "training")
    return TrainingLoopConfig(
        max_steps=_integer(mapping, "max_steps", "training"),
        micro_batch_size=_integer(mapping, "micro_batch_size", "training"),
        gradient_accumulation_steps=_integer(mapping, "gradient_accumulation_steps", "training"),
        learning_rate=_number(mapping, "learning_rate", "training"),
        weight_decay=_number(mapping, "weight_decay", "training"),
        max_grad_norm=_number(mapping, "max_grad_norm", "training"),
        warmup_steps=_integer(mapping, "warmup_steps", "training"),
    )


def _parse_precision(raw: object) -> PrecisionConfig:
    mapping = _mapping(raw, "precision")
    _reject_unknown(mapping, {"dtype", "allow_tf32", "use_grad_scaler"}, "precision")
    dtype = _string(mapping, "dtype", "precision")
    if dtype not in {"fp32", "bf16", "fp16"}:
        raise TrainingConfigError("precision.dtype must be fp32, bf16, or fp16")
    return PrecisionConfig(
        dtype=cast(Literal["fp32", "bf16", "fp16"], dtype),
        allow_tf32=_boolean(mapping, "allow_tf32", "precision"),
        use_grad_scaler=_boolean(mapping, "use_grad_scaler", "precision"),
    )


def _parse_checkpoint(raw: object) -> CheckpointConfig:
    mapping = _mapping(raw, "checkpoint")
    _reject_unknown(mapping, {"output_dir", "save_steps", "keep_last", "resume"}, "checkpoint")
    resume = _string(mapping, "resume", "checkpoint")
    if resume not in {"none", "auto", "exact"}:
        raise TrainingConfigError("checkpoint.resume must be none, auto, or exact")
    return CheckpointConfig(
        output_dir=_string(mapping, "output_dir", "checkpoint"),
        save_steps=_integer(mapping, "save_steps", "checkpoint"),
        keep_last=_integer(mapping, "keep_last", "checkpoint"),
        resume=cast(Literal["none", "auto", "exact"], resume),
    )


def training_config_from_mapping(raw: object) -> M1TrainingConfig:
    """Validate a decoded YAML object as an M1 training configuration."""

    root = _mapping(raw, "config")
    allowed = {"schema_version", "run", "model", "data", "training", "precision", "checkpoint"}
    _reject_unknown(root, allowed, "config")
    schema_version = _string(root, "schema_version", "config")
    if schema_version != "1.0":
        raise TrainingConfigError("schema_version must be '1.0'")
    try:
        model = TinyGPTConfig.from_mapping(_required(root, "model", "config"))
    except TinyGPTConfigError as exc:
        raise TrainingConfigError(str(exc)) from exc
    return M1TrainingConfig(
        schema_version="1.0",
        run=_parse_run(_required(root, "run", "config")),
        model=model,
        data=_parse_data(_required(root, "data", "config")),
        training=_parse_training(_required(root, "training", "config")),
        precision=_parse_precision(_required(root, "precision", "config")),
        checkpoint=_parse_checkpoint(_required(root, "checkpoint", "config")),
    )


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
