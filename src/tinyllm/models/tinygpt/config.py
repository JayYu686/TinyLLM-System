"""Configuration for the TinyGPT decoder-only model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class TinyGPTConfigError(ValueError):
    """Raised when a TinyGPT configuration is invalid."""


def _integer(mapping: dict[str, Any], key: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TinyGPTConfigError(f"model.{key} must be an integer")
    return value


def _number(mapping: dict[str, Any], key: str, default: float | None = None) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinyGPTConfigError(f"model.{key} must be a number")
    return float(value)


def _boolean(mapping: dict[str, Any], key: str, default: bool | None = None) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise TinyGPTConfigError(f"model.{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class TinyGPTConfig:
    """Validated architecture parameters for TinyGPT."""

    vocab_size: int = 256
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 6
    intermediate_size: int = 512
    max_sequence_length: int = 128
    rope_theta: float = 10_000.0
    rms_norm_epsilon: float = 1.0e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise TinyGPTConfigError("model.vocab_size must be at least 2")
        if self.hidden_size <= 0:
            raise TinyGPTConfigError("model.hidden_size must be positive")
        if self.num_layers <= 0:
            raise TinyGPTConfigError("model.num_layers must be positive")
        if self.num_heads <= 0:
            raise TinyGPTConfigError("model.num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise TinyGPTConfigError("model.hidden_size must be divisible by model.num_heads")
        if self.head_dimension % 2 != 0:
            raise TinyGPTConfigError("attention head dimension must be even for RoPE")
        if self.intermediate_size <= 0:
            raise TinyGPTConfigError("model.intermediate_size must be positive")
        if self.max_sequence_length < 2:
            raise TinyGPTConfigError("model.max_sequence_length must be at least 2")
        if self.rope_theta <= 0:
            raise TinyGPTConfigError("model.rope_theta must be positive")
        if self.rms_norm_epsilon <= 0:
            raise TinyGPTConfigError("model.rms_norm_epsilon must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise TinyGPTConfigError("model.dropout must be in [0, 1)")

    @property
    def head_dimension(self) -> int:
        """Return the dimension of one attention head."""

        return self.hidden_size // self.num_heads

    @classmethod
    def from_mapping(cls, raw: object) -> TinyGPTConfig:
        """Build a config from a mapping while rejecting unknown fields."""

        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise TinyGPTConfigError("model must be a string-keyed mapping")
        mapping: dict[str, Any] = raw
        allowed = {
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "intermediate_size",
            "max_sequence_length",
            "rope_theta",
            "rms_norm_epsilon",
            "dropout",
            "tie_word_embeddings",
        }
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise TinyGPTConfigError(f"unknown model field(s): {', '.join(unknown)}")
        defaults = cls()
        return cls(
            vocab_size=_integer(mapping, "vocab_size", defaults.vocab_size),
            hidden_size=_integer(mapping, "hidden_size", defaults.hidden_size),
            num_layers=_integer(mapping, "num_layers", defaults.num_layers),
            num_heads=_integer(mapping, "num_heads", defaults.num_heads),
            intermediate_size=_integer(mapping, "intermediate_size", defaults.intermediate_size),
            max_sequence_length=_integer(
                mapping, "max_sequence_length", defaults.max_sequence_length
            ),
            rope_theta=_number(mapping, "rope_theta", defaults.rope_theta),
            rms_norm_epsilon=_number(mapping, "rms_norm_epsilon", defaults.rms_norm_epsilon),
            dropout=_number(mapping, "dropout", defaults.dropout),
            tie_word_embeddings=_boolean(
                mapping, "tie_word_embeddings", defaults.tie_word_embeddings
            ),
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-serializable representation."""

        return asdict(self)
