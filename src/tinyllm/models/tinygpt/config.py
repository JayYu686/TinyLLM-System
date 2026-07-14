"""Pydantic configuration for the TinyGPT decoder-only model."""

from __future__ import annotations

from typing import Any

from pydantic import Field, ValidationError, model_validator

from tinyllm.schemas.base import StrictSchema

TinyGPTConfigError = ValidationError


class TinyGPTConfig(StrictSchema):
    """Strict, immutable architecture parameters for TinyGPT."""

    vocab_size: int = Field(default=256, ge=2)
    hidden_size: int = Field(default=192, gt=0)
    num_layers: int = Field(default=4, gt=0)
    num_heads: int = Field(default=6, gt=0)
    intermediate_size: int = Field(default=512, gt=0)
    max_sequence_length: int = Field(default=128, ge=2)
    rope_theta: float = Field(default=10_000.0, gt=0)
    rms_norm_epsilon: float = Field(default=1.0e-6, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    tie_word_embeddings: bool = True

    @model_validator(mode="after")
    def validate_attention_partition(self) -> TinyGPTConfig:
        """Require complete, even attention heads for RoPE."""

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if self.head_dimension % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        return self

    @property
    def head_dimension(self) -> int:
        """Return the dimension of one attention head."""

        return self.hidden_size // self.num_heads

    @classmethod
    def from_mapping(cls, raw: object) -> TinyGPTConfig:
        """Build a config from a mapping while rejecting unknown fields."""

        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return self.model_dump(mode="json")
