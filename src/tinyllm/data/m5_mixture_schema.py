"""Strict manifests for deterministic M5.2 supervised-token mixtures."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5MixtureArtifactFile(StrictSchema):
    """One immutable private mixture payload."""

    path: Literal["sequences.npz"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5MixtureManifest(StrictSchema):
    """Content-addressed M5.2 mixture with exact supervised-token accounting."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-ablation-mixture"] = "m5-ablation-mixture"
    mixture_version: str = Field(pattern=r"^m5-ablation-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    pilot_dataset_version: str = Field(pattern=r"^m5-reasoning-pilot-v1-[0-9a-f]{8}$")
    pilot_content_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-v1"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[0, 3000, 5000]
    nonthinking_supervised_tokens: int = Field(ge=0)
    thinking_supervised_tokens: int = Field(ge=0)
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(ge=0)
    thinking_sequence_count: int = Field(ge=0)
    nonthinking_source_sequences: int = Field(gt=0)
    thinking_source_sequences: int = Field(gt=0)
    nonthinking_reuse_count: int = Field(ge=0)
    thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=2)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M5MixtureManifest:
        """Bind the ratio, sequence totals, and content-addressed version."""

        if self.mixture_version != f"m5-ablation-mixture-v1-{self.content_sha256[:8]}":
            raise ValueError("mixture version does not match content hash")
        if self.nonthinking_supervised_tokens + self.thinking_supervised_tokens != 1_000_000:
            raise ValueError("mixture supervised-token counts must total exactly 1M")
        if self.thinking_supervised_tokens != self.thinking_fraction_basis_points * 100:
            raise ValueError("Thinking supervised-token count does not match ratio")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("mixture mode sequence counts do not equal total sequences")
        if self.thinking_fraction_basis_points == 0 and self.thinking_sequence_count != 0:
            raise ValueError("zero-Think mixture cannot contain Thinking sequences")
        if self.thinking_fraction_basis_points > 0 and self.thinking_sequence_count == 0:
            raise ValueError("Thinking mixture must contain Thinking sequences")
        return self
