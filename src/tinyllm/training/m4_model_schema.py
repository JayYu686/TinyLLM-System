"""Dependency-free public Schemas for the pinned M4 model artifact."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M4ModelArtifactFile(StrictSchema):
    """One local immutable file used by the pinned Qwen model."""

    path: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M4ModelArtifactManifest(StrictSchema):
    """Verified identity of the complete local Qwen3-8B snapshot."""

    schema_version: Literal["1.0"] = "1.0"
    repository: str
    revision: str
    license: str
    model_type: str
    architecture: str
    hidden_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    weight_bytes: int = Field(gt=0)
    tensor_count: int = Field(gt=0)
    files: tuple[M4ModelArtifactFile, ...]
    content_sha256: str = Field(pattern=SHA256_PATTERN)
