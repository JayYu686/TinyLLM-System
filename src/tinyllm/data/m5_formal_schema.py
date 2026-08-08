"""Strict contracts for the M5.3 immutable dual-mode training dataset."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5FormalArtifactFile(StrictSchema):
    """One immutable payload in the formal M5 dataset."""

    path: Literal["sequences.npz", "epoch_plan.npy"]
    role: Literal["source_sequences", "epoch_plan"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5FormalDatasetManifest(StrictSchema):
    """Content-addressed 50M-token view over the selected R1 source mixture."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-dual-sft"] = "m5-dual-sft"
    dataset_version: str = Field(pattern=r"^m5-dual-sft-v1-[0-9a-f]{8}$")
    source_mixture_version: Literal["m5-format-repair-mixture-v1-1396b60b"]
    source_manifest_sha256: Literal[
        "2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"
    ]
    source_content_sha256: str = Field(pattern=SHA256_PATTERN)
    authorization_gate_sha256: str = Field(pattern=SHA256_PATTERN)
    build_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-v1"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    thinking_fraction_basis_points: Literal[3000]
    source_supervised_tokens: Literal[1_000_000]
    source_nonthinking_tokens: Literal[700_000]
    source_thinking_tokens: Literal[300_000]
    repeated_epochs: Literal[50]
    target_supervised_tokens: Literal[50_000_000]
    nonthinking_supervised_tokens: Literal[35_000_000]
    thinking_supervised_tokens: Literal[15_000_000]
    source_sequence_count: int = Field(gt=0)
    sequence_count: int = Field(gt=0)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifacts: tuple[M5FormalArtifactFile, M5FormalArtifactFile]

    @model_validator(mode="after")
    def validate_identity(self) -> M5FormalDatasetManifest:
        """Bind version, ratios, repeated view, and artifact roles."""

        if self.dataset_version != f"m5-dual-sft-v1-{self.content_sha256[:8]}":
            raise ValueError("formal M5 dataset version does not match content hash")
        if self.sequence_count != self.source_sequence_count * self.repeated_epochs:
            raise ValueError("formal M5 sequence count does not match repeated epochs")
        if (
            self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
            or self.thinking_supervised_tokens * 10_000
            != self.target_supervised_tokens * self.thinking_fraction_basis_points
        ):
            raise ValueError("formal M5 supervised-token ratio is inconsistent")
        expected_artifacts = (
            ("sequences.npz", "source_sequences"),
            ("epoch_plan.npy", "epoch_plan"),
        )
        if tuple((item.path, item.role) for item in self.artifacts) != expected_artifacts:
            raise ValueError("formal M5 artifacts must be ordered sequences then plan")
        return self
