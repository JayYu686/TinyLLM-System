"""Strict public contracts for pinned Qwen3 tokenization and assistant labels."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.data.processing_schema import DataSplit
from tinyllm.data.schema import DataSourceName
from tinyllm.schemas.base import StrictSchema

TokenizationRejectionReason = Literal["no_supervised_tokens", "sequence_too_long"]


class TokenizerIdentity(StrictSchema):
    """Pinned files, backend, vocabulary, and special-token contract."""

    repository: Literal["Qwen/Qwen3-0.6B"]
    revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    tokenizer_file: Literal["tokenizer.json"]
    tokenizer_file_size: int = Field(gt=0)
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_config_file: Literal["tokenizer_config.json"]
    tokenizer_config_file_size: int = Field(gt=0)
    tokenizer_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: Literal["tokenizers"]
    backend_version: Literal["0.21.4"]
    vocab_size: int = Field(gt=0)
    pad_token: str = Field(min_length=1)
    pad_token_id: int = Field(ge=0)
    bos_token: str = Field(min_length=1)
    bos_token_id: int = Field(ge=0)
    eos_token: str = Field(min_length=1)
    eos_token_id: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_token_ids(self) -> TokenizerIdentity:
        """Require distinct in-vocabulary special tokens."""

        token_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("pad, BOS, and EOS token IDs must be distinct")
        if any(token_id >= self.vocab_size for token_id in token_ids):
            raise ValueError("special token IDs must be inside the tokenizer vocabulary")
        return self


class ChatTemplateIdentity(StrictSchema):
    """Frozen non-thinking ChatML rendering and supervision policy."""

    template_id: Literal["qwen3-chatml-nonthinking-v1"]
    template_sha256: Literal["d41161e0416a1047b0f31cce1497e610a4050fbe4d3fb7bda19cc56a1523cb33"]
    thinking: Literal[False]
    add_generation_prompt: Literal[False]
    assistant_only_loss: Literal[True]
    supervise_assistant_end_token: Literal[True]


class M2TokenizationConfig(StrictSchema):
    """Complete M2.3a tokenizer, template, and length policy."""

    schema_version: Literal["1.0"] = "1.0"
    tokenizer: TokenizerIdentity
    template: ChatTemplateIdentity
    max_sequence_length: int = Field(gt=1)
    overlength_policy: Literal["reject"]


class TokenizedSample(StrictSchema):
    """One processed sample encoded with an assistant-only causal-LM mask."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    split: DataSplit
    component_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_keys: tuple[str, ...] = Field(min_length=1)
    origin_sample_ids: tuple[str, ...] = Field(min_length=1)
    origin_record_sha256s: tuple[str, ...] = Field(min_length=1)
    language: str = Field(min_length=2, max_length=16)
    license: str = Field(min_length=1, max_length=64)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_sequence_length: int = Field(gt=1)
    input_ids: tuple[int, ...] = Field(min_length=1)
    labels: tuple[int, ...] = Field(min_length=1)
    token_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_token_contract(self) -> TokenizedSample:
        """Bind lengths, labels, identity, and vocabulary-independent mask semantics."""

        if not self.id.startswith(f"{self.source}:"):
            raise ValueError("tokenized sample ID must match its source")
        if self.id not in self.origin_sample_ids:
            raise ValueError("tokenized sample ID must be present in origin sample IDs")
        if len(self.input_ids) != len(self.labels) or len(self.input_ids) != self.token_count:
            raise ValueError("input IDs, labels, and token count must have equal lengths")
        if self.token_count > self.max_sequence_length:
            raise ValueError("token count exceeds configured maximum sequence length")
        if any(token_id < 0 for token_id in self.input_ids):
            raise ValueError("input token IDs must be non-negative")
        supervised = 0
        for token_id, label in zip(self.input_ids, self.labels, strict=True):
            if label == -100:
                continue
            if label != token_id:
                raise ValueError("supervised labels must equal their input token IDs")
            supervised += 1
        if supervised != self.supervised_token_count:
            raise ValueError("supervised token count does not match labels")
        return self


class TokenizationRejectedRecord(StrictSchema):
    """Content-free data-level rejection produced after a valid tokenizer run."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    split: DataSplit
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: TokenizationRejectionReason
    observed_token_count: int = Field(ge=0)
    max_sequence_length: int = Field(gt=1)

    @model_validator(mode="after")
    def validate_rejection(self) -> TokenizationRejectedRecord:
        """Keep the rejection reason consistent with observed non-sensitive counts."""

        if not self.sample_id.startswith(f"{self.source}:"):
            raise ValueError("rejected tokenization sample ID must match its source")
        if self.reason == "sequence_too_long":
            if self.observed_token_count <= self.max_sequence_length:
                raise ValueError("overlength rejection requires a count above the maximum")
        elif self.observed_token_count > self.max_sequence_length:
            raise ValueError("no-supervision rejection cannot also be overlength")
        return self
