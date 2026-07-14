"""Pinned Qwen3 tokenization with offset-aligned assistant-only labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import tokenizers  # type: ignore[import-untyped]
import yaml
from pydantic import ValidationError
from tokenizers import Tokenizer

from tinyllm.data.processing_schema import ProcessedSample
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization_schema import (
    M2TokenizationConfig,
    TokenizationRejectedRecord,
    TokenizedSample,
    TokenizerIdentity,
)

_CHATML_START = "<|im_start|>"
_CHATML_END = "<|im_end|>"
_QWEN3_MESSAGE_FORMAT = f"{_CHATML_START}{{role}}\n{{content}}{_CHATML_END}\n"
QWEN3_NONTHINKING_TEMPLATE_SPEC: dict[str, object] = {
    "add_generation_prompt": False,
    "assistant_supervision": "content_and_im_end",
    "id": "qwen3-chatml-nonthinking-v1",
    "message_format": _QWEN3_MESSAGE_FORMAT,
    "mode": "non-thinking",
}
QWEN3_NONTHINKING_TEMPLATE_SHA256 = (
    "d41161e0416a1047b0f31cce1497e610a4050fbe4d3fb7bda19cc56a1523cb33"
)


class TokenizerContractError(ValueError):
    """Raised when tokenizer identity or offset behavior violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class TokenEncoding:
    """Backend-neutral token IDs and character offsets."""

    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]


class OffsetTokenizer(Protocol):
    """Minimal tokenizer interface needed by the deterministic M2 pipeline."""

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including added tokens."""

    def token_to_id(self, token: str) -> int | None:
        """Resolve one special token without implicit fallback."""

    def encode(self, text: str) -> TokenEncoding:
        """Encode without model-added BOS/EOS tokens and retain character offsets."""


@dataclass(frozen=True, slots=True)
class RenderedConversation:
    """Rendered ChatML text and half-open assistant supervision spans."""

    text: str
    assistant_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class TokenizationBatch:
    """Deterministically ordered tokenized samples and data-level rejections."""

    samples: tuple[TokenizedSample, ...]
    rejected: tuple[TokenizationRejectedRecord, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _template_hash() -> str:
    payload = json.dumps(
        QWEN3_NONTHINKING_TEMPLATE_SPEC,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_m2_tokenization_config(path: Path) -> M2TokenizationConfig:
    """Load a strict formal M2.3a YAML configuration."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise TokenizerContractError("tokenization config must use a .yaml or .yml extension")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TokenizerContractError(f"cannot read tokenization config: {path}") from exc
    except yaml.YAMLError as exc:
        raise TokenizerContractError(f"invalid YAML in tokenization config: {path}") from exc
    try:
        config = M2TokenizationConfig.model_validate(decoded)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
        raise TokenizerContractError("; ".join(messages)) from exc
    if _template_hash() != config.template.template_sha256:
        raise TokenizerContractError("built-in Chat Template hash does not match configuration")
    return config


class TokenizersBackend:
    """Rust-tokenizers adapter that verifies a local pinned artifact before use."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_files(
        cls,
        tokenizer_path: Path,
        tokenizer_config_path: Path,
        identity: TokenizerIdentity,
    ) -> TokenizersBackend:
        """Load only after checking size, SHA256, backend version, vocab, and special IDs."""

        if not tokenizer_path.is_file():
            raise TokenizerContractError(f"tokenizer file does not exist: {tokenizer_path}")
        if not tokenizer_config_path.is_file():
            raise TokenizerContractError(
                f"tokenizer config file does not exist: {tokenizer_config_path}"
            )
        if tokenizer_path.stat().st_size != identity.tokenizer_file_size:
            raise TokenizerContractError("tokenizer file size does not match pinned identity")
        if _sha256_file(tokenizer_path) != identity.tokenizer_sha256:
            raise TokenizerContractError("tokenizer file SHA256 does not match pinned identity")
        if tokenizer_config_path.stat().st_size != identity.tokenizer_config_file_size:
            raise TokenizerContractError("tokenizer config size does not match pinned identity")
        if _sha256_file(tokenizer_config_path) != identity.tokenizer_config_sha256:
            raise TokenizerContractError("tokenizer config SHA256 does not match pinned identity")
        if tokenizers.__version__ != identity.backend_version:
            raise TokenizerContractError(
                "tokenizers backend must be "
                f"{identity.backend_version}, got {tokenizers.__version__}"
            )
        try:
            backend = cls(Tokenizer.from_file(str(tokenizer_path)))
        except Exception as exc:
            raise TokenizerContractError("tokenizer artifact cannot be parsed") from exc
        backend.validate_identity(identity)
        return backend

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size including added special tokens."""

        return cast(int, self._tokenizer.get_vocab_size(with_added_tokens=True))

    def token_to_id(self, token: str) -> int | None:
        """Resolve one token exactly."""

        return cast(int | None, self._tokenizer.token_to_id(token))

    def encode(self, text: str) -> TokenEncoding:
        """Encode full rendered text without implicit special-token insertion."""

        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return TokenEncoding(ids=tuple(encoding.ids), offsets=tuple(encoding.offsets))

    def validate_identity(self, identity: TokenizerIdentity) -> None:
        """Check vocabulary and all frozen special-token identities."""

        if self.vocab_size != identity.vocab_size:
            raise TokenizerContractError("tokenizer vocabulary size does not match pinned identity")
        expected = {
            identity.pad_token: identity.pad_token_id,
            identity.bos_token: identity.bos_token_id,
            identity.eos_token: identity.eos_token_id,
        }
        for token, expected_id in expected.items():
            if self.token_to_id(token) != expected_id:
                raise TokenizerContractError(f"special token ID mismatch for {token}")


def render_qwen3_nonthinking(messages: tuple[ImportedMessage, ...]) -> RenderedConversation:
    """Render the frozen system/user/assistant ChatML subset without thinking tags."""

    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    length = 0
    for message in messages:
        header = f"{_CHATML_START}{message.role}\n"
        parts.append(header)
        length += len(header)
        content_start = length
        parts.append(message.content)
        length += len(message.content)
        parts.append(_CHATML_END)
        length += len(_CHATML_END)
        if message.role == "assistant":
            assistant_spans.append((content_start, length))
        parts.append("\n")
        length += 1
    return RenderedConversation(text="".join(parts), assistant_spans=tuple(assistant_spans))


def _validate_backend(backend: OffsetTokenizer, identity: TokenizerIdentity) -> None:
    if backend.vocab_size != identity.vocab_size:
        raise TokenizerContractError("tokenizer backend vocabulary does not match configuration")
    for token, expected_id in (
        (identity.pad_token, identity.pad_token_id),
        (identity.bos_token, identity.bos_token_id),
        (identity.eos_token, identity.eos_token_id),
    ):
        if backend.token_to_id(token) != expected_id:
            raise TokenizerContractError(f"tokenizer backend special ID mismatch for {token}")


def _labels_from_offsets(
    *,
    text: str,
    encoding: TokenEncoding,
    assistant_spans: tuple[tuple[int, int], ...],
    vocab_size: int,
) -> tuple[int, ...]:
    if len(encoding.ids) != len(encoding.offsets):
        raise TokenizerContractError("token ID and offset counts differ")
    labels: list[int] = []
    previous_start = 0
    for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True):
        if token_id < 0 or token_id >= vocab_size:
            raise TokenizerContractError("encoded token ID is outside the configured vocabulary")
        if start < previous_start or start < 0 or end <= start or end > len(text):
            raise TokenizerContractError("tokenizer returned invalid or non-monotonic offsets")
        previous_start = start
        overlaps = [
            (span_start, span_end)
            for span_start, span_end in assistant_spans
            if start < span_end and end > span_start
        ]
        if not overlaps:
            labels.append(-100)
            continue
        if not any(start >= span_start and end <= span_end for span_start, span_end in overlaps):
            raise TokenizerContractError("token crosses an assistant supervision boundary")
        labels.append(token_id)
    return tuple(labels)


def tokenize_processed_sample(
    sample: ProcessedSample,
    *,
    backend: OffsetTokenizer,
    config: M2TokenizationConfig,
) -> TokenizedSample | TokenizationRejectedRecord:
    """Tokenize one processed sample or return a content-free data-level rejection."""

    _validate_backend(backend, config.tokenizer)
    rendered = render_qwen3_nonthinking(sample.messages)
    encoding = backend.encode(rendered.text)
    labels = _labels_from_offsets(
        text=rendered.text,
        encoding=encoding,
        assistant_spans=rendered.assistant_spans,
        vocab_size=config.tokenizer.vocab_size,
    )
    token_count = len(encoding.ids)
    if token_count > config.max_sequence_length:
        return TokenizationRejectedRecord(
            sample_id=sample.id,
            source=sample.source,
            split=sample.split,
            content_sha256=sample.content_sha256,
            reason="sequence_too_long",
            observed_token_count=token_count,
            max_sequence_length=config.max_sequence_length,
        )
    supervised_token_count = sum(label != -100 for label in labels)
    if supervised_token_count == 0:
        return TokenizationRejectedRecord(
            sample_id=sample.id,
            source=sample.source,
            split=sample.split,
            content_sha256=sample.content_sha256,
            reason="no_supervised_tokens",
            observed_token_count=token_count,
            max_sequence_length=config.max_sequence_length,
        )
    assistant_count = sum(message.role == "assistant" for message in sample.messages)
    if sum(label == config.tokenizer.eos_token_id for label in labels) < assistant_count:
        raise TokenizerContractError("assistant end token is not supervised for every response")
    return TokenizedSample(
        id=sample.id,
        source=sample.source,
        split=sample.split,
        component_id=sample.component_id,
        group_keys=sample.group_keys,
        origin_sample_ids=sample.origin_sample_ids,
        origin_record_sha256s=sample.origin_record_sha256s,
        language=sample.metadata.language,
        license=sample.metadata.license,
        content_sha256=sample.content_sha256,
        rendered_sha256=hashlib.sha256(rendered.text.encode()).hexdigest(),
        tokenizer_sha256=config.tokenizer.tokenizer_sha256,
        template_sha256=config.template.template_sha256,
        max_sequence_length=config.max_sequence_length,
        input_ids=encoding.ids,
        labels=labels,
        token_count=token_count,
        supervised_token_count=supervised_token_count,
    )


def tokenize_processed_samples(
    samples: Iterable[ProcessedSample],
    *,
    backend: OffsetTokenizer,
    config: M2TokenizationConfig,
) -> TokenizationBatch:
    """Tokenize a unique sample set in stable ID order; contract failures abort the batch."""

    materialized = tuple(samples)
    ids = [sample.id for sample in materialized]
    if len(ids) != len(set(ids)):
        raise TokenizerContractError("processed sample IDs must be unique before tokenization")
    _validate_backend(backend, config.tokenizer)
    accepted: list[TokenizedSample] = []
    rejected: list[TokenizationRejectedRecord] = []
    for sample in sorted(materialized, key=lambda item: item.id):
        result = tokenize_processed_sample(sample, backend=backend, config=config)
        if isinstance(result, TokenizedSample):
            accepted.append(result)
        else:
            rejected.append(result)
    return TokenizationBatch(samples=tuple(accepted), rejected=tuple(rejected))
