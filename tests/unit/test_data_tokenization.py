from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, models  # type: ignore[import-untyped]

from tinyllm.data import (
    OASST1_SOURCE,
    QWEN3_NONTHINKING_TEMPLATE_SHA256,
    QWEN3_THINKING_TEMPLATE_SHA256,
    ImportedMessage,
    ImportedSample,
    ImportedSampleMetadata,
    M2TokenizationConfig,
    OffsetTokenizer,
    ProcessedSample,
    TokenEncoding,
    TokenizationRejectedRecord,
    TokenizedSample,
    TokenizerContractError,
    TokenizerIdentity,
    TokenizersBackend,
    load_m2_processing_config,
    load_m2_tokenization_config,
    process_imported_samples,
    render_qwen3_nonthinking,
    render_qwen3_thinking,
    tokenize_processed_sample,
    tokenize_processed_samples,
    tokenize_thinking_messages,
)

PROCESSING_CONFIG = Path("configs/data/m2_processing.yaml")
TOKENIZATION_CONFIG = Path("configs/data/m2_tokenization.yaml")


class CharacterTokenizer(OffsetTokenizer):
    def __init__(self, *, mode: str = "normal") -> None:
        self.mode = mode
        self._special = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}

    @property
    def vocab_size(self) -> int:
        return 512

    def token_to_id(self, token: str) -> int | None:
        return self._special.get(token)

    def encode(self, text: str) -> TokenEncoding:
        if self.mode == "empty":
            return TokenEncoding(ids=(), offsets=())
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            matched = next(
                (token for token in self._special if text.startswith(token, index)), None
            )
            if matched is not None:
                ids.append(self._special[matched])
                offsets.append((index, index + len(matched)))
                index += len(matched)
                continue
            ids.append(10 + ord(text[index]) % 400)
            offsets.append((index, index + 1))
            index += 1
        if self.mode == "missing_offset":
            offsets.pop()
        elif self.mode == "out_of_bounds":
            offsets[-1] = (offsets[-1][0], len(text) + 1)
        elif self.mode == "bad_token_id":
            ids[-1] = self.vocab_size
        elif self.mode == "cross_boundary":
            assistant = text.index("assistant\n") + len("assistant\n")
            token_index = next(
                index for index, (start, _end) in enumerate(offsets) if start == assistant
            )
            offsets[token_index] = (assistant - 1, assistant + 1)
        elif self.mode == "omit_eos":
            ids = [99 if token_id == 2 else token_id for token_id in ids]
        return TokenEncoding(ids=tuple(ids), offsets=tuple(offsets))


def processed_sample(
    suffix: str = "sample",
    *,
    messages: tuple[ImportedMessage, ...] | None = None,
) -> ProcessedSample:
    source = ImportedSample(
        id=f"oasst1:{suffix}",
        source="oasst1",
        messages=messages
        or (
            ImportedMessage(role="user", content="question"),
            ImportedMessage(role="assistant", content="answer"),
        ),
        metadata=ImportedSampleMetadata(
            language="en",
            category="conversation",
            license="apache-2.0",
            source_revision=OASST1_SOURCE.revision,
            source_record_id=f"record-{suffix}",
            group_ids=(f"tree-{suffix}",),
            raw_record_sha256s=(hashlib.sha256(suffix.encode()).hexdigest(),),
        ),
    )
    result = process_imported_samples([source], config=load_m2_processing_config(PROCESSING_CONFIG))
    return result.samples[0]


def fake_config(*, max_sequence_length: int = 1024) -> M2TokenizationConfig:
    raw = load_m2_tokenization_config(TOKENIZATION_CONFIG).to_dict()
    raw["max_sequence_length"] = max_sequence_length
    raw["tokenizer"].update(
        {
            "tokenizer_file_size": 1,
            "tokenizer_sha256": "a" * 64,
            "tokenizer_config_file_size": 1,
            "tokenizer_config_sha256": "b" * 64,
            "vocab_size": 512,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    )
    return M2TokenizationConfig.model_validate(raw)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_qwen_tokenizer_identity_and_template_are_frozen(tmp_path: Path) -> None:
    config = load_m2_tokenization_config(TOKENIZATION_CONFIG)

    assert config.tokenizer.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config.tokenizer.tokenizer_sha256 == (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    )
    assert config.tokenizer.vocab_size == 151_669
    assert config.tokenizer.eos_token_id == 151_645
    assert config.template.template_sha256 == QWEN3_NONTHINKING_TEMPLATE_SHA256
    assert config.template.thinking is False

    with pytest.raises(TokenizerContractError, match="extension"):
        load_m2_tokenization_config(tmp_path / "config.json")
    with pytest.raises(TokenizerContractError, match="cannot read"):
        load_m2_tokenization_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(TokenizerContractError, match="invalid YAML"):
        load_m2_tokenization_config(invalid)
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(TokenizerContractError, match="tokenizer"):
        load_m2_tokenization_config(incomplete)


def test_nonthinking_render_is_exact_and_marks_assistant_content_plus_end() -> None:
    messages = (
        ImportedMessage(role="system", content="system text"),
        ImportedMessage(role="user", content="question"),
        ImportedMessage(role="assistant", content="answer"),
    )

    rendered = render_qwen3_nonthinking(messages)

    assert rendered.text == (
        "<|im_start|>system\nsystem text<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\nanswer<|im_end|>\n"
    )
    assert "<think>" not in rendered.text
    assert len(rendered.assistant_spans) == 1
    start, end = rendered.assistant_spans[0]
    assert rendered.text[start:end] == "answer<|im_end|>"


def test_thinking_render_is_exact_and_supervises_trace_answer_and_end() -> None:
    messages = (
        ImportedMessage(role="system", content="system text"),
        ImportedMessage(role="user", content="question"),
        ImportedMessage(role="assistant", content="answer"),
    )
    backend = CharacterTokenizer()
    tokenizer = fake_config().tokenizer

    rendered = render_qwen3_thinking(messages, assistant_reasoning=("reason",))
    tokenized = tokenize_thinking_messages(
        messages,
        assistant_reasoning=("reason",),
        backend=backend,
        tokenizer=tokenizer,
    )

    assert QWEN3_THINKING_TEMPLATE_SHA256 == (
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    )
    assert rendered.text == (
        "<|im_start|>system\nsystem text<|im_end|>\n"
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nreason\n</think>\n\nanswer<|im_end|>\n"
    )
    start, end = rendered.assistant_spans[0]
    assert rendered.text[start:end] == "<think>\nreason\n</think>\n\nanswer<|im_end|>"
    encoding = backend.encode(rendered.text)
    supervised_text = "".join(
        rendered.text[offset_start:offset_end]
        for label, (offset_start, offset_end) in zip(
            tokenized.labels, encoding.offsets, strict=True
        )
        if label != -100
    )
    assert supervised_text == "<think>\nreason\n</think>\n\nanswer<|im_end|>"


@pytest.mark.parametrize(
    ("reasoning", "answer", "message"),
    [
        ((), "answer", "count"),
        (("   ",), "answer", "non-empty"),
        (("<think>nested",), "answer", "nested"),
        (("reason",), "answer</think>", "nested"),
    ],
)
def test_thinking_render_rejects_ambiguous_or_empty_payloads(
    reasoning: tuple[str, ...],
    answer: str,
    message: str,
) -> None:
    messages = (
        ImportedMessage(role="user", content="question"),
        ImportedMessage(role="assistant", content=answer),
    )

    with pytest.raises(TokenizerContractError, match=message):
        render_qwen3_thinking(messages, assistant_reasoning=reasoning)


def test_thinking_render_requires_an_assistant_response() -> None:
    with pytest.raises(TokenizerContractError, match="at least one Assistant"):
        render_qwen3_thinking(
            (ImportedMessage(role="user", content="question"),),
            assistant_reasoning=(),
        )


def test_assistant_only_labels_mask_headers_user_and_trailing_newline() -> None:
    sample = processed_sample()
    backend = CharacterTokenizer()
    config = fake_config()

    result = tokenize_processed_sample(sample, backend=backend, config=config)

    assert isinstance(result, TokenizedSample)
    rendered = render_qwen3_nonthinking(sample.messages)
    encoding = backend.encode(rendered.text)
    supervised_text = "".join(
        rendered.text[start:end]
        for label, (start, end) in zip(result.labels, encoding.offsets, strict=True)
        if label != -100
    )
    assert supervised_text == "answer<|im_end|>"
    assert result.labels.count(config.tokenizer.eos_token_id) == 1
    assert result.supervised_token_count == len("answer") + 1
    assert result.rendered_sha256 == hashlib.sha256(rendered.text.encode()).hexdigest()


def test_all_assistant_turns_include_a_supervised_end_token() -> None:
    sample = processed_sample(
        "multi-turn",
        messages=(
            ImportedMessage(role="user", content="q1"),
            ImportedMessage(role="assistant", content="a1"),
            ImportedMessage(role="user", content="q2"),
            ImportedMessage(role="assistant", content="a2"),
        ),
    )
    config = fake_config()

    result = tokenize_processed_sample(sample, backend=CharacterTokenizer(), config=config)

    assert isinstance(result, TokenizedSample)
    assert result.labels.count(config.tokenizer.eos_token_id) == 2


def test_data_level_overlength_and_no_supervision_rejections_are_content_free() -> None:
    private_content = "private-content-must-not-leak"
    sample = processed_sample(
        "rejected",
        messages=(
            ImportedMessage(role="user", content=private_content),
            ImportedMessage(role="assistant", content="answer"),
        ),
    )
    overlength = tokenize_processed_sample(
        sample, backend=CharacterTokenizer(), config=fake_config(max_sequence_length=10)
    )
    no_labels = tokenize_processed_sample(
        sample, backend=CharacterTokenizer(mode="empty"), config=fake_config()
    )

    assert isinstance(overlength, TokenizationRejectedRecord)
    assert overlength.reason == "sequence_too_long"
    assert isinstance(no_labels, TokenizationRejectedRecord)
    assert no_labels.reason == "no_supervised_tokens"
    assert private_content not in overlength.model_dump_json()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_offset", "counts differ"),
        ("out_of_bounds", "invalid or non-monotonic"),
        ("bad_token_id", "outside"),
        ("cross_boundary", "crosses"),
        ("omit_eos", "end token"),
    ],
)
def test_tokenizer_contract_failures_abort_instead_of_skipping(mode: str, message: str) -> None:
    with pytest.raises(TokenizerContractError, match=message):
        tokenize_processed_sample(
            processed_sample(mode), backend=CharacterTokenizer(mode=mode), config=fake_config()
        )


def test_batch_is_deterministic_and_rejects_duplicate_sample_ids() -> None:
    first = processed_sample("a")
    second = processed_sample("b")
    config = fake_config()
    backend = CharacterTokenizer()

    forward = tokenize_processed_samples([second, first], backend=backend, config=config)
    reverse = tokenize_processed_samples([first, second], backend=backend, config=config)

    assert forward == reverse
    assert [sample.id for sample in forward.samples] == [first.id, second.id]
    with pytest.raises(TokenizerContractError, match="unique"):
        tokenize_processed_samples([first, first], backend=backend, config=config)


def test_backend_checks_local_files_hash_vocab_and_special_tokens(tmp_path: Path) -> None:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<|endoftext|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "[UNK]": 3,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")

    raw = load_m2_tokenization_config(TOKENIZATION_CONFIG).tokenizer.to_dict()
    raw.update(
        {
            "tokenizer_file_size": tokenizer_path.stat().st_size,
            "tokenizer_sha256": sha256(tokenizer_path),
            "tokenizer_config_file_size": config_path.stat().st_size,
            "tokenizer_config_sha256": sha256(config_path),
            "vocab_size": 4,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    )
    identity = TokenizerIdentity.model_validate(raw)

    backend = TokenizersBackend.from_files(tokenizer_path, config_path, identity)
    assert backend.vocab_size == 4
    assert backend.token_to_id("<|im_end|>") == 2

    bad_identity = TokenizerIdentity.model_validate({**raw, "tokenizer_sha256": "0" * 64})
    with pytest.raises(TokenizerContractError, match="SHA256"):
        TokenizersBackend.from_files(tokenizer_path, config_path, bad_identity)
    config_path.write_text("changed", encoding="utf-8")
    with pytest.raises(TokenizerContractError, match="config size"):
        TokenizersBackend.from_files(tokenizer_path, config_path, identity)


def test_tokenized_and_rejection_schemas_refuse_inconsistent_counts() -> None:
    tokenized = tokenize_processed_sample(
        processed_sample("schema"), backend=CharacterTokenizer(), config=fake_config()
    )
    assert isinstance(tokenized, TokenizedSample)
    with pytest.raises(ValidationError, match="equal lengths"):
        TokenizedSample.model_validate({**tokenized.model_dump(), "labels": (-100,)})
    with pytest.raises(ValidationError, match="input token IDs"):
        TokenizedSample.model_validate(
            {**tokenized.model_dump(), "input_ids": (-1, *tokenized.input_ids[1:])}
        )
    with pytest.raises(ValidationError, match="above the maximum"):
        TokenizationRejectedRecord(
            sample_id="oasst1:short",
            source="oasst1",
            split="train",
            content_sha256="a" * 64,
            reason="sequence_too_long",
            observed_token_count=10,
            max_sequence_length=10,
        )
