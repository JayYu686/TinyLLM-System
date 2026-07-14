from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests.unit.test_data_packing import balanced_samples, build, digest
from tests.unit.test_evaluation_schema import evaluation_config, evaluation_item
from tinyllm.data import (
    ImportedMessage,
    M2DatasetManifest,
    OffsetTokenizer,
    PackedSequence,
    TokenEncoding,
    TokenizedSample,
    load_m2_packing_config,
    load_m2_tokenization_config,
    pack_tokenized_samples,
    tokenize_messages,
)
from tinyllm.evaluation import (
    EvaluationBuildConfig,
    EvaluationContractError,
    build_evaluation_manifest,
    fingerprint_token_sequence,
    load_evaluation_build_config,
    load_evaluation_items,
    run_contamination_check,
    scan_exact_contamination,
)

TOKENIZATION_CONFIG = Path("configs/data/m2_tokenization.yaml")
PACKING_CONFIG = Path("configs/data/m2_packing.yaml")


class PinnedCharacterTokenizer(OffsetTokenizer):
    """Offset-preserving test backend with the formal Qwen identity surface."""

    def __init__(self) -> None:
        self._special = {
            "<|endoftext|>": 151_643,
            "<|im_start|>": 151_644,
            "<|im_end|>": 151_645,
        }

    @property
    def vocab_size(self) -> int:
        return 151_669

    def token_to_id(self, token: str) -> int | None:
        return self._special.get(token)

    def encode(self, text: str) -> TokenEncoding:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            matched = next(
                (token for token in self._special if text.startswith(token, index)),
                None,
            )
            if matched is not None:
                ids.append(self._special[matched])
                offsets.append((index, index + len(matched)))
                index += len(matched)
                continue
            ids.append(100 + ord(text[index]))
            offsets.append((index, index + 1))
            index += 1
        return TokenEncoding(ids=tuple(ids), offsets=tuple(offsets))


@dataclass(frozen=True, slots=True)
class FakeVerifiedDataset:
    manifest: M2DatasetManifest
    packs: tuple[PackedSequence, ...]

    def iter_packs(self) -> Iterator[PackedSequence]:
        yield from self.packs


def training_pack(
    *,
    prompt: str,
    answer: str,
    sample_id: str = "oasst1:private-record-17",
) -> PackedSequence:
    tokenization_config = load_m2_tokenization_config(TOKENIZATION_CONFIG)
    tokens = tokenize_messages(
        (
            ImportedMessage(role="user", content=prompt),
            ImportedMessage(role="assistant", content=answer),
        ),
        backend=PinnedCharacterTokenizer(),
        config=tokenization_config,
    )
    sample = TokenizedSample(
        id=sample_id,
        source="oasst1",
        split="train",
        component_id=digest(f"component-{sample_id}"),
        group_keys=(f"oasst1:group-{sample_id.split(':', 1)[1]}",),
        origin_sample_ids=(sample_id,),
        origin_record_sha256s=(digest(f"record-{sample_id}"),),
        language="en",
        license="apache-2.0",
        content_sha256=digest(f"content-{sample_id}"),
        rendered_sha256=tokens.rendered_sha256,
        tokenizer_sha256=tokenization_config.tokenizer.tokenizer_sha256,
        template_sha256=tokenization_config.template.template_sha256,
        max_sequence_length=tokenization_config.max_sequence_length,
        input_ids=tokens.input_ids,
        labels=tokens.labels,
        token_count=len(tokens.input_ids),
        supervised_token_count=sum(label != -100 for label in tokens.labels),
    )
    return pack_tokenized_samples(
        (sample,),
        config=load_m2_packing_config(PACKING_CONFIG),
    )[0]


def fake_dataset(*packs: PackedSequence) -> FakeVerifiedDataset:
    return FakeVerifiedDataset(
        manifest=build(balanced_samples()).manifest,
        packs=tuple(packs),
    )


def test_token_fingerprint_encoding_is_length_delimited_and_strict() -> None:
    expected = hashlib.sha256(
        (3).to_bytes(8, "big") + struct.pack(">III", 1, 256, 151_645)
    ).hexdigest()

    assert fingerprint_token_sequence((1, 256, 151_645)) == expected
    assert fingerprint_token_sequence((1, 23)) != fingerprint_token_sequence((12, 3))
    with pytest.raises(EvaluationContractError, match="cannot be empty"):
        fingerprint_token_sequence(())
    with pytest.raises(EvaluationContractError, match="outside UInt32"):
        fingerprint_token_sequence((-1,))


def test_manifest_is_order_independent_and_binds_config_and_counts() -> None:
    items = (
        evaluation_item(1, prompt="First prompt", answer="first"),
        evaluation_item(2, prompt="Second prompt", answer="second"),
    )
    config = evaluation_config(expected_items=2)

    first = build_evaluation_manifest(items, config=config)
    second = build_evaluation_manifest(reversed(items), config=config)

    assert first == second
    assert first.suite_version.endswith(first.content_sha256[:8])
    assert first.item_count == 2
    with pytest.raises(EvaluationContractError, match="item count"):
        build_evaluation_manifest(items[:1], config=config)


def test_exact_scanner_detects_full_and_prompt_only_without_private_ids() -> None:
    exact = evaluation_item(1, prompt="Shared prompt", answer="training answer")
    prompt_only = evaluation_item(2, prompt="Shared prompt", answer="different answer")
    clean = evaluation_item(3, prompt="Clean prompt", answer="clean answer")
    items = (clean, prompt_only, exact)
    config = evaluation_config(expected_items=3)
    manifest = build_evaluation_manifest(items, config=config)
    dataset = fake_dataset(training_pack(prompt="Shared prompt", answer="training answer"))

    report = scan_exact_contamination(
        dataset,
        items,
        manifest=manifest,
        backend=PinnedCharacterTokenizer(),
        tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
    )

    assert report.status == "contaminated"
    assert report.checked_training_samples == 1
    assert report.contaminated_items == 2
    assert report.full_sequence_matches == 1
    assert report.prompt_prefix_matches == 2
    assert report.near_dedup == "not_evaluated"
    assert {match.evaluation_item_id for match in report.matches} == {
        exact.id,
        prompt_only.id,
    }
    serialized = report.model_dump_json()
    assert "oasst1:private-record-17" not in serialized
    assert hashlib.sha256(b"oasst1:private-record-17").hexdigest() in serialized


def test_clean_scanner_is_deterministic_and_reports_zero_matches() -> None:
    clean = evaluation_item(prompt="No matching prompt", answer="new answer")
    manifest = build_evaluation_manifest((clean,), config=evaluation_config())
    dataset = fake_dataset(training_pack(prompt="Training prompt", answer="training answer"))

    first = scan_exact_contamination(
        dataset,
        (clean,),
        manifest=manifest,
        backend=PinnedCharacterTokenizer(),
        tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
    )
    second = scan_exact_contamination(
        dataset,
        (clean,),
        manifest=manifest,
        backend=PinnedCharacterTokenizer(),
        tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
    )

    assert first == second
    assert first.status == "clean"
    assert first.matches == ()
    assert first.contaminated_items == 0


def test_scanner_refuses_duplicate_train_samples_or_identity_drift() -> None:
    item = evaluation_item(prompt="Prompt", answer="answer")
    manifest = build_evaluation_manifest((item,), config=evaluation_config())
    pack = training_pack(prompt="Prompt", answer="answer")

    with pytest.raises(EvaluationContractError, match="more than once"):
        scan_exact_contamination(
            fake_dataset(pack, pack),
            (item,),
            manifest=manifest,
            backend=PinnedCharacterTokenizer(),
            tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
        )

    changed_item = evaluation_item(prompt="Changed prompt", answer="answer")
    with pytest.raises(EvaluationContractError, match="content does not match"):
        scan_exact_contamination(
            fake_dataset(pack),
            (changed_item,),
            manifest=manifest,
            backend=PinnedCharacterTokenizer(),
            tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
        )

    raw = evaluation_config().to_dict()
    raw["max_sequence_length"] = 2048
    incompatible_config = EvaluationBuildConfig.model_validate(raw)
    incompatible_manifest = build_evaluation_manifest((item,), config=incompatible_config)
    with pytest.raises(EvaluationContractError, match="maximum lengths"):
        scan_exact_contamination(
            fake_dataset(pack),
            (item,),
            manifest=incompatible_manifest,
            backend=PinnedCharacterTokenizer(),
            tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
        )


def test_jsonl_and_yaml_loaders_are_strict_deterministic_and_content_safe(
    tmp_path: Path,
) -> None:
    first = evaluation_item(1)
    second = evaluation_item(2, prompt="Other", answer="other")
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in (second, first))
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(evaluation_config(expected_items=2).to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    assert load_evaluation_items(jsonl) == (first, second)
    assert load_evaluation_build_config(config_path) == evaluation_config(expected_items=2)

    private = "private-evaluation-content-must-not-leak"
    jsonl.write_text(json.dumps({"prompt": private}) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationContractError) as captured:
        load_evaluation_items(jsonl)
    assert private not in str(captured.value)
    assert "line 1" in str(captured.value)

    config_path.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="invalid YAML"):
        load_evaluation_build_config(config_path)


def test_high_level_check_wires_strict_inputs_verified_registry_and_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = evaluation_item(prompt="Clean high-level prompt", answer="clean")
    evaluation_set = tmp_path / "items.jsonl"
    evaluation_set.write_text(
        json.dumps(item.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(evaluation_config().to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    dataset = fake_dataset(training_pack(prompt="Other prompt", answer="other"))
    cache_calls: list[bool] = []

    def acquire(*_args: object, **kwargs: object) -> Path:
        cache_calls.append(bool(kwargs["offline"]))
        return tmp_path / "verified-tokenizer-artifact"

    class BackendFactory:
        @staticmethod
        def from_files(*_args: object, **_kwargs: object) -> PinnedCharacterTokenizer:
            return PinnedCharacterTokenizer()

    monkeypatch.setattr(
        "tinyllm.evaluation.contamination.open_registered_dataset",
        lambda **_kwargs: dataset,
    )
    monkeypatch.setattr("tinyllm.evaluation.contamination.acquire_pinned_artifact", acquire)
    monkeypatch.setattr("tinyllm.evaluation.contamination.TokenizersBackend", BackendFactory)

    report = run_contamination_check(
        artifact_root=tmp_path,
        dataset_version=dataset.manifest.dataset_version,
        evaluation_set_path=evaluation_set,
        evaluation_config_path=config_path,
        tokenization_config_path=TOKENIZATION_CONFIG,
    )

    assert report.status == "clean"
    assert cache_calls == [True, True]
