from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    DataImportManifest,
    DataProcessingManifest,
    DatasetBuild,
    M2PackingConfig,
    PackedSequence,
    PackingError,
    TokenizationBatch,
    TokenizedSample,
    build_m2_dataset,
    load_m2_packing_config,
    load_m2_tokenization_config,
    pack_tokenized_samples,
)

PACKING_CONFIG = Path("configs/data/m2_packing.yaml")
TOKENIZATION_CONFIG = Path("configs/data/m2_tokenization.yaml")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def packing_config(*, maximum: int = 1024) -> M2PackingConfig:
    raw = load_m2_packing_config(PACKING_CONFIG).to_dict()
    raw["packing"]["max_sequence_length"] = maximum
    return M2PackingConfig.model_validate(raw)


def tokenized_sample(
    source: str,
    suffix: str,
    *,
    language: str,
    token_count: int = 10,
    split: str = "train",
    maximum: int = 1024,
    token_offset: int = 100,
) -> TokenizedSample:
    sample_id = f"{source}:{suffix}"
    input_ids = tuple(range(token_offset, token_offset + token_count))
    labels = (-100,) * (token_count - 1) + (input_ids[-1],)
    tokenizer = load_m2_tokenization_config(TOKENIZATION_CONFIG).tokenizer
    return TokenizedSample.model_validate(
        {
            "id": sample_id,
            "source": source,
            "split": split,
            "component_id": digest(f"component-{sample_id}"),
            "group_keys": (f"{source}:group-{suffix}",),
            "origin_sample_ids": (sample_id,),
            "origin_record_sha256s": (digest(f"record-{sample_id}"),),
            "language": language,
            "license": "apache-2.0" if source == "oasst1" else "mit",
            "content_sha256": digest(f"content-{sample_id}"),
            "rendered_sha256": digest(f"rendered-{sample_id}"),
            "tokenizer_sha256": tokenizer.tokenizer_sha256,
            "template_sha256": load_m2_tokenization_config(
                TOKENIZATION_CONFIG
            ).template.template_sha256,
            "max_sequence_length": maximum,
            "input_ids": input_ids,
            "labels": labels,
            "token_count": token_count,
            "supervised_token_count": 1,
        }
    )


def balanced_samples(*, copies: int = 1) -> tuple[TokenizedSample, ...]:
    samples: list[TokenizedSample] = []
    targets = (("oasst1", "zh", 3), ("oasst1", "en", 3), ("commitpackft", "en", 4))
    for source, language, target_count in targets:
        for index in range(target_count * copies):
            samples.append(
                tokenized_sample(
                    source,
                    f"{language}-{index:02d}",
                    language=language,
                    token_offset=1000 + len(samples) * 20,
                )
            )
    return tuple(samples)


def source_manifest(source: str, *, accepted: int) -> DataImportManifest:
    descriptor = OASST1_SOURCE if source == "oasst1" else COMMITPACKFT_SOURCE
    license_name = "apache-2.0" if source == "oasst1" else "mit"
    return DataImportManifest(
        source=descriptor,
        input_sha256=digest(f"{source}-input-{accepted}"),
        config_sha256=digest(f"{source}-config"),
        source_rows=accepted,
        candidate_samples=accepted,
        accepted_samples=accepted,
        rejected_samples=0,
        rejection_counts={},
        license_counts={license_name: accepted},
    )


def processing_manifest(samples: tuple[TokenizedSample, ...]) -> DataProcessingManifest:
    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in ("test", "train", "validation")
    }
    return DataProcessingManifest(
        input_sha256=digest("processing-input"),
        config_sha256=digest("processing-config"),
        output_sha256=digest("processing-output"),
        input_samples=len(samples),
        normalized_samples=len(samples),
        output_samples=len(samples),
        rejected_samples=0,
        normalization_rejections=0,
        exact_duplicates=0,
        component_count=len(samples),
        rejection_counts={},
        split_counts=split_counts,
        split_sha256s={
            "test": digest("test"),
            "train": digest("train"),
            "validation": digest("validation"),
        },
    )


def build(samples: tuple[TokenizedSample, ...]) -> DatasetBuild:
    source_counts = {
        source: sum(sample.source == source for sample in samples)
        for source in ("commitpackft", "oasst1")
    }
    return build_m2_dataset(
        TokenizationBatch(samples=samples, rejected=()),
        tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
        packing_config=load_m2_packing_config(PACKING_CONFIG),
        processing_manifest=processing_manifest(samples),
        source_manifests=(
            source_manifest("oasst1", accepted=source_counts["oasst1"]),
            source_manifest("commitpackft", accepted=source_counts["commitpackft"]),
        ),
    )


def test_formal_packing_config_is_strict_and_frozen(tmp_path: Path) -> None:
    config = load_m2_packing_config(PACKING_CONFIG)

    assert config.balance.targets() == {
        "commitpackft:en": 4000,
        "oasst1:en": 3000,
        "oasst1:zh": 3000,
    }
    assert config.packing.algorithm == "best-fit-decreasing-v1"
    assert config.packing.reset_position_ids is True
    assert config.packing.pad_to_max_length is False

    with pytest.raises(PackingError, match="extension"):
        load_m2_packing_config(tmp_path / "config.json")
    with pytest.raises(PackingError, match="cannot read"):
        load_m2_packing_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(PackingError, match="invalid YAML"):
        load_m2_packing_config(invalid)
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(PackingError, match="dataset_name"):
        load_m2_packing_config(incomplete)

    raw = config.to_dict()
    raw["balance"]["commitpackft_en_basis_points"] = 3999
    with pytest.raises(ValidationError, match="sum to 10000"):
        M2PackingConfig.model_validate(raw)


def test_best_fit_decreasing_is_split_local_and_boundary_aware() -> None:
    samples = (
        tokenized_sample("oasst1", "a", language="en", token_count=6, maximum=10),
        tokenized_sample("oasst1", "b", language="en", token_count=4, maximum=10),
        tokenized_sample("oasst1", "c", language="en", token_count=4, maximum=10),
        tokenized_sample(
            "commitpackft",
            "d",
            language="en",
            token_count=2,
            split="test",
            maximum=10,
        ),
    )

    packs = pack_tokenized_samples(reversed(samples), config=packing_config(maximum=10))

    train_packs = [pack for pack in packs if pack.split == "train"]
    test_packs = [pack for pack in packs if pack.split == "test"]
    assert sorted(pack.token_count for pack in train_packs) == [4, 10]
    assert len(test_packs) == 1
    assert test_packs[0].sample_ids == ("commitpackft:d",)
    full_train = next(pack for pack in train_packs if pack.token_count == 10)
    assert full_train.sample_token_counts == (6, 4)
    assert full_train.position_ids == (*range(6), *range(4))
    assert full_train.segment_ids == (0,) * 6 + (1,) * 4
    assert all(
        len(pack.input_ids) == len(pack.labels) == len(pack.position_ids) == pack.token_count
        for pack in packs
    )


def test_packing_is_deterministic_and_rejects_identity_or_config_drift() -> None:
    samples = (
        tokenized_sample("oasst1", "a", language="en", maximum=10),
        tokenized_sample("commitpackft", "b", language="en", maximum=10),
    )
    config = packing_config(maximum=10)

    assert pack_tokenized_samples(samples, config=config) == pack_tokenized_samples(
        reversed(samples), config=config
    )
    with pytest.raises(PackingError, match="unique"):
        pack_tokenized_samples((*samples, samples[0]), config=config)
    with pytest.raises(PackingError, match="maximum sequence length"):
        pack_tokenized_samples(
            [tokenized_sample("oasst1", "wrong-max", language="en", maximum=11)],
            config=config,
        )


def test_build_balances_exact_target_and_is_content_addressed() -> None:
    samples = balanced_samples()

    first = build(samples)
    second = build(tuple(reversed(samples)))

    assert first == second
    assert first.manifest.train_stratum_token_counts == {
        "commitpackft:en": 40,
        "oasst1:en": 30,
        "oasst1:zh": 30,
    }
    assert first.manifest.train_stratum_basis_points == {
        "commitpackft:en": 4000,
        "oasst1:en": 3000,
        "oasst1:zh": 3000,
    }
    assert first.manifest.dataset_version.endswith(first.manifest.content_sha256[:8])
    assert first.manifest.balance_rejections == 0
    assert first.manifest.total_tokens == 100
    assert {sample_id for pack in first.packs for sample_id in pack.sample_ids} == {
        sample.id for sample in samples
    }
    assert "created_at" not in type(first.manifest).model_fields
    assert "path" not in type(first.manifest).model_fields

    changed = list(samples)
    changed[0] = TokenizedSample.model_validate(
        {
            **changed[0].model_dump(),
            "input_ids": (*changed[0].input_ids[:-1], changed[0].input_ids[-1] + 1),
            "labels": (*changed[0].labels[:-1], changed[0].labels[-1] + 1),
        }
    )
    assert build(tuple(changed)).manifest.content_sha256 != first.manifest.content_sha256


def test_balance_downsampling_is_auditable_and_within_tolerance() -> None:
    samples = tuple(
        tokenized_sample(source, f"{language}-{index}", language=language)
        for source, language in (
            ("oasst1", "zh"),
            ("oasst1", "en"),
            ("commitpackft", "en"),
        )
        for index in range(5)
    )

    result = build(samples)

    assert result.manifest.balance_rejections == 2
    assert result.manifest.rejection_counts == {"balance_downsampled": 2}
    assert len(result.balance_rejected) == 2
    assert all(record.reason == "balance_downsampled" for record in result.balance_rejected)
    targets = load_m2_packing_config(PACKING_CONFIG).balance.targets()
    for stratum, target in targets.items():
        actual = result.manifest.train_stratum_basis_points[stratum]
        assert abs(actual - target) <= 300


def test_build_refuses_missing_or_unsupported_train_strata() -> None:
    missing_zh = tuple(sample for sample in balanced_samples() if sample.language != "zh")
    with pytest.raises(PackingError, match="missing required Stratum.*oasst1:zh"):
        build(missing_zh)

    unsupported = list(balanced_samples())
    raw = unsupported[0].model_dump()
    raw["language"] = "fr"
    unsupported[0] = TokenizedSample.model_validate(raw)
    with pytest.raises(PackingError, match="outside the frozen"):
        build(tuple(unsupported))


def test_build_refuses_incomplete_lineage_or_stage_counts() -> None:
    samples = balanced_samples()
    processing = processing_manifest(samples)
    tokenization = TokenizationBatch(samples=samples, rejected=())
    config = load_m2_packing_config(PACKING_CONFIG)
    token_config = load_m2_tokenization_config(TOKENIZATION_CONFIG)

    with pytest.raises(PackingError, match="exactly two"):
        build_m2_dataset(
            tokenization,
            tokenization_config=token_config,
            packing_config=config,
            processing_manifest=processing,
            source_manifests=(source_manifest("oasst1", accepted=10),),
        )
    with pytest.raises(PackingError, match="source accepted counts"):
        build_m2_dataset(
            tokenization,
            tokenization_config=token_config,
            packing_config=config,
            processing_manifest=processing,
            source_manifests=(
                source_manifest("oasst1", accepted=5),
                source_manifest("commitpackft", accepted=4),
            ),
        )

    bad_split_manifest = DataProcessingManifest.model_validate(
        {
            **processing.model_dump(),
            "split_counts": {"test": 1, "train": len(samples) - 1, "validation": 0},
        }
    )
    with pytest.raises(PackingError, match="split outcomes"):
        build_m2_dataset(
            tokenization,
            tokenization_config=token_config,
            packing_config=config,
            processing_manifest=bad_split_manifest,
            source_manifests=(
                source_manifest("oasst1", accepted=6),
                source_manifest("commitpackft", accepted=4),
            ),
        )

    drifted = list(samples)
    drifted[0] = TokenizedSample.model_validate(
        {**drifted[0].model_dump(), "max_sequence_length": 1025}
    )
    with pytest.raises(PackingError, match="does not match Tokenizer config"):
        build_m2_dataset(
            TokenizationBatch(samples=tuple(drifted), rejected=()),
            tokenization_config=token_config,
            packing_config=config,
            processing_manifest=processing,
            source_manifests=(
                source_manifest("oasst1", accepted=6),
                source_manifest("commitpackft", accepted=4),
            ),
        )


def test_packed_sequence_schema_detects_corruption() -> None:
    sample = tokenized_sample("oasst1", "schema", language="en", maximum=10)
    pack = pack_tokenized_samples([sample], config=packing_config(maximum=10))[0]

    with pytest.raises(ValidationError, match="content"):
        PackedSequence.model_validate(
            {**pack.model_dump(), "input_ids": (999, *pack.input_ids[1:])}
        )
    with pytest.raises(ValidationError, match="reset"):
        PackedSequence.model_validate(
            {**pack.model_dump(), "position_ids": (1, *pack.position_ids[1:])}
        )
    with pytest.raises(ValidationError, match="segment IDs"):
        PackedSequence.model_validate(
            {**pack.model_dump(), "segment_ids": (1, *pack.segment_ids[1:])}
        )
