from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    DataProcessingError,
    DataProcessingManifest,
    DeduplicationConfig,
    GroupedSplitConfig,
    ImportedMessage,
    ImportedSample,
    ImportedSampleMetadata,
    M2ProcessingConfig,
    NormalizationConfig,
    PipelineRejectedRecord,
    load_m2_processing_config,
    process_imported_samples,
)

CONFIG_PATH = Path("configs/data/m2_processing.yaml")


def imported_sample(
    suffix: str,
    *,
    source: str = "oasst1",
    groups: tuple[str, ...] = ("group-a",),
    user: str = "question",
    assistant: str = "answer",
) -> ImportedSample:
    is_oasst = source == "oasst1"
    revision = OASST1_SOURCE.revision if is_oasst else COMMITPACKFT_SOURCE.revision
    return ImportedSample(
        id=f"{source}:{suffix}",
        source=source,  # type: ignore[arg-type]
        messages=(
            ImportedMessage(role="user", content=user),
            ImportedMessage(role="assistant", content=assistant),
        ),
        metadata=ImportedSampleMetadata(
            language="en",
            category="conversation" if is_oasst else "code_edit",
            license="apache-2.0" if is_oasst else "mit",
            source_revision=revision,
            source_record_id=f"record-{suffix}",
            group_ids=groups,
            raw_record_sha256s=(hashlib.sha256(suffix.encode()).hexdigest(),),
        ),
    )


def processing_config(**normalization_overrides: int) -> M2ProcessingConfig:
    raw = load_m2_processing_config(CONFIG_PATH).to_dict()
    raw["normalization"].update(normalization_overrides)
    return M2ProcessingConfig.model_validate(raw)


def test_formal_processing_config_is_explicit_and_strict(tmp_path: Path) -> None:
    config = load_m2_processing_config(CONFIG_PATH)

    assert config.split.seed == 42
    assert config.split.train_basis_points == 9_800
    assert config.deduplication.near is False
    assert config.normalization.unicode_form == "nfc"

    with pytest.raises(ValidationError, match="sum to 10000"):
        GroupedSplitConfig(
            seed=42,
            train_basis_points=9_000,
            validation_basis_points=100,
            test_basis_points=100,
        )
    with pytest.raises(ValidationError, match="source priority"):
        DeduplicationConfig(
            exact=True,
            near=False,
            source_priority=("oasst1", "oasst1"),
        )
    with pytest.raises(ValidationError, match="max sample"):
        NormalizationConfig(
            unicode_form="nfc",
            normalize_line_endings=True,
            strip_bom=True,
            trim_outer_whitespace=True,
            reject_control_characters=True,
            max_message_chars=100,
            max_sample_chars=10,
        )

    with pytest.raises(DataProcessingError, match="extension"):
        load_m2_processing_config(tmp_path / "config.json")
    missing = tmp_path / "missing.yaml"
    with pytest.raises(DataProcessingError, match="cannot read"):
        load_m2_processing_config(missing)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(DataProcessingError, match="invalid YAML"):
        load_m2_processing_config(invalid)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("schema_version: '1.0'\nunknown: true\n", encoding="utf-8")
    with pytest.raises(DataProcessingError, match="normalization"):
        load_m2_processing_config(unknown)


def test_normalization_is_nfc_lf_and_preserves_internal_whitespace() -> None:
    sample = imported_sample(
        "normalize",
        user="\ufeff  Cafe\u0301\r\nline\t  \r\n ",
        assistant="  answer\r\n  indented  ",
    )

    result = process_imported_samples([sample], config=processing_config())

    assert result.samples[0].messages[0].content == "Café\nline"
    assert result.samples[0].messages[1].content == "answer\n  indented"
    assert result.manifest.normalization_rejections == 0


@pytest.mark.parametrize(
    ("user", "overrides", "reason"),
    [
        ("\ufeff ", {}, "empty_after_normalization"),
        ("bad\x00value", {}, "forbidden_control_character"),
        ("123456", {"max_message_chars": 5, "max_sample_chars": 20}, "message_too_long"),
        ("123456", {"max_message_chars": 10, "max_sample_chars": 10}, "sample_too_long"),
    ],
)
def test_normalization_failure_reasons_retain_no_content(
    user: str, overrides: dict[str, int], reason: str
) -> None:
    result = process_imported_samples(
        [imported_sample("rejected", user=user, assistant="123456")],
        config=processing_config(**overrides),
    )

    assert result.samples == ()
    assert result.rejected[0].reason == reason
    assert user not in result.rejected[0].model_dump_json()
    assert result.manifest.output_samples == 0
    assert result.manifest.component_count == 0


def test_exact_dedup_prefers_oasst_and_merges_all_lineage_and_groups() -> None:
    commit = imported_sample(
        "duplicate-commit",
        source="commitpackft",
        groups=("repository/a", "repository/shared"),
        user="same question",
        assistant="same answer",
    )
    oasst = imported_sample(
        "duplicate-oasst",
        groups=("tree/a",),
        user="same question",
        assistant="same answer",
    )

    result = process_imported_samples([commit, oasst], config=processing_config())

    assert len(result.samples) == 1
    kept = result.samples[0]
    assert kept.id == oasst.id
    assert kept.origin_sample_ids == tuple(sorted((commit.id, oasst.id)))
    assert kept.group_keys == (
        "commitpackft:repository/a",
        "commitpackft:repository/shared",
        "oasst1:tree/a",
    )
    assert len(kept.origin_record_sha256s) == 2
    assert result.rejected[0].reason == "exact_duplicate"
    assert result.rejected[0].duplicate_of_sample_id == oasst.id
    assert result.rejected[0].content_sha256 == kept.content_sha256
    assert result.manifest.exact_duplicates == 1


def test_multi_repository_and_duplicate_bridges_form_one_component() -> None:
    duplicate_oasst = imported_sample(
        "bridge-oasst", groups=("tree-bridge",), user="shared", assistant="shared"
    )
    duplicate_commit = imported_sample(
        "bridge-commit",
        source="commitpackft",
        groups=("repo-a",),
        user="shared",
        assistant="shared",
    )
    multi_repo = imported_sample(
        "multi-repo",
        source="commitpackft",
        groups=("repo-a", "repo-b"),
        user="different-a",
        assistant="answer-a",
    )
    repo_b = imported_sample(
        "repo-b",
        source="commitpackft",
        groups=("repo-b",),
        user="different-b",
        assistant="answer-b",
    )

    result = process_imported_samples(
        [repo_b, multi_repo, duplicate_commit, duplicate_oasst],
        config=processing_config(),
    )

    assert len(result.samples) == 3
    assert len({sample.component_id for sample in result.samples}) == 1
    assert len({sample.split for sample in result.samples}) == 1
    assert result.manifest.component_count == 1


def test_determinism_is_independent_of_input_iteration_order() -> None:
    samples = [
        imported_sample("c", groups=("tree-c",), user="q3", assistant="a3"),
        imported_sample("a", groups=("tree-a",), user="q1", assistant="a1"),
        imported_sample("b", groups=("tree-b",), user="q2", assistant="a2"),
    ]
    config = processing_config()

    forward = process_imported_samples(samples, config=config)
    reverse = process_imported_samples(reversed(samples), config=config)

    assert forward == reverse
    assert forward.manifest.input_sha256 == reverse.manifest.input_sha256
    assert forward.manifest.output_sha256 == reverse.manifest.output_sha256
    assert forward.manifest.split_sha256s == reverse.manifest.split_sha256s


def test_no_group_key_crosses_splits_across_many_components() -> None:
    samples = [
        imported_sample(
            f"sample-{index}",
            groups=(f"tree-{index // 2}",),
            user=f"question {index}",
            assistant=f"answer {index}",
        )
        for index in range(40)
    ]

    result = process_imported_samples(samples, config=processing_config())
    splits_by_group: dict[str, set[str]] = {}
    for sample in result.samples:
        for group_key in sample.group_keys:
            splits_by_group.setdefault(group_key, set()).add(sample.split)

    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert result.manifest.component_count == 20
    assert sum(result.manifest.split_counts.values()) == 40


def test_duplicate_imported_ids_are_an_integrity_error() -> None:
    sample = imported_sample("duplicate-id")

    with pytest.raises(DataProcessingError, match="duplicate imported sample ID"):
        process_imported_samples([sample, sample], config=processing_config())


def test_pipeline_rejection_and_manifest_schemas_guard_evidence() -> None:
    with pytest.raises(ValidationError, match="requires content hash"):
        PipelineRejectedRecord(
            sample_id="oasst1:duplicate",
            source="oasst1",
            reason="exact_duplicate",
        )
    with pytest.raises(ValidationError, match="only exact duplicates"):
        PipelineRejectedRecord(
            sample_id="oasst1:bad",
            source="oasst1",
            reason="message_too_long",
            duplicate_of_sample_id="oasst1:kept",
        )
    with pytest.raises(ValidationError, match="message index"):
        PipelineRejectedRecord(
            sample_id="oasst1:bad",
            source="oasst1",
            reason="message_too_long",
        )

    valid_result = process_imported_samples(
        [imported_sample("manifest")], config=processing_config()
    )
    valid = valid_result.manifest
    with pytest.raises(ValidationError, match="content hash"):
        type(valid_result.samples[0]).model_validate(
            {**valid_result.samples[0].model_dump(), "content_sha256": "0" * 64}
        )
    with pytest.raises(ValidationError, match="output and rejected"):
        DataProcessingManifest.model_validate({**valid.model_dump(), "input_samples": 2})
