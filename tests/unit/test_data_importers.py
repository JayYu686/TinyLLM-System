from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    COMMITPACKFT_LICENSE_ALLOWLIST,
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    CommitPackFTImportConfig,
    DataImportManifest,
    ImportedMessage,
    ImportedSample,
    ImportedSampleMetadata,
    OASST1ImportConfig,
    import_commitpackft,
    import_oasst1,
)

FIXTURES = Path("tests/fixtures/data")


def load_fixture(name: str) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def oasst_row(message_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "message_id": message_id,
        "parent_id": None,
        "text": "prompt",
        "role": "prompter",
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "message_tree_id": "tree-test",
        "tree_state": "ready_for_export",
    }
    row.update(overrides)
    return row


def commit_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "commit": "a" * 40,
        "old_file": "example.py",
        "new_file": "example.py",
        "old_contents": "old = True\n",
        "new_contents": "new = True\n",
        "subject": "Rename the value",
        "lang": "Python",
        "license": "mit",
        "repos": "owner/repository",
    }
    row.update(overrides)
    return row


def test_pinned_sources_and_conservative_license_policy_are_frozen() -> None:
    assert OASST1_SOURCE.revision == "fdf72ae0827c1cda404aff25b6603abec9e3399b"
    assert COMMITPACKFT_SOURCE.revision == "fc56fe33c030c6daa414c2b112c932b8eed085e6"
    assert OASST1_SOURCE.dataset_card_sha256 == (
        "68483ac2fcc2f3f7779f453352363827678070ef35b6d746ccb2ca6958540fff"
    )
    assert {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    } == COMMITPACKFT_LICENSE_ALLOWLIST


def test_oasst_import_accepts_positive_ready_path_and_records_lineage() -> None:
    rows = load_fixture("oasst1.synthetic.json")
    first = import_oasst1(rows)
    second = import_oasst1(rows)

    assert first == second
    assert len(first.samples) == 1
    sample = first.samples[0]
    assert [message.role for message in sample.messages] == ["user", "assistant"]
    assert sample.metadata.group_ids == ("tree-alpha",)
    assert len(sample.metadata.raw_record_sha256s) == 2
    assert sample.metadata.license == "apache-2.0"
    assert first.manifest.source == OASST1_SOURCE
    assert first.manifest.source_rows == 3
    assert first.manifest.candidate_samples == 2
    assert first.manifest.rejection_counts == {"review_not_positive": 1}
    assert first.manifest.license_counts == {"apache-2.0": 1}
    assert len(first.manifest.input_sha256) == 64
    assert len(first.manifest.config_sha256) == 64


@pytest.mark.parametrize(
    ("answer_overrides", "reason"),
    [
        ({"tree_state": "prompt_lottery_waiting"}, "not_ready"),
        ({"deleted": True}, "deleted"),
        ({"review_result": False}, "review_not_positive"),
        ({"lang": "de"}, "unsupported_language"),
        ({"text": "  "}, "empty_content"),
        ({"parent_id": "missing"}, "missing_parent"),
    ],
)
def test_oasst_path_filter_reasons_are_stable(
    answer_overrides: dict[str, object], reason: str
) -> None:
    prompt = oasst_row("prompt")
    answer_fields: dict[str, object] = {
        "parent_id": "prompt",
        "text": "answer",
        "role": "assistant",
    }
    answer_fields.update(answer_overrides)
    answer = oasst_row("answer", **answer_fields)
    result = import_oasst1([prompt, answer])

    assert result.samples == ()
    assert result.manifest.rejection_counts == {reason: 1}


def test_oasst_rejects_malformed_duplicate_role_and_invalid_chains() -> None:
    malformed = oasst_row("malformed")
    malformed.pop("review_result")
    duplicate_a = oasst_row("duplicate")
    duplicate_b = oasst_row("duplicate", text="other")
    unsupported = oasst_row("unsupported", role="tool")
    bad_root = oasst_row("bad-root", role="assistant")

    result = import_oasst1(
        [malformed, duplicate_a, duplicate_b, unsupported, bad_root, {"message_id": "tiny"}]
    )

    assert result.manifest.rejection_counts == {
        "duplicate_source_id": 2,
        "invalid_conversation": 1,
        "malformed_row": 2,
        "unsupported_role": 1,
    }
    assert all("prompt" not in record.model_dump_json() for record in result.rejected)


def test_oasst_rejects_cycle_and_tree_identity_drift() -> None:
    cycle_prompt = oasst_row("cycle-prompt", parent_id="cycle-answer")
    cycle_answer = oasst_row(
        "cycle-answer", parent_id="cycle-prompt", role="assistant", text="answer"
    )
    tree_prompt = oasst_row("tree-prompt")
    tree_answer = oasst_row(
        "tree-answer",
        parent_id="tree-prompt",
        role="assistant",
        text="answer",
        message_tree_id="tree-other",
    )

    result = import_oasst1([cycle_prompt, cycle_answer, tree_prompt, tree_answer])

    assert result.manifest.rejection_counts == {"invalid_conversation": 2}


def test_commitpack_import_filters_language_and_per_sample_license() -> None:
    result = import_commitpackft(load_fixture("commitpackft.synthetic.json"))

    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.metadata.group_ids == ("example/alpha", "example/shared")
    assert sample.metadata.license == "mit"
    assert sample.messages[0].content.startswith("Update tool.py")
    assert sample.messages[1].content == "print('new')\n"
    assert result.manifest.source == COMMITPACKFT_SOURCE
    assert result.manifest.rejection_counts == {"not_python": 1, "unsupported_license": 1}
    assert result.manifest.license_counts == {"mit": 1}


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"subject": "  "}, "empty_instruction"),
        ({"new_contents": "\n"}, "empty_content"),
        ({"lang": "Go"}, "not_python"),
        ({"license": "AGPL-3.0"}, "unsupported_license"),
    ],
)
def test_commitpack_filter_reasons_are_stable(overrides: dict[str, object], reason: str) -> None:
    result = import_commitpackft([commit_row(**overrides)])
    assert result.manifest.rejection_counts == {reason: 1}
    assert result.samples == ()


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"commit": None}, "commit"),
        ({"license": None}, "license"),
        ({"new_contents": None}, "new_contents"),
        ({"old_file": None, "new_file": None}, "new_file|old_file"),
    ],
)
def test_commitpack_malformed_rows_are_rejected_without_payload(
    overrides: dict[str, object], field: str
) -> None:
    row = commit_row(**overrides)
    row["private_payload"] = "must-not-appear-in-rejection"
    result = import_commitpackft([row])

    assert result.rejected[0].reason == "malformed_row"
    assert result.rejected[0].field == field
    assert "must-not-appear" not in result.rejected[0].model_dump_json()


def test_commitpack_missing_repository_has_a_distinct_filter_reason() -> None:
    result = import_commitpackft([commit_row(repos=" , ")])

    assert result.rejected[0].reason == "missing_repository"
    assert result.rejected[0].field == "repos"


def test_commitpack_accepts_empty_old_file_for_new_file_and_normalizes_license() -> None:
    result = import_commitpackft(
        [
            commit_row(
                old_contents="",
                old_file=None,
                new_file="new_file.py",
                license="Apache_2.0",
                repos="owner/z, owner/a,owner/z",
            )
        ]
    )

    assert result.samples[0].metadata.license == "apache-2.0"
    assert result.samples[0].metadata.group_ids == ("owner/a", "owner/z")
    assert "Current file:\n" in result.samples[0].messages[0].content


def test_import_config_and_public_schemas_are_strict() -> None:
    with pytest.raises(ValidationError, match="allowed languages"):
        OASST1ImportConfig(allowed_languages=("zh", "en"))
    with pytest.raises(ValidationError, match="en/zh"):
        OASST1ImportConfig(allowed_languages=("de",))
    with pytest.raises(ValidationError, match="allowed licenses"):
        CommitPackFTImportConfig(allowed_licenses=("MIT",))
    with pytest.raises(ValidationError, match="reviewed M2 allowlist"):
        CommitPackFTImportConfig(allowed_licenses=("agpl-3.0",))
    with pytest.raises(ValidationError, match="Extra inputs"):
        ImportedMessage.model_validate({"role": "user", "content": "hello", "extra": True})
    with pytest.raises(ValidationError, match="alternate"):
        ImportedSample(
            id="oasst1:abc",
            source="oasst1",
            messages=(
                ImportedMessage(role="user", content="one"),
                ImportedMessage(role="assistant", content="two"),
                ImportedMessage(role="assistant", content="three"),
            ),
            metadata=ImportedSampleMetadata(
                language="en",
                category="conversation",
                license="apache-2.0",
                source_revision=OASST1_SOURCE.revision,
                source_record_id="record",
                group_ids=("group",),
                raw_record_sha256s=("a" * 64,),
            ),
        )


def test_manifest_rejects_inconsistent_or_unsorted_counts() -> None:
    base = {
        "source": OASST1_SOURCE,
        "input_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "source_rows": 1,
        "candidate_samples": 1,
        "accepted_samples": 0,
        "rejected_samples": 1,
        "rejection_counts": {"z": 1, "a": 1},
        "license_counts": {},
    }
    with pytest.raises(ValidationError, match="sorted"):
        DataImportManifest.model_validate(base)
    with pytest.raises(ValidationError, match="candidate"):
        DataImportManifest.model_validate(
            {**base, "candidate_samples": 2, "rejection_counts": {"malformed_row": 1}}
        )


def test_input_hash_changes_with_input_order_and_content() -> None:
    first = commit_row(commit="a" * 40)
    second = commit_row(commit="b" * 40)
    original = import_commitpackft([first, second]).manifest.input_sha256
    reordered = import_commitpackft([second, first]).manifest.input_sha256
    changed = import_commitpackft([first, commit_row(commit="c" * 40)]).manifest.input_sha256

    assert len({original, reordered, changed}) == 3
