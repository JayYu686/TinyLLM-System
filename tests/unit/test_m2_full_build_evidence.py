from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def test_full_m2_build_evidence_is_internally_consistent() -> None:
    evidence = cast(
        dict[str, Any],
        json.loads(Path("reports/m2/raw/full_dataset_build.json").read_text(encoding="utf-8")),
    )
    dataset = evidence["dataset"]
    rejections = dataset["rejection_counts"]

    assert evidence["status"] == "pass"
    assert evidence["code"]["git_dirty"] is False
    assert dataset["dataset_version"].endswith(dataset["content_sha256"][:8])
    assert sum(dataset["source_token_counts"].values()) == dataset["total_tokens"]
    assert sum(dataset["language_token_counts"].values()) == dataset["total_tokens"]
    assert sum(dataset["split_token_counts"].values()) == dataset["total_tokens"]
    assert (
        sum(dataset["split_supervised_token_counts"].values()) == dataset["total_supervised_tokens"]
    )
    assert sum(dataset["split_sample_counts"].values()) == dataset["balanced_samples"]
    assert sum(dataset["split_pack_counts"].values()) == dataset["packed_sequences"]
    assert (
        sum(dataset["train_stratum_token_counts"].values())
        == dataset["split_token_counts"]["train"]
    )
    assert dataset["processed_samples"] == (
        sum(dataset["imported_samples"].values())
        - rejections["processing.exact_duplicate"]
        - rejections["processing.forbidden_control_character"]
    )
    assert dataset["tokenized_samples"] == (
        dataset["processed_samples"] - rejections["tokenization.sequence_too_long"]
    )
    assert dataset["balanced_samples"] == (
        dataset["tokenized_samples"] - rejections["balance.balance_downsampled"]
    )
    assert [run["created"] for run in evidence["runs"]] == [True, False]
    assert all(run["exit_status"] == 0 for run in evidence["runs"])
    assert all(run["verified"] is True for run in evidence["runs"])
    assert evidence["runs"][1]["filesystem_outputs"] == 0
    assert all(evidence["verification"].values())
