from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from tinyllm.data.m10_devops import (
    REPAIR_CATEGORY_COUNTS,
    REPAIR_LANGUAGE_COUNTS,
    REPAIR_V4_CATEGORY_COUNTS,
    REPAIR_V4_LANGUAGE_COUNTS,
    build_manifest,
    render_samples,
    scan_authored_duplicates,
)
from tinyllm.data.m10_devops_schema import M10DevOpsContentReviewResult
from tinyllm.data.m10_repair import (
    build_repair_samples,
    build_repair_v3_samples,
    build_repair_v4_samples,
    validate_repair_samples,
    validate_repair_v3_samples,
    validate_repair_v4_samples,
)


def test_repair_source_is_deterministic_and_fact_grounded() -> None:
    samples = build_repair_samples()
    rebuilt = build_repair_samples()
    quality = validate_repair_samples(samples)

    assert render_samples(samples) == render_samples(rebuilt)
    assert Counter(item.category for item in samples) == Counter(REPAIR_CATEGORY_COUNTS)
    assert Counter(item.language for item in samples) == Counter(REPAIR_LANGUAGE_COUNTS)
    assert quality == {
        "schema_version": "1.0",
        "status": "pass",
        "item_count": 2400,
        "tool_grounded_samples": 1440,
        "recovery_single_call_samples": 360,
        "clarification_question_samples": 240,
        "banned_generic_answer_matches": 0,
        "unique_final_answers": 2208,
        "maximum_exact_final_answer_frequency": 8,
    }


def test_repair_manifest_and_duplicate_gate_are_versioned() -> None:
    samples = build_repair_samples()
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)

    assert manifest.dataset_version == "m10-devops-training-v2-8461493c"
    assert manifest.source_revision == "m10-devops-training-v2"
    assert manifest.seed == 20260825
    assert manifest.training_permitted is False
    assert duplicate.status == "pass"
    assert duplicate.exact_duplicate_pairs == 0
    assert duplicate.cross_group_near_duplicate_pairs == 0


def test_repair_review_uses_a_distinct_protocol_identity() -> None:
    review = M10DevOpsContentReviewResult(
        review_version="m10-devops-content-review-v2",
        reviewed_at=datetime(2026, 8, 25, tzinfo=UTC),
        source_dataset_version="m10-devops-training-v2-8461493c",
        source_pending_manifest_sha256="a" * 64,
        source_items_sha256="b" * 64,
        source_content_sha256="c" * 64,
        source_review_packet_sha256="d" * 64,
        source_duplicate_report_sha256="e" * 64,
        source_contamination_report_sha256="f" * 64,
        approved_manifest_sha256="0" * 64,
        category_counts={category: 10 for category in REPAIR_CATEGORY_COUNTS},
        language_counts={"en": 40, "zh": 40},
    )

    assert review.review_version == "m10-devops-content-review-v2"


def test_repair_v3_source_covers_planning_and_entity_failures() -> None:
    samples = build_repair_v3_samples()
    rebuilt = build_repair_v3_samples()
    quality = validate_repair_v3_samples(samples)

    assert render_samples(samples) == render_samples(rebuilt)
    assert quality == {
        "schema_version": "1.0",
        "status": "pass",
        "item_count": 2400,
        "tool_grounded_samples": 1440,
        "recovery_single_call_samples": 360,
        "clarification_question_samples": 240,
        "sequential_two_step_samples": 480,
        "parallel_two_call_samples": 120,
        "entity_preserved_samples": 960,
        "banned_generic_answer_matches": 0,
        "unique_final_answers": 2208,
        "maximum_exact_final_answer_frequency": 8,
    }


def test_repair_v3_manifest_has_a_new_immutable_identity() -> None:
    samples = build_repair_v3_samples()
    manifest = build_manifest(samples)

    assert manifest.dataset_version.startswith("m10-devops-training-v3-")
    assert manifest.source_revision == "m10-devops-training-v3"
    assert manifest.seed == 20260827
    assert manifest.training_permitted is False


def test_repair_v4_source_expands_unique_runtime_aligned_coverage() -> None:
    samples = build_repair_v4_samples()
    quality = validate_repair_v4_samples(samples)
    manifest = build_manifest(samples)

    assert Counter(item.category for item in samples) == Counter(REPAIR_V4_CATEGORY_COUNTS)
    assert Counter(item.language for item in samples) == Counter(REPAIR_V4_LANGUAGE_COUNTS)
    assert quality["status"] == "pass"
    assert quality["item_count"] == 9600
    assert quality["sequential_two_step_samples"] == 1920
    assert quality["parallel_two_call_samples"] == 480
    assert manifest.dataset_version.startswith("m10-devops-training-v4-")
    assert manifest.seed == 20260829
    assert manifest.training_permitted is False
