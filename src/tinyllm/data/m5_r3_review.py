"""Finalize the private M5.2-R3 P2 maintainer content review."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from tinyllm.data.m5_r3_p2_schema import M5R3P2Result
from tinyllm.data.m5_r3_review_schema import (
    M5_R3_P2_PRIVATE_RAW_SHA256,
    M5_R3_P2_PUBLIC_RESULT_SHA256,
    M5R3ContentReviewJudgment,
    M5R3ContentReviewResult,
)


class M5R3ContentReviewError(ValueError):
    """Raised when content-review lineage, coverage, or authority differs."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_m5_r3_content_judgments(path: Path) -> tuple[M5R3ContentReviewJudgment, ...]:
    """Load strict private judgments without accepting drafts as maintainer decisions."""

    try:
        judgments = tuple(
            M5R3ContentReviewJudgment.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as exc:
        raise M5R3ContentReviewError("M5 R3 content judgments are invalid") from exc
    if not judgments or any(item.reviewer_role != "maintainer" for item in judgments):
        raise M5R3ContentReviewError("M5 R3 content review lacks maintainer judgments")
    return judgments


def finalize_m5_r3_content_review(
    *,
    public_result_path: Path,
    private_raw_path: Path,
    judgments_path: Path,
    reviewed_at: datetime | None = None,
) -> M5R3ContentReviewResult:
    """Bind all 33 private judgments to the exact passing P2 evidence."""

    if (
        _sha256(public_result_path) != M5_R3_P2_PUBLIC_RESULT_SHA256
        or _sha256(private_raw_path) != M5_R3_P2_PRIVATE_RAW_SHA256
    ):
        raise M5R3ContentReviewError("M5 R3 P2 review source SHA256 differs")
    try:
        public = M5R3P2Result.model_validate_json(public_result_path.read_text(encoding="utf-8"))
        raw = cast(dict[str, object], json.loads(private_raw_path.read_text(encoding="utf-8")))
        samples = cast(list[dict[str, object]], raw["samples"])
    except (OSError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        raise M5R3ContentReviewError("M5 R3 P2 review source is invalid") from exc
    if (
        public.status != "pass"
        or not public.formal_source_expansion_authorized
        or len(samples) != public.accepted_samples
        or raw.get("samples_sha256") != public.samples_sha256
    ):
        raise M5R3ContentReviewError("M5 R3 P2 source is not review-authorized")

    judgments = load_m5_r3_content_judgments(judgments_path)
    sample_by_id = {cast(str, item["task_id"]): item for item in samples}
    judgment_by_id = {item.task_id: item for item in judgments}
    if len(judgment_by_id) != len(judgments) or set(judgment_by_id) != set(sample_by_id):
        raise M5R3ContentReviewError("M5 R3 content-review task coverage differs")

    family_counts: Counter[str] = Counter(cast(str, sample["task_family"]) for sample in samples)
    language_counts: Counter[str] = Counter(cast(str, sample["language"]) for sample in samples)
    passed = sum(item.passed for item in judgments)
    timestamp = datetime.now(UTC) if reviewed_at is None else reviewed_at
    return M5R3ContentReviewResult(
        schema_version="1.0",
        review_version="m5-r3-p2-content-review-v1",
        reviewed_at=timestamp,
        status="approved" if passed == len(judgments) else "rejected",
        reviewer_role="maintainer",
        source_pilot_version=public.pilot_version,
        source_public_result_sha256=M5_R3_P2_PUBLIC_RESULT_SHA256,
        source_private_raw_sha256=M5_R3_P2_PRIVATE_RAW_SHA256,
        source_samples_sha256=("2d73a4d62b657b98e9da2539d7cf10fdc8a2f3af369e4db4956348b5b79c3ea8"),
        reviewed_items=33,
        passed_items=passed,
        rejected_items=len(judgments) - passed,
        family_counts={
            "config": family_counts["config"],
            "log_diagnosis": family_counts["log_diagnosis"],
        },
        language_counts={"en": language_counts["en"], "zh": language_counts["zh"]},
        private_judgments_sha256=_sha256(judgments_path),
        formal_source_expansion_authorized=passed == len(judgments),
        r3_mixture_authorized=False,
        r3_training_authorized=False,
        consumes_m6_frozen_results=False,
    )
