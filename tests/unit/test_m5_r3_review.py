from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

import tinyllm.data.m5_r3_review as review_module
from tinyllm.data.m5_r3_review import (
    M5R3ContentReviewError,
    finalize_m5_r3_content_review,
)
from tinyllm.data.m5_r3_review_schema import (
    M5_R3_P2_PRIVATE_RAW_SHA256,
    M5_R3_P2_PUBLIC_RESULT_SHA256,
)

PUBLIC_RESULT = Path("reports/m5/raw/m5_r3_p2.json")
CRITERIA = (
    "label_matches_direct_evidence",
    "rationale_supports_selected_label",
    "no_unsupported_or_alternative_claims",
)


def _write_judgments(
    path: Path,
    samples: list[dict[str, object]],
    *,
    role: str = "maintainer",
    rejected_task_id: str | None = None,
    omit_last: bool = False,
) -> None:
    if omit_last:
        samples = samples[:-1]
    lines = []
    for sample in samples:
        task_id = cast(str, sample["task_id"])
        passed = task_id != rejected_task_id
        lines.append(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "task_id": task_id,
                    "reviewer_role": role,
                    "criteria": [
                        {"criterion": criterion, "passed": passed} for criterion in CRITERIA
                    ],
                    "passed": passed,
                    "rationale": "逐条核对了标签、证据和短推理。",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def review_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[dict[str, object]]]:
    samples: list[dict[str, object]] = []
    for family, language_counts in (
        ("config", {"en": 13, "zh": 4}),
        ("log_diagnosis", {"en": 11, "zh": 5}),
    ):
        short = "config" if family == "config" else "log"
        index = 0
        for language, count in language_counts.items():
            for _ in range(count):
                samples.append(
                    {
                        "task_id": (f"m5-reasoning:pilot:r3p1-{short}-{language}-{index:03d}"),
                        "task_family": family,
                        "language": language,
                    }
                )
                index += 1
    private_raw = tmp_path / "raw.json"
    private_raw.write_text(
        json.dumps(
            {
                "samples": samples,
                "samples_sha256": (
                    "2d73a4d62b657b98e9da2539d7cf10fdc8a2f3af369e4db4956348b5b79c3ea8"
                ),
            }
        ),
        encoding="utf-8",
    )

    def fake_sha256(path: Path) -> str:
        if path == PUBLIC_RESULT:
            return M5_R3_P2_PUBLIC_RESULT_SHA256
        if path == private_raw:
            return M5_R3_P2_PRIVATE_RAW_SHA256
        return "a" * 64

    monkeypatch.setattr(review_module, "_sha256", fake_sha256)
    return private_raw, samples


def test_m5_r3_content_review_approves_only_complete_maintainer_pass(
    tmp_path: Path,
    review_sources: tuple[Path, list[dict[str, object]]],
) -> None:
    private_raw, samples = review_sources
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(judgments, samples)

    result = finalize_m5_r3_content_review(
        public_result_path=PUBLIC_RESULT,
        private_raw_path=private_raw,
        judgments_path=judgments,
        reviewed_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    assert result.status == "approved"
    assert result.reviewed_items == result.passed_items == 33
    assert result.family_counts == {"config": 17, "log_diagnosis": 16}
    assert result.language_counts == {"en": 24, "zh": 9}
    assert result.formal_source_expansion_authorized is True
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False


def test_m5_r3_content_review_retains_rejection(
    tmp_path: Path,
    review_sources: tuple[Path, list[dict[str, object]]],
) -> None:
    private_raw, samples = review_sources
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(
        judgments,
        samples,
        rejected_task_id="m5-reasoning:pilot:r3p1-config-en-000",
    )

    result = finalize_m5_r3_content_review(
        public_result_path=PUBLIC_RESULT,
        private_raw_path=private_raw,
        judgments_path=judgments,
        reviewed_at=datetime(2026, 7, 30, tzinfo=UTC),
    )

    assert result.status == "rejected"
    assert result.passed_items == 32
    assert result.rejected_items == 1
    assert result.formal_source_expansion_authorized is False


@pytest.mark.parametrize(
    ("role", "omit_last"),
    [("codex_draft", False), ("maintainer", True)],
)
def test_m5_r3_content_review_rejects_draft_or_incomplete_coverage(
    tmp_path: Path,
    review_sources: tuple[Path, list[dict[str, object]]],
    role: str,
    omit_last: bool,
) -> None:
    private_raw, samples = review_sources
    judgments = tmp_path / "judgments.jsonl"
    _write_judgments(
        judgments,
        samples,
        role=role,
        omit_last=omit_last,
    )

    with pytest.raises(M5R3ContentReviewError):
        finalize_m5_r3_content_review(
            public_result_path=PUBLIC_RESULT,
            private_raw_path=private_raw,
            judgments_path=judgments,
        )
