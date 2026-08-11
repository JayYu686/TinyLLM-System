from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from scripts.build_m2_domain_eval import _verify_or_write
from tinyllm.evaluation import (
    EvaluationSetManifest,
    HumanRubricScorer,
    build_evaluation_manifest,
    load_evaluation_build_config,
    load_evaluation_items,
)

CONFIG = Path("configs/eval/m2_domain_v1.yaml")
ITEMS = Path("evals/domain/v1/items.jsonl")
MANIFEST = Path("evals/domain/v1/manifest.json")
V2_CONFIG = Path("configs/eval/m6_domain_v2.yaml")
V2_ITEMS = Path("evals/domain/v2/items.jsonl")
V2_MANIFEST = Path("evals/domain/v2/manifest.json")
V3_CONFIG = Path("configs/eval/m6_domain_v3.yaml")
V3_ITEMS = Path("evals/domain/v3/items.jsonl")
V3_MANIFEST = Path("evals/domain/v3/manifest.json")
V4_CONFIG = Path("configs/eval/m6_domain_v4.yaml")
V4_ITEMS = Path("evals/domain/v4/items.jsonl")
V4_MANIFEST = Path("evals/domain/v4/manifest.json")


def test_frozen_domain_artifacts_are_reproducible_and_internally_consistent() -> None:
    config = load_evaluation_build_config(CONFIG)
    items = load_evaluation_items(ITEMS)
    committed = EvaluationSetManifest.model_validate_json(MANIFEST.read_text(encoding="utf-8"))

    assert build_evaluation_manifest(items, config=config) == committed
    assert len(items) == 300
    assert [item.id for item in items] == sorted(item.id for item in items)
    assert len({tuple(item.prompt_messages) for item in items}) == 300
    assert Counter(item.scorer.kind for item in items) == {
        "exact_match": 135,
        "human_rubric": 40,
        "json_object": 80,
        "multiple_choice": 45,
    }


def test_domain_content_has_complete_license_and_language_pairing() -> None:
    items = load_evaluation_items(ITEMS)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    english_only = 0
    for item in items:
        assert item.provenance.origin == "tinyllm-authored"
        assert item.provenance.license == "Apache-2.0"
        assert item.provenance.redistribution_allowed is True
        pair_tags = [tag for tag in item.tags if tag.startswith("bilingual-pair-")]
        if pair_tags:
            assert len(pair_tags) == 1
            pairs[(item.category, pair_tags[0])].append(item.language)
        else:
            assert "english-only" in item.tags
            assert item.language == "en"
            english_only += 1

    assert len(pairs) == 90
    assert all(sorted(languages) == ["en", "zh"] for languages in pairs.values())
    assert english_only == 120


def test_domain_scorers_match_category_policy() -> None:
    items = load_evaluation_items(ITEMS)
    expected = {
        "config": "json_object",
        "json": "json_object",
        "linux": "exact_match",
        "logs": "multiple_choice",
        "python": "exact_match",
        "refusal": "human_rubric",
        "short_code": "exact_match",
    }
    for item in items:
        assert item.scorer.kind == expected[item.category]
        if isinstance(item.scorer, HumanRubricScorer):
            assert len(item.scorer.criteria) == 3
            assert item.scorer.pass_threshold == 3
            assert item.scorer.retain_judgment_rationale is True


def test_committed_item_order_matches_canonical_id_order() -> None:
    rows = [
        cast(dict[str, Any], json.loads(line))
        for line in ITEMS.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)


def test_domain_generator_check_mode_accepts_committed_outputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_m2_domain_eval.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_m6_v2_holdout_is_reproducible_and_prompt_independent() -> None:
    config = load_evaluation_build_config(V2_CONFIG)
    items = load_evaluation_items(V2_ITEMS)
    committed = EvaluationSetManifest.model_validate_json(V2_MANIFEST.read_text(encoding="utf-8"))
    v1_prompts = {tuple(item.prompt_messages) for item in load_evaluation_items(ITEMS)}

    assert build_evaluation_manifest(items, config=config) == committed
    assert committed.suite_version == "tinyllm-domain-holdout-v1-c0c948cc"
    assert len(items) == 300
    assert len({tuple(item.prompt_messages) for item in items}) == 300
    assert not v1_prompts.intersection(tuple(item.prompt_messages) for item in items)
    assert Counter(item.scorer.kind for item in items) == {
        "exact_match": 135,
        "human_rubric": 40,
        "json_object": 80,
        "multiple_choice": 45,
    }
    assert all("v2 templates" in item.provenance.source_note for item in items)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_m2_domain_eval.py",
            "--suite-version",
            "v2",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_m6_v3_holdout_is_frozen_before_r2_and_independent() -> None:
    config = load_evaluation_build_config(V3_CONFIG)
    items = load_evaluation_items(V3_ITEMS)
    committed = EvaluationSetManifest.model_validate_json(V3_MANIFEST.read_text(encoding="utf-8"))
    earlier_prompts = {
        tuple(item.prompt_messages)
        for path in (ITEMS, V2_ITEMS)
        for item in load_evaluation_items(path)
    }

    assert build_evaluation_manifest(items, config=config) == committed
    assert committed.suite_version == "tinyllm-domain-holdout-v1-2b167ce6"
    assert len(items) == 300
    assert len({tuple(item.prompt_messages) for item in items}) == 300
    assert not earlier_prompts.intersection(tuple(item.prompt_messages) for item in items)
    assert all("v3 templates" in item.provenance.source_note for item in items)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_m2_domain_eval.py",
            "--suite-version",
            "v3",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_m6_v4_final_audit_is_reproducible_and_prompt_independent() -> None:
    config = load_evaluation_build_config(V4_CONFIG)
    items = load_evaluation_items(V4_ITEMS)
    committed = EvaluationSetManifest.model_validate_json(V4_MANIFEST.read_bytes())
    earlier_prompts = {
        tuple(item.prompt_messages)
        for path in (ITEMS, V2_ITEMS, V3_ITEMS)
        for item in load_evaluation_items(path)
    }

    assert build_evaluation_manifest(items, config=config) == committed
    assert committed.suite_version == "tinyllm-domain-final-audit-v1-bac25144"
    assert len(items) == 300
    assert len({tuple(item.prompt_messages) for item in items}) == 300
    assert not earlier_prompts.intersection(tuple(item.prompt_messages) for item in items)
    assert all("v4 templates" in item.provenance.source_note for item in items)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_m2_domain_eval.py",
            "--suite-version",
            "v4",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_domain_generator_check_mode_refuses_drift_without_writing(tmp_path: Path) -> None:
    output = tmp_path / "items.jsonl"

    assert _verify_or_write(output, "expected\n", check=True) is False
    assert not output.exists()
    output.write_text("stale\n", encoding="utf-8")
    assert _verify_or_write(output, "expected\n", check=True) is False
    assert output.read_text(encoding="utf-8") == "stale\n"
