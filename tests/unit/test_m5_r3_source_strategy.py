from __future__ import annotations

from pathlib import Path

import pytest

from scripts.review_m5_r3_teacher_source_strategy import build_parser
from tinyllm.data.m5_r3_source_strategy import (
    M5R3SourceStrategyError,
    load_m5_r3_teacher_source_strategy_config,
    m5_r3_teacher_source_strategy_config_sha256,
    review_m5_r3_teacher_source_strategy,
)

CONFIG = Path("configs/data/m5_r3_teacher_source_strategy.yaml")
R2 = Path("reports/m5/raw/m5_r2_length_diagnostic.json")
P0 = Path("reports/m5/raw/m5_r3_p0.json")
P0_R1 = Path("reports/m5/raw/m5_r3_p0_r1.json")


def test_teacher_source_strategy_freezes_two_stage_p1_without_expansion() -> None:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)

    assert config.selected_strategy == "two_stage_solve_compress"
    assert config.controlled_baseline == "deterministic_rule_trace"
    assert config.pilot.pilot_version == "m5-r3-p1-two-stage-v1"
    assert config.pilot.solver.mode == "thinking"
    assert config.pilot.solver.max_new_tokens == 896
    assert config.pilot.compressor.mode == "nonthinking"
    assert config.pilot.compressor.max_new_tokens == 256
    assert config.pilot.trace_policy.max_reasoning_tokens == 192
    assert config.pilot.controlled_baseline_policy.training_source_authorized is False
    assert config.formal_source_expansion_authorized is False
    assert config.r3_mixture_authorized is False
    assert config.r3_training_authorized is False
    assert config.consume_m6_frozen_results is False
    assert m5_r3_teacher_source_strategy_config_sha256(config) == (
        "6a59d3a83d9420d7f44bd3432c98b1d296d946d1c99abedaa5048837244fa2d6"
    )


def test_teacher_source_strategy_review_is_bound_to_real_rejected_evidence() -> None:
    result = review_m5_r3_teacher_source_strategy(
        config_path=CONFIG,
        r2_decision_path=R2,
        p0_result_path=P0,
        p0_r1_result_path=P0_R1,
    )

    assert result.status == "two_stage_contract_authorized"
    assert result.quality_metric is False
    assert tuple(item.experiment for item in result.observations) == ("r2", "p0", "p0_r1")
    assert result.observations[0].projected_format_basis_points == (9800, 9650)
    assert result.observations[0].unresolved_format_items == (4, 7)
    assert result.observations[1].accepted_samples == 10
    assert result.observations[1].accepted_per_family == (5, 5)
    assert result.observations[2].accepted_samples == 12
    assert result.observations[2].accepted_per_family == (4, 8)
    assert tuple(item.disposition for item in result.alternatives) == (
        "rejected",
        "rejected",
        "selected_for_p1",
        "control_only",
    )
    assert result.p1_contract_implementation_authorized is True
    assert result.p1_gpu_pilot_authorized is False
    assert result.formal_source_expansion_authorized is False
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False


def test_teacher_source_strategy_rejects_parent_hash_drift(tmp_path: Path) -> None:
    drifted = tmp_path / "p0-r1.json"
    drifted.write_text(P0_R1.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(M5R3SourceStrategyError, match="SHA256 differs"):
        review_m5_r3_teacher_source_strategy(
            config_path=CONFIG,
            r2_decision_path=R2,
            p0_result_path=P0,
            p0_r1_result_path=drifted,
        )


def test_teacher_source_strategy_loader_and_cli_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(M5R3SourceStrategyError, match="must use YAML"):
        load_m5_r3_teacher_source_strategy_config(tmp_path / "config.json")
    with pytest.raises(M5R3SourceStrategyError, match="cannot be read"):
        load_m5_r3_teacher_source_strategy_config(tmp_path / "missing.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(M5R3SourceStrategyError, match="violates its schema"):
        load_m5_r3_teacher_source_strategy_config(invalid)

    args = build_parser().parse_args(["--output", "review.json"])
    assert args.config == CONFIG
    assert args.r2_decision == R2
    assert args.p0_result == P0
    assert args.p0_r1_result == P0_R1
    assert args.output == Path("review.json")
