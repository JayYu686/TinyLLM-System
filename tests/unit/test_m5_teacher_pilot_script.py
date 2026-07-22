from __future__ import annotations

import pytest

from scripts.run_m5_teacher_pilot import build_parser, generation_seed


def test_teacher_pilot_generation_seeds_are_stable_and_distinct() -> None:
    assert generation_seed(100, 0, 0) == 100
    assert generation_seed(100, 0, 1) == 101
    assert generation_seed(100, 1, 0) == 102


def test_teacher_pilot_generation_seed_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="invalid"):
        generation_seed(100, -1, 0)
    with pytest.raises(ValueError, match="invalid"):
        generation_seed(100, 0, 2)


def test_teacher_pilot_cli_freezes_default_scale() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--model-dir",
            "/model",
            "--gpu-index",
            "9",
            "--raw-output",
            "/private/raw.json",
            "--public-output",
            "public.json",
        ]
    )

    assert args.tasks_per_family == 20
    assert args.timeout_seconds == 7200
