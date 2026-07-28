from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_m5_r2_failures import build_parser as build_analysis_parser
from scripts.run_m5_r2_length_replay import (
    _export_sha256,
)
from scripts.run_m5_r2_length_replay import (
    build_parser as build_replay_parser,
)
from scripts.select_m5_r2_diagnostic import build_parser as build_selection_parser
from tinyllm.evaluation.m5_r2_diagnostic import M5R2DiagnosticError


def test_r2_replay_cli_requires_explicit_lineage_paths_and_gpu() -> None:
    args = build_replay_parser().parse_args(
        [
            "--artifact-root",
            "/artifacts",
            "--source-evaluation",
            "/evaluations/seed42",
            "--training-run",
            "/runs/seed42",
            "--model-dir",
            "/runs/seed42/exports/model",
            "--tokenizer-dir",
            "/cache/model",
            "--output-dir",
            "/evaluations/r2/seed42",
            "--gpu-index",
            "5",
        ]
    )

    assert args.gpu_index == 5
    assert args.timeout_seconds == 14_400
    assert args.config == Path("configs/eval/m5_r2_length_replay.yaml")
    assert args.worker is False


def test_r2_replay_export_hash_rejects_symlink(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    target = tmp_path / "target"
    target.write_text("private model fixture", encoding="utf-8")
    (model_dir / "model.safetensors").symlink_to(target)

    with pytest.raises(M5R2DiagnosticError, match="non-regular"):
        _export_sha256(model_dir)


def test_r2_analysis_and_selection_clis_have_explicit_private_inputs() -> None:
    analysis = build_analysis_parser().parse_args(
        [
            "--seed42-evaluation",
            "/private/seed42",
            "--seed20260727-evaluation",
            "/private/seed20260727",
            "--tokenizer-dir",
            "/cache/model",
            "--output",
            "reports/m5/raw/offline.json",
        ]
    )
    selection = build_selection_parser().parse_args(
        [
            "--seed42-summary",
            "/private/r2/seed42/summary.json",
            "--seed20260727-summary",
            "/private/r2/seed20260727/summary.json",
            "--output",
            "reports/m5/raw/decision.json",
        ]
    )

    assert analysis.output == Path("reports/m5/raw/offline.json")
    assert selection.output == Path("reports/m5/raw/decision.json")
