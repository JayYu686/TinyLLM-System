from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.run_m5_r3_p2 import build_parser as build_gpu_parser
from scripts.run_m5_r3_p2_cpu_smoke import main, run_smoke
from tinyllm.data.m5_r3_p2_schema import M5R3P2CPUSmoke


def test_m5_r3_p2_cpu_smoke_authorizes_only_real_gpu_pilot() -> None:
    committed = M5R3P2CPUSmoke.model_validate_json(
        Path("reports/m5/raw/m5_r3_p2_cpu_smoke.json").read_text(encoding="utf-8")
    )
    result = run_smoke()

    assert result == committed
    assert result.model_generated is False
    assert result.quality_metric is False
    assert result.accepted_samples == 40
    assert result.fallback_solver_items == 6
    assert result.isolated_compressor_items == 40
    assert result.p2_gpu_pilot_authorized is True
    assert result.formal_source_expansion_authorized is False
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False


def test_m5_r3_p2_cpu_smoke_cli_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "p2.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_m5_r3_p2_cpu_smoke.py", "--output", str(output)],
    )

    assert main() == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {}


def test_m5_r3_p2_gpu_cli_requires_private_parent_and_outputs() -> None:
    args = build_gpu_parser().parse_args(
        [
            "--historical-pilot-artifact",
            "/private/historical.json",
            "--parent-p1-generation-artifact",
            "/private/p1.generations.json",
            "--model-dir",
            "/models/qwen3-8b",
            "--tokenizer-dir",
            "/models/qwen3-0.6b",
            "--gpu-index",
            "7",
            "--raw-output",
            "/private/p2.raw.json",
            "--public-output",
            "reports/m5/raw/m5_r3_p2.json",
        ]
    )

    assert args.config == Path("configs/data/m5_r3_p2.yaml")
    assert args.parent_p1_result == Path("reports/m5/raw/m5_r3_p1.json")
    assert args.gpu_index == 7
