from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_m5_r3_p0_cpu_smoke import build_smoke_payload


@pytest.mark.integration
def test_m5_r3_p0_cpu_smoke_matches_committed_public_evidence() -> None:
    payload = build_smoke_payload(
        Path("configs/data/m5_r3_p0.yaml"),
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    committed = json.loads(
        Path("reports/m5/raw/m5_r3_p0_cpu_smoke.json").read_text(encoding="utf-8")
    )

    assert payload == committed
    assert payload["model_generated"] is False
    assert payload["quality_metric"] is False
    assert payload["gpu_used"] is False
    assert payload["input_tasks"] == 40
    assert payload["accepted_samples"] == 40
    contamination = payload["contamination"]
    assert isinstance(contamination, dict)
    assert contamination["status"] == "pass"


@pytest.mark.integration
def test_m5_r3_p0_r1_cpu_smoke_matches_committed_public_evidence() -> None:
    payload = build_smoke_payload(
        Path("configs/data/m5_r3_p0_r1.yaml"),
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    committed = json.loads(
        Path("reports/m5/raw/m5_r3_p0_r1_cpu_smoke.json").read_text(encoding="utf-8")
    )

    assert payload == committed
    assert payload["model_generated"] is False
    assert payload["quality_metric"] is False
    assert payload["gpu_used"] is False
    assert payload["input_tasks"] == 40
    assert payload["accepted_samples"] == 40
    contamination = payload["contamination"]
    assert isinstance(contamination, dict)
    assert contamination["status"] == "pass"
