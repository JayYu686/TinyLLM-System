from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_m5_reasoning_data_smoke import build_smoke_payload


@pytest.mark.integration
def test_m5_reasoning_cpu_smoke_matches_committed_public_evidence() -> None:
    payload = build_smoke_payload(Path("configs/data/m5_reasoning.yaml"))
    committed = json.loads(
        Path("reports/m5/raw/reasoning_data_smoke.json").read_text(encoding="utf-8")
    )

    assert payload == committed
    assert payload["model_generated"] is False
    assert payload["quality_metric"] is False
    contamination = payload["contamination_report"]
    assert isinstance(contamination, dict)
    assert contamination["status"] == "pass"
    assert contamination["matches"] == []
    dev = payload["dev_manifest"]
    assert isinstance(dev, dict)
    assert dev["task_count"] == 200
    pilot = payload["pilot_smoke"]
    assert isinstance(pilot, dict)
    manifest = pilot["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["accepted_samples"] == 50
