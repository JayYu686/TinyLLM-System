from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_m5_r3_p1_cpu_smoke import main, run_smoke


@pytest.mark.integration
def test_m5_r3_p1_cpu_smoke_authorizes_only_real_gpu_pilot() -> None:
    result = run_smoke()
    committed = json.loads(
        Path("reports/m5/raw/m5_r3_p1_cpu_smoke.json").read_text(encoding="utf-8")
    )

    assert result.to_dict() == committed
    assert result.evidence_kind == "synthetic_cpu_contract_smoke"
    assert result.model_generated is False
    assert result.quality_metric is False
    assert result.accepted_samples == 40
    assert all(item.gate_passed for item in result.family_results)
    assert result.control.status == "pass"
    assert result.p1_gpu_pilot_authorized is True
    assert result.formal_source_expansion_authorized is False
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False


@pytest.mark.integration
def test_m5_r3_p1_cpu_smoke_cli_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "smoke.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["run_m5_r3_p1_cpu_smoke.py", "--output", str(output)],
    )

    assert main() == 2
