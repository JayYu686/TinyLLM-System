from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_m1_checkpoint_smoke import run_checkpoint_smoke


@pytest.mark.integration
def test_atomic_checkpoint_smoke() -> None:
    payload = run_checkpoint_smoke(Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"))
    retention = payload["retention"]
    corruption = payload["corruption_probe"]
    assert isinstance(retention, dict)
    assert isinstance(corruption, dict)

    assert payload["status"] == "pass"
    assert retention["retained"] == [
        "checkpoint-step-00000002",
        "checkpoint-step-00000003",
        "checkpoint-step-00000004",
    ]
    assert retention["latest"] == "checkpoint-step-00000004"
    assert corruption["detected_error_code"] == "CHECKPOINT_CORRUPT"
