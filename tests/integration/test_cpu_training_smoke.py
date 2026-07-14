from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_m1_cpu_smoke import run_cpu_smoke


@pytest.mark.integration
def test_cpu_training_loss_decreases() -> None:
    payload = run_cpu_smoke(Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"))
    loss = payload["loss"]
    state = payload["state"]
    assert isinstance(loss, dict)
    assert isinstance(state, dict)

    assert payload["status"] == "pass"
    assert state["global_step"] == 30
    assert float(loss["last_over_first"]) < 0.5
