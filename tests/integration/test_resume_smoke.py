from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_m1_resume_smoke import run_resume_smoke


@pytest.mark.integration
def test_cpu_resume_semantics_smoke() -> None:
    payload = run_resume_smoke(Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"))
    exact = payload["exact_resume"]
    failures = payload["failure_matrix"]
    assert isinstance(exact, dict)
    assert isinstance(failures, dict)

    assert payload["status"] == "pass"
    assert exact["first_resumed_global_step"] == 11
    assert exact["parameters_bitwise_equal"] is True
    assert exact["optimizer_state_bitwise_equal"] is True
    assert exact["loss_lr_and_metrics_bitwise_equal"] is True
    assert failures["bad_hash"]["code"] == "CHECKPOINT_CORRUPT"
