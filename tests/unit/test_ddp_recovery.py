from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.training.ddp_recovery import (
    _truncate_metrics,
    _validate_future_injection_steps,
)
from tinyllm.training.metrics import TrainingStepMetrics


def _metric(step: int) -> TrainingStepMetrics:
    return TrainingStepMetrics(
        global_step=step,
        micro_step=step,
        epoch=0,
        loss=1.0 / step,
        learning_rate=0.01,
        gradient_norm=1.0,
        gradient_clipped=False,
        tokens_seen=step * 8,
    )


def test_metric_reconciliation_discards_only_rows_after_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps(_metric(step).to_dict()) + "\n" for step in range(1, 4)),
        encoding="utf-8",
    )

    retained, discarded = _truncate_metrics(path, checkpoint_step=2)

    assert (retained, discarded) == (2, 1)
    assert [json.loads(line)["global_step"] for line in path.read_text().splitlines()] == [
        1,
        2,
    ]


def test_metric_reconciliation_and_failure_injection_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(_metric(2).to_dict()) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical row"):
        _truncate_metrics(path, checkpoint_step=2)

    _validate_future_injection_steps(
        current_step=2,
        stop_after_step=3,
        fail_after_step=None,
    )
    with pytest.raises(ValueError, match="stop-after-step"):
        _validate_future_injection_steps(
            current_step=2,
            stop_after_step=2,
            fail_after_step=None,
        )
    with pytest.raises(ValueError, match="fail-after-step"):
        _validate_future_injection_steps(
            current_step=2,
            stop_after_step=None,
            fail_after_step=1,
        )
