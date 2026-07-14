from __future__ import annotations

import torch

from scripts.run_m1_gpu_interruption_smoke import (
    WorkerRun,
    _canonical_interrupted_metrics,
    _compare_to_baseline,
    _max_loss_abs_diff,
    _max_model_abs_diff,
)


def metric(step: int, loss: float) -> dict[str, object]:
    return {
        "event": "optimizer_step",
        "global_step": step,
        "learning_rate": 0.01,
        "loss": loss,
    }


def test_canonical_metrics_discard_steps_newer_than_sigkill_checkpoint() -> None:
    initial = WorkerRun(
        events=tuple(metric(step, float(step)) for step in range(1, 8)),
        returncode=-9,
        stderr="",
    )
    resumed = WorkerRun(
        events=(
            {"event": "resumed", "global_step": 5},
            *(metric(step, float(step)) for step in range(6, 11)),
        ),
        returncode=0,
        stderr="",
    )

    canonical, resume_step = _canonical_interrupted_metrics(initial, resumed)

    assert resume_step == 5
    assert [event["global_step"] for event in canonical] == list(range(1, 11))


def test_tolerance_comparison_checks_loss_parameters_state_and_sampler() -> None:
    baseline_metrics = [metric(1, 1.0), metric(2, 0.9)]
    candidate_metrics = [metric(1, 1.0 + 1.0e-7), metric(2, 0.9)]
    baseline_payload = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "trainer_state": {"global_step": 2},
        "sampler": {"cursor": 4},
    }
    candidate_payload = {
        "model": {"weight": torch.tensor([1.0, 2.0 + 1.0e-7])},
        "trainer_state": {"global_step": 2},
        "sampler": {"cursor": 4},
    }

    comparison = _compare_to_baseline(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_payload=baseline_payload,
        candidate_payload=candidate_payload,
        loss_abs_tolerance=1.0e-6,
        parameter_abs_tolerance=1.0e-6,
    )

    assert comparison["status"] == "pass"
    assert _max_loss_abs_diff(baseline_metrics, candidate_metrics) <= 1.0e-6
    assert _max_model_abs_diff(baseline_payload["model"], candidate_payload["model"]) <= 1.0e-6


def test_shape_or_step_mismatch_fails_comparison() -> None:
    assert _max_model_abs_diff({"weight": torch.ones(2)}, {"weight": torch.ones(3)}) == float("inf")
    assert _max_loss_abs_diff([metric(1, 1.0)], []) == float("inf")
