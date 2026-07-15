from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import run_m3_ddp_recovery_smoke as recovery_smoke
from tinyllm.schemas import generate_run_id
from tinyllm.training import DDPRecoveryResult


def test_recovery_loss_difference_requires_equal_complete_series() -> None:
    assert recovery_smoke._max_abs_difference((1.0, 2.0), (1.25, 1.5)) == 0.5

    with pytest.raises(RuntimeError, match="equal non-empty"):
        recovery_smoke._max_abs_difference((1.0,), ())


def test_recovery_metric_reader_rejects_repeated_optimizer_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"global_step": step, "loss": 1.0}) for step in (1, 1)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing or repeated"):
        recovery_smoke._losses(run_dir)


def test_formal_recovery_smoke_rejects_dirty_git_before_gpu_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery_smoke,
        "read_git_identity",
        lambda _: ("a" * 40, True),
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        recovery_smoke.run_smoke(
            config_path=Path("configs/pretrain/tinygpt_debug_ddp_recovery_2gpu_bf16_smoke.yaml"),
            output_root=tmp_path / "runs",
            evidence_dir=tmp_path / "evidence",
            gpu_indices=(4, 5),
            timeout_seconds=30,
        )

    assert not (tmp_path / "evidence").exists()


def _result_mapping() -> dict[str, object]:
    config_hash = "a" * 64
    return {
        "status": "succeeded",
        "mode": "fresh",
        "run_id": generate_run_id(
            "recovery",
            config_hash,
            now=datetime(2026, 7, 15, tzinfo=UTC),
            nonce="cafe",
        ),
        "artifact_dir": Path("/tmp/recovery"),
        "config_sha256": config_hash,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "backend": "gloo",
        "world_size": 2,
        "global_step": 6,
        "checkpoint_id": "checkpoint-step-00000006",
        "model_parameter_sha256": "c" * 64,
        "resumed_from_step": None,
        "durable_metric_records": 6,
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"artifact_dir": Path("relative")}, "absolute"),
        ({"config_sha256": "d" * 64}, "config hash"),
        ({"checkpoint_id": "checkpoint-step-00000005"}, "global_step"),
        ({"durable_metric_records": 5}, "durable row"),
        ({"resumed_from_step": 2}, "fresh phase"),
        ({"mode": "exact_resume"}, "must advance"),
        ({"mode": "exact_resume", "resumed_from_step": 6}, "must advance"),
    ],
)
def test_recovery_result_rejects_mislabelled_evidence(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DDPRecoveryResult.model_validate({**_result_mapping(), **updates})
