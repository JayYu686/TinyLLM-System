from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinyllm.training.ddp_schema import DDPCorrectnessSummary, DDPTrainingResult


def _torchrun(
    *,
    config: Path,
    output_root: Path,
    processes: int,
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("torchrun")
    assert executable.is_file(), "the constrained PyTorch environment must provide torchrun"
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            str(executable),
            "--standalone",
            f"--nproc-per-node={processes}",
            "-m",
            "tinyllm.training.ddp_worker",
            "--config",
            str(config),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=environment,
    )


@pytest.mark.integration
def test_two_process_gloo_ddp_writes_only_rank_zero_correctness_artifacts(
    tmp_path: Path,
) -> None:
    completed = _torchrun(
        config=Path("configs/pretrain/tinygpt_debug_ddp_cpu_smoke.yaml").resolve(),
        output_root=tmp_path,
        processes=2,
    )

    assert completed.returncode == 0, completed.stderr
    result = DDPTrainingResult.model_validate_json(completed.stdout)
    run_directories = tuple(path for path in tmp_path.iterdir() if path.is_dir())
    assert run_directories == (result.artifact_dir,)
    assert result.summary.status == "pass"
    assert result.summary.backend == "gloo"
    assert result.summary.world_size == 2
    assert result.summary.global_batch_size == 8
    assert result.summary.optimizer_steps == result.summary.durable_metric_records == 2
    assert result.summary.sampler_union_samples == result.summary.sampler_num_samples == 64
    assert result.summary.max_loss_reduction_abs_diff == 0
    assert result.summary.max_gradient_norm_abs_diff == 0

    metrics = (result.artifact_dir / "metrics.jsonl").read_text().splitlines()
    events = [
        json.loads(line) for line in (result.artifact_dir / "events.jsonl").read_text().splitlines()
    ]
    summary = DDPCorrectnessSummary.model_validate_json(
        (result.artifact_dir / "correctness.json").read_text()
    )
    assert len(metrics) == 2
    assert [event["event"] for event in events] == ["ddp_run_started", "ddp_run_succeeded"]
    assert summary == result.summary
    assert json.loads((result.artifact_dir / "run.json").read_text())["status"] == "succeeded"
    assert not tuple((result.artifact_dir / "checkpoints").iterdir())


@pytest.mark.integration
def test_torchrun_world_size_mismatch_fails_before_artifact_creation(tmp_path: Path) -> None:
    completed = _torchrun(
        config=Path("configs/pretrain/tinygpt_debug_ddp_cpu_smoke.yaml").resolve(),
        output_root=tmp_path,
        processes=1,
    )

    assert completed.returncode != 0
    assert "WORLD_SIZE does not match" in completed.stderr
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())
