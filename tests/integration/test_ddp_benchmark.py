from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinyllm.benchmark import DDPBenchmarkRunResult


@pytest.mark.integration
def test_two_process_cpu_benchmark_retains_raw_rank_windows(tmp_path: Path) -> None:
    torchrun = Path(sys.executable).with_name("torchrun")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            str(torchrun),
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "tinyllm.benchmark.ddp_worker",
            "--config",
            "configs/benchmark/m3_ddp_cpu_test.yaml",
            "--output-root",
            str(tmp_path / "runs"),
            "--profile",
            "strong",
            "--group",
            "standard",
            "--repeat",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    result = DDPBenchmarkRunResult.model_validate_json(completed.stdout)
    assert result.world_size == 2
    assert result.global_batch_size == 2
    assert result.warmup_steps == 1
    assert result.measurement_steps == 2
    assert len(result.rank_metrics) == 2
    assert all(len(item.step_time_ms) == 2 for item in result.rank_metrics)
    assert all(item.communication.status == "unavailable" for item in result.rank_metrics)
    assert all(item.profiler_trace_sha256 is not None for item in result.rank_metrics)
    assert (result.artifact_dir / "benchmark.json").is_file()
