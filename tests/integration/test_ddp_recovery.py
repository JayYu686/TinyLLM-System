from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinyllm.training import DDPRecoveryResult

CONFIG = Path("configs/pretrain/tinygpt_debug_ddp_recovery_cpu_smoke.yaml").resolve()


def _torchrun(
    *,
    output_root: Path,
    extra: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("torchrun")
    assert executable.is_file()
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            str(executable),
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "tinyllm.training.ddp_recovery_worker",
            "--config",
            str(CONFIG),
            "--output-root",
            str(output_root),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=environment,
    )


def _steps(run_dir: Path) -> list[int]:
    return [
        int(json.loads(line)["global_step"])
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.integration
def test_coordinated_interruption_exact_resume_has_no_repeated_steps(tmp_path: Path) -> None:
    interrupted = _torchrun(output_root=tmp_path, extra=("--stop-after-step", "2"))

    assert interrupted.returncode != 0
    interrupted_result = DDPRecoveryResult.model_validate_json(interrupted.stdout)
    assert interrupted_result.status == "interrupted"
    assert interrupted_result.global_step == 2
    run_dir = interrupted_result.artifact_dir
    checkpoint = run_dir / "checkpoints" / interrupted_result.checkpoint_id
    assert {path.name for path in checkpoint.iterdir()} == {
        "COMMITTED",
        "config.resolved.json",
        "environment.json",
        "manifest.json",
        "rank-00000.pt",
        "rank-00001.pt",
        "training_state.pt",
    }

    resumed = _torchrun(
        output_root=tmp_path,
        extra=("--resume-run", str(run_dir)),
    )

    assert resumed.returncode == 0, resumed.stderr
    resumed_result = DDPRecoveryResult.model_validate_json(resumed.stdout)
    assert resumed_result.status == "succeeded"
    assert resumed_result.mode == "exact_resume"
    assert resumed_result.run_id == interrupted_result.run_id
    assert resumed_result.resumed_from_step == 2
    assert _steps(run_dir) == [1, 2, 3, 4, 5, 6]


@pytest.mark.integration
def test_nonzero_rank_exit_preserves_diagnostics_and_resumes(tmp_path: Path) -> None:
    failed = _torchrun(
        output_root=tmp_path,
        extra=("--fail-rank", "1", "--fail-after-step", "2"),
    )

    assert failed.returncode != 0
    assert "rank      : 1" in failed.stderr
    assert "exitcode  : 17" in failed.stderr
    run_dirs = tuple(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    failure_files = tuple((run_dir / "failures").glob("*.json"))
    assert len(failure_files) == 1
    failure = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert failure == {
        "checkpoint_id": "checkpoint-step-00000002",
        "event": "forced_rank_exit",
        "exit_code": 17,
        "global_step": 2,
        "rank": 1,
        "resumable": True,
        "schema_version": "1.0",
    }

    resumed = _torchrun(
        output_root=tmp_path,
        extra=("--resume-run", str(run_dir)),
    )

    assert resumed.returncode == 0, resumed.stderr
    result = DDPRecoveryResult.model_validate_json(resumed.stdout)
    assert result.status == "succeeded"
    assert result.resumed_from_step == 2
    assert _steps(run_dir) == [1, 2, 3, 4, 5, 6]
