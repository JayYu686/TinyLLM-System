from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tinyllm.training.fsdp2_schema import FSDP2CorrectnessSummary, FSDP2TrainingResult


def _torchrun(
    *,
    config: Path,
    output_root: Path,
    processes: int,
    extra: tuple[str, ...] = (),
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
            "tinyllm.training.fsdp2_worker",
            "--config",
            str(config),
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


@pytest.mark.integration
def test_two_process_gloo_fsdp2_uses_explicit_cpu_mesh_and_shards_state(
    tmp_path: Path,
) -> None:
    completed = _torchrun(
        config=Path("configs/fsdp2/tinygpt_debug_gloo_smoke.yaml").resolve(),
        output_root=tmp_path,
        processes=2,
    )

    assert completed.returncode == 0, completed.stderr
    result = FSDP2TrainingResult.model_validate_json(completed.stdout)
    assert result.summary.status == "pass"
    assert result.summary.backend == "gloo"
    assert result.summary.device_type == "cpu"
    assert result.summary.world_size == 2
    assert result.summary.global_batch_size == 4
    assert result.summary.optimizer_steps == result.summary.durable_metric_records == 2
    assert result.summary.local_shard_parameter_sum == result.summary.logical_parameter_count
    assert all(item.parameters_are_dtensor for item in result.summary.rank_evidence)
    assert result.summary.max_loss_reduction_abs_diff == 0
    assert result.summary.max_gradient_norm_abs_diff == 0
    assert result.summary.initial_full_parameter_sha256 != (
        result.summary.final_full_parameter_sha256
    )

    hardware = json.loads((result.artifact_dir / "hardware.json").read_text())
    assert hardware["device_type"] == "cpu"
    assert all(rank["device"] == "cpu" for rank in hardware["ranks"])
    assert all(rank["physical_gpu_index"] is None for rank in hardware["ranks"])
    assert len((result.artifact_dir / "metrics.jsonl").read_text().splitlines()) == 2
    events = [
        json.loads(line) for line in (result.artifact_dir / "events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "fsdp2_run_started",
        "fsdp2_run_succeeded",
    ]
    summary = FSDP2CorrectnessSummary.model_validate_json(
        (result.artifact_dir / "correctness.json").read_text()
    )
    assert summary == result.summary
    assert not tuple((result.artifact_dir / "checkpoints").iterdir())


@pytest.mark.integration
def test_fsdp2_world_size_mismatch_fails_before_artifact_creation(tmp_path: Path) -> None:
    completed = _torchrun(
        config=Path("configs/fsdp2/tinygpt_debug_gloo_smoke.yaml").resolve(),
        output_root=tmp_path,
        processes=1,
    )

    assert completed.returncode != 0
    assert "WORLD_SIZE does not match" in completed.stderr
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


@pytest.mark.integration
def test_two_process_gloo_fsdp2_activation_checkpointing(tmp_path: Path) -> None:
    completed = _torchrun(
        config=Path(
            "configs/fsdp2/tinygpt_debug_gloo_activation_checkpointing_smoke.yaml"
        ).resolve(),
        output_root=tmp_path,
        processes=2,
    )

    assert completed.returncode == 0, completed.stderr
    result = FSDP2TrainingResult.model_validate_json(completed.stdout)
    assert result.summary.activation_checkpointing is True
    assert result.summary.activation_checkpointed_block_type == "TransformerBlock"
    assert result.summary.activation_checkpointed_block_count == 2
    assert result.summary.local_shard_parameter_sum == result.summary.logical_parameter_count


@pytest.mark.integration
def test_two_process_gloo_fsdp2_nonzero_rank_exit_retains_diagnostics(
    tmp_path: Path,
) -> None:
    completed = _torchrun(
        config=Path(
            "configs/fsdp2/tinygpt_debug_gloo_activation_checkpointing_smoke.yaml"
        ).resolve(),
        output_root=tmp_path,
        processes=2,
        extra=("--fail-rank", "1", "--fail-after-step", "1"),
    )

    assert completed.returncode != 0
    assert "rank      : 1" in completed.stderr
    assert "exitcode  : 17" in completed.stderr
    run_dirs = tuple(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    failure_files = tuple((run_dir / "failures").glob("*.json"))
    assert len(failure_files) == 1
    failure = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert failure["event"] == "forced_rank_exit"
    assert failure["rank"] == 1
    assert failure["exit_code"] == 17
    assert failure["global_step"] == 1
    assert failure["resumable"] is False
    assert failure["checkpoint_status"] == "not_evaluated_m4_1"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failure_injected"
    assert run["resumable"] is False
    assert not tuple((run_dir / "checkpoints").iterdir())
