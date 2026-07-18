from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_m4_fsdp2_rank_failure_smoke import (
    _torchrun_command,
    run_rank_failure_smoke,
)


def test_m4_rank_failure_command_selects_nonzero_rank(tmp_path: Path) -> None:
    command = _torchrun_command(
        Path("config.yaml"),
        tmp_path,
        2,
        fail_rank=1,
        fail_after_step=1,
    )

    assert command[1:4] == ["--standalone", "--nproc-per-node=2", "-m"]
    assert command[-4:] == ["--fail-rank", "1", "--fail-after-step", "1"]


def test_m4_rank_failure_rejects_disabled_activation_before_gpu_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="Activation Checkpointing"):
        run_rank_failure_smoke(
            config_path=Path("configs/fsdp2/tinygpt_debug_nccl_bf16_two_gpu_smoke.yaml"),
            output_root=tmp_path / "runs",
            evidence_dir=tmp_path / "evidence",
            gpu_indices=(5, 7),
            fail_rank=1,
            fail_after_step=1,
            timeout_seconds=30,
        )


def test_m4_rank_failure_rejects_rank_zero_before_gpu_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="nonzero member"):
        run_rank_failure_smoke(
            config_path=Path(
                "configs/fsdp2/tinygpt_debug_nccl_bf16_two_gpu_activation_checkpointing_smoke.yaml"
            ),
            output_root=tmp_path / "runs",
            evidence_dir=tmp_path / "evidence",
            gpu_indices=(5, 7),
            fail_rank=0,
            fail_after_step=1,
            timeout_seconds=30,
        )
