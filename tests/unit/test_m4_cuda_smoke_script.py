from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_m4_fsdp2_cuda_smoke import _torchrun_command, run_smoke


def test_m4_cuda_smoke_command_uses_the_fsdp2_worker(tmp_path: Path) -> None:
    command = _torchrun_command(Path("config.yaml"), tmp_path, 2)

    assert command[1:4] == ["--standalone", "--nproc-per-node=2", "-m"]
    assert "tinyllm.training.fsdp2_worker" in command


def test_m4_cuda_smoke_rejects_world_size_before_gpu_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GPU index count"):
        run_smoke(
            config_path=Path("configs/fsdp2/tinygpt_debug_nccl_bf16_single_gpu_smoke.yaml"),
            output_root=tmp_path / "runs",
            evidence_dir=tmp_path / "evidence",
            gpu_indices=(8, 9),
            timeout_seconds=30,
        )


def test_m4_cuda_smoke_rejects_cpu_config_before_gpu_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend=nccl"):
        run_smoke(
            config_path=Path("configs/fsdp2/tinygpt_debug_gloo_smoke.yaml"),
            output_root=tmp_path / "runs",
            evidence_dir=tmp_path / "evidence",
            gpu_indices=(9,),
            timeout_seconds=30,
        )
