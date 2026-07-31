from __future__ import annotations

import subprocess
import sys

import pytest

from scripts.run_m5_formal_ddp import _gpu_indices


def test_m5_formal_training_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_m5_formal_ddp.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--gpu-indices" in completed.stdout
    assert "--resume-run" in completed.stdout


def test_m5_formal_training_cli_rejects_non_four_gpu_selection() -> None:
    with pytest.raises(Exception, match="four distinct"):
        _gpu_indices("4,5,6")
