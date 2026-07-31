from __future__ import annotations

import subprocess
import sys


def test_m5_lora_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_m5_lora.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--gpu-index" in completed.stdout
    assert "--resume-run" in completed.stdout
