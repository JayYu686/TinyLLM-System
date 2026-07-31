from __future__ import annotations

import subprocess
import sys


def test_thinking_budget_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_m5_thinking_budget_eval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--training-run" in completed.stdout
    assert "--gpu-index" in completed.stdout
