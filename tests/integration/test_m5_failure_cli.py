from __future__ import annotations

import subprocess
import sys


def test_m5_failure_path_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_m5_failure_path_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "--output" in completed.stdout
