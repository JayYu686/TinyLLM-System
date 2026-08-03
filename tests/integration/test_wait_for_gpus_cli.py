from __future__ import annotations

import subprocess
import sys


def test_wait_for_gpus_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/wait_for_gpus.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--candidate-gpus" in completed.stdout
    assert "--count" in completed.stdout
    assert "--max-memory-used-mib" in completed.stdout
    assert "--prerequisite-path" in completed.stdout
