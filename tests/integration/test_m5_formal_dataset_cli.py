from __future__ import annotations

import subprocess
import sys


def test_m5_formal_dataset_cli_supports_direct_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_m5_formal_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--authorization-gate" in completed.stdout
    assert "--source-root" in completed.stdout
