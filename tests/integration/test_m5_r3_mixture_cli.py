from __future__ import annotations

import subprocess
import sys


def test_m5_r3_mixture_builder_supports_direct_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_m5_r3_mixture.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "label-aware mixture" in completed.stdout
