from __future__ import annotations

import subprocess
import sys


def test_dual_mode_correction_builder_supports_direct_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_m5_dual_mode_correction.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "dual-mode correction mixture" in completed.stdout
