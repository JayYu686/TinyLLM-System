from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_m5_r3_formal_source_cpu_smoke import main


def test_m5_r3_formal_cpu_smoke_cli_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "formal.json"
    output.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_m5_r3_formal_source_cpu_smoke.py", "--output", str(output)],
    )

    assert main() == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {}


def test_m5_r3_formal_gpu_runner_supports_direct_script_help() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_m5_r3_formal_source.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "resumable GPU shards" in completed.stdout
