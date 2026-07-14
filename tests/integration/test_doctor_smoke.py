import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_doctor_json_smoke_from_outside_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    src = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = str(src)
    completed = subprocess.run(
        [sys.executable, "-m", "tinyllm", "doctor", "--json"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode in {0, 3}
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["command"] == "tinyllm doctor"
    assert payload["status"] in {"pass", "warn", "fail"}
