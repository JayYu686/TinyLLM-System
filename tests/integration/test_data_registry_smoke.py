from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def test_committed_registry_smoke_reproduces_exactly() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "scripts.run_m2_registry_smoke"],
        check=True,
        capture_output=True,
        text=True,
    )
    rebuilt = cast(dict[str, Any], json.loads(process.stdout))
    committed = json.loads(Path("reports/m2/raw/registry_smoke.json").read_text(encoding="utf-8"))

    assert rebuilt == committed
    assert rebuilt["status"] == "pass"
    assert rebuilt["first_registration_created"] is True
    assert rebuilt["second_registration_created"] is False
    assert rebuilt["reconstructed_packs_identical"] is True
    assert rebuilt["corruption_refusal_code"] == "DATASET_CORRUPT"
