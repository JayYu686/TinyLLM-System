from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def test_committed_contamination_smoke_reproduces_exactly() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "scripts.run_m2_contamination_smoke"],
        check=True,
        capture_output=True,
        text=True,
    )
    rebuilt = cast(dict[str, Any], json.loads(process.stdout))
    committed = json.loads(
        Path("reports/m2/raw/contamination_smoke.json").read_text(encoding="utf-8")
    )

    assert rebuilt == committed
    assert rebuilt["status"] == "pass"
    assert rebuilt["full_sequence_matches"] == 1
    assert rebuilt["prompt_prefix_matches"] == 2
    assert rebuilt["clean_control_status"] == "clean"
    assert rebuilt["raw_training_sample_ids_published"] is False
