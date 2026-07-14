from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts.run_m2_packing_smoke import run_smoke


def test_committed_packing_smoke_rebuilds_exactly() -> None:
    committed = json.loads(
        Path("reports/m2/raw/packing_manifest_smoke.json").read_text(encoding="utf-8")
    )

    rebuilt = run_smoke(Path.cwd())

    assert rebuilt == committed
    assert rebuilt["status"] == "pass"
    assert rebuilt["scope"] == "public-synthetic-token-arrays"
    assert rebuilt["rebuild_after_input_reversal"] == "identical"
    manifest = cast(dict[str, Any], rebuilt["manifest"])
    assert manifest["dataset_version"] == "m2-sft-v1-b606b6d3"
    assert manifest["train_stratum_basis_points"] == {
        "commitpackft:en": 3846,
        "oasst1:en": 3076,
        "oasst1:zh": 3076,
    }
    assert manifest["split_pack_counts"] == {"test": 1, "train": 3, "validation": 1}
    assert manifest["balance_rejections"] == 2
    packs = cast(list[dict[str, Any]], rebuilt["packs"])
    assert (
        len({sample_id for pack in packs for sample_id in pack["sample_ids"]})
        == manifest["balanced_samples"]
    )
    assert all(pack["position_resets_verified"] for pack in packs)
    assert all(pack["segment_boundaries_verified"] for pack in packs)
