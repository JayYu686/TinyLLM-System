from __future__ import annotations

import json
from pathlib import Path

from tinyllm.data.m10_mixture_schema import M10FrozenMixtureReport


def test_public_m10_frozen_mixture_report_records_real_training_gate() -> None:
    path = Path("reports/m10/raw/m10_frozen_mixture.json")
    report = M10FrozenMixtureReport.model_validate_json(path.read_bytes())

    assert report.status == "pass"
    assert report.dataset_version == "m10-agent-sft-v1-4655d3e3"
    assert report.target_supervised_tokens == 1_000_000
    assert report.source_supervised_tokens == {
        "toolace": 300_000,
        "hermes_function_calling": 200_000,
        "tinyllm_devops": 200_000,
        "m6_domain_replay": 200_000,
        "m2_no_tool_replay": 100_000,
    }
    assert report.language_supervised_tokens == {"en": 700_000, "zh": 300_000}
    assert report.mode_supervised_tokens == {"nonthinking": 940_000, "thinking": 60_000}
    assert report.overlength_rejections == {
        "toolace": 81,
        "hermes_function_calling": 1094,
        "tinyllm_devops": 0,
        "m6_domain_replay": 0,
        "m2_no_tool_replay": 3,
    }
    assert report.exact_duplicate_drops == 3
    assert report.near_duplicate_drops == 0
    assert report.training_permitted is True

    serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "tool_response" not in serialized
    assert "messages" not in serialized
