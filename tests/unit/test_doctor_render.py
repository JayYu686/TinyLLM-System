from __future__ import annotations

import json

from tinyllm.doctor.render import render_json, render_text
from tinyllm.doctor.schema import CheckResult, DoctorError, DoctorReport


def sample_report() -> DoctorReport:
    return DoctorReport(
        generated_at="2026-07-14T00:00:00Z",
        status="warn",
        inventory={"gpu_count": 10},
        checks=(
            CheckResult("cuda", "pass", "CUDA is available", required=True),
            CheckResult(
                "nccl",
                "unavailable",
                "NCCL test tool is missing",
                remediation="Build nccl-tests before a distributed smoke test.",
            ),
        ),
        errors=(DoctorError(code="TOOL_MISSING", message="nccl-tests unavailable"),),
    )


def test_render_json_keeps_stable_public_shape() -> None:
    payload = json.loads(render_json(sample_report()))

    assert payload["schema_version"] == "1.0"
    assert payload["checks"][0]["id"] == "cuda"
    assert payload["errors"][0]["code"] == "TOOL_MISSING"


def test_render_text_includes_remediation_only_when_present() -> None:
    rendered = render_text(sample_report())

    assert "status: warn" in rendered
    assert "[PASS       ] cuda: CUDA is available" in rendered
    assert "remediation: Build nccl-tests" in rendered
