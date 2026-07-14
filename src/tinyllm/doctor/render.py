"""Human and JSON renderers for doctor reports."""

from __future__ import annotations

import json

from tinyllm.doctor.schema import DoctorReport


def render_json(report: DoctorReport) -> str:
    """Render deterministic, machine-readable JSON."""

    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def render_text(report: DoctorReport) -> str:
    """Render a concise human-readable report."""

    lines = [
        "TinyLLM-System doctor",
        f"status: {report.status}",
        f"generated_at: {report.generated_at}",
        "checks:",
    ]
    for check in report.checks:
        lines.append(f"  [{check.status.upper():11}] {check.check_id}: {check.summary}")
        if check.remediation:
            lines.append(f"                remediation: {check.remediation}")
    return "\n".join(lines)
