"""Versioned output types for ``tinyllm doctor``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CheckStatus = Literal["pass", "warn", "fail", "unavailable"]
ReportStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    """Result of one doctor check."""

    check_id: str
    status: CheckStatus
    summary: str
    required: bool = False
    evidence: dict[str, object] = field(default_factory=dict)
    remediation: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "id": self.check_id,
            "status": self.status,
            "summary": self.summary,
            "required": self.required,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DoctorError:
    """Sanitized error emitted while collecting a report."""

    code: str
    message: str
    context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass(frozen=True)
class DoctorReport:
    """Complete versioned doctor report."""

    generated_at: str
    status: ReportStatus
    inventory: dict[str, object]
    checks: tuple[CheckResult, ...]
    errors: tuple[DoctorError, ...] = ()
    schema_version: str = "1.0"
    command: str = "tinyllm doctor"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "schema_version": self.schema_version,
            "command": self.command,
            "status": self.status,
            "generated_at": self.generated_at,
            "inventory": self.inventory,
            "checks": [check.to_dict() for check in self.checks],
            "errors": [error.to_dict() for error in self.errors],
        }


def aggregate_status(checks: list[CheckResult]) -> ReportStatus:
    """Aggregate check results according to the doctor contract."""

    if any(check.required and check.status == "fail" for check in checks):
        return "fail"
    if any(check.status in {"warn", "fail", "unavailable"} for check in checks):
        return "warn"
    return "pass"
