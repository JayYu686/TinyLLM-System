"""Versioned Pydantic output types for ``tinyllm doctor``."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tinyllm.schemas.base import StrictSchema

CheckStatus = Literal["pass", "warn", "fail", "unavailable"]
ReportStatus = Literal["pass", "warn", "fail"]


class CheckResult(StrictSchema):
    """Result of one doctor check."""

    check_id: str
    status: CheckStatus
    summary: str
    required: bool = False
    evidence: dict[str, object] = Field(default_factory=dict)
    remediation: str | None = None

    def __init__(
        self,
        check_id: str,
        status: CheckStatus,
        summary: str,
        *,
        required: bool = False,
        evidence: dict[str, object] | None = None,
        remediation: str | None = None,
    ) -> None:
        """Preserve the established positional collector interface."""

        data: dict[str, object] = {
            "check_id": check_id,
            "status": status,
            "summary": summary,
            "required": required,
            "evidence": evidence or {},
            "remediation": remediation,
        }
        super().__init__(**data)

    def to_dict(self) -> dict[str, object]:
        """Return the stable public shape used by doctor JSON."""

        return {
            "id": self.check_id,
            "status": self.status,
            "summary": self.summary,
            "required": self.required,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


class DoctorError(StrictSchema):
    """Sanitized error emitted while collecting a report."""

    code: str
    message: str
    context: dict[str, object] = Field(default_factory=dict)


class DoctorReport(StrictSchema):
    """Complete versioned doctor report."""

    generated_at: str
    status: ReportStatus
    inventory: dict[str, object]
    checks: tuple[CheckResult, ...]
    errors: tuple[DoctorError, ...] = ()
    schema_version: Literal["1.0"] = "1.0"
    command: Literal["tinyllm doctor"] = "tinyllm doctor"

    def to_dict(self) -> dict[str, object]:
        """Return the stable public shape used by existing consumers."""

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
