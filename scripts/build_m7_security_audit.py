#!/usr/bin/env python3
"""Build a deterministic M7 security audit from OSV snapshots and reviewed policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from tinyllm.deployment import M7SecurityAudit, M7VulnerabilityAssessment
from tinyllm.schemas import canonical_config_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.glob("*.json"), key=lambda path: path.name)
    if not paths:
        raise ValueError("OSV snapshot directory is empty")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("OSV snapshot contains an unsafe entry")
        digest.update(path.name.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _load_osv(root: Path) -> tuple[dict[str, dict[str, str]], int]:
    reviewed: dict[str, dict[str, str]] = {}
    observed: set[str] = set()
    for path in sorted(root.glob("*.json")):
        package = path.stem
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        for advisory in payload.get("vulns", []):
            identifier = advisory.get("id", "")
            if not isinstance(identifier, str) or not identifier.startswith("GHSA-"):
                continue
            observed.add(identifier)
            details = advisory.get("database_specific", {})
            severity = str(details.get("severity", "unknown")).lower()
            if severity in {"critical", "high"}:
                reviewed[identifier] = {"package": package, "severity": severity}
    return reviewed, len(observed)


def _versions(environment: Path) -> dict[str, str]:
    payload: Any = json.loads(environment.read_text(encoding="utf-8"))
    return {item["name"]: item["version"] for item in payload["packages"]}


def _write_new(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("output must be a new absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> M7SecurityAudit:
    policy: Any = yaml.safe_load(args.policy.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "1.0":
        raise ValueError("security policy schema version is invalid")
    reviewed, observed = _load_osv(args.osv_dir)
    decisions = {item["advisory_id"]: item for item in policy.get("assessments", [])}
    if set(decisions) != set(reviewed):
        missing = sorted(set(reviewed) - set(decisions))
        stale = sorted(set(decisions) - set(reviewed))
        raise ValueError(f"security policy drift: missing={missing}, stale={stale}")
    versions = _versions(args.environment)
    assessments = tuple(
        M7VulnerabilityAssessment(
            advisory_id=identifier,
            package=reviewed[identifier]["package"],
            installed_version=versions[reviewed[identifier]["package"]],
            severity=cast(Any, reviewed[identifier]["severity"]),
            disposition=decisions[identifier]["disposition"],
            rationale=decisions[identifier]["rationale"],
            controls=decisions[identifier].get("controls", ()),
        )
        for identifier in sorted(reviewed)
    )
    critical = sum(
        item.severity == "critical" and item.disposition == "unmitigated" for item in assessments
    )
    high = sum(
        item.severity == "high" and item.disposition == "unmitigated" for item in assessments
    )
    identity = canonical_config_hash(
        {
            "policy": policy,
            "osv": _tree_sha256(args.osv_dir),
            "environment": _sha256(args.environment),
            "control": _sha256(args.control_evidence),
        }
    )
    audit = M7SecurityAudit(
        audit_id=f"m7-security-{identity[:8]}",
        evaluated_at=datetime.now(UTC),
        profile=policy["profile"],
        environment_sha256=_sha256(args.environment),
        pip_audit_sha256=_sha256(args.pip_audit),
        osv_snapshot_sha256=_tree_sha256(args.osv_dir),
        control_evidence_sha256=_sha256(args.control_evidence),
        observed_advisories=observed,
        reviewed_critical_high_advisories=len(assessments),
        assessments=assessments,
        unmitigated_critical_vulnerabilities=critical,
        unmitigated_high_vulnerabilities=high,
        status="accepted" if critical == high == 0 else "rejected",
    )
    _write_new(
        args.output,
        (json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--osv-dir", type=Path, required=True)
    parser.add_argument("--pip-audit", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--control-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = build(parser.parse_args())
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
