"""Capability-bounded implementations for the reference DevOps MCP Server."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from tinyllm.agent.evidence import search_evidence
from tinyllm.agent.schema import (
    AGENT_RUN_PATTERN,
    AgentApprovalDecision,
    AgentToolCall,
    agent_tool_call_sha256,
)

MAX_DOCUMENT_BYTES = 65_536
MAX_RESULT_CHARACTERS = 32_000
SAFE_RUN_FIELDS = frozenset(
    {
        "run_id",
        "status",
        "config_sha256",
        "git_commit",
        "dataset_version",
        "dataset_manifest_sha256",
        "model_revision",
        "attention_architecture",
        "global_step",
        "supervised_tokens",
        "latest_checkpoint",
    }
)
SECRET_KEY = re.compile(r"(?:token|secret|password|api[_-]?key|authorization)", re.IGNORECASE)
PATCH_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


class DevOpsToolError(RuntimeError):
    """Raised when a reference tool request leaves its configured capability."""


def _root(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise DevOpsToolError(f"{label} must be an absolute non-symlink directory")
    return path.resolve(strict=True)


def _safe_relative(root: Path, relative: str, *, prefixes: Sequence[str]) -> Path:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or not any(candidate.parts[0] == prefix for prefix in prefixes)
    ):
        raise DevOpsToolError("path is outside the allowed roots")
    path = root.joinpath(candidate)
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise DevOpsToolError("symlink paths are not allowed")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise DevOpsToolError("path is missing or outside the allowed roots") from exc
    if not path.is_file():
        raise DevOpsToolError("path must identify a regular file")
    return path


def _read_bounded(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size > MAX_DOCUMENT_BYTES:
            raise DevOpsToolError("document exceeds the inspection limit")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DevOpsToolError("document cannot be read") from exc


def _scrub(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if SECRET_KEY.search(str(key)) else _scrub(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def _json_size_and_depth(value: object) -> tuple[int, int]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    pending: list[tuple[object, int]] = [(value, 1)]
    maximum = 0
    nodes = 0
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        nodes += 1
        if nodes > 4096:
            raise DevOpsToolError("structured value is too complex")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
    return len(payload.encode()), maximum


def _atomic_new(path: Path, payload: bytes) -> None:
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


class DevOpsTools:
    """Reference DevOps capabilities backed only by allowlisted local roots."""

    def __init__(self, *, project_root: Path, artifact_root: Path, index_dir: Path) -> None:
        self.project_root = _root(project_root, "project root")
        self.artifact_root = _root(artifact_root, "Artifact Store")
        self.index_dir = _root(index_dir, "evidence index")

    def search_evidence(self, query: str, top_k: int = 8) -> dict[str, object]:
        results = search_evidence(index_dir=self.index_dir, query=query, limit=top_k)
        return {"schema_version": "1.0", "results": [item.to_dict() for item in results]}

    def list_runs(self, limit: int = 20, status: str | None = None) -> dict[str, object]:
        if not 1 <= limit <= 100:
            raise DevOpsToolError("run list limit is invalid")
        runs_root = self.artifact_root / "runs"
        records: list[dict[str, object]] = []
        if runs_root.is_dir() and not runs_root.is_symlink():
            for path in sorted(runs_root.glob("**/run.json"), reverse=True):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    path.resolve(strict=True).relative_to(runs_root.resolve(strict=True))
                    value: Any = json.loads(_read_bounded(path))
                except (ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(value, dict) or (status and value.get("status") != status):
                    continue
                records.append({key: value[key] for key in SAFE_RUN_FIELDS if key in value})
                if len(records) >= limit:
                    break
        return {"schema_version": "1.0", "runs": records}

    def get_run(self, run_id: str) -> dict[str, object]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,180}", run_id):
            raise DevOpsToolError("Run ID is invalid")
        runs_root = (self.artifact_root / "runs").resolve(strict=True)
        matches: list[Path] = []
        for path in runs_root.glob("**/run.json"):
            if path.parent.name != run_id or not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve(strict=True).relative_to(runs_root)
            except ValueError:
                continue
            matches.append(path)
        if len(matches) != 1:
            raise DevOpsToolError("Run ID is missing or ambiguous")
        value: Any = json.loads(_read_bounded(matches[0]))
        if not isinstance(value, dict):
            raise DevOpsToolError("Run metadata is invalid")
        return {
            "schema_version": "1.0",
            "run": {key: value[key] for key in SAFE_RUN_FIELDS if key in value},
        }

    def read_log_excerpt(
        self, relative_path: str, start_line: int = 1, end_line: int = 100
    ) -> dict[str, object]:
        if start_line < 1 or end_line < start_line or end_line - start_line >= 200:
            raise DevOpsToolError("log line range is invalid")
        path = _safe_relative(
            self.artifact_root,
            relative_path,
            prefixes=("runs", "evaluations", "deployments"),
        )
        if path.suffix.lower() not in {".log", ".jsonl", ".txt"}:
            raise DevOpsToolError("log inspection requires a text log artifact")
        lines = _read_bounded(path).splitlines()
        excerpt = "\n".join(lines[start_line - 1 : end_line])[:MAX_RESULT_CHARACTERS]
        return {
            "schema_version": "1.0",
            "relative_path": relative_path,
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "excerpt": excerpt,
        }

    def query_metrics(
        self, relative_path: str, metric_names: list[str] | None = None, limit: int = 50
    ) -> dict[str, object]:
        if not 1 <= limit <= 200:
            raise DevOpsToolError("metrics record limit is invalid")
        path = _safe_relative(
            self.artifact_root,
            relative_path,
            prefixes=("runs", "evaluations", "deployments"),
        )
        if path.name not in {"metrics.jsonl", "summary.json"}:
            raise DevOpsToolError("metrics path must end in metrics.jsonl or summary.json")
        text = _read_bounded(path)
        try:
            values: list[object] = (
                [json.loads(line) for line in text.splitlines() if line]
                if path.suffix == ".jsonl"
                else [json.loads(text)]
            )
        except json.JSONDecodeError as exc:
            raise DevOpsToolError("metrics artifact is invalid JSON") from exc
        names = set(metric_names or ())
        if any(not PATCH_KEY.fullmatch(name) for name in names):
            raise DevOpsToolError("metric name is invalid")
        selected: list[object] = []
        for value in values[-limit:]:
            if isinstance(value, dict) and names:
                selected.append({key: value[key] for key in names if key in value})
            else:
                selected.append(value)
        if _json_size_and_depth(selected)[0] > MAX_RESULT_CHARACTERS:
            raise DevOpsToolError("metrics result exceeds the output limit")
        return {"schema_version": "1.0", "records": _scrub(selected)}

    def inspect_config(self, relative_path: str) -> dict[str, object]:
        roots: tuple[tuple[Path, tuple[str, ...]], ...] = (
            (self.project_root, ("configs",)),
            (self.artifact_root, ("runs", "deployments")),
        )
        path: Path | None = None
        for root, prefixes in roots:
            try:
                path = _safe_relative(root, relative_path, prefixes=prefixes)
                break
            except DevOpsToolError:
                continue
        if path is None or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            raise DevOpsToolError("configuration path is outside the allowed files")
        text = _read_bounded(path)
        try:
            value: Any = (
                json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
            )
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise DevOpsToolError("configuration document is invalid") from exc
        if _json_size_and_depth(value)[1] > 32:
            raise DevOpsToolError("configuration nesting exceeds the inspection limit")
        return {
            "schema_version": "1.0",
            "relative_path": relative_path,
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config": _scrub(value),
        }

    def apply_sandbox_config_patch(
        self,
        run_id: str,
        approval_id: str,
        call_id: str,
        source_relative_path: str,
        updates: dict[str, Any],
    ) -> dict[str, object]:
        if re.fullmatch(AGENT_RUN_PATTERN, run_id) is None:
            raise DevOpsToolError("Agent Run ID is invalid")
        source = _safe_relative(self.project_root, source_relative_path, prefixes=("configs",))
        if source.suffix.lower() not in {".yaml", ".yml"}:
            raise DevOpsToolError("sandbox patches support YAML configurations only")
        if not updates or any(not PATCH_KEY.fullmatch(str(key)) for key in updates):
            raise DevOpsToolError("sandbox patch keys are invalid")
        if any(SECRET_KEY.search(str(key)) for key in updates):
            raise DevOpsToolError("sandbox patch cannot set secrets")
        size, depth = _json_size_and_depth(updates)
        if size > 16_384 or depth > 16:
            raise DevOpsToolError("sandbox patch exceeds the structural limit")
        approval_path = (
            self.artifact_root / "agent-runs" / run_id / "approvals" / f"{approval_id}.json"
        )
        try:
            if approval_path.is_symlink():
                raise ValueError
            approval_path.resolve(strict=True).relative_to(self.artifact_root)
            decision = AgentApprovalDecision.model_validate_json(approval_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise DevOpsToolError("approval for the sandbox write is missing or invalid") from exc
        if decision.approval_id != approval_id or decision.decision != "approved":
            raise DevOpsToolError("sandbox write was not approved")
        try:
            proposed = AgentToolCall(
                call_id=call_id,
                server_id="tinyllm-devops",
                tool_name="apply_sandbox_config_patch",
                arguments={
                    "source_relative_path": source_relative_path,
                    "updates": updates,
                },
            )
        except ValueError as exc:
            raise DevOpsToolError("sandbox write identity is invalid") from exc
        if agent_tool_call_sha256(proposed) != decision.tool_call_sha256:
            raise DevOpsToolError("sandbox write differs from the approved tool call")
        sandboxes = self.artifact_root / "agent-sandboxes"
        if sandboxes.exists() and (not sandboxes.is_dir() or sandboxes.is_symlink()):
            raise DevOpsToolError("Agent sandbox collection is unsafe")
        sandboxes.mkdir(mode=0o700, exist_ok=True)
        sandbox = sandboxes / run_id
        if sandbox.exists() and (not sandbox.is_dir() or sandbox.is_symlink()):
            raise DevOpsToolError("Agent sandbox root is unsafe")
        sandbox.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = sandbox / source_relative_path
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            target.parent.resolve(strict=True).relative_to(sandbox.resolve(strict=True))
        except ValueError as exc:
            raise DevOpsToolError("sandbox path escaped its Agent Run") from exc
        current = sandbox
        for part in Path(source_relative_path).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise DevOpsToolError("sandbox path contains a symlink")
        try:
            value: Any = yaml.safe_load(_read_bounded(source))
            if not isinstance(value, dict):
                raise ValueError
            value.update(updates)
            payload = yaml.safe_dump(value, allow_unicode=True, sort_keys=True).encode()
            if target.exists() and not target.is_symlink():
                if target.read_bytes() == payload:
                    return {
                        "schema_version": "1.0",
                        "relative_path": target.relative_to(self.artifact_root).as_posix(),
                        "content_sha256": hashlib.sha256(payload).hexdigest(),
                        "approval_id": approval_id,
                    }
                raise DevOpsToolError("sandbox write target conflicts with the approved patch")
            if target.is_symlink():
                raise DevOpsToolError("sandbox patch target is a symlink")
            _atomic_new(target, payload)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise DevOpsToolError("sandbox configuration patch failed") from exc
        return {
            "schema_version": "1.0",
            "relative_path": target.relative_to(self.artifact_root).as_posix(),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "approval_id": approval_id,
        }
