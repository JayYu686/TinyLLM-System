"""Atomic, rebuildable SQLite query index over immutable Run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tinyllm.lineage.schema import RunIndexEntry, RunIndexListResult, RunIndexRebuildResult
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, RUN_ID_PATTERN, SHA256_PATTERN

INDEX_SCHEMA_VERSION = 1
DEFAULT_INDEX_RELATIVE_PATH = Path("registry/runs.sqlite3")

_DDL = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
);
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_relative_path TEXT UNIQUE NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    config_sha256 TEXT,
    git_commit TEXT,
    git_dirty INTEGER,
    strategy TEXT,
    world_size INTEGER,
    dataset_version TEXT,
    latest_checkpoint TEXT,
    global_step INTEGER,
    supervised_tokens INTEGER
);
CREATE INDEX runs_created_at_idx ON runs(created_at DESC, run_id DESC);
CREATE INDEX runs_status_idx ON runs(status, created_at DESC);
"""


class RunIndexErrorCode(StrEnum):
    """Stable failure classes for Run index operations."""

    INVALID_INPUT = "RUN_INDEX_INVALID_INPUT"
    SOURCE_NOT_FOUND = "RUN_INDEX_SOURCE_NOT_FOUND"
    SOURCE_CORRUPT = "RUN_INDEX_SOURCE_CORRUPT"
    INDEX_NOT_FOUND = "RUN_INDEX_NOT_FOUND"
    INDEX_CORRUPT = "RUN_INDEX_CORRUPT"
    WRITE_FAILED = "RUN_INDEX_WRITE_FAILED"


class RunIndexError(RuntimeError):
    """Run index failure with a stable public error code."""

    def __init__(self, code: RunIndexErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_string(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string when present")
    return value


def _optional_integer(source: dict[str, Any], key: str, *, minimum: int) -> int | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum} when present")
    return int(value)


def _optional_boolean(source: dict[str, Any], key: str) -> bool | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean when present")
    return value


def _coalesce_identity(
    source: dict[str, Any], first: str, second: str, *, pattern: str | None = None
) -> str | None:
    first_value = _optional_string(source, first)
    second_value = _optional_string(source, second)
    if first_value is not None and second_value is not None and first_value != second_value:
        raise ValueError(f"{first} and {second} disagree")
    value = first_value or second_value
    if value is not None and pattern is not None and re.fullmatch(pattern, value) is None:
        raise ValueError(f"{first}/{second} has an invalid digest")
    return value


def _created_at(source: dict[str, Any], run_id: str) -> datetime:
    raw = source.get("created_at")
    if raw is not None:
        if not isinstance(raw, str):
            raise ValueError("created_at must be an ISO-8601 string when present")
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return parsed.astimezone(UTC)
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError("invalid run_id")
    return datetime.strptime(match.group("timestamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _project_manifest(path: Path, *, artifact_root: Path, runs_root: Path) -> RunIndexEntry:
    if path.is_symlink() or not path.resolve().is_relative_to(runs_root.resolve()):
        raise ValueError("Run manifest must be a regular file below runs/")
    payload = path.read_bytes()
    source = json.loads(payload)
    if not isinstance(source, dict):
        raise ValueError("Run manifest root must be a JSON object")
    if source.get("schema_version") != "1.0":
        raise ValueError("Run manifest schema_version must be 1.0")
    run_id = source.get("run_id")
    status = source.get("status")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Run manifest has an invalid run_id")
    if path.parent.name != run_id:
        raise ValueError("Run manifest directory does not match run_id")
    if not isinstance(status, str) or not status:
        raise ValueError("Run manifest has an invalid status")
    git_commit = _optional_string(source, "git_commit")
    if git_commit is not None and re.fullmatch(GIT_COMMIT_PATTERN, git_commit) is None:
        raise ValueError("git_commit has an invalid digest")
    dataset_version = _coalesce_identity(source, "dataset_version", "mixture_version")
    config_sha256 = _coalesce_identity(
        source, "config_hash", "config_sha256", pattern=SHA256_PATTERN
    )
    relative_path = path.relative_to(artifact_root).as_posix()
    return RunIndexEntry(
        run_id=run_id,
        created_at=_created_at(source, run_id),
        status=status,
        manifest_relative_path=relative_path,
        manifest_sha256=_sha256_bytes(payload),
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=_optional_boolean(source, "git_dirty"),
        strategy=_optional_string(source, "strategy"),
        world_size=_optional_integer(source, "world_size", minimum=1),
        dataset_version=dataset_version,
        latest_checkpoint=_coalesce_identity(source, "latest_checkpoint", "checkpoint_id"),
        global_step=_optional_integer(source, "global_step", minimum=0),
        supervised_tokens=_optional_integer(source, "supervised_tokens", minimum=0),
    )


def _collect_entries(artifact_root: Path) -> tuple[RunIndexEntry, ...]:
    runs_root = artifact_root / "runs"
    if not runs_root.is_dir():
        raise RunIndexError(
            RunIndexErrorCode.SOURCE_NOT_FOUND,
            "Artifact Store runs directory does not exist",
        )
    entries: list[RunIndexEntry] = []
    seen: set[str] = set()
    for path in sorted(runs_root.rglob("run.json")):
        try:
            entry = _project_manifest(path, artifact_root=artifact_root, runs_root=runs_root)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, ValidationError) as exc:
            relative = path.relative_to(artifact_root).as_posix()
            raise RunIndexError(
                RunIndexErrorCode.SOURCE_CORRUPT,
                f"invalid Run manifest: {relative}: {exc}",
            ) from exc
        if entry.run_id in seen:
            raise RunIndexError(
                RunIndexErrorCode.SOURCE_CORRUPT,
                f"duplicate Run ID: {entry.run_id}",
            )
        seen.add(entry.run_id)
        entries.append(entry)
    return tuple(entries)


def _source_tree_sha256(entries: tuple[RunIndexEntry, ...]) -> str:
    payload = "".join(
        f"{entry.manifest_relative_path}\0{entry.manifest_sha256}\n" for entry in entries
    ).encode()
    return _sha256_bytes(payload)


def _entry_values(entry: RunIndexEntry) -> tuple[object, ...]:
    return (
        entry.run_id,
        entry.created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        entry.status,
        entry.manifest_relative_path,
        entry.manifest_sha256,
        entry.config_sha256,
        entry.git_commit,
        None if entry.git_dirty is None else int(entry.git_dirty),
        entry.strategy,
        entry.world_size,
        entry.dataset_version,
        entry.latest_checkpoint,
        entry.global_step,
        entry.supervised_tokens,
    )


def rebuild_run_index(
    artifact_root: Path,
    *,
    output_path: Path | None = None,
) -> RunIndexRebuildResult:
    """Rebuild a complete SQLite projection and publish it with atomic Rename."""

    if not artifact_root.is_absolute():
        raise RunIndexError(
            RunIndexErrorCode.INVALID_INPUT,
            "Artifact Store root must be absolute",
        )
    output = output_path or artifact_root / DEFAULT_INDEX_RELATIVE_PATH
    if not output.is_absolute():
        raise RunIndexError(RunIndexErrorCode.INVALID_INPUT, "index output must be absolute")
    if output.suffix != ".sqlite3":
        raise RunIndexError(RunIndexErrorCode.INVALID_INPUT, "index output must use .sqlite3")
    if output.resolve().is_relative_to((artifact_root / "runs").resolve()):
        raise RunIndexError(
            RunIndexErrorCode.INVALID_INPUT,
            "index output must remain outside the immutable runs tree",
        )
    if output.exists() and output.is_dir():
        raise RunIndexError(RunIndexErrorCode.INVALID_INPUT, "index output is a directory")
    entries = _collect_entries(artifact_root)
    source_tree_sha256 = _source_tree_sha256(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(_DDL)
            connection.executemany(
                """
                INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (_entry_values(entry) for entry in entries),
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(INDEX_SCHEMA_VERSION)),
                    ("source_tree_sha256", source_tree_sha256),
                    ("source_manifests", str(len(entries))),
                ),
            )
            connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise sqlite3.DatabaseError("SQLite integrity_check failed")
        finally:
            connection.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, sqlite3.Error) as exc:
        temporary.unlink(missing_ok=True)
        raise RunIndexError(
            RunIndexErrorCode.WRITE_FAILED,
            "cannot atomically rebuild Run index",
        ) from exc
    relative = (
        output.relative_to(artifact_root).as_posix()
        if output.is_relative_to(artifact_root)
        else output.name
    )
    return RunIndexRebuildResult(
        index_relative_path=relative,
        index_sha256=_sha256_file(output),
        source_tree_sha256=source_tree_sha256,
        source_manifests=len(entries),
        indexed_runs=len(entries),
    )


def _open_verified_index(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise RunIndexError(RunIndexErrorCode.INDEX_NOT_FOUND, "Run index does not exist")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()
        if integrity != ("ok",) or version != (INDEX_SCHEMA_VERSION,):
            raise sqlite3.DatabaseError("Run index failed integrity or version validation")
        return connection
    except sqlite3.Error as exc:
        if connection is not None:
            connection.close()
        raise RunIndexError(RunIndexErrorCode.INDEX_CORRUPT, "Run index is corrupt") from exc


def _row_to_entry(row: sqlite3.Row) -> RunIndexEntry:
    return RunIndexEntry(
        run_id=row["run_id"],
        created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
        status=row["status"],
        manifest_relative_path=row["manifest_relative_path"],
        manifest_sha256=row["manifest_sha256"],
        config_sha256=row["config_sha256"],
        git_commit=row["git_commit"],
        git_dirty=None if row["git_dirty"] is None else bool(row["git_dirty"]),
        strategy=row["strategy"],
        world_size=row["world_size"],
        dataset_version=row["dataset_version"],
        latest_checkpoint=row["latest_checkpoint"],
        global_step=row["global_step"],
        supervised_tokens=row["supervised_tokens"],
    )


def list_indexed_runs(
    index_path: Path,
    *,
    status: str | None = None,
    limit: int = 50,
) -> RunIndexListResult:
    """List newest indexed Runs with an optional exact status filter."""

    if limit < 1 or limit > 1000:
        raise RunIndexError(RunIndexErrorCode.INVALID_INPUT, "limit must be between 1 and 1000")
    if status is not None and (not status or len(status) > 64):
        raise RunIndexError(
            RunIndexErrorCode.INVALID_INPUT,
            "status must contain between 1 and 64 characters",
        )
    connection = _open_verified_index(index_path)
    try:
        connection.row_factory = sqlite3.Row
        if status is None:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM runs WHERE status = ?
                ORDER BY created_at DESC, run_id DESC LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        entries = tuple(_row_to_entry(row) for row in rows)
    except (sqlite3.Error, ValidationError, ValueError) as exc:
        raise RunIndexError(RunIndexErrorCode.INDEX_CORRUPT, "Run index is corrupt") from exc
    finally:
        connection.close()
    return RunIndexListResult(
        status_filter=status,
        limit=limit,
        returned_runs=len(entries),
        runs=entries,
    )


def show_indexed_run(index_path: Path, run_id: str) -> RunIndexEntry:
    """Return one indexed Run by exact ID."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise RunIndexError(RunIndexErrorCode.INVALID_INPUT, "invalid run_id")
    connection = _open_verified_index(index_path)
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    except sqlite3.Error as exc:
        raise RunIndexError(RunIndexErrorCode.INDEX_CORRUPT, "Run index is corrupt") from exc
    finally:
        connection.close()
    if row is None:
        raise RunIndexError(RunIndexErrorCode.INDEX_NOT_FOUND, "Run ID is not indexed")
    try:
        return _row_to_entry(row)
    except (ValidationError, ValueError) as exc:
        raise RunIndexError(RunIndexErrorCode.INDEX_CORRUPT, "Run index is corrupt") from exc
