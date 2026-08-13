"""Deterministic private SQLite FTS5 index for line-addressable Agent evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from tinyllm.agent.schema import EvidenceIndexManifest, EvidenceSearchResult

SourceKind = Literal["documentation", "report", "registry", "run_metadata"]
RUN_METADATA_FIELDS = (
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
)


class EvidenceIndexError(RuntimeError):
    """Raised when evidence cannot be indexed or queried safely."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_root(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise EvidenceIndexError(f"{label} must be an absolute non-symlink directory")
    return path.resolve(strict=True)


def _safe_files(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve(strict=True).relative_to(root)
            except ValueError:
                continue
            yield path


def _repository_documents(project_root: Path) -> Iterable[tuple[SourceKind, str, str]]:
    seen: set[Path] = set()
    patterns = ("README.md", "README.en.md", "docs/**/*.md", "reports/**/*.md")
    for path in _safe_files(project_root, patterns):
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(project_root).as_posix()
        kind: SourceKind = "report" if relative.startswith("reports/") else "documentation"
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvidenceIndexError(f"cannot read repository evidence: {relative}") from exc
        yield kind, relative, text


def _registry_documents(artifact_root: Path) -> Iterable[tuple[SourceKind, str, str]]:
    registry = artifact_root / "registry"
    if not registry.is_dir() or registry.is_symlink():
        return
    for path in _safe_files(registry.resolve(strict=True), ("**/*.json",)):
        relative = path.relative_to(artifact_root).as_posix()
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceIndexError(f"cannot normalize Registry evidence: {relative}") from exc
        yield "registry", relative, text


def _run_documents(artifact_root: Path) -> Iterable[tuple[SourceKind, str, str]]:
    runs = artifact_root / "runs"
    if not runs.is_dir() or runs.is_symlink():
        return
    for path in _safe_files(runs.resolve(strict=True), ("*/run.json", "*/*/run.json")):
        relative = path.relative_to(artifact_root).as_posix()
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceIndexError(f"cannot read Run metadata: {relative}") from exc
        if not isinstance(value, dict):
            raise EvidenceIndexError(f"Run metadata is not an object: {relative}")
        sanitized = {key: value[key] for key in RUN_METADATA_FIELDS if key in value}
        text = json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True)
        yield "run_metadata", relative, text


def _chunks(
    text: str, *, lines_per_chunk: int = 40, overlap: int = 5
) -> Iterable[tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return
    start = 0
    while start < len(lines):
        end = min(len(lines), start + lines_per_chunk)
        excerpt = "\n".join(lines[start:end]).strip()
        if excerpt:
            yield start + 1, end, excerpt
        if end == len(lines):
            break
        start = end - overlap


def _create_index(path: Path, documents: Sequence[tuple[SourceKind, str, str]]) -> tuple[int, int]:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute(
            "CREATE TABLE evidence ("
            "row_id INTEGER PRIMARY KEY, document_id TEXT NOT NULL, source_kind TEXT NOT NULL, "
            "relative_path TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, "
            "content_sha256 TEXT NOT NULL, excerpt TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE evidence_fts USING fts5(excerpt, content='evidence', "
            "content_rowid='row_id', tokenize='unicode61')"
        )
        chunks = 0
        for kind, relative, text in documents:
            content_hash = _sha256_bytes(text.encode())
            identity = hashlib.sha256(f"{kind}|{relative}|{content_hash}".encode()).hexdigest()
            document_id = f"doc-{identity[:16]}"
            for start_line, end_line, excerpt in _chunks(text):
                cursor = connection.execute(
                    "INSERT INTO evidence(document_id,source_kind,relative_path,"
                    "start_line,end_line,"
                    "content_sha256,excerpt) VALUES(?,?,?,?,?,?,?)",
                    (
                        document_id,
                        kind,
                        relative,
                        start_line,
                        end_line,
                        content_hash,
                        excerpt,
                    ),
                )
                connection.execute(
                    "INSERT INTO evidence_fts(rowid,excerpt) VALUES(?,?)",
                    (cursor.lastrowid, excerpt),
                )
                chunks += 1
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.commit()
        return len(documents), chunks
    finally:
        connection.close()


def rebuild_evidence_index(
    *, project_root: Path, artifact_root: Path, output_dir: Path
) -> EvidenceIndexManifest:
    """Build a new private index atomically from public and sanitized evidence."""

    project_root = _safe_root(project_root, "project root")
    artifact_root = _safe_root(artifact_root, "Artifact Store")
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise EvidenceIndexError("evidence index output must be a new absolute path")
    documents = sorted(
        (
            *_repository_documents(project_root),
            *_registry_documents(artifact_root),
            *_run_documents(artifact_root),
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not documents:
        raise EvidenceIndexError("no evidence documents were discovered")
    source_digest = hashlib.sha256()
    for kind, relative, content in documents:
        source_digest.update(f"{kind}|{relative}|{_sha256_bytes(content.encode())}\n".encode())
    source_sha256 = source_digest.hexdigest()
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir(parents=True, mode=0o700)
        database = temporary / "evidence.sqlite3"
        document_count, chunk_count = _create_index(database, documents)
        database.chmod(0o600)
        index_sha256 = _sha256_file(database)
        manifest = EvidenceIndexManifest(
            index_version=f"m8-evidence-{source_sha256[:8]}",
            built_at=datetime.now(UTC),
            source_root_sha256=source_sha256,
            documents=document_count,
            chunks=chunk_count,
            index_sha256=index_sha256,
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
        return manifest
    except (OSError, sqlite3.Error, ValueError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(exc, EvidenceIndexError):
            raise
        raise EvidenceIndexError("cannot build evidence index") from exc


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[^\W_][\w.-]{1,63}", query, flags=re.UNICODE)[:20]
    if not tokens:
        raise EvidenceIndexError("evidence query contains no searchable terms")
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search_evidence(
    *, index_dir: Path, query: str, limit: int = 8, excerpt_characters: int = 1200
) -> tuple[EvidenceSearchResult, ...]:
    """Search one immutable FTS5 index without exposing an arbitrary SQL surface."""

    if not 1 <= limit <= 20 or not 100 <= excerpt_characters <= 4000:
        raise EvidenceIndexError("evidence search limits are invalid")
    if not index_dir.is_absolute() or index_dir.is_symlink() or not index_dir.is_dir():
        raise EvidenceIndexError("evidence index path is unsafe")
    database = index_dir / "evidence.sqlite3"
    manifest_path = index_dir / "manifest.json"
    if database.is_symlink() or manifest_path.is_symlink():
        raise EvidenceIndexError("evidence index contains a symlink")
    try:
        manifest = EvidenceIndexManifest.model_validate_json(manifest_path.read_bytes())
        if _sha256_file(database) != manifest.index_sha256:
            raise EvidenceIndexError("evidence index hash drift detected")
        uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            rows = connection.execute(
                "SELECT e.document_id,e.source_kind,e.relative_path,e.start_line,e.end_line,"
                "e.content_sha256,e.excerpt,bm25(evidence_fts) AS rank "
                "FROM evidence_fts JOIN evidence e ON e.row_id=evidence_fts.rowid "
                "WHERE evidence_fts MATCH ? ORDER BY rank,e.relative_path,e.start_line LIMIT ?",
                (_fts_query(query), limit),
            ).fetchall()
        finally:
            connection.close()
    except EvidenceIndexError:
        raise
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise EvidenceIndexError("cannot search evidence index") from exc
    return tuple(
        EvidenceSearchResult(
            document_id=row[0],
            source_kind=row[1],
            relative_path=row[2],
            start_line=row[3],
            end_line=row[4],
            content_sha256=row[5],
            relevance_score=max(0.0, -float(row[7])),
            excerpt=str(row[6])[:excerpt_characters],
        )
        for row in rows
    )
