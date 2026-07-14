"""Deterministic, dependency-light importers for the two pinned M2 sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import field_validator

from tinyllm.data.schema import (
    DataImportManifest,
    ImportedMessage,
    ImportedSample,
    ImportedSampleMetadata,
    RejectedRecord,
    RejectionReason,
)
from tinyllm.data.sources import (
    COMMITPACKFT_LICENSE_ALLOWLIST,
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    normalize_license,
)
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import canonical_config_hash


class OASST1ImportConfig(StrictSchema):
    """Filtering contract for the pinned OASST1 snapshot."""

    schema_version: Literal["1.0"] = "1.0"
    allowed_languages: tuple[str, ...] = ("en", "zh")
    required_tree_state: Literal["ready_for_export"] = "ready_for_export"
    require_positive_review: Literal[True] = True
    include_deleted: Literal[False] = False

    @field_validator("allowed_languages")
    @classmethod
    def validate_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a stable, non-empty language policy."""

        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("allowed languages must be non-empty, unique, and sorted")
        if not set(value).issubset({"en", "zh"}):
            raise ValueError("allowed languages must stay within the M2 en/zh contract")
        return value


class CommitPackFTImportConfig(StrictSchema):
    """Language and source-license policy for CommitPackFT."""

    schema_version: Literal["1.0"] = "1.0"
    required_language: Literal["python"] = "python"
    prompt_template_version: Literal["commit-edit-v1"] = "commit-edit-v1"
    allowed_licenses: tuple[str, ...] = tuple(sorted(COMMITPACKFT_LICENSE_ALLOWLIST))

    @field_validator("allowed_licenses")
    @classmethod
    def validate_licenses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Prevent an ambiguous or non-canonical license policy."""

        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("allowed licenses must be non-empty, unique, and sorted")
        if any(normalize_license(item) != item for item in value):
            raise ValueError("allowed licenses must use normalized labels")
        if not set(value).issubset(COMMITPACKFT_LICENSE_ALLOWLIST):
            raise ValueError("allowed licenses must stay within the reviewed M2 allowlist")
        return value


@dataclass(frozen=True, slots=True)
class ImportResult:
    """In-memory M2 import result plus its deterministic evidence manifest."""

    manifest: DataImportManifest
    samples: tuple[ImportedSample, ...]
    rejected: tuple[RejectedRecord, ...]


@dataclass(frozen=True, slots=True)
class _OASSTNode:
    message_id: str
    parent_id: str | None
    text: str
    role: str
    language: str
    review_result: bool
    deleted: bool
    tree_state: str
    tree_id: str
    raw_hash: str


class _RowError(ValueError):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def _json_default(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _row_hash(row: Mapping[str, object]) -> str:
    try:
        payload = _canonical_json(dict(row))
    except (TypeError, ValueError):
        field_types = sorted((str(key), type(value).__qualname__) for key, value in row.items())
        payload = _canonical_json({"unserializable_field_types": field_types})
    return hashlib.sha256(payload).hexdigest()


def _input_hash(rows: tuple[Mapping[str, object], ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(_row_hash(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def _required_string(row: Mapping[str, object], field: str, *, allow_empty: bool = False) -> str:
    value = row.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _RowError(field)
    return value


def _optional_string(
    row: Mapping[str, object], field: str, *, allow_empty: bool = False
) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _RowError(field)
    return value


def _required_bool(row: Mapping[str, object], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise _RowError(field)
    return value


def _record_id(row: Mapping[str, object], index: int, preferred_field: str) -> str:
    value = row.get(preferred_field)
    return value if isinstance(value, str) and value.strip() else f"row-{index:08d}"


def _rejection(
    *,
    source: Literal["oasst1", "commitpackft"],
    revision: str,
    source_record_id: str,
    raw_hash: str,
    reason: RejectionReason,
    field: str | None = None,
) -> RejectedRecord:
    return RejectedRecord(
        source=source,
        source_revision=revision,
        source_record_id=source_record_id,
        raw_record_sha256=raw_hash,
        reason=reason,
        field=field,
    )


def _sample_id(source: Literal["oasst1", "commitpackft"], identity: str) -> str:
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"{source}:{suffix}"


def _build_manifest(
    *,
    source: Literal["oasst1", "commitpackft"],
    rows: tuple[Mapping[str, object], ...],
    config: StrictSchema,
    samples: list[ImportedSample],
    rejected: list[RejectedRecord],
) -> DataImportManifest:
    source_descriptor = OASST1_SOURCE if source == "oasst1" else COMMITPACKFT_SOURCE
    rejection_counts = Counter(record.reason for record in rejected)
    license_counts = Counter(sample.metadata.license for sample in samples)
    return DataImportManifest(
        source=source_descriptor,
        input_sha256=_input_hash(rows),
        config_sha256=canonical_config_hash(
            {"source": source_descriptor.to_dict(), "import": config.to_dict()}
        ),
        source_rows=len(rows),
        candidate_samples=len(samples) + len(rejected),
        accepted_samples=len(samples),
        rejected_samples=len(rejected),
        rejection_counts=dict(sorted(rejection_counts.items())),
        license_counts=dict(sorted(license_counts.items())),
    )


def _parse_oasst_node(row: Mapping[str, object]) -> _OASSTNode:
    tree_id = _required_string(row, "message_tree_id")
    return _OASSTNode(
        message_id=_required_string(row, "message_id"),
        parent_id=_optional_string(row, "parent_id"),
        text=_required_string(row, "text", allow_empty=True),
        role=_required_string(row, "role"),
        language=_required_string(row, "lang").lower(),
        review_result=_required_bool(row, "review_result"),
        deleted=_required_bool(row, "deleted"),
        tree_state=_required_string(row, "tree_state"),
        tree_id=tree_id,
        raw_hash=_row_hash(row),
    )


def _oasst_path(
    node: _OASSTNode, nodes: Mapping[str, _OASSTNode]
) -> tuple[list[_OASSTNode] | None, RejectionReason | None, str | None]:
    path: list[_OASSTNode] = []
    visited: set[str] = set()
    current = node
    while True:
        if current.message_id in visited:
            return None, "invalid_conversation", "parent_id"
        visited.add(current.message_id)
        path.append(current)
        if current.parent_id is None:
            break
        parent = nodes.get(current.parent_id)
        if parent is None:
            return None, "missing_parent", "parent_id"
        current = parent
    path.reverse()
    roles = [item.role for item in path]
    expected = ["prompter" if index % 2 == 0 else "assistant" for index in range(len(path))]
    if roles != expected:
        return None, "invalid_conversation", "role"
    if any(item.tree_id != path[0].tree_id for item in path):
        return None, "invalid_conversation", "message_tree_id"
    return path, None, None


def import_oasst1(
    rows: Iterable[Mapping[str, object]],
    *,
    config: OASST1ImportConfig | None = None,
) -> ImportResult:
    """Import licensed assistant-ending paths from the pinned OASST1 revision."""

    policy = config or OASST1ImportConfig()
    materialized = tuple(rows)
    rejected: list[RejectedRecord] = []
    parsed: list[_OASSTNode] = []
    for index, row in enumerate(materialized):
        try:
            parsed.append(_parse_oasst_node(row))
        except _RowError as exc:
            rejected.append(
                _rejection(
                    source="oasst1",
                    revision=OASST1_SOURCE.revision,
                    source_record_id=_record_id(row, index, "message_id"),
                    raw_hash=_row_hash(row),
                    reason="malformed_row",
                    field=exc.field,
                )
            )

    id_counts = Counter(node.message_id for node in parsed)
    nodes: dict[str, _OASSTNode] = {}
    for node in parsed:
        if id_counts[node.message_id] > 1:
            rejected.append(
                _rejection(
                    source="oasst1",
                    revision=OASST1_SOURCE.revision,
                    source_record_id=node.message_id,
                    raw_hash=node.raw_hash,
                    reason="duplicate_source_id",
                    field="message_id",
                )
            )
        elif node.role not in {"prompter", "assistant"}:
            rejected.append(
                _rejection(
                    source="oasst1",
                    revision=OASST1_SOURCE.revision,
                    source_record_id=node.message_id,
                    raw_hash=node.raw_hash,
                    reason="unsupported_role",
                    field="role",
                )
            )
        else:
            nodes[node.message_id] = node

    samples: list[ImportedSample] = []
    allowed_languages = set(policy.allowed_languages)
    for node in sorted(nodes.values(), key=lambda item: item.message_id):
        if node.role != "assistant":
            continue
        path, path_reason, path_field = _oasst_path(node, nodes)
        reason: RejectionReason | None = path_reason
        field = path_field
        if path is not None:
            if any(item.tree_state != policy.required_tree_state for item in path):
                reason, field = "not_ready", "tree_state"
            elif not policy.include_deleted and any(item.deleted for item in path):
                reason, field = "deleted", "deleted"
            elif policy.require_positive_review and any(not item.review_result for item in path):
                reason, field = "review_not_positive", "review_result"
            elif any(item.language not in allowed_languages for item in path):
                reason, field = "unsupported_language", "lang"
            elif any(not item.text.strip() for item in path):
                reason, field = "empty_content", "text"
        if reason is not None or path is None:
            rejected.append(
                _rejection(
                    source="oasst1",
                    revision=OASST1_SOURCE.revision,
                    source_record_id=node.message_id,
                    raw_hash=node.raw_hash,
                    reason=reason or "invalid_conversation",
                    field=field,
                )
            )
            continue

        messages = tuple(
            ImportedMessage(
                role="user" if item.role == "prompter" else "assistant", content=item.text
            )
            for item in path
        )
        samples.append(
            ImportedSample(
                id=_sample_id("oasst1", node.message_id),
                source="oasst1",
                messages=messages,
                metadata=ImportedSampleMetadata(
                    language=node.language,
                    category="conversation",
                    license="apache-2.0",
                    source_revision=OASST1_SOURCE.revision,
                    source_record_id=node.message_id,
                    group_ids=(path[0].tree_id,),
                    raw_record_sha256s=tuple(item.raw_hash for item in path),
                ),
            )
        )

    samples.sort(key=lambda sample: sample.id)
    rejected.sort(key=lambda item: (item.source_record_id, item.reason, item.raw_record_sha256))
    manifest = _build_manifest(
        source="oasst1",
        rows=materialized,
        config=policy,
        samples=samples,
        rejected=rejected,
    )
    return ImportResult(manifest=manifest, samples=tuple(samples), rejected=tuple(rejected))


def _parse_repositories(value: str) -> tuple[str, ...]:
    repositories = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not repositories:
        raise _RowError("repos")
    return repositories


def import_commitpackft(
    rows: Iterable[Mapping[str, object]],
    *,
    config: CommitPackFTImportConfig | None = None,
) -> ImportResult:
    """Import Python code-edit samples that pass the explicit source-license allowlist."""

    policy = config or CommitPackFTImportConfig()
    materialized = tuple(rows)
    samples: list[ImportedSample] = []
    rejected: list[RejectedRecord] = []
    for index, row in enumerate(materialized):
        record_id = _record_id(row, index, "commit")
        raw_hash = _row_hash(row)
        try:
            commit = _required_string(row, "commit")
            language = _required_string(row, "lang").strip().lower()
            license_name = normalize_license(_required_string(row, "license"))
            repositories_value = _required_string(row, "repos", allow_empty=True)
            instruction = _required_string(row, "subject", allow_empty=True)
            old_contents = _optional_string(row, "old_contents", allow_empty=True) or ""
            new_contents = _required_string(row, "new_contents", allow_empty=True)
            new_file = _optional_string(row, "new_file")
            old_file = _optional_string(row, "old_file")
            filename = new_file or old_file
            if filename is None:
                raise _RowError("new_file|old_file")
        except _RowError as exc:
            rejected.append(
                _rejection(
                    source="commitpackft",
                    revision=COMMITPACKFT_SOURCE.revision,
                    source_record_id=record_id,
                    raw_hash=raw_hash,
                    reason="malformed_row",
                    field=exc.field,
                )
            )
            continue

        try:
            repositories = _parse_repositories(repositories_value)
        except _RowError:
            rejected.append(
                _rejection(
                    source="commitpackft",
                    revision=COMMITPACKFT_SOURCE.revision,
                    source_record_id=record_id,
                    raw_hash=raw_hash,
                    reason="missing_repository",
                    field="repos",
                )
            )
            continue

        reason: RejectionReason | None = None
        field: str | None = None
        if language != policy.required_language:
            reason, field = "not_python", "lang"
        elif license_name not in policy.allowed_licenses:
            reason, field = "unsupported_license", "license"
        elif not instruction.strip():
            reason, field = "empty_instruction", "subject"
        elif not new_contents.strip():
            reason, field = "empty_content", "new_contents"
        if reason is not None:
            rejected.append(
                _rejection(
                    source="commitpackft",
                    revision=COMMITPACKFT_SOURCE.revision,
                    source_record_id=record_id,
                    raw_hash=raw_hash,
                    reason=reason,
                    field=field,
                )
            )
            continue

        prompt = (
            f"Update {filename} according to this instruction:\n{instruction.strip()}\n\n"
            f"Current file:\n{old_contents}"
        )
        identity = f"{commit}\n{','.join(repositories)}\n{filename}\n{raw_hash}"
        samples.append(
            ImportedSample(
                id=_sample_id("commitpackft", identity),
                source="commitpackft",
                messages=(
                    ImportedMessage(role="user", content=prompt),
                    ImportedMessage(role="assistant", content=new_contents),
                ),
                metadata=ImportedSampleMetadata(
                    language="en",
                    category="code_edit",
                    license=license_name,
                    source_revision=COMMITPACKFT_SOURCE.revision,
                    source_record_id=commit,
                    group_ids=repositories,
                    raw_record_sha256s=(raw_hash,),
                ),
            )
        )

    samples.sort(key=lambda sample: sample.id)
    rejected.sort(key=lambda item: (item.source_record_id, item.reason, item.raw_record_sha256))
    manifest = _build_manifest(
        source="commitpackft",
        rows=materialized,
        config=policy,
        samples=samples,
        rejected=rejected,
    )
    return ImportResult(manifest=manifest, samples=tuple(samples), rejected=tuple(rejected))
