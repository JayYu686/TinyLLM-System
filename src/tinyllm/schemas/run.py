"""Run identity and lifecycle schemas."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema

RUN_ID_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}T\d{6}Z)-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)-"
    r"(?P<config_hash>[a-f0-9]{8})-(?P<nonce>[a-f0-9]{4})$"
)
SHA256_PATTERN = r"^[a-f0-9]{64}$"
GIT_COMMIT_PATTERN = r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$"


class RunStatus(StrEnum):
    """Persisted states for one training or evaluation run."""

    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RESUMABLE = "resumable"
    NON_RESUMABLE = "non_resumable"
    TERMINATED = "terminated"


def canonical_config_hash(config: Any) -> str:
    """Hash a resolved config using canonical UTF-8 JSON."""

    if isinstance(config, StrictSchema):
        config = config.to_dict()
    try:
        payload = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("resolved config must be canonical-JSON serializable") from exc
    return hashlib.sha256(payload).hexdigest()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("run name must contain at least one ASCII letter or digit")
    return slug[:48].rstrip("-")


def generate_run_id(
    name: str,
    config_hash: str,
    *,
    now: datetime | None = None,
    nonce: str | None = None,
) -> str:
    """Generate a sortable run ID bound to a resolved configuration hash."""

    if not re.fullmatch(SHA256_PATTERN, config_hash):
        raise ValueError("config_hash must be a lowercase SHA256 digest")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("run ID timestamp must be timezone-aware")
    suffix = nonce or secrets.token_hex(2)
    if not re.fullmatch(r"[a-f0-9]{4}", suffix):
        raise ValueError("run ID nonce must contain four lowercase hexadecimal characters")
    timestamp = current.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{_slugify(name)}-{config_hash[:8]}-{suffix}"


def validate_run_id(run_id: str) -> None:
    """Reject malformed IDs and path-traversal input."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("invalid run_id")


class RunManifest(StrictSchema):
    """Portable JSON fact record for one run."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    name: str = Field(min_length=1, max_length=128)
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    config_hash: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    artifact_root: Path
    strategy: Literal["single", "ddp", "fsdp2", "zero3"]
    world_size: int = Field(ge=1)
    dataset_version: str | None = None
    tokenizer_revision: str | None = None

    @model_validator(mode="after")
    def validate_identity_and_time(self) -> RunManifest:
        """Bind the ID to the config and preserve monotonic timestamps."""

        match = RUN_ID_PATTERN.fullmatch(self.run_id)
        if match is None:
            raise ValueError("invalid run_id")
        if match.group("config_hash") != self.config_hash[:8]:
            raise ValueError("run_id config hash does not match config_hash")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("run timestamps must be timezone-aware")
        if not self.artifact_root.is_absolute():
            raise ValueError("artifact_root must be an absolute path")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self
