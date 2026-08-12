"""Versioned schemas for the rebuildable Run query index."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import GIT_COMMIT_PATTERN, SHA256_PATTERN, validate_run_id


class RunIndexEntry(StrictSchema):
    """A content-free query projection of one immutable ``run.json`` fact record."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    created_at: datetime
    status: str = Field(min_length=1, max_length=64)
    manifest_relative_path: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    git_commit: str | None = Field(default=None, pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool | None = None
    strategy: str | None = Field(default=None, min_length=1, max_length=64)
    world_size: int | None = Field(default=None, ge=1)
    dataset_version: str | None = Field(default=None, min_length=1, max_length=256)
    latest_checkpoint: str | None = Field(default=None, min_length=1, max_length=256)
    global_step: int | None = Field(default=None, ge=0)
    supervised_tokens: int | None = Field(default=None, ge=0)

    @field_validator("run_id")
    @classmethod
    def require_valid_run_id(cls, value: str) -> str:
        """Keep query identities aligned with the public Run ID contract."""

        validate_run_id(value)
        return value

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous local timestamps in the lineage index."""

        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("manifest_relative_path")
    @classmethod
    def require_run_manifest_path(cls, value: str) -> str:
        """Keep public index records path-relative and below ``runs/``."""

        if not value.startswith("runs/") or not value.endswith("/run.json"):
            raise ValueError("manifest path must be relative below runs/")
        if ".." in value.split("/") or value.startswith("/"):
            raise ValueError("manifest path must not escape the Artifact Store")
        return value


class RunIndexRebuildResult(StrictSchema):
    """Stable result emitted after an atomic index rebuild."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"] = "succeeded"
    index_schema_version: Literal[1] = 1
    index_relative_path: str = "registry/runs.sqlite3"
    index_sha256: str = Field(pattern=SHA256_PATTERN)
    source_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifests: int = Field(ge=0)
    indexed_runs: int = Field(ge=0)

    @model_validator(mode="after")
    def require_complete_projection(self) -> RunIndexRebuildResult:
        """A successful rebuild cannot silently skip source manifests."""

        if self.source_manifests != self.indexed_runs:
            raise ValueError("every source manifest must be indexed")
        if self.index_relative_path.startswith("/") or ".." in self.index_relative_path.split("/"):
            raise ValueError("index path must be a safe relative path")
        return self


class RunIndexListResult(StrictSchema):
    """Stable paginated result for ``tinyllm run list``."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"] = "succeeded"
    status_filter: str | None = None
    limit: int = Field(ge=1, le=1000)
    returned_runs: int = Field(ge=0)
    runs: tuple[RunIndexEntry, ...]

    @model_validator(mode="after")
    def require_consistent_count(self) -> RunIndexListResult:
        """Bind the pagination count to the actual returned records."""

        if self.returned_runs != len(self.runs):
            raise ValueError("returned_runs must match runs")
        if self.returned_runs > self.limit:
            raise ValueError("returned_runs cannot exceed limit")
        return self
