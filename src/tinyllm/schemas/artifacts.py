"""Artifact-root and run-layout contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator

from tinyllm.schemas.base import StrictSchema

DEFAULT_ARTIFACT_ROOT = Path("/data/yujielun/tinyllm")


class ArtifactRoots(StrictSchema):
    """Resolve private artifact locations without creating them."""

    schema_version: Literal["1.0"] = "1.0"
    root: Path = DEFAULT_ARTIFACT_ROOT

    @field_validator("root")
    @classmethod
    def require_absolute_root(cls, value: Path) -> Path:
        """Prevent a run layout from depending on the caller's working directory."""

        if not value.is_absolute():
            raise ValueError("artifact root must be an absolute path")
        return value

    @property
    def cache(self) -> Path:
        """Return the shared download/cache directory."""

        return self.root / "cache"

    @property
    def datasets(self) -> Path:
        """Return the versioned dataset directory."""

        return self.root / "datasets"

    @property
    def models(self) -> Path:
        """Return the model and export directory."""

        return self.root / "models"

    @property
    def runs(self) -> Path:
        """Return the immutable run directory root."""

        return self.root / "runs"

    @property
    def registry(self) -> Path:
        """Return the rebuildable local registry directory."""

        return self.root / "registry"

    def run_directory(self, run_id: str) -> Path:
        """Resolve a validated run ID below the run root."""

        from tinyllm.schemas.run import validate_run_id

        validate_run_id(run_id)
        return self.runs / run_id
