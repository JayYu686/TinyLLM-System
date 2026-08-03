from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from pydantic import ValidationError

import scripts.run_m5_failure_path_smoke as smoke_script
from tinyllm.training.m5_failure import (
    M5FailurePathError,
    normalize_cuda_oom,
    require_child_success,
    require_dataset_identity,
    require_file_sha256,
    require_finite_metric,
    require_storage_capacity,
    require_world_size,
)
from tinyllm.training.m5_failure_schema import M5_FAILURE_PATHS, M5FailurePathEvidence


class Usage(NamedTuple):
    total: int
    used: int
    free: int


def test_storage_preflight_uses_existing_ancestor_and_rejects_shortage(tmp_path: Path) -> None:
    observed: list[Path] = []

    def usage(path: Path) -> Usage:
        observed.append(path)
        return Usage(total=100, used=99, free=1)

    with pytest.raises(M5FailurePathError, match="storage preflight failed"):
        require_storage_capacity(
            tmp_path / "future" / "runs",
            minimum_free_bytes=2,
            disk_usage=usage,
        )

    assert observed == [tmp_path.resolve()]


def test_common_guards_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(M5FailurePathError, match="World Size mismatch"):
        require_world_size(actual=2, expected=4)
    with pytest.raises(M5FailurePathError, match="identity drift"):
        require_dataset_identity(
            actual_version="changed",
            expected_version="frozen",
            actual_manifest_sha256="a" * 64,
            expected_manifest_sha256="b" * 64,
        )
    with pytest.raises(M5FailurePathError, match="non-finite loss"):
        require_finite_metric("loss", float("inf"))
    with pytest.raises(M5FailurePathError, match="code 17"):
        require_child_success(17)
    assert "CUDA out of memory" in str(normalize_cuda_oom(torch.OutOfMemoryError("injected")))

    checkpoint = tmp_path / "training_state.pt"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(M5FailurePathError, match="integrity"):
        require_file_sha256(
            checkpoint,
            expected_sha256=hashlib.sha256(b"valid").hexdigest(),
        )


def test_safe_failure_smoke_covers_complete_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        smoke_script,
        "read_git_identity",
        lambda _: ("a" * 40, False),
    )
    generated_at = datetime(2026, 8, 3, tzinfo=UTC)

    result = smoke_script.run_smoke(project_root=tmp_path, generated_at=generated_at)

    assert result.status == "passed"
    assert result.generated_at == generated_at
    assert tuple(item.name for item in result.cases) == M5_FAILURE_PATHS
    assert all(item.status == "rejected_as_expected" for item in result.cases)
    assert result.model_generated is False
    assert result.quality_metric is False


def test_failure_evidence_rejects_reordered_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        smoke_script,
        "read_git_identity",
        lambda _: ("a" * 40, False),
    )
    result = smoke_script.run_smoke(
        project_root=tmp_path,
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
    )

    with pytest.raises(ValidationError, match="incomplete or unordered"):
        M5FailurePathEvidence.model_validate(
            result.model_dump() | {"cases": tuple(reversed(result.cases))}
        )
