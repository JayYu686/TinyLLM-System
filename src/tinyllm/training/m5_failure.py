"""Shared, side-effect-free guards for M5 training failure paths."""

from __future__ import annotations

import hashlib
import math
import shutil
from collections.abc import Callable
from pathlib import Path

M5_FULL_SFT_MINIMUM_FREE_BYTES = 64 * 1024**3
M5_LORA_MINIMUM_FREE_BYTES = 16 * 1024**3


class M5FailurePathError(RuntimeError):
    """One expected M5 preflight or runtime guard rejected unsafe state."""


def existing_storage_root(path: Path) -> Path:
    """Return the closest existing ancestor used for filesystem capacity checks."""

    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise M5FailurePathError("M5 storage path has no existing ancestor")
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def require_storage_capacity(
    path: Path,
    *,
    minimum_free_bytes: int,
    disk_usage: Callable[[Path], object] | None = None,
) -> int:
    """Reject a Run before GPU allocation when its filesystem lacks headroom."""

    if minimum_free_bytes <= 0:
        raise ValueError("M5 minimum free bytes must be positive")
    root = existing_storage_root(path)
    usage = shutil.disk_usage(root) if disk_usage is None else disk_usage(root)
    free = getattr(usage, "free", None)
    if type(free) is not int:
        raise M5FailurePathError("M5 storage preflight returned invalid capacity data")
    free_bytes = free
    if free_bytes < minimum_free_bytes:
        raise M5FailurePathError(
            "M5 storage preflight failed: "
            f"requires {minimum_free_bytes} free bytes, observed {free_bytes}"
        )
    return free_bytes


def require_world_size(*, actual: int, expected: int) -> None:
    """Reject an incompatible distributed launch before model construction."""

    if actual != expected:
        raise M5FailurePathError(f"M5 World Size mismatch: expected {expected}, observed {actual}")


def require_dataset_identity(
    *,
    actual_version: str,
    expected_version: str,
    actual_manifest_sha256: str,
    expected_manifest_sha256: str,
) -> None:
    """Reject silent dataset version or manifest drift."""

    if actual_version != expected_version or actual_manifest_sha256 != expected_manifest_sha256:
        raise M5FailurePathError("M5 Dataset identity drift detected")


def require_finite_metric(name: str, value: float) -> None:
    """Reject NaN and infinity before optimizer state can advance."""

    if not math.isfinite(value):
        raise M5FailurePathError(f"M5 training produced non-finite {name}")


def require_file_sha256(path: Path, *, expected_sha256: str) -> None:
    """Reject a missing or corrupted Checkpoint payload."""

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise M5FailurePathError("M5 Checkpoint payload is unavailable") from exc
    if digest != expected_sha256:
        raise M5FailurePathError("M5 Checkpoint integrity validation failed")


def require_child_success(return_code: int) -> None:
    """Reject a failed training child instead of accepting partial artifacts."""

    if return_code != 0:
        raise M5FailurePathError(f"M5 training child exited unsuccessfully with code {return_code}")


def normalize_cuda_oom(exc: BaseException) -> M5FailurePathError:
    """Create a stable failure for a caught CUDA out-of-memory exception."""

    return M5FailurePathError(f"M5 CUDA out of memory: {exc}")
