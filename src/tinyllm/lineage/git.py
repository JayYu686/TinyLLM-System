"""Read-only Git identity collection for run and smoke evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path


def read_git_identity(project_root: Path) -> tuple[str, bool]:
    """Return the checked-out commit and whether tracked source differs from it."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, bool(status)
