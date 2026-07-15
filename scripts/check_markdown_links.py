#!/usr/bin/env python3
"""Validate repository-local targets in Markdown links."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")
IGNORED_PREFIXES = ("#", "http://", "https://", "mailto:")
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-baseline",
}


def markdown_files(root: Path) -> list[Path]:
    """Return public Markdown files without inspecting local tool caches."""

    return sorted(
        path for path in root.rglob("*.md") if not any(part in IGNORED_PARTS for part in path.parts)
    )


def broken_links(path: Path) -> list[str]:
    """Return missing repository-local link targets in one file."""

    failures: list[str] = []
    for match in LINK_PATTERN.finditer(path.read_text(encoding="utf-8")):
        raw_target = match.group("target").strip()
        if raw_target.startswith("<") and raw_target.endswith(">"):
            raw_target = raw_target[1:-1]
        target = unquote(raw_target.split("#", maxsplit=1)[0])
        if not target or target.startswith(IGNORED_PREFIXES):
            continue
        candidate = path.parent / target
        if not candidate.exists():
            failures.append(f"{path}: missing local link target: {raw_target}")
    return failures


def main() -> int:
    """Validate every Markdown file below the repository root."""

    root = Path(__file__).resolve().parents[1]
    failures = [failure for path in markdown_files(root) for failure in broken_links(path)]
    if failures:
        print("\n".join(failures))
        return 1
    print(f"checked {len(markdown_files(root))} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
