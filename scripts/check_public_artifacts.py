#!/usr/bin/env python3
"""Reject secrets and private host identity in public repository artifacts."""

from __future__ import annotations

import re
from pathlib import Path

SKIPPED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-baseline",
}
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
REPORT_PRIVATE_PATTERNS = {
    "private home path": re.compile(r"/home/yujielun(?:/|\b)"),
    "private hostname": re.compile(r"\bsitonholy\b", re.IGNORECASE),
    "Windows path": re.compile(r"\b[A-Za-z]:\\"),
}


def public_text_files(root: Path) -> list[Path]:
    """Return repository text artifacts while excluding local caches."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in TEXT_SUFFIXES
        and not any(part in SKIPPED_PARTS for part in path.parts)
    )


def find_violations(root: Path) -> list[str]:
    """Return path-level policy violations without echoing secret values."""

    violations: list[str] = []
    report_root = root / "reports"
    for path in public_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{path.relative_to(root)}: possible {label}")
        if path.is_relative_to(report_root):
            for label, pattern in REPORT_PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    violations.append(f"{path.relative_to(root)}: {label}")
    return violations


def main() -> int:
    """Check committed public material before it is pushed."""

    root = Path(__file__).resolve().parents[1]
    violations = find_violations(root)
    if violations:
        print("\n".join(violations))
        return 1
    print("public artifact policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
