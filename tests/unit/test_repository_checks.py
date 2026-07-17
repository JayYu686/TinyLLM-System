from __future__ import annotations

from pathlib import Path

from scripts.check_markdown_links import broken_links, markdown_files
from scripts.check_public_artifacts import find_violations, public_text_files


def test_markdown_link_check_reports_missing_local_target(tmp_path: Path) -> None:
    document = tmp_path / "README.md"
    document.write_text(
        "[missing](docs/missing.md) [remote](https://example.com)\n",
        encoding="utf-8",
    )

    assert broken_links(document) == [f"{document}: missing local link target: docs/missing.md"]


def test_public_artifact_check_redacts_violation_value(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "hardware" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("host: sitonholy\n", encoding="utf-8")

    violations = find_violations(tmp_path)

    assert violations == ["reports/hardware/report.md: private hostname"]
    assert "sitonholy" not in violations[0]


def test_repository_checks_ignore_isolated_dependency_environments(tmp_path: Path) -> None:
    for environment in (".venv-baseline", ".venv-m4"):
        private = tmp_path / environment / "package"
        private.mkdir(parents=True)
        (private / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")
        (private / "metadata.txt").write_text("host: sitonholy\n", encoding="utf-8")
    public = tmp_path / "README.md"
    public.write_text("# Public\n", encoding="utf-8")

    assert markdown_files(tmp_path) == [public]
    assert public_text_files(tmp_path) == [public]
