from __future__ import annotations

from pathlib import Path

from scripts.check_markdown_links import broken_links
from scripts.check_public_artifacts import find_violations


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
