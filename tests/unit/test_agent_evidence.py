from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.agent import EvidenceIndexError, rebuild_evidence_index, search_evidence


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    artifact = tmp_path / "artifacts"
    (project / "docs").mkdir(parents=True)
    (project / "reports" / "m7").mkdir(parents=True)
    (artifact / "registry" / "candidates" / "candidate").mkdir(parents=True)
    (artifact / "runs" / "run-1").mkdir(parents=True)
    (project / "README.md").write_text("# TinyLLM\n\nServing platform.\n", encoding="utf-8")
    (project / "docs" / "recovery.md").write_text(
        "# Recovery\n\nBackend crash recovery preserves the production model.\n",
        encoding="utf-8",
    )
    (project / "reports" / "m7" / "result.md").write_text(
        "# Result\n\nReadiness recovered after the backend crash.\n",
        encoding="utf-8",
    )
    (artifact / "registry" / "candidates" / "candidate" / "model.json").write_text(
        json.dumps({"model_version": "candidate", "status": "Candidate"}),
        encoding="utf-8",
    )
    (artifact / "runs" / "run-1" / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "succeeded",
                "git_commit": "a" * 40,
                "private_path": "/secret/path",
                "prompt": "must not be indexed",
            }
        ),
        encoding="utf-8",
    )
    return project, artifact


def test_rebuild_and_search_evidence_index(tmp_path: Path) -> None:
    project, artifact = _roots(tmp_path)
    output = tmp_path / "index"

    manifest = rebuild_evidence_index(
        project_root=project, artifact_root=artifact, output_dir=output
    )
    results = search_evidence(index_dir=output, query="backend recovery", limit=5)

    assert manifest.documents == 5
    assert manifest.chunks == 5
    assert manifest.index_version.startswith("m8-evidence-")
    assert results
    assert results[0].relative_path in {
        "docs/recovery.md",
        "reports/m7/result.md",
    }
    assert results[0].start_line == 1
    assert results[0].end_line == 3
    assert all(not result.relative_path.startswith("/") for result in results)


def test_run_metadata_is_allowlisted_before_indexing(tmp_path: Path) -> None:
    project, artifact = _roots(tmp_path)
    output = tmp_path / "index"
    rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=output)

    assert search_evidence(index_dir=output, query="succeeded")
    assert search_evidence(index_dir=output, query="private_path") == ()
    assert search_evidence(index_dir=output, query="prompt") == ()


def test_search_rejects_hash_drift_and_invalid_limits(tmp_path: Path) -> None:
    project, artifact = _roots(tmp_path)
    output = tmp_path / "index"
    rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=output)
    database = output / "evidence.sqlite3"
    database.write_bytes(database.read_bytes() + b"drift")

    with pytest.raises(EvidenceIndexError, match="hash drift"):
        search_evidence(index_dir=output, query="recovery")
    with pytest.raises(EvidenceIndexError, match="limits"):
        search_evidence(index_dir=output, query="recovery", limit=21)


def test_index_paths_and_queries_fail_closed(tmp_path: Path) -> None:
    project, artifact = _roots(tmp_path)
    output = tmp_path / "index"
    output.mkdir()
    with pytest.raises(EvidenceIndexError, match="new absolute"):
        rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=output)

    output = tmp_path / "fresh-index"
    rebuild_evidence_index(project_root=project, artifact_root=artifact, output_dir=output)
    with pytest.raises(EvidenceIndexError, match="searchable terms"):
        search_evidence(index_dir=output, query="' ; --")
