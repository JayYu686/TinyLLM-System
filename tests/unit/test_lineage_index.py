from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.cli import main
from tinyllm.lineage import (
    RunIndexError,
    RunIndexErrorCode,
    list_indexed_runs,
    rebuild_run_index,
    show_indexed_run,
)

RUN_A = "20260714T000000Z-first-run-aaaaaaaa-0001"
RUN_B = "20260715T000000Z-second-run-bbbbbbbb-0002"


def write_run(
    root: Path,
    run_id: str,
    *,
    campaign: str = "campaign",
    status: str = "succeeded",
    legacy: bool = False,
) -> Path:
    directory = root / "runs" / campaign / run_id
    directory.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "global_step": 12,
    }
    if not legacy:
        payload.update(
            {
                "config_sha256": run_id.rsplit("-", 2)[-2] + "0" * 56,
                "git_commit": "a" * 40,
                "git_dirty": False,
                "strategy": "ddp",
                "world_size": 2,
                "dataset_version": "dataset-v1",
            }
        )
    path = directory / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_index_rebuild_list_show_and_legacy_projection(tmp_path: Path) -> None:
    write_run(tmp_path, RUN_A, legacy=True)
    write_run(tmp_path, RUN_B, status="failed")

    result = rebuild_run_index(tmp_path)

    index = tmp_path / "registry" / "runs.sqlite3"
    assert result.indexed_runs == 2
    assert result.source_manifests == 2
    assert index.is_file()
    listed = list_indexed_runs(index, limit=10)
    assert [entry.run_id for entry in listed.runs] == [RUN_B, RUN_A]
    assert listed.runs[1].git_commit is None
    assert list_indexed_runs(index, status="failed", limit=10).runs[0].run_id == RUN_B
    shown = show_indexed_run(index, RUN_A)
    assert shown.created_at.isoformat() == "2026-07-14T00:00:00+00:00"
    assert shown.manifest_relative_path == f"runs/campaign/{RUN_A}/run.json"


def test_run_index_rebuild_replaces_old_snapshot(tmp_path: Path) -> None:
    write_run(tmp_path, RUN_A)
    first = rebuild_run_index(tmp_path)
    write_run(tmp_path, RUN_B)

    second = rebuild_run_index(tmp_path)

    assert first.indexed_runs == 1
    assert second.indexed_runs == 2
    assert first.source_tree_sha256 != second.source_tree_sha256
    assert not list((tmp_path / "registry").glob(".runs.sqlite3.tmp-*"))


def test_run_index_refuses_duplicate_or_malformed_fact_records(tmp_path: Path) -> None:
    write_run(tmp_path, RUN_A, campaign="one")
    write_run(tmp_path, RUN_A, campaign="two")

    with pytest.raises(RunIndexError) as duplicate:
        rebuild_run_index(tmp_path)

    assert duplicate.value.code == RunIndexErrorCode.SOURCE_CORRUPT
    assert "duplicate" in str(duplicate.value)


def test_run_index_cli_emits_stable_json_and_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_run(tmp_path, RUN_A)

    assert main(["run", "rebuild", "--artifact-root", str(tmp_path), "--json"]) == 0
    rebuild_payload = json.loads(capsys.readouterr().out)
    assert rebuild_payload["status"] == "succeeded"
    assert rebuild_payload["indexed_runs"] == 1

    assert (
        main(
            [
                "run",
                "show",
                RUN_A,
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run_id"] == RUN_A

    assert (
        main(
            [
                "run",
                "show",
                "../../escape",
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "RUN_INDEX_INVALID_INPUT"


def test_run_index_rejects_invalid_roots_and_outputs(tmp_path: Path) -> None:
    with pytest.raises(RunIndexError) as relative:
        rebuild_run_index(Path("relative"))
    assert relative.value.code == RunIndexErrorCode.INVALID_INPUT

    with pytest.raises(RunIndexError) as missing:
        rebuild_run_index(tmp_path)
    assert missing.value.code == RunIndexErrorCode.SOURCE_NOT_FOUND

    write_run(tmp_path, RUN_A)
    with pytest.raises(RunIndexError) as relative_output:
        rebuild_run_index(tmp_path, output_path=Path("relative.sqlite3"))
    assert relative_output.value.code == RunIndexErrorCode.INVALID_INPUT

    with pytest.raises(RunIndexError) as bad_suffix:
        rebuild_run_index(tmp_path, output_path=tmp_path / "registry" / "runs.db")
    assert bad_suffix.value.code == RunIndexErrorCode.INVALID_INPUT

    with pytest.raises(RunIndexError) as inside_runs:
        rebuild_run_index(tmp_path, output_path=tmp_path / "runs" / "index.sqlite3")
    assert inside_runs.value.code == RunIndexErrorCode.INVALID_INPUT

    output_directory = tmp_path / "registry" / "directory"
    output_directory.mkdir(parents=True)
    with pytest.raises(RunIndexError) as directory_output:
        rebuild_run_index(tmp_path, output_path=output_directory)
    assert directory_output.value.code == RunIndexErrorCode.INVALID_INPUT


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"status": ""}, "status"),
        ({"created_at": "2026-07-14T00:00:00"}, "timezone-aware"),
        ({"created_at": 42}, "ISO-8601"),
        ({"git_commit": "bad"}, "git_commit"),
        ({"git_dirty": "false"}, "boolean"),
        ({"strategy": ""}, "strategy"),
        ({"world_size": 0}, "world_size"),
        ({"global_step": True}, "global_step"),
        ({"supervised_tokens": -1}, "supervised_tokens"),
        ({"config_sha256": "bad"}, "invalid digest"),
        ({"config_hash": "c" * 64}, "disagree"),
        ({"mixture_version": "other-dataset"}, "disagree"),
        ({"latest_checkpoint": "checkpoint-a", "checkpoint_id": "checkpoint-b"}, "disagree"),
    ],
)
def test_run_index_rejects_invalid_optional_projection_fields(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    path = write_run(tmp_path, RUN_A)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    if "mixture_version" in updates:
        payload["dataset_version"] = "dataset-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunIndexError, match=message) as error:
        rebuild_run_index(tmp_path)

    assert error.value.code == RunIndexErrorCode.SOURCE_CORRUPT


def test_run_index_refuses_directory_identity_mismatch_and_symlink(tmp_path: Path) -> None:
    path = write_run(tmp_path, RUN_A)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = RUN_B
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunIndexError, match="directory"):
        rebuild_run_index(tmp_path)

    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(payload), encoding="utf-8")
    path.symlink_to(outside)
    with pytest.raises(RunIndexError, match="regular file"):
        rebuild_run_index(tmp_path)


def test_run_index_queries_fail_closed_for_missing_corrupt_and_invalid_inputs(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(RunIndexError) as not_found:
        list_indexed_runs(missing)
    assert not_found.value.code == RunIndexErrorCode.INDEX_NOT_FOUND

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(RunIndexError) as corrupt_error:
        list_indexed_runs(corrupt)
    assert corrupt_error.value.code == RunIndexErrorCode.INDEX_CORRUPT

    write_run(tmp_path, RUN_A)
    result = rebuild_run_index(tmp_path)
    index = tmp_path / result.index_relative_path
    with pytest.raises(RunIndexError) as invalid_limit:
        list_indexed_runs(index, limit=0)
    assert invalid_limit.value.code == RunIndexErrorCode.INVALID_INPUT
    with pytest.raises(RunIndexError) as empty_status:
        list_indexed_runs(index, status="")
    assert empty_status.value.code == RunIndexErrorCode.INVALID_INPUT
    with pytest.raises(RunIndexError) as long_status:
        list_indexed_runs(index, status="x" * 65)
    assert long_status.value.code == RunIndexErrorCode.INVALID_INPUT
    with pytest.raises(RunIndexError) as missing_id:
        show_indexed_run(index, RUN_B)
    assert missing_id.value.code == RunIndexErrorCode.INDEX_NOT_FOUND


def test_run_index_write_failure_is_atomic_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_run(tmp_path, RUN_A)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("tinyllm.lineage.index.os.replace", fail_replace)
    with pytest.raises(RunIndexError) as error:
        rebuild_run_index(tmp_path)

    assert error.value.code == RunIndexErrorCode.WRITE_FAILED
    assert not list((tmp_path / "registry").glob(".runs.sqlite3.tmp-*"))
