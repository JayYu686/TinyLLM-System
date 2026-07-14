import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tinyllm.cli import main
from tinyllm.data import RegisteredDatasetSummary


def test_help_lists_doctor_train_and_data(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "doctor" in output
    assert "train" in output
    assert "data" in output

    assert main(["data", "--help"]) == 0
    data_output = capsys.readouterr().out
    assert "prepare" in data_output
    assert "inspect" in data_output


def test_version_is_stable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "tinyllm 0.1.0a1"


def test_missing_project_root_returns_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing"
    code = main(["doctor", "--json", "--project-root", str(missing)])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "CLI_OUTPUT_ERROR"


def test_missing_output_parent_returns_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "missing" / "report.json"
    code = main(["doctor", "--json", "--output", str(output)])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "CLI_OUTPUT_ERROR"


def test_invalid_command_returns_click_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["not-a-command"]) == 2
    assert "No such command" in capsys.readouterr().err


def test_output_directory_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor", "--json", "--output", str(tmp_path)])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert "output path is a directory" in payload["error"]["message"]


def test_train_rejects_invalid_runtime_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "train",
            "--config",
            "configs/pretrain/tinygpt_debug_cpu_smoke.yaml",
            "--device",
            "tpu",
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert "device must be" in payload["error"]["message"]

    code = main(
        [
            "train",
            "--config",
            "configs/pretrain/tinygpt_debug_cpu_smoke.yaml",
            "--resume-mode",
            "guess",
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert "resume mode must be" in payload["error"]["message"]

    code = main(
        [
            "train",
            "--config",
            "configs/pretrain/tinygpt_debug_cpu_smoke.yaml",
            "--resume-run",
            "/definitely/missing/run",
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert "resume Run directory" in payload["error"]["message"]


def test_data_inspect_exposes_pinned_contract_as_stable_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["data", "inspect", "--source", "commitpackft", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "ok"
    assert payload["stage"] == "import_contract"
    assert payload["sources"][0]["source"]["dataset_id"] == "bigcode/commitpackft"
    assert "mit" in payload["sources"][0]["source_license_allowlist"]


def test_data_inspect_rejects_unknown_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "inspect", "--source", "floating-latest", "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert "data source must be" in payload["error"]["message"]


def test_data_inspect_rejects_invalid_or_missing_registered_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "data",
                "inspect",
                "--dataset-version",
                "../../escape",
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "DATASET_INVALID_INPUT"

    assert (
        main(
            [
                "data",
                "inspect",
                "--dataset-version",
                "m2-sft-v1-aaaaaaaa",
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "DATASET_NOT_FOUND"


def test_data_prepare_offline_miss_is_preflight_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "data",
                "prepare",
                "--artifact-root",
                str(tmp_path),
                "--offline",
                "--json",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "DATA_ACQUISITION_ERROR"
    assert "offline cache miss" in payload["error"]["message"]


def test_data_prepare_emits_stable_path_free_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = RegisteredDatasetSummary(
        status="ok",
        operation="prepare",
        created=True,
        verified=True,
        dataset_name="m2-sft",
        dataset_version="m2-sft-v1-aaaaaaaa",
        content_sha256="a" * 64,
        storage_format="numpy-sharded-v1",
        registered_at=datetime(2026, 7, 14, tzinfo=UTC),
        git_commit="b" * 40,
        git_dirty=False,
        source_rows={"commitpackft": 4, "oasst1": 6},
        imported_samples={"commitpackft": 4, "oasst1": 6},
        processed_samples=10,
        tokenized_samples=10,
        balanced_samples=10,
        packed_sequences=1,
        total_tokens=100,
        total_supervised_tokens=10,
        rejection_counts={},
        registered_files=20,
        registered_bytes=4096,
    )
    monkeypatch.setattr("tinyllm.cli.prepare_m2_dataset", lambda **_kwargs: summary)

    assert (
        main(
            [
                "data",
                "prepare",
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == summary.to_dict()
    assert str(tmp_path) not in json.dumps(payload)
