import json
from pathlib import Path

import pytest

from tinyllm.cli import main


def test_help_lists_doctor_and_train(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "doctor" in output
    assert "train" in output


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
