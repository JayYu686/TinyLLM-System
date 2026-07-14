from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.cli import main
from tinyllm.training import run_single_device_training


@pytest.mark.integration
def test_public_training_command_writes_full_run_layout_and_can_exact_resume(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml")

    first = run_single_device_training(
        config_path=config_path,
        output_root=tmp_path,
        device="auto",
    )
    resumed = run_single_device_training(
        config_path=config_path,
        output_root=tmp_path,
        device="cpu",
        resume_run=first.artifact_dir,
        resume_mode="exact",
    )

    assert first.status == "succeeded"
    assert first.global_step == 30
    assert resumed.run_id == first.run_id
    assert resumed.resumed_from_step == 30
    assert resumed.checkpoint_id == first.checkpoint_id
    assert (first.artifact_dir / "run.json").is_file()
    assert (first.artifact_dir / "events.jsonl").is_file()
    assert (first.artifact_dir / "metrics.jsonl").is_file()
    assert (first.artifact_dir / "config.original.yaml").is_file()
    assert (first.artifact_dir / "config.resolved.json").is_file()
    assert (first.artifact_dir / "environment.json").is_file()
    assert (first.artifact_dir / "hardware.json").is_file()
    assert (first.artifact_dir / "evaluations").is_dir()
    assert (first.artifact_dir / "exports").is_dir()

    warm = run_single_device_training(
        config_path=config_path,
        output_root=tmp_path,
        device="cpu",
        resume_run=first.artifact_dir,
        resume_mode="warm",
    )
    assert warm.run_id != first.run_id
    assert warm.resume_mode == "warm"
    assert warm.resumed_from_step == 30
    assert warm.global_step == 30


@pytest.mark.integration
def test_train_cli_emits_stable_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "train",
            "--config",
            "configs/pretrain/tinygpt_debug_cpu_smoke.yaml",
            "--device",
            "cpu",
            "--output",
            str(tmp_path),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded"
    assert payload["global_step"] == 30
    assert payload["resume_mode"] == "none"
