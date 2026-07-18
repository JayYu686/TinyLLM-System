from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import yaml

from tinyllm.training import FSDP2RecoveryResult
from tinyllm.training.checkpoint import CheckpointError, CheckpointErrorCode
from tinyllm.training.fsdp2_checkpoint import FSDP2CheckpointStore

CONFIG = Path("configs/fsdp2/tinygpt_debug_gloo_dcp_recovery_smoke.yaml").resolve()


def _torchrun(
    *,
    output_root: Path,
    extra: tuple[str, ...] = (),
    config_path: Path = CONFIG,
) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("torchrun")
    assert executable.is_file()
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            str(executable),
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "tinyllm.training.fsdp2_recovery_worker",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )


def _metrics(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.integration
def test_fsdp2_dcp_interruption_exact_resume_matches_uninterrupted_run(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    recovery_root = tmp_path / "recovery"
    baseline_root.mkdir()
    recovery_root.mkdir()

    baseline_phase = _torchrun(output_root=baseline_root)
    assert baseline_phase.returncode == 0, baseline_phase.stderr
    baseline = FSDP2RecoveryResult.model_validate_json(baseline_phase.stdout)

    interrupted_phase = _torchrun(
        output_root=recovery_root,
        extra=("--stop-after-step", "2"),
    )
    assert interrupted_phase.returncode != 0
    interrupted = FSDP2RecoveryResult.model_validate_json(interrupted_phase.stdout)
    assert interrupted.status == "interrupted"
    checkpoint = interrupted.artifact_dir / "checkpoints" / interrupted.checkpoint_id
    assert (checkpoint / ".metadata").is_file()
    assert len(tuple(checkpoint.glob("*.distcp"))) == 2
    assert {path.name for path in checkpoint.glob("rank-*.pt")} == {
        "rank-00000.pt",
        "rank-00001.pt",
    }

    resumed_phase = _torchrun(
        output_root=recovery_root,
        extra=("--resume-run", str(interrupted.artifact_dir)),
    )
    assert resumed_phase.returncode == 0, resumed_phase.stderr
    resumed = FSDP2RecoveryResult.model_validate_json(resumed_phase.stdout)

    assert resumed.mode == "exact_resume"
    assert resumed.resumed_from_step == 2
    assert resumed.run_id == interrupted.run_id
    assert resumed.model_parameter_sha256 == baseline.model_parameter_sha256
    assert _metrics(resumed.artifact_dir) == _metrics(baseline.artifact_dir)
    assert [cast(int, item["global_step"]) for item in _metrics(resumed.artifact_dir)] == list(
        range(1, 7)
    )

    source = interrupted.artifact_dir / "checkpoints" / interrupted.checkpoint_id
    cases = tmp_path / "invalid"
    for name in ("corrupt-shard", "missing-marker", "missing-rank"):
        root = cases / name
        target = root / interrupted.checkpoint_id
        target.parent.mkdir(parents=True)
        shutil.copytree(source, target)
        if name == "corrupt-shard":
            shard = next(target.glob("*.distcp"))
            with shard.open("ab") as stream:
                stream.write(b"corrupt")
        else:
            if name == "missing-marker":
                (target / "COMMITTED").unlink()
            else:
                (target / "rank-00001.pt").unlink()
        store = FSDP2CheckpointStore(root, keep_last=2)
        with pytest.raises(CheckpointError) as caught:
            store.validate(interrupted.checkpoint_id, expected_world_size=2)
        assert caught.value.code in {
            CheckpointErrorCode.CHECKPOINT_CORRUPT,
            CheckpointErrorCode.CHECKPOINT_INCOMPLETE,
        }

    store = FSDP2CheckpointStore(interrupted.artifact_dir / "checkpoints", keep_last=2)
    with pytest.raises(CheckpointError) as caught:
        store.validate(interrupted.checkpoint_id, expected_world_size=4)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE

    drift_config = tmp_path / "config-drift.yaml"
    raw_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw_config["training"]["learning_rate"] = 4.0e-4
    drift_config.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config_drift = _torchrun(
        output_root=recovery_root,
        config_path=drift_config,
        extra=("--resume-run", str(interrupted.artifact_dir)),
    )
    assert config_drift.returncode != 0
    assert "CHECKPOINT_INCOMPATIBLE" in config_drift.stderr

    data_drift_run = tmp_path / "data-drift-run"
    shutil.copytree(interrupted.artifact_dir, data_drift_run)
    run_manifest_path = data_drift_run / "run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["dataset_version"] = "different-dataset-version"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_drift = _torchrun(
        output_root=recovery_root,
        extra=("--resume-run", str(data_drift_run)),
    )
    assert data_drift.returncode != 0
    assert "CHECKPOINT_INCOMPATIBLE" in data_drift.stderr
