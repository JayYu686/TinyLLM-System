from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tinyllm.cli as cli_module
import tinyllm.evaluation.m6_candidate as m6_candidate_module
from tinyllm.cli import main
from tinyllm.evaluation import (
    M6CandidateImportError,
    M6CandidateImportResult,
    M6ModelIdentity,
    import_m5_candidate_evidence,
    load_m6_candidate_import,
    model_export_sha256,
    sha256_file,
)
from tinyllm.training.m5_ablation_schema import M5AblationRunResult, M5CheckpointManifest


def _candidate_import() -> M6CandidateImportResult:
    return M6CandidateImportResult(
        status="succeeded",
        protocol_version="m6-release-v1",
        config_sha256="a" * 64,
        source_run_id="20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
        source_result_sha256="b" * 64,
        source_git_commit="c" * 40,
        source_environment_sha256="d" * 64,
        source_hardware_sha256="e" * 64,
        checkpoint_manifest_sha256="f" * 64,
        snapshot_sha256="1" * 64,
        model=M6ModelIdentity(
            role="candidate",
            repository="Qwen/Qwen3-0.6B",
            base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            attention_architecture="gqa",
            adaptation="full_sft",
            model_artifact_sha256=(
                "b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c"
            ),
            model_parameters=596_049_920,
            training_run_id="20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
            training_checkpoint_id="checkpoint-tokens-0010000532",
            training_tokens=10_000_532,
            training_config_sha256="2" * 64,
            dataset_version="m5-dual-sft-v1-b5b9e839",
            dataset_manifest_sha256="3" * 64,
        ),
    )


def test_m6_candidate_import_freezes_selected_snapshot() -> None:
    imported = _candidate_import()
    assert imported.model.training_tokens == 10_000_532
    with pytest.raises(ValidationError, match="frozen M5 10M snapshot"):
        M6CandidateImportResult.model_validate(
            imported.to_dict()
            | {"model": imported.model.to_dict() | {"training_tokens": 20_001_758}}
        )


def test_m6_candidate_import_accepts_only_aligned_correction_identity() -> None:
    model = _candidate_import().model.model_copy(
        update={
            "model_artifact_sha256": "4" * 64,
            "training_run_id": "20260810T095257Z-m6-dual-mode-fix-seed42-ffd49bd9-aeb6",
            "training_checkpoint_id": "checkpoint-tokens-0001000000",
            "training_tokens": 1_000_000,
            "training_config_sha256": "5" * 64,
            "dataset_version": "m5-dual-mode-correction-mixture-v1-4bc342d4",
            "dataset_manifest_sha256": (
                "db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c"
            ),
        }
    )
    imported = _candidate_import().model_copy(
        update={
            "source_kind": "m6-dual-mode-correction",
            "protocol_version": "m6-release-v2",
            "model": model,
        }
    )

    assert imported.model.training_tokens == 1_000_000
    with pytest.raises(ValidationError, match="correction contract"):
        M6CandidateImportResult.model_validate(
            imported.to_dict() | {"model": imported.model.to_dict() | {"training_tokens": 999_999}}
        )


def test_m6_candidate_import_accepts_only_preregistered_gate_replay() -> None:
    model = _candidate_import().model.model_copy(
        update={
            "model_artifact_sha256": "4" * 64,
            "training_run_id": "20260812T000000Z-m6-gate-replay-r3-seed42-a1b2c3d4-cafe",
            "training_checkpoint_id": "checkpoint-tokens-0001000000",
            "training_tokens": 1_000_000,
            "training_config_sha256": "5" * 64,
            "dataset_version": "m6-gate-replay-mixture-v1-6c169970",
            "dataset_manifest_sha256": (
                "c5ceb1e5597a8e253d7c370484f9aa06d22b0a26dbfe597043d9302d8e580fa9"
            ),
        }
    )
    imported = _candidate_import().model_copy(
        update={
            "source_kind": "m6-gate-replay",
            "protocol_version": "m6-release-v3",
            "model": model,
        }
    )

    assert imported.model.dataset_version == "m6-gate-replay-mixture-v1-6c169970"
    with pytest.raises(ValidationError, match="gate-replay contract"):
        M6CandidateImportResult.model_validate(
            imported.to_dict()
            | {"model": imported.model.to_dict() | {"dataset_manifest_sha256": "0" * 64}}
        )


def test_m6_candidate_import_accepts_only_warm_started_domain_generalization() -> None:
    model = _candidate_import().model.model_copy(
        update={
            "model_artifact_sha256": "4" * 64,
            "training_run_id": "20260812T000000Z-m6-domain-generalization-r4-seed42-a1b2c3d4-cafe",
            "training_checkpoint_id": "checkpoint-tokens-0001000000",
            "training_tokens": 1_000_000,
            "training_config_sha256": "5" * 64,
            "dataset_version": "m6-domain-generalization-mixture-v1-6c2f59e6",
            "dataset_manifest_sha256": (
                "40c7a85edb392b165e2a05f50dbe998cc62ffe96115af27896bf8d5d15401eb9"
            ),
        }
    )
    imported = _candidate_import().model_copy(
        update={
            "source_kind": "m6-domain-generalization",
            "protocol_version": "m6-release-v4",
            "model": model,
        }
    )

    assert imported.model.dataset_version == "m6-domain-generalization-mixture-v1-6c2f59e6"
    with pytest.raises(ValidationError, match="domain-generalization contract"):
        M6CandidateImportResult.model_validate(
            imported.to_dict()
            | {"model": imported.model.to_dict() | {"dataset_manifest_sha256": "0" * 64}}
        )


def test_m6_correction_import_binds_run_checkpoint_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = (tmp_path / "20260810T095257Z-m6-dual-mode-fix-seed42-ffd49bd9-aeb6").resolve()
    model = run / "exports/model"
    checkpoint = run / "checkpoints/checkpoint-tokens-0001000000"
    model.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"model")
    (run / "result.json").write_text("{}", encoding="utf-8")
    (run / "config.original.yaml").write_text("config", encoding="utf-8")
    (run / "environment.json").write_text("{}", encoding="utf-8")
    (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
    state = checkpoint / "training_state.pt"
    state.write_bytes(b"state")
    config_sha256 = "5" * 64
    mixture_sha256 = "db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c"
    result = SimpleNamespace(
        status="succeeded",
        run_id=run.name,
        git_dirty=False,
        config_sha256=config_sha256,
        supervised_tokens=1_000_000,
        latest_checkpoint="checkpoint-tokens-0001000000",
        mixture_version="m5-dual-mode-correction-mixture-v1-4bc342d4",
        mixture_manifest_sha256=mixture_sha256,
        export_sha256="4" * 64,
        gpu_name="NVIDIA GeForce RTX 3090",
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        physical_gpu_index=4,
        git_commit="6" * 40,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_version=result.mixture_version,
            mix_manifest_sha256=mixture_sha256,
        ),
        evaluation=SimpleNamespace(consume_m6_frozen_results=False),
    )
    manifest = SimpleNamespace(
        run_id=run.name,
        checkpoint_id=result.latest_checkpoint,
        supervised_tokens=1_000_000,
        config_sha256=config_sha256,
        mixture_version=result.mixture_version,
        mixture_manifest_sha256=mixture_sha256,
        git_commit=result.git_commit,
        pinned=True,
        pin_reason="final",
        file=SimpleNamespace(path="training_state.pt", sha256=sha256_file(state)),
    )
    monkeypatch.setattr(
        M5AblationRunResult,
        "model_validate_json",
        lambda _: result,
    )
    monkeypatch.setattr(
        M5CheckpointManifest,
        "model_validate_json",
        lambda _: manifest,
    )
    monkeypatch.setattr(m6_candidate_module, "load_m5_sft_config", lambda _: config)
    monkeypatch.setattr(m6_candidate_module, "canonical_config_hash", lambda _: config_sha256)
    monkeypatch.setattr(m6_candidate_module, "model_export_sha256", lambda _: "4" * 64)

    imported = import_m5_candidate_evidence(
        release_config_path=Path("configs/eval/m6_release_v2.yaml"),
        source_run=run,
        model_dir=model,
    )

    assert imported.source_kind == "m6-dual-mode-correction"
    assert imported.model.training_run_id == run.name
    assert imported.model.training_checkpoint_id == result.latest_checkpoint
    assert imported.model.model_artifact_sha256 == "4" * 64


def test_m6_candidate_export_hash_is_stable_and_rejects_symlinks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
    assert model_export_sha256(first.resolve()) == model_export_sha256(second.resolve())
    (second / "unsafe").symlink_to(second / "config.json")
    with pytest.raises(M6CandidateImportError, match="non-regular"):
        model_export_sha256(second.resolve())

    empty = (tmp_path / "empty").resolve()
    empty.mkdir()
    with pytest.raises(M6CandidateImportError, match="empty"):
        model_export_sha256(empty)
    with pytest.raises(M6CandidateImportError, match="missing or unsafe"):
        model_export_sha256((tmp_path / "missing").resolve())


def test_m6_candidate_import_cli_emits_stable_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imported = _candidate_import()
    monkeypatch.setattr(cli_module, "import_m5_candidate_evidence", lambda **_: imported)
    source = (tmp_path / "source").resolve()
    model = (tmp_path / "model").resolve()
    output = (tmp_path / "candidate.json").resolve()

    assert (
        main(
            [
                "--json",
                "eval",
                "m6-import-candidate",
                "--source-run",
                str(source),
                "--model-dir",
                str(model),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["model"]["role"] == "candidate"


def test_m6_candidate_import_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(_candidate_import().to_dict() | {"unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(M6CandidateImportError, match="invalid"):
        load_m6_candidate_import(path)


def test_m6_candidate_cli_rejects_relative_paths_and_maps_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "eval",
                "m6-import-candidate",
                "--source-run",
                "relative",
                "--model-dir",
                str((tmp_path / "model").resolve()),
                "--output",
                str((tmp_path / "output.json").resolve()),
            ]
        )
        == 2
    )
    capsys.readouterr()
    monkeypatch.setattr(
        cli_module,
        "import_m5_candidate_evidence",
        lambda **_: (_ for _ in ()).throw(M6CandidateImportError("broken lineage")),
    )
    assert (
        main(
            [
                "--json",
                "eval",
                "m6-import-candidate",
                "--source-run",
                str((tmp_path / "source").resolve()),
                "--model-dir",
                str((tmp_path / "model").resolve()),
                "--output",
                str((tmp_path / "output.json").resolve()),
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ("M6_CANDIDATE_IMPORT_FAILED")


def test_m6_candidate_domain_cli_runs_shared_gpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imported = _candidate_import()
    monkeypatch.setattr(cli_module, "load_m6_candidate_import", lambda _: imported)
    monkeypatch.setattr(cli_module, "preflight_baseline_gpu", lambda _: None)
    monkeypatch.setattr(
        cli_module,
        "run_m6_domain_pass",
        lambda **_: SimpleNamespace(
            status="awaiting_human_review",
            evaluation_id="candidate-thinking",
            objective_correct_items=200,
        ),
    )
    paths = [str((tmp_path / name).resolve()) for name in ("import", "model", "tokenizer", "out")]

    assert (
        main(
            [
                "eval",
                "m6-candidate-domain",
                "--candidate-import",
                paths[0],
                "--model-dir",
                paths[1],
                "--tokenizer-dir",
                paths[2],
                "--output-dir",
                paths[3],
                "--gpu-index",
                "7",
                "--mode",
                "thinking",
            ]
        )
        == 0
    )
    assert "candidate-thinking" in capsys.readouterr().out


def test_m6_candidate_import_binds_full_m5_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15"
    checkpoint_id = "checkpoint-tokens-0010000532"
    snapshot_root = run / "evaluations" / checkpoint_id
    model_dir = snapshot_root / "model"
    checkpoint_root = run / "checkpoints" / checkpoint_id
    model_dir.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    shutil.copyfile("configs/sft/m5_formal_qwen3_0_6b.yaml", run / "config.original.yaml")
    (run / "environment.json").write_text('{"environment": "test"}\n', encoding="utf-8")
    (run / "hardware.json").write_text('{"hardware": "test"}\n', encoding="utf-8")
    environment_sha = sha256_file(run / "environment.json")
    hardware_sha = sha256_file(run / "hardware.json")
    run_id = run.name
    config_sha = "d39dad3534730dfde08d526f24a69344d3be1341a097e610eaec7038041ad676"
    git_commit = "c406e6760c6ea6b5eb19966740af4c494983576d"
    dataset_sha = "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    manifest = {
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "global_step": 10,
        "local_sequence_cursor": 10,
        "supervised_tokens": 10_000_532,
        "dataset_epoch": 10.000532,
        "config_sha256": config_sha,
        "dataset_version": "m5-dual-sft-v1-b5b9e839",
        "dataset_manifest_sha256": dataset_sha,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "git_commit": git_commit,
        "environment_sha256": environment_sha,
        "hardware_sha256": hardware_sha,
        "file": {"path": "training_state.pt", "size_bytes": 1, "sha256": "4" * 64},
        "pinned": True,
        "pin_reason": "evaluation",
    }
    (checkpoint_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    manifest_sha = sha256_file(checkpoint_root / "manifest.json")
    export_sha = "b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c"
    snapshot = {
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "target_tokens": 10_000_000,
        "supervised_tokens": 10_000_532,
        "checkpoint_manifest_sha256": manifest_sha,
        "export_sha256": export_sha,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "git_commit": git_commit,
    }
    (snapshot_root / "snapshot.json").write_text(
        json.dumps(snapshot, sort_keys=True), encoding="utf-8"
    )
    checkpoints = (
        checkpoint_id,
        "checkpoint-tokens-0020001758",
        "checkpoint-tokens-0030002588",
        "checkpoint-tokens-0040004805",
        "checkpoint-tokens-0050000000",
    )
    result = {
        "status": "succeeded",
        "mode": "exact_resume",
        "run_id": run_id,
        "config_sha256": config_sha,
        "git_commit": git_commit,
        "git_dirty": False,
        "environment_sha256": environment_sha,
        "hardware_sha256": hardware_sha,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "dataset_version": "m5-dual-sft-v1-b5b9e839",
        "dataset_manifest_sha256": dataset_sha,
        "thinking_fraction_basis_points": 3000,
        "seed": 42,
        "world_size": 4,
        "global_step": 100,
        "local_sequence_cursor": 100,
        "supervised_tokens": 50_000_000,
        "completed_dataset_epochs": 50.0,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "duration_seconds": 60.0,
        "rank_memory": tuple(
            {
                "rank": rank,
                "physical_gpu_index": rank + 4,
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "peak_allocated_bytes": 10,
                "peak_reserved_bytes": 20,
            }
            for rank in range(4)
        ),
        "latest_checkpoint": checkpoints[-1],
        "evaluation_checkpoints": checkpoints,
        "evaluation_export_sha256s": (export_sha, "5" * 64, "6" * 64, "7" * 64, "8" * 64),
        "resumed_from_tokens": 2_000_000,
        "export_sha256": "9" * 64,
    }
    (run / "result.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(m6_candidate_module, "model_export_sha256", lambda _: export_sha)
    output = (tmp_path / "candidate.json").resolve()

    imported = import_m5_candidate_evidence(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        source_run=run.resolve(),
        model_dir=model_dir.resolve(),
        output_path=output,
    )

    assert imported.model.model_artifact_sha256 == export_sha
    assert load_m6_candidate_import(output) == imported
    (run / "environment.json").write_text('{"environment": "changed"}\n', encoding="utf-8")
    with pytest.raises(M6CandidateImportError, match="incomplete or inconsistent"):
        import_m5_candidate_evidence(
            release_config_path=Path("configs/eval/m6_release.yaml"),
            source_run=run.resolve(),
            model_dir=model_dir.resolve(),
        )
    (run / "environment.json").write_text('{"environment": "test"}\n', encoding="utf-8")
    with pytest.raises(M6CandidateImportError, match="output path must be absolute"):
        import_m5_candidate_evidence(
            release_config_path=Path("configs/eval/m6_release.yaml"),
            source_run=run.resolve(),
            model_dir=model_dir.resolve(),
            output_path=Path("relative.json"),
        )
