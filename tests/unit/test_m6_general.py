from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import tinyllm.cli as cli_module
import tinyllm.evaluation.m6_general as m6_general_module
from tinyllm.cli import main
from tinyllm.evaluation import (
    BaselinePreflightError,
    GeneralBaselineSummary,
    GeneralTaskResult,
    M6CandidateImportError,
    M6CandidateImportResult,
    M6GeneralError,
    M6GeneralPassSummary,
    M6ModelIdentity,
    load_m6_release_config,
    run_m6_general_pass,
    sha256_file,
)
from tinyllm.schemas import canonical_config_hash


def _candidate_model() -> M6ModelIdentity:
    return M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=("b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c"),
        model_parameters=596_049_920,
        training_run_id="20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
        training_checkpoint_id="checkpoint-tokens-0010000532",
        training_tokens=10_000_532,
        training_config_sha256="d" * 64,
        dataset_version="m5-dual-sft-v1-b5b9e839",
        dataset_manifest_sha256="e" * 64,
    )


def _general_summary() -> GeneralBaselineSummary:
    samples = (2376, 10042, 1838)
    tasks = tuple(
        GeneralTaskResult.model_validate(
            {
                "task": task,
                "samples": count,
                "acc": 0.5,
                "acc_stderr": 0.01,
                "acc_norm": 0.6,
                "acc_norm_stderr": 0.01,
            }
        )
        for task, count in zip(
            ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa"),
            samples,
            strict=True,
        )
    )
    return GeneralBaselineSummary(
        harness_version="0.4.12",
        model_parameters=596_049_920,
        tasks=tasks,  # type: ignore[arg-type]
        evaluation_seconds=10.0,
    )


def _candidate_import() -> M6CandidateImportResult:
    return M6CandidateImportResult(
        status="succeeded",
        protocol_version="m6-release-v1",
        config_sha256=canonical_config_hash(
            load_m6_release_config(Path("configs/eval/m6_release.yaml"))
        ),
        source_run_id="20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
        source_result_sha256="a" * 64,
        source_git_commit="b" * 40,
        source_environment_sha256="c" * 64,
        source_hardware_sha256="d" * 64,
        checkpoint_manifest_sha256="e" * 64,
        snapshot_sha256="f" * 64,
        model=_candidate_model(),
    )


def test_m6_general_pass_writes_complete_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = (tmp_path / "general").resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    model_dir = (tmp_path / "model").resolve()
    model_dir.mkdir()
    tokenizer_dir = (tmp_path / "tokenizer").resolve()
    tokenizer_dir.mkdir()

    def run_general(*_args: object, **kwargs: object) -> GeneralBaselineSummary:
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        assert kwargs["tokenizer_path"] == tokenizer_dir
        raw = output_path / "raw/model"
        raw.mkdir(parents=True)
        (raw / "results.json").write_text("{}\n", encoding="utf-8")
        return _general_summary()

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(total_memory=24, major=8, minor=6),
        get_device_name=lambda _: "NVIDIA GeForce RTX 3090",
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setattr("tinyllm.evaluation.m6_general.torch.cuda", cuda)
    monkeypatch.setattr(m6_general_module, "read_git_identity", lambda _: ("a" * 40, False))
    monkeypatch.setattr(
        m6_general_module,
        "model_export_sha256",
        lambda _: _candidate_model().model_artifact_sha256,
    )
    monkeypatch.setattr(m6_general_module, "run_general_evaluation", run_general)
    monkeypatch.setattr(
        m6_general_module,
        "_environment_payload",
        lambda: {"schema_version": "1.0", "environment": "test"},
    )
    monkeypatch.setattr(
        m6_general_module,
        "_hardware_payload",
        lambda _: {"schema_version": "1.0", "hardware": "test"},
    )

    result = run_m6_general_pass(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        artifact_root=artifact_root,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        output_dir=output,
        project_root=Path.cwd(),
        physical_gpu_index=3,
        model_identity=_candidate_model(),
        expected_config_sha256=_candidate_import().config_sha256,
    )

    assert result.general.aggregate_basis_points == 6000
    assert result.physical_gpu_index == 3
    assert result.environment_sha256 == sha256_file(output / "environment.json")
    assert (
        M6GeneralPassSummary.model_validate_json((output / "summary.json").read_bytes()) == result
    )


def test_m6_general_runtime_snapshots_are_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm_eval = ModuleType("lm_eval")
    lm_eval.__dict__["__version__"] = "0.4.12"
    transformers = ModuleType("transformers")
    transformers.__dict__["__version__"] = "4.57.6"
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    cuda = SimpleNamespace(
        get_device_properties=lambda _: SimpleNamespace(total_memory=24, major=8, minor=6),
        get_device_name=lambda _: "NVIDIA GeForce RTX 3090",
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setattr("tinyllm.evaluation.m6_general.torch.cuda", cuda)

    environment = m6_general_module._environment_payload()
    hardware = m6_general_module._hardware_payload(7)

    assert environment["lm_eval"] == "0.4.12"
    assert hardware["physical_gpu_index"] == 7
    assert hardware["gpu_name"] == "NVIDIA GeForce RTX 3090"


def test_m6_general_cli_runs_preflight_and_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imported = _candidate_import()
    result = M6GeneralPassSummary(
        status="succeeded",
        evaluation_id="candidate-general",
        protocol_version="m6-release-v1",
        config_sha256=imported.config_sha256,
        git_commit="a" * 40,
        git_dirty=False,
        model=imported.model,
        general=m6_general_module._general_result(_general_summary()),
        physical_gpu_index=3,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=10.0,
        environment_sha256="a" * 64,
        hardware_sha256="b" * 64,
        raw_results_sha256="c" * 64,
    )
    monkeypatch.setattr(cli_module, "load_m6_candidate_import", lambda _: imported)
    monkeypatch.setattr(cli_module, "preflight_baseline_gpu", lambda _: None)
    monkeypatch.setattr(cli_module, "run_m6_general_pass", lambda **_: result)
    paths = [
        str((tmp_path / name).resolve())
        for name in ("candidate.json", "model", "tokenizer", "artifacts", "general")
    ]

    assert (
        main(
            [
                "--json",
                "eval",
                "m6-candidate-general",
                "--candidate-import",
                paths[0],
                "--model-dir",
                paths[1],
                "--tokenizer-dir",
                paths[2],
                "--artifact-root",
                paths[3],
                "--output-dir",
                paths[4],
                "--gpu-index",
                "3",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation_id"] == "candidate-general"


def test_m6_general_pass_rejects_preflight_and_lineage_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = (tmp_path / "general").resolve()
    kwargs = {
        "release_config_path": Path("configs/eval/m6_release.yaml"),
        "artifact_root": (tmp_path / "artifacts").resolve(),
        "model_dir": (tmp_path / "model").resolve(),
        "tokenizer_dir": (tmp_path / "tokenizer").resolve(),
        "output_dir": output,
        "project_root": Path.cwd(),
        "physical_gpu_index": 3,
        "model_identity": _candidate_model(),
        "expected_config_sha256": _candidate_import().config_sha256,
    }
    cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setattr("tinyllm.evaluation.m6_general.torch.cuda", cuda)
    with pytest.raises(M6GeneralError, match="exactly one visible"):
        run_m6_general_pass(**kwargs)  # type: ignore[arg-type]

    cuda.is_available = lambda: True
    cuda.device_count = lambda: 1
    monkeypatch.setattr(m6_general_module, "read_git_identity", lambda _: ("a" * 40, True))
    with pytest.raises(M6GeneralError, match="clean Git"):
        run_m6_general_pass(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(m6_general_module, "read_git_identity", lambda _: ("a" * 40, False))
    output.mkdir()
    with pytest.raises(M6GeneralError, match="output absent"):
        run_m6_general_pass(**kwargs)  # type: ignore[arg-type]
    output.rmdir()

    with pytest.raises(M6GeneralError, match="config identities"):
        run_m6_general_pass(
            **(kwargs | {"expected_config_sha256": "0" * 64})  # type: ignore[arg-type]
        )
    monkeypatch.setattr(m6_general_module, "model_export_sha256", lambda _: "0" * 64)
    with pytest.raises(M6GeneralError, match="imported Candidate"):
        run_m6_general_pass(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (M6CandidateImportError, 2),
        (BaselinePreflightError, 3),
        (M6GeneralError, 6),
    ],
)
def test_m6_general_cli_maps_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[Exception],
    expected_code: int,
) -> None:
    imported = _candidate_import()
    if failure is M6CandidateImportError:
        monkeypatch.setattr(
            cli_module,
            "load_m6_candidate_import",
            lambda _: (_ for _ in ()).throw(M6CandidateImportError("invalid import")),
        )
    else:
        monkeypatch.setattr(cli_module, "load_m6_candidate_import", lambda _: imported)
        if failure is BaselinePreflightError:
            monkeypatch.setattr(
                cli_module,
                "preflight_baseline_gpu",
                lambda _: (_ for _ in ()).throw(BaselinePreflightError("busy")),
            )
        else:
            monkeypatch.setattr(cli_module, "preflight_baseline_gpu", lambda _: None)
            monkeypatch.setattr(
                cli_module,
                "run_m6_general_pass",
                lambda **_: (_ for _ in ()).throw(M6GeneralError("failed")),
            )
    paths = [
        str((tmp_path / name).resolve())
        for name in ("candidate.json", "model", "tokenizer", "artifacts", "general")
    ]
    assert (
        main(
            [
                "eval",
                "m6-candidate-general",
                "--candidate-import",
                paths[0],
                "--model-dir",
                paths[1],
                "--tokenizer-dir",
                paths[2],
                "--artifact-root",
                paths[3],
                "--output-dir",
                paths[4],
                "--gpu-index",
                "3",
            ]
        )
        == expected_code
    )
