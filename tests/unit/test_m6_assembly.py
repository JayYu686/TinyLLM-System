from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import tinyllm.cli as cli_module
from tinyllm.cli import main
from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_assembly import (
    M6AssemblyError,
    _load_domain_component,
    _load_general_component,
    assemble_m6_base_evaluation,
    assemble_m6_candidate_evaluation,
)
from tinyllm.evaluation.m6_schema import (
    M6BaseImportResult,
    M6CandidateImportResult,
    M6DomainItemScore,
    M6DomainModeResult,
    M6DomainPassSummary,
    M6GeneralPassSummary,
    M6GeneralResult,
    M6GeneralTaskResult,
    M6ModelIdentity,
)
from tinyllm.schemas import canonical_config_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(role: str) -> M6ModelIdentity:
    values: dict[str, Any] = {
        "role": role,
        "repository": "Qwen/Qwen3-0.6B",
        "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "model_artifact_sha256": (
            "a" * 64
            if role == "base"
            else "b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c"
        ),
        "model_parameters": 596049920,
        "adaptation": "base" if role == "base" else "full_sft",
    }
    if role == "candidate":
        values.update(
            {
                "training_run_id": ("20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15"),
                "training_checkpoint_id": "checkpoint-tokens-0010000532",
                "training_tokens": 10000532,
                "training_config_sha256": "b" * 64,
                "dataset_version": "m5-dual-sft-v1-b5b9e839",
                "dataset_manifest_sha256": "c" * 64,
            }
        )
    return M6ModelIdentity.model_validate(values)


def _mode(mode: str) -> M6DomainModeResult:
    decoded = [
        json.loads(line)
        for line in Path("evals/domain/v1/items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scores: list[M6DomainItemScore] = []
    for item in decoded:
        pair = next(
            (tag for tag in item["tags"] if tag.startswith("bilingual-pair-")),
            None,
        )
        cluster = (
            f"pair:{item['category']}:{pair.rsplit('-', 1)[1]}"
            if pair is not None
            else f"singleton:{item['id']}"
        )
        kind = item["scorer"]["kind"]
        scores.append(
            M6DomainItemScore(
                item_id=item["id"],
                cluster_id=cluster,
                language=item["language"],
                category=item["category"],
                scorer_kind=kind,
                correct=False,
                json_valid=False if kind == "json_object" else None,
                format_valid=True,
                visible_reasoning_leakage=False,
            )
        )
    return M6DomainModeResult(
        mode=cast(Any, mode),
        items=tuple(scores),
        evaluated_items=300,
        correct_items=0,
        score_basis_points=0,
        format_valid_items=300,
        format_valid_basis_points=10000,
        json_items=80,
        json_valid_items=0,
        json_valid_basis_points=0,
        visible_reasoning_leakage_items=0,
        visible_reasoning_leakage_basis_points=0,
        natural_thinking_closed_items=300 if mode == "thinking" else 0,
        budget_forced_close_items=0,
        forced_close_basis_points=0,
        generated_tokens=300,
        injected_tokens=0,
    )


def _general() -> M6GeneralResult:
    task_data = (
        ("tinyllm_arc_easy", 2376, 0.45),
        ("tinyllm_hellaswag", 10042, 0.40),
        ("tinyllm_piqa", 1838, 0.65),
    )
    tasks = tuple(
        M6GeneralTaskResult(
            task=cast(Any, task),
            samples=samples,
            acc=value,
            acc_stderr=0.01,
            acc_norm=value,
            acc_norm_stderr=0.01,
        )
        for task, samples, value in task_data
    )
    return M6GeneralResult(
        harness_version="0.4.12",
        metric="acc_norm",
        aggregation="equal-task-mean",
        tasks=cast(Any, tasks),
        aggregate_basis_points=5000,
    )


def _domain_component(
    root: Path,
    *,
    mode: str,
    model: M6ModelIdentity,
    config_sha256: str,
) -> tuple[Path, Path, M6DomainPassSummary]:
    directory = root / mode
    directory.mkdir()
    raw = directory / "results.jsonl"
    environment = directory / "environment.json"
    hardware = directory / "hardware.json"
    judgments = root / f"{mode}-judgments.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    environment.write_text("{}\n", encoding="utf-8")
    hardware.write_text("{}\n", encoding="utf-8")
    judgments.write_text("{}\n", encoding="utf-8")
    summary = M6DomainPassSummary(
        status="succeeded",
        evaluation_id=f"m6-{mode}",
        protocol_version="m6-release-v1",
        suite_version="tinyllm-domain-v1-83bdd8ef",
        config_sha256=config_sha256,
        git_commit="d" * 40,
        git_dirty=False,
        model=model,
        mode=cast(Any, mode),
        evaluated_items=300,
        objective_items=260,
        objective_correct_items=0,
        human_review_pending=0,
        human_reviewed=40,
        human_passed=0,
        json_items=80,
        json_valid_items=0,
        format_valid_items=300,
        visible_reasoning_leakage_items=0,
        natural_thinking_closed_items=300 if mode == "thinking" else 0,
        budget_forced_close_items=0,
        generated_tokens=300,
        injected_tokens=0,
        duration_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        physical_gpu_index=0,
        gpu_name="NVIDIA GeForce RTX 3090",
        environment_sha256=_sha(environment),
        hardware_sha256=_sha(hardware),
        raw_results_sha256=_sha(raw),
        human_review_sha256=_sha(judgments),
    )
    (directory / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    (directory / "mode_result.json").write_text(_mode(mode).model_dump_json(), encoding="utf-8")
    return directory.resolve(), judgments.resolve(), summary


def _general_component(
    root: Path,
    *,
    model: M6ModelIdentity,
    config_sha256: str,
) -> Path:
    directory = root / "general"
    raw = directory / "raw"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text("{}\n", encoding="utf-8")
    environment = directory / "environment.json"
    hardware = directory / "hardware.json"
    environment.write_text("{}\n", encoding="utf-8")
    hardware.write_text("{}\n", encoding="utf-8")
    from tinyllm.evaluation.m6_base import sha256_tree

    summary = M6GeneralPassSummary(
        status="succeeded",
        evaluation_id="m6-general",
        protocol_version="m6-release-v1",
        config_sha256=config_sha256,
        git_commit="e" * 40,
        git_dirty=False,
        model=model,
        general=_general(),
        physical_gpu_index=0,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=1.0,
        environment_sha256=_sha(environment),
        hardware_sha256=_sha(hardware),
        raw_results_sha256=sha256_tree(raw),
    )
    (directory / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    return directory.resolve()


def test_candidate_assembly_validates_components_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    config_sha256 = canonical_config_hash(release)
    model = _model("candidate")
    imported = M6CandidateImportResult(
        status="succeeded",
        protocol_version="m6-release-v1",
        config_sha256=config_sha256,
        source_run_id="20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15",
        source_result_sha256="1" * 64,
        source_git_commit="2" * 40,
        source_environment_sha256="3" * 64,
        source_hardware_sha256="4" * 64,
        checkpoint_manifest_sha256="5" * 64,
        snapshot_sha256="6" * 64,
        model=model,
    )
    import_path = (tmp_path / "candidate-import.json").resolve()
    import_path.write_text(imported.model_dump_json(), encoding="utf-8")
    thinking, thinking_judgments, _ = _domain_component(
        tmp_path, mode="thinking", model=model, config_sha256=config_sha256
    )
    nonthinking, nonthinking_judgments, _ = _domain_component(
        tmp_path, mode="nonthinking", model=model, config_sha256=config_sha256
    )
    general = _general_component(tmp_path, model=model, config_sha256=config_sha256)
    monkeypatch.setattr(
        "tinyllm.evaluation.m6_assembly.read_git_identity", lambda _path: ("f" * 40, False)
    )
    output = (tmp_path / "evaluation.json").resolve()

    result = assemble_m6_candidate_evaluation(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        candidate_import_path=import_path,
        thinking_pass_directory=thinking,
        thinking_judgments_path=thinking_judgments,
        nonthinking_pass_directory=nonthinking,
        nonthinking_judgments_path=nonthinking_judgments,
        general_pass_directory=general,
        output_path=output,
        project_root=Path("."),
    )
    repeated = assemble_m6_candidate_evaluation(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        candidate_import_path=import_path,
        thinking_pass_directory=thinking,
        thinking_judgments_path=thinking_judgments,
        nonthinking_pass_directory=nonthinking,
        nonthinking_judgments_path=nonthinking_judgments,
        general_pass_directory=general,
        output_path=output,
        project_root=Path("."),
    )

    assert result == repeated
    assert result.model.role == "candidate"
    assert result.human_review_complete is True
    assert result.lineage_complete is True
    assert json.loads(output.read_text())["evaluation_id"] == result.evaluation_id

    thinking_judgments.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(M6AssemblyError, match="lineage"):
        assemble_m6_candidate_evaluation(
            release_config_path=Path("configs/eval/m6_release.yaml"),
            candidate_import_path=import_path,
            thinking_pass_directory=thinking,
            thinking_judgments_path=thinking_judgments,
            nonthinking_pass_directory=nonthinking,
            nonthinking_judgments_path=nonthinking_judgments,
            general_pass_directory=general,
            output_path=output,
            project_root=Path("."),
        )


def test_base_assembly_reuses_imported_nonthinking_and_general(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    config_sha256 = canonical_config_hash(release)
    model = _model("base")
    thinking, judgments, _ = _domain_component(
        tmp_path, mode="thinking", model=model, config_sha256=config_sha256
    )
    imported = M6BaseImportResult(
        status="succeeded",
        protocol_version="m6-release-v1",
        config_sha256=config_sha256,
        source_run_id="base-run",
        source_config_sha256="1" * 64,
        source_git_commit="2" * 40,
        source_evaluation_sha256="3" * 64,
        source_domain_results_sha256="4" * 64,
        source_human_review_sha256="5" * 64,
        source_general_tree_sha256="6" * 64,
        source_environment_sha256="7" * 64,
        source_hardware_sha256="8" * 64,
        model=model,
        nonthinking=_mode("nonthinking"),
        general=_general(),
    )
    import_path = (tmp_path / "base-import.json").resolve()
    import_path.write_text(imported.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "tinyllm.evaluation.m6_assembly.read_git_identity", lambda _path: ("9" * 40, False)
    )

    result = assemble_m6_base_evaluation(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        base_import_path=import_path,
        thinking_pass_directory=thinking,
        thinking_judgments_path=judgments,
        output_path=(tmp_path / "base-evaluation.json").resolve(),
        project_root=Path("."),
    )

    assert result.model.role == "base"
    assert result.domain_modes[1] == imported.nonthinking
    assert result.general == imported.general


def test_component_loaders_reject_tampering(tmp_path: Path) -> None:
    release = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    config_sha256 = canonical_config_hash(release)
    model = _model("candidate")
    thinking, judgments, _ = _domain_component(
        tmp_path, mode="thinking", model=model, config_sha256=config_sha256
    )
    general = _general_component(tmp_path, model=model, config_sha256=config_sha256)

    summary, result = _load_domain_component(
        thinking,
        judgments,
        expected_mode="thinking",
        expected_model=model,
        expected_config_sha256=config_sha256,
    )
    assert summary.status == "succeeded"
    assert result.mode == "thinking"
    assert (
        _load_general_component(
            general, expected_model=model, expected_config_sha256=config_sha256
        ).general
        == _general()
    )

    (general / "raw" / "result.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(M6AssemblyError, match="lineage"):
        _load_general_component(
            general,
            expected_model=model,
            expected_config_sha256=config_sha256,
        )


def test_m6_assemble_cli_routes_both_roles_and_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = {
        name: str((tmp_path / name).resolve())
        for name in (
            "import.json",
            "thinking",
            "thinking.jsonl",
            "nonthinking",
            "nonthinking.jsonl",
            "general",
            "evaluation.json",
        )
    }
    calls: list[str] = []

    def result(role: str) -> SimpleNamespace:
        return SimpleNamespace(
            evaluation_id=f"m6-{role}",
            model=SimpleNamespace(role=role),
            model_dump_json=lambda *, indent: json.dumps(
                {"evaluation_id": f"m6-{role}", "indent": indent}
            ),
        )

    def assemble_base(**_kwargs: object) -> SimpleNamespace:
        calls.append("base")
        return result("base")

    def assemble_candidate(**_kwargs: object) -> SimpleNamespace:
        calls.append("candidate")
        return result("candidate")

    monkeypatch.setattr(
        cli_module,
        "assemble_m6_base_evaluation",
        assemble_base,
    )
    monkeypatch.setattr(
        cli_module,
        "assemble_m6_candidate_evaluation",
        assemble_candidate,
    )
    common = [
        "eval",
        "m6-assemble",
        "--evidence-import",
        paths["import.json"],
        "--thinking-pass",
        paths["thinking"],
        "--thinking-judgments",
        paths["thinking.jsonl"],
        "--output",
        paths["evaluation.json"],
        "--json",
    ]

    assert main([*common, "--role", "base"]) == 0
    assert json.loads(capsys.readouterr().out)["evaluation_id"] == "m6-base"
    assert (
        main(
            [
                *common,
                "--role",
                "candidate",
                "--nonthinking-pass",
                paths["nonthinking"],
                "--nonthinking-judgments",
                paths["nonthinking.jsonl"],
                "--general-pass",
                paths["general"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluation_id"] == "m6-candidate"
    assert calls == ["base", "candidate"]


def test_m6_assemble_cli_rejects_invalid_shapes_and_maps_artifact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    absolute = str((tmp_path / "artifact").resolve())
    common = [
        "eval",
        "m6-assemble",
        "--evidence-import",
        absolute,
        "--thinking-pass",
        absolute,
        "--thinking-judgments",
        absolute,
        "--output",
        absolute,
        "--json",
    ]

    assert main([*common, "--role", "unknown"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "CLI_OUTPUT_ERROR"
    assert main([*common, "--role", "candidate"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "M6_ASSEMBLY_FAILED"

    def fail(**_kwargs: object) -> None:
        raise M6AssemblyError("tampered")

    monkeypatch.setattr(cli_module, "assemble_m6_base_evaluation", fail)
    assert main([*common, "--role", "base"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["message"] == "tampered"

    relative = common.copy()
    relative[relative.index(absolute)] = "relative.json"
    assert main([*relative, "--role", "base"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "CLI_OUTPUT_ERROR"
