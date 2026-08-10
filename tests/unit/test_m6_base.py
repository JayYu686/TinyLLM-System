from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import tinyllm.cli as cli_module
import tinyllm.evaluation.m6_base as m6_base_module
from tinyllm.cli import main
from tinyllm.evaluation import (
    BaselineEvaluationResult,
    DomainBaselineSummary,
    GeneralBaselineSummary,
    GeneralTaskResult,
    HumanRubricJudgment,
    M6BaseImportError,
    M6ContractError,
    domain_cluster_id,
    import_m2_base_evidence,
    load_baseline_config,
    load_evaluation_items,
    load_m6_base_import,
    model_artifact_sha256,
    score_domain_response,
    sha256_file,
    sha256_tree,
)
from tinyllm.schemas import RunManifest, RunStatus, canonical_config_hash


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, values: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _synthetic_source_run(tmp_path: Path) -> tuple[Path, Path]:
    config = load_baseline_config(Path("configs/eval/m2_baseline.yaml"))
    config_sha = canonical_config_hash(config)
    run_id = "20260809T000000Z-qwen3-base-a2bae098-abcd"
    run = tmp_path / run_id
    run.mkdir()
    shutil.copyfile("configs/eval/m2_baseline.yaml", run / "config.original.yaml")
    items = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))
    results = tuple(
        score_domain_response(
            item,
            item.reference_answer,
            prompt_tokens=1,
            generated_tokens=1,
            finish_reason="eos",
        )
        for item in items
    )
    judgments = tuple(
        HumanRubricJudgment(
            item_id=item.id,
            criterion_results=(True, True, True),
            passed=True,
            rationale="All three frozen rubric criteria are satisfied.",
            reviewer_role="maintainer",
        )
        for item in items
        if item.scorer.kind == "human_rubric"
    )
    _write_jsonl(
        run / "evaluations/domain/results.jsonl",
        tuple(item.to_dict() for item in results),
    )
    _write_jsonl(
        run / "evaluations/domain/human_review/judgments.jsonl",
        tuple(item.to_dict() for item in judgments),
    )
    raw = run / "evaluations/general/raw/model"
    _write_json(raw / "results.json", {"complete": True})
    general = GeneralBaselineSummary(
        harness_version="0.4.12",
        model_parameters=596_049_920,
        tasks=(
            GeneralTaskResult(
                task="tinyllm_arc_easy",
                samples=2376,
                acc=0.5,
                acc_stderr=0.01,
                acc_norm=0.5,
                acc_norm_stderr=0.01,
            ),
            GeneralTaskResult(
                task="tinyllm_hellaswag",
                samples=10042,
                acc=0.5,
                acc_stderr=0.01,
                acc_norm=0.5,
                acc_norm_stderr=0.01,
            ),
            GeneralTaskResult(
                task="tinyllm_piqa",
                samples=1838,
                acc=0.5,
                acc_stderr=0.01,
                acc_norm=0.5,
                acc_norm_stderr=0.01,
            ),
        ),
        evaluation_seconds=1.0,
    )
    git_commit = "a" * 40
    summary = BaselineEvaluationResult(
        status="succeeded",
        mode="formal",
        run_id=run_id,
        config_sha256=config_sha,
        git_commit=git_commit,
        git_dirty=False,
        model_repository="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        domain=DomainBaselineSummary(
            status="complete",
            suite_version="tinyllm-domain-v1-83bdd8ef",
            evaluated_items=300,
            objective_items=260,
            objective_correct=260,
            human_review_pending=0,
            human_reviewed=40,
            human_passed=40,
            json_items=80,
            json_valid=80,
        ),
        general=general,
    )
    _write_json(run / "evaluations/summary.json", summary.to_dict())
    now = datetime(2026, 8, 9, tzinfo=UTC)
    _write_json(
        run / "run.json",
        RunManifest(
            run_id=run_id,
            name="qwen3-base",
            status=RunStatus.SUCCEEDED,
            created_at=now,
            updated_at=now,
            config_hash=config_sha,
            git_commit=git_commit,
            git_dirty=False,
            artifact_root=tmp_path,
            strategy="evaluation",
            world_size=1,
            dataset_version="tinyllm-domain-v1-83bdd8ef",
            tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        ).to_dict(),
    )
    _write_json(run / "environment.json", {"schema_version": "1.0"})
    _write_json(run / "hardware.json", {"schema_version": "1.0"})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return run, model_dir


def test_m6_cluster_mapping_has_90_pairs_and_120_singletons() -> None:
    items = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))
    clusters = {domain_cluster_id(item) for item in items}

    assert len(clusters) == 210
    assert sum(cluster.startswith("pair:") for cluster in clusters) == 90
    assert sum(cluster.startswith("singleton:") for cluster in clusters) == 120


def test_m6_tree_hash_is_path_independent_and_rejects_symlinks(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_json(first / "nested/result.json", {"value": 1})
    shutil.copytree(first, second)

    assert sha256_tree(first) == sha256_tree(second)
    (second / "unsafe").symlink_to(second / "nested/result.json")
    with pytest.raises(RuntimeError, match="symlinks"):
        sha256_tree(second)

    with pytest.raises(RuntimeError, match="missing or unsafe"):
        sha256_tree(tmp_path / "missing")


def test_m6_model_artifact_hash_validates_every_pinned_file(tmp_path: Path) -> None:
    model_dir = (tmp_path / "model").resolve()
    model_dir.mkdir()
    model_file = model_dir / "config.json"
    model_file.write_bytes(b"pinned model bytes")
    expected = (
        SimpleNamespace(
            filename=model_file.name,
            size_bytes=model_file.stat().st_size,
            sha256=sha256_file(model_file),
        ),
    )

    identity = model_artifact_sha256(model_dir, expected)

    assert len(identity) == 64
    model_file.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="differs from pinned input"):
        model_artifact_sha256(model_dir, expected)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        model_artifact_sha256(tmp_path / "missing", expected)


def test_m6_import_reuses_only_verified_compatible_base_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run, model_dir = _synthetic_source_run(tmp_path)
    monkeypatch.setattr(m6_base_module, "model_artifact_sha256", lambda *_args: "b" * 64)
    output = tmp_path / "base-import.json"

    result = import_m2_base_evidence(
        release_config_path=Path("configs/eval/m6_release.yaml"),
        source_run=source_run,
        model_dir=model_dir,
        project_root=Path("."),
        output_path=output,
    )

    assert result.status == "succeeded"
    assert result.nonthinking.correct_items == 300
    assert result.nonthinking.json_valid_items == 80
    assert result.nonthinking.visible_reasoning_leakage_items == 0
    assert result.general.aggregate_basis_points == 5000
    assert output.is_file()
    assert load_m6_base_import(output) == result

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="result is invalid"):
        load_m6_base_import(output)

    (source_run / "evaluations/general/raw/model/results.json").unlink()
    with pytest.raises(RuntimeError, match="contains no files"):
        import_m2_base_evidence(
            release_config_path=Path("configs/eval/m6_release.yaml"),
            source_run=source_run,
            model_dir=model_dir,
            project_root=Path("."),
        )


@pytest.mark.parametrize(
    "release_config",
    ("configs/eval/m6_release_v2.yaml", "configs/eval/m6_release_v3.yaml"),
)
def test_m6_holdout_import_reuses_general_but_requires_fresh_domain_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_config: str,
) -> None:
    source_run, model_dir = _synthetic_source_run(tmp_path)
    monkeypatch.setattr(m6_base_module, "model_artifact_sha256", lambda *_args: "b" * 64)

    result = import_m2_base_evidence(
        release_config_path=Path(release_config),
        source_run=source_run,
        model_dir=model_dir,
        project_root=Path("."),
    )

    assert result.source_domain_results_sha256 is None
    assert result.source_human_review_sha256 is None
    assert result.nonthinking is None
    assert result.general.aggregate_basis_points == 5000


def test_m6_base_import_cli_rejects_relative_artifact_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "eval",
                "m6-import-base",
                "--source-run",
                "relative/run",
                "--model-dir",
                "relative/model",
                "--output",
                "relative/result.json",
                "--json",
            ]
        )
        == 2
    )
    assert "must be absolute" in json.loads(capsys.readouterr().err)["error"]["message"]


def test_m6_base_import_cli_emits_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = type(
        "Result",
        (),
        {
            "source_run_id": "source-run",
            "nonthinking": type("Mode", (), {"correct_items": 16})(),
            "model_dump_json": lambda self, *, indent: json.dumps(
                {"status": "succeeded", "indent": indent}
            ),
        },
    )()
    monkeypatch.setattr(cli_module, "import_m2_base_evidence", lambda **_kwargs: result)

    code = main(
        [
            "eval",
            "m6-import-base",
            "--source-run",
            str((tmp_path / "run").resolve()),
            "--model-dir",
            str((tmp_path / "model").resolve()),
            "--output",
            str((tmp_path / "result.json").resolve()),
            "--json",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "succeeded", "indent": 2}


@pytest.mark.parametrize(
    ("error", "exit_code", "error_code"),
    (
        (M6ContractError("bad release"), 2, "M6_CONFIG_ERROR"),
        (M6BaseImportError("bad source"), 3, "M6_BASE_IMPORT_FAILED"),
    ),
)
def test_m6_base_import_cli_maps_contract_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: RuntimeError,
    exit_code: int,
    error_code: str,
) -> None:
    def fail(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli_module, "import_m2_base_evidence", fail)
    code = main(
        [
            "eval",
            "m6-import-base",
            "--source-run",
            str((tmp_path / "run").resolve()),
            "--model-dir",
            str((tmp_path / "model").resolve()),
            "--output",
            str((tmp_path / "result.json").resolve()),
            "--json",
        ]
    )

    assert code == exit_code
    assert json.loads(capsys.readouterr().err)["error"]["code"] == error_code
