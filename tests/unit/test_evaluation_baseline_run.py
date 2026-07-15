from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import tinyllm.evaluation.baseline_run as baseline_run_module
from tinyllm.evaluation import (
    BaselineContractError,
    BaselineEvaluationResult,
    BaselinePreflightError,
    BaselineRunConfig,
    BaselineRuntimeError,
    DomainBaselineSummary,
    EvaluationItem,
    GeneralBaselineSummary,
    GeneralTaskResult,
    HumanReviewCommit,
    build_domain_summary,
    complete_baseline_human_review,
    load_baseline_config,
    load_evaluation_items,
    load_general_summary,
    run_baseline_evaluation,
    score_domain_response,
)
from tinyllm.evaluation.baseline_run import (
    _run_general_baseline,
    _runtime_environment,
    _runtime_hardware,
)
from tinyllm.schemas import RunManifest, RunStatus, canonical_config_hash, generate_run_id

FORMAL_CONFIG = Path("configs/eval/m2_baseline.yaml")
SMOKE_CONFIG = Path("configs/eval/m2_baseline_smoke.yaml")


def _general_payload(*, samples: int = 2) -> dict[str, object]:
    task_results = {
        task: {
            "name": task,
            "alias": task,
            "sample_len": samples,
            "acc,none": 0.5,
            "acc_stderr,none": 0.5,
            "acc_norm,none": 0.5,
            "acc_norm_stderr,none": 0.5,
        }
        for task in ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa")
    }
    config = load_baseline_config(SMOKE_CONFIG)
    task_configs = {
        task.task: {
            "dataset_path": task.dataset,
            "dataset_kwargs": {"revision": task.dataset_revision},
            "unsafe_code": False,
            "num_fewshot": 0,
            "output_type": "multiple_choice",
        }
        for task in config.general.tasks
    }
    return {
        "results": task_results,
        "configs": task_configs,
        "config": {
            "model_num_parameters": 596_049_920,
            "model_dtype": "torch.bfloat16",
            "limit": 2.0,
            "random_seed": 42,
            "numpy_seed": 42,
            "torch_seed": 42,
            "fewshot_seed": 42,
        },
        "lm_eval_version": "0.4.12",
        "transformers_version": "4.57.6",
        "max_length": 1024,
        "chat_template_sha": config.general.tokenizer_chat_template_sha256,
        "model_source": "hf",
        "fewshot_as_multiturn": True,
        "total_evaluation_time_seconds": "2.5",
    }


def _write_general_payload(root: Path, payload: dict[str, object]) -> None:
    result_dir = root / "private-model-name"
    result_dir.mkdir(parents=True)
    (result_dir / "results_timestamp.json").write_text(json.dumps(payload), encoding="utf-8")


def _general_summary() -> GeneralBaselineSummary:
    tasks = (
        GeneralTaskResult(
            task="tinyllm_arc_easy",
            samples=2,
            acc=0.5,
            acc_stderr=0.5,
            acc_norm=0.5,
            acc_norm_stderr=0.5,
        ),
        GeneralTaskResult(
            task="tinyllm_hellaswag",
            samples=2,
            acc=0.5,
            acc_stderr=0.5,
            acc_norm=0.5,
            acc_norm_stderr=0.5,
        ),
        GeneralTaskResult(
            task="tinyllm_piqa",
            samples=2,
            acc=0.5,
            acc_stderr=0.5,
            acc_norm=0.5,
            acc_norm_stderr=0.5,
        ),
    )
    return GeneralBaselineSummary(
        harness_version="0.4.12",
        model_parameters=596_049_920,
        tasks=tasks,
        evaluation_seconds=2.5,
    )


def test_general_parser_accepts_complete_output_and_rejects_partial_counts(
    tmp_path: Path,
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    output = tmp_path / "valid"
    _write_general_payload(output, _general_payload())

    summary = load_general_summary(config, output_path=output)

    assert summary.model_parameters == 596_049_920
    assert tuple(task.task for task in summary.tasks) == (
        "tinyllm_arc_easy",
        "tinyllm_hellaswag",
        "tinyllm_piqa",
    )

    partial = tmp_path / "partial"
    _write_general_payload(partial, _general_payload(samples=1))
    with pytest.raises(BaselineRuntimeError, match="sample count mismatch"):
        load_general_summary(config, output_path=partial)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("lm_eval_version",), "0.0", "result version"),
        (("transformers_version",), "0.0", "Transformers result version"),
        (("max_length",), 2048, "maximum length"),
        (("chat_template_sha",), "bad", "Chat Template hash"),
        (("model_source",), "local", "model source"),
        (("config", "model_dtype"), "torch.float16", "dtype"),
        (("config", "limit"), None, "limit"),
        (("config", "random_seed"), 7, "seeds"),
        (("config", "model_num_parameters"), 0, "positive"),
        (
            ("configs", "tinyllm_arc_easy", "dataset_path"),
            "floating/dataset",
            "task config mismatch",
        ),
        (("results", "tinyllm_arc_easy", "acc,none"), 2.0, "outside"),
        (("total_evaluation_time_seconds",), "not-a-number", "must be numeric"),
    ],
)
def test_general_parser_rejects_drifted_metadata_and_metrics(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _general_payload()
    target = cast(dict[str, Any], payload)
    for component in path[:-1]:
        target = cast(dict[str, Any], target[component])
    target[path[-1]] = value
    output = tmp_path / "drifted"
    _write_general_payload(output, payload)

    with pytest.raises(BaselineRuntimeError, match=message):
        load_general_summary(load_baseline_config(SMOKE_CONFIG), output_path=output)


def test_domain_summary_rejects_missing_or_duplicate_results() -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    items = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))[:2]
    result = score_domain_response(
        items[0],
        items[0].reference_answer,
        prompt_tokens=1,
        generated_tokens=1,
        finish_reason="eos",
    )
    with pytest.raises(BaselineRuntimeError, match="count or item identity"):
        build_domain_summary(config, (result,))
    with pytest.raises(BaselineRuntimeError, match="count or item identity"):
        build_domain_summary(config, (result, result))


def test_general_runner_records_commands_offline_environment_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    calls: list[tuple[Sequence[str], dict[str, str]]] = []
    return_codes = iter((0, 0, 1, 0, 2))

    def run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        calls.append((command, environment))
        code = next(return_codes)
        return subprocess.CompletedProcess(command, code, "stdout", "stderr")

    monkeypatch.setattr("tinyllm.evaluation.baseline_run.subprocess.run", run)
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run.load_general_summary",
        lambda *_args, **_kwargs: _general_summary(),
    )
    artifact_root = (tmp_path / "artifacts").resolve()
    model_path = (tmp_path / "model").resolve()
    model_path.mkdir()

    summary = _run_general_baseline(
        config,
        project_root=Path.cwd().resolve(),
        artifact_root=artifact_root,
        model_path=model_path,
        output_path=tmp_path / "successful",
        device="cuda",
        offline=True,
    )
    assert summary == _general_summary()
    assert calls[0][1]["HF_DATASETS_OFFLINE"] == "1"
    assert calls[0][1]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][1]["TRANSFORMERS_OFFLINE"] == "1"
    assert (tmp_path / "successful/validation.command.json").is_file()
    assert "returncode=0" in (tmp_path / "successful/run.log").read_text()

    with pytest.raises(BaselineRuntimeError, match="validation failed"):
        _run_general_baseline(
            config,
            project_root=Path.cwd().resolve(),
            artifact_root=artifact_root,
            model_path=model_path,
            output_path=tmp_path / "validation-failed",
            device="cpu",
            offline=False,
        )

    with pytest.raises(BaselineRuntimeError, match="execution failed with code 2"):
        _run_general_baseline(
            config,
            project_root=Path.cwd().resolve(),
            artifact_root=artifact_root,
            model_path=model_path,
            output_path=tmp_path / "execution-failed",
            device="cpu",
            offline=False,
        )


def test_runtime_lineage_collectors_capture_packages_and_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)

    class FakeCuda:
        @staticmethod
        def get_device_properties(index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(
                name="NVIDIA GeForce RTX 3090",
                total_memory=24 * 1024**3,
                major=8,
                minor=6,
            )

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    runtime: Any = SimpleNamespace(
        torch=SimpleNamespace(
            __version__="2.7.1+cu118",
            version=SimpleNamespace(cuda="11.8"),
            cuda=FakeCuda(),
        )
    )

    def run(command: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if tuple(command[:3]) == (sys.executable, "-m", "pip"):
            return subprocess.CompletedProcess(command, 0, "z-package==1\na-package==2\n", "")
        return subprocess.CompletedProcess(command, 0, "5, 535.261.03, 4, 0, 29\n", "")

    monkeypatch.setattr("tinyllm.evaluation.baseline_run.subprocess.run", run)
    environment = _runtime_environment(
        config,
        runtime=runtime,
        device="cuda",
        git_branch="agent/test",
    )
    assert environment["pip_freeze"] == ["a-package==2", "z-package==1"]
    assert environment["cuda_runtime"] == "11.8"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    hardware = _runtime_hardware(runtime, device="cuda")
    assert hardware["physical_device"] == 5
    assert hardware["logical_device"] == 0
    assert hardware["driver_version"] == "535.261.03"
    assert hardware["bf16_supported"] is True

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert _runtime_hardware(runtime, device="cpu")["device"] == "cpu"
    with pytest.raises(BaselinePreflightError, match="numeric physical GPU"):
        _runtime_hardware(runtime, device="cuda")


def _patch_smoke_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    runtime = SimpleNamespace(torch=SimpleNamespace())
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run.acquire_baseline_model",
        lambda *_args, **_kwargs: model_path,
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run.load_baseline_runtime",
        lambda _config: runtime,
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run._runtime_environment",
        lambda *_args, **_kwargs: {"schema_version": "1.0", "device": "cpu"},
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run._runtime_hardware",
        lambda *_args, **_kwargs: {"schema_version": "1.0", "device": "cpu"},
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run.read_git_identity",
        lambda _root: ("a" * 40, True),
    )
    monkeypatch.setattr("tinyllm.evaluation.baseline_run._git_branch", lambda _root: "agent/test")
    monkeypatch.setattr("tinyllm.evaluation.baseline_run.seed_baseline", lambda *_args: None)

    class FakeBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr("tinyllm.evaluation.baseline_run.TransformersDomainBackend", FakeBackend)

    def domain_results(
        config: BaselineRunConfig,
        items: Sequence[EvaluationItem],
        *,
        backend: object,
    ) -> tuple[Any, ...]:
        del config, backend
        return tuple(
            score_domain_response(
                item,
                item.reference_answer,
                prompt_tokens=10,
                generated_tokens=10,
                finish_reason="eos",
            )
            for item in items
        )

    monkeypatch.setattr("tinyllm.evaluation.baseline_run.run_domain_generation", domain_results)


def test_smoke_runner_writes_private_raw_outputs_and_path_free_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_smoke_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run._run_general_baseline",
        lambda *_args, **_kwargs: _general_summary(),
    )
    artifact_root = (tmp_path / "artifacts").resolve()

    result = run_baseline_evaluation(
        config_path=SMOKE_CONFIG,
        project_root=Path.cwd().resolve(),
        artifact_root=artifact_root,
        device="cpu",
        offline=True,
    )

    run_directory = artifact_root / "runs" / result.run_id
    assert result.status == "succeeded"
    assert result.git_dirty is True
    assert str(artifact_root) not in result.model_dump_json()
    assert (run_directory / "config.original.yaml").is_file()
    assert (run_directory / "environment.json").is_file()
    assert (run_directory / "hardware.json").is_file()
    assert (run_directory / "evaluations/summary.json").is_file()
    raw_lines = (run_directory / "evaluations/domain/results.jsonl").read_text().splitlines()
    assert len(raw_lines) == 2
    assert json.loads(raw_lines[0])["response"]
    run_manifest = json.loads((run_directory / "run.json").read_text())
    assert run_manifest["strategy"] == "evaluation"
    assert run_manifest["status"] == "succeeded"


def test_smoke_runner_retains_failed_run_when_general_eval_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_smoke_runtime(monkeypatch, tmp_path)

    def fail(*_args: object, **_kwargs: object) -> GeneralBaselineSummary:
        raise BaselineRuntimeError("general evaluation failed")

    monkeypatch.setattr("tinyllm.evaluation.baseline_run._run_general_baseline", fail)
    artifact_root = (tmp_path / "artifacts").resolve()

    with pytest.raises(BaselineRuntimeError, match="general evaluation failed"):
        run_baseline_evaluation(
            config_path=SMOKE_CONFIG,
            project_root=Path.cwd().resolve(),
            artifact_root=artifact_root,
            device="cpu",
            offline=True,
        )

    runs = tuple((artifact_root / "runs").iterdir())
    assert len(runs) == 1
    assert json.loads((runs[0] / "run.json").read_text())["status"] == "failed"
    assert "baseline_failed" in (runs[0] / "events.jsonl").read_text()


def test_formal_runner_requires_clean_main_before_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run.read_git_identity",
        lambda _root: ("a" * 40, False),
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_run._git_branch", lambda _root: "agent/not-main"
    )
    model_accessed = False

    def acquire(*_args: object, **_kwargs: object) -> Path:
        nonlocal model_accessed
        model_accessed = True
        return tmp_path

    monkeypatch.setattr("tinyllm.evaluation.baseline_run.acquire_baseline_model", acquire)

    with pytest.raises(BaselineContractError, match="clean main"):
        run_baseline_evaluation(
            config_path=FORMAL_CONFIG,
            project_root=Path.cwd().resolve(),
            artifact_root=(tmp_path / "artifacts").resolve(),
            device="cuda",
            offline=True,
        )
    assert model_accessed is False


def test_domain_smoke_items_used_by_runner_are_frozen_first_two() -> None:
    items = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))[:2]
    assert tuple(item.id for item in items) == ("domain-config-001", "domain-config-002")


def test_human_review_commit_recovers_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_config = load_baseline_config(SMOKE_CONFIG).model_dump()
    raw_config["domain"]["limit"] = 1
    config = BaselineRunConfig.model_validate(raw_config)
    config_hash = canonical_config_hash(config)
    now = datetime(2026, 7, 15, tzinfo=UTC)
    run_id = generate_run_id(config.run_slug, config_hash, now=now, nonce="cafe")
    artifact_root = (tmp_path / "artifacts").resolve()
    run_directory = artifact_root / "runs" / run_id
    (run_directory / "evaluations/domain").mkdir(parents=True)
    human_item = next(
        item
        for item in load_evaluation_items(Path("evals/domain/v1/items.jsonl"))
        if item.scorer.kind == "human_rubric"
    )
    domain_result = score_domain_response(
        human_item,
        "The supplied evidence is insufficient; provide the complete error log and timestamp.",
        prompt_tokens=10,
        generated_tokens=15,
        finish_reason="eos",
    )
    domain_summary = DomainBaselineSummary(
        status="awaiting_human_review",
        suite_version="tinyllm-domain-v1-83bdd8ef",
        evaluated_items=1,
        objective_items=0,
        objective_correct=0,
        human_review_pending=1,
        human_reviewed=0,
        human_passed=0,
        json_items=0,
        json_valid=0,
    )
    result = BaselineEvaluationResult(
        status="awaiting_human_review",
        mode="smoke",
        run_id=run_id,
        config_sha256=config_hash,
        git_commit="a" * 40,
        git_dirty=True,
        model_repository="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        domain=domain_summary,
        general=_general_summary(),
    )
    manifest = RunManifest(
        run_id=run_id,
        name=config.run_slug,
        status=RunStatus.EVALUATING,
        created_at=now,
        updated_at=now,
        config_hash=config_hash,
        git_commit="a" * 40,
        git_dirty=True,
        artifact_root=artifact_root,
        strategy="evaluation",
        world_size=1,
        dataset_version=config.domain.suite_version,
        tokenizer_revision=config.model.revision,
    )
    (run_directory / "run.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (run_directory / "config.resolved.json").write_text(config.model_dump_json(), encoding="utf-8")
    (run_directory / "evaluations/summary.json").write_text(
        result.model_dump_json(), encoding="utf-8"
    )
    (run_directory / "evaluations/domain/results.jsonl").write_text(
        domain_result.model_dump_json() + "\n", encoding="utf-8"
    )
    (run_directory / "events.jsonl").write_text("", encoding="utf-8")
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "item_id": human_item.id,
                "criterion_results": [True, True, True],
                "passed": True,
                "rationale": "The response satisfies all three frozen criteria.",
                "reviewer_role": "maintainer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    real_atomic_json = baseline_run_module._atomic_json
    interrupted = False

    def interrupt_after_review_publish(path: Path, value: object) -> None:
        nonlocal interrupted
        if path == run_directory / "evaluations/domain/summary.json" and not interrupted:
            interrupted = True
            raise OSError("simulated outer-summary interruption")
        real_atomic_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(baseline_run_module, "_atomic_json", interrupt_after_review_publish)
        with pytest.raises(OSError, match="outer-summary interruption"):
            complete_baseline_human_review(
                run_id=run_id,
                artifact_root=artifact_root,
                judgments_path=judgments,
            )
    assert (run_directory / "evaluations/domain/human_review/commit.json").is_file()
    assert json.loads((run_directory / "run.json").read_text())["status"] == "evaluating"

    completed = complete_baseline_human_review(
        run_id=run_id,
        artifact_root=artifact_root,
        judgments_path=judgments,
    )

    assert completed.status == "succeeded"
    assert completed.domain.human_review_pending == 0
    assert completed.domain.human_reviewed == completed.domain.human_passed == 1
    commit = HumanReviewCommit.model_validate_json(
        (run_directory / "evaluations/domain/human_review/commit.json").read_text()
    )
    assert commit.run_id == run_id
    assert commit.judgment_count == 1
    assert json.loads((run_directory / "run.json").read_text())["status"] == "succeeded"
    assert "baseline_human_review_completed" in (run_directory / "events.jsonl").read_text()

    repeated = complete_baseline_human_review(
        run_id=run_id,
        artifact_root=artifact_root,
        judgments_path=judgments,
    )
    assert repeated == completed
    events = [
        json.loads(line) for line in (run_directory / "events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "baseline_human_review_completed" for event in events) == 1

    changed_judgment = json.loads(judgments.read_text())
    changed_judgment["rationale"] = "A different reviewer rationale must not replace the commit."
    judgments.write_text(json.dumps(changed_judgment) + "\n", encoding="utf-8")
    with pytest.raises(BaselineContractError, match="does not match this request"):
        complete_baseline_human_review(
            run_id=run_id,
            artifact_root=artifact_root,
            judgments_path=judgments,
        )
