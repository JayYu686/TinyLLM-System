import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tinyllm.benchmark.supervisor import BenchmarkPreflightError
from tinyllm.cli import main
from tinyllm.data import RegisteredDatasetSummary
from tinyllm.evaluation import (
    BaselineContractError,
    BaselineEvaluationResult,
    BaselineGpuPreflight,
    BaselinePreflightError,
    ContaminationMatch,
    ContaminationReport,
    DomainBaselineSummary,
    EvaluationContractError,
    GeneralBaselineSummary,
    GeneralTaskResult,
)


def test_help_lists_doctor_train_data_eval_and_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "doctor" in output
    assert "train" in output
    assert "data" in output
    assert "eval" in output
    assert "benchmark" in output

    assert main(["data", "--help"]) == 0
    data_output = capsys.readouterr().out
    assert "prepare" in data_output
    assert "inspect" in data_output

    assert main(["eval", "--help"]) == 0
    eval_output = capsys.readouterr().out
    assert "contamination" in eval_output
    assert "baseline" in eval_output
    assert "baseline-review" in eval_output

    assert main(["benchmark", "--help"]) == 0
    benchmark_output = capsys.readouterr().out
    assert "train" in benchmark_output


def test_benchmark_train_uses_preflight_exit_class(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(**_kwargs: object) -> None:
        raise BenchmarkPreflightError("selected GPU is busy")

    monkeypatch.setattr("tinyllm.cli.run_formal_benchmark", reject)
    code = main(
        [
            "benchmark",
            "train",
            "--config",
            "configs/benchmark/m3_tinygpt_120m_ddp.yaml",
            "--output-root",
            str(tmp_path),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--profile",
            "weak",
            "--repeat",
            "1",
            "--gpu-indices",
            "5,6",
            "--json",
        ]
    )

    assert code == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "BENCHMARK_PREFLIGHT_FAILED"


def test_version_is_stable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "tinyllm 0.3.0b1"


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


def contamination_report(*, contaminated: bool) -> ContaminationReport:
    matches = (
        (
            ContaminationMatch(
                evaluation_item_id="domain-python-001",
                match_kind="prompt_prefix",
                fingerprint_sha256="c" * 64,
                training_sample_id_sha256="d" * 64,
            ),
        )
        if contaminated
        else ()
    )
    return ContaminationReport(
        status="contaminated" if contaminated else "clean",
        fingerprint_algorithm="token-sequence-sha256-v1",
        near_dedup="not_evaluated",
        evaluation_suite_version="tinyllm-smoke-v1-aaaaaaaa",
        evaluation_content_sha256="a" * 64,
        dataset_version="m2-sft-v1-bbbbbbbb",
        dataset_content_sha256="b" * 64,
        checked_evaluation_items=1,
        checked_training_samples=10,
        contaminated_items=1 if contaminated else 0,
        full_sequence_matches=0,
        prompt_prefix_matches=1 if contaminated else 0,
        matches=matches,
    )


@pytest.mark.parametrize(("contaminated", "exit_code"), [(False, 0), (True, 6)])
def test_eval_contamination_emits_stable_report_and_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    contaminated: bool,
    exit_code: int,
) -> None:
    report = contamination_report(contaminated=contaminated)
    monkeypatch.setattr("tinyllm.cli.run_contamination_check", lambda **_kwargs: report)

    code = main(
        [
            "eval",
            "contamination",
            "--evaluation-set",
            "eval.jsonl",
            "--config",
            "eval.yaml",
            "--dataset-version",
            "m2-sft-v1-bbbbbbbb",
            "--artifact-root",
            str(tmp_path),
            "--json",
        ]
    )

    assert code == exit_code
    assert json.loads(capsys.readouterr().out) == report.to_dict()


def test_eval_contamination_maps_contract_error_to_usage_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs: object) -> ContaminationReport:
        raise EvaluationContractError("invalid frozen evaluation set")

    monkeypatch.setattr("tinyllm.cli.run_contamination_check", fail)

    code = main(
        [
            "eval",
            "contamination",
            "--evaluation-set",
            "eval.jsonl",
            "--config",
            "eval.yaml",
            "--dataset-version",
            "m2-sft-v1-bbbbbbbb",
            "--artifact-root",
            str(tmp_path),
            "--json",
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "EVALUATION_CONFIG_ERROR"


def baseline_result() -> BaselineEvaluationResult:
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
            acc=0.0,
            acc_stderr=0.0,
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
    return BaselineEvaluationResult(
        status="succeeded",
        mode="smoke",
        run_id="20260715T000000Z-qwen3-baseline-smoke-aaaaaaaa-cafe",
        config_sha256="a" * 64,
        git_commit="b" * 40,
        git_dirty=True,
        model_repository="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        domain=DomainBaselineSummary(
            status="complete",
            suite_version="tinyllm-domain-v1-83bdd8ef",
            evaluated_items=2,
            objective_items=2,
            objective_correct=1,
            human_review_pending=0,
            human_reviewed=0,
            human_passed=0,
            json_items=2,
            json_valid=2,
        ),
        general=GeneralBaselineSummary(
            harness_version="0.4.12",
            model_parameters=596_049_920,
            tasks=tasks,
            evaluation_seconds=2.5,
        ),
    )


def test_eval_baseline_requires_explicit_gpu_and_emits_path_free_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        main(
            [
                "eval",
                "baseline",
                "--config",
                "configs/eval/m2_baseline_smoke.yaml",
                "--artifact-root",
                str(tmp_path),
                "--json",
            ]
        )
        == 2
    )
    assert "--gpu-index" in json.loads(capsys.readouterr().err)["error"]["message"]

    seen: dict[str, object] = {}

    def preflight(index: int) -> BaselineGpuPreflight:
        seen["index"] = index
        return BaselineGpuPreflight(
            physical_index=index,
            memory_used_mib=1,
            utilization_percent=0,
            temperature_c=30,
        )

    def run(**_kwargs: object) -> BaselineEvaluationResult:
        seen["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return baseline_result()

    monkeypatch.setattr("tinyllm.cli.preflight_baseline_gpu", preflight)
    monkeypatch.setattr("tinyllm.cli.run_baseline_evaluation", run)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert (
        main(
            [
                "eval",
                "baseline",
                "--config",
                "configs/eval/m2_baseline_smoke.yaml",
                "--artifact-root",
                str(tmp_path),
                "--gpu-index",
                "5",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == baseline_result().run_id
    assert str(tmp_path) not in json.dumps(payload)
    assert seen == {"index": 5, "visible": "5"}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous


def test_eval_baseline_maps_gpu_preflight_to_exit_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_index: int) -> BaselineGpuPreflight:
        raise BaselinePreflightError("selected physical GPU is busy")

    monkeypatch.setattr("tinyllm.cli.preflight_baseline_gpu", fail)
    code = main(
        [
            "eval",
            "baseline",
            "--config",
            "configs/eval/m2_baseline_smoke.yaml",
            "--artifact-root",
            str(tmp_path),
            "--gpu-index",
            "5",
            "--json",
        ]
    )
    assert code == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "BASELINE_PREFLIGHT_FAILED"


def test_eval_baseline_review_emits_result_and_maps_contract_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = baseline_result()
    monkeypatch.setattr("tinyllm.cli.complete_baseline_human_review", lambda **_kwargs: result)
    code = main(
        [
            "eval",
            "baseline-review",
            "--run-id",
            result.run_id,
            "--judgments",
            str(tmp_path / "judgments.jsonl"),
            "--artifact-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == result.to_dict()

    def fail(**_kwargs: object) -> BaselineEvaluationResult:
        raise BaselineContractError("Run is not awaiting human review")

    monkeypatch.setattr("tinyllm.cli.complete_baseline_human_review", fail)
    code = main(
        [
            "eval",
            "baseline-review",
            "--run-id",
            result.run_id,
            "--judgments",
            str(tmp_path / "judgments.jsonl"),
            "--artifact-root",
            str(tmp_path),
            "--json",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "BASELINE_REVIEW_ERROR"

    code = main(
        [
            "eval",
            "baseline-review",
            "--run-id",
            result.run_id,
            "--judgments",
            str(tmp_path / "judgments.jsonl"),
            "--artifact-root",
            "relative",
            "--json",
        ]
    )
    assert code == 2
    assert "absolute" in json.loads(capsys.readouterr().err)["error"]["message"]
