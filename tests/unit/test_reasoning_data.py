from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    M5ReasoningDataConfig,
    M5ReasoningDatasetManifest,
    M5TeacherSmokeResult,
    ReasoningDataError,
    ReasoningRejectedRecord,
    ReasoningTask,
    TeacherGenerationRecord,
    build_reasoning_dataset,
    build_reasoning_task_manifest,
    build_synthetic_teacher_generations,
    generate_reasoning_dev_tasks,
    generate_reasoning_pilot_tasks,
    load_m5_reasoning_data_config,
    parse_teacher_output,
    verify_reasoning_answer,
)
from tinyllm.data.reasoning_schema import TeacherFinishReason

CONFIG_PATH = Path("configs/data/m5_reasoning.yaml")


def _config() -> M5ReasoningDataConfig:
    return load_m5_reasoning_data_config(CONFIG_PATH)


def _generation(
    task: ReasoningTask,
    candidate_index: int,
    output: str,
    *,
    token_count: int = 64,
    finish_reason: TeacherFinishReason = "stop",
) -> TeacherGenerationRecord:
    return TeacherGenerationRecord(
        generation_id=f"{task.id}:candidate-{candidate_index}",
        task_id=task.id,
        candidate_index=candidate_index,
        seed=42 + candidate_index,
        prompt_sha256=task.prompt_sha256,
        status="succeeded",
        finish_reason=finish_reason,
        raw_output=output,
        raw_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        observed_token_count=token_count,
    )


def test_formal_config_freezes_teacher_verifier_and_dev_distribution(tmp_path: Path) -> None:
    config = _config()

    assert config.teacher.repository == "Qwen/Qwen3-8B"
    assert config.teacher.attention_architecture == "gqa"
    assert config.teacher.local_files_only is True
    assert config.sampling.candidate_count == 2
    assert config.dev.task_family_counts == {
        "config": 40,
        "json": 40,
        "linux": 40,
        "log_diagnosis": 40,
        "python": 40,
    }
    assert config.dev.language_counts_per_family == {"en": 28, "zh": 12}

    with pytest.raises(ReasoningDataError, match="extension"):
        load_m5_reasoning_data_config(tmp_path / "config.json")
    with pytest.raises(ReasoningDataError, match="cannot read"):
        load_m5_reasoning_data_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(ReasoningDataError, match="invalid YAML"):
        load_m5_reasoning_data_config(invalid)


def test_reasoning_dev_has_exact_frozen_distribution_and_identity() -> None:
    config = _config()
    first = generate_reasoning_dev_tasks(config)
    second = generate_reasoning_dev_tasks(config)
    manifest = build_reasoning_task_manifest(first, config=config)

    assert first == second
    assert len(first) == 200
    assert manifest.task_count == 200
    assert manifest.task_family_counts == config.dev.task_family_counts
    assert manifest.language_counts == {"en": 140, "zh": 60}
    assert manifest.task_set_version.startswith("m5-reasoning-dev-v1-")
    assert all(task.id.startswith("m5-reasoning:dev:") for task in first)
    assert all(task.template_family.startswith("dev.") for task in first)
    assert build_reasoning_task_manifest(reversed(first), config=config) == manifest


def test_reasoning_dev_rejects_wrong_size_and_mixed_split() -> None:
    config = _config()
    dev_tasks = generate_reasoning_dev_tasks(config)
    pilot_tasks = generate_reasoning_pilot_tasks(seed=7, tasks_per_family=10)

    with pytest.raises(ReasoningDataError, match="exactly 200"):
        build_reasoning_task_manifest(dev_tasks[:-1], config=config)
    with pytest.raises(ReasoningDataError, match="exactly one split"):
        build_reasoning_task_manifest((*dev_tasks, pilot_tasks[0]), config=config)


def test_pilot_generation_requires_exact_70_30_compatible_size() -> None:
    with pytest.raises(ReasoningDataError, match="multiple of 10"):
        generate_reasoning_pilot_tasks(seed=1, tasks_per_family=9)

    tasks = generate_reasoning_pilot_tasks(seed=1, tasks_per_family=10)
    assert len(tasks) == 50
    assert sum(task.language == "en" for task in tasks) == 35
    assert sum(task.language == "zh" for task in tasks) == 15
    assert all(task.template_family.startswith("pilot.") for task in tasks)


def test_reasoning_task_schema_binds_split_canonical_json_and_hashes() -> None:
    task = generate_reasoning_pilot_tasks(seed=1, tasks_per_family=10)[0]

    with pytest.raises(ValidationError, match="prompt hash"):
        ReasoningTask.model_validate({**task.model_dump(), "prompt_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="canonical"):
        ReasoningTask.model_validate(
            {**task.model_dump(), "expected_answer_json": '{"z": 1}'},
        )
    with pytest.raises(ValidationError, match="split namespace"):
        ReasoningTask.model_validate({**task.model_dump(), "split": "reasoning_dev"})


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("answer only", "missing_think_block"),
        ("<think>a</think><think>b</think>{}", "multiple_think_blocks"),
        ("prefix<think>a</think>{}", "nested_think_tag"),
        ("<think>  </think>{}", "empty_reasoning"),
        ("<think>reason</think>  ", "empty_final_answer"),
        ("<think><think>nested</think></think>{}", "multiple_think_blocks"),
    ],
)
def test_teacher_parser_rejects_malformed_visible_reasoning(output: str, reason: str) -> None:
    parsed, actual_reason = parse_teacher_output(output)
    assert parsed is None
    assert actual_reason == reason


def test_teacher_parser_accepts_optional_im_end_suffix() -> None:
    parsed, reason = parse_teacher_output('<think>reasoning</think>\n\n{"result":1}<|im_end|>')

    assert reason is None
    assert parsed is not None
    assert parsed.reasoning_content == "reasoning"
    assert parsed.final_answer == '{"result":1}'


def test_verifier_never_executes_and_distinguishes_invalid_from_wrong_json() -> None:
    config = _config()
    task = generate_reasoning_pilot_tasks(seed=1, tasks_per_family=10)[0]
    generation = _generation(task, 0, f"<think>x</think>{task.expected_answer_json}")

    accepted = verify_reasoning_answer(
        task=task,
        generation=generation,
        final_answer=task.expected_answer_json,
        config=config,
    )
    invalid = verify_reasoning_answer(
        task=task,
        generation=generation,
        final_answer="__import__('os').system('echo unsafe')",
        config=config,
    )
    wrong = verify_reasoning_answer(
        task=task,
        generation=generation,
        final_answer='{"wrong":true}',
        config=config,
    )

    assert accepted.passed is True
    assert invalid.reason == "invalid_final_json"
    assert wrong.reason == "answer_mismatch"


def test_synthetic_cpu_build_selects_first_passing_candidate_deterministically() -> None:
    config = _config()
    tasks = generate_reasoning_pilot_tasks(seed=1, tasks_per_family=10)
    generations = build_synthetic_teacher_generations(tasks, config=config)

    first = build_reasoning_dataset(tasks, generations, config=config)
    second = build_reasoning_dataset(reversed(tasks), reversed(generations), config=config)

    assert first == second
    assert first.manifest.input_tasks == 50
    assert first.manifest.accepted_samples == 50
    assert first.manifest.rejected_tasks == 0
    assert first.manifest.rejection_counts == {"invalid_final_json": 10}
    assert first.manifest.verified_candidates == 60
    assert first.manifest.unused_candidates == 40
    assert first.manifest.dataset_version.startswith("m5-reasoning-pilot-v1-")
    assert len(first.samples) == 50
    assert all("synthetic" in sample.reasoning_content for sample in first.samples)


def test_build_records_generation_parsing_length_and_exhaustion_failures() -> None:
    config = _config()
    task = generate_reasoning_pilot_tasks(seed=2, tasks_per_family=10)[0]
    failed = TeacherGenerationRecord(
        generation_id=f"{task.id}:candidate-0",
        task_id=task.id,
        candidate_index=0,
        seed=1,
        prompt_sha256=task.prompt_sha256,
        status="failed",
        finish_reason="error",
        observed_token_count=0,
        error_code="model_generate_failed",
    )
    length_output = "<think>partial</think>"
    length = _generation(task, 1, length_output, finish_reason="length")

    build = build_reasoning_dataset([task], [failed, length], config=config)

    assert build.samples == ()
    assert build.manifest.failed_generations == 1
    assert build.manifest.rejection_counts == {
        "no_candidate_passed": 1,
        "teacher_generation_failed": 1,
        "teacher_length_limit": 1,
    }
    assert all(record.model_dump().get("raw_output") is None for record in build.rejected)


def test_build_records_sequence_and_parse_failure_before_accepting_fallback() -> None:
    config = _config()
    task = generate_reasoning_pilot_tasks(seed=3, tasks_per_family=10)[0]
    overlength = _generation(
        task,
        0,
        f"<think>long</think>{task.expected_answer_json}",
        token_count=1025,
    )
    fallback = _generation(
        task,
        1,
        f"<think>valid fallback</think>\n\n{task.expected_answer_json}",
    )

    build = build_reasoning_dataset([task], [overlength, fallback], config=config)

    assert build.manifest.accepted_samples == 1
    assert build.manifest.rejection_counts == {"sequence_too_long": 1}
    rejection = build.rejected[0]
    assert rejection.observed_token_count == 1025
    assert rejection.max_sequence_length == 1024


def test_generation_lineage_integrity_rejects_unknown_duplicate_and_gapped_candidates() -> None:
    config = _config()
    tasks = generate_reasoning_pilot_tasks(seed=4, tasks_per_family=10)
    task = tasks[0]
    valid = _generation(task, 0, f"<think>x</think>{task.expected_answer_json}")

    with pytest.raises(ReasoningDataError, match="unique"):
        build_reasoning_dataset([task], [valid, valid], config=config)
    changed = valid.model_copy(update={"prompt_sha256": "0" * 64})
    with pytest.raises(ReasoningDataError, match="prompt hash"):
        build_reasoning_dataset([task], [changed], config=config)
    candidate_one = _generation(task, 1, f"<think>x</think>{task.expected_answer_json}")
    with pytest.raises(ReasoningDataError, match="contiguous"):
        build_reasoning_dataset([task], [candidate_one], config=config)


def test_rejection_and_manifest_schemas_reject_incoherent_evidence() -> None:
    config = _config()
    tasks = generate_reasoning_pilot_tasks(seed=5, tasks_per_family=10)
    generations = build_synthetic_teacher_generations(tasks, config=config)
    build = build_reasoning_dataset(tasks, generations, config=config)

    with pytest.raises(ValidationError, match="selection-level"):
        ReasoningRejectedRecord(
            task_id=tasks[0].id,
            generation_id=f"{tasks[0].id}:candidate-0",
            stage="selection",
            reason="no_candidate_passed",
            prompt_sha256=tasks[0].prompt_sha256,
        )
    with pytest.raises(ValidationError, match="accepted and rejected tasks"):
        M5ReasoningDatasetManifest.model_validate(
            {**build.manifest.model_dump(), "input_tasks": 51}
        )


def test_failed_teacher_record_cannot_hide_output_or_tokens() -> None:
    task = generate_reasoning_pilot_tasks(seed=6, tasks_per_family=10)[0]
    with pytest.raises(ValidationError, match="cannot retain output"):
        TeacherGenerationRecord(
            generation_id=f"{task.id}:candidate-0",
            task_id=task.id,
            candidate_index=0,
            seed=1,
            prompt_sha256=task.prompt_sha256,
            status="failed",
            finish_reason="error",
            raw_output="secret partial output",
            raw_output_sha256=hashlib.sha256(b"secret partial output").hexdigest(),
            observed_token_count=1,
            error_code="generation_error",
        )


def test_teacher_smoke_result_cannot_mark_dirty_or_unverified_run_as_pass() -> None:
    config = _config()
    common = {
        "status": "pass",
        "generated_at": datetime(2026, 7, 22, tzinfo=UTC),
        "model": config.teacher,
        "sampling": config.sampling,
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "physical_gpu_index": 9,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "torch_version": "2.7.1+cu118",
        "transformers_version": "4.57.6",
        "input_token_count": 64,
        "generated_token_counts": (32, 24),
        "generation_attempts": 2,
        "accepted_samples": 1,
        "rejection_counts": {},
        "dataset_version": "m5-reasoning-pilot-v1-1234abcd",
        "duration_seconds": 1.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
        "raw_artifact_sha256": "c" * 64,
    }

    assert M5TeacherSmokeResult.model_validate(common).status == "pass"
    with pytest.raises(ValidationError, match="clean lineage"):
        M5TeacherSmokeResult.model_validate({**common, "git_dirty": True})
    with pytest.raises(ValidationError, match="token counts"):
        M5TeacherSmokeResult.model_validate({**common, "generation_attempts": 1})
