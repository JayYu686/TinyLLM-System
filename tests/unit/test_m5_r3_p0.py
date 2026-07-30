from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.run_m5_r3_p0 import (
    M5R3P0EnvironmentError,
    _subprocess_command,
    _verify_model_directory,
    _verify_policy_python,
    build_parser,
)
from tinyllm.data.m5_r3_p0 import (
    M5R3P0Error,
    build_m5_r3_p0_dataset,
    check_m5_r3_p0_contamination,
    generate_m5_r3_p0_tasks,
    load_m5_r3_p0_config,
    m5_r3_p0_config_sha256,
    m5_r3_p0_generation_seed,
    select_m5_r3_p0_candidate,
)
from tinyllm.data.reasoning import (
    generate_reasoning_dev_tasks,
    generate_reasoning_pilot_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.data.reasoning_schema import TeacherFinishReason, TeacherGenerationRecord
from tinyllm.data.tokenization import OffsetTokenizer, TokenEncoding

CONFIG = Path("configs/data/m5_r3_p0.yaml")
R1_CONFIG = Path("configs/data/m5_r3_p0_r1.yaml")
REASONING_CONFIG = Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


class _WordTokenizer:
    @property
    def vocab_size(self) -> int:
        return 200_000

    def token_to_id(self, token: str) -> int | None:
        del token
        return 1

    def encode(self, text: str) -> TokenEncoding:
        length = 193 if text.startswith("overlong") else max(1, len(text.split()))
        return TokenEncoding(
            ids=tuple(range(length)),
            offsets=tuple((index, index + 1) for index in range(length)),
        )


class _SequenceOverflowTokenizer(_WordTokenizer):
    def encode(self, text: str) -> TokenEncoding:
        if "sequence-overflow" in text and not text.startswith("sequence-overflow"):
            return TokenEncoding(
                ids=tuple(range(1025)),
                offsets=tuple((index, index + 1) for index in range(1025)),
            )
        return super().encode(text)


class _EmptyTokenizer(_WordTokenizer):
    def encode(self, text: str) -> TokenEncoding:
        del text
        return TokenEncoding(ids=(), offsets=())


class _RepeatedTokenizer(_WordTokenizer):
    def encode(self, text: str) -> TokenEncoding:
        if text.startswith("loop"):
            return TokenEncoding(
                ids=(1,) * 20,
                offsets=tuple((index, index + 1) for index in range(20)),
            )
        return super().encode(text)


def _record(
    task: Any,
    *,
    reasoning: str,
    candidate_index: int = 0,
    finish_reason: TeacherFinishReason = "stop",
    raw_output: str | None = None,
) -> TeacherGenerationRecord:
    output = raw_output or f"<think>{reasoning}</think>\n\n{task.expected_answer_json}"
    return TeacherGenerationRecord(
        generation_id=f"{task.id}:candidate-{candidate_index}",
        task_id=task.id,
        candidate_index=candidate_index,
        seed=20260731 + candidate_index,
        prompt_sha256=task.prompt_sha256,
        status="succeeded",
        finish_reason=finish_reason,
        raw_output=output,
        raw_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        observed_token_count=100,
    )


def _inputs(
    config_path: Path = CONFIG,
) -> tuple[Any, Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    config = load_m5_r3_p0_config(config_path)
    reasoning = load_m5_reasoning_data_config(REASONING_CONFIG)
    tasks = generate_m5_r3_p0_tasks(config)
    dev = generate_reasoning_dev_tasks(reasoning)
    historical = generate_reasoning_pilot_tasks(
        seed=reasoning.pilot_task_seed,
        tasks_per_family=20,
        task_contract_version=reasoning.task_contract_version,
    )
    return config, reasoning, tasks, dev, historical


def test_p0_tasks_are_deterministic_balanced_and_diverse() -> None:
    config, _reasoning, tasks, _dev, _historical = _inputs()

    assert tasks == generate_m5_r3_p0_tasks(config)
    assert len(tasks) == 40
    assert len({item.prompt_sha256 for item in tasks}) == 40
    assert sum(item.task_family == "config" for item in tasks) == 20
    assert sum(item.task_family == "log_diagnosis" for item in tasks) == 20
    assert sum(item.language == "en" for item in tasks) == 28
    assert sum(item.language == "zh" for item in tasks) == 12
    assert all("192" in item.prompt for item in tasks)


def test_p0_r1_has_independent_identity_and_only_changes_prompt_control() -> None:
    p0_config, _reasoning, p0_tasks, _dev, _historical = _inputs()
    r1_config, _reasoning, r1_tasks, r1_dev, r1_historical = _inputs(R1_CONFIG)

    assert r1_config.pilot_version == "m5-r3-p0-r1-v1"
    assert r1_config.parent_p0_public_result_sha256 == (
        "5eff250ef4cde98d044c992a0aaf7e2eb75342faa9c377d265a25945a3d4388b"
    )
    assert m5_r3_p0_config_sha256(p0_config) == (
        "ffd32c3d3ac9e7a235243f643be9018c1554909e2214f23c281039b83e5a9219"
    )
    assert m5_r3_p0_config_sha256(r1_config) == (
        "6f890910fba0120003133217d788ad4f30e2f0b932f5fd28a47f98bf5513880a"
    )
    assert r1_config.teacher == p0_config.teacher
    assert r1_config.verifier == p0_config.verifier
    assert r1_config.trace_policy == p0_config.trace_policy
    assert r1_config.gate == p0_config.gate
    assert r1_config.sampling.model_dump(exclude={"base_seed"}) == (
        p0_config.sampling.model_dump(exclude={"base_seed"})
    )
    assert r1_config.task_seed != p0_config.task_seed
    assert r1_config.sampling.base_seed != p0_config.sampling.base_seed
    assert len(r1_tasks) == len(p0_tasks) == 40
    assert {item.id for item in r1_tasks}.isdisjoint(item.id for item in p0_tasks)
    assert {item.template_family for item in r1_tasks} == {
        "pilot.config.r3-targeted-p0r1.v1",
        "pilot.log_diagnosis.r3-targeted-p0r1.v1",
    }
    assert all(item.id.startswith("m5-reasoning:pilot:r3p0r1-") for item in r1_tasks)
    english = tuple(item for item in r1_tasks if item.language == "en")
    chinese = tuple(item for item in r1_tasks if item.language == "zh")
    assert all("state the selected label first" in item.prompt for item in english)
    assert all("cite exactly one direct evidence fragment" in item.prompt for item in english)
    assert all("do not discuss other labels" in item.prompt for item in english)
    assert all("先给出所选标签" in item.prompt for item in chinese)
    assert all("引用输入中的一处直接证据" in item.prompt for item in chinese)
    assert all("不要讨论其他标签" in item.prompt for item in chinese)

    contamination = check_m5_r3_p0_contamination(
        r1_tasks,
        dev_tasks=r1_dev,
        historical_tasks=r1_historical,
    )
    assert contamination.status == "pass"
    assert contamination.p0_tasks_sha256 == (
        "4cc14273c8351b94c3221c3b7c0e934afb026169534f9a0cc2d8d862b46d0688"
    )


def test_p0_config_loader_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(M5R3P0Error, match="must use YAML"):
        load_m5_r3_p0_config(tmp_path / "config.json")
    with pytest.raises(M5R3P0Error, match="cannot be read"):
        load_m5_r3_p0_config(tmp_path / "missing.yaml")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[unclosed", encoding="utf-8")
    with pytest.raises(M5R3P0Error, match="invalid YAML"):
        load_m5_r3_p0_config(invalid_yaml)

    invalid_schema = tmp_path / "invalid-schema.yaml"
    invalid_schema.write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(M5R3P0Error, match="violates its schema"):
        load_m5_r3_p0_config(invalid_schema)

    mismatched_seed = tmp_path / "mismatched-seed.yaml"
    mismatched_seed.write_text(
        R1_CONFIG.read_text(encoding="utf-8").replace(
            "base_seed: 20260802",
            "base_seed: 20260731",
        ),
        encoding="utf-8",
    )
    with pytest.raises(M5R3P0Error, match="violates its schema"):
        load_m5_r3_p0_config(mismatched_seed)


def test_p0_contamination_passes_and_detects_historical_collision() -> None:
    _config, _reasoning, tasks, dev, historical = _inputs()

    passed = check_m5_r3_p0_contamination(
        tasks,
        dev_tasks=dev,
        historical_tasks=historical,
    )
    failed = check_m5_r3_p0_contamination(
        tasks,
        dev_tasks=dev,
        historical_tasks=(tasks[0], *historical[1:]),
    )

    assert passed.status == "pass"
    assert passed.p0_tasks_sha256 == (
        "cf0fee91d2a3362cb3fd6567485b12596d278bac889fcfc8abc8453acc337f2f"
    )
    assert failed.status == "fail"
    assert failed.historical_exact_prompt_matches == 1
    assert failed.historical_normalized_prompt_matches == 1
    assert failed.historical_template_family_overlaps == 1


def test_p0_candidate_rejects_length_and_duplicate_without_repair() -> None:
    config, reasoning, tasks, _dev, _historical = _inputs()
    tokenizer = cast(OffsetTokenizer, cast(Any, _WordTokenizer()))
    first = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="brief unique reasoning"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=tokenizer,
        existing_trace_hashes=frozenset(),
    )
    assert first.sample is not None
    trace_hash = cast(str, first.normalized_trace_sha256)

    duplicate = select_m5_r3_p0_candidate(
        tasks[1],
        (_record(tasks[1], reasoning="brief unique reasoning"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=tokenizer,
        existing_trace_hashes=frozenset({trace_hash}),
    )
    length = select_m5_r3_p0_candidate(
        tasks[1],
        (
            _record(
                tasks[1],
                reasoning="complete but length limited",
                finish_reason="length",
            ),
        ),
        config=config,
        reasoning_config=reasoning,
        tokenizer=tokenizer,
        existing_trace_hashes=frozenset(),
    )

    assert duplicate.sample is None
    assert duplicate.audits[0].rejection_reason == "duplicate_normalized_trace"
    assert length.sample is None
    assert length.audits[0].rejection_reason == "teacher_length_limit"


def test_p0_candidate_rejects_overlong_reasoning() -> None:
    config, reasoning, tasks, _dev, _historical = _inputs()
    result = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="overlong private reasoning"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
        existing_trace_hashes=frozenset(),
    )

    assert result.sample is None
    assert result.audits[0].reasoning_tokens == 193
    assert result.audits[0].rejection_reason == "reasoning_over_192_tokens"


def test_p0_candidate_rejects_malformed_and_wrong_answers() -> None:
    config, reasoning, tasks, _dev, _historical = _inputs()
    tokenizer = cast(OffsetTokenizer, cast(Any, _WordTokenizer()))
    malformed = select_m5_r3_p0_candidate(
        tasks[0],
        (
            _record(
                tasks[0],
                reasoning="unused",
                raw_output="<think>unclosed reasoning",
            ),
        ),
        config=config,
        reasoning_config=reasoning,
        tokenizer=tokenizer,
        existing_trace_hashes=frozenset(),
    )
    wrong = select_m5_r3_p0_candidate(
        tasks[0],
        (
            _record(
                tasks[0],
                reasoning="brief but wrong",
                raw_output='<think>brief but wrong</think>\n\n{"issue":"cuda_oom"}',
            ),
        ),
        config=config,
        reasoning_config=reasoning,
        tokenizer=tokenizer,
        existing_trace_hashes=frozenset(),
    )

    assert malformed.sample is None
    assert malformed.audits[0].rejection_reason == "missing_think_block"
    assert wrong.sample is None
    assert wrong.audits[0].rejection_reason == "answer_mismatch"


def test_p0_candidate_rejects_repeated_lines_and_long_sequence() -> None:
    config, reasoning, tasks, _dev, _historical = _inputs()
    repeated = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="same evidence\nsame evidence"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
        existing_trace_hashes=frozenset(),
    )
    overflow = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="sequence-overflow trace"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _SequenceOverflowTokenizer())),
        existing_trace_hashes=frozenset(),
    )

    assert repeated.sample is None
    assert repeated.audits[0].rejection_reason == "identical_line_repetition"
    assert overflow.sample is None
    assert overflow.audits[0].rejection_reason == "sequence_over_1024_tokens"


def test_p0_candidate_rejects_runtime_empty_and_repeated_tokens() -> None:
    config, reasoning, tasks, _dev, _historical = _inputs()
    failed = TeacherGenerationRecord(
        generation_id=f"{tasks[0].id}:candidate-0",
        task_id=tasks[0].id,
        candidate_index=0,
        seed=20260731,
        prompt_sha256=tasks[0].prompt_sha256,
        status="failed",
        finish_reason="error",
        observed_token_count=0,
        error_code="generation_runtime_error",
    )
    runtime = select_m5_r3_p0_candidate(
        tasks[0],
        (failed,),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
        existing_trace_hashes=frozenset(),
    )
    empty = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="brief"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _EmptyTokenizer())),
        existing_trace_hashes=frozenset(),
    )
    repeated = select_m5_r3_p0_candidate(
        tasks[0],
        (_record(tasks[0], reasoning="loop tokens"),),
        config=config,
        reasoning_config=reasoning,
        tokenizer=cast(OffsetTokenizer, cast(Any, _RepeatedTokenizer())),
        existing_trace_hashes=frozenset(),
    )

    assert runtime.audits[0].rejection_reason == "generation_runtime_error"
    assert empty.audits[0].rejection_reason == "empty_reasoning"
    assert repeated.audits[0].rejection_reason == "repeated_8gram_over_500bp"


def test_p0_cpu_build_passes_both_family_gates() -> None:
    config, reasoning, tasks, dev, historical = _inputs()
    generations = tuple(
        _record(task, reasoning=f"brief evidence and label for {task.id}") for task in tasks
    )
    result = build_m5_r3_p0_dataset(
        tasks,
        generations,
        config=config,
        reasoning_config=reasoning,
        dev_tasks=dev,
        historical_tasks=historical,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
    )

    assert len(result.samples) == 40
    assert all(item.gate_passed for item in result.family_results)
    assert result.rejection_counts == {}
    assert len({item.content_sha256 for item in result.samples}) == 40


def test_p0_cpu_build_records_empty_family_failures() -> None:
    config, reasoning, tasks, dev, historical = _inputs()
    result = build_m5_r3_p0_dataset(
        tasks,
        (),
        config=config,
        reasoning_config=reasoning,
        dev_tasks=dev,
        historical_tasks=historical,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
    )

    assert result.samples == ()
    assert result.rejection_counts == {"no_candidate_passed": 40}
    assert all(not item.gate_passed for item in result.family_results)
    assert all(item.reasoning_tokens_min is None for item in result.family_results)


def test_p0_generation_seed_and_cli_are_stable() -> None:
    assert m5_r3_p0_generation_seed(100, 0, 0) == 100
    assert m5_r3_p0_generation_seed(100, 1, 1) == 103
    with pytest.raises(M5R3P0Error, match="invalid"):
        m5_r3_p0_generation_seed(100, -1, 0)

    args = build_parser().parse_args(
        [
            "--historical-pilot-artifact",
            "/private/pilot.json",
            "--model-dir",
            f"/models/{REVISION}",
            "--tokenizer-dir",
            "/models/tokenizer",
            "--gpu-index",
            "7",
            "--raw-output",
            "/private/r3-p0.json",
            "--public-output",
            "reports/m5/raw/m5_r3_p0.json",
        ]
    )
    assert args.config == CONFIG
    assert args.gpu_index == 7
    assert args.timeout_seconds == 7200
    assert args.policy_python == Path(".venv/bin/python")
    assert args.parent_p0_result == Path("reports/m5/raw/m5_r3_p0.json")

    command = _subprocess_command(
        args,
        interpreter=Path("/runtime/python"),
        mode="--worker",
        generation_output=Path("/private/generations.json"),
    )
    assert command[0] == "/runtime/python"
    assert command[-1] == "--worker"
    assert "/private/generations.json" in command


def test_p0_policy_python_preflight_is_fail_closed(tmp_path: Path) -> None:
    _verify_policy_python(Path(sys.executable), Path.cwd())

    failing = tmp_path / "python"
    failing.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    failing.chmod(0o755)
    with pytest.raises(M5R3P0EnvironmentError, match="tokenizers 0.21.4"):
        _verify_policy_python(failing, Path.cwd())
    with pytest.raises(M5R3P0EnvironmentError, match="unavailable"):
        _verify_policy_python(tmp_path / "missing", Path.cwd())


def test_p0_teacher_snapshot_requires_pinned_qwen3_gqa(tmp_path: Path) -> None:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        _verify_model_directory(snapshot, REVISION)

    for name in (
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    _verify_model_directory(snapshot, REVISION)
