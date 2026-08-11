from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch

import tinyllm.cli as cli_module
import tinyllm.evaluation.m6_domain as m6_domain_module
from tinyllm.cli import main
from tinyllm.evaluation import (
    BaselinePreflightError,
    HumanRubricJudgment,
    M6DomainError,
    M6DomainPassSummary,
    M6ModelIdentity,
    build_m6_domain_transcript,
    finalize_m6_domain_pass,
    load_evaluation_items,
    load_m6_release_config,
    parse_m6_final_answer,
    repair_m6_json_answer,
    run_m6_domain_pass,
    sha256_file,
)
from tinyllm.evaluation.m5_thinking_budget_schema import EARLY_STOPPING_TEXT
from tinyllm.evaluation.m6_domain import THINKING_FINAL_SEPARATOR
from tinyllm.schemas import canonical_config_hash


def _base_model() -> M6ModelIdentity:
    return M6ModelIdentity(
        role="base",
        repository="Qwen/Qwen3-0.6B",
        base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        adaptation="base",
        model_artifact_sha256="a" * 64,
        model_parameters=596_049_920,
    )


def _write_jsonl(path: Path, values: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def test_m6_final_answer_parser_separates_modes_and_rejects_leakage() -> None:
    assert parse_m6_final_answer(
        "private trace\n</think>\n\nfinal",
        mode="thinking",
    ) == ("final", True, False)
    assert parse_m6_final_answer("final", mode="nonthinking") == ("final", True, False)
    assert parse_m6_final_answer("<think>leak</think> final", mode="nonthinking") == (
        "<think>leak</think> final",
        False,
        True,
    )
    assert parse_m6_final_answer("unfinished", mode="thinking") == ("", False, False)


def test_m6_transcript_scores_only_final_answer() -> None:
    item = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))[0]
    response = f"incorrect private trace\n</think>\n\n{item.reference_answer}"
    result = build_m6_domain_transcript(
        item,
        mode="thinking",
        prompt="prompt",
        response=response,
        first_pass_response=response,
        continuation_response="",
        controller_action="natural_complete",
        prompt_tokens=10,
        first_pass_tokens=20,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )

    assert result.final_answer == item.reference_answer
    assert result.automatic_correct is True
    assert result.format_valid is True


def test_m6_json_repair_changes_only_the_syntax_shell_and_retains_raw_answer() -> None:
    items = {item.id: item for item in load_evaluation_items(Path("evals/domain/v3/items.jsonl"))}
    wrapped, wrap_action = repair_m6_json_answer(
        items["domain-json-001"], "[36,38,40]", policy="json-syntax-only-v1"
    )
    fragment, fragment_action = repair_m6_json_answer(
        items["domain-json-012"],
        '"id":38,"ok":true',
        policy="json-syntax-only-v1",
    )
    closed, close_action = repair_m6_json_answer(
        items["domain-json-034"],
        '{"enabled":false,"name":"worker-37","retries":38,"tags":[]',
        policy="json-syntax-only-v1",
    )

    assert (wrapped, wrap_action) == ('{"even":[36,38,40]}', "wrap_single_key")
    assert (fragment, fragment_action) == ('{"id":38,"ok":true}', "brace_member_fragment")
    assert close_action == "close_object"
    assert json.loads(closed)["name"] == "worker-37"
    assert repair_m6_json_answer(
        items["domain-json-001"],
        '{"even":[1]}',
        policy="json-syntax-only-v1",
    ) == ('{"even":[1]}', "none")

    transcript = build_m6_domain_transcript(
        items["domain-json-001"],
        mode="nonthinking",
        prompt="prompt",
        response="[36,38,40]",
        first_pass_response="[36,38,40]",
        continuation_response="",
        controller_action="not_applicable",
        prompt_tokens=10,
        first_pass_tokens=5,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
        json_repair_policy="json-syntax-only-v1",
    )
    assert transcript.final_answer == '{"even":[36,38,40]}'
    assert transcript.raw_final_answer == "[36,38,40]"
    assert transcript.output_repair_action == "wrap_single_key"
    assert transcript.json_valid is True
    assert transcript.automatic_correct is True


def test_m6_json_repair_v2_handles_only_generalized_shell_failures() -> None:
    items = {item.id: item for item in load_evaluation_items(Path("evals/domain/v3/items.jsonl"))}
    cases = (
        (
            "domain-config-003",
            '```json\n{"data":{},"logging":{},"model":{},"training":{}}\n```',
            "unwrap_json_fence",
        ),
        (
            "domain-config-008",
            '{"data":{"workers":39,"model":{},"training":{}}}',
            "promote_required_keys",
        ),
        ("domain-json-008", "enabled", "wrap_bareword_single_key"),
        ("domain-json-012", "{id:38,ok:true}", "quote_bare_keys"),
        ("domain-json-037", '["even"]=>[38,40,42]', "arrow_single_key"),
    )
    for item_id, answer, expected_action in cases:
        repaired, action = repair_m6_json_answer(
            items[item_id],
            answer,
            policy="json-syntax-only-v2",
        )
        assert action == expected_action
        decoded = json.loads(repaired)
        scorer = items[item_id].scorer
        assert scorer.kind == "json_object"
        assert set(scorer.required_keys).issubset(decoded)

    unchanged, action = repair_m6_json_answer(
        items["domain-json-012"],
        '{"id":38,"ok":false}',
        policy="json-syntax-only-v2",
    )
    assert (unchanged, action) == ('{"id":38,"ok":false}', "none")


def test_m6_domain_review_finalizes_all_300_content_free_scores(tmp_path: Path) -> None:
    items = load_evaluation_items(Path("evals/domain/v1/items.jsonl"))
    transcripts = tuple(
        build_m6_domain_transcript(
            item,
            mode="thinking",
            prompt=f"prompt:{item.id}",
            response=f"trace\n</think>\n\n{item.reference_answer}",
            first_pass_response=f"trace\n</think>\n\n{item.reference_answer}",
            continuation_response="",
            controller_action="natural_complete",
            prompt_tokens=10,
            first_pass_tokens=20,
            continuation_tokens=0,
            injected_tokens=0,
            finish_reason="eos",
        )
        for item in items
    )
    pass_dir = tmp_path / "pass"
    raw_path = pass_dir / "results.jsonl"
    _write_jsonl(raw_path, tuple(item.to_dict() for item in transcripts))
    summary = M6DomainPassSummary(
        status="awaiting_human_review",
        evaluation_id="m6-base-thinking",
        protocol_version="m6-release-v1",
        suite_version="tinyllm-domain-v1-83bdd8ef",
        config_sha256="b" * 64,
        git_commit="c" * 40,
        git_dirty=False,
        model=_base_model(),
        mode="thinking",
        evaluated_items=300,
        objective_items=260,
        objective_correct_items=260,
        human_review_pending=40,
        human_reviewed=0,
        human_passed=0,
        json_items=80,
        json_valid_items=80,
        format_valid_items=300,
        visible_reasoning_leakage_items=0,
        natural_thinking_closed_items=300,
        budget_forced_close_items=0,
        generated_tokens=6000,
        injected_tokens=0,
        duration_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        environment_sha256="d" * 64,
        hardware_sha256="e" * 64,
        raw_results_sha256=sha256_file(raw_path),
    )
    (pass_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    judgments = tuple(
        HumanRubricJudgment(
            item_id=item.id,
            criterion_results=(True, True, True),
            passed=True,
            rationale="All frozen rubric criteria are satisfied.",
            reviewer_role="maintainer",
        )
        for item in items
        if item.scorer.kind == "human_rubric"
    )
    judgment_path = tmp_path / "judgments.jsonl"
    _write_jsonl(judgment_path, tuple(item.to_dict() for item in judgments))

    result = finalize_m6_domain_pass(
        project_root=Path("."),
        pass_directory=pass_dir,
        judgments_path=judgment_path,
    )

    assert result.correct_items == 300
    assert result.format_valid_items == 300
    assert result.natural_thinking_closed_items == 300
    assert (pass_dir / "mode_result.json").is_file()
    completed = M6DomainPassSummary.model_validate_json(
        (pass_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert completed.status == "succeeded"
    assert completed.human_passed == 40


def test_m6_domain_review_rejects_incomplete_judgments(tmp_path: Path) -> None:
    judgment_path = tmp_path / "empty.jsonl"
    judgment_path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        finalize_m6_domain_pass(
            project_root=Path("."),
            pass_directory=tmp_path / "missing",
            judgments_path=judgment_path,
        )


def test_m6_domain_cli_rejects_invalid_mode_before_gpu_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "eval",
                "m6-domain",
                "--base-import",
                str((tmp_path / "base.json").resolve()),
                "--model-dir",
                str((tmp_path / "model").resolve()),
                "--tokenizer-dir",
                str((tmp_path / "model").resolve()),
                "--output-dir",
                str((tmp_path / "output").resolve()),
                "--gpu-index",
                "5",
                "--mode",
                "automatic",
                "--json",
            ]
        )
        == 2
    )
    assert "thinking or nonthinking" in json.loads(capsys.readouterr().err)["error"]["message"]


@pytest.mark.parametrize(
    ("mode", "natural_closed", "forced_closed"),
    (("thinking", 299, 1), ("nonthinking", 0, 0)),
)
def test_m6_domain_pass_exercises_controller_and_writes_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["thinking", "nonthinking"],
    natural_closed: int,
    forced_closed: int,
) -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        padding_side = "right"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            assert not tokenize
            assert add_generation_prompt
            if messages == [{"role": "user", "content": "TinyLLM template probe."}]:
                if enable_thinking:
                    return (
                        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                return (
                    "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
            return "frozen prompt"

        def __call__(
            self,
            prompts: list[str],
            *,
            padding: bool,
            return_tensors: str,
        ) -> dict[str, torch.Tensor]:
            assert padding
            assert return_tensors == "pt"
            if all(prompt == "frozen prompt" for prompt in prompts):
                return {
                    "input_ids": torch.tensor([[1, 2]] * len(prompts), dtype=torch.long),
                    "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
                }
            continuation_rows: list[list[int]] = []
            for prompt in prompts:
                if prompt.endswith(EARLY_STOPPING_TEXT):
                    continuation_rows.append([1, 2, 12, 31])
                elif prompt.endswith(THINKING_FINAL_SEPARATOR):
                    continuation_rows.append([1, 2, 11, 32])
                else:
                    raise AssertionError("unexpected continuation context")
            return {
                "input_ids": torch.tensor(continuation_rows, dtype=torch.long),
                "attention_mask": torch.ones((len(continuation_rows), 4), dtype=torch.long),
            }

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            if text == EARLY_STOPPING_TEXT:
                return [31]
            if text == THINKING_FINAL_SEPARATOR:
                return [32]
            return []

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens
            assert not clean_up_tokenization_spaces
            pieces = {
                0: "",
                10: "private trace</think>\n\nfinal",
                11: "private trace</think>",
                12: "private trace",
                20: "\n\nfinal",
                31: EARLY_STOPPING_TEXT,
                32: THINKING_FINAL_SEPARATOR,
                99: "",
            }
            return "".join(pieces[token_id] for token_id in token_ids)

    class FakeModel:
        def __init__(self) -> None:
            self.first_batches = 0
            self.continuation_tail_ids: list[list[int]] = []

        def to(self, device: torch.device) -> FakeModel:
            assert device.type == "cpu"
            return self

        def eval(self) -> FakeModel:
            return self

        def generate(self, **kwargs: Any) -> torch.Tensor:
            inputs = kwargs["input_ids"]
            assert isinstance(inputs, torch.Tensor)
            if inputs.shape[1] == 2:
                if mode == "thinking":
                    assert kwargs["stop_strings"] == ("</think>",)
                    assert kwargs["tokenizer"] is tokenizer
                else:
                    assert "stop_strings" not in kwargs
                rows = [[10, 99, 0] for _ in range(inputs.shape[0])]
                if self.first_batches == 0:
                    rows[1] = [11, 0, 0]
                    rows[2] = [12, 0, 0]
                self.first_batches += 1
            else:
                assert "stop_strings" not in kwargs
                assert kwargs["do_sample"] is False
                self.continuation_tail_ids.append([int(value) for value in inputs[:, -1].tolist()])
                rows = [[20, 99] for _ in range(inputs.shape[0])]
            generated = torch.tensor(rows, dtype=torch.long)
            return torch.cat((inputs, generated), dim=1)

    tokenizer = FakeTokenizer()
    model = FakeModel()
    fake_transformers = SimpleNamespace(
        __version__="4.57.1",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    cpu_device = torch.device("cpu")
    monkeypatch.setattr(torch, "device", lambda *_args: cpu_device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _d: 456)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _d: SimpleNamespace(total_memory=24, major=8, minor=6),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _d: "Synthetic RTX 3090",
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        m6_domain_module,
        "read_git_identity",
        lambda _root: ("c" * 40, False),
    )
    monkeypatch.setattr(
        m6_domain_module,
        "model_artifact_sha256",
        lambda *_args: "a" * 64,
    )
    model_dir = (tmp_path / "model").resolve()
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
            }
        ),
        encoding="utf-8",
    )
    release_path = Path("configs/eval/m6_development_v3_controlled.yaml")
    release = load_m6_release_config(release_path)
    output = (tmp_path / f"m6-{mode}").resolve()

    result = run_m6_domain_pass(
        release_config_path=release_path,
        model_dir=model_dir,
        tokenizer_dir=model_dir,
        output_dir=output,
        project_root=Path("."),
        physical_gpu_index=5,
        model_identity=_base_model(),
        mode=mode,
        expected_config_sha256=canonical_config_hash(release),
    )

    assert result.status == "awaiting_human_review"
    assert result.evaluated_items == 300
    assert result.natural_thinking_closed_items == natural_closed
    assert result.budget_forced_close_items == forced_closed
    assert result.peak_allocated_bytes == 123
    assert result.physical_gpu_index == 5
    assert model.first_batches == 75
    assert model.continuation_tail_ids == ([[32, 31]] if mode == "thinking" else [])
    assert (output / "environment.json").is_file()
    assert (output / "hardware.json").is_file()
    assert (output / "results.jsonl").is_file()
    assert (output / "summary.json").is_file()


def test_m6_domain_cli_runs_preflight_and_restores_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    imported = SimpleNamespace(model=_base_model(), config_sha256="b" * 64)
    seen: dict[str, object] = {}
    result = SimpleNamespace(
        status="awaiting_human_review",
        evaluation_id="m6-thinking-base",
        objective_correct_items=1,
        model_dump_json=lambda *, indent: json.dumps(
            {"status": "awaiting_human_review", "indent": indent}
        ),
    )
    monkeypatch.setattr(cli_module, "load_m6_base_import", lambda _path: imported)

    def preflight(index: int) -> None:
        seen["preflight"] = index

    def run(**kwargs: object) -> object:
        seen["visible"] = sys.modules["os"].environ.get("CUDA_VISIBLE_DEVICES")
        seen["mode"] = kwargs["mode"]
        return result

    monkeypatch.setattr(cli_module, "preflight_baseline_gpu", preflight)
    monkeypatch.setattr(cli_module, "run_m6_domain_pass", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8")
    code = main(
        [
            "eval",
            "m6-domain",
            "--base-import",
            str((tmp_path / "base.json").resolve()),
            "--model-dir",
            str((tmp_path / "model").resolve()),
            "--tokenizer-dir",
            str((tmp_path / "model").resolve()),
            "--output-dir",
            str((tmp_path / "output").resolve()),
            "--gpu-index",
            "5",
            "--mode",
            "thinking",
            "--json",
        ]
    )

    assert code == 0
    assert seen == {"preflight": 5, "visible": "5", "mode": "thinking"}
    assert sys.modules["os"].environ["CUDA_VISIBLE_DEVICES"] == "8"
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_human_review"


@pytest.mark.parametrize(
    ("error", "exit_code", "error_code"),
    (
        (BaselinePreflightError("busy"), 3, "M6_PREFLIGHT_FAILED"),
        (M6DomainError("generation failed"), 6, "M6_DOMAIN_FAILED"),
    ),
)
def test_m6_domain_cli_maps_runtime_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: RuntimeError,
    exit_code: int,
    error_code: str,
) -> None:
    imported = SimpleNamespace(model=_base_model(), config_sha256="b" * 64)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cli_module, "load_m6_base_import", lambda _path: imported)

    def fail(_index: int) -> None:
        raise error

    monkeypatch.setattr(cli_module, "preflight_baseline_gpu", fail)
    code = main(
        [
            "eval",
            "m6-domain",
            "--base-import",
            str((tmp_path / "base.json").resolve()),
            "--model-dir",
            str((tmp_path / "model").resolve()),
            "--tokenizer-dir",
            str((tmp_path / "model").resolve()),
            "--output-dir",
            str((tmp_path / "output").resolve()),
            "--gpu-index",
            "5",
            "--json",
        ]
    )

    assert code == exit_code
    assert "CUDA_VISIBLE_DEVICES" not in sys.modules["os"].environ
    assert json.loads(capsys.readouterr().err)["error"]["code"] == error_code


def test_m6_domain_review_cli_emits_result_and_maps_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(
        mode="thinking",
        correct_items=200,
        format_valid_items=299,
        model_dump_json=lambda *, indent: json.dumps({"mode": "thinking", "indent": indent}),
    )
    monkeypatch.setattr(cli_module, "finalize_m6_domain_pass", lambda **_kwargs: result)
    arguments = [
        "eval",
        "m6-domain-review",
        "--pass-directory",
        str((tmp_path / "pass").resolve()),
        "--judgments",
        str((tmp_path / "judgments.jsonl").resolve()),
        "--json",
    ]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"mode": "thinking", "indent": 2}

    def fail(**_kwargs: object) -> None:
        raise M6DomainError("review failed")

    monkeypatch.setattr(cli_module, "finalize_m6_domain_pass", fail)
    assert main(arguments) == 6
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "M6_REVIEW_FAILED"
