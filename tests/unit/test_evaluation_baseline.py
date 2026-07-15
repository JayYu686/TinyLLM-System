from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from tests.unit.test_evaluation_schema import evaluation_item
from tinyllm.data import PinnedDataArtifact
from tinyllm.evaluation import (
    BaselineContractError,
    BaselineGpuPreflight,
    BaselinePreflightError,
    BaselineRuntimeError,
    DomainItemResult,
    EvaluationItem,
    GeneratedResponse,
    HumanRubricJudgment,
    TransformersDomainBackend,
    acquire_baseline_model,
    build_lm_eval_command,
    build_lm_eval_validation_command,
    load_baseline_config,
    load_baseline_runtime,
    load_evaluation_items,
    preflight_baseline_gpu,
    render_generation_prompt,
    run_domain_generation,
    score_domain_response,
    seed_baseline,
    validate_baseline_inputs,
)
from tinyllm.evaluation.lm_tasks import _clean_hellaswag_text, process_hellaswag

FORMAL_CONFIG = Path("configs/eval/m2_baseline.yaml")
SMOKE_CONFIG = Path("configs/eval/m2_baseline_smoke.yaml")
DOMAIN_ITEMS = Path("evals/domain/v1/items.jsonl")


def _items_by_scorer() -> dict[str, EvaluationItem]:
    items = load_evaluation_items(DOMAIN_ITEMS)
    return {item.scorer.kind: item for item in items}


def test_formal_and_smoke_configs_freeze_distinct_execution_modes(tmp_path: Path) -> None:
    formal = load_baseline_config(FORMAL_CONFIG)
    smoke = load_baseline_config(SMOKE_CONFIG)

    assert formal.mode == "formal"
    assert formal.domain.limit is None
    assert formal.general.limit is None
    assert smoke.mode == "smoke"
    assert smoke.domain.limit == 2
    assert smoke.general.limit == 2

    raw = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    raw["domain"]["limit"] = 1
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(BaselineContractError, match="formal Baseline cannot"):
        load_baseline_config(invalid)


def test_config_refuses_unknown_fields_or_wrong_extension(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["unreviewed_override"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(BaselineContractError, match="Extra inputs"):
        load_baseline_config(invalid)
    with pytest.raises(BaselineContractError, match="extension"):
        load_baseline_config(tmp_path / "config.json")


def test_input_validation_binds_suite_and_task_adapter_hashes(tmp_path: Path) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    project_root = tmp_path.resolve()
    shutil.copytree("evals", project_root / "evals")
    shutil.copytree("configs/eval/lm_eval", project_root / "configs/eval/lm_eval")

    items = validate_baseline_inputs(config, project_root=project_root)
    assert len(items) == 2

    adapter = project_root / config.general.tasks[0].adapter_path
    adapter.write_text(adapter.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(BaselineContractError, match="Adapter hash mismatch"):
        validate_baseline_inputs(config, project_root=project_root)


def test_generation_prompt_is_frozen_qwen3_nonthinking_chatml() -> None:
    item = evaluation_item(prompt="Return one word.")

    rendered = render_generation_prompt(item)

    assert rendered == (
        "<|im_start|>user\nReturn one word.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_domain_objective_scorers_and_human_review_state() -> None:
    items = _items_by_scorer()
    exact = items["exact_match"]
    multiple_choice = items["multiple_choice"]
    json_item = items["json_object"]
    human = items["human_rubric"]

    exact_result = score_domain_response(
        exact,
        f"  {exact.reference_answer}\n",
        prompt_tokens=10,
        generated_tokens=2,
        finish_reason="eos",
    )
    choice_result = score_domain_response(
        multiple_choice,
        multiple_choice.reference_answer,
        prompt_tokens=10,
        generated_tokens=8,
        finish_reason="eos",
    )
    json_result = score_domain_response(
        json_item,
        json_item.reference_answer,
        prompt_tokens=10,
        generated_tokens=20,
        finish_reason="length",
    )
    human_result = score_domain_response(
        human,
        "The evidence is insufficient; provide the complete log and timestamp.",
        prompt_tokens=10,
        generated_tokens=12,
        finish_reason="eos",
    )

    assert exact_result.automatic_correct is True
    assert choice_result.automatic_correct is True
    assert json_result.automatic_correct is True
    assert json_result.json_valid is True
    assert human_result.automatic_correct is None
    assert human_result.human_review_required is True
    assert human_result.response_sha256


def test_domain_scorers_reject_bad_finish_reason_and_invalid_json() -> None:
    items = _items_by_scorer()
    with pytest.raises(BaselineContractError, match="finish reason"):
        score_domain_response(
            items["exact_match"],
            "answer",
            prompt_tokens=1,
            generated_tokens=1,
            finish_reason="cancelled",
        )

    result = score_domain_response(
        items["json_object"],
        "not JSON",
        prompt_tokens=1,
        generated_tokens=2,
        finish_reason="eos",
    )
    assert result.automatic_correct is False
    assert result.json_valid is False


def test_human_judgment_requires_three_of_three_and_rationale() -> None:
    passed = HumanRubricJudgment(
        item_id="domain-refusal-001",
        criterion_results=(True, True, True),
        passed=True,
        rationale="The response satisfies all frozen criteria.",
        reviewer_role="maintainer",
    )
    assert passed.passed is True

    with pytest.raises(ValidationError, match="must equal all three"):
        HumanRubricJudgment(
            item_id="domain-refusal-001",
            criterion_results=(True, False, True),
            passed=True,
            rationale="One criterion failed.",
            reviewer_role="maintainer",
        )


def test_lm_eval_command_is_reviewable_and_smoke_is_bounded(tmp_path: Path) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    project_root = Path.cwd().resolve()
    command = build_lm_eval_command(
        config,
        project_root=project_root,
        model_path=(tmp_path / "model").resolve(),
        output_path=(tmp_path / "results").resolve(),
        device="cuda:0",
    )

    joined = " ".join(command)
    assert "tinyllm_arc_easy,tinyllm_hellaswag,tinyllm_piqa" in joined
    assert "enable_thinking=False" in joined
    assert "--apply_chat_template" in command
    assert "--log_samples" in command
    assert "--check_integrity" not in command
    assert command[-2:] == ("--limit", "2")

    validation = build_lm_eval_validation_command(config, project_root=project_root)
    assert validation[3:5] == ("validate", "--tasks")
    assert "tinyllm_arc_easy,tinyllm_hellaswag,tinyllm_piqa" in validation

    with pytest.raises(BaselineContractError, match="absolute"):
        build_lm_eval_command(
            config,
            project_root=project_root,
            model_path=Path("relative"),
            output_path=(tmp_path / "results").resolve(),
            device="cpu",
        )


def test_hellaswag_adapter_matches_frozen_transform() -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.transformed: dict[str, object] | None = None

        def map(self, function: object) -> FakeDataset:
            assert callable(function)
            self.transformed = function(
                {
                    "activity_label": "Cooking",
                    "ctx_a": "A person starts",
                    "ctx_b": "mixing [noise] batter",
                    "endings": [" [title] serves it", "drops it"],
                    "label": "1",
                }
            )
            return self

    dataset = FakeDataset()
    assert _clean_hellaswag_text(" A [noise]  test ") == "A  test"
    assert process_hellaswag(dataset) is dataset
    assert dataset.transformed == {
        "query": "Cooking: A person starts Mixing batter",
        "choices": [" serves it", "drops it"],
        "gold": 1,
    }


def test_domain_result_schema_binds_response_hash() -> None:
    item = _items_by_scorer()["exact_match"]
    result = score_domain_response(
        item,
        item.reference_answer,
        prompt_tokens=1,
        generated_tokens=1,
        finish_reason="eos",
    )
    raw = json.loads(result.model_dump_json())
    raw["response"] = "tampered"

    with pytest.raises(ValidationError, match="SHA256"):
        type(result).model_validate(raw)


def test_model_acquisition_binds_config_and_one_snapshot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    calls: list[tuple[str, bool]] = []

    def acquire(artifact: PinnedDataArtifact, *, cache_root: Path, offline: bool) -> Path:
        name = artifact.name
        filename = artifact.cache_path.name
        calls.append((name, offline))
        return cache_root / "models/Qwen/Qwen3-0.6B/revision" / filename

    monkeypatch.setattr("tinyllm.evaluation.baseline_runtime.acquire_pinned_artifact", acquire)
    model_path = acquire_baseline_model(
        config,
        cache_root=tmp_path.resolve(),
        offline=True,
    )

    assert model_path == tmp_path / "models/Qwen/Qwen3-0.6B/revision"
    assert len(calls) == 5
    assert all(offline for _name, offline in calls)

    changed = config.model_dump()
    changed["model"]["files"][0]["size_bytes"] = 1
    drifted = type(config).model_validate(changed)
    with pytest.raises(BaselineRuntimeError, match="artifact contract"):
        acquire_baseline_model(drifted, cache_root=tmp_path.resolve(), offline=True)


def test_model_acquisition_maps_verified_cache_failures(
    tmp_path: Path,
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    with pytest.raises(BaselineRuntimeError, match="offline cache miss"):
        acquire_baseline_model(config, cache_root=tmp_path.resolve(), offline=True)


def test_domain_generation_preserves_order_and_backend_accounting() -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    items = validate_baseline_inputs(config, project_root=Path.cwd().resolve())

    class FakeBackend:
        def __init__(self) -> None:
            self.prompts: tuple[str, ...] = ()

        def generate(
            self,
            prompts: Sequence[str],
            *,
            batch_size: int,
            max_sequence_length: int,
            max_new_tokens: int,
        ) -> tuple[GeneratedResponse, ...]:
            assert batch_size == 4
            assert max_sequence_length == 1024
            assert max_new_tokens == 512
            self.prompts = tuple(prompts)
            return tuple(
                GeneratedResponse(
                    response=item.reference_answer,
                    prompt_tokens=20 + index,
                    generated_tokens=4,
                    finish_reason="eos",
                )
                for index, item in enumerate(items)
            )

    backend = FakeBackend()
    results = run_domain_generation(config, items, backend=backend)

    assert all(isinstance(result, DomainItemResult) for result in results)
    assert [result.item_id for result in results] == [item.id for item in items]
    assert [result.prompt_tokens for result in results] == [20, 21]
    assert all(prompt.endswith("<think>\n\n</think>\n\n") for prompt in backend.prompts)


def test_gpu_preflight_accepts_idle_card_and_rejects_busy_or_hot_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = "4, 1, 0, 30\n5, 2048, 0, 40\n6, 1, 0, 81\n"
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, inventory, ""),
    )

    assert preflight_baseline_gpu(4) == BaselineGpuPreflight(
        physical_index=4,
        memory_used_mib=1,
        utilization_percent=0,
        temperature_c=30,
    )
    with pytest.raises(BaselinePreflightError, match="busy"):
        preflight_baseline_gpu(5)
    with pytest.raises(BaselinePreflightError, match="too hot"):
        preflight_baseline_gpu(6)
    with pytest.raises(BaselinePreflightError, match="does not exist"):
        preflight_baseline_gpu(9)


def test_gpu_preflight_rejects_invalid_index_command_and_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BaselinePreflightError, match="non-negative"):
        preflight_baseline_gpu(-1)

    def command_failure(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr("tinyllm.evaluation.baseline_runtime.subprocess.run", command_failure)
    with pytest.raises(BaselinePreflightError, match="cannot inspect"):
        preflight_baseline_gpu(0)

    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_runtime.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "bad,row\n", ""),
    )
    with pytest.raises(BaselinePreflightError, match="invalid GPU inventory"):
        preflight_baseline_gpu(0)


def test_runtime_version_contract_accepts_exact_stack_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    fake_torch = SimpleNamespace(__version__=config.software.torch)
    fake_transformers = SimpleNamespace(__version__=config.software.transformers)
    expected = {
        "accelerate": config.software.accelerate,
        "datasets": config.software.datasets,
        "lm_eval": config.software.lm_eval,
        "safetensors": config.software.safetensors,
        "tokenizers": config.software.tokenizers,
        "transformers": config.software.transformers,
    }
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_runtime._installed_version", expected.__getitem__
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_runtime.importlib.import_module",
        lambda name: fake_torch if name == "torch" else fake_transformers,
    )

    runtime = load_baseline_runtime(config)
    assert runtime.torch.__version__ == config.software.torch
    assert runtime.transformers.__version__ == config.software.transformers

    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_runtime._installed_version", lambda _name: "0.0"
    )
    with pytest.raises(BaselinePreflightError, match="dependency mismatch"):
        load_baseline_runtime(config)


def test_transformers_domain_backend_enforces_template_and_token_accounting(
    tmp_path: Path,
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    model_path = (tmp_path / "model").resolve()
    model_path.mkdir()

    class FakeTensor:
        def __init__(self, values: list[list[int]]) -> None:
            self.values = values

        @property
        def shape(self) -> tuple[int, int]:
            return (len(self.values), len(self.values[0]))

        def sum(self, *, dim: int) -> list[int]:
            assert dim == 1
            return [sum(row) for row in self.values]

        def to(self, _device: str) -> FakeTensor:
            return self

        def __getitem__(self, key: tuple[slice, slice]) -> FakeTensor:
            rows, columns = key
            return FakeTensor([row[columns] for row in self.values[rows]])

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[list[int]]:
            return self.values

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = (99,)
        padding_side = "right"

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return (
                "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )

        def __call__(self, batch: list[str], **_kwargs: object) -> dict[str, FakeTensor]:
            assert len(batch) == 2
            return {
                "input_ids": FakeTensor([[10, 11], [20, 21]]),
                "attention_mask": FakeTensor([[1, 1], [1, 1]]),
            }

        def decode(self, values: list[int], **_kwargs: object) -> str:
            return ",".join(str(value) for value in values)

    class FakeModel:
        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

        def eval(self) -> FakeModel:
            return self

        def generate(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["do_sample"] is False
            assert kwargs["eos_token_id"] == [99]
            return SimpleNamespace(sequences=FakeTensor([[10, 11, 99, 5], [20, 21, 6, 7]]))

    tokenizer = FakeTokenizer()
    model = FakeModel()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model),
    )
    fake_cuda = SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=fake_cuda,
        inference_mode=nullcontext,
    )
    runtime = cast(Any, SimpleNamespace(torch=fake_torch, transformers=fake_transformers))

    backend = TransformersDomainBackend(
        runtime=runtime,
        config=config,
        model_path=model_path,
        device="cpu",
    )
    assert tokenizer.padding_side == "left"
    responses = backend.generate(
        ("first", "second"),
        batch_size=2,
        max_sequence_length=2,
        max_new_tokens=2,
    )
    assert responses == (
        GeneratedResponse(response="99", prompt_tokens=2, generated_tokens=1, finish_reason="eos"),
        GeneratedResponse(
            response="6,7", prompt_tokens=2, generated_tokens=2, finish_reason="length"
        ),
    )
    with pytest.raises(BaselineContractError, match="exceeds"):
        backend.generate(
            ("first", "second"),
            batch_size=2,
            max_sequence_length=1,
            max_new_tokens=2,
        )


def test_transformers_domain_backend_preflight_and_seed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_baseline_config(SMOKE_CONFIG)
    missing = (tmp_path / "missing").resolve()
    runtime = cast(Any, SimpleNamespace(torch=SimpleNamespace(), transformers=SimpleNamespace()))
    with pytest.raises(BaselinePreflightError, match="existing absolute"):
        TransformersDomainBackend(
            runtime=runtime,
            config=config,
            model_path=missing,
            device="cpu",
        )

    calls: list[tuple[str, int]] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda value: calls.append(("cuda", value)),
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        manual_seed=lambda value: calls.append(("torch", value)),
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_inference.random.seed",
        lambda value: calls.append(("python", value)),
    )
    monkeypatch.setattr(
        "tinyllm.evaluation.baseline_inference.np.random.seed",
        lambda value: calls.append(("numpy", value)),
    )
    seed_baseline(config, cast(Any, SimpleNamespace(torch=fake_torch)))
    assert calls == [("python", 42), ("numpy", 42), ("torch", 42), ("cuda", 42)]
