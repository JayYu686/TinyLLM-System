from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tinyllm.data.m6_gate_repair as repair_module
import tinyllm.data.m6_gate_replay as replay_module
from tinyllm.data import (
    M5MixtureSequence,
    build_m6_gate_repair_mixture,
    generate_m6_gate_repair_tasks,
    open_m5_ablation_mixture,
)


def test_gate_repair_tasks_are_balanced_compact_and_eval_independent() -> None:
    tasks = generate_m6_gate_repair_tasks()
    eval_prompts = {
        str(json.loads(line)["prompt_messages"][0]["content"])
        for variant in ("v1", "v2", "v3")
        for line in Path(f"evals/domain/{variant}/items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }

    assert len(tasks) == 680
    assert len({task.task_id for task in tasks}) == 680
    assert Counter(task.kind for task in tasks) == {
        "refusal": 300,
        "python": 120,
        "json": 80,
        "linux": 60,
        "logs": 60,
        "short_code": 60,
    }
    assert Counter(task.language for task in tasks) == {"en": 476, "zh": 204}
    assert not {task.prompt for task in tasks}.intersection(eval_prompts)
    assert max(len(task.reasoning) for task in tasks) < 180


def test_gate_repair_refusals_state_limits_and_request_evidence() -> None:
    refusals = [task for task in generate_m6_gate_repair_tasks() if task.kind == "refusal"]

    assert len(refusals) == 300
    assert all(
        ("insufficient" in task.final_answer and "Please provide" in task.final_answer)
        if task.language == "en"
        else ("证据不足" in task.final_answer and "请提供" in task.final_answer)
        for task in refusals
    )


def _sequence(mode: int) -> M5MixtureSequence:
    return M5MixtureSequence(
        input_ids=(1,) * 1024,
        labels=(-100,) + (1,) * 1000 + (-100,) * 23,
        attention_mask=(1,) * 1001 + (0,) * 23,
        mode=mode,
    )


def test_gate_repair_builder_commits_reopens_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair_module, "_evaluation_prompts", lambda _: set())
    monkeypatch.setattr(
        repair_module,
        "_paired_sequences",
        lambda *_, **__: ((_sequence(0),), (_sequence(1),), 49),
    )
    monkeypatch.setattr(
        repair_module,
        "general_nonthinking_correction_sources",
        lambda **_: (_sequence(0),),
    )
    monkeypatch.setattr(
        repair_module,
        "open_registered_dataset",
        lambda **_: SimpleNamespace(manifest=SimpleNamespace(content_sha256="f" * 64)),
    )
    output_root = tmp_path / "output"
    manifest = build_m6_gate_repair_mixture(
        artifact_root=tmp_path / "artifacts",
        tokenizer_config_path=tmp_path / "tokenizer.yaml",
        model_dir=tmp_path / "model",
        project_root=tmp_path,
        output_root=output_root,
        build_seed=20260811,
    )
    destination = output_root / manifest.mixture_version
    reopened = open_m5_ablation_mixture(destination)
    repeated = build_m6_gate_repair_mixture(
        artifact_root=tmp_path / "artifacts",
        tokenizer_config_path=tmp_path / "tokenizer.yaml",
        model_dir=tmp_path / "model",
        project_root=tmp_path,
        output_root=output_root,
        build_seed=20260811,
    )

    assert reopened.manifest == manifest
    assert repeated == manifest
    assert manifest.nonthinking_supervised_tokens == 700_000
    assert manifest.thinking_supervised_tokens == 300_000


def test_gate_repair_pair_tokenization_builds_both_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(
        tokenizer_file="tokenizer.json", tokenizer_config_file="config.json"
    )
    monkeypatch.setattr(
        repair_module,
        "load_m2_tokenization_config",
        lambda _: SimpleNamespace(tokenizer=tokenizer),
    )
    monkeypatch.setattr(
        "tinyllm.data.m6_gate_repair.TokenizersBackend.from_files",
        lambda *_, **__: object(),
    )
    encoded = SimpleNamespace(input_ids=(1, 2, 3), labels=(-100, 2, 3))
    monkeypatch.setattr(
        repair_module,
        "tokenize_nonthinking_sft_messages",
        lambda *_, **__: encoded,
    )
    monkeypatch.setattr(
        repair_module,
        "tokenize_thinking_messages",
        lambda *_, **__: encoded,
    )

    nonthinking, thinking, maximum = repair_module._paired_sequences(
        (generate_m6_gate_repair_tasks()[0],),
        tokenizer_config_path=tmp_path / "tokenizer.yaml",
        model_dir=tmp_path,
    )

    assert nonthinking[0].mode == 0
    assert thinking[0].mode == 1
    assert maximum == 2


def test_gate_repair_evaluation_prompt_reader_fails_closed(tmp_path: Path) -> None:
    for variant in ("v1", "v2", "v3"):
        directory = tmp_path / f"evals/domain/{variant}"
        directory.mkdir(parents=True)
        (directory / "items.jsonl").write_text(
            json.dumps({"prompt_messages": [{"content": f"prompt-{variant}"}]}) + "\n",
            encoding="utf-8",
        )

    assert repair_module._evaluation_prompts(tmp_path) == {
        "prompt-v1",
        "prompt-v2",
        "prompt-v3",
    }
    (tmp_path / "evals/domain/v2/items.jsonl").write_text("invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contamination"):
        repair_module._evaluation_prompts(tmp_path)


def test_gate_replay_materializes_verified_source_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = np.asarray([[1, 2], [3, 4]], dtype="<i4")
    labels = np.asarray([[-100, 2], [-100, 4]], dtype="<i4")
    masks = np.asarray([[1, 1], [1, 1]], dtype="u1")
    modes = np.asarray([0, 1], dtype="u1")
    np.savez(
        tmp_path / "sequences.npz",
        input_ids=input_ids,
        labels=labels,
        attention_masks=masks,
        modes=modes,
    )
    monkeypatch.setattr(
        replay_module,
        "open_m5_ablation_mixture",
        lambda _: SimpleNamespace(
            manifest=SimpleNamespace(
                artifact=SimpleNamespace(path="sequences.npz"),
                sequence_count=2,
            )
        ),
    )

    sequences = replay_module._source_sequences(tmp_path)

    assert [item.mode for item in sequences] == [0, 1]
    assert [item.supervised_tokens for item in sequences] == [1, 1]
