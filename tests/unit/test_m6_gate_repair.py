from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tinyllm.data import generate_m6_gate_repair_tasks


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
