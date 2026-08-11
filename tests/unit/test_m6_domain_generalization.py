from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_m6_domain_generalization import generate_domain_generalization_tasks


def test_r4_training_inventory_covers_all_families_without_frozen_prompt_overlap() -> None:
    tasks = generate_domain_generalization_tasks()
    frozen_prompts = {
        str(json.loads(line)["prompt_messages"][0]["content"])
        for path in sorted(Path("evals/domain").glob("v*/items.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    }

    assert len(tasks) == len({task.task_id for task in tasks}) == 900
    assert Counter(task.category for task in tasks) == {
        "config": 120,
        "json": 120,
        "linux": 135,
        "logs": 135,
        "python": 150,
        "refusal": 120,
        "short_code": 120,
    }
    assert not {task.prompt for task in tasks}.intersection(frozen_prompts)
    assert all(task.reasoning and task.final_answer for task in tasks)
