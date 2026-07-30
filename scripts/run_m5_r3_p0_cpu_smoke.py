#!/usr/bin/env python3
"""Build deterministic content-free CPU contract evidence for M5.2-R3-P0/P0-R1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from tinyllm.data import (
    OffsetTokenizer,
    TeacherGenerationRecord,
    TokenEncoding,
    build_m5_r3_p0_dataset,
    generate_m5_r3_p0_tasks,
    generate_reasoning_dev_tasks,
    generate_reasoning_pilot_tasks,
    load_m5_r3_p0_config,
    load_m5_reasoning_data_config,
    m5_r3_p0_config_sha256,
)


class _FixtureTokenizer:
    """Minimal deterministic tokenizer used only to exercise P0 contract logic."""

    @property
    def vocab_size(self) -> int:
        return 200_000

    def token_to_id(self, token: str) -> int | None:
        del token
        return 1

    def encode(self, text: str) -> TokenEncoding:
        token_count = max(1, len(text.split()))
        return TokenEncoding(
            ids=tuple(range(token_count)),
            offsets=tuple((index, index + 1) for index in range(token_count)),
        )


def build_smoke_payload(
    config_path: Path,
    reasoning_config_path: Path,
) -> dict[str, object]:
    """Exercise P0/P0-R1 generation, contamination, selection, and family Gates on CPU."""

    config = load_m5_r3_p0_config(config_path)
    reasoning_config = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_m5_r3_p0_tasks(config)
    dev_tasks = generate_reasoning_dev_tasks(reasoning_config)
    historical_tasks = generate_reasoning_pilot_tasks(
        seed=reasoning_config.pilot_task_seed,
        tasks_per_family=20,
        task_contract_version=reasoning_config.task_contract_version,
    )
    generations: list[TeacherGenerationRecord] = []
    for task in tasks:
        reasoning = f"synthetic contract trace for {task.id}"
        raw_output = f"<think>{reasoning}</think>\n\n{task.expected_answer_json}"
        generations.append(
            TeacherGenerationRecord(
                generation_id=f"{task.id}:candidate-0",
                task_id=task.id,
                candidate_index=0,
                seed=config.sampling.base_seed,
                prompt_sha256=task.prompt_sha256,
                status="succeeded",
                finish_reason="stop",
                raw_output=raw_output,
                raw_output_sha256=hashlib.sha256(raw_output.encode()).hexdigest(),
                observed_token_count=64,
            )
        )
    build = build_m5_r3_p0_dataset(
        tasks,
        generations,
        config=config,
        reasoning_config=reasoning_config,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
        tokenizer=cast(OffsetTokenizer, _FixtureTokenizer()),
    )
    return {
        "schema_version": "1.0",
        "evidence_kind": "synthetic_cpu_contract_smoke",
        "model_generated": False,
        "quality_metric": False,
        "gpu_used": False,
        "config_sha256": m5_r3_p0_config_sha256(config),
        "task_set_sha256": build.task_set_sha256,
        "samples_sha256": build.samples_sha256,
        "input_tasks": len(build.tasks),
        "accepted_samples": len(build.samples),
        "contamination": build.contamination.to_dict(),
        "family_results": [item.to_dict() for item in build.family_results],
        "rejection_counts": build.rejection_counts,
    }


def main() -> int:
    """Write deterministic path-free CPU contract evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_p0.yaml"),
    )
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_smoke_payload(args.config, args.reasoning_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
