#!/usr/bin/env python3
"""Build public synthetic M5.1 CPU evidence without invoking a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import (
    build_reasoning_dataset,
    build_reasoning_task_manifest,
    build_synthetic_teacher_generations,
    generate_reasoning_dev_tasks,
    generate_reasoning_pilot_tasks,
    load_m5_reasoning_data_config,
    summarize_reasoning_build,
)


def build_smoke_payload(config_path: Path) -> dict[str, object]:
    """Run deterministic Dev and synthetic Pilot construction in memory."""

    config = load_m5_reasoning_data_config(config_path)
    dev_tasks = generate_reasoning_dev_tasks(config)
    dev_manifest = build_reasoning_task_manifest(dev_tasks, config=config)
    pilot_tasks = generate_reasoning_pilot_tasks(
        seed=config.pilot_task_seed,
        tasks_per_family=10,
    )
    generations = build_synthetic_teacher_generations(pilot_tasks, config=config)
    build = build_reasoning_dataset(
        pilot_tasks,
        generations,
        config=config,
        dev_tasks=dev_tasks,
    )
    return {
        "evidence_kind": "synthetic_cpu_contract_smoke",
        "model_generated": False,
        "quality_metric": False,
        "dev_manifest": dev_manifest.to_dict(),
        "contamination_report": build.contamination.to_dict(),
        "pilot_smoke": summarize_reasoning_build(build),
    }


def main() -> int:
    """Write deterministic path-free evidence to the explicitly selected location."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_reasoning.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_smoke_payload(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
