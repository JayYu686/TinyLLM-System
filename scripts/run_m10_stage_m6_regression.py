#!/usr/bin/env python3
"""Run the frozen M6 general-regression suite for an M10 stage subject."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tinyllm.evaluation.baseline_runtime import preflight_baseline_gpu
from tinyllm.training.m10_lora_general import run_m10_lora_general_pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/yujielun/tinyllm"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--release-config", type=Path, default=Path("configs/eval/m6_release_v7.yaml")
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    args = parser.parse_args()

    preflight_baseline_gpu(args.gpu_index)
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        result = run_m10_lora_general_pass(
            artifact_root=args.artifact_root,
            subject_id=args.subject,
            output_dir=args.output_dir,
            project_root=args.project_root,
            release_config_path=args.release_config,
            physical_gpu_index=args.gpu_index,
        )
    finally:
        if previous_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
