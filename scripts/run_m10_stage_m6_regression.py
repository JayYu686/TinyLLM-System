#!/usr/bin/env python3
"""Run the frozen M6 general-regression suite for an M10 stage subject."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tinyllm.deployment import resolve_m10_stage_evaluation_subject
from tinyllm.evaluation import load_m6_release_config, run_m6_general_pass
from tinyllm.evaluation.baseline_runtime import preflight_baseline_gpu
from tinyllm.schemas import canonical_config_hash


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

    resolved = resolve_m10_stage_evaluation_subject(args.artifact_root, args.subject)
    release = load_m6_release_config(args.release_config)
    preflight_baseline_gpu(args.gpu_index)
    previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        result = run_m6_general_pass(
            release_config_path=args.release_config,
            artifact_root=args.artifact_root,
            model_dir=resolved.model_dir,
            tokenizer_dir=resolved.tokenizer_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
            physical_gpu_index=args.gpu_index,
            model_identity=resolved.model,
            expected_config_sha256=canonical_config_hash(release),
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
