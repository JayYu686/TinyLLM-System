#!/usr/bin/env python3
"""Export deterministic public JSON Schemas from Pydantic models."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from pydantic import BaseModel

from tinyllm.data import (
    CommitPackFTImportConfig,
    DataImportManifest,
    DataProcessingManifest,
    ImportedSample,
    M2ProcessingConfig,
    OASST1ImportConfig,
    PipelineRejectedRecord,
    ProcessedSample,
    RejectedRecord,
    SamplerState,
)
from tinyllm.schemas.checkpoint import CheckpointCommitMarker, CheckpointManifest
from tinyllm.schemas.resume import ResumeResult
from tinyllm.schemas.run import RunManifest
from tinyllm.schemas.training_run import TrainingRunResult
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.metrics import TrainerState, TrainingStepMetrics

SCHEMAS: dict[str, type[BaseModel]] = {
    "checkpoint-manifest-v1.schema.json": CheckpointManifest,
    "checkpoint-commit-marker-v1.schema.json": CheckpointCommitMarker,
    "commitpackft-import-config-v1.schema.json": CommitPackFTImportConfig,
    "data-import-manifest-v1.schema.json": DataImportManifest,
    "data-processing-manifest-v1.schema.json": DataProcessingManifest,
    "imported-sample-v1.schema.json": ImportedSample,
    "m2-processing-config-v1.schema.json": M2ProcessingConfig,
    "m1-training-config-v1.schema.json": M1TrainingConfig,
    "oasst1-import-config-v1.schema.json": OASST1ImportConfig,
    "pipeline-rejected-record-v1.schema.json": PipelineRejectedRecord,
    "processed-sample-v1.schema.json": ProcessedSample,
    "rejected-record-v1.schema.json": RejectedRecord,
    "run-manifest-v1.schema.json": RunManifest,
    "resume-result-v1.schema.json": ResumeResult,
    "sampler-state-v1.schema.json": SamplerState,
    "trainer-state-v1.schema.json": TrainerState,
    "training-step-metrics-v1.schema.json": TrainingStepMetrics,
    "training-run-result-v1.schema.json": TrainingRunResult,
}


def render_schema(model: type[BaseModel]) -> str:
    """Render one schema using canonical formatting."""

    return json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write schemas, or verify that committed snapshots are current."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when a committed schema differs instead of rewriting it.",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "schemas"
    if not args.check:
        output_dir.mkdir(exist_ok=True)
    stale: list[str] = []
    for filename, model in SCHEMAS.items():
        path = output_dir / filename
        rendered = render_schema(model)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                stale.append(filename)
        else:
            path.write_text(rendered, encoding="utf-8")
    if stale:
        parser.error(
            "stale schema snapshots: " + ", ".join(stale) + "; run scripts/export_schemas.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
