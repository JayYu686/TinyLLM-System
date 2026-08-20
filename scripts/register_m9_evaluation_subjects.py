#!/usr/bin/env python3
"""Register hash-verified Qwen3-8B M9 baseline subjects outside Production."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from tinyllm.deployment import (
    M9EvaluationSubjectRecord,
    effective_artifact_sha256,
    evaluation_artifact_sha256,
    evaluation_subject_id,
    publish_evaluation_subject,
)
from tinyllm.evaluation import M6ModelIdentity

REVISION: Literal["b968826d9c46dd6066d109eabc6255188de91218"] = (
    "b968826d9c46dd6066d109eabc6255188de91218"
)
MODEL_FILES = tuple(
    sorted(
        (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        )
    )
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_historical_evidence(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("historical M5 evidence is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("historical M5 evidence must be a JSON object")
    model = value.get("model")
    training = value.get("training")
    limitations = value.get("limitations")
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("historical M5 evidence lacks model or training lineage")
    expected_model = {
        "repository": "Qwen/Qwen3-8B",
        "revision": REVISION,
        "attention_architecture": "gqa",
        "adaptation": "lora",
        "total_parameters": 8_234_382_336,
    }
    if any(model.get(key) != expected for key, expected in expected_model.items()):
        raise ValueError("historical M5 model identity differs from the frozen comparison")
    expected_training = {
        "run_id": "20260731T125617Z-m5-formal-qwen3-8b-lora-cc363170-e922",
        "config_sha256": "cc363170bda3d7637124664a9b742c16dff6eea6a2030bdae98b76ea35efc85f",
        "dataset_version": "m5-dual-sft-v1-b5b9e839",
        "dataset_manifest_sha256": (
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        "supervised_tokens": 10_000_000,
        "lora_rank": 16,
    }
    if any(training.get(key) != expected for key, expected in expected_training.items()):
        raise ValueError("historical M5 training lineage differs from the frozen comparison")
    if not isinstance(limitations, dict) or any(
        limitations.get(key) is not False
        for key in ("candidate_status_claimed", "production_status_claimed")
    ):
        raise ValueError("historical M5 evidence does not preserve its non-promotion boundary")
    return value


def _record(
    *,
    kind: Literal["base", "historical_lora"],
    model: M6ModelIdentity,
    model_dir: Path,
    base_sha256: str,
    tokenizer_sha256: str,
    source_sha256: str,
    adapter_dir: Path | None = None,
    adapter_sha256: str | None = None,
) -> M9EvaluationSubjectRecord:
    if kind not in {"base", "historical_lora"}:
        raise ValueError("unsupported M9 evaluation subject kind")
    subject_id = evaluation_subject_id(
        kind=kind,
        model=model,
        base_model_artifact_sha256=base_sha256,
        tokenizer_artifact_sha256=tokenizer_sha256,
        adapter_artifact_sha256=adapter_sha256,
        source_evidence_sha256=source_sha256,
    )
    return M9EvaluationSubjectRecord(
        subject_id=subject_id,
        kind=kind,
        created_at=datetime.now(UTC),
        model=model,
        model_dir=model_dir,
        model_files=MODEL_FILES,
        base_model_artifact_sha256=base_sha256,
        tokenizer_dir=model_dir,
        tokenizer_files=TOKENIZER_FILES,
        tokenizer_artifact_sha256=tokenizer_sha256,
        adapter_dir=adapter_dir,
        adapter_files=ADAPTER_FILES if adapter_dir is not None else (),
        adapter_artifact_sha256=adapter_sha256,
        effective_artifact_sha256=effective_artifact_sha256(base_sha256, adapter_sha256),
        source_evidence_sha256=source_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/yujielun/tinyllm"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--historical-evidence", type=Path, required=True)
    args = parser.parse_args()

    if not args.artifact_root.is_absolute():
        raise ValueError("Artifact root must be absolute")
    model_dir = args.model_dir.resolve(strict=True)
    adapter_dir = args.adapter_dir.resolve(strict=True)
    evidence_path = args.historical_evidence.resolve(strict=True)
    _load_historical_evidence(evidence_path)

    base_sha256 = evaluation_artifact_sha256(model_dir, MODEL_FILES)
    tokenizer_sha256 = evaluation_artifact_sha256(model_dir, TOKENIZER_FILES)
    adapter_sha256 = evaluation_artifact_sha256(adapter_dir, ADAPTER_FILES)
    base_source_sha256 = _sha256_file(model_dir / "README.md")
    historical_source_sha256 = _sha256_file(evidence_path)

    base_model = M6ModelIdentity(
        role="base",
        repository="Qwen/Qwen3-8B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="base",
        model_artifact_sha256=base_sha256,
        model_parameters=8_234_382_336,
    )
    historical_model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-8B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="lora",
        model_artifact_sha256=effective_artifact_sha256(base_sha256, adapter_sha256),
        model_parameters=8_234_382_336,
        training_run_id="20260731T125617Z-m5-formal-qwen3-8b-lora-cc363170-e922",
        training_checkpoint_id="checkpoint-tokens-0010000000",
        training_tokens=10_000_000,
        training_config_sha256=("cc363170bda3d7637124664a9b742c16dff6eea6a2030bdae98b76ea35efc85f"),
        dataset_version="m5-dual-sft-v1-b5b9e839",
        dataset_manifest_sha256=(
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        adapter_sha256=adapter_sha256,
    )
    records = (
        _record(
            kind="base",
            model=base_model,
            model_dir=model_dir,
            base_sha256=base_sha256,
            tokenizer_sha256=tokenizer_sha256,
            source_sha256=base_source_sha256,
        ),
        _record(
            kind="historical_lora",
            model=historical_model,
            model_dir=model_dir,
            base_sha256=base_sha256,
            tokenizer_sha256=tokenizer_sha256,
            source_sha256=historical_source_sha256,
            adapter_dir=adapter_dir,
            adapter_sha256=adapter_sha256,
        ),
    )
    published = []
    for record in records:
        stored, record_sha256 = publish_evaluation_subject(args.artifact_root, record)
        published.append(
            {
                "subject_id": stored.subject_id,
                "kind": stored.kind,
                "evaluation_subject_sha256": record_sha256,
                "effective_artifact_sha256": stored.effective_artifact_sha256,
                "production_eligible": stored.production_eligible,
            }
        )
    print(json.dumps({"schema_version": "1.0", "subjects": published}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
