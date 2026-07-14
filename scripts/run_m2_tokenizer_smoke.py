#!/usr/bin/env python3
"""Reproduce the public M2.3a tokenizer smoke from pinned local tokenizer files."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import tokenizers  # type: ignore[import-untyped]

from tinyllm.data import (
    TokenizersBackend,
    import_commitpackft,
    import_oasst1,
    load_m2_processing_config,
    load_m2_tokenization_config,
    process_imported_samples,
    tokenize_processed_samples,
)


def _fixture(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _sequence_hash(values: tuple[int, ...]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_smoke(project_root: Path, tokenizer_dir: Path) -> dict[str, object]:
    """Return machine-readable evidence without exposing source text or private paths."""

    fixture_root = project_root / "tests" / "fixtures" / "data"
    oasst = import_oasst1(_fixture(fixture_root / "oasst1.synthetic.json"))
    commitpack = import_commitpackft(_fixture(fixture_root / "commitpackft.synthetic.json"))
    processing = process_imported_samples(
        (*oasst.samples, *commitpack.samples),
        config=load_m2_processing_config(project_root / "configs/data/m2_processing.yaml"),
    )
    config = load_m2_tokenization_config(project_root / "configs/data/m2_tokenization.yaml")
    backend = TokenizersBackend.from_files(
        tokenizer_dir / config.tokenizer.tokenizer_file,
        tokenizer_dir / config.tokenizer.tokenizer_config_file,
        config.tokenizer,
    )
    result = tokenize_processed_samples(processing.samples, backend=backend, config=config)
    samples = [
        {
            "id": sample.id,
            "source": sample.source,
            "split": sample.split,
            "token_count": sample.token_count,
            "supervised_token_count": sample.supervised_token_count,
            "rendered_sha256": sample.rendered_sha256,
            "input_ids_sha256": _sequence_hash(sample.input_ids),
            "labels_sha256": _sequence_hash(sample.labels),
            "supervised_eos_count": sample.labels.count(config.tokenizer.eos_token_id),
            "input_ids": list(sample.input_ids),
            "labels": list(sample.labels),
        }
        for sample in result.samples
    ]
    return {
        "status": "pass",
        "backend": {
            "name": "tokenizers",
            "version": tokenizers.__version__,
            "vocab_size": backend.vocab_size,
        },
        "tokenizer": config.tokenizer.to_dict(),
        "template": config.template.to_dict(),
        "input_processing_sha256": processing.manifest.output_sha256,
        "accepted_samples": len(result.samples),
        "rejected_samples": len(result.rejected),
        "samples": samples,
        "rejected": [record.to_dict() for record in result.rejected],
    }


def main() -> int:
    """Run the smoke and print stable JSON to standard output."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_smoke(project_root, args.tokenizer_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
