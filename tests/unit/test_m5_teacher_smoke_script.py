from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_m5_teacher_smoke import _verify_model_directory, build_parser

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def test_teacher_smoke_parser_requires_explicit_hardware_and_artifacts() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/data/m5_reasoning.yaml",
            "--model-dir",
            f"/models/{REVISION}",
            "--gpu-index",
            "9",
            "--raw-output",
            "/private/raw.json",
            "--public-output",
            "reports/m5/raw/teacher.json",
        ]
    )

    assert args.gpu_index == 9
    assert args.timeout_seconds == 900
    assert args.worker is False


def test_teacher_snapshot_check_rejects_wrong_revision_and_missing_files(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong-revision"
    wrong.mkdir()
    with pytest.raises(RuntimeError, match="pinned revision"):
        _verify_model_directory(wrong, REVISION)

    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        _verify_model_directory(snapshot, REVISION)


def test_teacher_snapshot_check_requires_qwen3_8b_gqa_identity(tmp_path: Path) -> None:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    for name in (
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / name).write_text("{}", encoding="utf-8")
    config_path = snapshot / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    _verify_model_directory(snapshot, REVISION)
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="GQA identity"):
        _verify_model_directory(snapshot, REVISION)
