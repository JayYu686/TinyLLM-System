from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tinyllm.data.m10_canonical import (
    M10CanonicalImportError,
    _RowError,
    build_external_import_report,
    canonicalize_hermes_row,
    canonicalize_toolace_row,
    import_external_source,
    write_external_import,
)
from tinyllm.data.m10_canonical_schema import M10CanonicalTrainingSample

CONFIG = Path("configs/data/m10_agent.yaml")
TOOLACE_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
HERMES_REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"


def _toolace_tool(name: str = "Read Log") -> dict[str, object]:
    return {
        "name": name,
        "description": "Read a bounded log excerpt.",
        "parameters": {
            "type": "dict",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "required": None,
    }


def _toolace_system(tools: list[dict[str, object]]) -> str:
    return (
        "Use evidence.\nHere is a list of functions in JSON format that you can invoke:\n"
        + json.dumps(tools)
        + ". \nShould you decide to return the function call(s).\nAnswer safely."
    )


def _toolace_rows() -> list[object]:
    tool = _toolace_tool()
    return [
        {
            "system": _toolace_system([tool]),
            "conversations": [
                {"from": "user", "value": "Diagnose the failed run."},
                {"from": "assistant", "value": '[Read Log(path="run.log")]'},
                {"from": "tool", "value": '{"line":"CUDA OOM"}'},
                {"from": "assistant", "value": "The evidence indicates a CUDA OOM."},
            ],
        },
        {"bad": "shape"},
    ]


def _hermes_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "inspect_config",
            "description": "Inspect a registered config.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def _hermes_rows() -> list[object]:
    return [
        {
            "category": "fixture",
            "subcategory": "legacy",
            "task": "inspect",
            "id": "fixture-one",
            "tools": json.dumps([_hermes_tool()]),
            "conversations": [
                {"from": "system", "value": "Use the provided evidence."},
                {"from": "human", "value": "检查配置。"},
                {
                    "from": "gpt",
                    "value": (
                        r"<tool_call>\n{'name':'inspect_config','arguments':"
                        r"{'path':'config.yaml'}}\n</tool_call>"
                    ),
                },
                {
                    "from": "tool",
                    "value": '<tool_response>{"timeout":120}</tool_response>',
                },
                {"from": "gpt", "value": "配置中的超时是 120 秒。"},
            ],
        }
    ]


def _write_config(tmp_path: Path, toolace: Path, hermes: Path) -> Path:
    decoded: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for index, path in ((0, toolace), (1, hermes)):
        decoded["sources"][index]["artifacts"] = [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
    config = tmp_path / "m10.yaml"
    config.write_text(yaml.safe_dump(decoded, sort_keys=False), encoding="utf-8")
    return config


def test_toolace_canonicalization_normalizes_schema_calls_and_masks() -> None:
    sample = canonicalize_toolace_row(_toolace_rows()[0], revision=TOOLACE_REVISION, row_index=7)

    assert sample.source_record_id == "toolace-00007"
    assert sample.language == "en"
    assert sample.tools[0].name == "read_log"
    assert sample.tools[0].input_schema["type"] == "object"
    assert [message.role for message in sample.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert sample.messages[2].tool_calls[0].name == "read_log"
    assert sample.messages[3].tool_call_ids == (sample.messages[2].tool_calls[0].id,)
    assert all(message.supervised == (message.role == "assistant") for message in sample.messages)
    assert "Here is a list of functions" not in (sample.messages[0].content or "")


def test_hermes_canonicalization_accepts_safe_legacy_call_and_strips_result_tags() -> None:
    sample = canonicalize_hermes_row(_hermes_rows()[0], revision=HERMES_REVISION, row_index=3)

    assert sample.source_record_id == "hermes-00003"
    assert sample.language == "zh"
    assert sample.messages[2].tool_calls[0].arguments == {"path": "config.yaml"}
    assert sample.messages[3].content == '{"timeout":120}'
    assert sample.messages[-1].supervised is True


def test_canonicalization_rejects_unknown_call_and_visible_reasoning() -> None:
    unknown = _toolace_rows()[0]
    assert isinstance(unknown, dict)
    unknown["conversations"][1]["value"] = '[Unknown(path="run.log")]'
    with pytest.raises(_RowError) as unknown_error:
        canonicalize_toolace_row(unknown, revision=TOOLACE_REVISION, row_index=0)
    assert unknown_error.value.reason == "malformed_tool_call"

    visible = _hermes_rows()[0]
    assert isinstance(visible, dict)
    visible["conversations"][-1]["value"] = "<think>private</think> final"
    with pytest.raises(_RowError) as visible_error:
        canonicalize_hermes_row(visible, revision=HERMES_REVISION, row_index=0)
    assert visible_error.value.reason == "visible_reasoning"


def test_external_import_build_write_and_public_report_are_deterministic(
    tmp_path: Path,
) -> None:
    toolace_path = tmp_path / "toolace.json"
    hermes_path = tmp_path / "hermes.json"
    toolace_path.write_text(json.dumps(_toolace_rows()), encoding="utf-8")
    hermes_path.write_text(json.dumps(_hermes_rows()), encoding="utf-8")
    config = _write_config(tmp_path, toolace_path, hermes_path)

    toolace = import_external_source(
        config_path=config, source_id="toolace", artifact_path=toolace_path
    )
    hermes = import_external_source(
        config_path=config,
        source_id="hermes_function_calling",
        artifact_path=hermes_path,
    )

    assert toolace.manifest.accepted_rows == 1
    assert toolace.manifest.rejection_counts == {"invalid_row_shape": 1}
    assert hermes.manifest.accepted_rows == 1
    output = write_external_import(tmp_path / "out", toolace)
    assert write_external_import(tmp_path / "out", toolace) == output
    assert (output / "COMMITTED.json").is_file()
    report = build_external_import_report((toolace, hermes))
    assert report.status == "pass"
    assert report.total_source_rows == 3
    assert report.total_accepted_rows == 2
    assert "Diagnose the failed run" not in report.model_dump_json()

    manifest = output / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M10CanonicalImportError, match="already differs"):
        write_external_import(tmp_path / "out", toolace)


def test_canonical_sample_rejects_content_hash_drift() -> None:
    sample = canonicalize_toolace_row(_toolace_rows()[0], revision=TOOLACE_REVISION, row_index=0)
    payload = sample.to_dict()
    payload["content_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="sample SHA256"):
        M10CanonicalTrainingSample.model_validate(payload)
