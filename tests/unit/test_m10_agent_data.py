from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from scripts.profile_m10_agent_sources import build_parser
from tinyllm.data.m10_agent import (
    M10AgentDataError,
    _profile_hermes,
    _profile_toolace,
    _verify_artifact,
    load_m10_agent_data_config,
    m10_agent_data_config_sha256,
    parse_toolace_calls,
)
from tinyllm.data.m10_agent_schema import (
    M10AgentArtifactSpec,
    M10AgentDataConfig,
    M10AgentSourceSpec,
    M10ExternalSourceProfileReport,
)

CONFIG = Path("configs/data/m10_agent.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_source(path: Path, *, source_id: str) -> M10AgentSourceSpec:
    dataset_id = "lockon/ToolACE" if source_id == "toolace" else "test/hermes"
    return M10AgentSourceSpec.model_validate(
        {
            "source_id": source_id,
            "source_kind": "external",
            "dataset_id": dataset_id,
            "revision": "a" * 40,
            "license": "apache-2.0",
            "mixture_basis_points": 3000 if source_id == "toolace" else 2000,
            "readiness": "ready",
            "redistributable": False,
            "artifacts": [
                {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            ],
            "license_evidence_sha256": "b" * 64,
        }
    )


def _toolace_system(tools: list[dict[str, object]]) -> str:
    return (
        "Fixture policy.\nHere is a list of functions in JSON format that you can invoke:\n"
        + json.dumps(tools)
        + ". \nShould you decide to return the function call(s).\n"
    )


def _toolace_tool(name: str) -> dict[str, object]:
    return {
        "name": name,
        "description": "fixture",
        "parameters": {
            "type": "dict",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "required": None,
    }


def _hermes_tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "fixture",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


def test_m10_config_freezes_mix_masks_and_sealed_release() -> None:
    config = load_m10_agent_data_config(CONFIG)

    assert config.status == "preregistered"
    assert config.training_permitted is False
    assert [source.mixture_basis_points for source in config.sources] == [
        3000,
        2000,
        2000,
        2000,
        1000,
    ]
    assert config.language_target.english_basis_points == 7000
    assert config.supervision.mask_tool_results is True
    assert config.supervision.synthetic_cot_teacher_data is False
    assert config.contamination.targets[1].target_id == "m9_release"
    assert config.contamination.targets[1].visibility == "sealed_private"
    assert len(m10_agent_data_config_sha256(config)) == 64


def test_m10_config_rejects_unknown_fields_and_premature_ready_state(tmp_path: Path) -> None:
    decoded: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decoded["unexpected"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(decoded), encoding="utf-8")

    with pytest.raises(M10AgentDataError, match="violates its schema"):
        load_m10_agent_data_config(invalid)

    decoded.pop("unexpected")
    authored = decoded["sources"][2]
    authored["readiness"] = "ready"
    authored["content_sha256"] = "c" * 64
    authored["manifest_sha256"] = "d" * 64
    with pytest.raises(ValidationError, match="training is permitted only"):
        M10AgentDataConfig.model_validate(decoded)


def test_parse_toolace_calls_handles_real_name_and_nested_literal_shapes() -> None:
    calls = parse_toolace_calls(
        '[User Feed (Video Posts) V2(query="x,y"), '
        'Get Agent\'s Active Listings($top=10, filters={"active": true})]'
    )

    assert calls == (
        ("User Feed (Video Posts) V2", {"query": "x,y"}),
        (
            "Get Agent's Active Listings",
            {"$top": 10, "filters": {"active": True}},
        ),
    )


def test_parse_toolace_calls_rejects_executable_argument_and_bad_delimiters() -> None:
    with pytest.raises(M10AgentDataError, match="safe literal"):
        parse_toolace_calls("[unsafe(value=__import__('os').system('id'))]")
    with pytest.raises(M10AgentDataError, match="unbalanced"):
        parse_toolace_calls("[broken(query=[1, 2)]")


def test_toolace_profile_is_content_free_and_rejects_name_collision(tmp_path: Path) -> None:
    tools = [_toolace_tool("read log")]
    collision_tools = [_toolace_tool("read log"), _toolace_tool("read_log")]
    rows = [
        {
            "system": _toolace_system(tools),
            "conversations": [
                {"from": "user", "value": "fixture question"},
                {"from": "assistant", "value": '[read log(query="error")]'},
                {"from": "tool", "value": '{"result":"fixture"}'},
                {"from": "assistant", "value": "fixture answer"},
            ],
        },
        {
            "system": _toolace_system(tools),
            "conversations": [
                {"from": "user", "value": "no tool fixture"},
                {"from": "assistant", "value": "clarification required"},
            ],
        },
        {
            "system": _toolace_system(collision_tools),
            "conversations": [
                {"from": "user", "value": "collision fixture"},
                {"from": "assistant", "value": '[read log(query="x")]'},
            ],
        },
    ]
    artifact = tmp_path / "toolace.json"
    artifact.write_text(json.dumps(rows), encoding="utf-8")

    profile = _profile_toolace(_external_source(artifact, source_id="toolace"), artifact)

    assert profile.source_rows == 3
    assert profile.accepted_shape_rows == 2
    assert profile.rejected_shape_rows == 1
    assert profile.tool_name_collision_rows == 1
    assert profile.parsed_tool_calls == 2
    assert profile.dict_to_object_normalizations == 4
    assert "fixture question" not in profile.model_dump_json()


def test_hermes_profile_accepts_json_and_safe_legacy_literal_calls(tmp_path: Path) -> None:
    tool = _hermes_tool("inspect_config")
    rows = [
        {
            "category": "fixture",
            "subcategory": "json",
            "task": "call",
            "id": "one",
            "tools": json.dumps([tool]),
            "conversations": [
                {"from": "system", "value": "fixture system"},
                {"from": "human", "value": "fixture user"},
                {
                    "from": "gpt",
                    "value": (
                        '<tool_call>{"name":"inspect_config","arguments":{"query":"x"}}</tool_call>'
                    ),
                },
                {"from": "tool", "value": "<tool_response>{}</tool_response>"},
                {"from": "gpt", "value": "fixture final"},
            ],
        },
        {
            "category": "fixture",
            "subcategory": "legacy",
            "task": "call",
            "id": "two",
            "tools": json.dumps([tool]),
            "conversations": [
                {"from": "system", "value": "fixture system"},
                {"from": "human", "value": "fixture user"},
                {
                    "from": "gpt",
                    "value": (
                        r"<tool_call>\n{'arguments': {'query': 'x', "
                        r"'name': 'inspect_config'}}\n</tool_call>"
                    ),
                },
            ],
        },
        {
            "category": "fixture",
            "subcategory": "bad-role",
            "task": "reject",
            "id": "three",
            "tools": json.dumps([tool]),
            "conversations": [
                {"from": "system", "value": "fixture system"},
                {"from": "human", "value": "fixture user"},
                {"from": "tool", "value": "fixture result"},
                {"from": "gpt", "value": "fixture final"},
            ],
        },
    ]
    artifact = tmp_path / "hermes.json"
    artifact.write_text(json.dumps(rows), encoding="utf-8")

    profile = _profile_hermes(
        _external_source(artifact, source_id="hermes_function_calling"),
        artifact,
    )

    assert profile.accepted_shape_rows == 2
    assert profile.rejected_shape_rows == 1
    assert profile.parsed_tool_calls == 2
    assert profile.malformed_tool_calls == 0
    assert profile.rejection_counts[0].reason == "invalid_role_path"


def test_artifact_verification_fails_closed_on_hash_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "data.json"
    artifact.write_text("[]", encoding="utf-8")
    source = _external_source(artifact, source_id="toolace")
    artifact.write_text("[{}]", encoding="utf-8")

    with pytest.raises(M10AgentDataError, match="size differs"):
        _verify_artifact(source, artifact)


def test_profile_report_rejects_totals_and_exposes_no_source_content(tmp_path: Path) -> None:
    tools = [_toolace_tool("query")]
    toolace_path = tmp_path / "toolace.json"
    toolace_path.write_text(
        json.dumps(
            [
                {
                    "system": _toolace_system(tools),
                    "conversations": [
                        {"from": "user", "value": "private fixture"},
                        {"from": "assistant", "value": "clarify"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    profile = _profile_toolace(_external_source(toolace_path, source_id="toolace"), toolace_path)

    with pytest.raises(ValidationError, match="frozen order"):
        M10ExternalSourceProfileReport(
            profile_version="m10-external-source-profile-v1",
            data_config_sha256="a" * 64,
            profiles=(profile, profile),
            total_source_rows=2,
            total_accepted_shape_rows=2,
            total_rejected_shape_rows=0,
        )


def test_m10_profile_cli_requires_both_private_artifacts() -> None:
    args = build_parser().parse_args(
        [
            "--toolace-artifact",
            "/private/toolace.json",
            "--hermes-artifact",
            "/private/hermes.json",
            "--output",
            "/private/profile.json",
        ]
    )

    assert args.config == CONFIG
    assert args.toolace_artifact == Path("/private/toolace.json")
    assert args.hermes_artifact == Path("/private/hermes.json")


def test_external_artifact_schema_rejects_paths() -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        M10AgentArtifactSpec(filename="../data.json", size_bytes=1, sha256="a" * 64)
