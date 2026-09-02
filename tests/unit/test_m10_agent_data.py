from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import tinyllm.data.m10_agent as m10_module
from scripts.profile_m10_agent_sources import build_parser
from tinyllm.data.m10_agent import (
    M10AgentDataError,
    _extract_toolace_tools,
    _load_json_array,
    _matching_parenthesis,
    _parse_hermes_calls,
    _profile_hermes,
    _profile_toolace,
    _role_path,
    _safe_tool_name,
    _sha256_file,
    _source,
    _split_top_level,
    _tool_name_collision,
    _toolace_expressions,
    _toolace_role_path_valid,
    _valid_hermes_tools,
    _valid_toolace_tools,
    _verify_artifact,
    load_m10_agent_data_config,
    m10_agent_data_config_sha256,
    parse_toolace_calls,
)
from tinyllm.data.m10_agent_schema import (
    M10AgentArtifactSpec,
    M10AgentContaminationPolicy,
    M10AgentDataConfig,
    M10AgentDedupPolicy,
    M10AgentSourceSpec,
    M10ExternalSourceProfile,
    M10ExternalSourceProfileReport,
    M10SourceRolePathCount,
)

CONFIG = Path("configs/data/m10_agent.yaml")
REPAIR_CONFIG = Path("configs/data/m10_agent_repair.yaml")
REPAIR_V3_CONFIG = Path("configs/data/m10_agent_repair_v3.yaml")
REPAIR_V4_CONFIG = Path("configs/data/m10_agent_repair_v4.yaml")


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


def test_m10_repair_config_increases_grounded_devops_supervision() -> None:
    config = load_m10_agent_data_config(REPAIR_CONFIG)

    assert config.config_version == "m10-agent-data-v2"
    assert config.status == "preregistered"
    assert config.training_permitted is False
    assert [source.mixture_basis_points for source in config.sources] == [
        2000,
        1000,
        4000,
        2000,
        1000,
    ]
    assert config.sources[2].revision == "m10-devops-training-v2-8461493c"


def test_m10_repair_v3_config_preserves_mix_with_new_authored_identity() -> None:
    config = load_m10_agent_data_config(REPAIR_V3_CONFIG)

    assert config.config_version == "m10-agent-data-v2"
    assert [source.mixture_basis_points for source in config.sources] == [
        2000,
        1000,
        4000,
        2000,
        1000,
    ]
    assert config.sources[2].revision == "m10-devops-training-v3-a5645bc5"
    assert config.sources[2].readiness == "pending_build"


def test_m10_repair_v4_config_focuses_approved_runtime_aligned_source() -> None:
    config = load_m10_agent_data_config(REPAIR_V4_CONFIG)

    assert config.config_version == "m10-agent-data-v3"
    assert [source.mixture_basis_points for source in config.sources] == [
        500,
        500,
        7000,
        1500,
        500,
    ]
    assert config.sources[2].revision == "m10-devops-training-v4-f13ae053"
    assert config.sources[2].readiness == "pending_build"


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


def test_config_loader_and_json_reader_fail_closed(tmp_path: Path) -> None:
    wrong_suffix = tmp_path / "config.json"
    wrong_suffix.write_text("{}", encoding="utf-8")
    with pytest.raises(M10AgentDataError, match="must use YAML"):
        load_m10_agent_data_config(wrong_suffix)
    with pytest.raises(M10AgentDataError, match="cannot be read"):
        load_m10_agent_data_config(tmp_path / "missing.yaml")

    malformed_yaml = tmp_path / "malformed.yaml"
    malformed_yaml.write_text("value: [", encoding="utf-8")
    with pytest.raises(M10AgentDataError, match="invalid YAML"):
        load_m10_agent_data_config(malformed_yaml)

    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("[", encoding="utf-8")
    with pytest.raises(M10AgentDataError, match="invalid JSON"):
        _load_json_array(malformed_json)
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("[]", encoding="utf-8")
    with pytest.raises(M10AgentDataError, match="non-empty JSON array"):
        _load_json_array(empty_json)
    with pytest.raises(M10AgentDataError, match="cannot be read"):
        _load_json_array(tmp_path / "absent.json")
    with pytest.raises(M10AgentDataError, match="cannot be read"):
        _sha256_file(tmp_path / "absent.bin")


def test_source_and_artifact_lookup_reject_ambiguity_and_drift(tmp_path: Path) -> None:
    config = load_m10_agent_data_config(CONFIG)
    with pytest.raises(M10AgentDataError, match="exactly once"):
        _source(config, "missing")

    authored = config.sources[2]
    with pytest.raises(M10AgentDataError, match="exactly one selected artifact"):
        _verify_artifact(authored, tmp_path / "missing.json")

    artifact = tmp_path / "data.json"
    artifact.write_text("aa", encoding="utf-8")
    source = _external_source(artifact, source_id="toolace")
    source = source.model_copy(
        update={"artifacts": (source.artifacts[0].model_copy(update={"sha256": "0" * 64}),)}
    )
    with pytest.raises(M10AgentDataError, match="SHA256 differs"):
        _verify_artifact(source, artifact)

    missing_source = _external_source(artifact, source_id="toolace")
    with pytest.raises(M10AgentDataError, match="cannot be inspected"):
        _verify_artifact(missing_source, tmp_path / "data.json.missing")


def test_role_and_tool_helpers_cover_invalid_shapes() -> None:
    assert _role_path(None, role_key="from") == "invalid"
    assert _role_path([{"value": "x"}], role_key="from") == "invalid"
    assert _role_path([{"from": "tool-1"}], role_key="from") == "invalid"
    assert _safe_tool_name("123 inspect") == "tool_123_inspect"

    assert _tool_name_collision(["bad"], wrapped=False) is False
    assert _tool_name_collision([{"name": 1}], wrapped=False) is False
    assert _tool_name_collision([{"name": "***"}], wrapped=False) is False
    assert _valid_hermes_tools({}) is False
    assert _valid_hermes_tools([{"bad": "shape"}]) is False
    assert _valid_hermes_tools([{"type": "tool", "function": {}}]) is False


def test_hermes_call_parser_counts_malformed_variants() -> None:
    tools = [_hermes_tool("inspect_config")]
    messages = [
        {"from": "user", "value": "skip"},
        {"from": "gpt", "value": "<tool_call>{bad}</tool_call>"},
        {"from": "gpt", "value": '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'},
        {"from": "gpt", "value": "<tool_call>{}</tool_call><tool_call>"},
    ]

    parsed, malformed = _parse_hermes_calls(messages, tools)

    assert parsed == 0
    assert malformed == 4


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("a)", "unbalanced delimiters"),
        ('a="unterminated', "unbalanced quotes"),
        ("a=1,,b=2", "empty component"),
    ],
)
def test_top_level_split_rejects_malformed_values(value: str, message: str) -> None:
    with pytest.raises(M10AgentDataError, match=message):
        _split_top_level(value)


def test_top_level_and_parenthesis_parsers_handle_escapes_and_failures() -> None:
    assert _split_top_level(r'query="a\\\"b", values=[1, 2]') == (
        r'query="a\\\"b"',
        "values=[1, 2]",
    )
    assert _matching_parenthesis('(query="x)"), tail', 0) == 11
    with pytest.raises(M10AgentDataError, match="unbalanced arguments"):
        _matching_parenthesis("(query='x'", 0)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("missing", "expression is malformed"),
        ("(x=1)", "name is invalid"),
        ("tool(x=1),", "trailing comma"),
    ],
)
def test_toolace_expression_parser_rejects_invalid_boundaries(value: str, message: str) -> None:
    with pytest.raises(M10AgentDataError, match=message):
        _toolace_expressions(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("tool(x=1)", "outer brackets"),
        ("[]", "cannot be empty"),
        ("[***(x=1)]", "name is invalid"),
        ("[tool(x=1); other(y=2)]", "comma separated"),
        ("[tool(bad)]", "argument name is invalid"),
        ("[tool(x=1, x=2)]", "repeats an argument"),
    ],
)
def test_toolace_call_parser_rejects_contract_violations(value: str, message: str) -> None:
    with pytest.raises(M10AgentDataError, match=message):
        parse_toolace_calls(value)


def test_safe_parser_covers_empty_call_and_escaped_parenthesis_boundaries() -> None:
    assert _toolace_expressions("") == ()
    assert parse_toolace_calls("[GetCompetitions()]") == (("GetCompetitions", {}),)
    escaped = r'(query="a\\\"b")'
    assert _matching_parenthesis(escaped, 0) == len(escaped) - 1
    assert (
        _toolace_role_path_valid(
            [{"from": "user"}, {"from": "assistant"}, {"from": "assistant"}],
            set(),
        )
        is False
    )


def test_toolace_schema_extraction_rejects_envelope_and_schema_variants() -> None:
    with pytest.raises(M10AgentDataError, match="lacks the frozen"):
        _extract_toolace_tools("missing")
    with pytest.raises(M10AgentDataError, match="invalid JSON"):
        _extract_toolace_tools(
            "Here is a list of functions in JSON format that you can invoke:\n"
            "[bad]. \nShould you decide to return the function call(s)."
        )


def test_toolace_schema_extraction_rejects_non_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m10_module, "_TOOLACE_SCHEMA_SUFFIX", "}. END")

    with pytest.raises(M10AgentDataError, match="must be an array"):
        _extract_toolace_tools(
            "Here is a list of functions in JSON format that you can invoke:\n{}. END"
        )


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        ({}, (False, 0, 0)),
        (["bad"], (False, 0, 0)),
        ([{"name": "x", "parameters": {"type": "array", "properties": {}}}], (False, 0, 0)),
        ([{"name": "x", "parameters": {"type": "object"}}], (False, 0, 0)),
        (
            [
                {
                    "name": "x",
                    "parameters": {"type": "object", "properties": {}},
                    "required": "bad",
                }
            ],
            (False, 0, 0),
        ),
        (
            [
                {
                    "name": "x",
                    "parameters": {"type": "object", "properties": {}},
                    "required": [],
                }
            ],
            (True, 0, 0),
        ),
    ],
)
def test_toolace_schema_validation_rejects_unsupported_shapes(
    tools: object, expected: tuple[bool, int, int]
) -> None:
    assert _valid_toolace_tools(tools) == expected


@pytest.mark.parametrize(
    ("messages", "calls"),
    [
        ([], set()),
        ([{"from": "assistant"}], set()),
        ([{"from": "user"}, {"from": "tool"}, {"from": "assistant"}], set()),
        ([{"from": "user"}, {"from": "user"}, {"from": "assistant"}], set()),
        ([{"from": "user"}, {"from": "unknown"}, {"from": "assistant"}], set()),
    ],
)
def test_toolace_role_path_validation_rejects_invalid_transitions(
    messages: list[dict[str, object]], calls: set[int]
) -> None:
    assert _toolace_role_path_valid(messages, calls) is False


def _minimal_profile(source_id: str) -> M10ExternalSourceProfile:
    return M10ExternalSourceProfile.model_validate(
        {
            "source_id": source_id,
            "dataset_id": f"fixture/{source_id}",
            "revision": "a" * 40,
            "artifacts": [{"filename": "data.json", "size_bytes": 1, "sha256": "b" * 64}],
            "source_rows": 1,
            "accepted_shape_rows": 1,
            "rejected_shape_rows": 0,
            "role_paths": [{"role_path": "user>assistant", "rows": 1}],
            "rejection_counts": [],
            "rows_with_tool_definitions": 1,
            "tool_definitions": 1,
            "tools_per_row_min": 1,
            "tools_per_row_max": 1,
            "tools_per_row_mean_milli": 1000,
            "tool_call_candidate_rows": 0,
            "no_tool_candidate_rows": 1,
            "parsed_tool_calls": 0,
            "malformed_tool_calls": 0,
            "dict_to_object_normalizations": 0,
            "null_required_normalizations": 0,
            "tool_name_collision_rows": 0,
        }
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"accepted_shape_rows": 0}, "acceptance counts"),
        ({"role_paths": (M10SourceRolePathCount(role_path="user", rows=2),)}, "role path counts"),
        ({"accepted_shape_rows": 0, "rejected_shape_rows": 1}, "rejection counts"),
        ({"tool_call_candidate_rows": 1}, "tool/no-tool"),
        ({"tools_per_row_min": 2}, "distribution bounds"),
    ],
)
def test_external_profile_schema_rejects_inconsistent_counts(
    updates: dict[str, object], message: str
) -> None:
    profile = _minimal_profile("toolace")
    with pytest.raises(ValidationError, match=message):
        M10ExternalSourceProfile.model_validate({**profile.to_dict(), **updates})


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"total_source_rows": 3}, "total source rows"),
        ({"total_accepted_shape_rows": 1}, "total accepted rows"),
        ({"total_rejected_shape_rows": 1}, "total rejected rows"),
    ],
)
def test_profile_report_rejects_each_aggregate_drift(
    updates: dict[str, object], message: str
) -> None:
    toolace = _minimal_profile("toolace")
    hermes = _minimal_profile("hermes_function_calling")
    base: dict[str, object] = {
        "profile_version": "m10-external-source-profile-v1",
        "data_config_sha256": "a" * 64,
        "profiles": [toolace.to_dict(), hermes.to_dict()],
        "total_source_rows": 2,
        "total_accepted_shape_rows": 2,
        "total_rejected_shape_rows": 0,
    }
    with pytest.raises(ValidationError, match=message):
        M10ExternalSourceProfileReport.model_validate({**base, **updates})


def test_source_schema_rejects_cross_kind_identity_and_readiness() -> None:
    artifact = {"filename": "data.json", "size_bytes": 1, "sha256": "a" * 64}
    external = {
        "source_id": "toolace",
        "source_kind": "external",
        "dataset_id": "fixture/external",
        "revision": "r1",
        "license": "apache-2.0",
        "mixture_basis_points": 3000,
        "readiness": "ready",
        "redistributable": False,
        "artifacts": [artifact],
        "license_evidence_sha256": "b" * 64,
    }
    invalid_variants = (
        {**external, "source_id": "tinyllm_devops"},
        {**external, "artifacts": []},
        {**external, "readiness": "pending_build"},
        {
            **external,
            "source_id": "toolace",
            "source_kind": "registered_replay",
            "artifacts": [],
        },
        {
            **external,
            "source_id": "m6_domain_replay",
            "source_kind": "registered_replay",
            "artifacts": [],
        },
        {
            **external,
            "source_id": "m6_domain_replay",
            "source_kind": "registered_replay",
            "artifacts": [],
            "content_sha256": "c" * 64,
            "manifest_sha256": "d" * 64,
            "readiness": "pending_build",
        },
        {**external, "source_id": "toolace", "source_kind": "authored", "artifacts": []},
        {**external, "source_id": "tinyllm_devops", "source_kind": "authored"},
        {
            **external,
            "source_id": "tinyllm_devops",
            "source_kind": "authored",
            "artifacts": [],
            "readiness": "ready",
        },
    )
    for value in invalid_variants:
        with pytest.raises(ValidationError):
            M10AgentSourceSpec.model_validate(value)


def test_nested_policy_and_data_schema_reject_order_or_threshold_drift() -> None:
    config = load_m10_agent_data_config(CONFIG)
    dedup = config.deduplication.to_dict()
    dedup["near_dedup_fields"] = ["tool_schema", "prompt"]
    with pytest.raises(ValidationError, match="near dedup fields"):
        M10AgentDedupPolicy.model_validate(dedup)

    contamination = config.contamination.to_dict()
    contamination["targets"][0], contamination["targets"][1] = (
        contamination["targets"][1],
        contamination["targets"][0],
    )
    with pytest.raises(ValidationError, match="frozen order"):
        M10AgentContaminationPolicy.model_validate(contamination)
    contamination = config.contamination.to_dict()
    contamination["targets"][1]["visibility"] = "public"
    with pytest.raises(ValidationError, match="sealed_private"):
        M10AgentContaminationPolicy.model_validate(contamination)

    decoded: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decoded["sources"][0], decoded["sources"][1] = decoded["sources"][1], decoded["sources"][0]
    with pytest.raises(ValidationError, match="frozen order"):
        M10AgentDataConfig.model_validate(decoded)
    decoded = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decoded["sources"][0]["mixture_basis_points"] = 2999
    with pytest.raises(ValidationError, match="token shares"):
        M10AgentDataConfig.model_validate(decoded)


def test_fully_frozen_data_config_is_trainable() -> None:
    decoded: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    authored = decoded["sources"][2]
    authored["readiness"] = "ready"
    authored["content_sha256"] = "c" * 64
    authored["manifest_sha256"] = "d" * 64
    decoded["status"] = "frozen"
    decoded["training_permitted"] = True

    frozen = M10AgentDataConfig.model_validate(decoded)

    assert frozen.status == "frozen"
    assert frozen.training_permitted is True


def test_profile_wrapper_builds_report_from_verified_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m10_agent_data_config(CONFIG)
    toolace = _minimal_profile("toolace")
    hermes = _minimal_profile("hermes_function_calling")
    monkeypatch.setattr(m10_module, "load_m10_agent_data_config", lambda _path: config)
    monkeypatch.setattr(m10_module, "_profile_toolace", lambda *_args: toolace)
    monkeypatch.setattr(m10_module, "_profile_hermes", lambda *_args: hermes)

    report = m10_module.profile_m10_external_sources(
        config_path=CONFIG,
        toolace_artifact=Path("toolace.json"),
        hermes_artifact=Path("hermes.json"),
    )

    assert report.total_source_rows == 2
    assert report.total_accepted_shape_rows == 2
