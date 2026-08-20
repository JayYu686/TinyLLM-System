from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.agent_eval.suite import tool_catalog
from tinyllm.data.m10_devops import (
    CATEGORY_COUNTS,
    LANGUAGE_COUNTS,
    ContaminationTarget,
    M10DevOpsDataError,
    _message,
    _prompt,
    _tool_call,
    _validate_arguments,
    build_devops_samples,
    build_manifest,
    build_public_report,
    load_bfcl_target,
    load_dataset,
    load_m6_domain_target,
    load_m9_target,
    render_review_packet,
    render_samples,
    scan_authored_duplicates,
    scan_contamination,
    write_dataset,
)
from tinyllm.data.m10_devops_schema import (
    M10DevOpsBuildReport,
    M10DevOpsContaminationReport,
    M10DevOpsDatasetManifest,
    M10DevOpsDuplicateReport,
    M10DevOpsTrainingMessage,
    M10DevOpsTrainingSample,
    canonical_json_sha256,
)


@pytest.fixture(scope="module")
def samples() -> tuple[M10DevOpsTrainingSample, ...]:
    return build_devops_samples()


def test_authored_dataset_is_deterministic_and_has_frozen_distribution(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    second = build_devops_samples()

    assert render_samples(samples) == render_samples(second)
    assert len(samples) == 2400
    assert Counter(item.category for item in samples) == Counter(CATEGORY_COUNTS)
    assert Counter(item.language for item in samples) == Counter(LANGUAGE_COUNTS)
    assert len({item.sample_id for item in samples}) == 2400
    assert len({_prompt(item) for item in samples}) == 2400
    assert all(item.mode == "nonthinking" for item in samples)
    assert all(len(item.available_tools) == 7 for item in samples)


def test_authored_supervision_and_tool_paths_are_consistent(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    expected_calls = {
        "single_tool": {1},
        "no_tool": {0},
        "wrong_tool_irrelevance": {0},
        "missing_argument_clarification": {0},
        "sequential_multi_step": {2},
        "parallel_independent_tools": {2},
        "tool_failure_recovery": {2},
        "grounding_approval_security": {1},
    }
    observed: dict[str, set[int]] = {}
    for sample in samples:
        assert all(
            message.supervised == (message.role == "assistant") for message in sample.messages
        )
        assert all("<think>" not in (message.content or "").lower() for message in sample.messages)
        count = sum(len(message.tool_calls) for message in sample.messages)
        observed.setdefault(sample.category, set()).add(count)
        if sample.category == "parallel_independent_tools":
            assert any(len(message.tool_calls) == 2 for message in sample.messages)

    assert observed == expected_calls


def test_manifest_keeps_unreviewed_content_out_of_training(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    pending = build_manifest(samples)
    approved = build_manifest(samples, review_status="approved")

    assert pending.dataset_version == approved.dataset_version
    assert pending.review_status == "pending"
    assert pending.training_permitted is False
    assert approved.review_status == "approved"
    assert approved.training_permitted is True
    assert pending.item_count == 2400
    assert pending.unique_group_count == 48
    assert pending.items_sha256 == hashlib.sha256(render_samples(samples)).hexdigest()


def test_duplicate_gate_allows_clustered_variants_but_rejects_cross_group_copy(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    clean = scan_authored_duplicates(samples)
    assert clean.status == "pass"
    assert clean.exact_duplicate_pairs == 0
    assert clean.clustered_near_duplicate_pairs > 0
    assert clean.cross_group_near_duplicate_pairs == 0

    payload = samples[0].to_dict()
    payload["sample_id"] = "m10-devops-en-single-tool-9999"
    payload["template_family"] = "family-single-tool-99"
    payload["group_id"] = "group-single-tool-99"
    payload.pop("content_sha256")
    payload["content_sha256"] = canonical_json_sha256(payload)
    copied = M10DevOpsTrainingSample.model_validate(payload)
    contaminated = (*samples[:-1], copied)

    report = scan_authored_duplicates(contaminated)
    assert report.status == "fail"
    assert report.exact_duplicate_pairs == 1
    assert report.cross_group_near_duplicate_pairs >= 1


def test_contamination_scan_is_content_free_and_detects_exact_prompt(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    manifest = build_manifest(samples)
    clean_targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"unrelated boundary prompt {target_id} alpha beta gamma delta",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    clean = scan_contamination(samples, manifest, clean_targets)
    assert clean.status == "pass"
    assert all(item.exact_matches == item.near_matches == 0 for item in clean.targets)
    assert clean.contains_evaluation_content is False

    first_prompt = _prompt(samples[0])
    collided = (
        ContaminationTarget(
            target_id="m9_dev",
            version="m9-dev-v1",
            content_sha256=canonical_json_sha256([first_prompt]),
            prompts=(first_prompt,),
        ),
        *clean_targets[1:],
    )
    failed = scan_contamination(samples, manifest, collided)
    assert failed.status == "fail"
    assert failed.targets[0].exact_matches == 1
    assert failed.targets[0].contains_target_content is False


def test_review_packet_is_fixed_stratified_sample(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    packet = render_review_packet(samples, build_manifest(samples))

    assert packet.count("\n## ") == 80
    assert "training_permitted=false" in packet
    assert "Assistant 工具调用（监督）" in packet
    assert "Tool 结果（屏蔽 Loss）" in packet
    assert "input_schema" not in packet


def test_dataset_round_trip_and_existing_artifact_drift(
    tmp_path: Path, samples: tuple[M10DevOpsTrainingSample, ...]
) -> None:
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)
    targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"separate target {target_id} one two three four five",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    contamination = scan_contamination(samples, manifest, targets)
    directory = write_dataset(tmp_path, samples, manifest, duplicate, contamination)

    loaded_manifest, loaded_samples = load_dataset(directory)
    assert loaded_manifest == manifest
    assert render_samples(loaded_samples) == render_samples(samples)
    assert write_dataset(tmp_path, samples, manifest, duplicate, contamination) == directory

    (directory / "duplicate-report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(M10DevOpsDataError, match="different content"):
        write_dataset(tmp_path, samples, manifest, duplicate, contamination)
    with pytest.raises(M10DevOpsDataError, match="hash mismatch"):
        load_dataset(directory)


def test_public_report_cannot_claim_ready_before_review(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)
    targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"independent {target_id} boundary six seven eight nine",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    contamination = scan_contamination(samples, manifest, targets)
    report = build_public_report(manifest, duplicate, contamination)

    assert report.status == "review_pending"
    assert report.training_permitted is False
    assert report.private_artifacts_only is True


def test_message_schema_rejects_mask_hash_and_visible_reasoning() -> None:
    valid = _message("assistant", content="Bounded answer.")
    payload = valid.to_dict()
    payload["supervised"] = False
    payload["message_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "message_sha256"}
    )
    with pytest.raises(ValidationError, match="only assistant"):
        M10DevOpsTrainingMessage.model_validate(payload)

    payload = valid.to_dict()
    payload["content"] = "<think>private trace</think> final"
    payload["message_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "message_sha256"}
    )
    with pytest.raises(ValidationError, match="chain-of-thought"):
        M10DevOpsTrainingMessage.model_validate(payload)

    payload = valid.to_dict()
    payload["message_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="message SHA256"):
        M10DevOpsTrainingMessage.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_valid_1",
                "supervised": False,
                "tool_calls": [],
            },
            "tool messages require",
        ),
        (
            {
                "role": "user",
                "content": "question",
                "name": "search_evidence",
                "supervised": False,
                "tool_calls": [],
            },
            "only tool messages",
        ),
        (
            {
                "role": "user",
                "content": "question",
                "supervised": False,
                "tool_calls": [
                    {
                        "id": "call_valid_1",
                        "type": "function",
                        "function": {"name": "search_evidence", "arguments": {"query": "x"}},
                    }
                ],
            },
            "only assistant messages",
        ),
        (
            {"role": "assistant", "content": None, "supervised": True, "tool_calls": []},
            "messages require content",
        ),
    ],
)
def test_message_schema_rejects_invalid_role_fields(
    payload: dict[str, object], message: str
) -> None:
    payload["message_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ValidationError, match=message):
        M10DevOpsTrainingMessage.model_validate(payload)


def test_sample_schema_rejects_hash_and_tool_result_drift(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    sample = next(item for item in samples if item.category == "single_tool")
    payload = sample.to_dict()
    payload["prompt_sha256"] = "0" * 64
    payload.pop("content_sha256")
    payload["content_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ValidationError, match="prompt SHA256"):
        M10DevOpsTrainingSample.model_validate(payload)

    payload = sample.to_dict()
    tool_result = next(item for item in payload["messages"] if item["role"] == "tool")
    tool_result["name"] = "query_metrics"
    canonical_message = {
        key: value for key, value in tool_result.items() if key != "message_sha256"
    }
    tool_result["message_sha256"] = canonical_json_sha256(canonical_message)
    payload.pop("content_sha256")
    payload["content_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ValidationError, match="does not match"):
        M10DevOpsTrainingSample.model_validate(payload)

    payload = sample.to_dict()
    payload["tool_schema_sha256"] = "f" * 64
    payload.pop("content_sha256")
    payload["content_sha256"] = canonical_json_sha256(payload)
    with pytest.raises(ValidationError, match="tool Schema SHA256"):
        M10DevOpsTrainingSample.model_validate(payload)

    payload = sample.to_dict()
    payload["content_sha256"] = "e" * 64
    with pytest.raises(ValidationError, match="content SHA256"):
        M10DevOpsTrainingSample.model_validate(payload)


def _rehash_sample(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("content_sha256", None)
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def test_sample_schema_enforces_category_tool_shape(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    single = next(item for item in samples if item.category == "single_tool").to_dict()
    single["category"] = "no_tool"
    single["sample_id"] = "m10-devops-en-no-tool-9999"
    with pytest.raises(ValidationError, match="no-call categories"):
        M10DevOpsTrainingSample.model_validate(_rehash_sample(single))

    no_tool = next(item for item in samples if item.category == "no_tool").to_dict()
    no_tool["category"] = "single_tool"
    no_tool["sample_id"] = "m10-devops-en-single-tool-9998"
    with pytest.raises(ValidationError, match="require at least one"):
        M10DevOpsTrainingSample.model_validate(_rehash_sample(no_tool))

    sequential = next(
        item for item in samples if item.category == "sequential_multi_step"
    ).to_dict()
    sequential["category"] = "parallel_independent_tools"
    sequential["sample_id"] = "m10-devops-en-parallel-independent-tools-9997"
    with pytest.raises(ValidationError, match="parallel trajectories"):
        M10DevOpsTrainingSample.model_validate(_rehash_sample(sequential))


def test_manifest_schema_rejects_approval_without_training_permission(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    payload = build_manifest(samples, review_status="approved").to_dict()
    payload["training_permitted"] = False
    with pytest.raises(ValidationError, match="approved content review"):
        M10DevOpsDatasetManifest.model_validate(payload)

    with pytest.raises(M10DevOpsDataError, match="exactly 2,400"):
        build_manifest(samples[:-1])


def test_tool_argument_validator_rejects_missing_extra_and_wrong_type() -> None:
    tools = {item.tool_name: item for item in tool_catalog()}
    with pytest.raises(M10DevOpsDataError, match="missing required"):
        _validate_arguments(_tool_call("call_test_1", "get_run", {}), tools)
    with pytest.raises(M10DevOpsDataError, match="unexpected argument"):
        _validate_arguments(
            _tool_call("call_test_2", "get_run", {"run_id": "run", "extra": True}), tools
        )
    with pytest.raises(M10DevOpsDataError, match="invalid argument type"):
        _validate_arguments(_tool_call("call_test_3", "get_run", {"run_id": 7}), tools)


def test_report_schemas_reject_inconsistent_status_and_hash(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    duplicate = scan_authored_duplicates(samples)
    payload = duplicate.to_dict()
    payload["status"] = "fail"
    payload["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    with pytest.raises(ValidationError, match="status is inconsistent"):
        M10DevOpsDuplicateReport.model_validate(payload)

    payload = duplicate.to_dict()
    payload["report_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="SHA256 is inconsistent"):
        M10DevOpsDuplicateReport.model_validate(payload)


def test_contamination_and_public_report_schemas_fail_closed(
    samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)
    targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"isolated boundary {target_id} alpha beta gamma delta epsilon",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    contamination = scan_contamination(samples, manifest, targets)

    payload = contamination.to_dict()
    payload["targets"] = list(reversed(payload["targets"]))
    payload["report_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    with pytest.raises(ValidationError, match="frozen order"):
        M10DevOpsContaminationReport.model_validate(payload)

    payload = contamination.to_dict()
    payload["report_sha256"] = "1" * 64
    with pytest.raises(ValidationError, match="SHA256 is inconsistent"):
        M10DevOpsContaminationReport.model_validate(payload)

    ready = build_public_report(
        build_manifest(samples, review_status="approved"), duplicate, contamination
    )
    assert ready.status == "ready"
    assert ready.training_permitted is True
    payload = ready.to_dict()
    payload["training_permitted"] = False
    with pytest.raises(ValidationError, match="build status is inconsistent"):
        M10DevOpsBuildReport.model_validate(payload)


def test_m9_target_loader_verifies_manifest_count(tmp_path: Path) -> None:
    directory = tmp_path / "m9"
    directory.mkdir()
    rows = [
        {"messages": [{"role": "user", "content": "first target"}]},
        {"messages": [{"role": "user", "content": "second target"}]},
    ]
    (directory / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "suite_version": "m9-test-v1",
                "content_sha256": "1" * 64,
                "item_count": 2,
            }
        ),
        encoding="utf-8",
    )

    target = load_m9_target(directory, target_id="m9_dev")
    assert target.prompts == ("first target", "second target")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["item_count"] = 3
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(M10DevOpsDataError, match="count"):
        load_m9_target(directory, target_id="m9_dev")


def test_m6_target_loader_requires_300_items(tmp_path: Path) -> None:
    directory = tmp_path / "m6"
    directory.mkdir()
    row = {"prompt_messages": [{"role": "user", "content": "domain boundary"}]}
    (directory / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"suite_version": "m6-test", "content_sha256": "2" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(M10DevOpsDataError, match="exactly 300"):
        load_m6_domain_target(directory)


def test_bfcl_loader_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(M10DevOpsDataError, match="cannot read BFCL"):
        load_bfcl_target(tmp_path)


def test_write_and_load_reject_unsafe_directories(
    tmp_path: Path, samples: tuple[M10DevOpsTrainingSample, ...]
) -> None:
    link = tmp_path / "linked-output"
    link.symlink_to(tmp_path / "target", target_is_directory=True)
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)
    targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"unrelated {target_id} boundary one two three four five",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    contamination = scan_contamination(samples, manifest, targets)
    with pytest.raises(M10DevOpsDataError, match="symbolic link"):
        write_dataset(link, samples, manifest, duplicate, contamination)
    with pytest.raises(M10DevOpsDataError, match="missing or unsafe"):
        load_dataset(tmp_path / "missing")
