from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

import tinyllm.data.m10_mixture as mixture
from tinyllm.data.m10_agent_schema import M10SourceId
from tinyllm.data.m10_canonical import (
    canonicalize_toolace_row,
    import_external_source,
    write_external_import,
)
from tinyllm.data.m10_devops import (
    ContaminationTarget,
    build_devops_samples,
    build_manifest,
    render_review_packet,
    scan_authored_duplicates,
    scan_contamination,
    write_dataset,
)
from tinyllm.data.m10_devops_review import (
    finalize_m10_devops_content_review,
    render_json,
)
from tinyllm.data.m10_devops_schema import M10DevOpsTrainingSample, canonical_json_sha256
from tinyllm.data.m10_mixture import (
    M10FrozenDataset,
    M10MixtureError,
    M10TextCandidate,
    _pad_sequence,
    build_frozen_mixture,
    build_public_report,
    deduplicate_text_candidates,
    load_frozen_mixture_config,
    open_frozen_mixture,
    render_agent_conversation,
    select_exact_sequences,
    write_frozen_mixture,
)
from tinyllm.data.m10_mixture_schema import (
    M10FrozenMixtureConfig,
    M10MixtureLanguage,
    M10MixtureMode,
)
from tinyllm.data.tokenization import TokenizersBackend


def _candidate(source_id: str, record_id: str) -> M10TextCandidate:
    tools: tuple[dict[str, object], ...] = (
        {
            "type": "function",
            "function": {
                "name": "get_run",
                "description": "Read one run.",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
    )
    messages: tuple[dict[str, object], ...] = (
        {"role": "system", "content": "Use evidence.", "tool_calls": (), "tool_call_ids": ()},
        {
            "role": "user",
            "content": "Inspect run one.",
            "tool_calls": (),
            "tool_call_ids": (),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": ({"id": "call_1", "name": "get_run", "arguments": {"run_id": "one"}},),
            "tool_call_ids": (),
        },
        {
            "role": "tool",
            "content": '{"status":"failed"}',
            "tool_calls": (),
            "tool_call_ids": ("call_1",),
        },
        {
            "role": "assistant",
            "content": "Run one failed.",
            "tool_calls": (),
            "tool_call_ids": (),
        },
    )
    return M10TextCandidate(
        source_id=source_id,  # type: ignore[arg-type]
        version="fixture-v1",
        record_id=record_id,
        record_sha256="a" * 64,
        group_id="fixture-group",
        language="en",
        tools=tools,
        messages=messages,
        prompt="Inspect run one.",
        prompt_sha256="b" * 64,
        tool_schema_text=json.dumps(tools, ensure_ascii=False, sort_keys=True),
        content_sha256="c" * 64,
    )


@pytest.fixture(scope="module")
def authored_samples() -> tuple[M10DevOpsTrainingSample, ...]:
    return build_devops_samples()


def _toolace_fixture_row() -> dict[str, object]:
    tool = {
        "name": "Read Log",
        "description": "Read a bounded log excerpt.",
        "parameters": {
            "type": "dict",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "required": None,
    }
    return {
        "system": (
            "Use evidence.\nHere is a list of functions in JSON format that you can invoke:\n"
            f"{json.dumps([tool])}. \nShould you decide to return the function call(s)."
        ),
        "conversations": [
            {"from": "user", "value": "Diagnose this run."},
            {"from": "assistant", "value": '[Read Log(path="run.log")]'},
            {"from": "tool", "value": '{"line":"OOM"}'},
            {"from": "assistant", "value": "The run exhausted memory."},
        ],
    }


def test_frozen_config_encodes_exact_source_language_mode_matrix() -> None:
    config = load_frozen_mixture_config(Path("configs/data/m10_agent_frozen.yaml"))

    assert sum(item.supervised_tokens for item in config.strata) == 1_000_000
    assert sum(item.supervised_tokens for item in config.strata if item.language == "zh") == 300_000
    assert (
        sum(item.supervised_tokens for item in config.strata if item.mode == "thinking") == 60_000
    )


def test_frozen_config_rejects_silent_ratio_drift() -> None:
    value = yaml.safe_load(Path("configs/data/m10_agent_frozen.yaml").read_text(encoding="utf-8"))
    value["strata"][0]["supervised_tokens"] -= 1

    with pytest.raises(ValueError, match="frozen matrix"):
        M10FrozenMixtureConfig.model_validate(value)


def test_repair_config_increases_grounded_devops_supervision() -> None:
    value = yaml.safe_load(Path("configs/data/m10_agent_frozen.yaml").read_text(encoding="utf-8"))
    value["config_version"] = "m10-agent-frozen-mixture-v2"
    value["build_seed"] = 20260825
    value["strata"] = [
        {
            "source_id": "toolace",
            "language": "en",
            "mode": "nonthinking",
            "supervised_tokens": 200000,
        },
        {
            "source_id": "hermes_function_calling",
            "language": "en",
            "mode": "nonthinking",
            "supervised_tokens": 100000,
        },
        {
            "source_id": "tinyllm_devops",
            "language": "en",
            "mode": "nonthinking",
            "supervised_tokens": 280000,
        },
        {
            "source_id": "tinyllm_devops",
            "language": "zh",
            "mode": "nonthinking",
            "supervised_tokens": 120000,
        },
        {
            "source_id": "m6_domain_replay",
            "language": "en",
            "mode": "nonthinking",
            "supervised_tokens": 70000,
        },
        {
            "source_id": "m6_domain_replay",
            "language": "en",
            "mode": "thinking",
            "supervised_tokens": 30000,
        },
        {
            "source_id": "m6_domain_replay",
            "language": "zh",
            "mode": "nonthinking",
            "supervised_tokens": 70000,
        },
        {
            "source_id": "m6_domain_replay",
            "language": "zh",
            "mode": "thinking",
            "supervised_tokens": 30000,
        },
        {
            "source_id": "m2_no_tool_replay",
            "language": "en",
            "mode": "nonthinking",
            "supervised_tokens": 20000,
        },
        {
            "source_id": "m2_no_tool_replay",
            "language": "zh",
            "mode": "nonthinking",
            "supervised_tokens": 80000,
        },
    ]

    config = M10FrozenMixtureConfig.model_validate(value)
    source_counts = {
        source: sum(item.supervised_tokens for item in config.strata if item.source_id == source)
        for source in (
            "toolace",
            "hermes_function_calling",
            "tinyllm_devops",
            "m6_domain_replay",
            "m2_no_tool_replay",
        )
    }

    assert source_counts == {
        "toolace": 200_000,
        "hermes_function_calling": 100_000,
        "tinyllm_devops": 400_000,
        "m6_domain_replay": 200_000,
        "m2_no_tool_replay": 100_000,
    }


def test_agent_renderer_supervises_calls_and_final_without_visible_cot() -> None:
    text, spans = render_agent_conversation(_candidate("toolace", "toolace-fixture"))

    assert len(spans) == 2
    assert "<tools>" in text
    assert '<tool_call>\n{"arguments":{"run_id":"one"},"name":"get_run"}' in text
    assert "<tool_response>" in text
    assert text.count("<think>\n\n</think>\n\n") == 2
    assert all("<think>" not in text[start:end] for start, end in spans)
    assert all(text[start:end].endswith("<|im_end|>") for start, end in spans)


def test_exact_dedup_prefers_approved_authored_source() -> None:
    toolace = _candidate("toolace", "toolace-fixture")
    authored = _candidate("tinyllm_devops", "m10-devops-en-fixture-0001")

    kept, report = deduplicate_text_candidates((toolace, authored))

    assert tuple(item.source_id for item in kept) == ("tinyllm_devops",)
    assert report["exact_duplicate_drops"] == 1
    assert report["near_duplicate_drops"] == 0
    assert report["source_duplicate_drops"] == {
        "toolace": 1,
        "hermes_function_calling": 0,
        "tinyllm_devops": 0,
    }


def test_exact_token_selector_cycles_and_trims_only_labels() -> None:
    sequence = _pad_sequence(
        (10, 11, 12, 13),
        (-100, 11, 12, 13),
        source_id="toolace",
        language="en",
        mode="nonthinking",
        record_sha256="a" * 64,
    )

    selected, reuse, partial = select_exact_sequences((sequence,), target=5, seed=7)

    assert sum(item.supervised_tokens for item in selected) == 5
    assert len(selected) == 2
    assert reuse == 1
    assert partial == 1
    assert selected[-1].input_ids == sequence.input_ids
    assert selected[-1].supervised_tokens == 2


def test_source_adapters_preserve_tool_calls_results_and_language(
    authored_samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    sample = canonicalize_toolace_row(
        _toolace_fixture_row(),
        revision="6bda777c88d21e5a204703c1ee45597a8fa4f734",
        row_index=0,
    )

    external = mixture._external_candidate(sample, version="toolace-fixture-v1")
    authored = mixture._devops_candidate(authored_samples[0], version="devops-fixture-v1")

    function = cast(dict[str, object], external.tools[0]["function"])
    assert function["name"] == "read_log"
    assert external.messages[2]["tool_calls"]
    assert external.messages[3]["tool_call_ids"]
    assert external.prompt == "Diagnose this run."
    assert authored.source_id == "tinyllm_devops"
    assert authored.messages[-1]["role"] == "assistant"
    assert authored.language in {"en", "zh"}


def test_external_loader_rebuilds_commit_manifest_and_content_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "toolace.json"
    source_path.write_text(json.dumps([_toolace_fixture_row()]), encoding="utf-8")
    config_value = yaml.safe_load(Path("configs/data/m10_agent.yaml").read_text(encoding="utf-8"))
    config_value["sources"][0]["artifacts"] = [
        {
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    ]
    config_path = tmp_path / "m10-agent.yaml"
    config_path.write_text(yaml.safe_dump(config_value, sort_keys=False), encoding="utf-8")
    build = import_external_source(
        config_path=config_path,
        source_id="toolace",
        artifact_path=source_path,
    )
    root = write_external_import(tmp_path / "imports", build)
    manifest_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()

    manifest, candidates = mixture.load_external_candidates(
        root,
        expected_version=build.manifest.import_version,
        expected_content=build.manifest.content_sha256,
        expected_manifest=manifest_sha,
    )

    assert manifest == build.manifest
    assert len(candidates) == 1
    assert candidates[0].record_id == "toolace-00000"
    with pytest.raises(M10MixtureError, match="identity or row count"):
        mixture.load_external_candidates(
            root,
            expected_version=build.manifest.import_version,
            expected_content="0" * 64,
            expected_manifest=manifest_sha,
        )


def test_approved_devops_loader_binds_review_to_pending_source(
    tmp_path: Path,
    authored_samples: tuple[M10DevOpsTrainingSample, ...],
) -> None:
    pending = build_manifest(authored_samples)
    duplicate = scan_authored_duplicates(authored_samples)
    targets = tuple(
        ContaminationTarget(
            target_id=target_id,
            version=f"{target_id}-fixture-v1",
            content_sha256=canonical_json_sha256([target_id]),
            prompts=(f"independent {target_id} boundary alpha beta gamma delta epsilon",),
        )
        for target_id in ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    )
    contamination = scan_contamination(authored_samples, pending, targets)
    dataset_root = write_dataset(
        tmp_path / "datasets", authored_samples, pending, duplicate, contamination
    )
    packet = tmp_path / "review-packet.md"
    packet.write_text(render_review_packet(authored_samples, pending), encoding="utf-8")
    approval, approved, _ = finalize_m10_devops_content_review(
        dataset_dir=dataset_root,
        review_packet_path=packet,
        reviewed_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    approval_root = tmp_path / "approval"
    approval_root.mkdir()
    (approval_root / "approved-manifest.json").write_bytes(render_json(approved))
    (approval_root / "approval.json").write_bytes(render_json(approval))
    approved_sha = hashlib.sha256(
        (approval_root / "approved-manifest.json").read_bytes()
    ).hexdigest()
    approval_sha = hashlib.sha256((approval_root / "approval.json").read_bytes()).hexdigest()

    manifest, candidates = mixture.load_approved_devops_candidates(
        dataset_root,
        approval_root,
        expected_version=approved.dataset_version,
        expected_content=approved.content_sha256,
        expected_manifest=approved_sha,
        expected_approval=approval_sha,
    )

    assert manifest == approved
    assert len(candidates) == len(authored_samples)
    with pytest.raises(M10MixtureError, match="approval file hash"):
        mixture.load_approved_devops_candidates(
            dataset_root,
            approval_root,
            expected_version=approved.dataset_version,
            expected_content=approved.content_sha256,
            expected_manifest=approved_sha,
            expected_approval="0" * 64,
        )


def test_immutable_json_and_jsonl_guards_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(M10MixtureError, match="missing or unsafe"):
        mixture._safe_json(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(M10MixtureError, match="cannot decode"):
        mixture._safe_json(invalid)
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(M10MixtureError, match="must be an object"):
        mixture._safe_json(invalid)

    with pytest.raises(M10MixtureError, match="JSONL is missing"):
        mixture._load_jsonl(tmp_path / "missing.jsonl")
    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(M10MixtureError, match="row is not an object"):
        mixture._load_jsonl(invalid_jsonl)
    invalid_jsonl.write_text("{\n", encoding="utf-8")
    with pytest.raises(M10MixtureError, match="cannot decode JSONL"):
        mixture._load_jsonl(invalid_jsonl)

    with pytest.raises(M10MixtureError, match="directory is missing"):
        mixture._verify_committed_files(tmp_path / "absent")
    committed = tmp_path / "committed"
    committed.mkdir()
    (committed / "COMMITTED.json").write_text('{"files":{}}', encoding="utf-8")
    with pytest.raises(M10MixtureError, match="no file identities"):
        mixture._verify_committed_files(committed)
    (committed / "COMMITTED.json").write_text(
        '{"files":{"../escape":"' + "0" * 64 + '"}}', encoding="utf-8"
    )
    with pytest.raises(M10MixtureError, match="unsafe filename"):
        mixture._verify_committed_files(committed)
    (committed / "COMMITTED.json").write_text(
        '{"files":{"items.jsonl":"' + "0" * 64 + '"}}', encoding="utf-8"
    )
    (committed / "items.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(M10MixtureError, match="hash differs"):
        mixture._verify_committed_files(committed)


def test_render_tokenization_and_sequence_guards() -> None:
    base = _candidate("toolace", "toolace-fixture")
    plain = replace(base, tools=(), messages=base.messages[1:])
    text, spans = render_agent_conversation(plain)
    assert text.startswith("<|im_start|>system")
    assert "# Tools" not in text
    assert spans

    unsupported = replace(
        base,
        messages=({"role": "developer", "content": "bad", "tool_calls": ()},),
    )
    with pytest.raises(M10MixtureError, match="unsupported"):
        render_agent_conversation(unsupported)
    no_assistant = replace(base, messages=(base.messages[1],))
    with pytest.raises(M10MixtureError, match="no supervised"):
        render_agent_conversation(no_assistant)

    assert mixture._labels_from_offsets((1, 2), ((0, 1), (1, 2)), ((1, 2),)) == (-100, 2)
    with pytest.raises(M10MixtureError, match="crosses"):
        mixture._labels_from_offsets((1,), ((0, 2),), ((1, 2),))
    with pytest.raises(M10MixtureError, match="length"):
        _pad_sequence(
            (1,),
            (1,),
            source_id="toolace",
            language="en",
            mode="nonthinking",
            record_sha256="a" * 64,
        )
    with pytest.raises(M10MixtureError, match="no shifted"):
        _pad_sequence(
            (1, 2),
            (-100, -100),
            source_id="toolace",
            language="en",
            mode="nonthinking",
            record_sha256="a" * 64,
        )

    class CharacterBackend:
        def __init__(self, *, overlength: bool = False) -> None:
            self.overlength = overlength

        def encode(self, value: str) -> SimpleNamespace:
            size = 2049 if self.overlength else len(value)
            return SimpleNamespace(
                ids=tuple(range(size)), offsets=tuple((index, index + 1) for index in range(size))
            )

        def decode(self, _ids: object) -> str:
            return "检查" if self.overlength else "inspect"

    sequences, rejected = mixture.tokenize_text_candidates(
        (base,), backend=cast(Any, CharacterBackend())
    )
    assert len(sequences) == 1
    assert not rejected
    sequences, rejected = mixture.tokenize_text_candidates(
        (base,), backend=cast(Any, CharacterBackend(overlength=True))
    )
    assert not sequences
    assert rejected == Counter({"toolace": 1})
    assert mixture._decoded_language(cast(Any, CharacterBackend()), (1, 2)) == "en"
    assert mixture._decoded_language(cast(Any, CharacterBackend(overlength=True)), (1, 2)) == "zh"


def test_near_dedup_and_contamination_rejection_paths() -> None:
    prompt = "inspect the failed run and report the bounded evidence from the registered log"
    left = _unique_candidate("toolace", "toolace-near", "en", prompt)
    right = _unique_candidate(
        "hermes_function_calling",
        "hermes-near",
        "en",
        prompt + " safely",
    )
    kept, duplicate = deduplicate_text_candidates((left, right))
    assert len(kept) == 1
    assert duplicate["near_duplicate_drops"] == 1

    targets = tuple(
        SimpleNamespace(
            target_id=target_id,
            version=f"{target_id}-v1",
            content_sha256=f"{index:064x}",
            prompts=(prompt if index == 1 else f"unrelated {target_id} boundary",),
        )
        for index, target_id in enumerate(
            ("m9_dev", "m9_release", "bfcl_core", "m6_domain"), start=1
        )
    )
    with pytest.raises(M10MixtureError, match="overlap"):
        mixture.scan_text_contamination((left,), targets=targets)
    with pytest.raises(M10MixtureError, match="frozen order"):
        mixture.scan_text_contamination((left,), targets=tuple(reversed(targets)))
    with pytest.raises(M10MixtureError, match="invalid target"):
        select_exact_sequences((), target=1, seed=1)
    with pytest.raises(M10MixtureError, match="partial"):
        mixture._trim_supervision(
            _pad_sequence(
                (1, 2),
                (-100, 2),
                source_id="toolace",
                language="en",
                mode="nonthinking",
                record_sha256="a" * 64,
            ),
            1,
        )


def _unique_candidate(
    source_id: M10SourceId, record_id: str, language: str, prompt: str
) -> M10TextCandidate:
    candidate = _candidate(source_id, record_id)
    messages = list(candidate.messages)
    messages[1] = {**messages[1], "content": prompt}
    return replace(
        candidate,
        language=cast(M10MixtureLanguage, language),
        messages=tuple(messages),
        prompt=prompt,
        prompt_sha256=(record_id.encode().hex() + "0" * 64)[:64],
    )


def _dense_sequence(
    source_id: M10SourceId, language: str, mode: str, ordinal: int
) -> mixture.M10TrainingSequence:
    ids = tuple(range(2048))
    return mixture._pad_sequence(
        ids,
        (-100,) + ids[1:],
        source_id=source_id,
        language=cast(M10MixtureLanguage, language),
        mode=cast(M10MixtureMode, mode),
        record_sha256=f"{ordinal:064x}",
    )


def test_synthetic_full_build_is_atomic_reopenable_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path("configs/data/m10_agent_frozen.yaml")
    source_config = Path("configs/data/m10_agent.yaml")
    config = load_frozen_mixture_config(config_path)
    real_sha = mixture._sha256_file

    toolace = (_unique_candidate("toolace", "toolace-one", "en", "ToolACE one"),)
    hermes = (_unique_candidate("hermes_function_calling", "hermes-one", "en", "Hermes one"),)
    devops = (
        _unique_candidate("tinyllm_devops", "m10-devops-en-fixture-0001", "en", "DevOps English"),
        _unique_candidate("tinyllm_devops", "m10-devops-zh-fixture-0002", "zh", "检查 DevOps 中文"),
    )
    manifests = {
        "toolace": SimpleNamespace(accepted_rows=1),
        "hermes": SimpleNamespace(accepted_rows=1),
    }

    def fake_external(root: Path, **_: object) -> tuple[object, tuple[M10TextCandidate, ...]]:
        return (
            (manifests["toolace"], toolace)
            if "toolace" in root.parts
            else (
                manifests["hermes"],
                hermes,
            )
        )

    def fake_text_tokenization(
        candidates: tuple[M10TextCandidate, ...], **_: object
    ) -> tuple[tuple[mixture.M10TrainingSequence, ...], Counter[M10SourceId]]:
        return (
            tuple(
                _dense_sequence(item.source_id, item.language, "nonthinking", index + 1)
                for index, item in enumerate(candidates)
            ),
            Counter(),
        )

    targets = {
        "m9_dev": SimpleNamespace(
            target_id="m9_dev",
            version=config.contamination.m9_dev_version,
            content_sha256="1" * 64,
            prompts=("unrelated dev boundary",),
        ),
        "m9_release": SimpleNamespace(
            target_id="m9_release",
            version=config.contamination.m9_release_version,
            content_sha256="2" * 64,
            prompts=("unrelated release boundary",),
        ),
        "bfcl_core": SimpleNamespace(
            target_id="bfcl_core",
            version=config.contamination.bfcl_version,
            content_sha256="3" * 64,
            prompts=("unrelated BFCL boundary",),
        ),
        "m6_domain": SimpleNamespace(
            target_id="m6_domain",
            version=config.contamination.m6_domain_version,
            content_sha256="4" * 64,
            prompts=("unrelated domain boundary",),
        ),
    }
    m6_sequences = tuple(
        _dense_sequence("m6_domain_replay", language, mode, index + 20)
        for index, (language, mode) in enumerate(
            (
                ("en", "nonthinking"),
                ("en", "thinking"),
                ("zh", "nonthinking"),
                ("zh", "thinking"),
            )
        )
    )
    m2_sequences = (
        _dense_sequence("m2_no_tool_replay", "en", "nonthinking", 30),
        _dense_sequence("m2_no_tool_replay", "zh", "nonthinking", 31),
    )

    def fake_sha(path: Path) -> str:
        if path.name == "manifest.json" and "m6-domain-generalization" in path.parts:
            return config.inputs.m6_domain_replay.manifest_sha256
        if path.name == "manifest.json" and "m2-sft" in path.parts:
            return config.inputs.m2_no_tool_replay.manifest_sha256
        return real_sha(path)

    monkeypatch.setattr(mixture, "_sha256_file", fake_sha)
    monkeypatch.setattr(mixture, "load_external_candidates", fake_external)
    monkeypatch.setattr(
        mixture,
        "load_approved_devops_candidates",
        lambda *args, **kwargs: (SimpleNamespace(item_count=2), devops),
    )
    monkeypatch.setattr(TokenizersBackend, "from_files", lambda *args, **kwargs: object())
    monkeypatch.setattr(mixture, "tokenize_text_candidates", fake_text_tokenization)
    monkeypatch.setattr(
        mixture,
        "load_m9_target",
        lambda _path, *, target_id: targets[target_id],
    )
    monkeypatch.setattr(mixture, "load_bfcl_target", lambda _path: targets["bfcl_core"])
    monkeypatch.setattr(mixture, "load_m6_domain_target", lambda _path: targets["m6_domain"])
    monkeypatch.setattr(
        mixture,
        "load_m6_replay_candidates",
        lambda *args, **kwargs: (
            SimpleNamespace(content_sha256=config.inputs.m6_domain_replay.content_sha256),
            m6_sequences,
        ),
    )
    monkeypatch.setattr(
        mixture,
        "load_m2_replay_candidates",
        lambda *args, **kwargs: (
            SimpleNamespace(content_sha256=config.inputs.m2_no_tool_replay.content_sha256),
            m2_sequences,
            0,
        ),
    )

    build = build_frozen_mixture(
        config_path=config_path,
        source_config_path=source_config,
        tokenizer_config_path=Path("configs/data/m2_tokenization.yaml"),
        model_dir=tmp_path / "model",
        artifact_root=tmp_path / "artifacts",
        m9_dev_dir=tmp_path / "dev",
        m9_release_dir=tmp_path / "release",
        bfcl_data_root=tmp_path / "bfcl",
        m6_domain_dir=tmp_path / "domain",
    )

    manifest = write_frozen_mixture(tmp_path / "frozen", build)
    reopened = open_frozen_mixture(tmp_path / "frozen" / manifest.dataset_version)
    repeated = write_frozen_mixture(tmp_path / "frozen", build)
    report = build_public_report(reopened)
    dataset = M10FrozenDataset(tmp_path / "frozen" / manifest.dataset_version)

    assert reopened == repeated == manifest
    assert manifest.target_supervised_tokens == 1_000_000
    assert manifest.source_supervised_tokens == {
        "toolace": 300_000,
        "hermes_function_calling": 200_000,
        "tinyllm_devops": 200_000,
        "m6_domain_replay": 200_000,
        "m2_no_tool_replay": 100_000,
    }
    assert manifest.language_supervised_tokens == {"en": 700_000, "zh": 300_000}
    assert manifest.mode_supervised_tokens == {"nonthinking": 940_000, "thinking": 60_000}
    assert report.training_permitted is True
    assert len(dataset) == manifest.sequence_count
    assert set(dataset[0]) == {"input_ids", "labels", "attention_mask"}
    assert tuple(dataset[0]["input_ids"].shape) == (2048,)

    source = tmp_path / "frozen" / manifest.dataset_version
    cases = {
        "missing-array": "arrays differ",
        "token-shape": "token arrays",
        "dtype": "dtype differs",
        "non-binary-mask": "not binary",
        "supervised-padding": "padding carries",
        "label-mismatch": "labels are not masked",
        "unknown-code": "unknown codes",
        "accounting": "token accounting",
    }
    for case, expected_error in cases.items():
        corrupted = tmp_path / f"corrupted-{case}" / manifest.dataset_version
        shutil.copytree(source, corrupted)
        sequence_path = corrupted / "sequences.npz"
        with np.load(sequence_path, allow_pickle=False) as opened:
            arrays = {name: np.asarray(opened[name]).copy() for name in opened.files}
        if case == "missing-array":
            arrays.pop("modes")
        elif case == "token-shape":
            arrays["input_ids"] = arrays["input_ids"][:, :-1]
        elif case == "dtype":
            arrays["input_ids"] = arrays["input_ids"].astype("<i8")
        elif case == "non-binary-mask":
            arrays["attention_masks"][0, 0] = 2
        elif case == "supervised-padding":
            arrays["attention_masks"][0, -1] = 0
        elif case == "label-mismatch":
            arrays["labels"][0, 1] += 1
        elif case == "unknown-code":
            arrays["source_ids"][0] = 99
        else:
            arrays["source_ids"][0] = 1
        with sequence_path.open("wb") as handle:
            cast(Any, np.savez)(handle, **arrays)
        manifest_path = corrupted / "manifest.json"
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_value["artifact"]["size_bytes"] = sequence_path.stat().st_size
        manifest_value["artifact"]["sha256"] = hashlib.sha256(
            sequence_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        commit_path = corrupted / "COMMITTED.json"
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["files"]["sequences.npz"] = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
        commit["files"]["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        commit_path.write_text(
            json.dumps(commit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(M10MixtureError, match=expected_error):
            open_frozen_mixture(corrupted)
