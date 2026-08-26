from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.rescore_agent_evaluation import rescore_evaluation
from tinyllm.agent_eval import (
    AgentEvalItemResult,
    AgentEvalSummary,
    aggregate_results,
    load_suite,
    score_task,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def test_rescore_preserves_generation_and_changes_only_protocol(tmp_path: Path) -> None:
    manifest, tasks = load_suite(Path("evals/agent/dev/v1"))
    source = tmp_path / "source"
    source.mkdir()
    results = tuple(
        score_task(
            task,
            run_id=f"run-{index}",
            status="succeeded",
            calls=(),
            final_answer="Need more information?",
        )
        for index, task in enumerate(tasks)
    )
    items = b"".join(
        json.dumps(
            item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
        for item in results
    )
    source_summary = AgentEvalSummary(
        evaluation_id="m9-agent-eval-12345678",
        evaluated_at=datetime(2026, 8, 25, tzinfo=UTC),
        suite_version=manifest.suite_version,
        suite_content_sha256=manifest.content_sha256,
        model_id="fixture-model",
        model_revision="revision-1",
        model_artifact_sha256="a" * 64,
        parent_model_id="fixture-parent",
        environment_sha256="b" * 64,
        hardware_sha256="c" * 64,
        physical_gpu_index=0,
        gpu_name="fixture-gpu",
        driver_version="fixture-driver",
        gateway_version="0.1.0",
        agent_runtime_version="0.1.0",
        git_commit="d" * 40,
        git_dirty=False,
        metrics=aggregate_results(results),
        item_results_sha256=hashlib.sha256(items).hexdigest(),
        completed=True,
    )
    (source / "items.jsonl").write_bytes(items)
    (source / "summary.json").write_bytes(_json_bytes(source_summary.to_dict()))

    output = tmp_path / "output"
    rescored = rescore_evaluation(
        suite_directory=Path("evals/agent/dev/v1"),
        source_directory=source,
        output_directory=output,
        protocol="m10-agent-scoring-v2",
        evaluated_at=datetime(2026, 8, 25, 1, tzinfo=UTC),
    )

    stored = tuple(
        AgentEvalItemResult.model_validate_json(line)
        for line in (output / "items.jsonl").read_bytes().splitlines()
    )
    assert rescored.scoring_protocol == "m10-agent-scoring-v2"
    assert len(stored) == 80
    assert all(item.scoring_protocol == "m10-agent-scoring-v2" for item in stored)
    assert json.loads((output / "source.json").read_bytes())["model_generation_repeated"] is False
