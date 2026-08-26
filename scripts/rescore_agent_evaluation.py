#!/usr/bin/env python3
"""Re-score immutable Agent outputs under a newer scoring protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tinyllm.agent_eval import (
    AgentEvalItemResult,
    AgentEvalSummary,
    AgentScoringProtocol,
    aggregate_results,
    canonical_json_sha256,
    load_suite,
    score_task,
)


class AgentRescoreError(ValueError):
    """Raised when source evidence cannot be re-scored exactly."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def rescore_evaluation(
    *,
    suite_directory: Path,
    source_directory: Path,
    output_directory: Path,
    protocol: AgentScoringProtocol,
    evaluated_at: datetime | None = None,
) -> AgentEvalSummary:
    """Re-score stored normalized calls and answers without model generation."""

    if output_directory.exists():
        raise AgentRescoreError("Agent re-score output already exists")
    manifest, tasks = load_suite(suite_directory)
    task_by_id = {item.task_id: item for item in tasks}
    try:
        source_summary_bytes = (source_directory / "summary.json").read_bytes()
        source_summary = AgentEvalSummary.model_validate_json(source_summary_bytes)
        source_items_bytes = (source_directory / "items.jsonl").read_bytes()
        source_results = tuple(
            AgentEvalItemResult.model_validate_json(line)
            for line in source_items_bytes.splitlines()
            if line.strip()
        )
    except (OSError, ValueError) as exc:
        raise AgentRescoreError("Agent re-score source evidence is invalid") from exc
    if (
        not source_summary.completed
        or source_summary.suite_version != manifest.suite_version
        or source_summary.suite_content_sha256 != manifest.content_sha256
        or source_summary.item_results_sha256 != hashlib.sha256(source_items_bytes).hexdigest()
        or len(source_results) != manifest.item_count
        or {item.task_id for item in source_results} != set(task_by_id)
    ):
        raise AgentRescoreError("Agent re-score source lineage differs from the suite")

    results = tuple(
        score_task(
            task_by_id[item.task_id],
            run_id=item.run_id,
            status=item.status,
            calls=item.calls,
            final_answer=item.final_answer,
            evidence_citations=item.evidence_citations,
            duration_milliseconds=item.duration_milliseconds,
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            unapproved_write_attempts=item.unapproved_write_attempts,
            path_escape_attempts=item.path_escape_attempts,
            arbitrary_command_attempts=item.arbitrary_command_attempts,
            failure_reason=item.failure_reason,
            scoring_protocol=protocol,
        )
        for item in source_results
    )
    result_bytes = b"".join(
        json.dumps(
            item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
        for item in results
    )
    timestamp = datetime.now(UTC) if evaluated_at is None else evaluated_at
    if timestamp.tzinfo is None:
        raise AgentRescoreError("Agent re-score timestamp must be timezone-aware")
    identity = canonical_json_sha256(
        {
            "protocol": protocol,
            "source_summary_sha256": hashlib.sha256(source_summary_bytes).hexdigest(),
            "source_items_sha256": hashlib.sha256(source_items_bytes).hexdigest(),
            "rescored_items_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    )
    summary = AgentEvalSummary(
        scoring_protocol=protocol,
        evaluation_id=f"m9-agent-eval-{identity[:8]}",
        evaluated_at=timestamp,
        suite_version=source_summary.suite_version,
        suite_content_sha256=source_summary.suite_content_sha256,
        model_id=source_summary.model_id,
        model_revision=source_summary.model_revision,
        model_artifact_sha256=source_summary.model_artifact_sha256,
        parent_model_id=source_summary.parent_model_id,
        deployment_record_sha256=source_summary.deployment_record_sha256,
        evaluation_subject_sha256=source_summary.evaluation_subject_sha256,
        environment_sha256=source_summary.environment_sha256,
        hardware_sha256=source_summary.hardware_sha256,
        physical_gpu_index=source_summary.physical_gpu_index,
        gpu_name=source_summary.gpu_name,
        driver_version=source_summary.driver_version,
        gateway_version=source_summary.gateway_version,
        agent_runtime_version=source_summary.agent_runtime_version,
        git_commit=source_summary.git_commit,
        git_dirty=source_summary.git_dirty,
        metrics=aggregate_results(results),
        item_results_sha256=hashlib.sha256(result_bytes).hexdigest(),
        completed=True,
    )
    source_record = {
        "schema_version": "1.0",
        "kind": "agent_evaluation_rescore",
        "protocol": protocol,
        "source_evaluation_id": source_summary.evaluation_id,
        "source_summary_sha256": hashlib.sha256(source_summary_bytes).hexdigest(),
        "source_items_sha256": hashlib.sha256(source_items_bytes).hexdigest(),
        "model_generation_repeated": False,
    }
    _atomic_write(output_directory / "items.jsonl", result_bytes)
    _atomic_write(output_directory / "summary.json", _json_bytes(summary.to_dict()))
    _atomic_write(output_directory / "source.json", _json_bytes(source_record))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-directory", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("m10-agent-scoring-v2",),
        default="m10-agent-scoring-v2",
    )
    args = parser.parse_args()
    try:
        summary = rescore_evaluation(
            suite_directory=args.suite_directory,
            source_directory=args.source_directory,
            output_directory=args.output_directory,
            protocol=args.protocol,
        )
    except (AgentRescoreError, OSError, ValueError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 2
    print(summary.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
