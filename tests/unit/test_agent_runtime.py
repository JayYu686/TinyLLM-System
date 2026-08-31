from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentRunRequest,
    AgentRunStore,
    agent_tool_call_sha256,
    load_agent_config,
)
from tinyllm.agent.runtime import AgentRuntime, _explicit_read_plan, _normalize_tool_calls
from tinyllm.agent.schema import AgentMessage, AgentModelDecision

NOW = datetime.now(UTC)


class _Model:
    def __init__(self, decisions: list[AgentModelDecision]) -> None:
        self.decisions = deque(decisions)

    async def decide(
        self,
        *,
        messages: Any,
        observations: Any,
        mode: str,
        allowed_tools: Any,
    ) -> AgentModelDecision:
        del messages, observations, mode, allowed_tools
        return self.decisions.popleft()


class _Policy:
    def __init__(self, approval_required: bool) -> None:
        self.approval_required = approval_required


class _Client:
    def __init__(self, *, approval_required: bool = False) -> None:
        self.approval_required = approval_required
        self.tool_names = (
            "search_evidence",
            "get_run",
            "read_log_excerpt",
            "query_metrics",
            "inspect_config",
            "apply_sandbox_config_patch",
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def policy(self, name: str) -> _Policy:
        assert name in self.tool_names
        return _Policy(self.approval_required or name == "apply_sandbox_config_patch")

    async def validate_call(self, name: str, arguments: dict[str, Any]) -> None:
        assert name in self.tool_names
        assert isinstance(arguments, dict)

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"schema_version": "1.0", "evidence": "grounded"}


def _decision(tool_name: str, arguments: dict[str, Any] | None = None) -> AgentModelDecision:
    return AgentModelDecision.model_validate(
        {
            "tool_calls": [
                {
                    "call_id": "call_one",
                    "server_id": "tinyllm-devops",
                    "tool_name": tool_name,
                    "arguments": arguments or {},
                }
            ]
        }
    )


def _run(tmp_path: Path) -> tuple[AgentRunStore, str, tuple[AgentMessage, ...]]:
    store = AgentRunStore(tmp_path)
    request = AgentRunRequest.model_validate(
        {"messages": [{"role": "user", "content": "diagnose"}]}
    )
    record, _ = store.create(request, idempotency_key="runtime-create-operation-0001", now=NOW)
    return store, record.run_id, request.messages


def test_runtime_executes_read_tool_then_returns_grounded_answer(tmp_path: Path) -> None:
    store, run_id, messages = _run(tmp_path)
    client = _Client()
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model(
            [
                _decision("search_evidence", {"query": "failure"}),
                AgentModelDecision(message="ok [evidence:call_one]"),
            ]
        ),
        clients={"tinyllm-devops": client},  # type: ignore[dict-item]
    )

    answer = asyncio.run(runtime.run(run_id, messages=messages))

    assert answer == (
        "ok [evidence:call_one]\n\nEvidence trace: search_evidence: failure [evidence:call_one]"
    )
    assert store.load(run_id).status == "succeeded"
    assert client.calls == [("search_evidence", {"query": "failure"})]
    assert [event.event_type for event in store.events_after(run_id)] == [
        "run.started",
        "tool.call.proposed",
        "tool.started",
        "tool.completed",
        "model.delta",
        "message.completed",
        "run.completed",
    ]


def test_runtime_suspends_write_and_resumes_only_after_approval(tmp_path: Path) -> None:
    store, run_id, messages = _run(tmp_path)
    client = _Client()
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model(
            [
                _decision("apply_sandbox_config_patch", {"updates": {"seed": 42}}),
                AgentModelDecision(message="patched [evidence:call_one]"),
            ]
        ),
        clients={"tinyllm-devops": client},  # type: ignore[dict-item]
    )

    assert asyncio.run(runtime.run(run_id, messages=messages)) is None
    waiting = store.load(run_id)
    assert waiting.status == "waiting_approval"
    assert client.calls == []
    assert waiting.pending_approval_id is not None
    assert waiting.pending_tool_call is not None
    store.decide_approval(
        run_id,
        AgentApprovalDecision(
            approval_id=waiting.pending_approval_id,
            tool_call_sha256=agent_tool_call_sha256(waiting.pending_tool_call),
            decision="approved",
            idempotency_key="runtime-approval-operation-0001",
            decided_at=NOW,
        ),
    )

    answer = asyncio.run(runtime.resume_after_approval(run_id, messages=messages))

    assert answer == (
        "patched [evidence:call_one]\n\n"
        "Evidence trace: apply_sandbox_config_patch [evidence:call_one]"
    )
    assert store.load(run_id).status == "succeeded"
    assert client.calls[0][1]["run_id"] == run_id
    assert client.calls[0][1]["approval_id"] == waiting.pending_approval_id


def test_runtime_stops_repeated_identical_tool_loop(tmp_path: Path) -> None:
    store, run_id, messages = _run(tmp_path)
    repeated = _decision("search_evidence", {"query": "same"})
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model([repeated, repeated, repeated]),
        clients={"tinyllm-devops": _Client()},  # type: ignore[dict-item]
    )

    assert asyncio.run(runtime.run(run_id, messages=messages)) is None
    record = store.load(run_id)
    assert record.status == "failed"
    assert record.error_code == "AGENT_TOOL_LOOP"


def test_runtime_suppresses_one_duplicate_successful_call_and_reprompts(
    tmp_path: Path,
) -> None:
    store, run_id, messages = _run(tmp_path)
    client = _Client()
    repeated = _decision("search_evidence", {"query": "same"})
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model(
            [
                repeated,
                repeated,
                AgentModelDecision(message="done [evidence:call_one]"),
            ]
        ),
        clients={"tinyllm-devops": client},  # type: ignore[dict-item]
    )

    answer = asyncio.run(runtime.run(run_id, messages=messages))

    assert answer == (
        "done [evidence:call_one]\n\nEvidence trace: search_evidence: same [evidence:call_one]"
    )
    assert store.load(run_id).status == "succeeded"
    assert client.calls == [("search_evidence", {"query": "same"})]
    events = store.events_after(run_id)
    assert sum(event.event_type == "tool.started" for event in events) == 1
    assert sum(event.event_type == "tool.completed" for event in events) == 1


def test_runtime_deduplicates_identical_calls_within_one_model_decision(
    tmp_path: Path,
) -> None:
    store, run_id, messages = _run(tmp_path)
    client = _Client()
    duplicated = AgentModelDecision.model_validate(
        {
            "tool_calls": [
                {
                    "call_id": f"call_{index}",
                    "server_id": "tinyllm-devops",
                    "tool_name": "search_evidence",
                    "arguments": {"query": "same"},
                }
                for index in (1, 2)
            ]
        }
    )
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model([duplicated, AgentModelDecision(message="done [evidence:call_1]")]),
        clients={"tinyllm-devops": client},  # type: ignore[dict-item]
    )

    answer = asyncio.run(runtime.run(run_id, messages=messages))

    assert answer == (
        "done [evidence:call_1]\n\nEvidence trace: search_evidence: same [evidence:call_1]"
    )
    assert store.load(run_id).status == "succeeded"
    assert client.calls == [("search_evidence", {"query": "same"})]


def test_runtime_attaches_actual_evidence_identity_when_model_omits_citation(
    tmp_path: Path,
) -> None:
    store, run_id, messages = _run(tmp_path)
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model(
            [_decision("search_evidence", {"query": "failure"}), AgentModelDecision(message="ok")]
        ),
        clients={"tinyllm-devops": _Client()},  # type: ignore[dict-item]
    )

    answer = asyncio.run(runtime.run(run_id, messages=messages))
    record = store.load(run_id)
    assert record.status == "succeeded"
    assert answer == "ok\n\nEvidence trace: search_evidence: failure [evidence:call_one]"


def test_normalizes_run_id_and_orders_same_decision_by_request() -> None:
    run_identifier = "20260820T010000Z-agent-eval-a1b2c3d4-0001"
    log_path = f"runs/m9/{run_identifier}/logs/trainer.log"
    metrics_path = f"runs/m9/{run_identifier}/metrics.jsonl"
    messages = (
        AgentMessage(
            role="user",
            content=(
                f"Read Run {run_identifier}. Then read {log_path}, then query {metrics_path}."
            ),
        ),
    )
    decision = AgentModelDecision.model_validate(
        {
            "tool_calls": [
                {
                    "call_id": "call_metrics",
                    "server_id": "tinyllm-devops",
                    "tool_name": "query_metrics",
                    "arguments": {"relative_path": metrics_path},
                },
                {
                    "call_id": "call_log",
                    "server_id": "tinyllm-devops",
                    "tool_name": "read_log_excerpt",
                    "arguments": {"relative_path": log_path},
                },
                {
                    "call_id": "call_run",
                    "server_id": "tinyllm-devops",
                    "tool_name": "get_run",
                    "arguments": {"run_id": f"runs/m9/{run_identifier}"},
                },
            ]
        }
    )
    calls = _normalize_tool_calls(decision.tool_calls, messages=messages)

    assert [call.tool_name for call in calls] == ["get_run", "read_log_excerpt", "query_metrics"]
    assert calls[0].arguments == {"run_id": run_identifier}


def test_repairs_only_file_type_incompatible_read_routes() -> None:
    log_path = "runs/m9/run-one/logs/trainer.log"
    metrics_path = "runs/m9/run-one/metrics.jsonl"
    messages = (
        AgentMessage(
            role="user",
            content=f"Read {log_path} and query {metrics_path}.",
        ),
    )
    decision = AgentModelDecision.model_validate(
        {
            "tool_calls": [
                {
                    "call_id": "call_log",
                    "server_id": "tinyllm-devops",
                    "tool_name": "inspect_config",
                    "arguments": {"relative_path": log_path},
                },
                {
                    "call_id": "call_metrics",
                    "server_id": "tinyllm-devops",
                    "tool_name": "read_log_excerpt",
                    "arguments": {"relative_path": metrics_path},
                },
            ]
        }
    )
    calls = _normalize_tool_calls(decision.tool_calls, messages=messages)

    assert [call.tool_name for call in calls] == ["read_log_excerpt", "query_metrics"]


def test_runtime_executes_explicit_read_plan_before_model_decision(tmp_path: Path) -> None:
    store, run_id, _ = _run(tmp_path)
    run_identifier = "20260820T010000Z-agent-eval-a1b2c3d4-0001"
    log_path = f"runs/m9/{run_identifier}/logs/trainer.log"
    metrics_path = f"runs/m9/{run_identifier}/metrics.jsonl"
    messages = (
        AgentMessage(
            role="user",
            content=(
                f"Read Run {run_identifier}. Then read {log_path} lines 4 through 9. "
                f"Then query {metrics_path} for loss, latest 7."
            ),
        ),
    )
    client = _Client()
    runtime = AgentRuntime(
        config=load_agent_config(Path("configs/agent/m8_devops.yaml")),
        store=store,
        model=_Model([AgentModelDecision(message="done")]),
        clients={"tinyllm-devops": client},  # type: ignore[dict-item]
    )

    answer = asyncio.run(runtime.run(run_id, messages=messages))

    assert client.calls == [
        ("get_run", {"run_id": run_identifier}),
        (
            "read_log_excerpt",
            {"relative_path": log_path, "start_line": 4, "end_line": 9},
        ),
        (
            "query_metrics",
            {"relative_path": metrics_path, "metric_names": ["loss"], "limit": 7},
        ),
    ]
    assert answer is not None
    assert answer.count("[evidence:") == 3
    assert log_path in answer
    assert metrics_path in answer


def test_explicit_read_plan_rejects_unrequested_negated_and_write_paths() -> None:
    allowed = (
        "tinyllm-devops.get_run",
        "tinyllm-devops.read_log_excerpt",
        "tinyllm-devops.inspect_config",
    )
    messages = (
        AgentMessage(
            role="user",
            content=(
                "Do not read runs/m9/run-one/logs/trainer.log. "
                "Reference: runs/m9/run-one/logs/other.log. "
                "Change deployments/prod/config.yaml to use the candidate."
            ),
        ),
    )

    assert _explicit_read_plan(messages=messages, allowed_tools=allowed) == ()
