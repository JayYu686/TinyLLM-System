from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentRunRequest,
    AgentRunStore,
    AgentStoreError,
    AgentToolCall,
    agent_tool_call_sha256,
)

NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _request(text: str = "diagnose the run") -> AgentRunRequest:
    return AgentRunRequest.model_validate({"messages": [{"role": "user", "content": text}]})


def test_agent_store_create_is_idempotent_and_content_is_private(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    first, created = store.create(
        _request(), idempotency_key="client-create-operation-0001", now=NOW
    )
    repeated, repeated_created = store.create(
        _request(), idempotency_key="client-create-operation-0001", now=NOW
    )

    assert created is True
    assert repeated_created is False
    assert first == repeated
    run_dir = tmp_path / "agent-runs" / first.run_id
    assert "diagnose the run" not in (run_dir / "run.json").read_text(encoding="utf-8")
    assert "diagnose the run" in (run_dir / "request.json").read_text(encoding="utf-8")
    assert (run_dir / "run.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(AgentStoreError, match="another request"):
        store.create(
            _request("different"),
            idempotency_key="client-create-operation-0001",
            now=NOW,
        )


def test_agent_store_events_are_monotonic_and_replayable(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0002", now=NOW)
    first = store.append_event(record.run_id, "run.started", {"model": "production"}, now=NOW)
    second = store.append_event(
        record.run_id,
        "tool.call.proposed",
        {"tool": "search_evidence", "arguments": {"query": "failure"}},
        now=NOW + timedelta(seconds=1),
    )

    assert (first.sequence, second.sequence) == (1, 2)
    assert store.events_after(record.run_id) == (first, second)
    assert store.events_after(record.run_id, last_event_id=1) == (second,)
    assert store.load(record.run_id).last_event_sequence == 2
    with pytest.raises(AgentStoreError, match="non-negative"):
        store.events_after(record.run_id, last_event_id=-1)


def test_agent_store_approval_is_idempotent(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0003", now=NOW)
    approval_id = "approval-123456abcdef"
    pending_call = AgentToolCall(
        call_id="call_patch_1",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={},
    )
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id=approval_id,
        pending_tool_call=pending_call,
        now=NOW + timedelta(seconds=1),
    )
    decision = AgentApprovalDecision(
        approval_id=approval_id,
        tool_call_sha256=agent_tool_call_sha256(pending_call),
        decision="approved",
        idempotency_key="client-approval-operation-0001",
        decided_at=NOW + timedelta(seconds=2),
    )

    assert store.decide_approval(record.run_id, decision) is True
    assert store.decide_approval(record.run_id, decision) is False
    store.transition(record.run_id, status="running", now=NOW + timedelta(seconds=3))
    assert store.decide_approval(record.run_id, decision) is False
    conflict = decision.model_copy(update={"decision": "rejected"})
    with pytest.raises(AgentStoreError, match="different decision"):
        store.decide_approval(record.run_id, conflict)


def test_agent_store_explicit_cancel_is_idempotent(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0004", now=NOW)
    cancelled, changed = store.cancel(record.run_id, now=NOW + timedelta(seconds=1))
    repeated, repeated_changed = store.cancel(record.run_id, now=NOW + timedelta(seconds=2))

    assert changed is True
    assert repeated_changed is False
    assert cancelled.status == repeated.status == "cancelled"
    assert cancelled.completed_at is not None
    with pytest.raises(AgentStoreError, match="terminal"):
        store.append_event(record.run_id, "run.failed", {"code": "CANCELLED"})


def test_agent_store_expires_waiting_run_at_immutable_deadline(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(
        _request(),
        idempotency_key="client-expiry-operation-0001",
        now=NOW,
        expires_in=timedelta(seconds=10),
    )
    pending_call = AgentToolCall(
        call_id="call_expiry",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={"updates": {"seed": 42}},
    )
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id="approval-123456abcdea",
        pending_tool_call=pending_call,
        now=NOW + timedelta(seconds=1),
    )

    before, before_changed = store.expire_if_due(record.run_id, now=NOW + timedelta(seconds=9))
    expired, changed = store.expire_if_due(record.run_id, now=NOW + timedelta(seconds=30))
    repeated, repeated_changed = store.expire_if_due(record.run_id, now=NOW + timedelta(seconds=40))

    assert before.status == "waiting_approval"
    assert before_changed is False
    assert changed is True
    assert repeated_changed is False
    assert expired.status == repeated.status == "expired"
    assert expired.completed_at == record.expires_at
    assert store.events_after(record.run_id)[-1].data["code"] == "AGENT_RUN_EXPIRED"


def test_agent_store_rejects_corrupt_event_sequence(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0005", now=NOW)
    store.append_event(record.run_id, "run.started", {}, now=NOW)
    event_path = tmp_path / "agent-runs" / record.run_id / "events.jsonl"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["sequence"] = 2
    event_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AgentStoreError, match="monotonic"):
        store.events_after(record.run_id)


def test_agent_store_recovers_cursor_after_event_append_crash(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0007", now=NOW)
    first = store.append_event(record.run_id, "run.started", {}, now=NOW)
    run_path = tmp_path / "agent-runs" / record.run_id / "run.json"
    stale = json.loads(run_path.read_text(encoding="utf-8"))
    stale["last_event_sequence"] = 0
    run_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    second = store.append_event(
        record.run_id,
        "message.completed",
        {"content": "grounded answer"},
        now=NOW + timedelta(seconds=1),
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert store.load(record.run_id).last_event_sequence == 2


def test_agent_store_rejects_unsafe_roots_keys_and_transitions(tmp_path: Path) -> None:
    with pytest.raises(AgentStoreError, match="absolute"):
        AgentRunStore(Path("relative"))
    store = AgentRunStore(tmp_path)
    with pytest.raises(AgentStoreError, match="Idempotency-Key"):
        store.create(_request(), idempotency_key="short", now=NOW)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0006", now=NOW)
    with pytest.raises(AgentStoreError, match="transition"):
        store.transition(
            record.run_id,
            status="failed",
            now=NOW + timedelta(seconds=1),
        )


def test_langgraph_checkpoint_path_is_private_and_reused(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0008", now=NOW)

    first = store.langgraph_checkpoint_path(record.run_id)
    second = store.langgraph_checkpoint_path(record.run_id)

    assert first == second
    assert first.stat().st_mode & 0o777 == 0o600
    assert first.parent.stat().st_mode & 0o777 == 0o700


def test_approval_rejects_tool_call_hash_drift(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(_request(), idempotency_key="client-create-operation-0009", now=NOW)
    call = AgentToolCall(
        call_id="call_patch_2",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={"source_relative_path": "configs/train.yaml", "updates": {"seed": 42}},
    )
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id="approval-abcdef123456",
        pending_tool_call=call,
        now=NOW + timedelta(seconds=1),
    )
    with pytest.raises(AgentStoreError, match="bound"):
        store.decide_approval(
            record.run_id,
            AgentApprovalDecision(
                approval_id="approval-abcdef123456",
                tool_call_sha256="0" * 64,
                decision="approved",
                idempotency_key="client-approval-operation-0002",
                decided_at=NOW + timedelta(seconds=2),
            ),
        )
