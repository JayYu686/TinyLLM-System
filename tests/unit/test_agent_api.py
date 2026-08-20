from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyllm.agent.api import AgentExecutionService, create_agent_router
from tinyllm.agent.runtime import AgentRuntimeError
from tinyllm.agent.schema import AgentModelDecision, AgentRunRequest, AgentToolCall
from tinyllm.agent.store import AgentRunStore

TOKEN = "test-agent-bearer-token-with-at-least-32-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
REQUEST = {
    "schema_version": "1.0",
    "model": "production",
    "messages": [{"role": "user", "content": "diagnose"}],
    "mode": "nonthinking",
    "mcp_server_ids": ["tinyllm-devops"],
    "max_steps": 8,
}


class _ImmediateRuntime:
    async def run(self, run_id: str, *, messages: object, observations: object = ()) -> str:
        del messages, observations
        self.store.append_event(run_id, "message.completed", {"content": "grounded"})
        self.store.append_event(run_id, "run.completed", {"status": "succeeded"})
        self.store.transition(run_id, status="succeeded")
        return "grounded"

    async def resume_after_approval(
        self, run_id: str, *, messages: object, observations: object = ()
    ) -> str:
        return await self.run(run_id, messages=messages, observations=observations)

    def __init__(self, store: AgentRunStore) -> None:
        self.store = store
        self.model: object | None = None


def _client(tmp_path: Path) -> tuple[TestClient, AgentRunStore]:
    store = AgentRunStore(tmp_path)
    runtime = _ImmediateRuntime(store)
    service = AgentExecutionService(
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        run_timeout_seconds=10,
    )
    app = FastAPI()
    app.include_router(create_agent_router(store=store, service=service, bearer_token=TOKEN))
    return TestClient(app), store


def test_agent_api_requires_auth_and_create_idempotency(tmp_path: Path) -> None:
    client, _store = _client(tmp_path)
    assert client.post("/v1/agent/runs", json=REQUEST).status_code == 401
    assert client.post("/v1/agent/runs", headers=AUTH, json=REQUEST).status_code == 400
    headers = {**AUTH, "Idempotency-Key": "api-create-operation-0001"}
    with client:
        first = client.post("/v1/agent/runs", headers=headers, json=REQUEST)
        repeated = client.post("/v1/agent/runs", headers=headers, json=REQUEST)
    assert first.status_code == repeated.status_code == 200
    assert first.json()["run_id"] == repeated.json()["run_id"]


def test_agent_api_sse_replays_after_last_event_id(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    record, _ = store.create(
        AgentRunRequest.model_validate(REQUEST),
        idempotency_key="api-events-operation-0001",
        now=datetime.now(UTC),
    )
    store.transition(record.run_id, status="running")
    store.append_event(record.run_id, "run.started", {"model": "production"})
    store.append_event(record.run_id, "message.completed", {"content": "done"})
    store.append_event(record.run_id, "run.completed", {"status": "succeeded"})
    store.transition(record.run_id, status="succeeded")

    response = client.get(
        f"/v1/agent/runs/{record.run_id}/events",
        headers={**AUTH, "Last-Event-ID": "1"},
    )

    assert response.status_code == 200
    assert "id: 1\n" not in response.text
    assert "id: 2\n" in response.text
    assert "event: run.completed" in response.text
    invalid = client.get(
        f"/v1/agent/runs/{record.run_id}/events",
        headers={**AUTH, "Last-Event-ID": "-1"},
    )
    assert invalid.status_code == 400


def test_agent_api_explicit_cancel_is_idempotent(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    request = AgentRunRequest.model_validate(REQUEST)
    record, _ = store.create(
        request, idempotency_key="api-cancel-operation-0001", now=datetime.now(UTC)
    )

    first = client.post(f"/v1/agent/runs/{record.run_id}/cancel", headers=AUTH)
    second = client.post(f"/v1/agent/runs/{record.run_id}/cancel", headers=AUTH)

    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == second.json()["status"] == "cancelled"


def test_agent_execution_recovery_skips_waiting_approval(tmp_path: Path) -> None:
    store = AgentRunStore(tmp_path)
    runtime = _ImmediateRuntime(store)
    service = AgentExecutionService(
        store=store,
        runtime=runtime,  # type: ignore[arg-type]
        run_timeout_seconds=10,
    )
    assert service.recover() == 0
    assert AgentModelDecision(message="ok").message == "ok"


def test_agent_api_replays_approval_after_run_progressed(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    record, _ = store.create(
        AgentRunRequest.model_validate(REQUEST),
        idempotency_key="api-approval-create-operation-0001",
        now=datetime.now(UTC),
    )
    approval_id = "approval-123456abcdef"
    store.transition(
        record.run_id,
        status="waiting_approval",
        pending_approval_id=approval_id,
        pending_tool_call=AgentToolCall(
            call_id="call_api_patch",
            server_id="tinyllm-devops",
            tool_name="apply_sandbox_config_patch",
            arguments={"updates": {"seed": 42}},
        ),
    )
    headers = {**AUTH, "Idempotency-Key": "api-approval-operation-0001"}

    with client:
        first = client.post(
            f"/v1/agent/runs/{record.run_id}/approvals/{approval_id}",
            headers=headers,
            json={"schema_version": "1.0", "decision": "approved"},
        )
        for _ in range(100):
            if store.load(record.run_id).status == "succeeded":
                break
        repeated = client.post(
            f"/v1/agent/runs/{record.run_id}/approvals/{approval_id}",
            headers=headers,
            json={"schema_version": "1.0", "decision": "approved"},
        )
        conflict = client.post(
            f"/v1/agent/runs/{record.run_id}/approvals/{approval_id}",
            headers={**AUTH, "Idempotency-Key": "api-approval-operation-0002"},
            json={"schema_version": "1.0", "decision": "approved"},
        )

    assert first.status_code == repeated.status_code == 200
    assert repeated.json()["run_id"] == record.run_id
    assert conflict.status_code == 409


def test_agent_api_maps_missing_and_conflicting_resources(tmp_path: Path) -> None:
    client, store = _client(tmp_path)
    with pytest.raises(ValueError, match="32"):
        create_agent_router(
            store=store,
            service=cast(AgentExecutionService, object()),
            bearer_token="short",
        )
    headers = {**AUTH, "Idempotency-Key": "api-conflict-operation-0001"}
    wrong_model = {**REQUEST, "model": "unregistered"}
    assert client.post("/v1/agent/runs", headers=headers, json=wrong_model).status_code == 404
    assert client.get("/v1/agent/runs/invalid", headers=AUTH).status_code == 404
    assert client.get("/v1/agent/runs/invalid/events", headers=AUTH).status_code == 404
    assert client.post("/v1/agent/runs/invalid/cancel", headers=AUTH).status_code == 409

    record, _ = store.create(
        AgentRunRequest.model_validate(REQUEST),
        idempotency_key="api-conflict-create-0001",
        now=datetime.now(UTC),
    )
    approval_url = f"/v1/agent/runs/{record.run_id}/approvals/approval-123456abcdef"
    assert client.post(approval_url, headers=AUTH, json={"decision": "approved"}).status_code == 400
    assert (
        client.post(
            approval_url,
            headers={**AUTH, "Idempotency-Key": "api-conflict-approval-0001"},
            json={"decision": "approved"},
        ).status_code
        == 409
    )
    store.cancel(record.run_id)
    assert client.post(f"/v1/agent/runs/{record.run_id}/cancel", headers=AUTH).status_code == 200


class _FailureRuntime:
    def __init__(self, store: AgentRunStore, error: BaseException) -> None:
        self.store = store
        self.error = error
        self.model = _ClosableModel()

    async def run(self, run_id: str, *, messages: object, observations: object = ()) -> None:
        del run_id, messages, observations
        raise self.error

    async def resume_after_approval(
        self, run_id: str, *, messages: object, observations: object = ()
    ) -> None:
        del run_id, messages, observations
        raise self.error


class _ClosableModel:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _service_run(tmp_path: Path, key: str) -> tuple[AgentRunStore, str]:
    store = AgentRunStore(tmp_path)
    record, _ = store.create(
        AgentRunRequest.model_validate(REQUEST),
        idempotency_key=key,
        now=datetime.now(UTC),
    )
    return store, record.run_id


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (AgentRuntimeError("failed"), "AGENT_RUNTIME_FAILED"),
        (TimeoutError(), "AGENT_RUN_TIMEOUT"),
    ],
)
def test_execution_service_persists_runtime_failures(
    tmp_path: Path, error: BaseException, code: str
) -> None:
    store, run_id = _service_run(tmp_path, f"api-failure-{code.lower()}-0001")
    service = AgentExecutionService(
        store=store,
        runtime=cast(Any, _FailureRuntime(store, error)),
        run_timeout_seconds=10,
    )
    asyncio.run(service._execute(run_id))
    record = store.load(run_id)
    assert record.status == "failed"
    assert record.error_code == code
    service._fail_if_running(run_id, code, error)


def test_execution_service_recovers_cancels_and_closes(tmp_path: Path) -> None:
    store, run_id = _service_run(tmp_path, "api-recovery-operation-0001")
    runtime = _ImmediateRuntime(store)
    runtime.model = _ClosableModel()
    service = AgentExecutionService(
        store=store,
        runtime=cast(Any, runtime),
        run_timeout_seconds=10,
    )

    async def exercise() -> None:
        assert service.recover() == 1
        assert service.recover() == 1
        service.cancel(run_id)
        await service.shutdown()

    asyncio.run(exercise())
    assert isinstance(runtime.model, _ClosableModel)
    assert runtime.model.closed is True
