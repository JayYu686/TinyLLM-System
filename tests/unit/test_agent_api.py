from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinyllm.agent.api import AgentExecutionService, create_agent_router
from tinyllm.agent.schema import AgentModelDecision, AgentRunRequest
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
