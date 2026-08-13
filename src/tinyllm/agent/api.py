"""FastAPI Agent endpoints with durable idempotency, replay, and explicit cancellation."""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tinyllm.agent.runtime import AgentRuntime, AgentRuntimeError
from tinyllm.agent.schema import (
    AgentApprovalDecision,
    AgentApprovalRequest,
    AgentEvent,
    AgentRunRecord,
    AgentRunRequest,
)
from tinyllm.agent.store import TERMINAL_STATUSES, AgentRunStore, AgentStoreError


class AgentExecutionService:
    """Own in-process tasks while durable Store state remains the source of truth."""

    def __init__(
        self,
        *,
        store: AgentRunStore,
        runtime: AgentRuntime,
        run_timeout_seconds: float,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.run_timeout_seconds = run_timeout_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, record: AgentRunRecord) -> None:
        if record.run_id in self._tasks and not self._tasks[record.run_id].done():
            return
        self._tasks[record.run_id] = asyncio.create_task(
            self._execute(record.run_id), name=f"tinyllm-agent-{record.run_id}"
        )

    async def _execute(self, run_id: str) -> None:
        try:
            request = self.store.load_request(run_id)
            observations = tuple(
                event.data
                for event in self.store.events_after(run_id)
                if event.event_type == "tool.completed"
            )
            async with asyncio.timeout(self.run_timeout_seconds):
                await self.runtime.run(
                    run_id,
                    messages=request.messages,
                    observations=observations,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._fail_if_running(run_id, "AGENT_RUN_TIMEOUT")
        except (AgentRuntimeError, AgentStoreError, ValueError, OSError):
            self._fail_if_running(run_id, "AGENT_RUNTIME_FAILED")

    def resume(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            raise AgentStoreError("Agent Run is already executing")
        self._tasks[run_id] = asyncio.create_task(
            self._resume(run_id), name=f"tinyllm-agent-resume-{run_id}"
        )

    async def _resume(self, run_id: str) -> None:
        try:
            request = self.store.load_request(run_id)
            observations = tuple(
                event.data
                for event in self.store.events_after(run_id)
                if event.event_type == "tool.completed"
            )
            async with asyncio.timeout(self.run_timeout_seconds):
                await self.runtime.resume_after_approval(
                    run_id,
                    messages=request.messages,
                    observations=observations,
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._fail_if_running(run_id, "AGENT_RUN_TIMEOUT")
        except (AgentRuntimeError, AgentStoreError, ValueError, OSError):
            self._fail_if_running(run_id, "AGENT_RUNTIME_FAILED")

    def recover(self) -> int:
        """Schedule only active runs; waiting approvals remain durable and idle."""

        recovered = 0
        for record in self.store.list_records():
            if record.status in {"created", "running"}:
                self.start(record)
                recovered += 1
        return recovered

    def cancel(self, run_id: str) -> tuple[AgentRunRecord, bool]:
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        return self.store.cancel(run_id)

    def _fail_if_running(self, run_id: str, code: str) -> None:
        record = self.store.load(run_id)
        if record.status in TERMINAL_STATUSES:
            return
        self.store.append_event(run_id, "run.failed", {"code": code})
        self.store.transition(run_id, status="failed", error_code=code)


def _sse(event: AgentEvent) -> bytes:
    data = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {data}\n\n".encode()


def create_agent_router(
    *,
    store: AgentRunStore,
    service: AgentExecutionService,
    bearer_token: str,
    poll_seconds: float = 0.1,
) -> APIRouter:
    """Create the frozen M8 Agent API router without exposing runtime internals."""

    if len(bearer_token) < 32:
        raise ValueError("Agent API Bearer Token must contain at least 32 characters")
    router = APIRouter(prefix="/v1/agent", tags=["agent"])
    security = HTTPBearer(auto_error=False)
    auth_dependency = Security(security)

    async def authenticate(
        credentials: HTTPAuthorizationCredentials | None = auth_dependency,
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(credentials.credentials, bearer_token)
        ):
            raise HTTPException(status_code=401, detail="invalid authentication credentials")

    def required_idempotency(value: str | None) -> str:
        if value is None:
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        return value

    @router.post("/runs", dependencies=[Security(authenticate)])
    async def create_run(
        body: AgentRunRequest,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AgentRunRecord:
        try:
            record, created = store.create(
                body, idempotency_key=required_idempotency(idempotency_key)
            )
            if created:
                service.start(record)
            return record
        except AgentStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}", dependencies=[Security(authenticate)])
    async def get_run(run_id: str) -> AgentRunRecord:
        try:
            return store.load(run_id)
        except AgentStoreError as exc:
            raise HTTPException(status_code=404, detail="Agent Run was not found") from exc

    @router.get("/runs/{run_id}/events", dependencies=[Security(authenticate)])
    async def get_events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        if last_event_id is not None and last_event_id < 0:
            raise HTTPException(status_code=400, detail="Last-Event-ID must be non-negative")
        try:
            store.load(run_id)
        except AgentStoreError as exc:
            raise HTTPException(status_code=404, detail="Agent Run was not found") from exc

        async def stream() -> AsyncIterator[bytes]:
            cursor = last_event_id or 0
            while True:
                for event in store.events_after(run_id, cursor):
                    cursor = event.sequence
                    yield _sse(event)
                record = store.load(run_id)
                if record.status in TERMINAL_STATUSES and cursor >= record.last_event_sequence:
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(poll_seconds)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.post(
        "/runs/{run_id}/approvals/{approval_id}",
        dependencies=[Security(authenticate)],
    )
    async def decide_approval(
        run_id: str,
        approval_id: str,
        body: AgentApprovalRequest,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AgentRunRecord:
        try:
            decision = AgentApprovalDecision(
                approval_id=approval_id,
                decision=body.decision,
                idempotency_key=required_idempotency(idempotency_key),
                decided_at=datetime.now(UTC),
            )
            created = store.decide_approval(run_id, decision)
            if created:
                service.resume(run_id)
            return store.load(run_id)
        except AgentStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/cancel", dependencies=[Security(authenticate)])
    async def cancel_run(run_id: str) -> AgentRunRecord:
        try:
            record, _changed = service.cancel(run_id)
            return record
        except AgentStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
