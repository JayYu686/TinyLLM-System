"""Bounded, resumable DevOps Agent orchestration with local tool authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Protocol

from tinyllm.agent.mcp_client import MCPClientError, MCPPolicyClient
from tinyllm.agent.schema import AgentConfig, AgentMessage, AgentModelDecision, AgentToolCall
from tinyllm.agent.store import AgentRunStore, AgentStoreError


class AgentRuntimeError(RuntimeError):
    """Raised when a bounded Agent run cannot proceed safely."""


class AgentModel(Protocol):
    """Interface implemented by the M7 OpenAI Gateway adapter."""

    async def decide(
        self,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]],
        mode: str,
        allowed_tools: Sequence[str],
    ) -> AgentModelDecision: ...


ApprovalIDFactory = Callable[[AgentToolCall], str]


def deterministic_approval_id(call: AgentToolCall) -> str:
    payload = json.dumps(call.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return f"approval-{hashlib.sha256(payload).hexdigest()[:12]}"


class AgentRuntime:
    """Execute model decisions under local limits; suspend before every sandbox write."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        store: AgentRunStore,
        model: AgentModel,
        clients: dict[str, MCPPolicyClient],
        approval_id_factory: ApprovalIDFactory = deterministic_approval_id,
    ) -> None:
        self.config = config
        self.store = store
        self.model = model
        self.clients = clients
        self.approval_id_factory = approval_id_factory

    async def run(
        self,
        run_id: str,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]] = (),
    ) -> str | None:
        """Run until completion or an approval-safe suspension point."""

        record = self.store.load(run_id)
        if record.status not in {"created", "running"}:
            raise AgentRuntimeError("Agent Run is not executable")
        if record.status == "created":
            self.store.transition(run_id, status="running")
            self.store.append_event(run_id, "run.started", {"model": record.model})
        transcript_observations = list(observations)
        signatures = [
            self._signature_from_event(event.data)
            for event in self.store.events_after(run_id)
            if event.event_type == "tool.call.proposed"
        ]
        last_signature = signatures[-1] if signatures else None
        repeated = 0
        for signature in reversed(signatures):
            if signature != last_signature:
                break
            repeated += 1
        while True:
            record = self.store.load(run_id)
            if record.steps_completed >= min(record.max_steps, self.config.max_steps):
                self._fail(run_id, "AGENT_STEP_LIMIT")
                return None
            allowed = self._allowed_tools(record.mcp_server_ids)
            decision = await self.model.decide(
                messages=messages,
                observations=transcript_observations,
                mode=record.mode,
                allowed_tools=allowed,
            )
            next_steps = record.steps_completed + 1
            self.store.transition(run_id, status="running", steps_completed=next_steps)
            if decision.message is not None:
                if transcript_observations and not any(
                    f"[evidence:{item.get('call_id')}]" in decision.message
                    for item in transcript_observations
                    if isinstance(item.get("call_id"), str)
                ):
                    self._fail(run_id, "AGENT_GROUNDING_REQUIRED")
                    return None
                self.store.append_event(run_id, "model.delta", {"content": decision.message})
                self.store.append_event(run_id, "message.completed", {"content": decision.message})
                self.store.append_event(run_id, "run.completed", {"status": "succeeded"})
                self.store.transition(run_id, status="succeeded")
                return decision.message
            for call in decision.tool_calls:
                current = self.store.load(run_id)
                if current.tool_calls_completed >= self.config.max_tool_calls:
                    self._fail(run_id, "AGENT_TOOL_LIMIT")
                    return None
                signature = json.dumps(
                    [call.server_id, call.tool_name, call.arguments],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                repeated = repeated + 1 if signature == last_signature else 1
                last_signature = signature
                if repeated > self.config.same_tool_consecutive_limit:
                    self._fail(run_id, "AGENT_TOOL_LOOP")
                    return None
                client = self._client(call)
                policy = client.policy(call.tool_name)
                self.store.append_event(
                    run_id,
                    "tool.call.proposed",
                    {
                        "call_id": call.call_id,
                        "server_id": call.server_id,
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                    },
                )
                if policy.approval_required:
                    approval_id = self.approval_id_factory(call)
                    self.store.append_event(
                        run_id,
                        "approval.required",
                        {
                            "approval_id": approval_id,
                            "call_id": call.call_id,
                            "server_id": call.server_id,
                            "tool_name": call.tool_name,
                            "arguments": call.arguments,
                        },
                    )
                    self.store.transition(
                        run_id,
                        status="waiting_approval",
                        pending_approval_id=approval_id,
                        pending_tool_call=call,
                        steps_completed=next_steps,
                    )
                    return None
                observation = await self._execute(run_id, call, client)
                transcript_observations.append(observation)

    async def resume_after_approval(
        self,
        run_id: str,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]] = (),
    ) -> str | None:
        """Continue an approved write from the persisted safe node."""

        record = self.store.load(run_id)
        call = record.pending_tool_call
        approval_id = record.pending_approval_id
        if record.status != "waiting_approval" or call is None or approval_id is None:
            raise AgentRuntimeError("Agent Run has no pending approval")
        decision = self.store.load_approval(run_id, approval_id)
        if decision.decision == "rejected":
            self._fail(run_id, "AGENT_APPROVAL_REJECTED")
            return None
        arguments = dict(call.arguments)
        arguments.setdefault("approval_id", approval_id)
        arguments.setdefault("run_id", run_id)
        approved_call = call.model_copy(update={"arguments": arguments})
        self.store.transition(
            run_id,
            status="running",
            pending_approval_id=None,
            pending_tool_call=None,
        )
        observation = await self._execute(run_id, approved_call, self._client(approved_call))
        return await self.run(
            run_id,
            messages=messages,
            observations=(*observations, observation),
        )

    def _allowed_tools(self, server_ids: Sequence[str]) -> tuple[str, ...]:
        names: list[str] = []
        for server_id in server_ids:
            client = self.clients.get(server_id)
            if client is None:
                raise AgentRuntimeError("requested MCP Server is not registered")
            names.extend(f"{server_id}.{name}" for name in client.tool_names)
        return tuple(names)

    def _client(self, call: AgentToolCall) -> MCPPolicyClient:
        client = self.clients.get(call.server_id)
        if client is None:
            raise AgentRuntimeError("model proposed an unregistered MCP Server")
        client.policy(call.tool_name)
        return client

    async def _execute(
        self, run_id: str, call: AgentToolCall, client: MCPPolicyClient
    ) -> dict[str, object]:
        self.store.append_event(
            run_id,
            "tool.started",
            {"call_id": call.call_id, "server_id": call.server_id, "tool_name": call.tool_name},
        )
        current = self.store.load(run_id)
        attempted = current.tool_calls_completed + 1
        self.store.transition(run_id, status="running", tool_calls_completed=attempted)
        try:
            result = await client.call(call.tool_name, call.arguments)
        except (MCPClientError, OSError, TimeoutError) as exc:
            return self._fail_observation(run_id, call, exc)
        observation: dict[str, object] = {
            "call_id": call.call_id,
            "server_id": call.server_id,
            "tool_name": call.tool_name,
            "result": result,
        }
        self.store.append_event(run_id, "tool.completed", observation)
        return observation

    @staticmethod
    def _signature_from_event(data: dict[str, object]) -> str:
        return json.dumps(
            [data.get("server_id"), data.get("tool_name"), data.get("arguments")],
            sort_keys=True,
            separators=(",", ":"),
        )

    def _fail_observation(
        self, run_id: str, call: AgentToolCall, error: BaseException
    ) -> dict[str, object]:
        observation: dict[str, object] = {
            "call_id": call.call_id,
            "server_id": call.server_id,
            "tool_name": call.tool_name,
            "error": type(error).__name__,
        }
        self.store.append_event(run_id, "tool.completed", observation)
        return observation

    def _fail(self, run_id: str, code: str) -> None:
        try:
            self.store.append_event(run_id, "run.failed", {"code": code})
            self.store.transition(run_id, status="failed", error_code=code)
        except AgentStoreError as exc:
            raise AgentRuntimeError("Agent failure state could not be persisted") from exc
        return None
