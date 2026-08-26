"""Bounded LangGraph DevOps Agent with durable safe-node recovery."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypedDict, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tinyllm.agent.mcp_client import MCPClientError, MCPPolicyClient
from tinyllm.agent.schema import (
    AgentConfig,
    AgentMessage,
    AgentModelDecision,
    AgentToolCall,
    _reject_private_reasoning,
    agent_tool_call_sha256,
)
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
    return f"approval-{agent_tool_call_sha256(call)[:12]}"


class _GraphState(TypedDict, total=False):
    run_id: str
    messages: list[dict[str, Any]]
    observations: list[dict[str, object]]
    decision: dict[str, Any]
    pending_calls: list[dict[str, Any]]
    current_call: dict[str, Any]
    approval_id: str
    answer: str
    failed: bool
    retry_model: bool


class AgentRuntime:
    """Execute the frozen Agent graph and persist each safe node in SQLite."""

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

    def _builder(self) -> StateGraph[_GraphState, None, _GraphState, _GraphState]:
        graph = StateGraph(_GraphState)
        graph.add_node("receive_request", self._receive_request)
        graph.add_node("prepare_evidence_retrieval", self._prepare_evidence_retrieval)
        graph.add_node("model_decision", self._model_decision)
        graph.add_node("validate_tool_schema", self._validate_tool_schema)
        graph.add_node("enforce_tool_policy", self._enforce_tool_policy)
        graph.add_node("wait_for_approval", self._wait_for_approval)
        graph.add_node("call_mcp_tool", self._call_mcp_tool)
        graph.add_node("validate_observation", self._validate_observation)
        graph.add_node("complete_message", self._complete_message)
        graph.add_edge(START, "receive_request")
        graph.add_edge("receive_request", "prepare_evidence_retrieval")
        graph.add_conditional_edges(
            "prepare_evidence_retrieval",
            self._route_safe_node,
            {"model": "model_decision", "end": END},
        )
        graph.add_conditional_edges(
            "model_decision",
            self._route_decision,
            {
                "tool": "validate_tool_schema",
                "final": "complete_message",
                "model": "model_decision",
                "end": END,
            },
        )
        graph.add_edge("validate_tool_schema", "enforce_tool_policy")
        graph.add_conditional_edges(
            "enforce_tool_policy",
            self._route_policy,
            {"approval": "wait_for_approval", "execute": "call_mcp_tool", "end": END},
        )
        graph.add_conditional_edges(
            "wait_for_approval",
            self._route_after_approval,
            {"execute": "call_mcp_tool", "end": END},
        )
        graph.add_edge("call_mcp_tool", "validate_observation")
        graph.add_conditional_edges(
            "validate_observation",
            self._route_after_observation,
            {"tool": "validate_tool_schema", "model": "model_decision", "end": END},
        )
        graph.add_edge("complete_message", END)
        return graph

    async def run(
        self,
        run_id: str,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]] = (),
    ) -> str | None:
        """Run from creation or resume an interrupted active run at its latest safe node."""

        record = self.store.load(run_id)
        if record.status not in {"created", "running"}:
            raise AgentRuntimeError("Agent Run is not executable")
        initial: _GraphState = {
            "run_id": run_id,
            "messages": [item.to_dict() for item in messages],
            "observations": [dict(item) for item in observations],
            "pending_calls": [],
            "failed": False,
        }
        result = await self._invoke(
            run_id,
            initial if record.status == "created" else None,
            fallback=initial,
        )
        return self._answer(result)

    async def resume_after_approval(
        self,
        run_id: str,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]] = (),
    ) -> str | None:
        """Resume the LangGraph approval interrupt after verifying durable approval state."""

        del messages, observations
        record = self.store.load(run_id)
        approval_id = record.pending_approval_id
        call = record.pending_tool_call
        if record.status != "waiting_approval" or approval_id is None or call is None:
            raise AgentRuntimeError("Agent Run has no pending approval")
        decision = self.store.load_approval(run_id, approval_id)
        if decision.tool_call_sha256 != agent_tool_call_sha256(call):
            raise AgentRuntimeError("approval does not match the pending tool call")
        result = await self._invoke(run_id, Command(resume=decision.decision))
        return self._answer(result)

    async def _invoke(
        self,
        run_id: str,
        value: _GraphState | Command[Any] | None,
        *,
        fallback: _GraphState | None = None,
    ) -> dict[str, Any]:
        checkpoint = self.store.langgraph_checkpoint_path(run_id)
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 64}
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint)) as saver:
            if value is None and await saver.aget_tuple(cast(Any, config)) is None:
                if fallback is None:
                    raise AgentRuntimeError("Agent safe-node checkpoint is missing")
                value = fallback
            compiled = self._builder().compile(checkpointer=saver, name="tinyllm-devops-agent")
            result = await compiled.ainvoke(value, config=cast(Any, config))
        checkpoint.chmod(0o600)
        return cast(dict[str, Any], result)

    @staticmethod
    def _answer(result: dict[str, Any]) -> str | None:
        answer = result.get("answer")
        return answer if isinstance(answer, str) else None

    async def _receive_request(self, state: _GraphState) -> _GraphState:
        run_id = state["run_id"]
        record = self.store.load(run_id)
        if record.status == "created":
            self.store.transition(run_id, status="running")
            self.store.append_event(run_id, "run.started", {"model": record.model})
        elif record.status != "running":
            raise AgentRuntimeError("fresh Agent graph requires a created or running Run")
        return {}

    async def _prepare_evidence_retrieval(self, state: _GraphState) -> _GraphState:
        record = self.store.load(state["run_id"])
        allowed = self._allowed_tools(record.mcp_server_ids)
        if not any(name.endswith(".search_evidence") for name in allowed):
            self._fail(state["run_id"], "AGENT_EVIDENCE_TOOL_MISSING")
            return {"failed": True}
        return {}

    async def _model_decision(self, state: _GraphState) -> _GraphState:
        run_id = state["run_id"]
        record = self.store.load(run_id)
        if state.get("failed"):
            return {}
        if record.steps_completed >= min(record.max_steps, self.config.max_steps):
            self._fail(run_id, "AGENT_STEP_LIMIT")
            return {"failed": True}
        messages = tuple(AgentMessage.model_validate(item) for item in state["messages"])
        decision = await self.model.decide(
            messages=messages,
            observations=tuple(state.get("observations", [])),
            mode=record.mode,
            allowed_tools=self._allowed_tools(record.mcp_server_ids),
        )
        self.store.transition(
            run_id,
            status="running",
            steps_completed=record.steps_completed + 1,
        )
        observations = list(state.get("observations", []))
        pending: list[dict[str, Any]] = []
        duplicate_observations: list[dict[str, object]] = []
        for call in decision.tool_calls:
            signature = self._signature(call)
            succeeded = next(
                (
                    item
                    for item in reversed(observations)
                    if "result" in item and self._signature_from_event(item) == signature
                ),
                None,
            )
            if succeeded is None:
                pending.append(call.to_dict())
                continue
            already_suppressed = any(
                item.get("duplicate_suppressed") is True
                and self._signature_from_event(item) == signature
                for item in observations
            )
            if already_suppressed:
                self._fail(run_id, "AGENT_TOOL_LOOP")
                return {"failed": True, "pending_calls": [], "retry_model": False}
            duplicate_observations.append(
                {
                    "call_id": call.call_id,
                    "server_id": call.server_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result": {
                        "schema_version": "1.0",
                        "status": "already_succeeded",
                        "instruction": "Do not repeat this call; answer from the existing result.",
                    },
                    "duplicate_suppressed": True,
                }
            )
        return {
            "decision": decision.to_dict(),
            "pending_calls": pending,
            "observations": [*observations, *duplicate_observations],
            "retry_model": bool(duplicate_observations and not pending),
        }

    async def _validate_tool_schema(self, state: _GraphState) -> _GraphState:
        pending = list(state.get("pending_calls", []))
        if not pending:
            self._fail(state["run_id"], "AGENT_TOOL_DECISION_EMPTY")
            return {"failed": True}
        call = AgentToolCall.model_validate(pending.pop(0))
        client = self._client(call)
        await client.validate_call(call.tool_name, call.arguments)
        return {"current_call": call.to_dict(), "pending_calls": pending}

    async def _enforce_tool_policy(self, state: _GraphState) -> _GraphState:
        run_id = state["run_id"]
        call = AgentToolCall.model_validate(state["current_call"])
        record = self.store.load(run_id)
        if record.tool_calls_completed >= self.config.max_tool_calls:
            self._fail(run_id, "AGENT_TOOL_LIMIT")
            return {"failed": True}
        signature = self._signature(call)
        signatures = [
            self._signature_from_event(event.data)
            for event in self.store.events_after(run_id)
            if event.event_type == "tool.call.proposed"
        ]
        repeated = 1
        for prior in reversed(signatures):
            if prior != signature:
                break
            repeated += 1
        if repeated > self.config.same_tool_consecutive_limit:
            self._fail(run_id, "AGENT_TOOL_LOOP")
            return {"failed": True}
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
        if not policy.approval_required:
            return {"approval_id": ""}
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
                "tool_call_sha256": agent_tool_call_sha256(call),
            },
        )
        self.store.transition(
            run_id,
            status="waiting_approval",
            pending_approval_id=approval_id,
            pending_tool_call=call,
        )
        return {"approval_id": approval_id}

    async def _wait_for_approval(self, state: _GraphState) -> _GraphState:
        decision = interrupt(
            {
                "approval_id": state["approval_id"],
                "tool_call_sha256": agent_tool_call_sha256(
                    AgentToolCall.model_validate(state["current_call"])
                ),
            }
        )
        if decision not in {"approved", "rejected"}:
            raise AgentRuntimeError("approval resume payload is invalid")
        run_id = state["run_id"]
        record = self.store.load(run_id)
        if record.pending_approval_id != state["approval_id"]:
            raise AgentRuntimeError("approval state changed before graph resume")
        persisted = self.store.load_approval(run_id, state["approval_id"])
        if persisted.decision != decision:
            raise AgentRuntimeError("approval resume differs from durable decision")
        if decision == "rejected":
            self._fail(run_id, "AGENT_APPROVAL_REJECTED")
            return {"failed": True}
        self.store.transition(
            run_id,
            status="running",
            pending_approval_id=None,
            pending_tool_call=None,
        )
        return {}

    async def _call_mcp_tool(self, state: _GraphState) -> _GraphState:
        run_id = state["run_id"]
        call = AgentToolCall.model_validate(state["current_call"])
        arguments = dict(call.arguments)
        approval_id = state.get("approval_id", "")
        if approval_id:
            arguments.update(
                {"approval_id": approval_id, "run_id": run_id, "call_id": call.call_id}
            )
            call = call.model_copy(update={"arguments": arguments})
        observation = await self._execute(run_id, call, self._client(call))
        return {"observations": [*state.get("observations", []), observation]}

    async def _validate_observation(self, state: _GraphState) -> _GraphState:
        observations = state.get("observations", [])
        if not observations:
            self._fail(state["run_id"], "AGENT_OBSERVATION_MISSING")
            return {"failed": True}
        try:
            _reject_private_reasoning(observations[-1])
        except ValueError:
            self._fail(state["run_id"], "AGENT_OBSERVATION_UNSAFE")
            return {"failed": True}
        return {"approval_id": "", "current_call": {}}

    async def _complete_message(self, state: _GraphState) -> _GraphState:
        run_id = state["run_id"]
        decision = AgentModelDecision.model_validate(state["decision"])
        assert decision.message is not None
        observations = state.get("observations", [])
        answer = decision.message
        evidence_ids = tuple(
            item["call_id"]
            for item in observations
            if isinstance(item.get("call_id"), str) and "result" in item
        )
        if evidence_ids and not any(f"[evidence:{call_id}]" in answer for call_id in evidence_ids):
            citations = " ".join(f"[evidence:{call_id}]" for call_id in evidence_ids)
            answer = f"{answer.rstrip()}\n\nEvidence: {citations}"
        self.store.append_event(run_id, "model.delta", {"content": answer})
        self.store.append_event(run_id, "message.completed", {"content": answer})
        self.store.append_event(run_id, "run.completed", {"status": "succeeded"})
        self.store.transition(run_id, status="succeeded")
        return {"answer": answer}

    def _route_decision(self, state: _GraphState) -> str:
        if state.get("failed"):
            return "end"
        if state.get("retry_model"):
            return "model"
        decision = AgentModelDecision.model_validate(state["decision"])
        return "tool" if decision.tool_calls else "final"

    @staticmethod
    def _route_safe_node(state: _GraphState) -> str:
        return "end" if state.get("failed") else "model"

    def _route_policy(self, state: _GraphState) -> str:
        if state.get("failed"):
            return "end"
        return "approval" if state.get("approval_id") else "execute"

    @staticmethod
    def _route_after_approval(state: _GraphState) -> str:
        return "end" if state.get("failed") else "execute"

    @staticmethod
    def _route_after_observation(state: _GraphState) -> str:
        if state.get("failed"):
            return "end"
        return "tool" if state.get("pending_calls") else "model"

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
        self.store.transition(
            run_id,
            status="running",
            tool_calls_completed=current.tool_calls_completed + 1,
        )
        try:
            result = await client.call(call.tool_name, call.arguments)
            observation: dict[str, object] = {
                "call_id": call.call_id,
                "server_id": call.server_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "result": result,
            }
        except (MCPClientError, OSError, TimeoutError) as exc:
            observation = {
                "call_id": call.call_id,
                "server_id": call.server_id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "error": type(exc).__name__,
            }
        self.store.append_event(run_id, "tool.completed", observation)
        return observation

    @staticmethod
    def _signature(call: AgentToolCall) -> str:
        return json.dumps(
            [call.server_id, call.tool_name, call.arguments],
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _signature_from_event(data: dict[str, object]) -> str:
        return json.dumps(
            [data.get("server_id"), data.get("tool_name"), data.get("arguments")],
            sort_keys=True,
            separators=(",", ":"),
        )

    def _fail(self, run_id: str, code: str) -> None:
        try:
            self.store.append_event(run_id, "run.failed", {"code": code})
            self.store.transition(run_id, status="failed", error_code=code)
        except AgentStoreError as exc:
            raise AgentRuntimeError("Agent failure state could not be persisted") from exc
