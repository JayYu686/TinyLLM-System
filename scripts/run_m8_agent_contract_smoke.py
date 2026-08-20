#!/usr/bin/env python3
"""Exercise M8 approval, restart recovery, and sandbox write through real stdio MCP."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentMessage,
    AgentModelDecision,
    AgentRunRequest,
    AgentRunStore,
    AgentToolCall,
    M8AgentContractEvidence,
    agent_tool_call_sha256,
    load_agent_config,
)
from tinyllm.agent.mcp_client import MCPPolicyClient
from tinyllm.agent.runtime import AgentRuntime
from tinyllm.lineage.git import read_git_identity
from tinyllm.schemas import canonical_config_hash


class _ScriptedModel:
    def __init__(self, decisions: Sequence[AgentModelDecision]) -> None:
        self._decisions = deque(decisions)

    async def decide(
        self,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]],
        mode: str,
        allowed_tools: Sequence[str],
    ) -> AgentModelDecision:
        del messages, observations, mode, allowed_tools
        return self._decisions.popleft()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_new(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("output must be a new absolute path")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _runtime(
    *,
    project_root: Path,
    artifact_root: Path,
    evidence_index: Path,
    store: AgentRunStore,
    decisions: Sequence[AgentModelDecision],
) -> tuple[AgentRuntime, MCPPolicyClient]:
    config = load_agent_config(project_root / "configs/agent/m8_devops.yaml")
    server = config.mcp_servers[0]
    client = MCPPolicyClient(
        server,
        artifact_root=artifact_root,
        project_root=project_root,
        evidence_index=evidence_index,
        retry_delays_ms=config.read_retry_delays_ms,
    )
    return (
        AgentRuntime(
            config=config,
            store=store,
            model=_ScriptedModel(decisions),
            clients={server.server_id: client},
        ),
        client,
    )


async def _execute(args: argparse.Namespace) -> M8AgentContractEvidence:
    project_root: Path = args.project_root.resolve(strict=True)
    artifact_root: Path = args.artifact_root.resolve(strict=True)
    evidence_index: Path = args.evidence_index.resolve(strict=True)
    source_relative_path = "configs/benchmark/m7_inference.yaml"
    source = project_root / source_relative_path
    before = _sha256(source)
    store = AgentRunStore(artifact_root)
    request = AgentRunRequest(
        messages=(AgentMessage(role="user", content="Create an approved sandbox benchmark patch."),)
    )
    record, created = store.create(
        request,
        idempotency_key=f"m8-contract-create-{uuid.uuid4().hex}",
    )
    if not created:
        raise RuntimeError("contract smoke unexpectedly reused an Agent Run")
    call = AgentToolCall(
        call_id="call_m8_contract_patch",
        server_id="tinyllm-devops",
        tool_name="apply_sandbox_config_patch",
        arguments={
            "source_relative_path": source_relative_path,
            "updates": {"warmup_requests": 19},
        },
    )
    first, _client = _runtime(
        project_root=project_root,
        artifact_root=artifact_root,
        evidence_index=evidence_index,
        store=store,
        decisions=(AgentModelDecision(tool_calls=(call,)),),
    )
    await first.run(record.run_id, messages=request.messages)
    waiting = store.load(record.run_id)
    if waiting.pending_approval_id is None or waiting.pending_tool_call is None:
        raise RuntimeError("Agent did not reach durable approval state")
    decision = AgentApprovalDecision(
        approval_id=waiting.pending_approval_id,
        tool_call_sha256=agent_tool_call_sha256(waiting.pending_tool_call),
        decision="approved",
        idempotency_key=f"m8-contract-approval-{uuid.uuid4().hex}",
        decided_at=datetime.now(UTC),
    )
    approval_created = store.decide_approval(record.run_id, decision)

    # Constructing a new Runtime and MCP client simulates a process restart. LangGraph
    # resumes at the persisted approval interrupt instead of replaying the write proposal.
    restarted, client = _runtime(
        project_root=project_root,
        artifact_root=artifact_root,
        evidence_index=evidence_index,
        store=store,
        decisions=(AgentModelDecision(message="沙箱配置已更新。"),),
    )
    answer = await restarted.resume_after_approval(record.run_id, messages=request.messages)
    final = store.load(record.run_id)
    if final.status != "succeeded" or final.tool_calls_completed != 1:
        raise RuntimeError("Agent contract did not complete exactly one approved tool call")
    approval_repeated = not store.decide_approval(record.run_id, decision)

    internal_arguments = {
        **call.arguments,
        "approval_id": waiting.pending_approval_id,
        "run_id": record.run_id,
        "call_id": call.call_id,
    }
    repeated_write = await client.call(call.tool_name, internal_arguments)
    sandbox_relative = str(repeated_write["relative_path"])
    sandbox = artifact_root / sandbox_relative
    after = _sha256(source)
    events = store.events_after(record.run_id)
    event_types = tuple(event.event_type for event in events)
    restart_succeeded = bool(answer) and final.status == "succeeded"
    identity = {
        "run_id": record.run_id,
        "source_sha256": before,
        "sandbox_sha256": _sha256(sandbox),
        "event_types": event_types,
    }
    git_commit, git_dirty = read_git_identity(project_root)
    return M8AgentContractEvidence(
        validation_id=f"m8-agent-contract-{canonical_config_hash(identity)[:8]}",
        executed_at=datetime.now(UTC),
        git_commit=git_commit,
        git_dirty=git_dirty,
        transport="stdio",
        run_id=record.run_id,
        approval_id=waiting.pending_approval_id,
        source_relative_path=source_relative_path,
        source_sha256_before=before,
        source_sha256_after=after,
        sandbox_relative_path=sandbox_relative,
        sandbox_sha256=_sha256(sandbox),
        waiting_status="waiting_approval",
        final_status="succeeded",
        tool_calls_completed=1,
        event_types=event_types,
        source_unchanged=before == after,
        restart_resume_succeeded=restart_succeeded,
        idempotent_approval_succeeded=approval_created and approval_repeated,
        idempotent_write_succeeded=repeated_write["content_sha256"] == _sha256(sandbox),
        passed=not git_dirty,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(_execute(args))
    _atomic_new(args.output, result.to_dict())
    print(result.model_dump_json())
    if not result.passed:
        raise SystemExit(8)


if __name__ == "__main__":
    main()
