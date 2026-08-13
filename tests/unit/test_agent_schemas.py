from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.agent import (
    AgentApprovalDecision,
    AgentEvent,
    AgentModelDecision,
    AgentRunRecord,
    AgentRunRequest,
    AgentToolCall,
    EvidenceSearchResult,
    MCPServerConfig,
    MCPToolPolicy,
    load_agent_config,
)

NOW = datetime(2026, 8, 13, tzinfo=UTC)
RUN_ID = "agent-20260813T120000Z-1234abcd-beef"


def test_agent_request_freezes_messages_and_server_ids() -> None:
    request = AgentRunRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "diagnose the failed run"}],
            "mcp_server_ids": ["tinyllm-devops"],
        }
    )
    assert isinstance(request.messages, tuple)
    assert request.mcp_server_ids == ("tinyllm-devops",)
    assert request.max_steps == 8
    with pytest.raises(ValidationError, match="unique"):
        AgentRunRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "x"}],
                "mcp_server_ids": ["tinyllm-devops", "tinyllm-devops"],
            }
        )
    with pytest.raises(ValidationError, match="invalid"):
        AgentRunRequest.model_validate(
            {"messages": [{"role": "user", "content": "x"}], "mcp_server_ids": ["HTTP://x"]}
        )


def test_mcp_tool_policy_enforces_write_approval_and_no_retry() -> None:
    write = MCPToolPolicy(
        name="apply_sandbox_config_patch",
        access="sandbox_write",
        approval_required=True,
        max_attempts=1,
    )
    assert write.max_attempts == 1
    with pytest.raises(ValidationError, match="approval"):
        MCPToolPolicy(
            name="apply_sandbox_config_patch",
            access="sandbox_write",
            approval_required=False,
            max_attempts=1,
        )
    with pytest.raises(ValidationError, match="read-only"):
        MCPToolPolicy(
            name="search_evidence",
            access="read",
            approval_required=True,
            max_attempts=1,
        )


def test_mcp_server_registration_rejects_runtime_supplied_endpoints() -> None:
    tool = MCPToolPolicy(
        name="search_evidence", access="read", approval_required=False, max_attempts=3
    )
    stdio = MCPServerConfig.model_validate(
        {
            "server_id": "tinyllm-devops",
            "transport": "stdio",
            "command": Path("/usr/bin/env"),
            "args": ["python3"],
            "tools": [tool],
        }
    )
    assert stdio.args == ("python3",)
    with pytest.raises(ValidationError, match="absolute command"):
        MCPServerConfig.model_validate(
            {
                "server_id": "tinyllm-devops",
                "transport": "stdio",
                "command": Path("python"),
                "tools": [tool],
            }
        )
    with pytest.raises(ValidationError, match="HTTPS"):
        MCPServerConfig.model_validate(
            {
                "server_id": "remote-devops",
                "transport": "streamable_http",
                "url": "http://127.0.0.1:9000/mcp",
                "bearer_token_env": "TINYLLM_MCP_TOKEN",
                "tools": [tool],
            }
        )
    with pytest.raises(ValidationError, match="unique"):
        MCPServerConfig.model_validate(
            {
                "server_id": "tinyllm-devops",
                "transport": "stdio",
                "command": Path("/usr/bin/env"),
                "tools": [tool, tool],
            }
        )


def test_frozen_agent_config_loads() -> None:
    config = load_agent_config(Path("configs/agent/m8_devops.yaml"))
    assert config.max_steps == 8
    assert config.max_tool_calls == 12
    assert config.read_retry_delays_ms == (250, 500)
    assert config.mcp_servers[0].server_id == "tinyllm-devops"
    assert config.mcp_servers[0].tools[-1].approval_required is True


def test_agent_event_rejects_private_reasoning_fields_at_any_depth() -> None:
    event = AgentEvent(
        run_id=RUN_ID,
        sequence=1,
        event_type="run.started",
        created_at=NOW,
        data={"model": "production"},
    )
    assert event.sequence == 1
    with pytest.raises(ValidationError, match="private reasoning"):
        AgentEvent(
            run_id=RUN_ID,
            sequence=2,
            event_type="model.delta",
            created_at=NOW,
            data={"nested": [{"reasoning_content": "secret"}]},
        )


def test_model_decision_freezes_valid_tool_proposals() -> None:
    decision = AgentModelDecision.model_validate(
        {
            "tool_calls": [
                {
                    "call_id": "call_search_1",
                    "server_id": "tinyllm-devops",
                    "tool_name": "search_evidence",
                    "arguments": {"query": "backend failure"},
                }
            ]
        }
    )
    assert isinstance(decision.tool_calls, tuple)
    assert isinstance(decision.tool_calls[0], AgentToolCall)
    with pytest.raises(ValidationError, match="either"):
        AgentModelDecision()


def _run(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "run_id": RUN_ID,
        "request_sha256": "a" * 64,
        "model": "production",
        "mode": "nonthinking",
        "mcp_server_ids": ["tinyllm-devops"],
        "max_steps": 8,
        "status": "running",
        "created_at": NOW,
        "updated_at": NOW,
        "expires_at": NOW + timedelta(minutes=10),
        "steps_completed": 1,
        "tool_calls_completed": 0,
        "last_event_sequence": 1,
    }
    value.update(updates)
    return value


def test_agent_run_state_machine_schema_rejects_inconsistent_states() -> None:
    AgentRunRecord.model_validate(_run())
    waiting = AgentRunRecord.model_validate(
        _run(
            status="waiting_approval",
            pending_approval_id="approval-123456abcdef",
            pending_tool_call={
                "call_id": "call_patch_1",
                "server_id": "tinyllm-devops",
                "tool_name": "apply_sandbox_config_patch",
                "arguments": {},
            },
        )
    )
    assert waiting.pending_approval_id is not None
    with pytest.raises(ValidationError, match="completion time"):
        AgentRunRecord.model_validate(_run(status="succeeded"))
    with pytest.raises(ValidationError, match="approval state"):
        AgentRunRecord.model_validate(_run(status="waiting_approval"))
    with pytest.raises(ValidationError, match="error code"):
        AgentRunRecord.model_validate(
            _run(status="failed", completed_at=NOW + timedelta(seconds=1))
        )


def test_approval_and_evidence_require_safe_identifiers() -> None:
    approval = AgentApprovalDecision(
        approval_id="approval-123456abcdef",
        decision="approved",
        idempotency_key="client-operation-1234",
        decided_at=NOW,
    )
    assert approval.decision == "approved"
    with pytest.raises(ValidationError, match="timezone-aware"):
        AgentApprovalDecision(
            approval_id="approval-123456abcdef",
            decision="rejected",
            idempotency_key="client-operation-1234",
            decided_at=datetime(2026, 8, 13),
        )
    with pytest.raises(ValidationError, match="relative"):
        EvidenceSearchResult(
            document_id="doc-1234567890abcdef",
            source_kind="documentation",
            relative_path="../secret",
            start_line=1,
            end_line=1,
            content_sha256="a" * 64,
            relevance_score=1.0,
            excerpt="secret",
        )
