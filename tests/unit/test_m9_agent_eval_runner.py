from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tinyllm.agent import AgentMessage, AgentModelDecision, AgentToolCall
from tinyllm.agent_eval.runner import _agent_config, _evaluate_task
from tinyllm.agent_eval.schema import AgentEvalRunConfig
from tinyllm.agent_eval.suite import build_tasks


class _FakeGatewayAgentModel:
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.input_tokens = 10
        self.output_tokens = 5

    async def close(self) -> None:
        return None

    async def decide(
        self,
        *,
        messages: Sequence[AgentMessage],
        observations: Sequence[dict[str, object]],
        mode: str,
        allowed_tools: Sequence[str],
    ) -> AgentModelDecision:
        del messages, mode, allowed_tools
        if not observations:
            return AgentModelDecision(
                tool_calls=(
                    AgentToolCall(
                        call_id="call_fixture_get_run",
                        server_id="tinyllm-devops",
                        tool_name="get_run",
                        arguments={"run_id": "20260820T011500Z-serving-smoke-b2c3d4e5-0002"},
                    ),
                )
            )
        call_id = observations[-1].get("call_id")
        assert isinstance(call_id, str)
        return AgentModelDecision(
            message=(
                f"Run 20260820T011500Z-serving-smoke-b2c3d4e5-0002 succeeded. [evidence:{call_id}]"
            )
        )


def _config() -> AgentEvalRunConfig:
    return AgentEvalRunConfig(
        config_id="m9-agent-eval-unit",
        gateway_base_url="http://127.0.0.1:8000",
        bearer_token_env="TINYLLM_TEST_TOKEN",
        model="production",
        max_concurrency=1,
        physical_gpu_index=4,
    )


def test_evaluate_task_crosses_real_agent_graph_and_fixture_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tinyllm.agent_eval.runner as runner

    monkeypatch.setattr(runner, "GatewayAgentModel", _FakeGatewayAgentModel)
    task = build_tasks("dev")[0]

    result = asyncio.run(
        _evaluate_task(
            task,
            config=_config(),
            bearer_token="x" * 32,
            output_root=tmp_path,
        )
    )

    assert result.status == "succeeded"
    assert result.task_success is True
    assert result.calls[0].tool_name == "get_run"
    assert len(result.evidence_citations) == 1
    assert result.evidence_citations[0].startswith("call_plan_")
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_eval_config_rejects_remote_gateway_and_unknown_fields() -> None:
    payload: dict[str, Any] = _config().to_dict()
    payload["gateway_base_url"] = "https://example.com"
    with pytest.raises(ValidationError, match="loopback"):
        AgentEvalRunConfig.model_validate(payload)

    payload = _config().to_dict()
    payload["secret"] = "embedded-token"
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentEvalRunConfig.model_validate(payload)


def test_scoring_v3_enables_strict_explicit_tool_intent() -> None:
    task = build_tasks("dev")[0]

    strict = _agent_config(
        task,
        _config().model_copy(update={"scoring_protocol": "m10-agent-scoring-v3"}),
    )
    legacy = _agent_config(task, _config())

    assert strict.require_explicit_tool_intent is True
    assert legacy.require_explicit_tool_intent is False
