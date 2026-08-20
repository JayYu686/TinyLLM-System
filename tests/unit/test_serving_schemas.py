from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.serving.config import (
    GatewayConfig,
    ServingConfigError,
    load_gateway_config,
)
from tinyllm.serving.schema import ChatCompletionRequest, ChatMessage


def _request(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "production",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    "message,error",
    [
        ({"role": "tool", "content": "result"}, "tool_call_id"),
        (
            {"role": "user", "content": "x", "tool_calls": [{"id": "call"}]},
            "only assistant",
        ),
        ({"role": "assistant", "content": None}, "content or tool_calls"),
    ],
)
def test_message_role_invariants(message: dict[str, object], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        ChatMessage.model_validate(message)


@pytest.mark.parametrize(
    "updates,error",
    [
        (
            {"max_tokens": 1, "max_completion_tokens": 1},
            "mutually exclusive",
        ),
        ({"stream_options": {"include_usage": True}}, "stream=true"),
        ({"tool_choice": "required"}, "requires tools"),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "known", "parameters": {}},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            },
            "supplied tool",
        ),
        ({"stop": []}, "between one and four"),
        ({"stop": ["1", "2", "3", "4", "5"]}, "between one and four"),
    ],
)
def test_chat_request_invariants(updates: dict[str, object], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        ChatCompletionRequest.model_validate(_request(**updates))


def test_chat_sequences_are_frozen() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            stream=True,
            stream_options={"include_usage": True},
            stop=["one"],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "known", "parameters": {}},
                }
            ],
            tool_choice={"type": "function", "function": {"name": "known"}},
        )
    )
    assert request.stop == ("one",)
    assert request.tools is not None
    assert isinstance(request.messages, tuple)


def test_chat_history_bound_covers_bfcl_parallel_tool_loops() -> None:
    messages = [{"role": "user", "content": "step"} for _ in range(131)]

    request = ChatCompletionRequest.model_validate(_request(messages=messages))

    assert len(request.messages) == 131
    with pytest.raises(ValidationError, match="at most 1024"):
        ChatCompletionRequest.model_validate(_request(messages=messages * 8))


def test_gateway_config_secure_defaults_and_loader(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(
        "schema_version: '1.0'\n"
        "config_id: m7-gateway-unit\n"
        "trusted_hosts: [127.0.0.1, localhost]\n",
        encoding="utf-8",
    )
    config = load_gateway_config(path)
    assert config.trusted_hosts == ("127.0.0.1", "localhost")

    with pytest.raises(ValidationError, match="loopback"):
        GatewayConfig(config_id="m7-gateway-unit", backend_base_url="https://example.com")
    with pytest.raises(ValidationError, match="explicit loopback"):
        GatewayConfig(config_id="m7-gateway-unit", trusted_hosts=("example.com",))
    with pytest.raises(ValidationError, match="explicit loopback"):
        GatewayConfig(config_id="m7-gateway-unit", trusted_hosts=())

    with pytest.raises(ServingConfigError, match="invalid"):
        load_gateway_config(tmp_path / "missing.yaml")
