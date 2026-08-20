from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tinyllm.cli import _agent_api_request, main


def test_agent_cli_help_exposes_frozen_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agent", "--help"]) == 0
    output = capsys.readouterr().out
    for command in ("run", "approve", "cancel", "index"):
        assert command in output
    assert main(["agent", "index", "--help"]) == 0
    assert "rebuild" in capsys.readouterr().out


def test_agent_cli_run_approve_and_cancel_emit_stable_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def request(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"run_id": "agent-20260813T120000Z-1234abcd-beef", "status": "running"}

    monkeypatch.setattr("tinyllm.cli._agent_api_request", request)
    assert main(["agent", "run", "diagnose", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert calls[-1]["path"] == "/v1/agent/runs"
    body = calls[-1]["body"]
    assert isinstance(body, dict)
    assert body["mode"] == "nonthinking"

    assert (
        main(
            [
                "agent",
                "approve",
                "agent-20260813T120000Z-1234abcd-beef",
                "approval-123456abcdef",
                "--decision",
                "rejected",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert calls[-1]["body"] == {"schema_version": "1.0", "decision": "rejected"}

    assert (
        main(
            [
                "agent",
                "cancel",
                "agent-20260813T120000Z-1234abcd-beef",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert str(calls[-1]["path"]).endswith("/cancel")


@pytest.mark.parametrize(
    "arguments",
    [
        ["agent", "run", " ", "--json"],
        ["agent", "run", "x", "--mode", "invalid", "--json"],
        [
            "agent",
            "approve",
            "agent-20260813T120000Z-1234abcd-beef",
            "approval-123456abcdef",
            "--decision",
            "maybe",
            "--json",
        ],
    ],
)
def test_agent_cli_rejects_invalid_inputs(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(arguments) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "AGENT_ERROR"


def test_agent_cli_maps_transport_failures_to_exit_eight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("offline")

    monkeypatch.setattr("tinyllm.cli._agent_api_request", reject)
    assert main(["agent", "run", "diagnose", "--json"]) == 8
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "AGENT_ERROR"


class _Response:
    def __init__(self, payload: object, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def raise_for_status(self) -> None:
        if self.fail:
            raise httpx.HTTPStatusError(
                "failure",
                request=httpx.Request("POST", "http://127.0.0.1:8000"),
                response=httpx.Response(500),
            )

    def json(self) -> object:
        return self.payload


def test_agent_api_helper_enforces_loopback_secret_and_response_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="loopback"):
        _agent_api_request(
            method="GET", base_url="https://example.com", token_env="MISSING", path="/runs"
        )
    monkeypatch.delenv("TEST_AGENT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="32-character"):
        _agent_api_request(
            method="GET",
            base_url="http://127.0.0.1:8000",
            token_env="TEST_AGENT_TOKEN",
            path="/runs",
        )
    monkeypatch.setenv("TEST_AGENT_TOKEN", "x" * 32)
    captured: dict[str, Any] = {}

    def request(*args: object, **kwargs: object) -> _Response:
        captured["args"] = args
        captured.update(kwargs)
        return _Response({"status": "ok"})

    monkeypatch.setattr(httpx, "request", request)
    result = _agent_api_request(
        method="POST",
        base_url="http://127.0.0.1:8000/",
        token_env="TEST_AGENT_TOKEN",
        path="/v1/agent/runs",
        body={"value": 1},
        idempotency_key="operation-123456",
    )
    assert result == {"status": "ok"}
    assert captured["trust_env"] is False
    assert captured["headers"]["Idempotency-Key"] == "operation-123456"

    monkeypatch.setattr(httpx, "request", lambda *_args, **_kwargs: _Response([]))
    with pytest.raises(RuntimeError, match="invalid response"):
        _agent_api_request(
            method="GET",
            base_url="http://localhost:8000",
            token_env="TEST_AGENT_TOKEN",
            path="/runs",
        )
    monkeypatch.setattr(httpx, "request", lambda *_args, **_kwargs: _Response({}, fail=True))
    with pytest.raises(RuntimeError, match="request failed"):
        _agent_api_request(
            method="GET",
            base_url="http://localhost:8000",
            token_env="TEST_AGENT_TOKEN",
            path="/runs",
        )


def test_agent_index_rebuild_cli_persists_private_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    artifact = tmp_path / "artifact"
    output = tmp_path / "indexes" / "m8"
    (project / "docs").mkdir(parents=True)
    (artifact / "registry").mkdir(parents=True)
    (project / "README.md").write_text("# Agent evidence\n", encoding="utf-8")

    assert (
        main(
            [
                "agent",
                "index",
                "rebuild",
                "--project-root",
                str(project),
                "--artifact-root",
                str(artifact),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["index_version"].startswith("m8-evidence-")
    assert (output / "manifest.json").is_file()
    assert (
        main(
            [
                "agent",
                "index",
                "rebuild",
                "--project-root",
                str(project),
                "--artifact-root",
                str(artifact),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 8
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "AGENT_ERROR"
