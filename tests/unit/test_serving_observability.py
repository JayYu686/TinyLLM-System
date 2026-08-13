from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tinyllm.serving.observability import StructuredEventLog


def test_structured_log_writes_content_free_private_jsonl(tmp_path: Path) -> None:
    path = tmp_path.resolve() / "private" / "events.jsonl"
    event_log = StructuredEventLog(path)

    asyncio.run(event_log.write("gateway.request", request_id="req_unit", status_code=200))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "gateway.request"
    assert record["request_id"] == "req_unit"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_structured_log_rejects_unsafe_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        StructuredEventLog(Path("events.jsonl"))
    directory = tmp_path.resolve() / "events.jsonl"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular"):
        StructuredEventLog(directory)
