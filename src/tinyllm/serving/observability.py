"""Private structured event logging without request or model content."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path


class StructuredEventLog:
    """Append content-free JSONL events to a same-user-only Artifact file."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute() or path.is_symlink():
            raise ValueError("structured event log path must be absolute and non-symlink")
        if path.exists() and not path.is_file():
            raise ValueError("structured event log must be a regular file")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        self._path = path
        self._lock = asyncio.Lock()

    async def write(self, event: str, **fields: object) -> None:
        record = {
            "schema_version": "1.0",
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        async with self._lock:
            await asyncio.to_thread(self._append, payload)

    def _append(self, payload: bytes) -> None:
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
