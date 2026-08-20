"""Atomic private Artifact Store for Agent Runs, events, approvals, and cancellation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from tinyllm.agent.schema import (
    AGENT_RUN_PATTERN,
    IDEMPOTENCY_KEY_PATTERN,
    AgentApprovalDecision,
    AgentEvent,
    AgentEventType,
    AgentRunRecord,
    AgentRunRequest,
    AgentToolCall,
    agent_tool_call_sha256,
)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "expired"}


class AgentStoreError(RuntimeError):
    """Raised when a durable Agent state transition is invalid."""


def _canonical_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text}\n".encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_new(path: Path, payload: bytes) -> None:
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


def _atomic_replace(path: Path, payload: bytes) -> None:
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


class AgentRunStore:
    """Persist bounded Agent state while keeping message content in a private request file."""

    def __init__(self, artifact_root: Path) -> None:
        if not artifact_root.is_absolute() or artifact_root.is_symlink():
            raise AgentStoreError("Artifact Store must be an absolute non-symlink path")
        self._root = artifact_root / "agent-runs"
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)

    def _directory(self, run_id: str) -> Path:
        try:
            if re.fullmatch(AGENT_RUN_PATTERN, run_id) is None:
                raise ValueError
            path = self._root / run_id
            if path.exists() and (not path.is_dir() or path.is_symlink()):
                raise ValueError
            return path
        except ValueError as exc:
            raise AgentStoreError("Agent Run ID is invalid") from exc

    def create(
        self,
        request: AgentRunRequest,
        *,
        idempotency_key: str,
        now: datetime | None = None,
        expires_in: timedelta = timedelta(minutes=10),
    ) -> tuple[AgentRunRecord, bool]:
        """Create a Run once, returning an existing identity for a repeated key."""

        if re.fullmatch(IDEMPOTENCY_KEY_PATTERN, idempotency_key) is None:
            raise AgentStoreError("Idempotency-Key is invalid")
        if expires_in <= timedelta(0) or expires_in > timedelta(hours=24):
            raise AgentStoreError("Agent Run expiry is invalid")
        request_payload = _canonical_bytes(request.to_dict())
        request_sha256 = _sha256(request_payload)
        key_sha256 = _sha256(idempotency_key.encode())
        for pointer in self._root.glob("*/idempotency.json"):
            if pointer.is_symlink():
                continue
            try:
                value: Any = json.loads(pointer.read_text(encoding="utf-8"))
                if value.get("key_sha256") == key_sha256:
                    if value.get("request_sha256") != request_sha256:
                        raise AgentStoreError("Idempotency-Key was reused with another request")
                    return self.load(str(value["run_id"])), False
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
                raise AgentStoreError("Agent idempotency index is corrupt") from exc
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise AgentStoreError("Agent creation timestamp must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        prefix = timestamp.strftime("%Y%m%dT%H%M%SZ")
        run_id = f"agent-{prefix}-{request_sha256[:8]}-{secrets.token_hex(2)}"
        directory = self._directory(run_id)
        try:
            directory.mkdir(mode=0o700)
            record = AgentRunRecord(
                run_id=run_id,
                request_sha256=request_sha256,
                model=request.model,
                mode=request.mode,
                mcp_server_ids=request.mcp_server_ids,
                max_steps=request.max_steps,
                status="created",
                created_at=timestamp,
                updated_at=timestamp,
                expires_at=timestamp + expires_in,
                steps_completed=0,
                tool_calls_completed=0,
                last_event_sequence=0,
            )
            _atomic_new(directory / "request.json", request_payload)
            _atomic_new(directory / "run.json", _canonical_bytes(record.to_dict()))
            _atomic_new(
                directory / "idempotency.json",
                _canonical_bytes(
                    {
                        "schema_version": "1.0",
                        "run_id": run_id,
                        "key_sha256": key_sha256,
                        "request_sha256": request_sha256,
                    }
                ),
            )
            (directory / "events.jsonl").touch(mode=0o600)
            return record, True
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("cannot create Agent Run") from exc

    def load(self, run_id: str) -> AgentRunRecord:
        """Load the content-minimized Run projection."""

        path = self._directory(run_id) / "run.json"
        try:
            return AgentRunRecord.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("Agent Run is missing or corrupt") from exc

    def load_request(self, run_id: str) -> AgentRunRequest:
        """Load the private request body for runtime execution or safe-node recovery."""

        path = self._directory(run_id) / "request.json"
        try:
            if path.is_symlink():
                raise ValueError
            return AgentRunRequest.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("Agent Run request is missing or corrupt") from exc

    def list_records(self) -> tuple[AgentRunRecord, ...]:
        """List valid Agent records in deterministic creation order."""

        records: list[AgentRunRecord] = []
        for path in sorted(self._root.glob("agent-*/run.json")):
            if path.is_symlink():
                continue
            try:
                records.append(AgentRunRecord.model_validate_json(path.read_bytes()))
            except (OSError, ValidationError, ValueError) as exc:
                raise AgentStoreError("Agent Run index contains a corrupt record") from exc
        return tuple(records)

    def langgraph_checkpoint_path(self, run_id: str) -> Path:
        """Return a private, pre-created SQLite file for LangGraph safe-node state."""

        directory = self._directory(run_id) / "langgraph"
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError
            directory.chmod(0o700)
            path = directory / "checkpoints.sqlite3"
            if not path.exists():
                with path.open("xb") as handle:
                    os.fchmod(handle.fileno(), 0o600)
            if path.is_symlink() or not path.is_file():
                raise ValueError
            path.chmod(0o600)
            return path
        except (OSError, ValueError) as exc:
            raise AgentStoreError("LangGraph checkpoint path is unsafe") from exc

    def append_event(
        self,
        run_id: str,
        event_type: AgentEventType,
        data: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> AgentEvent:
        """Append one durable event and atomically advance the state cursor."""

        record = self.load(run_id)
        if record.status in TERMINAL_STATUSES:
            raise AgentStoreError("cannot append to a terminal Agent Run")
        existing = self.events_after(run_id)
        durable_sequence = existing[-1].sequence if existing else 0
        if record.last_event_sequence > durable_sequence:
            raise AgentStoreError("Agent Run cursor is ahead of its durable event log")
        timestamp = now or datetime.now(UTC)
        event = AgentEvent(
            run_id=run_id,
            sequence=durable_sequence + 1,
            event_type=event_type,
            created_at=timestamp,
            data=data,
        )
        path = self._directory(run_id) / "events.jsonl"
        try:
            payload = _canonical_bytes(event.to_dict())
            with path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            updated = record.model_copy(
                update={"updated_at": timestamp, "last_event_sequence": event.sequence}
            )
            _atomic_replace(
                self._directory(run_id) / "run.json", _canonical_bytes(updated.to_dict())
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("cannot persist Agent event") from exc
        return event

    def events_after(self, run_id: str, last_event_id: int = 0) -> tuple[AgentEvent, ...]:
        """Replay all events after a per-Run SSE cursor."""

        if last_event_id < 0:
            raise AgentStoreError("Last-Event-ID must be non-negative")
        path = self._directory(run_id) / "events.jsonl"
        try:
            events = tuple(
                AgentEvent.model_validate_json(line)
                for line in path.read_bytes().splitlines()
                if line
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("Agent event log is missing or corrupt") from exc
        if tuple(event.sequence for event in events) != tuple(range(1, len(events) + 1)):
            raise AgentStoreError("Agent event sequence is not monotonic")
        return tuple(event for event in events if event.sequence > last_event_id)

    def decide_approval(self, run_id: str, decision: AgentApprovalDecision) -> bool:
        """Persist one approval exactly once; repeats must carry the same decision and key."""

        record = self.load(run_id)
        directory = self._directory(run_id) / "approvals"
        path = directory / f"{decision.approval_id}.json"
        payload = _canonical_bytes(decision.to_dict())
        if path.exists():
            if path.read_bytes() != payload:
                raise AgentStoreError("approval was repeated with a different decision")
            return False
        if (
            record.status != "waiting_approval"
            or record.pending_approval_id != decision.approval_id
        ):
            raise AgentStoreError("Agent Run is not waiting for this approval")
        assert record.pending_tool_call is not None
        if agent_tool_call_sha256(record.pending_tool_call) != decision.tool_call_sha256:
            raise AgentStoreError("approval is not bound to the pending tool call")
        directory.mkdir(mode=0o700, exist_ok=True)
        _atomic_new(path, payload)
        return True

    def load_approval(self, run_id: str, approval_id: str) -> AgentApprovalDecision:
        """Load one persisted approval decision without trusting caller-supplied state."""

        if re.fullmatch(r"^approval-[0-9a-f]{12}$", approval_id) is None:
            raise AgentStoreError("approval identity is invalid")
        path = self._directory(run_id) / "approvals" / f"{approval_id}.json"
        try:
            if path.is_symlink():
                raise ValueError
            return AgentApprovalDecision.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("approval decision is missing or corrupt") from exc

    def expire_if_due(
        self,
        run_id: str,
        *,
        now: datetime | None = None,
    ) -> tuple[AgentRunRecord, bool]:
        """Expire one nonterminal Run once its immutable deadline has passed."""

        record = self.load(run_id)
        if record.status in TERMINAL_STATUSES:
            return record, False
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise AgentStoreError("Agent expiry timestamp must be timezone-aware")
        if timestamp.astimezone(UTC) < record.expires_at:
            return record, False
        # The record contract keeps updated_at at or before expires_at. Persist the
        # transition at the immutable deadline even if the expiry sweep runs later.
        self.append_event(
            run_id,
            "run.failed",
            {"code": "AGENT_RUN_EXPIRED", "status": "expired"},
            now=record.expires_at,
        )
        return (
            self.transition(run_id, status="expired", now=record.expires_at),
            True,
        )

    def transition(
        self,
        run_id: str,
        *,
        status: Literal[
            "created",
            "running",
            "waiting_approval",
            "succeeded",
            "failed",
            "cancelled",
            "expired",
        ],
        now: datetime | None = None,
        pending_approval_id: str | None = None,
        pending_tool_call: AgentToolCall | None = None,
        error_code: str | None = None,
        steps_completed: int | None = None,
        tool_calls_completed: int | None = None,
    ) -> AgentRunRecord:
        """Apply one schema-validated state transition."""

        record = self.load(run_id)
        if record.status in TERMINAL_STATUSES:
            raise AgentStoreError("terminal Agent Run cannot transition")
        timestamp = now or datetime.now(UTC)
        terminal = status in TERMINAL_STATUSES
        updates: dict[str, object] = {
            "status": status,
            "updated_at": timestamp,
            "completed_at": timestamp if terminal else None,
            "pending_approval_id": pending_approval_id,
            "pending_tool_call": pending_tool_call,
            "error_code": error_code,
        }
        if steps_completed is not None:
            updates["steps_completed"] = steps_completed
        if tool_calls_completed is not None:
            updates["tool_calls_completed"] = tool_calls_completed
        try:
            updated = AgentRunRecord.model_validate(
                record.model_copy(update=updates), from_attributes=True
            )
            _atomic_replace(
                self._directory(run_id) / "run.json", _canonical_bytes(updated.to_dict())
            )
            return updated
        except (OSError, ValidationError, ValueError) as exc:
            raise AgentStoreError("Agent state transition is invalid") from exc

    def cancel(self, run_id: str, *, now: datetime | None = None) -> tuple[AgentRunRecord, bool]:
        """Explicitly cancel a nonterminal Run; repeated cancellation is idempotent."""

        record = self.load(run_id)
        if record.status == "cancelled":
            return record, False
        if record.status in TERMINAL_STATUSES:
            raise AgentStoreError("completed Agent Run cannot be cancelled")
        return self.transition(run_id, status="cancelled", now=now), True
