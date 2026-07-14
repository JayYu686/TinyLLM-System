"""Safe subprocess execution used by doctor collectors."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _clean_output(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).strip()


@dataclass(frozen=True)
class CommandResult:
    """Sanitized result of a subprocess invocation."""

    args: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    available: bool = True
    timed_out: bool = False


class Runner(Protocol):
    """Protocol that allows collectors to be tested with fixture commands."""

    def run(self, args: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run a command without invoking a shell."""


class SubprocessRunner:
    """Run bounded, read-only commands with deterministic locale settings."""

    def run(self, args: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        """Run a command and capture output without leaking environment values."""

        command = tuple(args)
        environment = os.environ.copy()
        environment.update({"LANG": "C", "LC_ALL": "C", "NO_COLOR": "1", "TERM": "dumb"})
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
            )
        except FileNotFoundError:
            return CommandResult(command, None, "", "command not found", available=False)
        except subprocess.TimeoutExpired:
            return CommandResult(command, None, "", "command timed out", timed_out=True)
        except OSError as exc:
            return CommandResult(command, None, "", type(exc).__name__, available=False)
        return CommandResult(
            command,
            completed.returncode,
            _clean_output(completed.stdout),
            _clean_output(completed.stderr),
        )
