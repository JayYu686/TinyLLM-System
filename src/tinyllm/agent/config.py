"""Strict YAML loader for the M8 Agent Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tinyllm.agent.schema import AgentConfig


class AgentConfigError(ValueError):
    """Raised when an Agent configuration violates the frozen contract."""


def load_agent_config(path: Path) -> AgentConfig:
    """Load one strict Agent configuration without environment interpolation."""

    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Agent configuration root must be an object")
        return AgentConfig.model_validate(payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise AgentConfigError(f"Agent configuration is invalid: {path}") from exc
