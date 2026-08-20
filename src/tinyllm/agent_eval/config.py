"""Strict YAML loading for the M9 Agent evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tinyllm.agent_eval.schema import (
    AgentEvalRunConfig,
    AgentGateConfig,
    BFCLCoreProfileConfig,
)


class AgentEvalConfigError(ValueError):
    """Raised when an M9 Agent evaluation configuration is missing or invalid."""


def load_agent_eval_config(path: Path) -> AgentEvalRunConfig:
    """Load one strict configuration without interpolating environment secrets."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AgentEvalConfigError("Agent evaluation config must be a YAML mapping")
        return AgentEvalRunConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise AgentEvalConfigError("Agent evaluation config is missing or invalid") from exc


def load_bfcl_profile_config(path: Path) -> BFCLCoreProfileConfig:
    """Load the frozen 1840-task BFCL profile without accepting scope drift."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AgentEvalConfigError("BFCL profile config must be a YAML mapping")
        return BFCLCoreProfileConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise AgentEvalConfigError("BFCL profile config is missing or invalid") from exc


def load_agent_gate_config(path: Path) -> AgentGateConfig:
    """Load the M9-frozen M10 gate thresholds without accepting partial overrides."""

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AgentEvalConfigError("Agent gate config must be a YAML mapping")
        return AgentGateConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise AgentEvalConfigError("Agent gate config is missing or invalid") from exc
