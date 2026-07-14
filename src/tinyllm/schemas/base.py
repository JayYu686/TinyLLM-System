"""Base types for strict, immutable, versioned schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictSchema(BaseModel):
    """Reject unknown fields and mutation at every persisted schema boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with stable field names."""

        return self.model_dump(mode="json")
