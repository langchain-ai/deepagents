"""Starter provider scaffold (Wave 4 P1, category: integration).

Copy this file into a new partner directory, rename `StarterProvider`,
and fill in the TODO blocks. Smoke tests live in `test_provider.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderHealth:
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason}


class StarterProvider:
    """Minimal provider scaffold.

    Subclasses (or copies) should override `name`, `capabilities`, `invoke`,
    and `health`. The base implementation is intentionally a no-op so the
    smoke tests pass without external dependencies.
    """

    name = "starter"

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supports_streaming": False,
            "supports_tools": False,
            "supports_embeddings": False,
            "max_context_tokens": 0,
        }

    def invoke(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        # TODO: replace with a real provider call
        return {
            "provider": self.name,
            "prompt": prompt,
            "completion": "",
            "kwargs": kwargs,
        }

    def health(self) -> dict[str, Any]:
        return ProviderHealth(ok=True).to_dict()
