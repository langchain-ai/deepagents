"""Mutable state derived from the active model."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class RuntimeState:
    """Model metadata that changes during a running session."""

    model_name: str | None = None
    model_provider: str | None = None
    model_context_limit: int | None = None
    model_unsupported_modalities: frozenset[str] = frozenset()


_runtime_state: RuntimeState | None = None
_runtime_state_lock = threading.Lock()


def get_runtime_state() -> RuntimeState:
    """Return the lazily initialized process-wide runtime state."""
    global _runtime_state  # noqa: PLW0603
    if _runtime_state is not None:
        return _runtime_state
    with _runtime_state_lock:
        if _runtime_state is None:
            _runtime_state = RuntimeState()
        return _runtime_state
