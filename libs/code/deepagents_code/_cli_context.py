"""Lightweight runtime context type for app model overrides.

Extracted from `configurable_model` so hot-path modules (`app`,
`textual_adapter`) can import `CLIContext` without pulling in the langchain
middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class CLIContextSchema:
    """Server-side schema for run-scoped CLI context."""

    model: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    effective_model: str | None = None


class CLIContext(TypedDict, total=False):
    """Runtime context passed via `context=` to the LangGraph graph.

    Carries per-invocation overrides that `ConfigurableModelMiddleware`
    reads from `request.runtime.context`.
    """

    model: str | None
    """Model spec to swap at runtime (e.g. `'provider:model'`)."""

    model_params: dict[str, Any]
    """Invocation params (e.g. `temperature`, `max_tokens`) to merge
    into `model_settings`."""

    effective_model: str | None
    """Resolved `provider:model` spec actually in use for this invocation.

    Unlike `model` (a swap *instruction* that is `None` when the base model is
    used), this carries the model in effect — whether from a `/model` override
    or the startup default — so `ResumeStateMiddleware` can record it to
    checkpoint state for restore on resume. `None` when no usable spec is
    resolved yet (e.g. credentials not configured), in which case nothing is
    recorded rather than a malformed spec.
    """
