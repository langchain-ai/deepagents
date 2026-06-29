"""Lightweight runtime context types for the CLI agent graph.

Carries per-run overrides (model swap/params, approval mode) passed via
`context=`. Extracted from `configurable_model` so hot-path modules (`app`,
`textual_adapter`) can import `CLIContext` without pulling in the langchain
middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class CLIContextSchema:
    """Declared `context_schema` for the agent graph.

    Registered via `context_schema=` when the graph is built, so LangGraph
    coerces each run's `context=` payload into this dataclass — in-process,
    `runtime.context` is a `CLIContextSchema` instance.

    It exists alongside `CLIContext` (below) because the payload is shaped
    differently on each side of the API boundary: in-process it is coerced to
    this dataclass, but over the LangGraph API server (RemoteGraph) it is
    serialized to JSON and arrives as a plain dict. Consumers
    (`configurable_model._get_context`, `_should_interrupt_tool_call`)
    therefore accept both shapes. `CLIContext` is the client-facing builder for
    constructing that payload.

    Fields mirror `CLIContext`; see its per-field docstrings for semantics.
    """

    model: str | None = None

    model_params: dict[str, Any] = field(default_factory=dict)

    effective_model: str | None = None

    auto_approve: bool = False

    approval_mode_key: str | None = None

    thread_id: str | None = None


class CLIContext(TypedDict, total=False):
    """Client-facing builder for the per-run graph context payload.

    Callers populate this and pass it via `context=` to `astream`/`ainvoke`.
    `ConfigurableModelMiddleware` and the `interrupt_on` `when` predicate read
    it from `request.runtime.context`. In-process LangGraph coerces it into
    `CLIContextSchema` (the registered `context_schema`); over the API it stays
    a plain dict — which is why consumers handle both shapes.
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

    auto_approve: bool
    """Whether gated tool calls should skip the human-approval interrupt.

    Sourced from the client session (not graph state) so the model cannot
    self-approve by writing state. The `interrupt_on` `when` predicate reads
    this to suppress interrupts at the source when "approve always" is on,
    avoiding the interrupt-then-auto-resolve round-trip.
    """

    approval_mode_key: str | None
    """Store key for the live approval-mode control record.

    The TUI updates this record when the user toggles approval mode mid-run.
    The server-side interrupt predicate reads it from the LangGraph Store on
    each gated tool call so auto-to-manual changes can take effect before the
    current stream returns.
    """

    thread_id: str | None
    """LangGraph thread ID for the active conversation.

    Mirrors `config.configurable.thread_id` into runtime context for model-call
    middleware that needs per-request session identity, including Fireworks
    session-affinity headers.
    """
