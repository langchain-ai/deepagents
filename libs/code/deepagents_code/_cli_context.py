"""Lightweight runtime context types for the CLI agent graph.

Carries per-run overrides (model swap/params, approval mode) passed via
`context=`. Extracted from `configurable_model` so hot-path modules (`app`,
`textual_adapter`) can import `CLIContext` without pulling in the langchain
middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

INHERIT_CLASSIFIER_MODEL = "__dcode_inherit_classifier__"
"""Per-run `classifier_model` value meaning "review with the main agent model".

An absent (or `None`) `classifier_model` only says the run carries no
preference, so the classifier keeps whatever the server resolved at startup
(`--auto-classifier-model`, `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL`,
`[models].auto_classifier`). `/auto model clear` needs the stronger statement
that reviews go back to the main agent model, which this sentinel carries.

It cannot collide with a real spec: `create_model` resolves `provider:model`
(or a bare model name) and has no provider or model named `__dcode_...`. A
control character such as a leading NUL would also be collision-proof, but this
value has to survive the trip to a remote deployment intact — the context is
serialized to JSON and may be persisted, and Postgres `text`/`jsonb` rejects NUL
outright. A stripped sentinel would silently read as "no preference" and leave a
startup classifier authorizing actions after the UI reported the clear, so the
sentinel stays plain ASCII.
"""


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

    profile_overrides: dict[str, Any] = field(default_factory=dict)

    model_context_limit: int | None = None

    classifier_model: str | None = None

    approval_mode: str = "manual"

    auto_approve: bool = False

    approval_mode_key: str | None = None

    thread_id: str | None = None

    turn_id: str | None = None

    offload_tool_call_id: str | None = None

    hooks_snapshot_id: str | None = None

    hooks_server_events: list[str] = field(default_factory=list)

    prompt_id: str | None = None


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

    profile_overrides: dict[str, Any]
    """Model profile metadata supplied by `--profile-override`."""

    model_context_limit: int | None
    """Effective context-window limit for profile-aware middleware."""

    classifier_model: str | None
    """Model spec the Auto approval classifier should use for this run.

    `None` (or absent) expresses no per-run preference, so the classifier keeps
    whatever the graph was built with — normally the main agent model, but a
    separate model when the session was launched with one.
    `INHERIT_CLASSIFIER_MODEL` overrides that startup value back to the main
    agent model. Set by `/auto model` so the switch takes effect without
    restarting the agent server.
    """

    approval_mode: str
    """`manual`, classifier-backed `auto`, or unrestricted `yolo`."""

    auto_approve: bool
    """Compatibility snapshot for clients predating the typed mode field."""

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

    turn_id: str | None
    """Current user-turn ID for binding trusted interactive responses."""

    offload_tool_call_id: str | None
    """The sole tool-call ID authorized during a server-driven `/offload` run.

    This is set by the client, not graph state, so model-generated calls cannot
    grant themselves permission to execute during the hidden compaction turn.
    """

    hooks_snapshot_id: str | None
    """Canonical Hooks v2 configuration hash for this session.

    Server-owned lifecycle middleware includes this id on interrupt requests so
    the client can reject mismatched resumes.
    """

    hooks_server_events: list[str]
    """Server-owned HookEvent names that have configured handlers.

    Middleware only interrupts for events listed here, avoiding a round-trip
    when the session snapshot has no matching handlers.
    """

    prompt_id: str | None
    """Optional per-turn prompt id projected into hook context."""
