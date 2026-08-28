"""Lightweight runtime context types for the CLI agent graph.

Carries per-run overrides (model swap/params, approval mode) passed via
`context=`. Extracted from `configurable_model` so hot-path modules (`app`,
`textual_adapter`) can import `CLIContext` without pulling in the langchain
middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

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

INHERIT_SUMMARIZATION_MODEL = "__dcode_inherit_summarization__"
"""Per-run `summarization_model` value meaning "use the main agent model".

An absent (or `None`) `summarization_model` leaves the graph's startup summary
model in effect. `/summarization-model clear` needs the stronger statement that
summaries go back to the main agent model, which this sentinel carries across
the remote JSON boundary without being mistaken for a model spec.
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

    summarization_model: str | None = None

    profile_overrides: dict[str, Any] = field(default_factory=dict)

    model_context_limit: int | None = None

    classifier_model: str | None = None

    approval_mode: str = "manual"

    auto_approve: bool = False

    approval_mode_key: str | None = None

    thread_id: str | None = None

    turn_id: str | None = None

    hooks_snapshot_id: str | None = None

    hooks_server_events: list[str] = field(default_factory=list)

    prompt_id: str | None = None

    @classmethod
    def from_payload(cls, payload: object) -> CLIContextSchema | None:
        """Coerce a run's `context=` payload into this schema.

        In-process LangGraph coerces the payload before consumers see it, so a
        schema instance is returned unchanged. Over the API server (RemoteGraph)
        the payload is JSON and arrives as a plain dict, which this rebuilds
        field by field. Single conversion point for every consumer: a second
        hand-rolled copy drifts field by field, and the copy that forgets a
        field drops it silently.

        Args:
            payload: The value read from `runtime.context`.

        Returns:
            The coerced schema, or `None` when the payload is neither a
            `CLIContextSchema` nor a dict.
        """
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return None
        data = cast("dict[str, Any]", payload)

        def _str(key: str) -> str | None:
            value = data.get(key)
            return value if isinstance(value, str) else None

        def _mapping(key: str) -> dict[str, Any]:
            # `dict(...)` on a non-mapping (`"x"`, `7`, ...) raises
            # `TypeError`/`ValueError`, which would abort the request before the
            # model handler runs — even when the malformed field is unrelated to
            # model selection. Fall back to empty instead.
            value = data.get(key)
            return dict(value) if isinstance(value, dict) else {}

        # `bool` is an `int` subclass, so exclude it explicitly.
        raw_limit = data.get("model_context_limit")
        limit = (
            raw_limit
            if isinstance(raw_limit, int) and not isinstance(raw_limit, bool)
            else None
        )
        approval_mode = _str("approval_mode")
        raw_auto_approve = data.get("auto_approve")
        # Only a real list is safe to iterate: `7` raises `TypeError`, and a
        # bare string (`"PreToolUse"`) would explode into per-character events.
        raw_events = data.get("hooks_server_events")
        events = (
            [event for event in raw_events if isinstance(event, str)]
            if isinstance(raw_events, list)
            else []
        )
        return cls(
            model=_str("model"),
            model_params=_mapping("model_params"),
            summarization_model=_str("summarization_model"),
            profile_overrides=_mapping("profile_overrides"),
            model_context_limit=limit,
            classifier_model=_str("classifier_model"),
            approval_mode=approval_mode or "manual",
            # Fail closed for malformed remote payloads. In particular,
            # `bool("false")` is `True` and would enable legacy YOLO mode.
            auto_approve=(
                raw_auto_approve if isinstance(raw_auto_approve, bool) else False
            ),
            approval_mode_key=_str("approval_mode_key"),
            thread_id=_str("thread_id"),
            turn_id=_str("turn_id"),
            hooks_snapshot_id=_str("hooks_snapshot_id"),
            hooks_server_events=events,
            prompt_id=_str("prompt_id"),
        )


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

    summarization_model: str | None
    """Model spec used only for context-compaction summary generation.

    `None` or an absent key keeps the graph's startup summary model.
    `INHERIT_SUMMARIZATION_MODEL` explicitly selects the main agent model. This
    value never changes the main model or its system-prompt identity: its only
    consumers are the summary-generation slots installed by
    `offload_middleware._summarization_for_runtime`, so compaction thresholds
    and token counting still track the main model.
    """

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
