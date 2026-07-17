"""Textual UI adapter for agent execution."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from typing import Protocol

    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        EditDecision,
        HITLRequest,
        RejectDecision,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command, Interrupt
    from pydantic import TypeAdapter

    from deepagents_code._ask_user_types import AskUserWidgetResult, Question
    from deepagents_code.resume_state import RubricResult

    # Type alias matching HITLResponse["decisions"] element type
    HITLDecision = ApproveDecision | EditDecision | RejectDecision

    class _TokensUpdateCallback(Protocol):
        """Callback signature for `_on_tokens_update`."""

        def __call__(self, count: int, *, approximate: bool = False) -> None: ...

    class _TokensShowCallback(Protocol):
        """Callback signature for `_on_tokens_show`."""

        def __call__(self, *, approximate: bool = False) -> None: ...


from deepagents_code._ask_user_types import AskUserRequest
from deepagents_code._cli_context import CLIContext
from deepagents_code._constants import SYSTEM_MESSAGE_PREFIX
from deepagents_code._session_stats import (
    ModelStats as ModelStats,
    ModelStatsKey as ModelStatsKey,
    SessionStats as SessionStats,
    SpinnerStatus as SpinnerStatus,
    format_token_count as format_token_count,
    print_usage_table as print_usage_table,
)
from deepagents_code._tool_stream import (
    UNRENDERABLE_TOOL_OUTPUT,
    ToolCallBuffer,
    ToolCallBufferKey,
    ToolStatus,
    build_tool_error_payload,
    build_tool_result_payload,
    build_tool_use_payload,
    count_unemitted_tool_calls,
    normalize_tool_status,
    tool_call_buffer_key,
)
from deepagents_code.config import build_stream_config, get_glyphs
from deepagents_code.file_ops import FileOpTracker
from deepagents_code.hooks import (
    dispatch_hook,
    dispatch_hook_fire_and_forget,
)
from deepagents_code.input import MediaTracker, parse_file_mentions
from deepagents_code.media_utils import create_multimodal_content
from deepagents_code.tool_display import format_tool_message_content
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    RubricResultMessage,
    SummarizationMessage,
    ToolCallMessage,
)

logger = logging.getLogger(__name__)

_hitl_adapter_cache: TypeAdapter | None = None
"""Lazy singleton for the HITL request validator."""

_ASK_USER_UNSUPPORTED_ERROR = "ask_user not supported by this UI"


def _dispatch_tool_use_hook(
    tool_name: str, tool_id: str, tool_args: dict[str, Any]
) -> None:
    """Dispatch a `tool.use` hook with the payload documented in `hooks`."""
    dispatch_hook_fire_and_forget(
        "tool.use", build_tool_use_payload(tool_name, tool_id, tool_args)
    )


def _dispatch_tool_error_hook(tool_name: str) -> None:
    """Dispatch a `tool.error` hook with the payload documented in `hooks`."""
    dispatch_hook_fire_and_forget("tool.error", build_tool_error_payload(tool_name))


def _dispatch_tool_result_hook(
    tool_name: str,
    tool_id: str | None,
    tool_args: dict[str, Any],
    tool_status: ToolStatus,
    tool_output: str,
) -> None:
    """Dispatch a `tool.result` hook with the payload documented in `hooks`.

    `tool_output` is truncated to `HOOK_TOOL_OUTPUT_LIMIT` inside the shared
    payload builder.
    """
    dispatch_hook_fire_and_forget(
        "tool.result",
        build_tool_result_payload(
            tool_name, tool_id, tool_args, tool_status, tool_output
        ),
    )


def _dispatch_terminal_tool_result_hooks(
    tool_messages: dict[str, ToolCallMessage],
    tool_output: str,
) -> list[str]:
    """Emit terminal `tool.error`/`tool.result` for still-pending tool widgets.

    Every widget in `tool_messages` already had its `tool.use` dispatched (that
    is when the widget is mounted), so any tool that reaches a terminal outcome
    *without* a streamed `ToolMessage` — a HITL rejection, a cancelled turn, or
    an aborted stream — would otherwise leave its `tool.use` unterminated. This
    closes each one with a `tool_status="error"` result carrying the widget's
    real `tool_name`/`args`, so the "every `tool.use` is closed by a matching
    terminal event" guarantee holds on those paths too.

    TUI-only: the headless surface reaches the equivalent state through
    `_run_agent_loop`'s orphan drain rather than widgets.

    Args:
        tool_messages: Map of tool-call id to its widget for the pending tools.
        tool_output: Terminal output string recorded on each `tool.result`.

    Returns:
        The tool-call ids that received terminal hooks. Callers track these
            (via `completed_tool_result_ids`) so a later synthetic `ToolMessage`
            — when the turn still resumes, e.g. alongside an answered `ask_user`
            — does not double-dispatch.
    """
    dispatched: list[str] = []
    for tool_id, tool_msg in list(tool_messages.items()):
        _dispatch_tool_error_hook(tool_msg.tool_name)
        _dispatch_tool_result_hook(
            tool_msg.tool_name,
            tool_id,
            tool_msg.args,
            "error",
            tool_output,
        )
        dispatched.append(tool_id)
    return dispatched


def _get_hitl_request_adapter(hitl_request_type: type) -> TypeAdapter:
    """Return a cached `TypeAdapter(HITLRequest)`.

    Avoids re-compiling the pydantic schema on every `execute_task_textual` call.

    Args:
        hitl_request_type: The `HITLRequest` class (passed in because
            it is imported locally by the caller).

    Returns:
        Shared `TypeAdapter` instance.
    """
    global _hitl_adapter_cache  # noqa: PLW0603
    if _hitl_adapter_cache is None:
        from pydantic import TypeAdapter

        _hitl_adapter_cache = TypeAdapter(hitl_request_type)
    return _hitl_adapter_cache


_ask_user_adapter_cache: TypeAdapter | None = None
"""Lazy singleton for the `ask_user` interrupt validator."""


def _get_ask_user_adapter() -> TypeAdapter:
    """Return a cached `TypeAdapter(AskUserRequest)`.

    Returns:
        Shared `TypeAdapter` instance.
    """
    global _ask_user_adapter_cache  # noqa: PLW0603
    if _ask_user_adapter_cache is None:
        from pydantic import TypeAdapter

        _ask_user_adapter_cache = TypeAdapter(AskUserRequest)
    return _ask_user_adapter_cache


def _is_summarization_chunk(metadata: dict | None) -> bool:
    """Check if a message chunk is from summarization middleware.

    The summarization model is invoked with
    `config={"metadata": {"lc_source": "summarization"}}`
    (see `langchain.agents.middleware.summarization`), which
    LangChain's callback system merges into the stream metadata dict.

    Args:
        metadata: The metadata dict from the stream chunk.

    Returns:
        Whether the chunk is from summarization and should be filtered.
    """
    if metadata is None:
        return False
    return metadata.get("lc_source") == "summarization"


def _is_auto_mode_classifier_chunk(metadata: dict | None) -> bool:
    """Check if a message chunk is internal Auto mode classifier output.

    The Auto mode authorization classifier is invoked with
    `config={"metadata": {"lc_source": "auto_mode_classifier"}}`
    (see `AutoModeHITLMiddleware` in `deepagents_code.auto_mode`), which
    LangChain's callback system merges into the stream metadata dict.

    Args:
        metadata: The metadata dict from the stream chunk.

    Returns:
        Whether the chunk should be hidden from the conversation transcript.
    """
    if metadata is None:
        return False
    return metadata.get("lc_source") == "auto_mode_classifier"


class RubricEvaluationEnd(NamedTuple):
    """A validated `rubric_evaluation_end` event forwarded to the caller.

    Bundling the two fields as named attributes (rather than two positional
    strings) makes the grading-run correlation self-documenting and removes the
    risk of transposing the run ID and the verdict at a call site.
    """

    grading_run_id: str
    """Correlation ID minted by `RubricMiddleware` for this grading run."""

    result: RubricResult
    """Terminal/loop verdict carried by the event."""


def _format_rubric_event(data: dict[str, Any]) -> str | None:
    """Format a concise rubric custom-stream event for the transcript.

    Args:
        data: Custom-stream rubric event payload.

    Returns:
        A user-visible summary for rubric events, or `None` for custom-stream
        events that are not rubric events.
    """
    glyphs = get_glyphs()
    event_type = data.get("type")
    if event_type == "rubric_evaluation_start":
        iteration = data.get("iteration", 0)
        show_iteration = data.get("show_iteration") is True
        label = (
            f" (iteration {iteration + 1})"
            if show_iteration and isinstance(iteration, int)
            else ""
        )
        return (
            f"{glyphs.hourglass} Checking acceptance criteria{label}{glyphs.ellipsis}"
        )
    if event_type != "rubric_evaluation_end":
        return None

    result = data.get("result")
    if result is None:
        return None
    if result == "satisfied":
        return f"{glyphs.checkmark} Acceptance criteria satisfied"
    if result == "needs_revision":
        return f"{glyphs.retry} Acceptance criteria not yet satisfied"
    if result == "max_iterations_reached":
        return (
            f"{glyphs.warning} Acceptance criteria not yet satisfied "
            "(iteration limit reached)"
        )
    if result == "failed":
        return f"{glyphs.warning} Rubric is invalid or cannot be evaluated"
    if result == "grader_error":
        return f"{glyphs.warning} Acceptance criteria check failed"
    # A `rubric_evaluation_end` with an unrecognized result is still a terminal
    # grading event; surface it rather than silently dropping it (e.g. if the
    # SDK adds a new verdict the chat would otherwise go quiet mid-turn).
    return f"{glyphs.warning} Acceptance criteria check ended"


def _format_rubric_details(data: dict[str, Any], *, goal_active: bool = False) -> str:
    """Format complete grader details without serializing or truncating payloads.

    Args:
        data: Custom-stream rubric event payload.
        goal_active: Whether the rubric belongs to an unfinished `/goal`.

    Returns:
        Plain text containing the full explanation, unmet criteria, and next step.
    """
    result = data.get("result")
    if result in {None, "satisfied"}:
        return ""

    sections: list[str] = []
    explanation = str(data.get("explanation") or "").strip()
    if explanation:
        sections.append(f"Explanation\n{explanation}")

    criteria = data.get("criteria")
    failing: list[tuple[str, str]] = []
    if isinstance(criteria, list):
        for criterion in criteria:
            if isinstance(criterion, dict) and criterion.get("passed") is False:
                name = str(criterion.get("name") or "Unnamed criterion").strip()
                gap = str(criterion.get("gap") or "").strip()
                failing.append((name, gap))
    if failing:
        lines = ["Unmet criteria"]
        for name, gap in failing:
            lines.append(f"- {name}" + (f"\n  {gap}" if gap else ""))
        sections.append("\n".join(lines))

    if result == "max_iterations_reached" and goal_active:
        next_step = (
            "The goal remains active. Continue with another prompt to resume or "
            "retry, use `/goal <objective>` to amend it, or `/goal clear` to clear it."
        )
    elif result in {"needs_revision", "max_iterations_reached"}:
        next_step = "Address every unmet criterion, then retry the check."
    elif result == "failed":
        next_step = "Review or replace the rubric before grading again."
    elif result == "grader_error":
        next_step = "Retry the check, or choose a different grader model."
    else:
        next_step = "Review the grader details before continuing."
    sections.append(f"Next step\n{next_step}")
    return "\n\n".join(sections)


class TextualUIAdapter:
    """Adapter for rendering agent output to Textual widgets.

    This adapter provides an abstraction layer between the agent execution and the
    Textual UI, allowing streaming output to be rendered as widgets.
    """

    def __init__(
        self,
        mount_message: Callable[..., Awaitable[None]],
        update_status: Callable[[str], None],
        request_approval: Callable[..., Awaitable[Any]],
        on_auto_approve_enabled: Callable[[], Awaitable[bool] | bool | None]
        | None = None,
        on_switch_to_manual: Callable[[], Awaitable[bool] | bool] | None = None,
        set_spinner: Callable[[SpinnerStatus], Awaitable[None]] | None = None,
        set_active_message: Callable[[str | None], None] | None = None,
        on_user_visible_output_started: Callable[[], None] | None = None,
        sync_message_content: Callable[[str, str], None] | None = None,
        sync_tool_message: Callable[[ToolCallMessage], None] | None = None,
        request_ask_user: (
            Callable[
                [list[Question]],
                Awaitable[asyncio.Future[AskUserWidgetResult] | None],
            ]
            | None
        ) = None,
        on_tool_complete: Callable[[], None] | None = None,
        on_subagent_event: Callable[[dict[str, Any]], None] | None = None,
        on_auto_mode_event: (
            Callable[[dict[str, Any]], Awaitable[None] | None] | None
        ) = None,
        on_approval_mode_fallback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the adapter."""
        self._mount_message = mount_message
        """Async callback to mount a message widget to the chat."""

        self._update_status = update_status
        """Callback to update the status bar text."""

        self._request_approval = request_approval
        """Async callback that returns a Future for HITL approval."""

        self._on_auto_approve_enabled = on_auto_approve_enabled
        """Callback invoked before a Manual approval enables Auto."""

        self._on_switch_to_manual = on_switch_to_manual
        """Callback that persists Manual before an Auto fallback resumes."""

        self._set_spinner = set_spinner
        """Callback to show/hide loading spinner."""

        self._set_active_message = set_active_message
        """Callback to set the active streaming message ID (pass `None` to clear)."""

        self._on_user_visible_output_started = on_user_visible_output_started
        """Callback fired after the first model text or tool-call widget renders.

        Hidden model and subagent output does not trigger it. A turn interrupted
        before any user-visible model output produces zero firings.
        """

        self._sync_message_content = sync_message_content
        """Callback to sync final message content back to the store after streaming."""

        self._sync_tool_message = sync_tool_message
        """Callback to sync a tool widget's mutable state back to the store."""

        self._request_ask_user = request_ask_user
        """Async callback for `ask_user` interrupts.

        When awaited, returns a `Future` that resolves to user answers.
        """

        self._on_tool_complete = on_tool_complete
        """Sync callback fired after each `ToolMessage` is processed.

        The app uses this to refresh the footer's git branch as soon as an
        agent-executed tool (e.g. `git checkout`) returns, instead of waiting
        for the full turn to finish.
        """

        self._on_subagent_event = on_subagent_event
        """Sync callback fired for each validated `subagent` custom-stream event."""

        self._on_auto_mode_event = on_auto_mode_event
        """Callback for compact sanitized Auto denial and fallback events."""

        self._on_approval_mode_fallback = on_approval_mode_fallback
        """Callback that synchronizes a fail-closed startup fallback to Manual."""

        # State tracking
        self._current_tool_messages: dict[str, ToolCallMessage] = {}
        """Map of tool call IDs to their message widgets."""

        # Token display callbacks (set by the app after construction)
        self._on_tokens_update: _TokensUpdateCallback | None = None
        """Called with total context tokens after each LLM response."""

        self._on_tokens_pending: Callable[[], None] | None = None
        """Called to show an unknown token count during streaming."""

        self._on_tokens_show: _TokensShowCallback | None = None
        """Called to restore the token display with the cached value."""

    def _sync_tool_widget(self, tool_msg: ToolCallMessage) -> None:
        """Sync a tool widget when the app provided a store callback.

        Total by contract: never raises. Call sites are scattered across the
        turn loop, some outside try/except, so a sync failure must not abort
        the turn — it is logged and swallowed here.
        """
        if self._sync_tool_message is None:
            return
        try:
            self._sync_tool_message(tool_msg)
        except Exception:
            logger.exception("Failed to sync tool widget state to store")

    def finalize_pending_tools_with_error(self, error: str) -> None:
        """Mark all pending/running tool widgets as error and clear tracking.

        This is used as a safety net when an unexpected exception aborts
        streaming before matching `ToolMessage` results are received.

        Args:
            error: Error text to display in each pending tool widget.
        """
        # Each pending widget already had its `tool.use` dispatched at mount, so
        # emit terminal hooks before dropping them — otherwise an aborted stream
        # leaves those `tool.use` events unterminated for audit consumers. Runs
        # before the widget updates so a `set_error` failure can't skip it.
        _dispatch_terminal_tool_result_hooks(self._current_tool_messages, error)
        for tool_msg in list(self._current_tool_messages.values()):
            tool_msg.set_error(error)
            self._sync_tool_widget(tool_msg)
        self._current_tool_messages.clear()

        # Clear active streaming message to avoid stale "active" state in the store.
        if self._set_active_message:
            self._set_active_message(None)


def _build_interrupted_ai_message(
    pending_text_by_namespace: dict[tuple, str],
    current_tool_messages: dict[str, Any],
) -> AIMessage | None:
    """Build an AIMessage capturing interrupted state (text + tool calls).

    Args:
        pending_text_by_namespace: Dict of accumulated text by namespace
        current_tool_messages: Dict of tool_id -> ToolCallMessage widget

    Returns:
        AIMessage with accumulated content and tool calls, or None if empty.
    """
    from langchain_core.messages import AIMessage

    main_ns_key = ()
    accumulated_text = pending_text_by_namespace.get(main_ns_key, "").strip()

    # Reconstruct tool_calls from displayed tool messages
    tool_calls = []
    for tool_id, tool_widget in list(current_tool_messages.items()):
        tool_calls.append(
            {
                "id": tool_id,
                "name": tool_widget._tool_name,
                "args": tool_widget._args,
            }
        )

    if not accumulated_text and not tool_calls:
        return None

    return AIMessage(
        content=accumulated_text,
        tool_calls=tool_calls or [],
    )


def _read_mentioned_file(file_path: Path, max_embed_bytes: int) -> str:
    """Read a mentioned file for inline embedding (sync, for use with to_thread).

    Args:
        file_path: Resolved path to the file.
        max_embed_bytes: Size threshold; larger files get a reference only.

    Returns:
        Markdown snippet with the file content or a size-exceeded reference.
    """
    file_size = file_path.stat().st_size
    if file_size > max_embed_bytes:
        size_kb = file_size // 1024
        return (
            f"\n### {file_path.name}\n"
            f"Path: `{file_path}`\n"
            f"Size: {size_kb}KB (too large to embed, "
            "use read_file tool to view)"
        )
    content = file_path.read_text(encoding="utf-8")
    return f"\n### {file_path.name}\nPath: `{file_path}`\n```text\n{content}\n```"


def _is_renderable_subagent_event(data: Any, *, is_main_agent: bool) -> bool:  # noqa: ANN401  # custom-stream payload is dynamic
    """Whether a `custom` payload is a subagent event this UI can render.

    Guards the live panel against unrelated/malformed custom events and against
    nested (subagent-to-subagent) emissions.

    Args:
        data: The `custom` stream payload.
        is_main_agent: Whether the event came from the main agent's namespace
            (the empty namespace). Nested emissions are ignored.

    Returns:
        True only for a well-formed subagent event from the main agent.
    """
    return is_main_agent and isinstance(data, dict) and data.get("type") == "subagent"


def _require_approval_mode_key(value: str | None) -> str:
    """Return a written Store key for fail-closed startup.

    Raises:
        RuntimeError: If the remote agent has no Store writer.
    """
    if value is None:
        msg = "Approval-mode Store writer is unavailable"
        raise RuntimeError(msg)
    return value


def _is_renderable_auto_mode_event(data: Any, *, is_main_agent: bool) -> bool:  # noqa: ANN401
    """Return whether a custom event is a sanitized top-level Auto event."""
    if (
        not is_main_agent
        or not isinstance(data, dict)
        or data.get("type") != "auto_mode"
    ):
        return False
    event = data.get("event")
    reason = data.get("reason")
    mode = data.get("mode")
    return (
        event in {"denial", "unavailable", "fallback", "warning"}
        and (reason is None or isinstance(reason, str))
        and (mode is None or (event == "fallback" and mode == "manual"))
    )


async def execute_task_textual(
    user_input: str,
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    assistant_id: str | None,
    session_state: Any,  # noqa: ANN401  # Dynamic session state type
    adapter: TextualUIAdapter,
    backend: Any = None,  # noqa: ANN401  # Dynamic backend type
    image_tracker: MediaTracker | None = None,
    context: CLIContext | None = None,
    *,
    sandbox_type: str | None = None,
    message_kwargs: dict[str, Any] | None = None,
    graph_input: dict[str, Any] | None = None,
    rubric: str | None = None,
    goal_active: bool = False,
    blocked_goal_retry_context: str | None = None,
    on_rubric_evaluation_end: Callable[[RubricEvaluationEnd], None] | None = None,
    turn_stats: SessionStats | None = None,
) -> SessionStats:
    """Execute a task with output directed to Textual UI.

    This is the Textual-compatible version of execute_task() that uses
    the TextualUIAdapter for all UI operations.

    Args:
        user_input: The user's input message
        agent: The LangGraph agent to execute
        assistant_id: The agent identifier
        session_state: Session state with a typed approval mode.
        adapter: The TextualUIAdapter for UI operations.
        backend: Optional backend for file operations.
        image_tracker: Optional tracker for images.
        context: Optional `CLIContext` with model override and params. The current
            mode is persisted and copied into runtime context before every stream
            iteration.
        sandbox_type: Sandbox provider name for trace metadata, or `None`
            if no sandbox is active.
        message_kwargs: Extra fields merged into the stream input message
            dict (e.g., `additional_kwargs` for persisting skill metadata
            in the checkpoint).
        graph_input: Prepared non-conversation input for a server-side graph
            operation. When provided, no user message or media is constructed.
        rubric: Acceptance criteria supplied to `RubricMiddleware` via graph
            input state.
        goal_active: Whether the rubric belongs to an unfinished `/goal`.
        blocked_goal_retry_context: One-turn model context for retrying a
            previously blocked goal. This is carried via runtime context so it
            is not parsed for file mentions or checkpointed as human input.
        on_rubric_evaluation_end: Optional callback receiving a validated
            `RubricEvaluationEnd` (grading run ID and verdict) for each
            main-agent `rubric_evaluation_end` event.
        turn_stats: Pre-created `SessionStats` to accumulate into.

            When the caller holds a reference to the same object, stats are
            available even if this coroutine is cancelled before it can return.

            If `None`, a new instance is created internally.

    Returns:
        Stats accumulated over this turn (request count, token counts,
            wall-clock time).

    Raises:
        ValidationError: If HITL request validation fails (re-raised).
        RuntimeError: If Manual cannot be persisted before graph execution.
    """
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        HITLRequest,
        RejectDecision,
    )
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.types import Command
    from pydantic import ValidationError

    from deepagents_code.approval_mode import ApprovalMode, awrite_approval_mode
    from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY, user_prompt_metadata

    hitl_request_adapter = _get_hitl_request_adapter(HITLRequest)
    ask_user_adapter = _get_ask_user_adapter()

    message_content: str | list[dict[str, Any]] | None = None
    if graph_input is None:
        prompt_text, mentioned_files = await asyncio.to_thread(
            parse_file_mentions, user_input
        )
        max_embed_bytes = 256 * 1024

        if mentioned_files:
            context_parts = [prompt_text, "\n\n## Referenced Files\n"]
            for file_path in mentioned_files:
                try:
                    part = await asyncio.to_thread(
                        _read_mentioned_file, file_path, max_embed_bytes
                    )
                    context_parts.append(part)
                except Exception as e:  # noqa: BLE001  # Resilient adapter error handling
                    context_parts.append(
                        f"\n### {file_path.name}\n[Error reading file: {e}]"
                    )
            final_input = "\n".join(context_parts)
        else:
            final_input = prompt_text

        images_to_send = []
        videos_to_send = []
        if image_tracker:
            images_to_send = image_tracker.get_images()
            videos_to_send = image_tracker.get_videos()
        if images_to_send or videos_to_send:
            message_content = create_multimodal_content(
                final_input, images_to_send, videos_to_send
            )
        else:
            message_content = final_input

    thread_id = session_state.thread_id
    # Advance the per-thread turn markers (coding-agent-v1 turn_id/turn_number)
    # once per user prompt, before building the stream config. `session_state`
    # is duck-typed (`Any`): the production `TextualSessionState` always has
    # `advance_turn`, but lightweight callers/test doubles may not, so probe for
    # it and degrade to no turn markers rather than raising.
    advance_turn = getattr(session_state, "advance_turn", None)
    if graph_input is None and callable(advance_turn):
        turn_id, turn_number = advance_turn()
    else:
        turn_id, turn_number = None, None
    # `build_stream_config` does blocking git filesystem reads and may shell out
    # to `git`; offload it so the Textual event loop stays responsive. Advancing
    # the turn markers above is pure/cheap and stays on the loop.
    #
    # `auto_approve` is sampled once here, at turn start, so it labels the trace
    # with the mode the turn began in. A mid-turn Shift+Tab toggle still changes
    # execution behavior (via `context`) but does not relabel this turn's trace.
    config = await asyncio.to_thread(
        build_stream_config,
        thread_id,
        assistant_id,
        sandbox_type=sandbox_type,
        turn_id=turn_id,
        turn_number=turn_number,
        auto_approve=bool(session_state.auto_approve),
    )

    await dispatch_hook("session.start", {"thread_id": thread_id})

    captured_input_tokens = 0
    captured_output_tokens = 0
    if turn_stats is None:
        turn_stats = SessionStats()
    start_time = time.monotonic()

    # Warn if token display callbacks are only partially wired — all three
    # should be set together to avoid inconsistent status-bar behavior.
    token_cbs = (
        adapter._on_tokens_update,
        adapter._on_tokens_pending,
        adapter._on_tokens_show,
    )
    if any(token_cbs) and not all(token_cbs):
        logger.warning(
            "Token callbacks partially wired (update=%s, pending=%s, show=%s); "
            "token display may behave inconsistently",
            adapter._on_tokens_update is not None,
            adapter._on_tokens_pending is not None,
            adapter._on_tokens_show is not None,
        )

    # Show unknown token count during streaming; the accurate count arrives at turn end.
    if adapter._on_tokens_pending:
        adapter._on_tokens_pending()

    file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=backend)
    # Fires at most once per turn, after the first main-agent text or tool-call
    # widget becomes visible, so hidden model activity cannot block prompt restore.
    user_visible_output_started = False

    def _notify_user_visible_output_started() -> None:
        """Fire the output-started callback once, on the first visible output.

        Call only from main-agent, post-filter paths: the "hidden output does
        not count" guarantee lives in the placement of the call sites (all sit
        after the subagent and summarization `continue`s), not in any check
        here — this helper only dedupes.
        """
        nonlocal user_visible_output_started
        if user_visible_output_started:
            return
        user_visible_output_started = True
        if adapter._on_user_visible_output_started:
            try:
                adapter._on_user_visible_output_started()
            except Exception:
                # A prompt-restore gate update must never abort agent
                # streaming — log and keep going (mirrors `_on_tool_complete`).
                logger.warning(
                    "on_user_visible_output_started callback failed",
                    exc_info=True,
                )

    displayed_tool_ids: set[str] = set()
    tool_call_buffers: dict[ToolCallBufferKey, ToolCallBuffer] = {}
    # Tool-call ids that already received terminal hooks before a resumed
    # `ToolMessage` can stream. When the turn still resumes, middleware
    # synthetic messages would otherwise re-dispatch `tool.result`; this set
    # suppresses those duplicates.
    completed_tool_result_ids: set[str] = set()

    # Track pending text and assistant messages PER NAMESPACE to avoid interleaving
    # when multiple subagents stream in parallel
    pending_text_by_namespace: dict[tuple, str] = {}
    assistant_message_by_namespace: dict[tuple, Any] = {}

    if image_tracker and graph_input is None:
        image_tracker.clear()

    if graph_input is None:
        user_msg: dict[str, Any] = {"role": "user", "content": message_content}
        if message_kwargs:
            user_msg.update(message_kwargs)
        additional_kwargs = user_msg.get("additional_kwargs")
        trusted_kwargs = (
            dict(additional_kwargs) if isinstance(additional_kwargs, dict) else {}
        )
        trusted_kwargs[USER_PROMPT_METADATA_KEY] = user_prompt_metadata(
            user_input,
            [str(path) for path in mentioned_files],
            turn_id=turn_id,
        )
        user_msg["additional_kwargs"] = trusted_kwargs
        stream_input: dict | Command = {
            "messages": [user_msg],
            "goal_criteria_request": None,
        }
        if rubric:
            stream_input["rubric"] = rubric
    else:
        stream_input = dict(graph_input)
    recover_interrupted_turn = not (
        graph_input is not None and graph_input.get("goal_criteria_request") is not None
    )

    # Track summarization lifecycle so spinner status and notification stay in sync.
    summarization_in_progress = False

    try:
        while True:
            interrupt_occurred = False
            suppress_resumed_output = False
            pending_interrupts: dict[str, HITLRequest] = {}
            pending_ask_user: dict[str, AskUserRequest] = {}

            if context is None:
                context = CLIContext()
            context["thread_id"] = thread_id
            if blocked_goal_retry_context is not None:
                context["blocked_goal_retry_context"] = blocked_goal_retry_context
            else:
                context.pop("blocked_goal_retry_context", None)
            raw_mode = getattr(session_state, "approval_mode", None)
            if raw_mode is None:
                raw_mode = (
                    ApprovalMode.YOLO
                    if getattr(session_state, "auto_approve", False)
                    else ApprovalMode.MANUAL
                )
            try:
                selected_mode = ApprovalMode(raw_mode)
            except (TypeError, ValueError):
                selected_mode = ApprovalMode.MANUAL
            context["approval_mode"] = selected_mode.value
            context["auto_approve"] = selected_mode is not ApprovalMode.MANUAL
            try:
                live_key = _require_approval_mode_key(
                    await awrite_approval_mode(
                        agent,
                        thread_id,
                        mode=selected_mode,
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to persist selected approval mode; forcing Manual",
                    exc_info=True,
                )
                try:
                    live_key = _require_approval_mode_key(
                        await awrite_approval_mode(
                            agent,
                            thread_id,
                            mode=ApprovalMode.MANUAL,
                        )
                    )
                except Exception as exc:
                    context["approval_mode"] = ApprovalMode.MANUAL.value
                    context["auto_approve"] = False
                    context.pop("approval_mode_key", None)
                    session_state.approval_mode = ApprovalMode.MANUAL
                    session_state.approval_mode_key = None
                    if adapter._on_approval_mode_fallback is not None:
                        adapter._on_approval_mode_fallback(ApprovalMode.MANUAL.value)
                    adapter._update_status("Approval mode fell back to Manual")
                    msg = (
                        "Manual approval mode could not be persisted; graph execution "
                        "is blocked until the Store is available."
                    )
                    raise RuntimeError(msg) from exc
                selected_mode = ApprovalMode.MANUAL
                session_state.approval_mode = ApprovalMode.MANUAL
                context["approval_mode"] = ApprovalMode.MANUAL.value
                context["auto_approve"] = False
                if adapter._on_approval_mode_fallback is not None:
                    adapter._on_approval_mode_fallback(ApprovalMode.MANUAL.value)
                adapter._update_status("Approval mode fell back to Manual")
            context["approval_mode_key"] = live_key
            session_state.approval_mode_key = live_key

            # Show the Thinking spinner before each astream iteration so
            # both the first turn and HITL/ask_user resumes surface feedback
            # while the model processes input. Skip when
            # `_current_tool_messages` is non-empty so running-tool
            # indicators remain the dominant signal.
            if adapter._set_spinner and not adapter._current_tool_messages:
                await adapter._set_spinner("Thinking")

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
                config=config,
                context=context,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:  # noqa: PLR2004  # stream chunk is a 3-tuple (namespace, mode, data)
                    logger.debug("Skipping non-3-tuple chunk: %s", type(chunk).__name__)
                    continue

                namespace, current_stream_mode, data = chunk

                # Convert namespace to hashable tuple for dict keys
                ns_key = tuple(namespace) if namespace else ()

                # Filter out subagent outputs - only show main agent (empty
                # namespace). Subagents run via Task tool and should only
                # report back to the main agent
                is_main_agent = ns_key == ()

                # Handle CUSTOM stream - live subagent fan-out events emitted by
                # the QuickJS task() bridge during a js_eval call. Validate at
                # this boundary before forwarding so unrelated/malformed or
                # nested custom events never reach the panel; forwarding must
                # never raise into the stream loop.
                if current_stream_mode == "custom":
                    rubric_message = data if isinstance(data, dict) else None
                    formatted_rubric_event = (
                        _format_rubric_event(rubric_message) if rubric_message else None
                    )
                    if (
                        formatted_rubric_event is not None
                        and rubric_message is not None
                        and is_main_agent
                    ):
                        details = (
                            _format_rubric_details(
                                rubric_message,
                                goal_active=goal_active,
                            )
                            if rubric_message.get("type") == "rubric_evaluation_end"
                            else ""
                        )
                        message = (
                            RubricResultMessage(formatted_rubric_event, details)
                            if details
                            else AppMessage(formatted_rubric_event)
                        )
                        await adapter._mount_message(message)
                        if (
                            on_rubric_evaluation_end is not None
                            and rubric_message.get("type") == "rubric_evaluation_end"
                        ):
                            grading_run_id = rubric_message.get("grading_run_id")
                            result = rubric_message.get("result")
                            if (
                                isinstance(grading_run_id, str)
                                and grading_run_id.strip()
                                and isinstance(result, str)
                            ):
                                # Structurally validated here; the verdict is
                                # cast to `RubricResult` at this boundary and the
                                # consumer re-checks it against the known set.
                                try:
                                    on_rubric_evaluation_end(
                                        RubricEvaluationEnd(
                                            grading_run_id=grading_run_id.strip(),
                                            result=cast("RubricResult", result),
                                        )
                                    )
                                except Exception:
                                    logger.warning(
                                        "on_rubric_evaluation_end callback failed",
                                        exc_info=True,
                                    )
                        continue
                    if formatted_rubric_event is not None:
                        # Rubric events come from the main agent today; a
                        # non-main namespace would be dropped by the gate above,
                        # so leave a breadcrumb if that ever changes.
                        logger.debug(
                            "Dropping rubric event from non-main namespace %r",
                            ns_key,
                        )
                    if (
                        adapter._on_subagent_event is not None
                        and _is_renderable_subagent_event(
                            data, is_main_agent=is_main_agent
                        )
                    ):
                        try:
                            adapter._on_subagent_event(data)
                        except Exception:
                            logger.exception("subagent panel event handler failed")
                    if (
                        adapter._on_auto_mode_event is not None
                        and _is_renderable_auto_mode_event(
                            data, is_main_agent=is_main_agent
                        )
                    ):
                        try:
                            callback_result = adapter._on_auto_mode_event(data)
                            if callback_result is not None:
                                await callback_result
                        except Exception:
                            logger.exception("Auto mode event handler failed")
                    continue

                # Handle UPDATES stream - for interrupts and todos
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for interrupts
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                iv = interrupt_obj.value
                                if (
                                    isinstance(iv, dict)
                                    and iv.get("type") == "ask_user"
                                ):
                                    try:
                                        validated_ask_user = (
                                            ask_user_adapter.validate_python(iv)
                                        )
                                        pending_ask_user[interrupt_obj.id] = (
                                            validated_ask_user
                                        )
                                        tool_id = validated_ask_user["tool_call_id"]
                                        if tool_id not in displayed_tool_ids:
                                            if adapter._set_spinner:
                                                await adapter._set_spinner(None)
                                            tool_args = {
                                                "questions": validated_ask_user[
                                                    "questions"
                                                ]
                                            }
                                            tool_msg = ToolCallMessage(
                                                "ask_user",
                                                tool_args,
                                            )
                                            try:
                                                await adapter._mount_message(tool_msg)
                                            except Exception:
                                                # Mount failed (e.g. a torn-down
                                                # DOM during shutdown). tool.use
                                                # is dispatched only on mount
                                                # success (below), so a failed
                                                # mount leaves no unterminated
                                                # tool.use to orphan if the turn
                                                # is then cancelled before the
                                                # ask_user resolution loop runs.
                                                # The id is left unlatched so a
                                                # re-observed interrupt can retry
                                                # the mount; the question is still
                                                # asked and closed by the
                                                # resolution loop, which
                                                # dispatches the terminal
                                                # tool.result independently of
                                                # this widget.
                                                logger.exception(
                                                    "Failed to mount ask_user "
                                                    "tool row for %s",
                                                    tool_id,
                                                )
                                            else:
                                                _notify_user_visible_output_started()
                                                # Fire tool.use and latch the id
                                                # together, only once the widget
                                                # is mounted, so the "every
                                                # tool.use is closed" guarantee
                                                # holds with no widget-less orphan
                                                # on the mount-failure path.
                                                # Gating on mount success also
                                                # keeps tool.use fire-once: a
                                                # failed mount never fires it, and
                                                # a successful mount latches the
                                                # id so a re-observed interrupt is
                                                # skipped.
                                                _dispatch_tool_use_hook(
                                                    "ask_user", tool_id, tool_args
                                                )
                                                displayed_tool_ids.add(tool_id)
                                                adapter._current_tool_messages[
                                                    tool_id
                                                ] = tool_msg
                                        interrupt_occurred = True
                                        await dispatch_hook("input.required", {})
                                    except ValidationError:
                                        logger.exception(
                                            "Invalid ask_user interrupt payload"
                                        )
                                        raise
                                else:
                                    try:
                                        validated_request = (
                                            hitl_request_adapter.validate_python(iv)
                                        )
                                        pending_interrupts[interrupt_obj.id] = (
                                            validated_request
                                        )
                                        interrupt_occurred = True
                                        await dispatch_hook("input.required", {})
                                    except ValidationError:  # noqa: TRY203  # Re-raise preserves exception context in handler
                                        raise

                    # Check for todo updates (not yet implemented in Textual UI)
                    chunk_data = next(iter(data.values())) if data else None
                    if (
                        chunk_data
                        and isinstance(chunk_data, dict)
                        and "todos" in chunk_data
                    ):
                        pass  # Future: render todo list widget

                # Handle MESSAGES stream - for content and tool calls
                elif current_stream_mode == "messages":
                    # Skip subagent outputs - only render main agent content in chat
                    if not is_main_agent:
                        logger.debug("Skipping subagent message ns=%s", ns_key)
                        continue

                    if not isinstance(data, tuple) or len(data) != 2:  # noqa: PLR2004  # message stream data is a 2-tuple (message, metadata)
                        logger.debug(
                            "Skipping non-2-tuple message data: type=%s",
                            type(data).__name__,
                        )
                        continue

                    message, metadata = data
                    logger.debug(
                        "Processing message: type=%s id=%s has_content_blocks=%s",
                        type(message).__name__,
                        getattr(message, "id", None),
                        hasattr(message, "content_blocks"),
                    )

                    # Filter out summarization model output, but keep UI feedback.
                    # The summarization model streams AIMessage chunks tagged
                    # with lc_source="summarization" in the callback metadata.
                    # These are hidden from the user; only the spinner and a
                    # notification widget provide feedback.
                    if _is_summarization_chunk(metadata):
                        if not summarization_in_progress:
                            summarization_in_progress = True
                            if adapter._set_spinner:
                                await adapter._set_spinner("Offloading")
                        continue

                    # Extract token usage before filtering hidden model output.
                    # Usage may be attached to any message chunk, including the
                    # internal Auto mode classifier response.
                    if hasattr(message, "usage_metadata"):
                        usage = message.usage_metadata
                        if usage:
                            input_toks = usage.get("input_tokens", 0)
                            output_toks = usage.get("output_tokens", 0)
                            total_toks = usage.get("total_tokens", 0)
                            from deepagents_code.config import settings

                            active_model = settings.model_name or ""
                            active_provider = settings.model_provider or ""
                            if input_toks or output_toks:
                                # Model gives split counts — preferred path
                                turn_stats.record_request(
                                    active_model,
                                    input_toks,
                                    output_toks,
                                    active_provider,
                                )
                                captured_input_tokens = max(
                                    captured_input_tokens, input_toks + output_toks
                                )
                            elif total_toks:
                                # Fallback: model gives only total (no split)
                                turn_stats.record_request(
                                    active_model, total_toks, 0, active_provider
                                )
                                captured_input_tokens = max(
                                    captured_input_tokens, total_toks
                                )

                    # The Auto mode authorization classifier is a nested model
                    # call. Its structured JSON is internal policy machinery,
                    # not assistant output for the conversation transcript.
                    if _is_auto_mode_classifier_chunk(metadata):
                        continue

                    # Regular (non-summarization) chunks resumed — summarization
                    # has finished. Mount the notification and reset the spinner.
                    if summarization_in_progress:
                        summarization_in_progress = False
                        try:
                            await adapter._mount_message(SummarizationMessage())
                        except Exception:
                            logger.debug(
                                "Failed to mount summarization notification",
                                exc_info=True,
                            )
                        if adapter._set_spinner and not adapter._current_tool_messages:
                            await adapter._set_spinner("Thinking")

                    if isinstance(message, HumanMessage):
                        content = message.text
                        # Flush pending text for this namespace
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if content and pending_text:
                            await _flush_assistant_text_ns(
                                adapter,
                                pending_text,
                                ns_key,
                                assistant_message_by_namespace,
                            )
                            pending_text_by_namespace[ns_key] = ""
                            # Drop the cached assistant bubble too, not just the
                            # pending text: a mid-turn HumanMessage (e.g. the
                            # rubric revision loop re-prompting the agent) means
                            # the next assistant text is a fresh response and
                            # must start a new bubble rather than appending to
                            # the pre-revision one.
                            assistant_message_by_namespace.pop(ns_key, None)
                        continue

                    if isinstance(message, ToolMessage):
                        tool_name = getattr(message, "name", "")
                        # Normalize to the two-value hook domain, fail-closed: an
                        # unexpected provider status is logged and treated as an
                        # error (see `normalize_tool_status`) rather than silently
                        # reported as success.
                        tool_status: ToolStatus = normalize_tool_status(
                            getattr(message, "status", "success"), tool_name
                        )
                        # Guard formatting *and* the str() coercion so a
                        # pathological __str__ on the content can't re-raise and
                        # skip the tool.result dispatch below. On failure use a
                        # sentinel rather than re-touching the offending content,
                        # so the terminal dispatch is genuinely unconditional.
                        try:
                            tool_content = format_tool_message_content(message.content)
                            output_str = str(tool_content) if tool_content else ""
                        except Exception:
                            logger.exception("Failed to format tool output")
                            output_str = UNRENDERABLE_TOOL_OUTPUT
                        record = file_op_tracker.complete_with_message(message)

                        # Update tool call status with output
                        tool_id = getattr(message, "tool_call_id", None)
                        if tool_id and tool_id in adapter._current_tool_messages:
                            # Pop before the widget calls so the dict drains even
                            # if set_success/set_error raises.
                            tool_msg = adapter._current_tool_messages.pop(tool_id)
                            # Dispatch the terminal hooks *before* touching the
                            # widget: a render failure must never drop this tool's
                            # tool.result/tool.error (which would leave its
                            # tool.use unterminated). The headless path likewise
                            # dispatches without depending on any widget.
                            if tool_status == "error":
                                _dispatch_tool_error_hook(tool_msg.tool_name)
                            _dispatch_tool_result_hook(
                                tool_msg.tool_name,
                                tool_id,
                                tool_msg.args,
                                tool_status,
                                output_str,
                            )
                            # Update the widget last, guarded: a set_success/
                            # set_error failure must not abort the turn and drop
                            # the remaining tools' hooks.
                            try:
                                if tool_status == "success":
                                    tool_msg.set_success(output_str)
                                else:
                                    tool_msg.set_error(output_str or "Error")
                                adapter._sync_tool_widget(tool_msg)
                            except Exception:
                                logger.exception(
                                    "Failed to update tool row for %s", tool_id
                                )
                        elif tool_id and tool_id in completed_tool_result_ids:
                            # This is a middleware synthetic ToolMessage for a
                            # tool whose terminal hooks already fired while the
                            # turn was resolving interrupts. Its widget was
                            # cleared, so it lands here — consume the id and skip
                            # re-dispatch to avoid a duplicate tool.result (with
                            # mismatched `{}` args).
                            completed_tool_result_ids.discard(tool_id)
                        else:
                            # The tool call was never mounted — either it has no
                            # tool_call_id, or its streamed args never parsed so
                            # no tool.use fired and no widget exists. Still emit
                            # tool.result (with {} args, since without a widget
                            # we lack the parsed args) so audit hooks observe
                            # every executed tool, matching the headless path.
                            # tool_id may be None here, mirroring headless.
                            # Reciprocal: headless always dispatches tool.result
                            # from `_process_message_chunk` since it has no
                            # widget concept; see `non_interactive.py`. The
                            # parity contract is documented in `_tool_stream`.
                            if tool_id:
                                # Warning, not info/debug: a real-id result with
                                # no mounted widget (its args never parsed, so no
                                # tool.use fired) means a hook consumer sees a
                                # `tool.result` with empty args for a tool that
                                # actually executed — degraded audit fidelity worth
                                # surfacing at default log levels, matching the
                                # headless path.
                                logger.warning(
                                    "ToolMessage tool_call_id=%s not in "
                                    "_current_tool_messages; no correlated "
                                    "tool.use, sending empty tool_args",
                                    tool_id,
                                )
                            if tool_status == "error":
                                _dispatch_tool_error_hook(tool_name)
                            _dispatch_tool_result_hook(
                                tool_name, tool_id, {}, tool_status, output_str
                            )

                        # Show file operation results - always show diffs in chat
                        if record:
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter,
                                    pending_text,
                                    ns_key,
                                    assistant_message_by_namespace,
                                )
                                pending_text_by_namespace[ns_key] = ""
                            if record.diff:
                                await adapter._mount_message(
                                    DiffMessage(
                                        record.diff,
                                        record.display_path,
                                        tool_name=record.tool_name,
                                    )
                                )

                        # Reshow spinner only when all in-flight tools have
                        # completed (avoids premature "Thinking..." when
                        # parallel tool calls are active). Must happen after
                        # the diff is mounted so the spinner stays at the
                        # bottom of the messages container.
                        if adapter._set_spinner and not adapter._current_tool_messages:
                            await adapter._set_spinner("Thinking")

                        if adapter._on_tool_complete is not None:
                            try:
                                adapter._on_tool_complete()
                            except Exception:
                                # A footer refresh failure must never abort
                                # agent streaming — log and keep going.
                                logger.warning(
                                    "on_tool_complete callback failed",
                                    exc_info=True,
                                )
                        continue

                    # Check if this is an AIMessageChunk with content
                    if not hasattr(message, "content_blocks"):
                        logger.debug(
                            "Message has no content_blocks: type=%s",
                            type(message).__name__,
                        )
                        continue

                    # Process content blocks
                    blocks = message.content_blocks
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "content_blocks count=%d blocks=%s",
                            len(blocks),
                            repr(blocks)[:500],
                        )
                    for block in blocks:
                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                # Track accumulated text for reference
                                pending_text = pending_text_by_namespace.get(ns_key, "")
                                pending_text += text
                                pending_text_by_namespace[ns_key] = pending_text

                                # Get or create assistant message for this namespace
                                current_msg = assistant_message_by_namespace.get(ns_key)
                                if current_msg is None:
                                    msg_id = f"asst-{uuid.uuid4().hex}"
                                    # Mark active BEFORE mounting so pruning
                                    # (triggered by mount) won't remove it
                                    # (_mount_message can trigger
                                    # _prune_old_messages if the window exceeds
                                    # WINDOW_SIZE.)
                                    if adapter._set_active_message:
                                        adapter._set_active_message(msg_id)
                                    current_msg = AssistantMessage(id=msg_id)
                                    await adapter._mount_message(current_msg)
                                    assistant_message_by_namespace[ns_key] = current_msg
                                    # Keep the Thinking spinner visible after
                                    # the streaming message so the user still
                                    # sees activity if the model pauses between
                                    # finishing text and emitting its next
                                    # action (e.g. a tool call). The mount
                                    # above placed the new message at the end
                                    # of the container; this re-anchors the
                                    # spinner after it.
                                    if (
                                        adapter._set_spinner
                                        and not adapter._current_tool_messages
                                    ):
                                        await adapter._set_spinner("Thinking")

                                # Append just the new text chunk for smoother
                                # streaming (uses MarkdownStream internally for
                                # better performance)
                                await current_msg.append_content(text)
                                _notify_user_visible_output_started()

                        elif block_type in {"tool_call_chunk", "tool_call"}:
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            buffer_key = tool_call_buffer_key(
                                chunk_index, chunk_id, len(tool_call_buffers)
                            )
                            buffer = tool_call_buffers.setdefault(
                                buffer_key, ToolCallBuffer()
                            )
                            buffer.ingest(
                                name=chunk_name, tool_id=chunk_id, args=chunk_args
                            )

                            buffer_name = buffer.name
                            buffer_id = buffer.tool_id
                            if buffer_name is None:
                                continue

                            # `parse_args` reassembles streamed JSON string
                            # fragments, deferring the parse until the value
                            # looks complete — which avoids re-parsing the whole
                            # prefix on every fragment (costly on the UI event
                            # loop for large `edit_file` blobs) — and returns
                            # None while still incomplete. Each `continue` leaves
                            # the buffer in `tool_call_buffers` so the next
                            # fragment keeps accumulating; it is popped only after
                            # a successful parse + mount below.
                            parsed_args = buffer.parse_args()
                            if parsed_args is None:
                                continue

                            # Flush pending text before tool call
                            pending_text = pending_text_by_namespace.get(ns_key, "")
                            if pending_text:
                                await _flush_assistant_text_ns(
                                    adapter,
                                    pending_text,
                                    ns_key,
                                    assistant_message_by_namespace,
                                )
                                pending_text_by_namespace[ns_key] = ""
                                assistant_message_by_namespace.pop(ns_key, None)

                            logger.debug(
                                "Tool call buffer: name=%s id=%s args=%s",
                                buffer_name,
                                buffer_id,
                                repr(parsed_args)[:200],
                            )
                            if (
                                buffer_id is not None
                                and buffer_id not in displayed_tool_ids
                            ):
                                displayed_tool_ids.add(buffer_id)
                                file_op_tracker.start_operation(
                                    buffer_name, parsed_args, buffer_id
                                )

                                # Keep the global "Thinking" spinner visible
                                # across tool calls rather than hiding it per
                                # tool: it's a stable turn-level indicator, and
                                # the tool's own progress now shows in its
                                # collapsed group row. Re-assert it so it stays
                                # pinned at the bottom as the new row mounts
                                # above it.
                                if adapter._set_spinner:
                                    await adapter._set_spinner("Thinking")

                                # Mount tool call message
                                logger.debug(
                                    "Mounting ToolCallMessage: %s(%s)",
                                    buffer_name,
                                    repr(parsed_args)[:200],
                                )
                                # Dispatch tool.use once the streamed call has a
                                # resolved id and parsed args. The headless
                                # surface dispatches from the stream loop
                                # instead; see the "Gate tool.use" comment in
                                # `non_interactive._process_ai_message`. Both
                                # gate on a resolved tool-call id and fire at
                                # most once per id — the parity contract is
                                # documented in `_tool_stream`.
                                _dispatch_tool_use_hook(
                                    buffer_name, buffer_id, parsed_args
                                )
                                tool_msg = ToolCallMessage(buffer_name, parsed_args)
                                try:
                                    await adapter._mount_message(tool_msg)
                                except Exception:
                                    # tool.use already fired. If the mount raises
                                    # (e.g. mounting into a torn-down DOM during
                                    # shutdown), still track the pending call so
                                    # the later real ToolMessage remains
                                    # authoritative for tool.result status/output.
                                    # If the stream ends first, the terminal
                                    # drains close this tool.use from the same
                                    # pending map.
                                    logger.exception(
                                        "Failed to mount tool widget for %s",
                                        buffer_id,
                                    )
                                else:
                                    _notify_user_visible_output_started()
                                    # Mark running so the group row reflects live
                                    # progress; the row itself is hidden inside
                                    # the group, so this drives state, not a
                                    # visible per-tool spinner.
                                    tool_msg.set_running()
                                    adapter._sync_tool_widget(tool_msg)
                                adapter._current_tool_messages[buffer_id] = tool_msg

                            if buffer_id is not None:
                                tool_call_buffers.pop(buffer_key, None)

                    if getattr(message, "chunk_position", None) == "last":
                        pending_text = pending_text_by_namespace.get(ns_key, "")
                        if pending_text:
                            await _flush_assistant_text_ns(
                                adapter,
                                pending_text,
                                ns_key,
                                assistant_message_by_namespace,
                            )
                            pending_text_by_namespace[ns_key] = ""
                            assistant_message_by_namespace.pop(ns_key, None)

            # Reset summarization state if stream ended mid-summarization
            # (e.g. middleware error, stream exhausted before regular chunks).
            if summarization_in_progress:
                summarization_in_progress = False
                try:
                    await adapter._mount_message(SummarizationMessage())
                except Exception:
                    logger.debug(
                        "Failed to mount summarization notification",
                        exc_info=True,
                    )
                if adapter._set_spinner and not adapter._current_tool_messages:
                    await adapter._set_spinner("Thinking")

            # Flush any remaining text from all namespaces
            for ns_key, pending_text in list(pending_text_by_namespace.items()):
                if pending_text:
                    await _flush_assistant_text_ns(
                        adapter, pending_text, ns_key, assistant_message_by_namespace
                    )
            pending_text_by_namespace.clear()
            assistant_message_by_namespace.clear()

            # Handle HITL after stream completes
            if interrupt_occurred:
                any_rejected = False
                ask_user_cancelled = False
                resume_payload: dict[str, Any] = {}

                # Tools mounted above start their spinner immediately, but a
                # tool blocked on HITL approval or `ask_user` input is not
                # actually running. Pause every in-flight row so none shows a
                # misleading "Running..."; the approve branches below call
                # `set_running` again to resume those that proceed. Guard each
                # row individually so a single bad widget can't abort the whole
                # interrupt handler (mirrors `clear_awaiting_approval` below).
                for tool_msg in adapter._current_tool_messages.values():
                    try:
                        tool_msg.pause_running()
                        adapter._sync_tool_widget(tool_msg)
                    except Exception:
                        logger.exception(
                            "Failed to pause running state on tool widget %s",
                            tool_msg.tool_name,
                        )

                for interrupt_id, ask_req in list(pending_ask_user.items()):
                    questions = ask_req["questions"]
                    tool_args = {"questions": questions}

                    if adapter._request_ask_user:
                        if adapter._set_spinner:
                            await adapter._set_spinner(None)
                        result: AskUserWidgetResult | dict[str, str] = {
                            "type": "error",
                            "error": "ask_user callback returned no response",
                        }
                        try:
                            future = await adapter._request_ask_user(questions)
                        except Exception:
                            logger.exception("Failed to mount ask_user widget")
                            result = {
                                "type": "error",
                                "error": "failed to display ask_user prompt",
                            }
                            future = None

                        if future is None:
                            logger.error(
                                "ask_user callback returned no Future; "
                                "reporting as error"
                            )
                        else:
                            try:
                                future_result = await future
                                if isinstance(future_result, dict):
                                    result = future_result
                                else:
                                    logger.error(
                                        "ask_user future returned non-dict result: %s",
                                        type(future_result).__name__,
                                    )
                                    result = {
                                        "type": "error",
                                        "error": "invalid ask_user widget result",
                                    }
                            except Exception:
                                logger.exception(
                                    "ask_user future resolution failed; "
                                    "reporting as error"
                                )
                                result = {
                                    "type": "error",
                                    "error": "failed to receive ask_user response",
                                }

                        result_type = result.get("type")
                        tool_id = ask_req["tool_call_id"]
                        if result_type == "answered":
                            answers = result.get("answers", [])
                            if isinstance(answers, list):
                                resume_payload[interrupt_id] = {"answers": answers}
                                output = "User answered"
                                tool_msg = adapter._current_tool_messages.pop(
                                    tool_id, None
                                )
                                _dispatch_tool_result_hook(
                                    "ask_user", tool_id, tool_args, "success", output
                                )
                                completed_tool_result_ids.add(tool_id)
                                if tool_msg is not None:
                                    try:
                                        tool_msg.set_success(output)
                                        adapter._sync_tool_widget(tool_msg)
                                    except Exception:
                                        logger.exception(
                                            "Failed to update ask_user row for %s",
                                            tool_id,
                                        )
                                else:
                                    logger.warning(
                                        "ask_user tool_id %s missing from "
                                        "_current_tool_messages on answered",
                                        tool_id,
                                    )
                            else:
                                logger.error(
                                    "ask_user answered payload had non-list "
                                    "answers: %s",
                                    type(answers).__name__,
                                )
                                resume_payload[interrupt_id] = {
                                    "status": "error",
                                    "error": "invalid ask_user answers payload",
                                    "answers": ["" for _ in questions],
                                }
                                any_rejected = True
                                output = "invalid ask_user answers payload"
                                tool_msg = adapter._current_tool_messages.pop(
                                    tool_id, None
                                )
                                _dispatch_tool_error_hook("ask_user")
                                _dispatch_tool_result_hook(
                                    "ask_user", tool_id, tool_args, "error", output
                                )
                                completed_tool_result_ids.add(tool_id)
                                if tool_msg is not None:
                                    try:
                                        tool_msg.set_error(output)
                                        adapter._sync_tool_widget(tool_msg)
                                    except Exception:
                                        logger.exception(
                                            "Failed to update ask_user row for %s",
                                            tool_id,
                                        )
                        elif result_type == "cancelled":
                            resume_payload[interrupt_id] = {
                                "status": "cancelled",
                                "answers": ["" for _ in questions],
                            }
                            any_rejected = True
                            # Halt the turn on cancel; error branches still
                            # resume so the agent can react to the failure.
                            ask_user_cancelled = True
                            tool_msg = adapter._current_tool_messages.pop(tool_id, None)
                            output = "Question cancelled"
                            _dispatch_tool_error_hook("ask_user")
                            _dispatch_tool_result_hook(
                                "ask_user", tool_id, tool_args, "error", output
                            )
                            completed_tool_result_ids.add(tool_id)
                            if tool_msg is not None:
                                try:
                                    tool_msg.set_rejected()
                                    adapter._sync_tool_widget(tool_msg)
                                except Exception:
                                    logger.exception(
                                        "Failed to update ask_user row for %s",
                                        tool_id,
                                    )
                            else:
                                logger.warning(
                                    "ask_user tool_id %s missing from "
                                    "_current_tool_messages on cancelled",
                                    tool_id,
                                )
                        else:
                            error_text = result.get("error")
                            if not isinstance(error_text, str) or not error_text:
                                error_text = "ask_user interaction failed"
                            resume_payload[interrupt_id] = {
                                "status": "error",
                                "error": error_text,
                                "answers": ["" for _ in questions],
                            }
                            any_rejected = True
                            tool_msg = adapter._current_tool_messages.pop(tool_id, None)
                            _dispatch_tool_error_hook("ask_user")
                            _dispatch_tool_result_hook(
                                "ask_user", tool_id, tool_args, "error", error_text
                            )
                            completed_tool_result_ids.add(tool_id)
                            if tool_msg is not None:
                                try:
                                    tool_msg.set_error(error_text)
                                    adapter._sync_tool_widget(tool_msg)
                                except Exception:
                                    logger.exception(
                                        "Failed to update ask_user row for %s",
                                        tool_id,
                                    )
                    else:
                        logger.warning(
                            "ask_user interrupt received but no UI callback is "
                            "registered; reporting as error"
                        )
                        resume_payload[interrupt_id] = {
                            "status": "error",
                            "error": _ASK_USER_UNSUPPORTED_ERROR,
                            "answers": ["" for _ in questions],
                        }
                        tool_id = ask_req["tool_call_id"]
                        tool_msg = adapter._current_tool_messages.pop(tool_id, None)
                        _dispatch_tool_error_hook("ask_user")
                        _dispatch_tool_result_hook(
                            "ask_user",
                            tool_id,
                            tool_args,
                            "error",
                            _ASK_USER_UNSUPPORTED_ERROR,
                        )
                        completed_tool_result_ids.add(tool_id)
                        if tool_msg is not None:
                            try:
                                tool_msg.set_error(_ASK_USER_UNSUPPORTED_ERROR)
                                adapter._sync_tool_widget(tool_msg)
                            except Exception:
                                logger.exception(
                                    "Failed to update ask_user row for %s", tool_id
                                )

                for interrupt_id, hitl_request in list(pending_interrupts.items()):
                    action_requests = hitl_request["action_requests"]

                    if (
                        getattr(session_state, "approval_mode", None)
                        is ApprovalMode.YOLO
                    ):
                        decisions: list[HITLDecision] = [
                            ApproveDecision(type="approve") for _ in action_requests
                        ]
                        resume_payload[interrupt_id] = {"decisions": decisions}
                        for tool_msg in list(adapter._current_tool_messages.values()):
                            tool_msg.set_running()
                            adapter._sync_tool_widget(tool_msg)
                    else:
                        # Batch approval - one dialog for all parallel tool calls
                        await dispatch_hook(
                            "permission.request",
                            {
                                "tool_names": [
                                    r.get("name", "") for r in action_requests
                                ]
                            },
                        )
                        # Hide shell tool widgets while the approval renders
                        # the same command; restore before processing the
                        # decision so subsequent status updates render on the
                        # visible widget. Only applies to single-tool
                        # approvals — the batch dialog doesn't render
                        # per-tool commands, so hiding the rows would leave
                        # the user with no preview of what's being approved.
                        suppressed_tool_msgs = (
                            [
                                tool_msg
                                for tool_msg in adapter._current_tool_messages.values()
                                if tool_msg.tool_name == "execute"
                            ]
                            if len(action_requests) == 1
                            else []
                        )
                        for tool_msg in suppressed_tool_msgs:
                            tool_msg.set_awaiting_approval()
                        try:
                            while True:
                                future = await adapter._request_approval(
                                    action_requests, assistant_id
                                )
                                decision = await future
                                if (
                                    isinstance(decision, dict)
                                    and decision.get("type") == "auto_approve_all"
                                    and adapter._on_auto_approve_enabled is not None
                                ):
                                    callback_result = adapter._on_auto_approve_enabled()
                                    enabled = (
                                        await callback_result
                                        if inspect.isawaitable(callback_result)
                                        else callback_result
                                    )
                                    if enabled is None:
                                        enabled = True
                                    if enabled is False:
                                        continue
                                break
                        finally:
                            for tool_msg in suppressed_tool_msgs:
                                try:
                                    tool_msg.clear_awaiting_approval()
                                except Exception:
                                    logger.exception(
                                        "Failed to clear awaiting-approval "
                                        "state on tool widget %s",
                                        tool_msg.tool_name,
                                    )

                        if isinstance(decision, dict):
                            decision_type = decision.get("type")

                            if decision_type == "auto_approve_all":
                                decisions = [
                                    ApproveDecision(type="approve")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_running()
                                    adapter._sync_tool_widget(tool_msg)
                                for action_request in action_requests:
                                    tool_name = action_request.get("name")
                                    if tool_name in {
                                        "write_file",
                                        "edit_file",
                                        "delete",
                                    }:
                                        args = action_request.get("args", {})
                                        if isinstance(args, dict):
                                            file_op_tracker.mark_hitl_approved(
                                                tool_name, args
                                            )

                            elif decision_type == "switch_manual":
                                if adapter._on_switch_to_manual is None:
                                    msg = "Manual mode callback is unavailable"
                                    raise RuntimeError(msg)
                                callback_result = adapter._on_switch_to_manual()
                                switched = (
                                    await callback_result
                                    if inspect.isawaitable(callback_result)
                                    else callback_result
                                )
                                if not switched:
                                    msg = "Manual mode could not be persisted"
                                    raise RuntimeError(msg)
                                decisions = [
                                    cast("HITLDecision", {"type": "switch_manual"})
                                    for _ in action_requests
                                ]

                            elif decision_type == "approve":
                                decisions = [
                                    ApproveDecision(type="approve")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_running()
                                    adapter._sync_tool_widget(tool_msg)
                                for action_request in action_requests:
                                    tool_name = action_request.get("name")
                                    if tool_name in {
                                        "write_file",
                                        "edit_file",
                                        "delete",
                                    }:
                                        args = action_request.get("args", {})
                                        if isinstance(args, dict):
                                            file_op_tracker.mark_hitl_approved(
                                                tool_name, args
                                            )

                            elif decision_type == "reject":
                                reject_message = decision.get("message")
                                reject_message = (
                                    reject_message
                                    if isinstance(reject_message, str)
                                    and reject_message.strip()
                                    else None
                                )
                                reject_decision: RejectDecision = (
                                    RejectDecision(
                                        type="reject", message=reject_message
                                    )
                                    if reject_message
                                    else RejectDecision(type="reject")
                                )
                                decisions = [reject_decision for _ in action_requests]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_rejected(reason=reject_message)
                                    adapter._sync_tool_widget(tool_msg)
                                # Bare reject aborts an ordinary conversation
                                # turn and shows the canned "Command rejected"
                                # banner. Server operations must receive every
                                # decision so their nested agent can finish
                                # without the rejected context. A supplied
                                # reason likewise resumes either kind of run.
                                if reject_message is None and graph_input is None:
                                    completed_tool_result_ids.update(
                                        _dispatch_terminal_tool_result_hooks(
                                            adapter._current_tool_messages,
                                            "Tool approval rejected",
                                        )
                                    )
                                    adapter._current_tool_messages.clear()
                                    any_rejected = True
                            else:
                                logger.warning(
                                    "Unexpected HITL decision type: %s",
                                    decision_type,
                                )
                                decisions = [
                                    RejectDecision(type="reject")
                                    for _ in action_requests
                                ]
                                for tool_msg in list(
                                    adapter._current_tool_messages.values()
                                ):
                                    tool_msg.set_rejected()
                                    adapter._sync_tool_widget(tool_msg)
                                completed_tool_result_ids.update(
                                    _dispatch_terminal_tool_result_hooks(
                                        adapter._current_tool_messages,
                                        "Tool approval rejected",
                                    )
                                )
                                adapter._current_tool_messages.clear()
                                any_rejected = True
                        else:
                            logger.warning(
                                "HITL decision was not a dict: %s",
                                type(decision).__name__,
                            )
                            decisions = [
                                RejectDecision(type="reject") for _ in action_requests
                            ]
                            for tool_msg in list(
                                adapter._current_tool_messages.values()
                            ):
                                tool_msg.set_rejected()
                                adapter._sync_tool_widget(tool_msg)
                            completed_tool_result_ids.update(
                                _dispatch_terminal_tool_result_hooks(
                                    adapter._current_tool_messages,
                                    "Tool approval rejected",
                                )
                            )
                            adapter._current_tool_messages.clear()
                            any_rejected = True

                        resume_payload[interrupt_id] = {"decisions": decisions}

                        if any_rejected:
                            break

                suppress_resumed_output = any_rejected

            if interrupt_occurred and resume_payload:
                if suppress_resumed_output and (
                    ask_user_cancelled or not pending_ask_user
                ):
                    message = (
                        "Question cancelled. Tell the agent what you'd like instead."
                        if ask_user_cancelled
                        else "Command rejected. Tell the agent what you'd like instead."
                    )
                    await adapter._mount_message(AppMessage(message))
                    turn_stats.wall_time_seconds = time.monotonic() - start_time
                    # Model call already completed (HITL interrupt fires after
                    # the model node); `ResumeStateMiddleware.after_model`
                    # persisted the count, so only refresh UI here.
                    _report_tokens(
                        adapter,
                        captured_input_tokens,
                        captured_output_tokens,
                    )
                    return turn_stats

                stream_input = Command(resume=resume_payload)
            else:
                # Clean stream end. Any tool still in `_current_tool_messages`
                # had its `tool.use` dispatched at mount but never received a
                # `ToolMessage` (e.g. a custom/remote graph that ends the turn
                # after emitting an unexecuted tool call). Close each one with a
                # terminal hook so the "every `tool.use` is terminated" guarantee
                # does not depend on the graph raising. This mirrors the headless
                # `_dispatch_orphaned_tool_result_hooks`, which likewise closes
                # orphans hooks-only (no widget mutation) on every loop exit —
                # the widget keeps its rendered state; only the audit stream and
                # the cross-turn `_current_tool_messages` tracking are settled.
                if adapter._current_tool_messages:
                    logger.info(
                        "Stream ended with %d un-resulted tool call(s); "
                        "closing with terminal hooks",
                        len(adapter._current_tool_messages),
                    )
                    _dispatch_terminal_tool_result_hooks(
                        adapter._current_tool_messages,
                        "Stream ended before tool result",
                    )
                    adapter._current_tool_messages.clear()
                # The end-of-stream diagnostic for buffered tool calls that never
                # fired a `tool.use` runs in the `finally` below, not here, so it
                # fires on cancel and mid-stream error too (not only this clean
                # end) — mirroring the headless surface, whose identical
                # diagnostic lives in `_run_agent_loop`'s `finally`.
                await dispatch_hook("task.complete", {"thread_id": thread_id})
                break

    except (asyncio.CancelledError, KeyboardInterrupt):
        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config=config,
            pending_text_by_namespace=pending_text_by_namespace,
            assistant_message_by_namespace=assistant_message_by_namespace,
            captured_input_tokens=captured_input_tokens,
            captured_output_tokens=captured_output_tokens,
            turn_stats=turn_stats,
            start_time=start_time,
            recover_interrupted_turn=recover_interrupted_turn,
        )
        return turn_stats
    finally:
        # Streamed text is coalesced in each AssistantMessage's `_pending_append`
        # buffer and flushed on a throttled timer, so up to one flush interval of
        # tokens can be in flight at any moment. Normal completion (the flush loop
        # above) and interrupt cleanup both clear the namespace dict, leaving this
        # a no-op there. The path that matters is a non-cancel mid-stream error
        # propagating to the caller: without this drain those buffered tokens are
        # never written and the user sees a silently truncated reply.
        try:
            await _stop_assistant_streams(adapter, assistant_message_by_namespace)
        except Exception:  # drain must not mask the original error
            logger.exception("Failed to drain assistant streams on exit")

        # Self-contained backstop for the "every `tool.use` is terminated" hook
        # guarantee. The clean-end branch, HITL-reject branches, and interrupt
        # cleanup each already drained `_current_tool_messages` and cleared it, so
        # this is a no-op on those paths. The one path it covers is a non-cancel
        # mid-stream error propagating to the caller: without it, the tools that
        # fired `tool.use` would be terminated only by the caller's
        # `finalize_pending_tools_with_error`, leaving the hook guarantee dependent
        # on the caller rather than owned here (a future second caller, or a
        # missing adapter, would leak an unterminated `tool.use`). Runs before the
        # exception reaches the caller, whose `finalize_pending_tools_with_error`
        # then finds an empty dict and no-ops, so no `tool.result` is dispatched
        # twice. Fail-loud and guarded so a dispatch problem can never mask the
        # error propagating from the stream.
        if adapter._current_tool_messages:
            logger.warning(
                "Turn exited with %d un-terminated tool call(s); closing with "
                "terminal hooks as a backstop",
                len(adapter._current_tool_messages),
            )
            try:
                adapter.finalize_pending_tools_with_error(
                    "Agent error before tool result"
                )
            except Exception:
                logger.warning(
                    "Backstop terminal tool close failed unexpectedly",
                    exc_info=True,
                )

        # Surface any buffered tool call that never mounted and never fired a
        # `tool.use`, so it would otherwise vanish with `tool_call_buffers` at turn
        # end with no trace. Two distinct cases (args that never parsed, and args
        # that parsed but carried no tool-call id) are classified by the shared
        # `count_unemitted_tool_calls`. In the `finally` so it fires on every exit
        # path — clean end, cancel, and mid-stream error — matching the headless
        # surface. Info, not warning: nothing executed for these and the
        # precondition (exiting mid-tool-call) is unusual; it only needs to be
        # greppable. Guarded so a logging failure can never mask a propagating
        # exception (`parse_args`, re-run inside the count, can raise on the
        # invariant-violating both-fields-set buffer).
        try:
            unemitted = count_unemitted_tool_calls(tool_call_buffers.values())
            if unemitted.unparsed:
                logger.info(
                    "Stream ended with %d tool call(s) whose arguments never "
                    "parsed; no tool.use was emitted for them",
                    unemitted.unparsed,
                )
            if unemitted.idless_parsed:
                logger.info(
                    "Stream ended with %d tool call(s) whose arguments parsed "
                    "but carried no tool-call id; no tool.use was emitted for "
                    "them",
                    unemitted.idless_parsed,
                )
        except Exception:
            logger.warning(
                "Unparsed tool-call buffer check failed unexpectedly",
                exc_info=True,
            )

    # Update token count and return stats. Persistence is handled inside the
    # graph by `ResumeStateMiddleware.after_model`, so this only refreshes UI.
    turn_stats.wall_time_seconds = time.monotonic() - start_time
    _report_tokens(
        adapter,
        captured_input_tokens,
        captured_output_tokens,
    )
    return turn_stats


async def _stop_assistant_streams(
    adapter: TextualUIAdapter,
    assistant_message_by_namespace: dict[tuple, Any] | None,
) -> None:
    """Finalize active assistant streams during interrupt cleanup."""
    if not assistant_message_by_namespace:
        return

    for current_msg in list(assistant_message_by_namespace.values()):
        try:
            await current_msg.stop_stream()
        except Exception:
            logger.warning("Failed to stop interrupted assistant stream", exc_info=True)
            continue

        if adapter._sync_message_content and current_msg.id:
            adapter._sync_message_content(current_msg.id, current_msg._content)

    assistant_message_by_namespace.clear()


async def _handle_interrupt_cleanup(
    *,
    adapter: TextualUIAdapter,
    agent: Any,  # noqa: ANN401  # Dynamic agent graph type
    config: RunnableConfig,
    pending_text_by_namespace: dict[tuple, str],
    assistant_message_by_namespace: dict[tuple, Any] | None = None,
    captured_input_tokens: int,
    captured_output_tokens: int,
    turn_stats: SessionStats,
    start_time: float,
    recover_interrupted_turn: bool = True,
) -> None:
    """Shared cleanup for CancelledError and KeyboardInterrupt.

    Args:
        adapter: UI adapter with display callbacks.
        agent: The LangGraph agent.
        config: Runnable config with `thread_id`.
        pending_text_by_namespace: Accumulated text per namespace.
        assistant_message_by_namespace: Active assistant message widgets per namespace.
        captured_input_tokens: Input tokens captured before interrupt.
        captured_output_tokens: Output tokens captured before interrupt.
        turn_stats: Stats for the current turn.
        start_time: Monotonic timestamp when the turn began.
        recover_interrupted_turn: Whether to append the normal partial assistant
            and cancellation messages for an interrupted conversation turn.

    Raises:
        ValueError: If proactive remote-run cancellation is attempted without a
            `thread_id` in `config` (a contract violation rather than a
            transient remote failure).
    """
    from langchain_core.messages import HumanMessage

    # Clear active message immediately so it won't block pruning.
    # If we don't do this, the store still thinks it's active and protects
    # from pruning, which breaks get_messages_to_prune(), potentially
    # blocking all future pruning.
    if adapter._set_active_message:
        adapter._set_active_message(None)

    # Hide spinner (may still show "Offloading" if interrupted mid-offload)
    if adapter._set_spinner:
        await adapter._set_spinner(None)

    await _stop_assistant_streams(adapter, assistant_message_by_namespace)

    if recover_interrupted_turn:
        await adapter._mount_message(AppMessage("Interrupted by user"))

    # Proactively cancel server-side runs before persisting recovery state, so
    # the aupdate_state writes below don't 409 against a still-busy thread. This
    # is defense-in-depth layered on top of aupdate_state's own 409 -> cancel ->
    # retry path (see RemoteAgent.aupdate_state); a failure here is not fatal.
    # Absent on local agents, so this is a no-op for them.
    cancel_active_runs = getattr(agent, "acancel_active_runs", None)
    if cancel_active_runs is not None:
        try:
            await cancel_active_runs(config)
        except ValueError:
            # A missing thread_id is a contract violation (a bug), not a
            # transient remote failure — surface it rather than downgrading it
            # to a warning alongside the swallowed network errors below.
            raise
        except Exception:
            # Remote cancel is best-effort defense-in-depth; transient remote
            # failures here are recovered by aupdate_state's 409 retry below.
            logger.warning(
                "Failed to cancel active remote runs for thread %s",
                config.get("configurable", {}).get("thread_id"),
                exc_info=True,
            )

    interrupted_msg = (
        _build_interrupted_ai_message(
            pending_text_by_namespace,
            adapter._current_tool_messages,
        )
        if recover_interrupted_turn
        else None
    )

    # Close out any tool whose `tool.use` fired but whose `ToolMessage` never
    # arrived because the turn was cancelled: emit terminal hooks before the
    # widgets are dropped, so a cancel path leaves no unterminated `tool.use`
    # (mirroring the HITL-reject branches). The turn does not resume from here,
    # so the returned ids need not be tracked for dedup.
    #
    # Dispatched *before* the `aupdate_state` writes below (not alongside the
    # `set_rejected` loop after them): those writes await a possibly-slow remote
    # checkpointer, and on an interactive quit the graceful-exit drain in
    # `app.py` snapshots the in-flight hook tasks right after cancelling this
    # worker. Scheduling the fire-and-forget hooks here — synchronously, as soon
    # as cancellation is observed — guarantees they are in that snapshot and get
    # drained, rather than being scheduled after a slow write and cancelled at
    # loop teardown (a silent audit gap). It reads `tool_msg.args`/`tool_name`,
    # both available regardless of the widget's rejected state.
    #
    # Guarded because this now sits *before* the recovery-state write below: the
    # dispatch never raises by construction today (pure payload builders, and
    # `dispatch_hook_fire_and_forget` swallows serialization inside its task), but
    # this function's whole contract is best-effort-must-not-propagate, so a
    # future change here must never skip the `aupdate_state` save or escape the
    # cancel handler.
    try:
        _dispatch_terminal_tool_result_hooks(
            adapter._current_tool_messages, "Turn cancelled"
        )
    except Exception:
        logger.warning("Terminal tool.result dispatch failed on cancel", exc_info=True)

    # Save accumulated state before marking tools as rejected (best-effort).
    # State update failures shouldn't prevent cleanup.
    from langsmith import tracing_context

    try:
        # tracing_context(enabled=False) suppresses only the UpdateState traced
        # run that each aupdate_state call would otherwise emit in LangSmith — it
        # does not affect any other tracing in the surrounding turn. These writes
        # are internal interrupt-recovery mechanics (partial AI message +
        # cancellation notice), not user-driven agent activity; surfacing them as
        # standalone peer runs alongside real agent turns clutters the trace view.
        with tracing_context(enabled=False):
            if recover_interrupted_turn:
                if interrupted_msg:
                    await agent.aupdate_state(config, {"messages": [interrupted_msg]})

                cancellation_msg = HumanMessage(
                    content=f"{SYSTEM_MESSAGE_PREFIX} Task interrupted by user. "
                    "Previous operation was cancelled."
                )
                cancellation_values: dict[str, Any] = {"messages": [cancellation_msg]}
                # Piggy-back the latest token count on this already-required
                # write instead of issuing a separate `aupdate_state`.
                # `after_model` never ran on the partial turn, so without this
                # the count would be stale on resume.
                captured_total = captured_input_tokens + captured_output_tokens
                if captured_total:
                    cancellation_values["_context_tokens"] = captured_total
                await agent.aupdate_state(config, cancellation_values)
    except (httpx.TransportError, httpx.TimeoutException) as e:
        logger.warning("Could not save interrupted state (network): %s", e)
    except Exception as exc:  # interrupt cleanup must not propagate
        logger.warning("Failed to save interrupted state", exc_info=True)
        # Surface via the chat surface — silent file-only warnings have
        # masked real state-write failures (validation, checkpointer
        # corruption) in past incidents. The mount is best-effort; the
        # adapter may already be tearing down.
        with contextlib.suppress(Exception):
            await adapter._mount_message(
                AppMessage(
                    f"Could not save interrupted state ({type(exc).__name__}). "
                    "Subsequent turns may see stale state."
                )
            )

    # Mark tools as rejected AFTER saving state. Terminal hooks for these were
    # already dispatched before the state writes above (see the comment there).
    # Guard each `set_rejected` — it does DOM work that can raise during
    # app-exit teardown — so a failure can't skip the `clear()` below. If it
    # did, `_current_tool_messages` would stay populated and the caller's
    # `finally` backstop would re-dispatch a duplicate terminal hook for every
    # id already closed at the top of this function.
    for tool_msg in list(adapter._current_tool_messages.values()):
        try:
            tool_msg.set_rejected()
            adapter._sync_tool_widget(tool_msg)
        except Exception:
            logger.exception(
                "Failed to mark tool row rejected during interrupt cleanup"
            )
    adapter._current_tool_messages.clear()

    # Keep the token count marked stale whenever interrupted state was captured,
    # including tool-only turns after assistant text was already flushed.
    approximate = interrupted_msg is not None

    turn_stats.wall_time_seconds = time.monotonic() - start_time
    _report_tokens(
        adapter,
        captured_input_tokens,
        captured_output_tokens,
        approximate=approximate,
    )


def _report_tokens(
    adapter: TextualUIAdapter,
    captured_input_tokens: int,
    captured_output_tokens: int,
    *,
    approximate: bool = False,
) -> None:
    """Refresh the token-count UI display.

    Persistence into graph state is owned by `ResumeStateMiddleware.after_model`
    (normal turns), `_handle_offload` (offload turns), and the interrupt-cleanup
    `aupdate_state` write (partial turns) — never this helper.

    Args:
        adapter: UI adapter with token callbacks.
        captured_input_tokens: Total input tokens captured during the turn.
        captured_output_tokens: Total output tokens captured during the turn.
        approximate: When `True`, signal to the UI that the count is stale
            (e.g. after an interrupted generation) by appending "+".
    """
    if captured_input_tokens or captured_output_tokens:
        if adapter._on_tokens_update:
            adapter._on_tokens_update(captured_input_tokens, approximate=approximate)
    elif adapter._on_tokens_show:
        adapter._on_tokens_show(approximate=approximate)


async def _flush_assistant_text_ns(
    adapter: TextualUIAdapter,
    text: str,
    ns_key: tuple,
    assistant_message_by_namespace: dict[tuple, Any],
) -> None:
    """Flush accumulated assistant text for a specific namespace.

    Finalizes the streaming by stopping the MarkdownStream.
    If no message exists yet, creates one with the full content.
    """
    if not text.strip():
        return

    current_msg = assistant_message_by_namespace.get(ns_key)
    if current_msg is None:
        # No message was created during streaming - create one with full content
        msg_id = f"asst-{uuid.uuid4().hex}"
        current_msg = AssistantMessage(text, id=msg_id)
        await adapter._mount_message(current_msg)
        await current_msg.write_initial_content()
        assistant_message_by_namespace[ns_key] = current_msg
    else:
        # Stop the stream to finalize the content
        await current_msg.stop_stream()

    # When the AssistantMessage was first mounted and recorded in the
    # MessageStore, it had empty content (streaming hadn't started yet).
    # Now that streaming is done, the widget holds the full text in
    # `_content`, but the store's MessageData still has `content=""`.
    # If the message is later pruned and re-hydrated, `to_widget()` would
    # recreate it from that stale empty string. This call copies the
    # widget's final content back into the store so re-hydration works.
    if adapter._sync_message_content and current_msg.id:
        adapter._sync_message_content(current_msg.id, current_msg._content)

    # Clear active message since streaming is done
    if adapter._set_active_message:
        adapter._set_active_message(None)
