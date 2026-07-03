"""Textual UI adapter for agent execution."""
# This module has complex streaming logic ported from execution.py

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

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
    from rich.console import Console

    from deepagents_code._ask_user_types import AskUserWidgetResult, Question

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
)
from deepagents_code.config import build_stream_config, get_glyphs
from deepagents_code.file_ops import FileOpTracker
from deepagents_code.formatting import format_duration
from deepagents_code.hooks import dispatch_hook
from deepagents_code.input import MediaTracker, parse_file_mentions
from deepagents_code.media_utils import create_multimodal_content
from deepagents_code.tool_display import format_tool_message_content
from deepagents_code.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    SummarizationMessage,
    ToolCallMessage,
)

logger = logging.getLogger(__name__)

_hitl_adapter_cache: TypeAdapter | None = None
"""Lazy singleton for the HITL request validator."""

_ASK_USER_UNSUPPORTED_ERROR = "ask_user not supported by this UI"


async def _dispatch_tool_use_hook(
    tool_name: str, tool_id: str, tool_args: dict[str, Any]
) -> None:
    """Dispatch a `tool.use` hook with the documented textual payload."""
    await dispatch_hook(
        "tool.use",
        {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "tool_args": tool_args,
        },
    )


async def _dispatch_tool_result_hook(
    tool_name: str,
    tool_id: str,
    tool_args: dict[str, Any],
    tool_status: str,
    tool_output: str,
) -> None:
    """Dispatch a `tool.result` hook with the documented textual payload."""
    await dispatch_hook(
        "tool.result",
        {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "tool_args": tool_args,
            "tool_status": tool_status,
            "tool_output": tool_output,
        },
    )


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


def print_usage_table(
    stats: SessionStats,
    wall_time: float,
    console: Console,
) -> None:
    """Print a model-usage stats table to a Rich console.

    Each row shows the serving provider alongside the model name. When the
    session spans multiple models each gets its own row with a totals row
    appended; single-model sessions show one row.

    Args:
        stats: Cumulative session stats.
        wall_time: Total wall-clock time in seconds.
        console: Rich console for output.
    """
    from rich.table import Table

    has_time = wall_time >= 0.1  # noqa: PLR2004
    if not (stats.request_count or stats.input_tokens or has_time):
        return

    if stats.per_model:
        multi_model = len(stats.per_model) > 1

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2, 0, 0),
            show_edge=False,
        )
        table.add_column("Provider", style="dim")
        table.add_column("Model", style="dim")
        table.add_column("Reqs", justify="right", style="dim")
        table.add_column("InputTok", justify="right", style="dim")
        table.add_column("OutputTok", justify="right", style="dim")

        if multi_model:
            for ms in stats.per_model.values():
                table.add_row(
                    ms.provider,
                    ms.model_name,
                    str(ms.request_count),
                    format_token_count(ms.input_tokens),
                    format_token_count(ms.output_tokens),
                )
            table.add_row(
                "",
                "Total",
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
            )
        else:
            ms = next(iter(stats.per_model.values()))
            table.add_row(
                ms.provider,
                ms.model_name,
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
            )

        console.print()
        console.print("[bold]Usage Stats[/bold]")
        console.print(table)
    if has_time:
        console.print()
        console.print(
            f"Agent active  {format_duration(wall_time)}",
            style="dim",
            highlight=False,
        )


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


def _format_rubric_event(data: dict[str, Any]) -> str | None:
    """Format a rubric custom-stream event for the chat transcript.

    Returns:
        A user-visible message for rubric events, or `None` for custom-stream
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
    explanation = str(data.get("explanation") or "").strip()
    if result is None:
        return None
    if result == "satisfied":
        return f"{glyphs.checkmark} Acceptance criteria satisfied"
    if result == "needs_revision":
        lines = [
            f"{glyphs.retry} Changes need revision"
            + (f": {explanation}" if explanation else ""),
        ]
        for criterion in data.get("criteria", []):
            if isinstance(criterion, dict) and criterion.get("passed") is False:
                name = str(criterion.get("name", "criterion"))
                gap = str(criterion.get("gap", "")).strip()
                lines.append(f"  {glyphs.error} {name}" + (f" — {gap}" if gap else ""))
        return "\n".join(lines)
    if result == "max_iterations_reached":
        return (
            f"{glyphs.warning} Acceptance criteria not satisfied "
            "(iteration limit reached)"
        )
    if result in {"failed", "grader_error"}:
        label = "grader failed" if result == "failed" else "grader error"
        return (
            f"{glyphs.warning} Rubric "
            + label
            + (f": {explanation}" if explanation else "")
        )
    # A `rubric_evaluation_end` with an unrecognized result is still a terminal
    # grading event; surface it rather than silently dropping it (e.g. if the
    # SDK adds a new verdict the chat would otherwise go quiet mid-turn).
    return f"{glyphs.warning} Rubric grading ended" + (
        f": {explanation}" if explanation else ""
    )


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
        on_auto_approve_enabled: Callable[[], Awaitable[None] | None] | None = None,
        set_spinner: Callable[[SpinnerStatus], Awaitable[None]] | None = None,
        set_active_message: Callable[[str | None], None] | None = None,
        sync_message_content: Callable[[str, str], None] | None = None,
        request_ask_user: (
            Callable[
                [list[Question]],
                Awaitable[asyncio.Future[AskUserWidgetResult] | None],
            ]
            | None
        ) = None,
        on_tool_complete: Callable[[], None] | None = None,
        on_subagent_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the adapter."""
        self._mount_message = mount_message
        """Async callback to mount a message widget to the chat."""

        self._update_status = update_status
        """Callback to update the status bar text."""

        self._request_approval = request_approval
        """Async callback that returns a Future for HITL approval."""

        self._on_auto_approve_enabled = on_auto_approve_enabled
        """Callback invoked when auto-approve is enabled via the HITL approval
        menu.

        Fired when the user selects "Auto-approve all" from an approval dialog,
        allowing the app to sync its status bar and session state.
        """

        self._set_spinner = set_spinner
        """Callback to show/hide loading spinner."""

        self._set_active_message = set_active_message
        """Callback to set the active streaming message ID (pass `None` to clear)."""

        self._sync_message_content = sync_message_content
        """Callback to sync final message content back to the store after streaming."""

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
        """Sync callback fired for each validated `subagent` custom-stream event.

        Drives the live subagent fan-out panel. Events originate from the
        QuickJS `task()` bridge during a `js_eval` call; payload strings are
        LLM/JS-authored and treated as untrusted by the panel renderer.
        """

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

    def finalize_pending_tools_with_error(self, error: str) -> None:
        """Mark all pending/running tool widgets as error and clear tracking.

        This is used as a safety net when an unexpected exception aborts
        streaming before matching `ToolMessage` results are received.

        Args:
            error: Error text to display in each pending tool widget.
        """
        for tool_msg in list(self._current_tool_messages.values()):
            tool_msg.set_error(error)
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
    rubric: str | None = None,
    blocked_goal_retry_context: str | None = None,
    turn_stats: SessionStats | None = None,
) -> SessionStats:
    """Execute a task with output directed to Textual UI.

    This is the Textual-compatible version of execute_task() that uses
    the TextualUIAdapter for all UI operations.

    Args:
        user_input: The user's input message
        agent: The LangGraph agent to execute
        assistant_id: The agent identifier
        session_state: Session state with auto_approve flag
        adapter: The TextualUIAdapter for UI operations
        backend: Optional backend for file operations
        image_tracker: Optional tracker for images
        context: Optional `CLIContext` with model override and params. The
            current approval mode (`session_state.auto_approve`) is written
            into `context["auto_approve"]` on every stream iteration before it
            is passed to the graph via `context=`, so the `interrupt_on` `when`
            predicate can suppress interrupts at the source.
        sandbox_type: Sandbox provider name for trace metadata, or `None`
            if no sandbox is active.
        message_kwargs: Extra fields merged into the stream input message
            dict (e.g., `additional_kwargs` for persisting skill metadata
            in the checkpoint).
        rubric: Acceptance criteria supplied to `RubricMiddleware` via graph
            input state.
        blocked_goal_retry_context: One-turn model context for retrying a
            previously blocked goal. This is carried via runtime context so it
            is not parsed for file mentions or checkpointed as human input.
        turn_stats: Pre-created `SessionStats` to accumulate into.

            When the caller holds a reference to the same object, stats are
            available even if this coroutine is cancelled before it can return.

            If `None`, a new instance is created internally.

    Returns:
        Stats accumulated over this turn (request count, token counts,
            wall-clock time).

    Raises:
        ValidationError: If HITL request validation fails (re-raised).
    """
    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        HITLRequest,
        RejectDecision,
    )
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.types import Command
    from pydantic import ValidationError

    from deepagents_code.approval_mode import awrite_approval_mode

    hitl_request_adapter = _get_hitl_request_adapter(HITLRequest)
    ask_user_adapter = _get_ask_user_adapter()

    # Parse file mentions and inject content if any — offload blocking I/O
    prompt_text, mentioned_files = await asyncio.to_thread(
        parse_file_mentions, user_input
    )

    # Max file size to embed inline (256KB, matching mistral-vibe)
    # Larger files get a reference instead - use read_file tool to view them
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

    # Include images and videos in the message content
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
    if callable(advance_turn):
        turn_id, turn_number = advance_turn()
    else:
        turn_id, turn_number = None, None
    # `build_stream_config` does blocking git filesystem reads and may shell out
    # to `git`; offload it so the Textual event loop stays responsive. Advancing
    # the turn markers above is pure/cheap and stays on the loop.
    config = await asyncio.to_thread(
        build_stream_config,
        thread_id,
        assistant_id,
        sandbox_type=sandbox_type,
        turn_id=turn_id,
        turn_number=turn_number,
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
    displayed_tool_ids: set[str] = set()
    tool_call_buffers: dict[str | int, dict] = {}

    # Track pending text and assistant messages PER NAMESPACE to avoid interleaving
    # when multiple subagents stream in parallel
    pending_text_by_namespace: dict[tuple, str] = {}
    assistant_message_by_namespace: dict[tuple, Any] = {}

    # Clear media from tracker after creating the message
    if image_tracker:
        image_tracker.clear()

    user_msg: dict[str, Any] = {"role": "user", "content": message_content}
    if message_kwargs:
        user_msg.update(message_kwargs)
    # Auto-approve is carried via run context (set per stream iteration below),
    # not graph state — so the initial input is a plain dict. A first-turn
    # `Command(update=...)` would be rebuilt with `goto=None` by the LangGraph
    # API server and crash `_control_branch` on a fresh thread.
    stream_input: dict | Command = {"messages": [user_msg]}
    if rubric:
        stream_input["rubric"] = rubric

    # Track summarization lifecycle so spinner status and notification stay in sync.
    summarization_in_progress = False

    try:
        while True:
            interrupt_occurred = False
            suppress_resumed_output = False
            pending_interrupts: dict[str, HITLRequest] = {}
            pending_ask_user: dict[str, AskUserRequest] = {}

            # Carry the current approval mode into run context so the
            # `interrupt_on` `when` predicate can suppress interrupts at the
            # source. Also write the live store item that the server-side
            # predicate re-reads on each tool call, so toggling approval mode
            # mid-stream (either direction) takes effect before the current
            # stream returns. Turning auto-approve off is the safety-critical
            # direction, but the same store write also propagates turning it on.
            if context is None:
                context = CLIContext()
            context["thread_id"] = thread_id
            if blocked_goal_retry_context is not None:
                context["blocked_goal_retry_context"] = blocked_goal_retry_context
            else:
                context.pop("blocked_goal_retry_context", None)
            auto_approve = bool(session_state.auto_approve)
            context["auto_approve"] = auto_approve
            try:
                live_key = await awrite_approval_mode(
                    agent,
                    thread_id,
                    auto_approve=auto_approve,
                )
            except Exception:
                logger.warning(
                    "Failed to write live approval mode; interrupting for safety",
                    exc_info=True,
                )
                context["auto_approve"] = False
                context.pop("approval_mode_key", None)
                session_state.approval_mode_key = None
            else:
                if live_key is None:
                    context.pop("approval_mode_key", None)
                else:
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
                    if formatted_rubric_event is not None and is_main_agent:
                        await adapter._mount_message(AppMessage(formatted_rubric_event))
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
                            # Panel rendering must never crash the stream loop.
                            logger.exception("subagent panel event handler failed")
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
                                            await _dispatch_tool_use_hook(
                                                "ask_user", tool_id, tool_args
                                            )
                                            tool_msg = ToolCallMessage(
                                                "ask_user",
                                                tool_args,
                                            )
                                            try:
                                                await adapter._mount_message(tool_msg)
                                            except Exception:
                                                logger.exception(
                                                    "Failed to mount ask_user "
                                                    "tool row for %s",
                                                    tool_id,
                                                )
                                            else:
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
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(message.content)
                        record = file_op_tracker.complete_with_message(message)

                        # Update tool call status with output
                        tool_id = getattr(message, "tool_call_id", None)
                        if tool_id and tool_id in adapter._current_tool_messages:
                            # Pop before widget calls so the dict drains even
                            # if set_success/set_error raises.
                            tool_msg = adapter._current_tool_messages.pop(tool_id)
                            output_str = str(tool_content) if tool_content else ""
                            if tool_status == "success":
                                tool_msg.set_success(output_str)
                            else:
                                tool_msg.set_error(output_str or "Error")
                                await dispatch_hook(
                                    "tool.error",
                                    {"tool_names": [tool_msg._tool_name]},
                                )
                            await _dispatch_tool_result_hook(
                                tool_msg._tool_name,
                                tool_id,
                                tool_msg._args,
                                tool_status,
                                output_str,
                            )
                        elif tool_id:
                            logger.debug(
                                "ToolMessage tool_call_id=%s not in "
                                "_current_tool_messages; spinner gating "
                                "may be stale",
                                tool_id,
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
                                    DiffMessage(record.diff, record.display_path)
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

                    # Extract token usage (before content_blocks check
                    # - usage may be on any chunk)
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

                        elif block_type in {"tool_call_chunk", "tool_call"}:
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            buffer_key: str | int
                            if chunk_index is not None:
                                buffer_key = chunk_index
                            elif chunk_id is not None:
                                buffer_key = chunk_id
                            else:
                                buffer_key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                buffer_key,
                                {
                                    "name": None,
                                    "id": None,
                                    "args": None,
                                    "args_parts": [],
                                },
                            )

                            if chunk_name:
                                buffer["name"] = chunk_name
                            if chunk_id:
                                buffer["id"] = chunk_id

                            if isinstance(chunk_args, dict):
                                buffer["args"] = chunk_args
                                buffer["args_parts"] = []
                            elif isinstance(chunk_args, str):
                                if chunk_args:
                                    parts: list[str] = buffer.setdefault(
                                        "args_parts", []
                                    )
                                    if not parts or chunk_args != parts[-1]:
                                        parts.append(chunk_args)
                            elif chunk_args is not None:
                                buffer["args"] = chunk_args

                            buffer_name = buffer.get("name")
                            buffer_id = buffer.get("id")
                            if buffer_name is None:
                                continue

                            # Resolve the tool arguments. String fragments are
                            # accumulated in `args_parts` and joined + parsed
                            # once the buffer holds a complete JSON value. Re-
                            # joining and re-parsing the whole prefix on every
                            # fragment is O(n^2) and ran on the UI event loop for
                            # large `edit_file` blobs. Each `continue` below
                            # leaves the buffer in `tool_call_buffers` so the next
                            # fragment keeps accumulating; it is popped only after
                            # a successful parse + mount.
                            direct_args = buffer.get("args")
                            if isinstance(direct_args, dict):
                                parsed_args = direct_args
                            elif direct_args is not None:
                                parsed_args = {"value": direct_args}
                            else:
                                parts = buffer.get("args_parts") or []
                                if not parts:
                                    continue
                                joined = "".join(parts)
                                stripped = joined.strip()
                                if not stripped:
                                    continue
                                # Objects/arrays can be large (e.g. `edit_file`
                                # blobs), so defer parsing until the closing
                                # bracket arrives. Scalars are always small and
                                # never end in `}`/`]`, so parse them eagerly
                                # rather than leaving them stuck unparsed.
                                if stripped[0] in "{[" and not stripped.endswith(
                                    ("}", "]")
                                ):
                                    continue
                                try:
                                    parsed_args = json.loads(joined)
                                except json.JSONDecodeError:
                                    continue
                                if not isinstance(parsed_args, dict):
                                    parsed_args = {"value": parsed_args}

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
                                await _dispatch_tool_use_hook(
                                    buffer_name, buffer_id, parsed_args
                                )
                                tool_msg = ToolCallMessage(buffer_name, parsed_args)
                                await adapter._mount_message(tool_msg)
                                adapter._current_tool_messages[buffer_id] = tool_msg
                                # Mark running so the group row reflects live
                                # progress; the row itself is hidden inside the
                                # group, so this drives state, not a visible
                                # per-tool spinner.
                                tool_msg.set_running()

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
                                if tool_msg is not None:
                                    tool_msg.set_success(output)
                                else:
                                    logger.warning(
                                        "ask_user tool_id %s missing from "
                                        "_current_tool_messages on answered",
                                        tool_id,
                                    )
                                await _dispatch_tool_result_hook(
                                    "ask_user", tool_id, tool_args, "success", output
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
                                if tool_msg is not None:
                                    tool_msg.set_error(output)
                                await dispatch_hook(
                                    "tool.error",
                                    {"tool_names": ["ask_user"]},
                                )
                                await _dispatch_tool_result_hook(
                                    "ask_user", tool_id, tool_args, "error", output
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
                            if tool_msg is not None:
                                tool_msg.set_rejected()
                            else:
                                logger.warning(
                                    "ask_user tool_id %s missing from "
                                    "_current_tool_messages on cancelled",
                                    tool_id,
                                )
                            output = "Question cancelled"
                            await dispatch_hook(
                                "tool.error",
                                {"tool_names": ["ask_user"]},
                            )
                            await _dispatch_tool_result_hook(
                                "ask_user", tool_id, tool_args, "error", output
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
                            if tool_msg is not None:
                                tool_msg.set_error(error_text)
                            await dispatch_hook(
                                "tool.error",
                                {"tool_names": ["ask_user"]},
                            )
                            await _dispatch_tool_result_hook(
                                "ask_user", tool_id, tool_args, "error", error_text
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
                        if tool_msg is not None:
                            tool_msg.set_error(_ASK_USER_UNSUPPORTED_ERROR)
                        await dispatch_hook(
                            "tool.error",
                            {"tool_names": ["ask_user"]},
                        )
                        await _dispatch_tool_result_hook(
                            "ask_user",
                            tool_id,
                            tool_args,
                            "error",
                            _ASK_USER_UNSUPPORTED_ERROR,
                        )

                for interrupt_id, hitl_request in list(pending_interrupts.items()):
                    action_requests = hitl_request["action_requests"]

                    if session_state.auto_approve:
                        decisions: list[HITLDecision] = [
                            ApproveDecision(type="approve") for _ in action_requests
                        ]
                        resume_payload[interrupt_id] = {"decisions": decisions}
                        for tool_msg in list(adapter._current_tool_messages.values()):
                            tool_msg.set_running()
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
                            future = await adapter._request_approval(
                                action_requests, assistant_id
                            )
                            decision = await future
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
                                session_state.auto_approve = True
                                # The resuming stream re-reads
                                # `session_state.auto_approve` into run context
                                # at the top of the loop, so the `interrupt_on`
                                # `when` predicate suppresses interrupts on the
                                # remaining tool calls in this turn — keeping it
                                # a single run instead of resuming after each.
                                if adapter._on_auto_approve_enabled:
                                    callback_result = adapter._on_auto_approve_enabled()
                                    if callback_result is not None:
                                        await callback_result
                                decisions = [
                                    ApproveDecision(type="approve")
                                    for _ in action_requests
                                ]
                                tool_msgs = list(
                                    adapter._current_tool_messages.values()
                                )
                                for tool_msg in tool_msgs:
                                    tool_msg.set_running()
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
                                adapter._current_tool_messages.clear()
                                # Bare reject aborts the turn and shows the
                                # canned "Command rejected" banner so the user
                                # can redirect. When a reason is supplied, the
                                # reason itself serves as feedback for the
                                # agent: keep `any_rejected=False` so the
                                # stream resumes and the banner is suppressed.
                                if reject_message is None:
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

    interrupted_msg = _build_interrupted_ai_message(
        pending_text_by_namespace,
        adapter._current_tool_messages,
    )

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
            if interrupted_msg:
                await agent.aupdate_state(config, {"messages": [interrupted_msg]})

            cancellation_msg = HumanMessage(
                content=f"{SYSTEM_MESSAGE_PREFIX} Task interrupted by user. "
                "Previous operation was cancelled."
            )
            cancellation_values: dict[str, Any] = {"messages": [cancellation_msg]}
            # Piggy-back the latest token count on this already-required write
            # instead of issuing a separate `aupdate_state`. `after_model` never
            # ran on the partial turn, so without this the count would be stale
            # on resume.
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

    # Mark tools as rejected AFTER saving state
    for tool_msg in list(adapter._current_tool_messages.values()):
        tool_msg.set_rejected()
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
