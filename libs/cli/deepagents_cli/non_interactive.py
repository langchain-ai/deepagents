"""Non-interactive execution mode for deepagents CLI.

Provides `run_non_interactive` which runs a single user task against the
agent graph, streams results to stdout, and exits with an appropriate code.

The agent runs inside a `langgraph dev` server subprocess, connected via
the `RemoteAgent` client (see `server_manager.server_session`).

Shell commands are gated by an optional allow-list (`--shell-allow-list`):

- Not set → shell disabled, all other tool calls auto-approved.
- `recommended` or explicit list → shell enabled, commands validated
    against the list; non-shell tools approved unconditionally.
- `all` → shell enabled, any command allowed, all tools auto-approved.

An optional quiet mode (`--quiet` / `-q`) redirects all console output to
stderr, leaving stdout exclusively for the agent's response text.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.human_in_the_loop import ActionRequest, HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.live import Live
from rich.markup import escape as escape_markup
from rich.spinner import Spinner as RichSpinner
from rich.style import Style
from rich.text import Text

from deepagents_cli._version import __version__
from deepagents_cli.agent import DEFAULT_AGENT_NAME
from deepagents_cli.config import (
    SHELL_ALLOW_ALL,
    SHELL_TOOL_NAMES,
    build_langsmith_thread_url,
    create_model,
    is_shell_command_allowed,
    settings,
)
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.hooks import dispatch_hook, dispatch_hook_fire_and_forget
from deepagents_cli.model_config import ModelConfigError
from deepagents_cli.sessions import generate_thread_id
from deepagents_cli.textual_adapter import SessionStats, print_usage_table
from deepagents_cli.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    summarize_issues,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class HITLIterationLimitError(RuntimeError):
    """Raised when the HITL interrupt loop exceeds `_MAX_HITL_ITERATIONS` rounds."""


_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)

_STREAM_CHUNK_LENGTH = 3
"""Expected element counts for the tuples emitted by agent.astream.

Stream chunks are 3-tuples: (namespace, stream_mode, data).
"""

_MESSAGE_DATA_LENGTH = 2
"""Message-mode data is a 2-tuple: (message_obj, metadata)."""

_MAX_HITL_ITERATIONS = 50
"""Safety cap on the number of HITL interrupt round-trips to prevent infinite
loops (e.g. when the agent keeps retrying rejected commands)."""


def _write_text(text: str) -> None:
    """Write agent response text to stdout (without a trailing newline).

    Uses `sys.stdout` directly (rather than the Rich Console) so that agent
    response text always appears on stdout, even in quiet mode where the
    Console is redirected to stderr.

    Args:
        text: The text string to write.
    """
    sys.stdout.write(text)
    sys.stdout.flush()


def _write_newline() -> None:
    """Write a newline to stdout (and flush)."""
    sys.stdout.write("\n")
    sys.stdout.flush()


class _ConsoleSpinner:
    """Animated spinner for non-interactive verbose output.

    Uses Rich's `Live` display with a transient braille-dot spinner that
    disappears when stopped, keeping terminal output clean.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: Live | None = None

    def start(self, message: str = "Working...") -> None:
        """Start the spinner with the given message.

        No-op if the spinner is already running. Fails silently if the console
        cannot support live display.

        Args:
            message: Status text to display next to the spinner.
        """
        if self._live is not None:
            return
        renderable = RichSpinner(
            "dots",
            text=Text(f" {message}", style="dim"),
            style="dim",
        )
        try:
            self._live = Live(renderable, console=self._console, transient=True)
            self._live.start()
        except (AttributeError, TypeError, OSError) as exc:
            logger.warning("Spinner start failed: %s", exc)
            self._live = None

    def stop(self) -> None:
        """Stop the spinner if running. Can be restarted with `start`."""
        if self._live is not None:
            try:
                self._live.stop()
            except (AttributeError, TypeError, OSError) as exc:
                logger.warning("Spinner stop failed: %s", exc)
            finally:
                self._live = None


@dataclass
class StreamState:
    """Mutable state accumulated while iterating over the agent stream."""

    quiet: bool = False
    """When `True`, diagnostic formatting that would otherwise go to stdout
    (e.g. separator newlines before tool notifications) is suppressed so that
    stdout contains only agent response text."""

    stream: bool = True
    """When `True` (default), text chunks are written to stdout as they arrive.

    When `False`, text is buffered in `full_response` and flushed after the
    agent finishes.
    """

    full_response: list[str] = field(default_factory=list)
    """Accumulated text fragments from the AI message stream."""

    tool_call_buffers: dict[int | str, dict[str, str | None]] = field(
        default_factory=dict
    )
    """Maps a tool-call index or ID to its name/ID metadata for in-progress
    tool calls."""

    pending_interrupts: dict[str, HITLRequest] = field(default_factory=dict)
    """Maps interrupt IDs to their validated HITL requests that are awaiting
    decisions."""

    hitl_response: dict[str, dict[str, list[dict[str, str]]]] = field(
        default_factory=dict
    )
    """Maps interrupt IDs to dicts containing a `'decisions'` key with a list of
    decision dicts (each having a `'type'` key of `'approve'` or `'reject'`).

    Used to resume the agent after HITL processing.
    """

    interrupt_occurred: bool = False
    """Flag indicating whether any HITL interrupt was received during the
    current stream pass."""

    stats: SessionStats = field(default_factory=SessionStats)
    """Accumulated model usage stats for this stream."""

    spinner: _ConsoleSpinner | None = None
    """Optional animated spinner shown during agent work in verbose mode."""


@dataclass
class ThreadUrlLookupState:
    """Best-effort background LangSmith thread URL lookup state.

    Thread safety: the background thread sets `url` then calls `done.set()`.
    Consumers must check `done.is_set()` before reading `url`.
    """

    done: threading.Event = field(default_factory=threading.Event)
    url: str | None = None


def _start_langsmith_thread_url_lookup(thread_id: str) -> ThreadUrlLookupState:
    """Start background LangSmith URL resolution without blocking.

    Args:
        thread_id: Thread identifier to resolve.

    Returns:
        Mutable lookup state whose completion can be checked later.
    """
    state = ThreadUrlLookupState()

    def _resolve() -> None:
        try:
            state.url = build_langsmith_thread_url(thread_id)
        except Exception:  # build_langsmith_thread_url already handles known errors
            logger.debug(
                "Could not resolve LangSmith thread URL for '%s'",
                thread_id,
                exc_info=True,
            )
        finally:
            state.done.set()

    threading.Thread(target=_resolve, daemon=True).start()
    return state


def _process_interrupts(
    data: dict[str, list[Interrupt]],
    state: StreamState,
    console: Console,
) -> None:
    """Extract HITL interrupts from an `updates` chunk and record them.

    Args:
        data: The `updates` dict that contains an `__interrupt__` key.
        state: Stream state to update with new pending interrupts.
        console: Rich console for user-visible warnings.
    """
    interrupts = data["__interrupt__"]
    if interrupts:
        for interrupt_obj in interrupts:
            try:
                validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                    interrupt_obj.value
                )
            except ValidationError:
                logger.warning(
                    "Rejecting malformed HITL interrupt %s (raw value: %r)",
                    interrupt_obj.id,
                    interrupt_obj.value,
                )
                console.print(
                    f"[yellow]Warning: Received malformed tool approval "
                    f"request (interrupt {interrupt_obj.id}). Rejecting.[/yellow]"
                )
                # Fail-closed: record a reject decision for malformed interrupts

                state.hitl_response[interrupt_obj.id] = {
                    "decisions": [{"type": "reject", "message": "Malformed interrupt"}]
                }
                continue
            state.pending_interrupts[interrupt_obj.id] = validated_request
            state.interrupt_occurred = True
            dispatch_hook_fire_and_forget("input.required", {})


def _process_ai_message(
    message_obj: AIMessage,
    state: StreamState,
    console: Console,
) -> None:
    """Extract text and tool-call blocks from an AI message and render them.

    When streaming is enabled, text blocks are written to stdout immediately;
    otherwise they are accumulated in `state.full_response` for deferred
    output. Tool-call blocks are buffered and their names are printed to the
    console.

    Args:
        message_obj: The `AIMessage` received from the stream.
        state: Stream state for accumulating response text and tool-call buffers.
        console: Rich console for formatted output.
    """
    # Extract token usage for stats accumulation
    usage = getattr(message_obj, "usage_metadata", None)
    if usage:
        input_toks = usage.get("input_tokens", 0)
        output_toks = usage.get("output_tokens", 0)
        total_toks = usage.get("total_tokens", 0)
        active_model = settings.model_name or ""
        if input_toks or output_toks:
            state.stats.record_request(active_model, input_toks, output_toks)
        elif total_toks:
            state.stats.record_request(active_model, total_toks, 0)

    if not hasattr(message_obj, "content_blocks"):
        logger.debug("AIMessage missing content_blocks attribute, skipping")
        return
    for block in message_obj.content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                if state.stream:
                    if state.spinner:
                        state.spinner.stop()
                    _write_text(text)
                state.full_response.append(text)
        elif block_type in {"tool_call_chunk", "tool_call"}:
            chunk_name = block.get("name")
            chunk_id = block.get("id")
            chunk_index = block.get("index")

            if chunk_index is not None:
                buffer_key: int | str = chunk_index
            elif chunk_id is not None:
                buffer_key = chunk_id
            else:
                buffer_key = f"unknown-{len(state.tool_call_buffers)}"

            if buffer_key not in state.tool_call_buffers:
                state.tool_call_buffers[buffer_key] = {"name": None, "id": None}
            if chunk_name:
                state.tool_call_buffers[buffer_key]["name"] = chunk_name
                if state.spinner:
                    state.spinner.stop()
                if state.full_response and not state.quiet:
                    _write_newline()
                console.print(
                    f"[dim]🔧 Calling tool: {escape_markup(chunk_name)}[/dim]",
                    highlight=False,
                )


def _process_message_chunk(
    data: tuple[AIMessage | ToolMessage, dict[str, str]],
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """Handle a `messages`-mode chunk from the stream.

    Dispatches to AI-message or tool-message processing depending on the
    message type.

    Args:
        data: A 2-tuple of `(message_obj, metadata)` from the messages
            stream mode.
        state: Shared stream state.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
    """
    if not isinstance(data, tuple) or len(data) != _MESSAGE_DATA_LENGTH:
        logger.debug(
            "Unexpected message-mode data (type=%s), skipping", type(data).__name__
        )
        return

    message_obj, metadata = data

    # The summarization middleware injects synthetic messages to compress
    # conversation history for the LLM. These are internal bookkeeping and
    # should not be rendered to the user.
    if metadata and metadata.get("lc_source") == "summarization":
        return

    if isinstance(message_obj, AIMessage):
        _process_ai_message(message_obj, state, console)
    elif isinstance(message_obj, ToolMessage):
        record = file_op_tracker.complete_with_message(message_obj)
        if record and record.diff:
            if state.spinner:
                state.spinner.stop()
            console.print(
                f"[dim]📝 {escape_markup(record.display_path)}[/dim]",
                highlight=False,
            )
        if state.spinner:
            state.spinner.start()


def _process_stream_chunk(
    chunk: object,
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """Route a single raw stream chunk to the appropriate handler.

    Only main-agent chunks are processed; sub-agent output is ignored so
    that only top-level content is rendered.

    Args:
        chunk: A raw element yielded by `agent.astream`.

            Expected to be a 3-tuple `(namespace, stream_mode, data)` for
            main-agent output.
        state: Shared stream state.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
    """
    if not isinstance(chunk, tuple) or len(chunk) != _STREAM_CHUNK_LENGTH:
        logger.debug(
            "Unexpected stream chunk (type=%s), skipping", type(chunk).__name__
        )
        return

    namespace, stream_mode, data = chunk
    is_main_agent = not namespace

    if not is_main_agent:
        return

    if stream_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
        _process_interrupts(cast("dict[str, list[Interrupt]]", data), state, console)
    elif stream_mode == "messages":
        _process_message_chunk(
            cast("tuple[AIMessage | ToolMessage, dict[str, str]]", data),
            state,
            console,
            file_op_tracker,
        )


def _make_hitl_decision(
    action_request: ActionRequest, console: Console
) -> dict[str, str]:
    """Decide whether to approve or reject a single action request.

    This function is only invoked when a restrictive shell allow-list is
    configured (not `all`). When shell is disabled or unrestricted,
    `interrupt_on` is empty and this function is bypassed entirely.

    Shell tools are always gated: if an allow-list is configured, the command
    is validated against it; if no allow-list is configured, shell commands
    are rejected outright (defense-in-depth — the caller should disable
    shell tools when no allow-list is present, but this function fails
    closed regardless). Non-shell tools are approved unconditionally.

    Args:
        action_request: The action-request dict emitted by the HITL middleware.

            Must contain at least a `name` key.
        console: Rich console for status output.

    Returns:
        Decision dict with a `type` key (`"approve"` or `"reject"`)
            and an optional `message` key with a human-readable explanation.
    """
    for warning in _collect_action_request_warnings(action_request):
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    action_name = action_request.get("name", "")

    if action_name in SHELL_TOOL_NAMES:
        if not settings.shell_allow_list:
            command = action_request.get("args", {}).get("command", "")
            console.print(
                f"\n[red]Shell command rejected (no allow-list configured): "
                f"{command}[/red]"
            )
            return {
                "type": "reject",
                "message": (
                    "Shell commands are not permitted in non-interactive mode "
                    "without a --shell-allow-list. Use --shell-allow-list to "
                    "specify allowed commands."
                ),
            }

        command = action_request.get("args", {}).get("command", "")

        if is_shell_command_allowed(command, settings.shell_allow_list):
            console.print(f"[dim]✓ Auto-approved: {escape_markup(command)}[/dim]")
            return {"type": "approve"}

        allowed_list_str = ", ".join(settings.shell_allow_list)
        console.print(f"\n[red]Shell command rejected:[/red] {escape_markup(command)}")
        console.print(
            f"[yellow]Allowed commands:[/yellow] {escape_markup(allowed_list_str)}"
        )
        return {
            "type": "reject",
            "message": (
                f"Command '{command}' is not in the allow-list. "
                f"Allowed commands: {allowed_list_str}. "
                f"Please use allowed commands or try another approach."
            ),
        }

    console.print(f"[dim]✓ Auto-approved action: {escape_markup(action_name)}[/dim]")
    return {"type": "approve"}


def _collect_action_request_warnings(action_request: ActionRequest) -> list[str]:
    """Collect Unicode/URL safety warnings for one action request.

    Recursively inspects all nested string values in action arguments.

    Returns:
        Warning messages for suspicious values in action arguments.
    """
    warnings: list[str] = []
    args = action_request.get("args", {})
    if not isinstance(args, dict):
        return warnings

    tool_name = str(action_request.get("name", "unknown"))

    for arg_path, text in iter_string_values(args):
        issues = detect_dangerous_unicode(text)
        if issues:
            warnings.append(
                f"{tool_name}.{arg_path} contains hidden Unicode "
                f"({summarize_issues(issues)})"
            )

        if looks_like_url_key(arg_path):
            safety = check_url_safety(text)
            if safety.safe:
                continue
            detail = format_warning_detail(safety.warnings)
            if safety.decoded_domain:
                detail = f"{detail}; decoded host: {safety.decoded_domain}"
            warnings.append(f"{tool_name}.{arg_path} URL warning: {detail}")

    return warnings


def _process_hitl_interrupts(state: StreamState, console: Console) -> None:
    """Iterate over pending HITL interrupts and build approval/rejection responses.

    After processing, `state.pending_interrupts` is cleared and decisions
    are written into `state.hitl_response` so the agent can be resumed.

    Args:
        state: Stream state containing the pending interrupts to process.
        console: Rich console for status output.
    """
    current_interrupts = dict(state.pending_interrupts)
    state.pending_interrupts.clear()

    for interrupt_id, hitl_request in current_interrupts.items():
        decisions = [
            _make_hitl_decision(action_request, console)
            for action_request in hitl_request["action_requests"]
        ]
        state.hitl_response[interrupt_id] = {"decisions": decisions}


async def _stream_agent(
    agent: Any,  # noqa: ANN401
    stream_input: dict[str, Any] | Command,
    config: RunnableConfig,
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """Consume the full agent stream and update *state* with results.

    Args:
        agent: The agent (Pregel or RemoteAgent).
        stream_input: Either the initial user message dict or a
            `Command(resume=...)` for HITL continuation.
        config: LangGraph runnable config (thread ID, metadata, etc.).
        state: Shared stream state.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
    """
    if state.spinner:
        state.spinner.start()
    try:
        async for chunk in agent.astream(
            stream_input,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config=config,
            durability="exit",
        ):
            _process_stream_chunk(chunk, state, console, file_op_tracker)
    finally:
        if state.spinner:
            state.spinner.stop()


async def _run_agent_loop(
    agent: Any,  # noqa: ANN401
    message: str,
    config: RunnableConfig,
    console: Console,
    file_op_tracker: FileOpTracker,
    *,
    quiet: bool = False,
    stream: bool = True,
    message_kwargs: dict[str, Any] | None = None,
    thread_url_lookup: ThreadUrlLookupState | None = None,
    max_turns: int | None = None,
) -> None:
    """Run the agent and handle HITL interrupts until the task completes.

    The loop processes at most `_MAX_HITL_ITERATIONS` rounds to prevent
    runaway retries (e.g. the agent repeatedly attempting rejected commands).

    Args:
        agent: The agent (Pregel or RemoteAgent).
        message: The user's task message.
        config: LangGraph runnable config.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
        quiet: Suppress diagnostic formatting on stdout.
        stream: When `True`, text is written to stdout as it arrives.

            When `False`, the full response is buffered and flushed at
            the end.
        message_kwargs: Extra fields merged into the initial HumanMessage
            dict (e.g., `additional_kwargs` for persisted skill metadata).
        thread_url_lookup: Optional non-blocking lookup state for rendering
            a fast-follow LangSmith thread link.
        max_turns: Optional user-supplied cap on agentic turns.

            When set, the user's value is honoured verbatim and overrides the
            internal safety default; exceeding it
            raises `HITLIterationLimitError`.

    Raises:
        HITLIterationLimitError: If the HITL iteration limit is exceeded.
    """
    spinner = None if quiet else _ConsoleSpinner(console)
    state = StreamState(quiet=quiet, stream=stream, spinner=spinner)
    user_msg: dict[str, Any] = {"role": "user", "content": message}
    if message_kwargs:
        user_msg.update(message_kwargs)
    stream_input: dict[str, Any] | Command = {"messages": [user_msg]}

    thread_id = config.get("configurable", {}).get("thread_id", "")
    await dispatch_hook("session.start", {"thread_id": thread_id})

    start_time = time.monotonic()

    # Initial stream
    await _stream_agent(agent, stream_input, config, state, console, file_op_tracker)

    # When --max-turns is explicitly set, honour the user's value verbatim;
    # the operator has opted in to a specific bound. The internal default
    # only applies when no flag is provided, guarding against unbounded
    # runaway loops in scripts that forgot to set one.
    effective_limit = max_turns if max_turns is not None else _MAX_HITL_ITERATIONS

    # Handle HITL interrupts
    iterations = 0
    while state.interrupt_occurred:
        iterations += 1
        if iterations > effective_limit:
            limit_source = (
                f"--max-turns {max_turns}"
                if max_turns is not None
                else f"the internal safety default of {_MAX_HITL_ITERATIONS}"
            )
            msg = (
                f"Exceeded {effective_limit} agentic turns ({limit_source}). "
                "The agent may be stuck retrying rejected commands. "
                "Increase --max-turns or break the task into smaller steps."
            )
            raise HITLIterationLimitError(msg)
        state.interrupt_occurred = False
        state.hitl_response.clear()
        _process_hitl_interrupts(state, console)
        stream_input = Command(resume=state.hitl_response)
        await _stream_agent(
            agent, stream_input, config, state, console, file_op_tracker
        )

    wall_time = time.monotonic() - start_time

    if state.full_response:
        if not state.stream:
            _write_text("".join(state.full_response))
        _write_newline()

    if not quiet:
        console.print()
        if (
            thread_url_lookup is not None
            and thread_url_lookup.done.is_set()
            and thread_url_lookup.url
        ):
            link_text = Text("View in LangSmith: ", style="dim")
            link_text.append(
                thread_url_lookup.url,
                style=Style(dim=True, link=thread_url_lookup.url),
            )
            console.print(link_text)
        console.print("[green]✓ Task completed[/green]")
        print_usage_table(state.stats, wall_time, console)

    await dispatch_hook("task.complete", {"thread_id": thread_id})
    await dispatch_hook("session.end", {"thread_id": thread_id})


def _build_non_interactive_header(
    assistant_id: str,
    thread_id: str,
    *,
    include_thread_link: bool = False,
) -> Text:
    """Build the non-interactive mode header with model, agent, and thread info.

    By default, this function avoids LangSmith network lookups and renders the
    thread ID as plain text. Callers can opt in to hyperlink resolution.

    Args:
        assistant_id: Agent identifier.
        thread_id: Thread identifier.
        include_thread_link: Whether to resolve and render a LangSmith link for
            the thread ID.

    Returns:
        Rich Text object with the formatted header line.
    """
    default_label = " (default)" if assistant_id == DEFAULT_AGENT_NAME else ""
    parts: list[tuple[str, str | Style]] = [
        (f"CLI: v{__version__}", "dim"),
        (" | ", "dim"),
        (f"Agent: {assistant_id}{default_label}", "dim"),
    ]

    if settings.model_name:
        parts.extend([(" | ", "dim"), (f"Model: {settings.model_name}", "dim")])

    parts.append((" | ", "dim"))

    thread_url = build_langsmith_thread_url(thread_id) if include_thread_link else None
    if thread_url:
        parts.extend(
            [
                ("Thread: ", "dim"),
                (thread_id, Style(dim=True, link=thread_url)),
            ]
        )
    else:
        parts.append((f"Thread: {thread_id}", "dim"))

    return Text.assemble(*parts)


async def run_non_interactive(
    message: str,
    assistant_id: str = "agent",
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
    sandbox_setup: str | None = None,
    *,
    initial_skill: str | None = None,
    profile_override: dict[str, Any] | None = None,
    quiet: bool = False,
    stream: bool = True,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool = False,
    max_turns: int | None = None,
) -> int:
    """Run a single task non-interactively and exit.

    The agent is created with `interactive=False`, which tailors the system
    prompt for autonomous headless execution (no clarification questions,
    reasonable assumptions).

    Shell access and auto-approval are controlled by `--shell-allow-list`:

    - Not set → shell disabled, all other tools auto-approved.
    - `recommended` or explicit list → shell enabled, commands gated by
        allow-list; non-shell tools approved unconditionally.
    - `all` → shell enabled, any command allowed, all tools auto-approved.

    Note: startup header rendering avoids synchronous LangSmith URL lookups.
    A background thread resolves the thread URL concurrently and the result is
    displayed after task completion if available.

    Args:
        message: The task/message to execute.
        assistant_id: Agent identifier for memory storage.
        model_name: Optional model name to use.
        model_params: Extra kwargs from `--model-params` to pass to the model.

            These override config file values.
        sandbox_type: Type of sandbox (`'none'`, `'agentcore'`,
            `'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).
        sandbox_id: Optional existing sandbox ID to reuse.
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        initial_skill: Optional skill name whose `SKILL.md` instructions wrap
            the user message before sending it to the agent.
        profile_override: Extra profile fields from `--profile-override`.

            Merged on top of config file profile overrides.
        quiet: When `True`, all console output (headers, status messages,
            tool notifications, HITL decisions, errors) is redirected to
            stderr so that only the agent's response text appears on stdout.
        stream: When `True` (default), text chunks are written to stdout
            as they arrive.

            When `False`, the full response is buffered and written to stdout in
            one shot after the agent finishes.
        mcp_config_path: Optional path to MCP servers JSON configuration file.
            Merged on top of auto-discovered configs (highest precedence).
        no_mcp: Disable all MCP tool loading.
        trust_project_mcp: When `True`, allow project-level stdio MCP
            servers. When `False` (default), project stdio servers are
            silently skipped.
        max_turns: Optional cap on agentic turns.

            When set, overrides the internal safety default; exceeding it
            raises `HITLIterationLimitError`.

    Returns:
        Exit code: 0 for success, 1 for error, 130 for keyboard interrupt.
    """
    # stderr=True routes all console.print() to stderr; agent response text
    # uses _write_text() -> sys.stdout directly.
    console = Console(stderr=True) if quiet else Console()

    message_kwargs: dict[str, Any] | None = None
    if initial_skill and initial_skill.strip():
        from deepagents_cli.skills.invocation import (
            build_skill_invocation_envelope,
            discover_skills_and_roots,
        )
        from deepagents_cli.skills.load import load_skill_content

        normalized_skill = initial_skill.strip().lower()
        try:
            skills, allowed_roots = discover_skills_and_roots(assistant_id)
            skill = next((s for s in skills if s["name"] == normalized_skill), None)
        except OSError as e:
            console.print(
                "[bold red]Error:[/bold red] "
                f"Could not load skill: {escape_markup(normalized_skill)}. "
                f"Filesystem error: {escape_markup(str(e))}"
            )
            return 1
        except Exception as e:  # noqa: BLE001
            console.print(
                "[bold red]Error:[/bold red] "
                f"Error loading skill: {escape_markup(normalized_skill)}. "
                f"Unexpected error: {type(e).__name__}: {escape_markup(str(e))}"
            )
            return 1

        if skill is None:
            console.print(
                "[bold red]Error:[/bold red] "
                f"Skill not found: {escape_markup(normalized_skill)}"
            )
            return 1

        try:
            content = load_skill_content(
                str(skill["path"]),
                allowed_roots=allowed_roots,
            )
        except PermissionError as e:
            console.print(f"[bold red]Error:[/bold red] {escape_markup(str(e))}")
            return 1
        except OSError as e:
            console.print(
                "[bold red]Error:[/bold red] "
                f"Could not load skill: {escape_markup(normalized_skill)}. "
                f"Filesystem error: {escape_markup(str(e))}"
            )
            return 1
        except Exception as e:  # noqa: BLE001
            console.print(
                "[bold red]Error:[/bold red] "
                f"Error loading skill: {escape_markup(normalized_skill)}. "
                f"Unexpected error: {type(e).__name__}: {escape_markup(str(e))}"
            )
            return 1
        if content is None:
            console.print(
                "[bold red]Error:[/bold red] "
                f"Could not read content for skill: {escape_markup(normalized_skill)}. "
                "Check that the SKILL.md file exists, is readable, "
                "and is saved as UTF-8."
            )
            return 1

        if not content.strip():
            console.print(
                "[bold red]Error:[/bold red] "
                f"Skill '{escape_markup(normalized_skill)}' has an empty "
                "SKILL.md file. "
                "Add instructions to the file before invoking."
            )
            return 1

        envelope = build_skill_invocation_envelope(skill, content, message)
        message = envelope.prompt
        message_kwargs = envelope.message_kwargs

    try:
        result = create_model(
            model_name,
            extra_kwargs=model_params,
            profile_overrides=profile_override,
        )
    except ModelConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1

    result.apply_to_settings()
    thread_id = generate_thread_id()

    from deepagents_cli.config import build_stream_config

    config: RunnableConfig = build_stream_config(
        thread_id, assistant_id, sandbox_type=sandbox_type
    )

    thread_url_lookup: ThreadUrlLookupState | None = None
    if not quiet:
        thread_url_lookup = _start_langsmith_thread_url_lookup(thread_id)
        console.print(Text("Running task non-interactively...", style="dim"))
        header = _build_non_interactive_header(assistant_id, thread_id)
        console.print(header)

    import asyncio

    from deepagents_cli.server_manager import server_session

    # Launch MCP preload concurrently with server startup
    mcp_task: asyncio.Task[Any] | None = None
    if not no_mcp and not quiet:
        try:
            from deepagents_cli.main import _preload_session_mcp_server_info

            mcp_task = asyncio.create_task(
                _preload_session_mcp_server_info(
                    mcp_config_path=mcp_config_path,
                    no_mcp=no_mcp,
                    trust_project_mcp=trust_project_mcp,
                )
            )
        except Exception:
            logger.warning("MCP metadata preload task creation failed", exc_info=True)

    try:
        enable_shell = bool(settings.shell_allow_list)
        shell_is_unrestricted = isinstance(
            settings.shell_allow_list, type(SHELL_ALLOW_ALL)
        )
        # Currently, non-shell tools have no HITL handler in non-interactive
        # mode, so interrupting on them just fragments LangSmith traces
        # without adding value. Gate only shell execution via middleware.
        use_auto_approve = not enable_shell or shell_is_unrestricted
        use_interrupt_shell_only = enable_shell and not shell_is_unrestricted
        # Extract the concrete allow-list to forward to the server subprocess.
        # settings.shell_allow_list is already validated at this point.
        restrictive_allow_list: list[str] | None = (
            list(settings.shell_allow_list)
            if use_interrupt_shell_only and settings.shell_allow_list
            else None
        )

        if not quiet:
            console.print(Text("Starting LangGraph server...", style="dim"))

        async with server_session(
            assistant_id=assistant_id,
            model_name=model_name,
            model_params=model_params,
            auto_approve=use_auto_approve,
            interrupt_shell_only=use_interrupt_shell_only,
            shell_allow_list=restrictive_allow_list,
            sandbox_type=sandbox_type,
            sandbox_id=sandbox_id,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=False,
            mcp_config_path=mcp_config_path,
            no_mcp=no_mcp,
            trust_project_mcp=trust_project_mcp,
            interactive=False,
        ) as (agent, _server_proc):
            # Collect MCP preload result (ran concurrently with server startup)
            if mcp_task is not None:
                try:
                    mcp_info = await mcp_task
                    if mcp_info:
                        tool_count = sum(len(s.tools) for s in mcp_info)
                        if tool_count:
                            label = "MCP tool" if tool_count == 1 else "MCP tools"
                            console.print(
                                f"[green]✓ Loaded {tool_count} {label}[/green]"
                            )
                except Exception:
                    logger.warning("MCP metadata preload failed", exc_info=True)

            if not quiet:
                console.print("[green]✓ Server ready[/green]")

            file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=None)

            await _run_agent_loop(
                agent,
                message,
                config,
                console,
                file_op_tracker,
                quiet=quiet,
                stream=stream,
                message_kwargs=message_kwargs,
                thread_url_lookup=thread_url_lookup,
                max_turns=max_turns,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except HITLIterationLimitError as e:
        console.print(f"\n[red]{escape_markup(str(e))}[/red]")
        console.print(
            "[yellow]Hint: The agent may be repeatedly attempting commands "
            "that are not in the allow-list. Consider expanding the "
            "--shell-allow-list or adjusting the task.[/yellow]"
        )
        return 1
    except (ValueError, OSError) as e:
        logger.exception("Error during non-interactive execution")
        console.print(f"\n[red]Error: {escape_markup(str(e))}[/red]")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during non-interactive execution")
        console.print(
            f"\n[red]Unexpected error ({type(e).__name__}): "
            f"{escape_markup(str(e))}[/red]"
        )
        return 1
    else:
        return 0
