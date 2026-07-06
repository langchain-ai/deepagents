"""Non-interactive execution mode.

Provides `run_non_interactive` which runs a single user task against the
agent graph, streams results to stdout, and exits with an appropriate code.

The agent runs inside a `langgraph dev` server subprocess, connected via
the `RemoteAgent` client (see `server_manager.server_session`).

Shell commands are gated by an optional allow-list (`--shell-allow-list`):

- Not set → shell disabled, all other tool calls auto-approved.
- `recommended` or explicit list → shell enabled, commands validated
    against the list; non-shell tools approved unconditionally.
- `all` → shell enabled, any command allowed, all tools auto-approved.

An optional quiet mode (`--quiet` / `-q`) suppresses stream-time diagnostics
(the tool-call and file-operation notifications) so stdout carries only the
agent's response text.
"""

from __future__ import annotations

import asyncio
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

from deepagents_code._cli_context import CLIContext
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
from deepagents_code._version import __version__
from deepagents_code.agent import DEFAULT_AGENT_NAME
from deepagents_code.config import (
    SHELL_ALLOW_ALL,
    build_langsmith_thread_url,
    create_model,
    is_shell_command_allowed,
    settings,
)
from deepagents_code.file_ops import FileOpTracker
from deepagents_code.hooks import (
    dispatch_hook,
    dispatch_hook_fire_and_forget,
    drain_pending_hooks,
)
from deepagents_code.model_config import ModelConfigError
from deepagents_code.sessions import generate_thread_id
from deepagents_code.textual_adapter import SessionStats, print_usage_table
from deepagents_code.tool_display import format_tool_message_content
from deepagents_code.unicode_security import (
    check_url_safety,
    detect_dangerous_unicode,
    format_warning_detail,
    iter_string_values,
    looks_like_url_key,
    summarize_issues,
)

if TYPE_CHECKING:
    from asyncio.subprocess import Process

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


def _make_stdio_encoding_safe() -> None:
    """Prevent `UnicodeEncodeError` from killing a non-interactive run.

    Legacy Windows consoles default to a locale code page (e.g. cp1252) that
    cannot encode glyphs like "✓"; the first `console.print()` containing one
    then crashes the whole run. Reconfiguring the streams with
    `errors="replace"` degrades unencodable characters to "?" instead. The
    stream encoding itself is left untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        # Streams replaced by non-reconfigurable objects (e.g. a plain
        # StringIO or a captured buffer) are left as-is.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError, TypeError):
            # Closed, detached, or otherwise non-reconfigurable stream (a
            # duck-typed `reconfigure` with a different signature raises
            # `TypeError`) — leave it as-is. This is a best-effort hardening
            # step; it must never itself crash the run.
            logger.debug(
                "Could not reconfigure %s error handler",
                getattr(stream, "name", stream),
                exc_info=True,
            )
            continue


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


async def _terminate_startup_process(proc: Process) -> None:
    """Terminate and reap a startup command subprocess.

    Args:
        proc: Process returned by `asyncio.create_subprocess_shell`.
    """
    import sys

    if proc.returncode is not None:
        return

    try:
        if sys.platform != "win32":
            import os
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    except OSError:
        logger.warning(
            "Failed to terminate startup command (pid=%s)",
            proc.pid,
            exc_info=True,
        )
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        logger.warning(
            "Startup command (pid=%s) did not exit after termination; sending SIGKILL",
            proc.pid,
        )
        try:
            if sys.platform != "win32":
                import os
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            return
        except OSError:
            logger.warning(
                "Failed to SIGKILL startup command (pid=%s); process may leak",
                proc.pid,
                exc_info=True,
            )
            return
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        except OSError:
            logger.warning(
                "Failed to reap startup command (pid=%s) after SIGKILL",
                proc.pid,
                exc_info=True,
            )
    except ProcessLookupError:
        pass
    except OSError:
        logger.warning(
            "Failed to wait on startup command (pid=%s) after SIGTERM; "
            "process may leak",
            proc.pid,
            exc_info=True,
        )


@dataclass(frozen=True)
class InFlightToolCall:
    """A tool call whose `tool.use` has fired but whose result has not arrived.

    Bundling the name and args into one record keeps them structurally in
    lock-step: a single `dict[str, InFlightToolCall]` cannot represent an id with
    args but no name (or vice versa), which two parallel dicts could. That
    removes the desync failure mode where an orphaned drain would emit a
    `tool.error` with an empty `tool_name`.
    """

    name: str
    """The tool name, carried so an orphaned call can be closed with a name."""

    args: dict[str, Any]
    """The parsed tool-call arguments, for correlating the matching result."""


@dataclass
class StreamState:
    """Mutable state accumulated while iterating over the agent stream."""

    quiet: bool = False
    """When `True`, stream-time diagnostics (the tool-call and file-operation
    notifications, plus the stdout separator newline preceding a tool call) are
    suppressed, so stdout carries only agent response text."""

    stream: bool = True
    """When `True` (default), text chunks are written to stdout as they arrive.

    When `False`, text is buffered in `full_response` and flushed after the
    agent finishes.
    """

    full_response: list[str] = field(default_factory=list)
    """Accumulated text fragments from the AI message stream."""

    tool_call_buffers: dict[ToolCallBufferKey, ToolCallBuffer] = field(
        default_factory=dict
    )
    """Maps a tool-call index or ID to its in-progress buffer: name, ID,
    accumulated argument fragments, and the display latch."""

    in_flight_tool_calls: dict[str, InFlightToolCall] = field(default_factory=dict)
    """Maps in-flight tool-call IDs (those whose `tool.use` has fired) to their
    name and parsed arguments, so the matching `tool.result` can be correlated.
    Entries are removed when the result arrives; any still present when the
    stream aborts are closed by `_dispatch_orphaned_tool_result_hooks`. One
    record per id keeps name and args structurally in lock-step (see
    `InFlightToolCall`)."""

    displayed_tool_call_ids: set[str] = field(default_factory=set)
    """Tool-call IDs whose non-interactive call line has already been printed."""

    emitted_tool_use_ids: set[str] = field(default_factory=set)
    """Tool-call IDs for which a `tool.use` has been dispatched.

    Monotonic: never cleared within a run, so `tool.use` fires at most once per
    id even if a call's arg chunks are redelivered after its result (mirrors the
    TUI's `displayed_tool_ids`). Result correlation and the orphan drain use the
    separate `in_flight_tool_calls`, which *is* cleared per result."""

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

    show_rubric_iterations: bool = False
    """Whether rubric lifecycle messages should include iteration numbers."""


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
        active_provider = settings.model_provider or ""
        if input_toks or output_toks:
            state.stats.record_request(
                active_model, input_toks, output_toks, active_provider
            )
        elif total_toks:
            state.stats.record_request(active_model, total_toks, 0, active_provider)

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
            chunk_args = block.get("args")

            buffer_key = tool_call_buffer_key(
                chunk_index, chunk_id, len(state.tool_call_buffers)
            )
            buffer = state.tool_call_buffers.setdefault(buffer_key, ToolCallBuffer())
            buffer.ingest(name=chunk_name, tool_id=chunk_id, args=chunk_args)

            buffer_name = buffer.name
            buffer_id = buffer.tool_id
            already_displayed = (
                isinstance(buffer_id, str)
                and buffer_id in state.displayed_tool_call_ids
            )
            if (
                isinstance(buffer_name, str)
                and not buffer.displayed
                and not already_displayed
            ):
                if state.spinner:
                    state.spinner.stop()
                if not state.quiet:
                    if state.full_response:
                        _write_newline()
                    console.print(
                        f"[dim]🔧 Calling tool: {escape_markup(buffer_name)}[/dim]",
                        highlight=False,
                    )
                buffer.displayed = True
                if isinstance(buffer_id, str):
                    state.displayed_tool_call_ids.add(buffer_id)
            elif isinstance(buffer_id, str) and buffer.displayed:
                state.displayed_tool_call_ids.add(buffer_id)

            # Gate tool.use on a resolved tool id so this surface matches the
            # interactive one, which dispatches at widget-mount time in
            # `textual_adapter.execute_task_textual`. Both gate on a resolved
            # tool-call id and fire at most once per id via a monotonic id set
            # (`emitted_tool_use_ids` here, `displayed_tool_ids` there) that is
            # never cleared within the run — see the "fire-once-per-id" clause of
            # the parity contract in `_tool_stream`. Gating on the monotonic set
            # rather than `in_flight_tool_calls` (which is cleared per result)
            # means a redelivery of a completed call's arg chunks does not
            # re-fire `tool.use` and spawn a spurious orphan.
            parsed_args = buffer.parse_args()
            if (
                isinstance(buffer_name, str)
                and buffer_id is not None
                and parsed_args is not None
                and buffer_id not in state.emitted_tool_use_ids
            ):
                dispatch_hook_fire_and_forget(
                    "tool.use",
                    build_tool_use_payload(buffer_name, buffer_id, parsed_args),
                )
                state.emitted_tool_use_ids.add(buffer_id)
                state.in_flight_tool_calls[buffer_id] = InFlightToolCall(
                    buffer_name, parsed_args
                )
            if (
                isinstance(buffer_id, str)
                and parsed_args is not None
                and buffer_id in state.emitted_tool_use_ids
            ):
                # Drop the buffer so a later turn that reuses this streaming
                # index (indices restart per message, per LangChain streaming
                # semantics) starts fresh rather than reusing this call's state.
                # This also clears a redelivered completed call's recreated
                # buffer, whose `tool.use` is already suppressed by the
                # monotonic `emitted_tool_use_ids`.
                state.tool_call_buffers.pop(buffer_key, None)


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
        tool_id = getattr(message_obj, "tool_call_id", None)
        correlated_tool_name = ""
        # Args come from the matching tool.use. They default to {} when the call
        # had no id to correlate on, or no tool.use fired (e.g. its args never
        # parsed). The seam is intentional — without a correlated id we cannot
        # pair them — but log the correlation miss so a lost pairing is not
        # completely silent.
        if isinstance(tool_id, str):
            in_flight = state.in_flight_tool_calls.pop(tool_id, None)
            tool_args = in_flight.args if in_flight is not None else None
            correlated_tool_name = in_flight.name if in_flight is not None else ""
            state.displayed_tool_call_ids.discard(tool_id)
            if tool_args is None:
                # Warning, not info/debug: a real-id result with no matching
                # tool.use means a hook consumer sees a `tool.result` with empty
                # args for a tool that actually executed (its args never parsed) —
                # degraded audit fidelity worth surfacing at default log levels,
                # consistent with the rest of this feature's severity philosophy.
                logger.warning(
                    "tool.result for %s has no correlated tool.use args; "
                    "sending empty tool_args",
                    tool_id,
                )
                tool_args = {}
        else:
            tool_args = {}
        record = file_op_tracker.complete_with_message(message_obj)
        if record and record.diff:
            if state.spinner:
                state.spinner.stop()
            if not state.quiet:
                console.print(
                    f"[dim]📝 {escape_markup(record.display_path)}[/dim]",
                    highlight=False,
                )
        tool_name = getattr(message_obj, "name", "")
        if not tool_name:
            tool_name = correlated_tool_name
        # Normalize to the two-value hook domain, fail-closed: an unexpected
        # provider status is logged and treated as an error (see
        # `normalize_tool_status`) rather than silently reported as success.
        tool_status: ToolStatus = normalize_tool_status(
            getattr(message_obj, "status", "success"), tool_name
        )
        # Format the content the same way the interactive surface does so
        # `tool_output` is identical across surfaces for list/structured content
        # (e.g. multimodal or MCP tools returning content blocks) rather than a
        # raw Python list repr here vs. extracted text there. Truncation to
        # HOOK_TOOL_OUTPUT_LIMIT happens inside build_tool_result_payload.
        # Guard formatting so a formatter error can't skip the tool.result
        # dispatch below; on failure use a sentinel (see the except) rather than
        # re-touching the offending content, so the dispatch stays unconditional.
        try:
            tool_content = format_tool_message_content(message_obj.content)
            tool_output = str(tool_content) if tool_content else ""
        except Exception:
            # Guard formatting *and* the str() coercion together: a pathological
            # __str__ must not re-raise past the fallback and skip the
            # tool.result dispatch below. Use a sentinel rather than re-touching
            # the offending content, so the dispatch stays unconditional.
            logger.exception("Failed to format tool output")
            tool_output = UNRENDERABLE_TOOL_OUTPUT
        # Headless always dispatches tool.result for every ToolMessage — there
        # are no widgets to skip. The TUI handles ToolMessages in three branches
        # in `textual_adapter.execute_task_textual`: the widget-backed path and
        # an `else` for unmounted tools both dispatch (mirroring this
        # always-dispatch behavior), while the `completed_tool_result_ids` branch
        # suppresses a duplicate rather than dispatching. See the parity contract
        # in `_tool_stream` for the full guarantee.
        if tool_status == "error":
            dispatch_hook_fire_and_forget(
                "tool.error",
                build_tool_error_payload(tool_name),
            )
        dispatch_hook_fire_and_forget(
            "tool.result",
            build_tool_result_payload(
                tool_name, tool_id, tool_args, tool_status, tool_output
            ),
        )
        if state.spinner:
            state.spinner.start()


def _process_rubric_event(
    data: dict[str, Any],
    state: StreamState,
    console: Console,
) -> None:
    """Render a `RubricMiddleware` lifecycle event from the custom stream.

    `RubricMiddleware` emits `rubric_evaluation_start` / `rubric_evaluation_end`
    dicts via `runtime.stream_writer`. Non-rubric custom payloads are ignored.

    Args:
        data: The custom-stream payload dict.
        state: Shared stream state (used to pause the spinner).
        console: Rich console for status output (stderr in `--quiet` mode).
    """
    event_type = data.get("type")
    if event_type not in {"rubric_evaluation_start", "rubric_evaluation_end"}:
        return

    if state.spinner:
        state.spinner.stop()

    if event_type == "rubric_evaluation_start":
        # `iteration` is untrusted streamed payload; only render the 1-based
        # number when it is actually an int and the user explicitly requested an
        # iteration cap. A non-int previously raised `TypeError` here and aborted
        # the whole non-interactive run.
        iteration = data.get("iteration", 0)
        label = (
            f" (iteration {iteration + 1})"
            if state.show_rubric_iterations and isinstance(iteration, int)
            else ""
        )
        console.print(
            f"[dim]⏳ Checking acceptance criteria{label}…[/dim]",
            highlight=False,
        )
        if state.spinner:
            state.spinner.start()
        return

    result = data.get("result")
    explanation = (data.get("explanation") or "").strip()
    if result == "satisfied":
        console.print("[green]✓ Acceptance criteria satisfied[/green]", highlight=False)
    elif result == "needs_revision":
        suffix = f": {escape_markup(explanation)}" if explanation else ""
        console.print(
            f"[yellow]↻ Changes need revision{suffix}[/yellow]", highlight=False
        )
        for criterion in data.get("criteria", []):
            if isinstance(criterion, dict) and not criterion.get("passed", True):
                name = escape_markup(str(criterion.get("name", "criterion")))
                gap = escape_markup(str(criterion.get("gap", "")).strip())
                detail = f" — {gap}" if gap else ""
                console.print(f"[yellow]  ✗ {name}{detail}[/yellow]", highlight=False)
    elif result == "max_iterations_reached":
        console.print(
            "[yellow]⚠ Acceptance criteria not satisfied "
            "(iteration limit reached)[/yellow]",
            highlight=False,
        )
    elif result in {"failed", "grader_error"}:
        label = "grader failed" if result == "failed" else "grader error"
        suffix = f": {escape_markup(explanation)}" if explanation else ""
        console.print(f"[red]⚠ Rubric {label}{suffix}[/red]", highlight=False)
    elif result is not None:
        # A `rubric_evaluation_end` with an unrecognized result is still a
        # terminal grading event; surface it rather than letting the run go
        # quiet mid-task (e.g. if the SDK adds a new verdict). Mirrors the
        # interactive fallback in `textual_adapter._format_rubric_event`.
        suffix = f": {escape_markup(explanation)}" if explanation else ""
        console.print(
            f"[yellow]⚠ Rubric grading ended{suffix}[/yellow]", highlight=False
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
    elif stream_mode == "custom" and isinstance(data, dict):
        _process_rubric_event(cast("dict[str, Any]", data), state, console)
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

    if action_name == "execute":
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
    context: CLIContext,
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
        context: Runtime context for model-call middleware.
    """
    if state.spinner:
        state.spinner.start()
    try:
        async for chunk in agent.astream(
            stream_input,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
            config=config,
            context=context,
            durability="exit",
        ):
            _process_stream_chunk(chunk, state, console, file_op_tracker)
    finally:
        if state.spinner:
            state.spinner.stop()


def _dispatch_orphaned_tool_result_hooks(state: StreamState, tool_output: str) -> None:
    """Close out `tool.use` events that never received a `ToolMessage`.

    On a normally-completing run every `tool.use` is followed by a `ToolMessage`
    that drains `in_flight_tool_calls`, so this is a no-op. When the stream is
    aborted mid-flight (e.g. a provider error between the tool call and its
    result), any id still present had its `tool.use` dispatched with no terminal
    event; emit `tool.error` + a `tool_status="error"` `tool.result` for each so
    the headless surface upholds the same "every `tool.use` is closed" guarantee
    as the TUI's `_dispatch_terminal_tool_result_hooks`.

    Args:
        state: The stream state whose in-flight tool maps are drained.
        tool_output: Terminal output recorded on each synthesized `tool.result`.
    """
    if state.in_flight_tool_calls:
        # A non-empty in-flight map here means real tool results were lost to a
        # mid-stream abort (a clean run drains every id via its result). Surface
        # it at warning — matching the TUI's backstop for the same class — so an
        # operator can tell a clean run from one that synthesized error closes,
        # rather than the drain being silent (degraded audit fidelity).
        logger.warning(
            "Stream ended with %d in-flight tool call(s) that never received a "
            "result; closing each with a synthetic tool.error/tool.result",
            len(state.in_flight_tool_calls),
        )
    for tool_id, in_flight in list(state.in_flight_tool_calls.items()):
        dispatch_hook_fire_and_forget(
            "tool.error", build_tool_error_payload(in_flight.name)
        )
        dispatch_hook_fire_and_forget(
            "tool.result",
            build_tool_result_payload(
                in_flight.name, tool_id, in_flight.args, "error", tool_output
            ),
        )
    state.in_flight_tool_calls.clear()


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
    rubric: str | None = None,
    show_rubric_iterations: bool = False,
) -> None:
    """Run the agent and handle HITL interrupts until the task completes.

    The loop is capped at `max_turns` when set,
    otherwise `_MAX_HITL_ITERATIONS`, to prevent runaway retries
    (e.g. the agent repeatedly attempting rejected commands).

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
        max_turns: Optional cap on total agentic turns (initial response plus
            HITL resumes).

            When `None`, falls back to `_MAX_HITL_ITERATIONS`.
        rubric: Acceptance criteria supplied to `RubricMiddleware` via the
            graph's `rubric` state field.

            `None` leaves it unset (no grading).
        show_rubric_iterations: Whether rubric lifecycle messages should include
            iteration numbers.

    Raises:
        HITLIterationLimitError: If the effective turn limit is exceeded.
    """
    spinner = None if quiet else _ConsoleSpinner(console)
    state = StreamState(
        quiet=quiet,
        stream=stream,
        spinner=spinner,
        show_rubric_iterations=show_rubric_iterations,
    )
    user_msg: dict[str, Any] = {"role": "user", "content": message}
    if message_kwargs:
        user_msg.update(message_kwargs)
    stream_input: dict[str, Any] | Command = {"messages": [user_msg]}
    if rubric is not None:
        stream_input["rubric"] = rubric

    thread_id = config.get("configurable", {}).get("thread_id", "")
    # An empty or missing thread ID carries no session identity, so leave it
    # unset in context rather than passing a blank string to model middleware.
    context_thread_id = thread_id if isinstance(thread_id, str) and thread_id else None
    context = CLIContext(thread_id=context_thread_id)
    await dispatch_hook("session.start", {"thread_id": thread_id})

    start_time = time.monotonic()

    try:
        # Initial stream
        await _stream_agent(
            agent, stream_input, config, state, console, file_op_tracker, context
        )

        # The internal default applies when --max-turns is omitted, guarding
        # against unbounded runaway loops in scripts that forgot to set one.
        effective_limit = max_turns if max_turns is not None else _MAX_HITL_ITERATIONS

        # The initial stream above counts as turn 1; each HITL resume is a
        # further turn. Raise before starting a resume that would exceed the
        # budget so the user-facing count matches the flag's semantics.
        turns = 1
        while state.interrupt_occurred:
            if turns >= effective_limit:
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
            turns += 1
            state.interrupt_occurred = False
            state.hitl_response.clear()
            _process_hitl_interrupts(state, console)
            stream_input = Command(resume=state.hitl_response)
            await _stream_agent(
                agent, stream_input, config, state, console, file_op_tracker, context
            )
    finally:
        # Close out any `tool.use` with no matching `ToolMessage` — e.g. a stream
        # aborted by a provider error mid-tool. On a clean run every id was
        # already drained by its result, so this is a no-op. Guarded so a
        # dispatch problem can never mask the exception propagating from the
        # stream (this runs on the error path too).
        try:
            _dispatch_orphaned_tool_result_hooks(
                state, "Stream ended before tool result"
            )
        except Exception:
            logger.warning(
                "Orphaned tool.result drain failed unexpectedly", exc_info=True
            )
        # Surface any buffered tool call whose args never parsed: it never
        # entered `in_flight_tool_calls` (so the orphan drain above skips it) and
        # would otherwise be dropped with `state` at scope exit with no trace.
        # Info, not warning — some of these may still have executed (their
        # `tool.result` fired with `{}` args and logged a correlation miss); this
        # only asserts the args never parsed. Guarded so a logging failure can
        # never mask an exception propagating from the stream.
        try:
            # Two distinct reasons a buffered call never fired tool.use — args
            # that never parsed, and args that parsed but carried no id (so
            # tool.use was gated out). The classification is shared with the TUI
            # via `count_unemitted_tool_calls`; each surface logs its own lines.
            unemitted = count_unemitted_tool_calls(state.tool_call_buffers.values())
            if unemitted.unparsed:
                logger.info(
                    "Stream ended with %d tool call(s) whose arguments never "
                    "parsed; no tool.use was emitted for them",
                    unemitted.unparsed,
                )
            if unemitted.idless_parsed:
                logger.info(
                    "Stream ended with %d tool call(s) whose arguments parsed but "
                    "carried no tool-call id; no tool.use was emitted for them",
                    unemitted.idless_parsed,
                )
        except Exception:
            logger.warning(
                "Unparsed tool-call buffer check failed unexpectedly",
                exc_info=True,
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
    rubric_active: bool = False,
) -> Text:
    """Build the non-interactive mode header with model, agent, and thread info.

    By default, this function avoids LangSmith network lookups and renders the
    thread ID as plain text. Callers can opt in to hyperlink resolution.

    Args:
        assistant_id: Agent identifier.
        thread_id: Thread identifier.
        include_thread_link: Whether to resolve and render a LangSmith link for
            the thread ID.
        rubric_active: Whether a rubric is active for this run; when `True`,
            appends a `Rubric: active` marker so the behavior change is visible.

    Returns:
        Rich Text object with the formatted header line.
    """
    default_label = " (default)" if assistant_id == DEFAULT_AGENT_NAME else ""
    parts: list[tuple[str, str | Style]] = [
        (f"App: v{__version__}", "dim"),
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

    if rubric_active:
        parts.extend([(" | ", "dim"), ("Rubric: active", "dim")])

    return Text.assemble(*parts)


async def _run_startup_command(
    command: str,
    console: Console,
    *,
    quiet: bool,
) -> None:
    """Run the `--startup-cmd` shell command before the agent loop.

    Stdout and stderr are routed through `console`. In `--quiet` mode the
    caller wires `console` to stderr so agent output on stdout stays clean;
    otherwise both streams land on stdout alongside the agent's response.
    Non-zero exits and timeouts emit a yellow warning but do not abort the
    session — the non-zero exit code is logged, not propagated.

    Args:
        command: Shell command to execute (subject to shell expansion).
        console: Rich console for status messages. Respects `quiet`.
        quiet: When `True`, suppresses the "Running startup command" header
            so piped output stays minimal; warnings still appear (on stderr
            when the caller wired the console there).

    Raises:
        asyncio.CancelledError: If the caller cancels while the startup command
            is running.
    """
    import sys

    if not quiet:
        console.print(Text(f"Running startup command: {command}", style="dim"))

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(sys.platform != "win32"),
        )
    except OSError as e:
        console.print(
            "[yellow]Warning:[/yellow] startup command failed to launch: "
            f"{escape_markup(str(e))}"
        )
        return

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=60
        )
    except asyncio.CancelledError:
        await _terminate_startup_process(proc)
        raise
    except TimeoutError:
        await _terminate_startup_process(proc)
        console.print("[yellow]Warning:[/yellow] startup command timed out (60s limit)")
        return

    stdout_text = (stdout_bytes or b"").decode(errors="replace").rstrip("\n")
    stderr_text = (stderr_bytes or b"").decode(errors="replace").rstrip("\n")
    if stdout_text:
        # Wrap in `Text` so Rich treats the shell output as literal — otherwise
        # brackets in tool output (e.g. `[INFO]`, `[1/3]`) would be parsed as
        # markup and either silently stripped or raise `MarkupError`.
        console.print(Text(stdout_text), highlight=False)
    if stderr_text:
        console.print(Text(stderr_text, style="dim"), highlight=False)

    if proc.returncode:
        console.print(
            "[yellow]Warning:[/yellow] startup command exited with code "
            f"{proc.returncode} — continuing anyway"
        )


async def run_non_interactive(
    message: str,
    assistant_id: str = DEFAULT_AGENT_NAME,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
    sandbox_snapshot_name: str | None = None,
    sandbox_setup: str | None = None,
    *,
    initial_skill: str | None = None,
    startup_cmd: str | None = None,
    profile_override: dict[str, Any] | None = None,
    quiet: bool = False,
    stream: bool = True,
    mcp_config_path: str | None = None,
    no_mcp: bool = False,
    trust_project_mcp: bool = False,
    enable_interpreter: bool | None = None,
    interpreter_ptc: str | list[str] | None = None,
    interpreter_ptc_acknowledge_unsafe: bool = False,
    max_turns: int | None = None,
    rubric: str | None = None,
    rubric_model: str | None = None,
    rubric_max_iterations: int | None = None,
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
        sandbox_snapshot_name: Snapshot (langsmith) or blueprint (runloop) name.
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        initial_skill: Optional skill name whose `SKILL.md` instructions wrap
            the user message before sending it to the agent.
        startup_cmd: Shell command to run at startup, before the agent runs.

            Output follows the same console routing as other app messages:
            stdout by default, stderr when `-q` is set. Non-zero exits and
            timeouts warn but do not abort the task.
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
        enable_interpreter: Enable the JS interpreter (`js_eval`) middleware
            on the main agent. `None` uses the sandbox-aware default.
        interpreter_ptc: Override for `settings.interpreter_ptc` (PTC
            allowlist for `js_eval`).
        interpreter_ptc_acknowledge_unsafe: Explicit acknowledgement for
            `interpreter_ptc="all"` outside of `auto_approve`.
        max_turns: Optional cap on total agentic turns. When `None`, the
            internal safety default applies.
        rubric: Acceptance criteria for `RubricMiddleware`. When provided, the
            agent self-evaluates against it and loops until satisfied.

            `None` disables rubric grading.
        rubric_model: Grader model spec; `None` reuses the main model.
        rubric_max_iterations: Grader iterations per rubric attempt; `None`
            uses the middleware default.

    Returns:
        Exit code: 0 for success, 1 for error, 124 when the `--max-turns`
            budget was exceeded (matching GNU `timeout`), 130 for keyboard
            interrupt.
    """
    _make_stdio_encoding_safe()

    # stderr=True routes all console.print() to stderr; agent response text
    # uses _write_text() -> sys.stdout directly.
    console = Console(stderr=True) if quiet else Console()

    if startup_cmd and startup_cmd.strip():
        await _run_startup_command(startup_cmd.strip(), console, quiet=quiet)

    message_kwargs: dict[str, Any] | None = None
    if initial_skill and initial_skill.strip():
        from deepagents_code.skills.invocation import (
            build_skill_invocation_envelope,
            discover_skills_and_roots,
        )
        from deepagents_code.skills.load import load_skill_content

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

    # One user turn per process: fresh turn id, turn_number 1.
    from uuid import uuid4

    from deepagents_code.config import build_stream_config

    config: RunnableConfig = build_stream_config(
        thread_id,
        assistant_id,
        sandbox_type=sandbox_type,
        turn_id=str(uuid4()),
        turn_number=1,
    )

    thread_url_lookup: ThreadUrlLookupState | None = None
    if not quiet:
        thread_url_lookup = _start_langsmith_thread_url_lookup(thread_id)
        console.print(Text("Running task non-interactively...", style="dim"))
        header = _build_non_interactive_header(
            assistant_id,
            thread_id,
            rubric_active=rubric is not None,
        )
        console.print(header)

    from deepagents_code.server_manager import server_session

    # Launch MCP preload concurrently with server startup
    mcp_task: asyncio.Task[Any] | None = None
    if not no_mcp and not quiet:
        try:
            from deepagents_code.main import _preload_session_mcp_server_info

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
            sandbox_snapshot_name=sandbox_snapshot_name,
            sandbox_setup=sandbox_setup,
            enable_shell=enable_shell,
            enable_ask_user=False,
            enable_interpreter=enable_interpreter,
            interpreter_ptc=interpreter_ptc,
            interpreter_ptc_acknowledge_unsafe=interpreter_ptc_acknowledge_unsafe,
            rubric_model=rubric_model,
            rubric_max_iterations=rubric_max_iterations,
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
                rubric=rubric,
                show_rubric_iterations=rubric_max_iterations is not None,
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
        # Dedicated exit code (matches GNU `timeout`) so CI can distinguish
        # a turn-budget hit from a generic failure that also returns 1.
        return 124
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
    finally:
        # Fire-and-forget hooks (tool.use/tool.result) run as background tasks;
        # await them here so the final tool.result is not cancelled when
        # asyncio.run tears the loop down. Never return from this block — that
        # would swallow the exit code determined above. drain_pending_hooks is
        # documented never to raise, but guard it anyway (mirroring app.py's
        # shutdown drain) so a future contract break can't replace the exit code
        # with an exception escaping the finally.
        try:
            await drain_pending_hooks()
        except Exception:
            logger.warning("Hook drain raised unexpectedly before exit", exc_info=True)
