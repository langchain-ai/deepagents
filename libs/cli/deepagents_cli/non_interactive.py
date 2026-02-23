"""Non-interactive execution mode for deepagents CLI.

Provides `run_non_interactive` which runs a single user task against the
agent graph, streams results to stdout, and exits with an appropriate code.

Shell commands are gated by an optional allow-list. When no allow-list is
set, shell is disabled and all other tool calls are auto-approved via the
`auto_approve` flag. When an allow-list is provided, shell is enabled and
all tool calls (shell and non-shell) pass through HITL, where non-shell
tools are approved unconditionally and shell commands are validated against
the list.

An optional quiet mode (`--quiet` / `-q`) redirects all console output to
stderr, leaving stdout exclusively for the agent's response text.

Note: in non-interactive mode (`-n`), auto-approval is determined solely by
whether a `--shell-allow-list` is present, not by the `--auto-approve` CLI
flag. See `run_non_interactive` for details.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.human_in_the_loop import ActionRequest, HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError
from rich.console import Console
from rich.style import Style
from rich.text import Text

from deepagents_cli.agent import DEFAULT_AGENT_NAME, create_cli_agent
from deepagents_cli.config import (
    SHELL_TOOL_NAMES,
    build_langsmith_thread_url,
    create_model,
    is_shell_command_allowed,
    settings,
)
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.model_config import ModelConfigError
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

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


@dataclass
class StreamState:
    """Mutable state accumulated while iterating over the agent stream.

    Attributes:
        quiet: When `True`, diagnostic formatting that would otherwise go
            to stdout (e.g. separator newlines before tool notifications)
            is suppressed so that stdout contains only agent response text.
        stream: When `True` (default), text chunks are written to stdout
            as they arrive. When `False`, text is buffered in `full_response`
            and flushed after the agent finishes.
        full_response: Accumulated text fragments from the AI message stream.
        tool_call_buffers: Maps a tool-call index or ID to its name/ID
            metadata for in-progress tool calls.
        pending_interrupts: Maps interrupt IDs to their validated HITL
            requests that are awaiting decisions.
        hitl_response: Maps interrupt IDs to dicts containing a `'decisions'`
            key with a list of decision dicts (each having a `'type'` key of
            `'approve'` or `'reject'`).

            Used to resume the agent after HITL processing.
        interrupt_occurred: Flag indicating whether any HITL interrupt was
            received during the current stream pass.
    """

    quiet: bool = False
    stream: bool = True
    full_response: list[str] = field(default_factory=list)
    tool_call_buffers: dict[int | str, dict[str, str | None]] = field(
        default_factory=dict
    )
    pending_interrupts: dict[str, HITLRequest] = field(default_factory=dict)
    hitl_response: dict[str, dict[str, list[dict[str, str]]]] = field(
        default_factory=dict
    )
    interrupt_occurred: bool = False


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
                if state.full_response and not state.quiet:
                    _write_newline()
                console.print(f"[dim]🔧 Calling tool: {chunk_name}[/dim]")


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
            console.print(f"[dim]📝 {record.display_path}[/dim]")


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

    Shell tools are always gated: if an allow-list is configured, the command
    is validated against it; if no allow-list is configured, shell commands
    are rejected outright (defense-in-depth -- the caller should disable
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
            console.print(f"[dim]✓ Auto-approved: {command}[/dim]")
            return {"type": "approve"}

        allowed_list_str = ", ".join(settings.shell_allow_list)
        console.print(f"\n[red]Shell command rejected:[/red] {command}")
        console.print(f"[yellow]Allowed commands:[/yellow] {allowed_list_str}")
        return {
            "type": "reject",
            "message": (
                f"Command '{command}' is not in the allow-list. "
                f"Allowed commands: {allowed_list_str}. "
                f"Please use allowed commands or try another approach."
            ),
        }

    console.print(f"[dim]✓ Auto-approved action: {action_name}[/dim]")
    return {"type": "approve"}


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
    agent: Pregel,
    stream_input: dict[str, Any] | Command,
    config: RunnableConfig,
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """Consume the full agent stream and update *state* with results.

    Args:
        agent: The compiled LangGraph agent.
        stream_input: Either the initial user message dict or a
            `Command(resume=...)` for HITL continuation.
        config: LangGraph runnable config (thread ID, metadata, etc.).
        state: Shared stream state.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
    """
    async for chunk in agent.astream(
        stream_input,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        _process_stream_chunk(chunk, state, console, file_op_tracker)


async def _run_agent_loop(
    agent: Pregel,
    message: str,
    config: RunnableConfig,
    console: Console,
    file_op_tracker: FileOpTracker,
    *,
    quiet: bool = False,
    stream: bool = True,
) -> None:
    """Run the agent and handle HITL interrupts until the task completes.

    The loop processes at most `_MAX_HITL_ITERATIONS` rounds to prevent
    runaway retries (e.g. the agent repeatedly attempting rejected commands).

    Args:
        agent: The compiled LangGraph agent.
        message: The user's task message.
        config: LangGraph runnable config.
        console: Rich console for formatted output.
        file_op_tracker: Tracker for file-operation diffs.
        quiet: Suppress diagnostic formatting on stdout.
        stream: When `True`, text is written to stdout as it arrives.

            When `False`, the full response is buffered and flushed at
            the end.

    Raises:
        HITLIterationLimitError: If the HITL iteration limit is exceeded.
    """
    state = StreamState(quiet=quiet, stream=stream)
    stream_input: dict[str, Any] | Command = {
        "messages": [{"role": "user", "content": message}]
    }

    # Initial stream
    await _stream_agent(agent, stream_input, config, state, console, file_op_tracker)

    # Handle HITL interrupts
    iterations = 0
    while state.interrupt_occurred:
        iterations += 1
        if iterations > _MAX_HITL_ITERATIONS:
            msg = (
                f"Exceeded {_MAX_HITL_ITERATIONS} HITL interrupt rounds. "
                "The agent may be stuck retrying rejected commands."
            )
            raise HITLIterationLimitError(msg)
        state.interrupt_occurred = False
        state.hitl_response.clear()
        _process_hitl_interrupts(state, console)
        stream_input = Command(resume=state.hitl_response)
        await _stream_agent(
            agent, stream_input, config, state, console, file_op_tracker
        )

    if state.full_response:
        if not state.stream:
            _write_text("".join(state.full_response))
        _write_newline()

    if not quiet:
        console.print()
        console.print("[green]✓ Task completed[/green]")


def _build_non_interactive_header(assistant_id: str, thread_id: str) -> Text:
    """Build the non-interactive mode header with model, agent, and thread info.

    The thread ID is rendered as a clickable hyperlink when LangSmith tracing
    is configured.

    Args:
        assistant_id: Agent identifier.
        thread_id: Thread identifier.

    Returns:
        Rich Text object with the formatted header line.
    """
    default_label = " (default)" if assistant_id == DEFAULT_AGENT_NAME else ""
    parts: list[tuple[str, str | Style]] = [
        (f"Agent: {assistant_id}{default_label}", "dim"),
    ]

    if settings.model_name:
        parts.extend([(" | ", "dim"), (f"Model: {settings.model_name}", "dim")])

    parts.append((" | ", "dim"))

    # Attempt to build a clickable thread link via LangSmith
    thread_url = build_langsmith_thread_url(thread_id)
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
    quiet: bool = False,
    stream: bool = True,
) -> int:
    """Run a single task non-interactively and exit.

    When no `shell_allow_list` is configured, shell execution is disabled
    and all other tool calls are auto-approved (no HITL prompts). When an
    allow-list **is** provided, shell execution is enabled but gated by the
    list; commands not in the list are rejected with an error message sent
    back to the agent.

    Note: when LangSmith tracing is configured, `_build_non_interactive_header`
    makes a synchronous network call (via `fetch_langsmith_project_url`) to
    resolve the thread URL. This blocks the event loop briefly at startup.

    Args:
        message: The task/message to execute.
        assistant_id: Agent identifier for memory storage.
        model_name: Optional model name to use.
        model_params: Extra kwargs from `--model-params` to pass to the model.

            These override config file values.
        sandbox_type: Type of sandbox (`'none'`, `'modal'`,
            `'runloop'`, `'daytona'`, `'langsmith'`).
        sandbox_id: Optional existing sandbox ID to reuse.
        sandbox_setup: Optional path to setup script to run in the sandbox
            after creation.
        quiet: When `True`, all console output (headers, status messages,
            tool notifications, HITL decisions, errors) is redirected to
            stderr so that only the agent's response text appears on stdout.
        stream: When `True` (default), text chunks are written to stdout
            as they arrive.

            When `False`, the full response is buffered and written to stdout in
            one shot after the agent finishes.

    Returns:
        Exit code: 0 for success, 1 for error, 130 for keyboard interrupt.
    """
    # stderr=True routes all console.print() to stderr; agent response text
    # uses _write_text() -> sys.stdout directly.
    console = Console(stderr=True) if quiet else Console()
    try:
        result = create_model(model_name, extra_kwargs=model_params)
    except ModelConfigError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1

    model = result.model
    result.apply_to_settings()
    thread_id = generate_thread_id()

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "assistant_id": assistant_id,
            "agent_name": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    }

    if not quiet:
        console.print("[dim]Running task non-interactively...[/dim]")
        header = _build_non_interactive_header(assistant_id, thread_id)
        console.print(header)
        console.print()

    sandbox_backend = None
    exit_stack = contextlib.ExitStack()

    if sandbox_type != "none":
        # Conditional: sandbox_factory transitively imports provider modules
        # and SDKs — skip that cost for the common no-sandbox path.
        from deepagents_cli.integrations.sandbox_factory import (
            create_sandbox,
        )

        try:
            sandbox_cm = create_sandbox(
                sandbox_type,
                sandbox_id=sandbox_id,
                setup_script_path=sandbox_setup,
            )
            sandbox_backend = exit_stack.enter_context(sandbox_cm)
        except (ImportError, ValueError, RuntimeError) as e:
            logger.exception("Sandbox creation failed")
            console.print(f"[red]Sandbox creation failed: {e}[/red]")
            return 1
        except NotImplementedError as e:
            logger.exception("Unsupported sandbox type %r", sandbox_type)
            console.print(
                f"[red]Sandbox type '{sandbox_type}' is not yet supported: {e}[/red]"
            )
            return 1

    try:
        async with get_checkpointer() as checkpointer:
            tools = [http_request, fetch_url]
            if settings.has_tavily:
                tools.append(web_search)

            # If an allow-list is provided, enable shell but disable
            # auto-approve so HITL can gate commands. If no allow-list, disable
            # shell entirely and auto-approve all other tools.
            enable_shell = bool(settings.shell_allow_list)
            use_auto_approve = not enable_shell

            agent, composite_backend = create_cli_agent(
                model=model,
                assistant_id=assistant_id,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=sandbox_type if sandbox_type != "none" else None,
                auto_approve=use_auto_approve,
                enable_shell=enable_shell,
                checkpointer=checkpointer,
            )

            file_op_tracker = FileOpTracker(
                assistant_id=assistant_id, backend=composite_backend
            )

            await _run_agent_loop(
                agent,
                message,
                config,
                console,
                file_op_tracker,
                quiet=quiet,
                stream=stream,
            )
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except HITLIterationLimitError as e:
        console.print(f"\n[red]{e}[/red]")
        console.print(
            "[yellow]Hint: The agent may be repeatedly attempting commands "
            "that are not in the allow-list. Consider expanding the "
            "--shell-allow-list or adjusting the task.[/yellow]"
        )
        return 1
    except (ValueError, OSError) as e:
        logger.exception("Error during non-interactive execution")
        console.print(f"\n[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during non-interactive execution")
        console.print(f"\n[red]Unexpected error ({type(e).__name__}): {e}[/red]")
        return 1
    finally:
        try:
            exit_stack.close()
        except (OSError, RuntimeError) as cleanup_err:
            msg = "Failed to clean up resources during exit"
            logger.warning("%s: %s", msg, cleanup_err, exc_info=True)
            console.print(
                f"[yellow]Warning: Resource cleanup failed: {cleanup_err}[/yellow]"
            )
