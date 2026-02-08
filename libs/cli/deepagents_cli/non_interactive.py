"""Non-interactive execution mode for deepagents CLI.

Provides `run_non_interactive` which runs a single user task against the
agent graph, streams results to stdout, and exits with an appropriate code.

Shell commands are gated by an optional allow-list; all other tool calls are
auto-approved so that the agent can operate without a human in the loop.
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

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import (
    SHELL_TOOL_NAMES,
    create_model,
    is_shell_command_allowed,
    settings,
)
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

logger = logging.getLogger(__name__)

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
    """Write text to stdout (without a trailing newline) for streaming output.

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
        full_response: Accumulated text fragments from the AI message stream.
        tool_call_buffers: Maps a tool-call index or ID to its name/ID
            metadata for in-progress tool calls.
        pending_interrupts: Maps interrupt IDs to their validated HITL
            requests that are awaiting decisions.
        hitl_response: Maps interrupt IDs to decision lists used to resume
            the agent after HITL processing.
        interrupt_occurred: Flag indicating whether any HITL interrupt was
            received during the current stream pass.
    """

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
) -> None:
    """Extract HITL interrupts from an `updates` chunk and record them.

    Args:
        data: The `updates` dict that contains an `__interrupt__` key.
        state: Stream state to update with new pending interrupts.
    """
    interrupts = data["__interrupt__"]
    if interrupts:
        for interrupt_obj in interrupts:
            try:
                validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                    interrupt_obj.value
                )
            except ValidationError:
                logger.warning("Skipping malformed HITL interrupt %s", interrupt_obj.id)
                continue
            state.pending_interrupts[interrupt_obj.id] = validated_request
            state.interrupt_occurred = True


def _process_ai_message(
    message_obj: AIMessage,
    state: StreamState,
    console: Console,
) -> None:
    """Extract text and tool-call blocks from an AI message and render them.

    Text blocks are streamed to stdout; tool-call blocks are buffered and
    their names are printed to the console.

    Args:
        message_obj: The `AIMessage` received from the stream.
        state: Stream state for accumulating response text and tool-call buffers.
        console: Rich console for formatted output.
    """
    if not hasattr(message_obj, "content_blocks"):
        return
    for block in message_obj.content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
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
                if state.full_response:
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
        return

    message_obj, metadata = data

    # Skip summarization chunks
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
        return

    namespace, stream_mode, data = chunk
    is_main_agent = not namespace

    if not is_main_agent:
        return

    if stream_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
        _process_interrupts(cast("dict[str, list[Interrupt]]", data), state)
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

    When a `shell_allow_list` is configured and the action is a shell tool,
    the command is validated against the allow-list. All other actions
    (non-shell tools, or shell when no allow-list is set) are approved
    unconditionally -- the allow-list is the **only** gating mechanism in
    non-interactive mode.

    Args:
        action_request: The action-request dict emitted by the HITL middleware.

            Must contain at least a `name` key.
        console: Rich console for status output.

    Returns:
        Decision dict with a `type` key (`"approve"` or `"reject"`)
            and an optional `message` key with a human-readable explanation.
    """
    action_name = action_request.get("name", "")

    if action_name in SHELL_TOOL_NAMES and settings.shell_allow_list:
        command = action_request.get("args", {}).get("command", "")

        if is_shell_command_allowed(command, settings.shell_allow_list):
            console.print(f"[dim]✓ Auto-approved: {command}[/dim]")
            return {"type": "approve"}

        allowed_list_str = ", ".join(settings.shell_allow_list)
        console.print(f"\n[red]❌ Shell command rejected:[/red] {command}")
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

    Raises:
        RuntimeError: If the HITL iteration limit is exceeded.
    """
    state = StreamState()
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
            raise RuntimeError(msg)
        state.interrupt_occurred = False
        state.hitl_response.clear()
        _process_hitl_interrupts(state, console)
        stream_input = Command(resume=state.hitl_response)
        await _stream_agent(
            agent, stream_input, config, state, console, file_op_tracker
        )

    if state.full_response:
        _write_newline()

    console.print("\n[green]✓ Task completed[/green]")


async def run_non_interactive(
    message: str,
    assistant_id: str = "agent",
    model_name: str | None = None,
    sandbox_type: str = "none",  # str (not None) to match argparse choices
    sandbox_id: str | None = None,
) -> int:
    """Run a single task non-interactively and exit.

    When no `shell_allow_list` is configured, shell execution is disabled
    and all other tool calls are auto-approved (no HITL prompts). When an
    allow-list **is** provided, shell execution is enabled but gated by the
    list; commands not in the list are rejected with an error message sent
    back to the agent.

    Args:
        message: The task/message to execute.
        assistant_id: Agent identifier for memory storage.
        model_name: Optional model name to use.
        sandbox_type: Type of sandbox (`'none'`, `'modal'`,
            `'runloop'`, `'daytona'`).
        sandbox_id: Optional existing sandbox ID to reuse.

    Returns:
        Exit code: 0 for success, 1 for error, 130 for keyboard interrupt.
    """
    console = Console()
    model = create_model(model_name)
    thread_id = generate_thread_id()

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "assistant_id": assistant_id,
            "agent_name": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    }

    console.print("[dim]Running task non-interactively...[/dim]")
    console.print(f"[dim]Agent: {assistant_id} | Thread: {thread_id}[/dim]\n")

    sandbox_backend = None
    exit_stack = contextlib.ExitStack()

    if sandbox_type != "none":
        from deepagents_cli.integrations.sandbox_factory import (  # noqa: PLC0415
            create_sandbox,
        )

        try:
            sandbox_cm = create_sandbox(sandbox_type, sandbox_id=sandbox_id)
            sandbox_backend = exit_stack.enter_context(sandbox_cm)
        except (ImportError, ValueError, RuntimeError) as e:
            console.print(f"[red]❌ Sandbox creation failed: {e}[/red]")
            return 1
        except NotImplementedError as e:
            console.print(
                f"[red]❌ Sandbox type '{sandbox_type}' is not yet supported: {e}[/red]"
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

            await _run_agent_loop(agent, message, config, console, file_op_tracker)
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except (ValueError, OSError) as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        return 1
    except (RuntimeError, TypeError) as e:
        logger.exception("Unexpected error during non-interactive execution")
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
        return 1
    finally:
        try:
            exit_stack.close()
        except Exception:
            logger.warning("Failed to clean up resources during exit", exc_info=True)
