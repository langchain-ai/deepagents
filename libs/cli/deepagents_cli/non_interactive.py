"""Non-interactive execution mode for deepagents CLI."""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.human_in_the_loop import HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter
from rich.console import Console

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model, is_shell_command_allowed, settings
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.integrations.sandbox_factory import create_sandbox
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)

# Constants for stream chunk validation
_STREAM_CHUNK_LENGTH = 3
_MESSAGE_DATA_LENGTH = 2


def _write_text(text: str) -> None:
    """Write text to stdout without newline for streaming output."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _write_newline() -> None:
    """Write a newline to stdout."""
    sys.stdout.write("\n")
    sys.stdout.flush()


@dataclass
class StreamState:
    """Tracks state during agent stream processing."""

    full_response: list[str] = field(default_factory=list)
    tool_call_buffers: dict[Any, dict[str, Any]] = field(default_factory=dict)
    pending_interrupts: dict[str, Any] = field(default_factory=dict)
    hitl_response: dict[str, Any] = field(default_factory=dict)
    interrupt_occurred: bool = False


def _process_interrupts(
    data: dict[str, Any],
    state: StreamState,
) -> None:
    """Process interrupt data and update state."""
    interrupts: list[Interrupt] = data["__interrupt__"]
    if interrupts:
        for interrupt_obj in interrupts:
            validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                interrupt_obj.value
            )
            state.pending_interrupts[interrupt_obj.id] = validated_request
            state.interrupt_occurred = True


def _process_text_block(block: dict[str, Any], state: StreamState) -> None:
    """Process a text block from the stream."""
    text = block.get("text", "")
    if text:
        _write_text(text)
        state.full_response.append(text)


def _process_tool_call_block(
    block: dict[str, Any],
    state: StreamState,
    console: Console,
) -> None:
    """Process a tool call block from the stream."""
    chunk_name = block.get("name")
    chunk_id = block.get("id")
    chunk_index = block.get("index")

    buffer_key = (
        chunk_index
        if chunk_index is not None
        else (
            chunk_id
            if chunk_id is not None
            else f"unknown-{len(state.tool_call_buffers)}"
        )
    )

    if buffer_key not in state.tool_call_buffers:
        state.tool_call_buffers[buffer_key] = {"name": None, "id": None}

    if chunk_name:
        state.tool_call_buffers[buffer_key]["name"] = chunk_name
        if state.full_response:
            _write_newline()
        console.print(f"[dim]🔧 Calling tool: {chunk_name}[/dim]")


def _process_ai_message(
    message_obj: AIMessage,
    state: StreamState,
    console: Console,
) -> None:
    """Process an AI message from the stream."""
    if hasattr(message_obj, "content_blocks"):
        for block in message_obj.content_blocks:
            if not isinstance(block, dict):
                continue
            typed_block = cast("dict[str, Any]", block)
            block_type = typed_block.get("type")
            if block_type == "text":
                _process_text_block(typed_block, state)
            elif block_type in {"tool_call_chunk", "tool_call"}:
                _process_tool_call_block(typed_block, state, console)


def _process_message_chunk(
    data: tuple[Any, Any],
    state: StreamState,
    console: Console,
    file_op_tracker: FileOpTracker,
) -> None:
    """Process a message chunk from the stream."""
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
    """Process a single chunk from the agent stream."""
    if not isinstance(chunk, tuple) or len(chunk) != _STREAM_CHUNK_LENGTH:
        return

    namespace, stream_mode, data = chunk
    is_main_agent = not namespace

    if not is_main_agent:
        return

    if stream_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
        _process_interrupts(cast("dict[str, Any]", data), state)
    elif stream_mode == "messages" and isinstance(data, tuple):
        _process_message_chunk(
            cast("tuple[Any, Any]", data), state, console, file_op_tracker
        )


def _make_hitl_decision(
    action_request: dict[str, Any], console: Console
) -> dict[str, str]:
    """Make a HITL decision for an action request.

    Returns:
        Decision dict with "type" key ("approve" or "reject") and optional "message".
    """
    action_name = action_request.get("name", "")

    if action_name == "shell" and settings.shell_allow_list:
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

    return {"type": "approve"}


def _process_hitl_interrupts(state: StreamState, console: Console) -> None:
    """Process pending HITL interrupts and prepare responses."""
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
    """Stream agent output and process chunks."""
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
    """Run the main agent loop with HITL interrupt handling."""
    state = StreamState()
    stream_input: dict[str, Any] | Command = {
        "messages": [{"role": "user", "content": message}]
    }

    # Initial stream
    await _stream_agent(agent, stream_input, config, state, console, file_op_tracker)

    # Handle HITL interrupts
    while state.interrupt_occurred:
        state.interrupt_occurred = False
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
    sandbox_type: str = "none",
    sandbox_id: str | None = None,
) -> int:
    """Run a single task non-interactively and exit.

    Args:
        message: The task/message to execute
        assistant_id: Agent identifier for memory storage
        model_name: Optional model name to use
        sandbox_type: Type of sandbox ("none", "modal", "runloop", "daytona")
        sandbox_id: Optional existing sandbox ID to reuse

    Returns:
        Exit code: 0 for success, 1 for error
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
        try:
            sandbox_cm = create_sandbox(sandbox_type, sandbox_id=sandbox_id)
            sandbox_backend = exit_stack.enter_context(sandbox_cm)
        except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
            console.print(f"[red]❌ Sandbox creation failed: {e}[/red]")
            return 1

    try:
        async with get_checkpointer() as checkpointer:
            tools = [http_request, fetch_url]
            if settings.has_tavily:
                tools.append(web_search)

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
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        return 1
    finally:
        with contextlib.suppress(Exception):
            exit_stack.close()
