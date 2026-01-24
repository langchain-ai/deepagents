"""Non-interactive execution mode for deepagents CLI."""
# ruff: noqa: T201, PLR0912, PLR0915, PLR2004

import contextlib
from datetime import UTC, datetime

from langchain.agents.middleware.human_in_the_loop import HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter
from rich.console import Console

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model, is_shell_command_allowed, settings
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search

_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)


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

    # Create model
    model = create_model(model_name)

    # Generate thread ID
    thread_id = generate_thread_id()

    # Create config
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "assistant_id": assistant_id,
            "agent_name": assistant_id,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    }

    console.print("[dim]Running task non-interactively...[/dim]")
    console.print(f"[dim]Agent: {assistant_id} | Thread: {thread_id}[/dim]\n")

    # Setup sandbox if needed
    sandbox_backend = None
    sandbox_cm = None

    if sandbox_type != "none":
        from deepagents_cli.integrations.sandbox_factory import create_sandbox

        try:
            sandbox_cm = create_sandbox(sandbox_type, sandbox_id=sandbox_id)
            sandbox_backend = sandbox_cm.__enter__()
        except (ImportError, ValueError, RuntimeError, NotImplementedError) as e:
            console.print(f"[red]❌ Sandbox creation failed: {e}[/red]")
            return 1

    try:
        # Use async context manager for checkpointer
        async with get_checkpointer() as checkpointer:
            # Create agent with conditional tools
            tools = [http_request, fetch_url]
            if settings.has_tavily:
                tools.append(web_search)

            # SECURITY: In non-interactive mode, shell is DISABLED by default.
            # Shell is only enabled if an explicit allow-list is configured via
            # DEEPAGENTS_SHELL_ALLOW_LIST environment variable.
            # This prevents arbitrary command execution without human oversight.
            enable_shell = bool(settings.shell_allow_list)

            # If shell is enabled (allow-list configured), use HITL for selective approval
            # Otherwise, auto-approve all non-shell tools
            use_auto_approve = not enable_shell

            # Create agent
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

            # Execute the task
            stream_input = {"messages": [{"role": "user", "content": message}]}

            file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=composite_backend)

            # Track assistant response
            full_response = []
            tool_call_buffers = {}
            pending_interrupts = {}
            hitl_response = {}
            interrupt_occurred = False

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue

                namespace, stream_mode, data = chunk

                # Only show main agent output (empty namespace)
                is_main_agent = not namespace

                # Handle interrupts (HITL requests)
                if stream_mode == "updates" and is_main_agent and "__interrupt__" in data:
                    interrupts: list[Interrupt] = data["__interrupt__"]
                    if interrupts:
                        for interrupt_obj in interrupts:
                            validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                                interrupt_obj.value
                            )
                            pending_interrupts[interrupt_obj.id] = validated_request
                            interrupt_occurred = True

                if stream_mode == "messages" and is_main_agent:
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue

                    message_obj, metadata = data

                    # Skip summarization chunks
                    if metadata and metadata.get("lc_source") == "summarization":
                        continue

                    if isinstance(message_obj, AIMessage):
                        # Process content blocks for streaming text
                        if hasattr(message_obj, "content_blocks"):
                            for block in message_obj.content_blocks:
                                block_type = block.get("type")

                                if block_type == "text":
                                    text = block.get("text", "")
                                    if text:
                                        print(text, end="", flush=True)
                                        full_response.append(text)

                                elif block_type in ("tool_call_chunk", "tool_call"):
                                    # Buffer tool call information
                                    chunk_name = block.get("name")
                                    chunk_id = block.get("id")
                                    chunk_index = block.get("index")

                                    buffer_key = (
                                        chunk_index
                                        if chunk_index is not None
                                        else (
                                            chunk_id
                                            if chunk_id is not None
                                            else f"unknown-{len(tool_call_buffers)}"
                                        )
                                    )

                                    if buffer_key not in tool_call_buffers:
                                        tool_call_buffers[buffer_key] = {"name": None, "id": None}

                                    if chunk_name:
                                        tool_call_buffers[buffer_key]["name"] = chunk_name
                                        # Show tool call when we get the name
                                        if full_response:
                                            print()  # Newline after any text
                                        console.print(f"[dim]🔧 Calling tool: {chunk_name}[/dim]")

                    elif isinstance(message_obj, ToolMessage):
                        # Track file operations
                        record = file_op_tracker.complete_with_message(message_obj)

                        # Show file operation diffs
                        if record and record.diff:
                            console.print(f"[dim]📝 {record.display_path}[/dim]")

            # Handle HITL interrupts - loop to handle multiple interrupt cycles
            while interrupt_occurred:
                interrupt_occurred = False  # Reset flag
                current_interrupts = dict(pending_interrupts)  # Copy to iterate
                pending_interrupts.clear()

                for interrupt_id, hitl_request in current_interrupts.items():
                    decisions = []

                    for action_request in hitl_request["action_requests"]:
                        action_name = action_request.get("name", "")

                        # Check if this is a shell command and if we have an allow-list
                        if action_name == "shell" and settings.shell_allow_list:
                            command = action_request.get("args", {}).get("command", "")

                            if is_shell_command_allowed(command, settings.shell_allow_list):
                                # Auto-approve allowed commands
                                console.print(f"[dim]✓ Auto-approved: {command}[/dim]")
                                decisions.append({"type": "approve"})
                            else:
                                # Reject disallowed commands but let agent continue
                                allowed_list_str = ", ".join(settings.shell_allow_list)
                                console.print(f"\n[red]❌ Shell command rejected:[/red] {command}")
                                console.print(
                                    f"[yellow]Allowed commands:[/yellow] {allowed_list_str}"
                                )
                                decisions.append(
                                    {
                                        "type": "reject",
                                        "message": (
                                            f"Command '{command}' is not in the allow-list. "
                                            f"Allowed commands: {allowed_list_str}. "
                                            f"Please use allowed commands or try another approach."
                                        ),
                                    }
                                )
                        else:
                            # For non-shell commands or when no allow-list, auto-approve
                            # (This maintains backward compatibility for non-interactive mode)
                            decisions.append({"type": "approve"})

                    hitl_response[interrupt_id] = {"decisions": decisions}

                # Resume with decisions
                stream_input = Command(resume=hitl_response)

                # Continue streaming with the resolution
                async for chunk in agent.astream(
                    stream_input,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                    config=config,
                    durability="exit",
                ):
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        continue

                    namespace, stream_mode, data = chunk
                    is_main_agent = not namespace

                    # Handle any new interrupts
                    if stream_mode == "updates" and is_main_agent and "__interrupt__" in data:
                        interrupts: list[Interrupt] = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                                    interrupt_obj.value
                                )
                                pending_interrupts[interrupt_obj.id] = validated_request
                                interrupt_occurred = True

                    if stream_mode == "messages" and is_main_agent:
                        if not isinstance(data, tuple) or len(data) != 2:
                            continue

                        message_obj, metadata = data

                        if isinstance(message_obj, AIMessage):
                            if hasattr(message_obj, "content_blocks"):
                                for block in message_obj.content_blocks:
                                    block_type = block.get("type")

                                    if block_type == "text":
                                        text = block.get("text", "")
                                        if text:
                                            print(text, end="", flush=True)
                                            full_response.append(text)

                                    elif block_type in ("tool_call_chunk", "tool_call"):
                                        # Show new tool calls
                                        chunk_name = block.get("name")
                                        if chunk_name:
                                            if full_response:
                                                print()  # Newline after any text
                                            console.print(
                                                f"[dim]🔧 Calling tool: {chunk_name}[/dim]"
                                            )

                        elif isinstance(message_obj, ToolMessage):
                            record = file_op_tracker.complete_with_message(message_obj)
                            if record and record.diff:
                                console.print(f"[dim]📝 {record.display_path}[/dim]")

            # Final newline
            if full_response:
                print()

            console.print("\n[green]✓ Task completed[/green]")
            return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]❌ Error: {e}[/red]")
        return 1
    finally:
        # Clean up sandbox if we created one
        if sandbox_cm is not None:
            with contextlib.suppress(Exception):
                sandbox_cm.__exit__(None, None, None)
