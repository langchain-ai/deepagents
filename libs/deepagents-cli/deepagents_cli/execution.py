"""Task execution and streaming logic for the CLI."""

import asyncio
import json
import sys
import termios
import tty

from langchain.agents.middleware.human_in_the_loop import (
    ActionRequest,
    ApproveDecision,
    Decision,
    HITLRequest,
    HITLResponse,
    RejectDecision,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel

from deepagents_cli.config import COLORS, console
from deepagents_cli.file_ops import FileOpTracker, build_approval_preview
from deepagents_cli.image_utils import create_multimodal_content
from deepagents_cli.input import ImageTracker, parse_file_mentions
from deepagents_cli.ui import (
    TokenTracker,
    format_tool_display,
    format_tool_message_content,
    render_diff_block,
    render_file_operation,
    render_todo_list,
)

_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)


def prompt_for_tool_approval(
    action_request: ActionRequest,
    assistant_id: str | None,
) -> Decision | dict:
    """Prompt user to approve/reject a tool action with arrow key navigation.

    Returns:
        Decision (ApproveDecision or RejectDecision) OR
        dict with {"type": "auto_approve_all"} to switch to auto-approve mode
    """
    description = action_request.get("description", "No description available")
    name = action_request["name"]
    args = action_request["args"]
    preview = build_approval_preview(name, args, assistant_id) if name else None

    body_lines = []
    if preview:
        body_lines.append(f"[bold]{preview.title}[/bold]")
        body_lines.extend(preview.details)
        if preview.error:
            body_lines.append(f"[red]{preview.error}[/red]")
    else:
        body_lines.append(description)

    # Display action info first
    console.print(
        Panel(
            "[bold yellow]⚠️  Tool Action Requires Approval[/bold yellow]\n\n"
            + "\n".join(body_lines),
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    if preview and preview.diff and not preview.error:
        console.print()
        render_diff_block(preview.diff, preview.diff_title or preview.title)

    options = ["approve", "reject", "auto-accept all going forward"]
    selected = 0  # Start with approve selected

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            # Hide cursor during menu interaction
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

            # Initial render flag
            first_render = True

            while True:
                if not first_render:
                    # Move cursor back to start of menu (up 3 lines, then to start of line)
                    sys.stdout.write("\033[3A\r")

                first_render = False

                # Display options vertically with ANSI color codes
                for i, option in enumerate(options):
                    sys.stdout.write("\r\033[K")  # Clear line from cursor to end

                    if i == selected:
                        if option == "approve":
                            # Green bold with filled checkbox
                            sys.stdout.write("\033[1;32m☑ Approve\033[0m\n")
                        elif option == "reject":
                            # Red bold with filled checkbox
                            sys.stdout.write("\033[1;31m☑ Reject\033[0m\n")
                        else:
                            # Blue bold with filled checkbox for auto-accept
                            sys.stdout.write("\033[1;34m☑ Auto-accept all going forward\033[0m\n")
                    elif option == "approve":
                        # Dim with empty checkbox
                        sys.stdout.write("\033[2m☐ Approve\033[0m\n")
                    elif option == "reject":
                        # Dim with empty checkbox
                        sys.stdout.write("\033[2m☐ Reject\033[0m\n")
                    else:
                        # Dim with empty checkbox
                        sys.stdout.write("\033[2m☐ Auto-accept all going forward\033[0m\n")

                sys.stdout.flush()

                # Read key
                char = sys.stdin.read(1)

                if char == "\x1b":  # ESC sequence (arrow keys)
                    next1 = sys.stdin.read(1)
                    next2 = sys.stdin.read(1)
                    if next1 == "[":
                        if next2 == "B":  # Down arrow
                            selected = (selected + 1) % len(options)
                        elif next2 == "A":  # Up arrow
                            selected = (selected - 1) % len(options)
                elif char in {"\r", "\n"}:  # Enter
                    sys.stdout.write("\r\n")  # Move to start of line and add newline
                    break
                elif char == "\x03":  # Ctrl+C
                    sys.stdout.write("\r\n")  # Move to start of line and add newline
                    raise KeyboardInterrupt
                elif char.lower() == "a":
                    selected = 0
                    sys.stdout.write("\r\n")  # Move to start of line and add newline
                    break
                elif char.lower() == "r":
                    selected = 1
                    sys.stdout.write("\r\n")  # Move to start of line and add newline
                    break

        finally:
            # Show cursor again
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (termios.error, AttributeError):
        # Fallback for non-Unix systems
        console.print("  ☐ (A)pprove  (default)")
        console.print("  ☐ (R)eject")
        console.print("  ☐ (Auto)-accept all going forward")
        choice = input("\nChoice (A/R/Auto, default=Approve): ").strip().lower()
        if choice in {"r", "reject"}:
            selected = 1
        elif choice in {"auto", "auto-accept"}:
            selected = 2
        else:
            selected = 0

    # Return decision based on selection
    if selected == 0:
        return ApproveDecision(type="approve")
    if selected == 1:
        return RejectDecision(type="reject", message="User rejected the command")
    # Return special marker for auto-approve mode
    return {"type": "auto_approve_all"}


async def execute_task(
    user_input: str,
    agent,
    assistant_id: str | None,
    session_state,
    token_tracker: TokenTracker | None = None,
    backend=None,
    image_tracker: ImageTracker | None = None,
) -> None:
    """Execute any task by passing it directly to the AI agent."""
    # Parse file mentions and inject content if any
    prompt_text, mentioned_files = parse_file_mentions(user_input)

    if mentioned_files:
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for file_path in mentioned_files:
            try:
                content = file_path.read_text()
                # Limit file content to reasonable size
                if len(content) > 50000:
                    content = content[:50000] + "\n... (file truncated)"
                context_parts.append(
                    f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
                )
            except Exception as e:
                context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")

        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    # Include images in the message content
    images_to_send = []
    if image_tracker:
        images_to_send = image_tracker.get_images()
    if images_to_send:
        message_content = create_multimodal_content(final_input, images_to_send)
    else:
        message_content = final_input

    config = {
        "configurable": {"thread_id": session_state.thread_id},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {},
    }

    has_responded = False
    captured_input_tokens = 0
    captured_output_tokens = 0
    current_todos = None  # Track current todo list state

    status = console.status(f"[bold {COLORS['thinking']}]Agent is thinking...", spinner="dots")
    status.start()
    spinner_active = True

    tool_icons = {
        "read_file": "📖",
        "write_file": "✏️",
        "edit_file": "✂️",
        "ls": "📁",
        "glob": "🔍",
        "grep": "🔎",
        "shell": "⚡",
        "execute": "🔧",
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "write_todos": "📋",
    }

    file_op_tracker = FileOpTracker(assistant_id=assistant_id, backend=backend)

    # Track which tool calls we've displayed to avoid duplicates
    displayed_tool_ids = set()
    # Buffer partial tool-call chunks keyed by streaming index
    tool_call_buffers: dict[str | int, dict] = {}
    # Buffer assistant text so we can render complete markdown segments
    pending_text = ""

    def flush_text_buffer(*, final: bool = False) -> None:
        """Flush accumulated assistant text as rendered markdown when appropriate."""
        nonlocal pending_text, spinner_active, has_responded
        if not final or not pending_text.strip():
            return
        if spinner_active:
            status.stop()
            spinner_active = False
        if not has_responded:
            console.print("●", style=COLORS["agent"], markup=False, end=" ")
            has_responded = True
        markdown = Markdown(pending_text.rstrip())
        console.print(markdown, style=COLORS["agent"])
        pending_text = ""

    # Clear images from tracker after creating the message
    # (they've been encoded into the message content)
    if image_tracker:
        image_tracker.clear()

    # Stream input - may need to loop if there are interrupts
    stream_input = {"messages": [{"role": "user", "content": message_content}]}

    try:
        while True:
            interrupt_occurred = False
            hitl_response: dict[str, HITLResponse] = {}
            suppress_resumed_output = False
            # Track all pending interrupts: {interrupt_id: request_data}
            pending_interrupts: dict[str, HITLRequest] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],  # Dual-mode for HITL support
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                # Unpack chunk - with subgraphs=True and dual-mode, it's (namespace, stream_mode, data)
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue

                _namespace, current_stream_mode, data = chunk

                # Handle UPDATES stream - for interrupts and todos
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for interrupts - collect ALL pending interrupts
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                # Interrupt has required fields: value (HITLRequest) and id (str)
                                # Validate the HITLRequest using TypeAdapter
                                try:
                                    validated_request = _HITL_REQUEST_ADAPTER.validate_python(
                                        interrupt_obj.value
                                    )
                                    pending_interrupts[interrupt_obj.id] = validated_request
                                    interrupt_occurred = True
                                except ValidationError as e:
                                    console.print(
                                        f"[yellow]Warning: Invalid HITL request data: {e}[/yellow]",
                                        style="dim",
                                    )
                                    raise

                    # Extract chunk_data from updates for todo checking
                    chunk_data = next(iter(data.values())) if data else None
                    if chunk_data and isinstance(chunk_data, dict):
                        # Check for todo updates
                        if "todos" in chunk_data:
                            new_todos = chunk_data["todos"]
                            if new_todos != current_todos:
                                current_todos = new_todos
                                # Stop spinner before rendering todos
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                console.print()
                                render_todo_list(new_todos)
                                console.print()

                # Handle MESSAGES stream - for content and tool calls
                elif current_stream_mode == "messages":
                    # Messages stream returns (message, metadata) tuples
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue

                    message, _metadata = data

                    if isinstance(message, HumanMessage):
                        content = message.text
                        if content:
                            flush_text_buffer(final=True)
                            if spinner_active:
                                status.stop()
                                spinner_active = False
                            if not has_responded:
                                console.print("●", style=COLORS["agent"], markup=False, end=" ")
                                has_responded = True
                            markdown = Markdown(content)
                            console.print(markdown, style=COLORS["agent"])
                            console.print()
                        continue

                    if isinstance(message, ToolMessage):
                        # Tool results are sent to the agent, not displayed to users
                        # Exception: show shell command errors to help with debugging
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(message.content)
                        record = file_op_tracker.complete_with_message(message)

                        # Reset spinner message after tool completes
                        if spinner_active:
                            status.update(f"[bold {COLORS['thinking']}]Agent is thinking...")

                        if tool_name == "shell" and tool_status != "success":
                            flush_text_buffer(final=True)
                            if tool_content:
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                console.print()
                                console.print(tool_content, style="red", markup=False)
                                console.print()
                        elif tool_content and isinstance(tool_content, str):
                            stripped = tool_content.lstrip()
                            if stripped.lower().startswith("error"):
                                flush_text_buffer(final=True)
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                console.print()
                                console.print(tool_content, style="red", markup=False)
                                console.print()

                        if record:
                            flush_text_buffer(final=True)
                            if spinner_active:
                                status.stop()
                                spinner_active = False
                            console.print()
                            render_file_operation(record)
                            console.print()
                            if not spinner_active:
                                status.start()
                                spinner_active = True

                        # For all other tools (web_search, http_request, etc.),
                        # results are hidden from user - agent will process and respond
                        continue

                    # Check if this is an AIMessageChunk
                    if not hasattr(message, "content_blocks"):
                        # Fallback for messages without content_blocks
                        continue

                    # Extract token usage if available
                    if token_tracker and hasattr(message, "usage_metadata"):
                        usage = message.usage_metadata
                        if usage:
                            input_toks = usage.get("input_tokens", 0)
                            output_toks = usage.get("output_tokens", 0)
                            if input_toks or output_toks:
                                captured_input_tokens = max(captured_input_tokens, input_toks)
                                captured_output_tokens = max(captured_output_tokens, output_toks)

                    # Process content blocks (this is the key fix!)
                    for block in message.content_blocks:
                        block_type = block.get("type")

                        # Handle text blocks
                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                pending_text += text

                        # Handle reasoning blocks
                        elif block_type == "reasoning":
                            flush_text_buffer(final=True)
                            reasoning = block.get("reasoning", "")
                            if reasoning and spinner_active:
                                status.stop()
                                spinner_active = False
                                # Could display reasoning differently if desired
                                # For now, skip it or handle minimally

                        # Handle tool call chunks
                        # Some models (OpenAI, Anthropic) stream tool_call_chunks
                        # Others (Gemini) don't stream them and just return the full tool_call
                        elif block_type in ("tool_call_chunk", "tool_call"):
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            # Use index as stable buffer key; fall back to id if needed
                            buffer_key: str | int
                            if chunk_index is not None:
                                buffer_key = chunk_index
                            elif chunk_id is not None:
                                buffer_key = chunk_id
                            else:
                                buffer_key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                buffer_key,
                                {"name": None, "id": None, "args": None, "args_parts": []},
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
                                    parts: list[str] = buffer.setdefault("args_parts", [])
                                    if not parts or chunk_args != parts[-1]:
                                        parts.append(chunk_args)
                                    buffer["args"] = "".join(parts)
                            elif chunk_args is not None:
                                buffer["args"] = chunk_args

                            buffer_name = buffer.get("name")
                            buffer_id = buffer.get("id")
                            if buffer_name is None:
                                continue

                            parsed_args = buffer.get("args")
                            if isinstance(parsed_args, str):
                                if not parsed_args:
                                    continue
                                try:
                                    parsed_args = json.loads(parsed_args)
                                except json.JSONDecodeError:
                                    # Wait for more chunks to form valid JSON
                                    continue
                            elif parsed_args is None:
                                continue

                            # Ensure args are in dict form for formatter
                            if not isinstance(parsed_args, dict):
                                parsed_args = {"value": parsed_args}

                            flush_text_buffer(final=True)
                            if buffer_id is not None:
                                if buffer_id not in displayed_tool_ids:
                                    displayed_tool_ids.add(buffer_id)
                                    file_op_tracker.start_operation(
                                        buffer_name, parsed_args, buffer_id
                                    )
                                else:
                                    file_op_tracker.update_args(buffer_id, parsed_args)
                            tool_call_buffers.pop(buffer_key, None)
                            icon = tool_icons.get(buffer_name, "🔧")

                            if spinner_active:
                                status.stop()

                            if has_responded:
                                console.print()

                            display_str = format_tool_display(buffer_name, parsed_args)
                            console.print(
                                f"  {icon} {display_str}",
                                style=f"dim {COLORS['tool']}",
                                markup=False,
                            )

                            # Restart spinner with context about which tool is executing
                            status.update(f"[bold {COLORS['thinking']}]Executing {display_str}...")
                            status.start()
                            spinner_active = True

                    if getattr(message, "chunk_position", None) == "last":
                        flush_text_buffer(final=True)

            # After streaming loop - handle interrupt if it occurred
            flush_text_buffer(final=True)

            # Handle human-in-the-loop after stream completes
            if interrupt_occurred:
                any_rejected = False

                for interrupt_id, hitl_request in pending_interrupts.items():
                    # Check if auto-approve is enabled
                    if session_state.auto_approve:
                        # Auto-approve all commands without prompting
                        decisions = []
                        for action_request in hitl_request["action_requests"]:
                            # Show what's being auto-approved (brief, dim message)
                            if spinner_active:
                                status.stop()
                                spinner_active = False

                            description = action_request.get("description", "tool action")
                            console.print()
                            console.print(f"  [dim]⚡ {description}[/dim]")

                            decisions.append({"type": "approve"})

                        hitl_response[interrupt_id] = {"decisions": decisions}

                        # Restart spinner for continuation
                        if not spinner_active:
                            status.start()
                            spinner_active = True
                    else:
                        # Normal HITL flow - stop spinner and prompt user
                        if spinner_active:
                            status.stop()
                            spinner_active = False

                        # Handle human-in-the-loop approval
                        decisions = []
                        for action_index, action_request in enumerate(
                            hitl_request["action_requests"]
                        ):
                            decision = prompt_for_tool_approval(
                                action_request,
                                assistant_id,
                            )

                            # Check if user wants to switch to auto-approve mode
                            if (
                                isinstance(decision, dict)
                                and decision.get("type") == "auto_approve_all"
                            ):
                                # Switch to auto-approve mode
                                session_state.auto_approve = True
                                console.print()
                                console.print("[bold blue]✓ Auto-approve mode enabled[/bold blue]")
                                console.print(
                                    "[dim]All future tool actions will be automatically approved.[/dim]"
                                )
                                console.print()

                                # Approve this action and all remaining actions in the batch
                                decisions.append({"type": "approve"})
                                for _remaining_action in hitl_request["action_requests"][
                                    action_index + 1 :
                                ]:
                                    decisions.append({"type": "approve"})
                                break
                            decisions.append(decision)

                            # Mark file operations as HIL-approved if user approved
                            if decision.get("type") == "approve":
                                tool_name = action_request.get("name")
                                if tool_name in {"write_file", "edit_file"}:
                                    file_op_tracker.mark_hitl_approved(
                                        tool_name, action_request.get("args", {})
                                    )

                        if any(decision.get("type") == "reject" for decision in decisions):
                            any_rejected = True

                        hitl_response[interrupt_id] = {"decisions": decisions}

                suppress_resumed_output = any_rejected

            if interrupt_occurred and hitl_response:
                if suppress_resumed_output:
                    if spinner_active:
                        status.stop()
                        spinner_active = False

                    console.print("[yellow]Command rejected.[/yellow]", style="bold")
                    console.print("Tell the agent what you'd like to do differently.")
                    console.print()
                    return

                # Resume the agent with the human decision
                stream_input = Command(resume=hitl_response)
                # Continue the while loop to restream
            else:
                # No interrupt, break out of while loop
                break

    except asyncio.CancelledError:
        # Event loop cancelled the task (e.g. Ctrl+C during streaming) - clean up and return
        if spinner_active:
            status.stop()
        console.print("\n[yellow]Interrupted by user[/yellow]")
        console.print("Updating agent state...", style="dim")

        try:
            await agent.aupdate_state(
                config=config,
                values={
                    "messages": [
                        HumanMessage(content="[The previous request was cancelled by the system]")
                    ]
                },
            )
            console.print("Ready for next command.\n", style="dim")
        except Exception as e:
            console.print(f"[red]Warning: Failed to update agent state: {e}[/red]\n")

        return

    except KeyboardInterrupt:
        # User pressed Ctrl+C - clean up and exit gracefully
        if spinner_active:
            status.stop()
        console.print("\n[yellow]Interrupted by user[/yellow]")
        console.print("Updating agent state...", style="dim")

        # Inform the agent synchronously (in async context)
        try:
            await agent.aupdate_state(
                config=config,
                values={
                    "messages": [
                        HumanMessage(content="[User interrupted the previous request with Ctrl+C]")
                    ]
                },
            )
            console.print("Ready for next command.\n", style="dim")
        except Exception as e:
            console.print(f"[red]Warning: Failed to update agent state: {e}[/red]\n")

        return

    if spinner_active:
        status.stop()

    if has_responded:
        console.print()
        # Track token usage (display only via /tokens command)
        if token_tracker and (captured_input_tokens or captured_output_tokens):
            token_tracker.add(captured_input_tokens, captured_output_tokens)
