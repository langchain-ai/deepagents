"""Task execution and streaming logic for the CLI."""

import json
import sys
import termios
import threading
import tty

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel

from .config import COLORS, console
from .file_ops import FileOpTracker, build_approval_preview
from .input import parse_file_mentions
from .ui import (
    TokenTracker,
    format_tool_display,
    format_tool_message_content,
    render_diff_block,
    render_file_operation,
    render_summary_panel,
    render_todo_list,
)


def is_summary_message(content: str) -> bool:
    """Detect if a message is from SummarizationMiddleware."""
    if not isinstance(content, str):
        return False
    content_lower = content.lower()
    # Common patterns from SummarizationMiddleware
    return (
        "conversation summary" in content_lower
        or "previous conversation" in content_lower
        or content.startswith("Summary:")
        or content.startswith("Conversation summary:")
        or "summarized the conversation" in content_lower
    )


def _extract_tool_args(action_request: dict) -> dict | None:
    """Best-effort extraction of tool call arguments from an action request."""
    if "tool_call" in action_request and isinstance(action_request["tool_call"], dict):
        args = action_request["tool_call"].get("args")
        if isinstance(args, dict):
            return args
    args = action_request.get("args")
    if isinstance(args, dict):
        return args
    return None


def prompt_for_tool_approval(action_request: dict, assistant_id: str | None) -> dict:
    """Prompt user to approve/reject a tool action with arrow key navigation."""
    description = action_request.get("description", "No description available")
    tool_name = action_request.get("name") or action_request.get("tool")
    tool_args = _extract_tool_args(action_request)
    preview = build_approval_preview(tool_name, tool_args, assistant_id) if tool_name else None

    body_lines = []
    if preview:
        body_lines.append(f"[bold]{preview.title}[/bold]")
        body_lines.extend(preview.details)
        if preview.error:
            body_lines.append(f"[red]{preview.error}[/red]")
        if description and description != "No description available":
            body_lines.append("")
            body_lines.append(description)
    else:
        body_lines.append(description)

    # Display action info first
    console.print()
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
    console.print()

    options = ["approve", "reject"]
    selected = 0  # Start with approve selected

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            # Initial render flag
            first_render = True

            while True:
                if not first_render:
                    # Move cursor back to start of menu (up 2 lines, then to start of line)
                    sys.stdout.write("\033[2A\r")

                first_render = False

                # Display options vertically with ANSI color codes
                for i, option in enumerate(options):
                    sys.stdout.write("\r\033[K")  # Clear line from cursor to end

                    if i == selected:
                        if option == "approve":
                            # Green bold with filled checkbox
                            sys.stdout.write("\033[1;32m☑ Approve\033[0m\n")
                        else:
                            # Red bold with filled checkbox
                            sys.stdout.write("\033[1;31m☑ Reject\033[0m\n")
                    elif option == "approve":
                        # Dim with empty checkbox
                        sys.stdout.write("\033[2m☐ Approve\033[0m\n")
                    else:
                        # Dim with empty checkbox
                        sys.stdout.write("\033[2m☐ Reject\033[0m\n")

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
                elif char == "\r" or char == "\n":  # Enter
                    sys.stdout.write("\033[1B\n")  # Move down past the menu
                    break
                elif char == "\x03":  # Ctrl+C
                    sys.stdout.write("\033[1B\n")  # Move down past the menu
                    raise KeyboardInterrupt
                elif char.lower() == "a":
                    selected = 0
                    sys.stdout.write("\033[1B\n")  # Move down past the menu
                    break
                elif char.lower() == "r":
                    selected = 1
                    sys.stdout.write("\033[1B\n")  # Move down past the menu
                    break

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (termios.error, AttributeError):
        # Fallback for non-Unix systems
        console.print("  ☐ (A)pprove  (default)")
        console.print("  ☐ (R)eject")
        choice = input("\nChoice (A/R, default=Approve): ").strip().lower()
        if choice == "r" or choice == "reject":
            selected = 1
        else:
            selected = 0

    console.print()

    # Return decision based on selection
    if selected == 0:
        return {"type": "approve"}
    return {"type": "reject", "message": "User rejected the command"}


def execute_task(
    user_input: str,
    agent,
    assistant_id: str | None,
    session_state,
    token_tracker: TokenTracker | None = None,
):
    """Execute any task by passing it directly to the AI agent."""
    console.print()

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

    config = {
        "configurable": {"thread_id": "main"},
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
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "write_todos": "📋",
    }

    file_op_tracker = FileOpTracker(assistant_id=assistant_id)

    # Track which tool calls we've displayed to avoid duplicates
    displayed_tool_ids = set()
    # Buffer partial tool-call chunks keyed by streaming index
    tool_call_buffers: dict[str | int, dict] = {}
    # Buffer assistant text so we can render complete markdown segments
    pending_text = ""
    # Track if we're buffering a summary message
    summary_mode = False
    summary_buffer = ""

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

    def flush_summary_buffer() -> None:
        """Render any buffered summary panel output."""
        nonlocal summary_mode, summary_buffer, spinner_active, has_responded
        if not summary_mode or not summary_buffer.strip():
            summary_mode = False
            summary_buffer = ""
            return
        if spinner_active:
            status.stop()
            spinner_active = False
        if not has_responded:
            console.print("●", style=COLORS["agent"], markup=False, end=" ")
            has_responded = True
        console.print()
        render_summary_panel(summary_buffer.strip())
        console.print()
        summary_mode = False
        summary_buffer = ""

    # Stream input - may need to loop if there are interrupts
    stream_input = {"messages": [{"role": "user", "content": final_input}]}

    try:
        while True:
            interrupt_occurred = False
            hitl_response = None
            suppress_resumed_output = False
            hitl_request = None

            for chunk in agent.stream(
                stream_input,
                stream_mode=["messages", "updates"],  # Dual-mode for HITL support
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                # Unpack chunk - with subgraphs=True and dual-mode, it's (namespace, stream_mode, data)
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue

                namespace, current_stream_mode, data = chunk

                # Handle UPDATES stream - for interrupts and todos
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for interrupts - just capture the data, don't handle yet
                    if "__interrupt__" in data:
                        interrupt_data = data["__interrupt__"]
                        if interrupt_data:
                            interrupt_obj = (
                                interrupt_data[0]
                                if isinstance(interrupt_data, tuple)
                                else interrupt_data
                            )
                            hitl_request = (
                                interrupt_obj.value
                                if hasattr(interrupt_obj, "value")
                                else interrupt_obj
                            )
                            interrupt_occurred = True

                    # Extract chunk_data from updates for todo checking
                    chunk_data = list(data.values())[0] if data else None
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

                    message, metadata = data

                    if isinstance(message, ToolMessage):
                        # Tool results are sent to the agent, not displayed to users
                        # Exception: show shell command errors to help with debugging
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(message.content)
                        record = file_op_tracker.complete_with_message(message)

                        if tool_name == "shell" and tool_status != "success":
                            flush_summary_buffer()
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
                                flush_summary_buffer()
                                flush_text_buffer(final=True)
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                console.print()
                                console.print(tool_content, style="red", markup=False)
                                console.print()

                        if record:
                            flush_summary_buffer()
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
                                if summary_mode:
                                    summary_buffer += text
                                    continue

                                if is_summary_message(text) or is_summary_message(
                                    pending_text + text
                                ):
                                    if pending_text:
                                        summary_buffer += pending_text
                                        pending_text = ""
                                    summary_mode = True
                                    summary_buffer += text
                                    continue

                                pending_text += text

                        # Handle reasoning blocks
                        elif block_type == "reasoning":
                            flush_summary_buffer()
                            flush_text_buffer(final=True)
                            reasoning = block.get("reasoning", "")
                            if reasoning:
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                # Could display reasoning differently if desired
                                # For now, skip it or handle minimally

                        # Handle tool call chunks
                        elif block_type == "tool_call_chunk":
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
                            if buffer_id is not None and buffer_id in displayed_tool_ids:
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

                            flush_summary_buffer()
                            flush_text_buffer(final=True)
                            if buffer_id is not None:
                                displayed_tool_ids.add(buffer_id)
                                file_op_tracker.start_operation(buffer_name, parsed_args, buffer_id)
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

                            if not spinner_active:
                                status.start()
                                spinner_active = True

                    if getattr(message, "chunk_position", None) == "last":
                        flush_summary_buffer()
                        flush_text_buffer(final=True)

            # After streaming loop - handle interrupt if it occurred
            flush_summary_buffer()
            flush_text_buffer(final=True)

            # Handle human-in-the-loop after stream completes
            if interrupt_occurred and hitl_request:
                # Check if auto-approve is enabled
                if session_state.auto_approve:
                    # Auto-approve all commands without prompting
                    decisions = []
                    for action_request in hitl_request.get("action_requests", []):
                        # Show what's being auto-approved (brief, dim message)
                        if spinner_active:
                            status.stop()
                            spinner_active = False

                        description = action_request.get("description", "tool action")
                        console.print()
                        console.print(f"  [dim]⚡ {description}[/dim]")

                        decisions.append({"type": "approve"})

                    hitl_response = {"decisions": decisions}

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
                    for action_request in hitl_request.get("action_requests", []):
                        decision = prompt_for_tool_approval(action_request, assistant_id)
                        decisions.append(decision)

                    suppress_resumed_output = any(
                        decision.get("type") == "reject" for decision in decisions
                    )
                    hitl_response = {"decisions": decisions}

            if interrupt_occurred and hitl_response:
                if suppress_resumed_output:
                    if spinner_active:
                        status.stop()
                        spinner_active = False

                    console.print("\n[yellow]Command rejected.[/yellow]", style="bold")
                    console.print("Tell the agent what you'd like to do differently.\n")
                    return

                # Resume the agent with the human decision
                stream_input = Command(resume=hitl_response)
                # Continue the while loop to restream
            else:
                # No interrupt, break out of while loop
                break

    except KeyboardInterrupt:
        # User pressed Ctrl+C - clean up and exit gracefully
        if spinner_active:
            status.stop()
        console.print("\n[yellow]Interrupted by user[/yellow]\n")

        # Inform the agent in background thread (non-blocking)
        def notify_agent():
            try:
                agent.update_state(
                    config=config,
                    values={
                        "messages": [
                            HumanMessage(
                                content="[User interrupted the previous request with Ctrl+C]"
                            )
                        ]
                    },
                )
            except Exception:
                pass

        threading.Thread(target=notify_agent, daemon=True).start()
        return

    if spinner_active:
        status.stop()

    if has_responded:
        console.print()

        # Track token usage (display only via /tokens command)
        if token_tracker and (captured_input_tokens or captured_output_tokens):
            token_tracker.add(captured_input_tokens, captured_output_tokens)

        console.print()
