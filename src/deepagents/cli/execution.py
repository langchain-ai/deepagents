"""Task execution and streaming logic for the CLI."""
import sys
import tty
import termios
import json
import threading

from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from rich.panel import Panel
from rich import box

from .config import console, COLORS
from .ui import render_todo_list, format_tool_message_content, TokenTracker
from .input import parse_file_mentions


def prompt_for_shell_approval(action_request: dict) -> dict:
    """Prompt user to approve/reject a shell command with arrow key navigation."""
    # Display command info first
    console.print()
    console.print(Panel(
        f"[bold yellow]⚠️  Shell Command Requires Approval[/bold yellow]\n\n"
        f"{action_request.get('description', 'No description available')}",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 1)
    ))
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
                    sys.stdout.write('\033[2A\r')

                first_render = False

                # Display options vertically with ANSI color codes
                for i, option in enumerate(options):
                    sys.stdout.write('\r\033[K')  # Clear line from cursor to end

                    if i == selected:
                        if option == "approve":
                            # Green bold with filled checkbox
                            sys.stdout.write('\033[1;32m☑ Approve\033[0m\n')
                        else:
                            # Red bold with filled checkbox
                            sys.stdout.write('\033[1;31m☑ Reject\033[0m\n')
                    else:
                        if option == "approve":
                            # Dim with empty checkbox
                            sys.stdout.write('\033[2m☐ Approve\033[0m\n')
                        else:
                            # Dim with empty checkbox
                            sys.stdout.write('\033[2m☐ Reject\033[0m\n')

                sys.stdout.flush()

                # Read key
                char = sys.stdin.read(1)

                if char == '\x1b':  # ESC sequence (arrow keys)
                    next1 = sys.stdin.read(1)
                    next2 = sys.stdin.read(1)
                    if next1 == '[':
                        if next2 == 'B':  # Down arrow
                            selected = (selected + 1) % len(options)
                        elif next2 == 'A':  # Up arrow
                            selected = (selected - 1) % len(options)
                elif char == '\r' or char == '\n':  # Enter
                    sys.stdout.write('\033[1B\n')  # Move down past the menu
                    break
                elif char == '\x03':  # Ctrl+C
                    sys.stdout.write('\033[1B\n')  # Move down past the menu
                    raise KeyboardInterrupt()
                elif char.lower() == 'a':
                    selected = 0
                    sys.stdout.write('\033[1B\n')  # Move down past the menu
                    break
                elif char.lower() == 'r':
                    selected = 1
                    sys.stdout.write('\033[1B\n')  # Move down past the menu
                    break

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (termios.error, AttributeError):
        # Fallback for non-Unix systems
        console.print("  ☐ (A)pprove  (default)")
        console.print("  ☐ (R)eject")
        choice = input("\nChoice (A/R, default=Approve): ").strip().lower()
        if choice == 'r' or choice == 'reject':
            selected = 1
        else:
            selected = 0

    console.print()

    # Return decision based on selection
    if selected == 0:
        return {"type": "approve"}
    else:
        return {"type": "reject", "message": "User rejected the command"}


def execute_task(user_input: str, agent, assistant_id: str | None, session_state, token_tracker: TokenTracker | None = None):
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
                context_parts.append(f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```")
            except Exception as e:
                context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")

        final_input = "\n".join(context_parts)
    else:
        final_input = prompt_text

    config = {
        "configurable": {"thread_id": "main"},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {}
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

    # Stream input - may need to loop if there are interrupts
    stream_input = {"messages": [{"role": "user", "content": final_input}]}

    try:
        while True:
            interrupt_occurred = False
            hitl_response = None
            suppress_resumed_output = False

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

                    # Check for interrupts
                    if "__interrupt__" in data:
                        interrupt_data = data["__interrupt__"]
                        if interrupt_data:
                            interrupt_obj = interrupt_data[0] if isinstance(interrupt_data, tuple) else interrupt_data
                            hitl_request = interrupt_obj.value if hasattr(interrupt_obj, 'value') else interrupt_obj

                            # Check if auto-approve is enabled
                            if session_state.auto_approve:
                                # Auto-approve all commands without prompting
                                decisions = []
                                for action_request in hitl_request.get("action_requests", []):
                                    # Show what's being auto-approved (brief, dim message)
                                    if spinner_active:
                                        status.stop()
                                        spinner_active = False

                                    description = action_request.get('description', 'tool action')
                                    console.print()
                                    console.print(f"  [dim]⚡ {description}[/dim]")

                                    decisions.append({"type": "approve"})

                                hitl_response = {"decisions": decisions}
                                interrupt_occurred = True

                                # Restart spinner for continuation
                                if not spinner_active:
                                    status.start()
                                    spinner_active = True

                                break
                            else:
                                # Normal HITL flow - stop spinner and prompt user
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False

                                # Handle human-in-the-loop approval
                                decisions = []
                                for action_request in hitl_request.get("action_requests", []):
                                    decision = prompt_for_shell_approval(action_request)
                                    decisions.append(decision)

                                suppress_resumed_output = any(decision.get("type") == "reject" for decision in decisions)
                                hitl_response = {"decisions": decisions}
                                interrupt_occurred = True
                                break

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

                        if tool_name == "shell" and tool_status != "success":
                            tool_content = format_tool_message_content(message.content)
                            if tool_content:
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                console.print()
                                console.print(tool_content, style="red", markup=False)
                                console.print()

                        # For all other tools (web_search, http_request, etc.),
                        # results are hidden from user - agent will process and respond
                        continue

                    # Check if this is an AIMessageChunk
                    if not hasattr(message, 'content_blocks'):
                        # Fallback for messages without content_blocks
                        continue

                    # Extract token usage if available
                    if token_tracker and hasattr(message, 'usage_metadata'):
                        usage = message.usage_metadata
                        if usage:
                            input_toks = usage.get('input_tokens', 0)
                            output_toks = usage.get('output_tokens', 0)
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
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False

                                if not has_responded:
                                    console.print("● ", style=COLORS["agent"], end="", markup=False)
                                    has_responded = True

                                # Print the text chunk directly (no cumulative diffing needed)
                                console.print(text, style=COLORS["agent"], end="", markup=False)

                        # Handle reasoning blocks
                        elif block_type == "reasoning":
                            reasoning = block.get("reasoning", "")
                            if reasoning:
                                if spinner_active:
                                    status.stop()
                                    spinner_active = False
                                # Could display reasoning differently if desired
                                # For now, skip it or handle minimally

                        # Handle tool call chunks
                        elif block_type == "tool_call_chunk":
                            tool_name = block.get("name")
                            tool_args = block.get("args", "")
                            tool_id = block.get("id")

                            # Only display when we have a complete tool call (name is present)
                            if tool_name:
                                icon = tool_icons.get(tool_name, "🔧")

                                if spinner_active:
                                    status.stop()

                                # Display tool call
                                if has_responded:
                                    console.print()  # New line after text

                                # Try to parse args if it's a string
                                try:
                                    if isinstance(tool_args, str) and tool_args:
                                        parsed_args = json.loads(tool_args)
                                        args_str = ", ".join(
                                            f"{k}={truncate_value(str(v), 50)}"
                                            for k, v in parsed_args.items()
                                        )
                                    else:
                                        args_str = str(tool_args)
                                except:
                                    args_str = str(tool_args)

                                console.print(f"  {icon} {tool_name}({args_str})", style=f"dim {COLORS['tool']}", markup=False)

                                if spinner_active:
                                    status.start()

            # After streaming loop - handle interrupt if it occurred
            if interrupt_occurred and hitl_response:
                if suppress_resumed_output:
                    if spinner_active:
                        status.stop()
                        spinner_active = False

                    console.print("\nCommand rejected. Returning to prompt.\n", style=COLORS["dim"])

                    # Resume agent in background thread to properly update graph state
                    # without blocking the user
                    def resume_after_rejection():
                        try:
                            agent.invoke(Command(resume=hitl_response), config=config)
                        except Exception:
                            pass  # Silently ignore errors

                    threading.Thread(target=resume_after_rejection, daemon=True).start()
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
                        "messages": [HumanMessage(content="[User interrupted the previous request with Ctrl+C]")]
                    }
                )
            except Exception:
                pass

        threading.Thread(target=notify_agent, daemon=True).start()
        return

    if spinner_active:
        status.stop()

    if has_responded:
        console.print()

        # Display token usage if available
        if token_tracker and (captured_input_tokens or captured_output_tokens):
            token_tracker.add(captured_input_tokens, captured_output_tokens)
            token_tracker.display_last()

        console.print()


# Import truncate_value for use in tool arg formatting
from .ui import truncate_value
