"""Textual UI application for deepagents-cli."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.events import MouseUp  # noqa: TC002 - used in type annotation
from textual.widgets import Static  # noqa: TC002 - used at runtime

from deepagents_cli.clipboard import copy_selection_to_clipboard
from deepagents_cli.textual_adapter import TextualUIAdapter, execute_task_textual
from deepagents_cli.widgets.approval import ApprovalMenu
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.loading import LoadingWidget
from deepagents_cli.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    SystemMessage,
    ToolCallMessage,
    UserMessage,
)
from deepagents_cli.widgets.status import StatusBar
from deepagents_cli.widgets.welcome import WelcomeBanner

if TYPE_CHECKING:
    from langgraph.pregel import Pregel
    from textual.app import ComposeResult
    from textual.worker import Worker


class TextualTokenTracker:
    """Token tracker that updates the status bar."""

    def __init__(self, update_callback: callable) -> None:
        """Initialize with a callback to update the display."""
        self._update_callback = update_callback
        self.current_context = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:  # noqa: ARG002
        """Update token count from a response."""
        self.current_context = input_tokens
        self._update_callback(input_tokens)

    def reset(self) -> None:
        """Reset token count."""
        self.current_context = 0
        self._update_callback(0)


class TextualSessionState:
    """Session state for the Textual app."""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        thread_id: str | None = None,
    ) -> None:
        """Initialize session state.

        Args:
            auto_approve: Whether to auto-approve tool calls
            thread_id: Optional thread ID (generates 8-char hex if not provided)
        """
        self.auto_approve = auto_approve
        self.thread_id = thread_id if thread_id else uuid.uuid4().hex[:8]

    def reset_thread(self) -> str:
        """Reset to a new thread. Returns the new thread_id."""
        self.thread_id = uuid.uuid4().hex[:8]
        return self.thread_id


class DeepAgentsApp(App):
    """Main Textual application for deepagents-cli."""

    TITLE = "DeepAgents"
    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False

    # Slow down scroll speed (default is 3 lines per scroll event)
    # Using 0.25 to require 4 scroll events per line - very smooth
    SCROLL_SENSITIVITY_Y = 0.25

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        Binding(
            "shift+tab", "toggle_auto_approve", "Toggle Auto-Approve", show=False, priority=True
        ),
        Binding("ctrl+o", "toggle_tool_output", "Toggle Tool Output", show=False),
        # Approval menu keys (handled at App level for reliability)
        Binding("up", "approval_up", "Up", show=False),
        Binding("k", "approval_up", "Up", show=False),
        Binding("down", "approval_down", "Down", show=False),
        Binding("j", "approval_down", "Down", show=False),
        Binding("enter", "approval_select", "Select", show=False),
        Binding("y", "approval_yes", "Yes", show=False),
        Binding("1", "approval_yes", "Yes", show=False),
        Binding("n", "approval_no", "No", show=False),
        Binding("2", "approval_no", "No", show=False),
        Binding("a", "approval_auto", "Auto", show=False),
        Binding("3", "approval_auto", "Auto", show=False),
    ]

    def __init__(
        self,
        *,
        agent: Pregel | None = None,
        assistant_id: str | None = None,
        backend: Any = None,  # noqa: ANN401  # CompositeBackend
        auto_approve: bool = False,
        cwd: str | Path | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepAgents application.

        Args:
            agent: Pre-configured LangGraph agent (optional for standalone mode)
            assistant_id: Agent identifier for memory storage
            backend: Backend for file operations
            auto_approve: Whether to start with auto-approve enabled
            cwd: Current working directory to display
            thread_id: Optional thread ID for session persistence
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._agent = agent
        self._assistant_id = assistant_id
        self._backend = backend
        self._auto_approve = auto_approve
        self._cwd = str(cwd) if cwd else str(Path.cwd())
        # Avoid collision with App._thread_id
        self._lc_thread_id = thread_id
        self._status_bar: StatusBar | None = None
        self._chat_input: ChatInput | None = None
        self._quit_pending = False
        self._session_state: TextualSessionState | None = None
        self._ui_adapter: TextualUIAdapter | None = None
        self._pending_approval: asyncio.Future | None = None
        self._pending_approval_widget: Any = None
        # Agent task tracking for interruption
        self._agent_worker: Worker[None] | None = None
        self._agent_running = False
        self._loading_widget: LoadingWidget | None = None
        self._token_tracker: TextualTokenTracker | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        # Main chat area with scrollable messages
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(id="welcome-banner")
            yield Container(id="messages")  # Container can have children mounted

        # Bottom app container - holds either ChatInput OR ApprovalMenu (swapped)
        # This is OUTSIDE VerticalScroll so arrow keys work in approval
        with Container(id="bottom-app-container"):
            yield ChatInput(cwd=self._cwd, id="input-area")

        # Status bar at bottom
        yield StatusBar(cwd=self._cwd, id="status-bar")

    async def on_mount(self) -> None:
        """Initialize components after mount."""
        self._status_bar = self.query_one("#status-bar", StatusBar)
        self._chat_input = self.query_one("#input-area", ChatInput)

        # Set initial auto-approve state
        if self._auto_approve:
            self._status_bar.set_auto_approve(enabled=True)

        # Create session state
        self._session_state = TextualSessionState(
            auto_approve=self._auto_approve,
            thread_id=self._lc_thread_id,
        )

        # Create token tracker that updates status bar
        self._token_tracker = TextualTokenTracker(self._update_tokens)

        # Create UI adapter if agent is provided
        if self._agent:
            self._ui_adapter = TextualUIAdapter(
                mount_message=self._mount_message,
                update_status=self._update_status,
                request_approval=self._request_approval,
                on_auto_approve_enabled=self._on_auto_approve_enabled,
                scroll_to_bottom=self._scroll_chat_to_bottom,
            )
            self._ui_adapter.set_token_tracker(self._token_tracker)

        # Focus the input (autocomplete is now built into ChatInput)
        self._chat_input.focus_input()

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message."""
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def _update_tokens(self, count: int) -> None:
        """Update the token count in status bar."""
        if self._status_bar:
            self._status_bar.set_tokens(count)

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll the chat area to the bottom.

        Uses anchor() for smoother streaming - keeps scroll locked to bottom
        as new content is added without causing visual jumps.
        """
        try:
            chat = self.query_one("#chat", VerticalScroll)
            # anchor() locks scroll to bottom and auto-scrolls as content grows
            # Much smoother than calling scroll_end() on every chunk
            chat.anchor()
        except NoMatches:
            pass

    async def _request_approval(
        self,
        action_request: Any,  # noqa: ANN401
        assistant_id: str | None,
    ) -> asyncio.Future:
        """Request user approval inline in the messages area.

        Returns a Future that resolves to the user's decision.
        Mounts ApprovalMenu in the messages area (inline with chat).
        ChatInput stays visible - user can still see it.

        If another approval is already pending, queue this one.
        """
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()

        # If there's already a pending approval, wait for it to complete first
        if self._pending_approval_widget is not None:
            while self._pending_approval_widget is not None:  # noqa: ASYNC110
                await asyncio.sleep(0.1)

        # Create menu with unique ID to avoid conflicts
        unique_id = f"approval-menu-{uuid.uuid4().hex[:8]}"
        menu = ApprovalMenu(action_request, assistant_id, id=unique_id)
        menu.set_future(result_future)

        # Store reference
        self._pending_approval_widget = menu

        # Pause the loading spinner during approval
        if self._loading_widget:
            self._loading_widget.pause("Awaiting decision")

        # Update status to show we're waiting for approval
        self._update_status("Waiting for approval...")

        # Mount approval inline in messages area (not replacing ChatInput)
        try:
            messages = self.query_one("#messages", Container)
            await messages.mount(menu)
            self._scroll_chat_to_bottom()
            # Focus approval menu
            self.call_after_refresh(menu.focus)
        except Exception as e:  # noqa: BLE001
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.set_exception(e)

        return result_future

    def _on_auto_approve_enabled(self) -> None:
        """Callback when auto-approve mode is enabled via HITL."""
        self._auto_approve = True
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=True)
        if self._session_state:
            self._session_state.auto_approve = True

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle submitted input from ChatInput widget."""
        value = event.value
        mode = event.mode

        # Reset quit pending state on any input
        self._quit_pending = False

        # Handle different modes
        if mode == "bash":
            # Bash command - strip the ! prefix
            await self._handle_bash_command(value.removeprefix("!"))
        elif mode == "command":
            # Slash command
            await self._handle_command(value)
        else:
            # Normal message - will be sent to agent
            await self._handle_user_message(value)

    def on_chat_input_mode_changed(self, event: ChatInput.ModeChanged) -> None:
        """Update status bar when input mode changes."""
        if self._status_bar:
            self._status_bar.set_mode(event.mode)

    async def on_approval_menu_decided(
        self,
        event: Any,  # noqa: ANN401, ARG002
    ) -> None:
        """Handle approval menu decision - remove from messages and refocus input."""
        # Remove ApprovalMenu using stored reference
        if self._pending_approval_widget:
            await self._pending_approval_widget.remove()
            self._pending_approval_widget = None

        # Resume the loading spinner after approval
        if self._loading_widget:
            self._loading_widget.resume()

        # Clear status message
        self._update_status("")

        # Refocus the chat input
        if self._chat_input:
            self.call_after_refresh(self._chat_input.focus_input)

    async def _handle_bash_command(self, command: str) -> None:
        """Handle a bash command (! prefix).

        Args:
            command: The bash command to execute
        """
        # Mount user message showing the bash command
        await self._mount_message(UserMessage(f"!{command}"))

        # Execute the bash command (shell=True is intentional for user-requested bash)
        try:
            result = await asyncio.to_thread(  # noqa: S604
                subprocess.run,
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self._cwd,
                timeout=60,
            )
            output = result.stdout.strip()
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr.strip()}"

            if output:
                # Display output as assistant message (uses markdown for code blocks)
                msg = AssistantMessage(f"```\n{output}\n```")
                await self._mount_message(msg)
                await msg.write_initial_content()
            else:
                await self._mount_message(SystemMessage("Command completed (no output)"))

            if result.returncode != 0:
                await self._mount_message(ErrorMessage(f"Exit code: {result.returncode}"))

            # Scroll to show the output
            self._scroll_chat_to_bottom()

        except subprocess.TimeoutExpired:
            await self._mount_message(ErrorMessage("Command timed out (60s limit)"))
        except OSError as e:
            await self._mount_message(ErrorMessage(str(e)))

    async def _handle_command(self, command: str) -> None:
        """Handle a slash command.

        Args:
            command: The slash command (including /)
        """
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/help":
            await self._mount_message(UserMessage(command))
            await self._mount_message(
                SystemMessage("Commands: /quit, /clear, /tokens, /threads, /help")
            )

        elif cmd == "/version":
            await self._mount_message(UserMessage(command))
            # Show CLI package version
            try:
                from deepagents_cli._version import __version__

                await self._mount_message(SystemMessage(f"deepagents version: {__version__}"))
            except Exception:
                await self._mount_message(SystemMessage("deepagents version: unknown"))
        elif cmd == "/clear":
            await self._clear_messages()
            # Reset thread to start fresh conversation
            if self._session_state:
                new_thread_id = self._session_state.reset_thread()
                await self._mount_message(SystemMessage(f"Started new session: {new_thread_id}"))
        elif cmd == "/threads":
            await self._mount_message(UserMessage(command))
            if self._session_state:
                await self._mount_message(
                    SystemMessage(f"Current session: {self._session_state.thread_id}")
                )
            else:
                await self._mount_message(SystemMessage("No active session"))
        elif cmd == "/tokens":
            await self._mount_message(UserMessage(command))
            if self._token_tracker and self._token_tracker.current_context > 0:
                count = self._token_tracker.current_context
                if count >= 1000:
                    formatted = f"{count / 1000:.1f}K"
                else:
                    formatted = str(count)
                await self._mount_message(SystemMessage(f"Current context: {formatted} tokens"))
            else:
                await self._mount_message(SystemMessage("No token usage yet"))
        else:
            await self._mount_message(UserMessage(command))
            await self._mount_message(SystemMessage(f"Unknown command: {cmd}"))

    async def _handle_user_message(self, message: str) -> None:
        """Handle a user message to send to the agent.

        Args:
            message: The user's message
        """
        # Mount the user message
        await self._mount_message(UserMessage(message))

        # Check if agent is available
        if self._agent and self._ui_adapter and self._session_state:
            # Show loading widget
            self._loading_widget = LoadingWidget("Thinking")
            await self._mount_message(self._loading_widget)
            self._agent_running = True

            # Disable cursor blink while agent is working
            if self._chat_input:
                self._chat_input.set_cursor_active(active=False)

            # Use run_worker to avoid blocking the main event loop
            # This allows the UI to remain responsive during agent execution
            self._agent_worker = self.run_worker(
                self._run_agent_task(message),
                exclusive=False,
            )
        else:
            await self._mount_message(
                SystemMessage("Agent not configured. Run with --agent flag or use standalone mode.")
            )

    async def _run_agent_task(self, message: str) -> None:
        """Run the agent task in a background worker.

        This runs in a worker thread so the main event loop stays responsive.
        """
        try:
            await execute_task_textual(
                user_input=message,
                agent=self._agent,
                assistant_id=self._assistant_id,
                session_state=self._session_state,
                adapter=self._ui_adapter,
                backend=self._backend,
            )
        except Exception as e:  # noqa: BLE001
            await self._mount_message(ErrorMessage(f"Agent error: {e}"))
        finally:
            # Clean up loading widget and agent state
            await self._cleanup_agent_task()

    async def _cleanup_agent_task(self) -> None:
        """Clean up after agent task completes or is cancelled."""
        self._agent_running = False
        self._agent_worker = None

        # Remove loading widget if present
        if self._loading_widget:
            with contextlib.suppress(Exception):
                await self._loading_widget.remove()
            self._loading_widget = None

        # Re-enable cursor blink now that agent is done
        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)

    async def _mount_message(self, widget: Static) -> None:
        """Mount a message widget to the messages area.

        Args:
            widget: The message widget to mount
        """
        try:
            messages = self.query_one("#messages", Container)
            await messages.mount(widget)
            # Scroll to bottom
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
        except NoMatches:
            pass

    async def _clear_messages(self) -> None:
        """Clear the messages area."""
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            # Widget not found - can happen during shutdown
            pass

    def action_quit_or_interrupt(self) -> None:
        """Handle Ctrl+C - interrupt agent, reject approval, or quit on double press.

        Priority order:
        1. If agent is running, interrupt it (preserve input)
        2. If approval menu is active, reject it
        3. If double press (quit_pending), quit
        4. Otherwise show quit hint
        """
        # If agent is running, interrupt it
        if self._agent_running and self._agent_worker:
            self._agent_worker.cancel()
            self._quit_pending = False
            return

        # If approval menu is active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()
            self._quit_pending = False
            return

        # Double Ctrl+C to quit
        if self._quit_pending:
            self.exit()
        else:
            self._quit_pending = True
            self.notify("Press Ctrl+C again to quit", timeout=3)

    def action_interrupt(self) -> None:
        """Handle escape key - interrupt agent or reject approval.

        This is the primary way to stop a running agent.
        """
        # If agent is running, interrupt it
        if self._agent_running and self._agent_worker:
            self._agent_worker.cancel()
            return

        # If approval menu is active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_quit_app(self) -> None:
        """Handle quit action (Ctrl+D)."""
        self.exit()

    def action_toggle_auto_approve(self) -> None:
        """Toggle auto-approve mode."""
        self._auto_approve = not self._auto_approve
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)
        if self._session_state:
            self._session_state.auto_approve = self._auto_approve

    def action_toggle_tool_output(self) -> None:
        """Toggle expand/collapse of the most recent tool output."""
        # Find all tool messages with output, get the most recent one
        try:
            tool_messages = list(self.query(ToolCallMessage))
            # Find ones with output, toggle the most recent
            for tool_msg in reversed(tool_messages):
                if tool_msg.has_output:
                    tool_msg.toggle_output()
                    return
        except Exception:
            pass

    # Approval menu action handlers (delegated from App-level bindings)
    # NOTE: These only activate when approval widget is pending AND input is not focused
    def action_approval_up(self) -> None:
        """Handle up arrow in approval menu."""
        # Only handle if approval is active (input handles its own up for history/completion)
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_up()

    def action_approval_down(self) -> None:
        """Handle down arrow in approval menu."""
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_move_down()

    def action_approval_select(self) -> None:
        """Handle enter in approval menu."""
        # Only handle if approval is active AND input is not focused
        if self._pending_approval_widget and not self._is_input_focused():
            self._pending_approval_widget.action_select()

    def _is_input_focused(self) -> bool:
        """Check if the chat input (or its text area) has focus."""
        if not self._chat_input:
            return False
        focused = self.focused
        if focused is None:
            return False
        # Check if focused widget is the text area inside chat input
        return focused.id == "chat-input" or focused in self._chat_input.walk_children()

    def action_approval_yes(self) -> None:
        """Handle yes/1 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_approve()

    def action_approval_no(self) -> None:
        """Handle no/2 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_approval_auto(self) -> None:
        """Handle auto/3 in approval menu."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_auto()

    def action_approval_escape(self) -> None:
        """Handle escape in approval menu - reject."""
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def on_mouse_up(self, event: MouseUp) -> None:  # noqa: ARG002
        """Copy selection to clipboard on mouse release."""
        copy_selection_to_clipboard(self)


async def run_textual_app(
    *,
    agent: Pregel | None = None,
    assistant_id: str | None = None,
    backend: Any = None,  # noqa: ANN401  # CompositeBackend
    auto_approve: bool = False,
    cwd: str | Path | None = None,
    thread_id: str | None = None,
) -> None:
    """Run the Textual application.

    Args:
        agent: Pre-configured LangGraph agent (optional)
        assistant_id: Agent identifier for memory storage
        backend: Backend for file operations
        auto_approve: Whether to start with auto-approve enabled
        cwd: Current working directory to display
        thread_id: Optional thread ID for session persistence
    """
    app = DeepAgentsApp(
        agent=agent,
        assistant_id=assistant_id,
        backend=backend,
        auto_approve=auto_approve,
        cwd=cwd,
        thread_id=thread_id,
    )
    await app.run_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_textual_app())
