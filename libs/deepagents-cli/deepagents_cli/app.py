"""Textual UI application for deepagents-cli."""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.widgets import Static

from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    SystemMessage,
    UserMessage,
)
from deepagents_cli.widgets.status import StatusBar
from deepagents_cli.widgets.welcome import WelcomeBanner

if TYPE_CHECKING:
    from langgraph.pregel import Pregel
    from textual.app import ComposeResult

    from deepagents_cli.textual_adapter import TextualUIAdapter


class TextualSessionState:
    """Session state for the Textual app."""

    def __init__(self, *, auto_approve: bool = False) -> None:
        """Initialize session state."""
        self.auto_approve = auto_approve
        self.thread_id = str(uuid.uuid4())


class DeepAgentsApp(App):
    """Main Textual application for deepagents-cli."""

    TITLE = "DeepAgents"
    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "quit_or_clear", "Quit", show=False),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        # Note: No escape binding - let child widgets handle it
    ]

    def __init__(
        self,
        *,
        agent: Pregel | None = None,
        assistant_id: str | None = None,
        backend: Any = None,  # noqa: ANN401  # CompositeBackend
        auto_approve: bool = False,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DeepAgents application.

        Args:
            agent: Pre-configured LangGraph agent (optional for standalone mode)
            assistant_id: Agent identifier for memory storage
            backend: Backend for file operations
            auto_approve: Whether to start with auto-approve enabled
            cwd: Current working directory to display
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._agent = agent
        self._assistant_id = assistant_id
        self._backend = backend
        self._auto_approve = auto_approve
        self._cwd = str(cwd) if cwd else str(Path.cwd())
        self._status_bar: StatusBar | None = None
        self._chat_input: ChatInput | None = None
        self._quit_pending = False
        self._session_state: TextualSessionState | None = None
        self._ui_adapter: TextualUIAdapter | None = None
        self._pending_approval: asyncio.Future | None = None
        self._pending_approval_widget: Any = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        # Main chat area with scrollable messages
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(id="welcome-banner")
            yield Static(id="messages")

        # Input area
        yield ChatInput(id="input-area")

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
        self._session_state = TextualSessionState(auto_approve=self._auto_approve)

        # Create UI adapter if agent is provided
        if self._agent:
            from deepagents_cli.textual_adapter import TextualUIAdapter

            self._ui_adapter = TextualUIAdapter(
                mount_message=self._mount_message,
                update_status=self._update_status,
                request_approval=self._request_approval,
                on_auto_approve_enabled=self._on_auto_approve_enabled,
                scroll_to_bottom=self._scroll_chat_to_bottom,
            )

        # Focus the input
        self._chat_input.focus_input()

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message."""
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll the chat area to the bottom."""
        try:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
        except LookupError:
            pass

    def _request_approval(
        self, action_request: Any, assistant_id: str | None  # noqa: ANN401
    ) -> asyncio.Future:
        """Request user approval using a modal screen.

        Returns a Future that resolves to the user's decision.
        Uses ModalScreen with a result callback (no worker required).
        """
        from deepagents_cli.widgets.approval import ApprovalScreen

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()

        def handle_result(decision: dict[str, str] | None) -> None:
            if result_future.done():
                return
            if decision is None:
                result_future.set_result({"type": "reject"})
            else:
                result_future.set_result(decision)

        try:
            screen = ApprovalScreen(
                action_request=action_request,
                assistant_id=assistant_id,
            )
            self.push_screen(screen, callback=handle_result)
        except Exception as exc:
            if not result_future.done():
                result_future.set_exception(exc)

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
            else:
                await self._mount_message(SystemMessage("Command completed (no output)"))

            if result.returncode != 0:
                await self._mount_message(
                    ErrorMessage(f"Exit code: {result.returncode}")
                )

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
            await self._mount_message(SystemMessage("Commands: /quit, /clear, /tokens, /help"))
        elif cmd == "/clear":
            await self._clear_messages()
        elif cmd == "/tokens":
            await self._mount_message(UserMessage(command))
            await self._mount_message(SystemMessage("Token tracking not yet implemented"))
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
            from deepagents_cli.textual_adapter import execute_task_textual

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
        else:
            await self._mount_message(
                SystemMessage("Agent not configured. Run with --agent flag or use standalone mode.")
            )

    async def _mount_message(self, widget: Static) -> None:
        """Mount a message widget to the messages area.

        Args:
            widget: The message widget to mount
        """
        try:
            messages = self.query_one("#messages", Static)
            await messages.mount(widget)
            # Scroll to bottom
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
        except LookupError:
            pass

    async def _clear_messages(self) -> None:
        """Clear the messages area."""
        try:
            messages = self.query_one("#messages", Static)
            await messages.remove_children()
        except LookupError:
            # Widget not found - can happen during shutdown
            pass

    def action_quit_or_clear(self) -> None:
        """Handle Ctrl+C - clear input or quit on double press."""
        if self._chat_input and self._chat_input.value:
            # Clear the input
            self._chat_input.value = ""
            self._quit_pending = False
        elif self._quit_pending:
            # Second Ctrl+C - quit
            self.exit()
        else:
            # First Ctrl+C with empty input - show hint
            self._quit_pending = True
            self.notify("Press Ctrl+C again to quit", timeout=3)

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


def run_textual_app(
    *,
    agent: Pregel | None = None,
    assistant_id: str | None = None,
    backend: Any = None,  # noqa: ANN401  # CompositeBackend
    auto_approve: bool = False,
    cwd: str | Path | None = None,
) -> None:
    """Run the Textual application.

    Args:
        agent: Pre-configured LangGraph agent (optional)
        assistant_id: Agent identifier for memory storage
        backend: Backend for file operations
        auto_approve: Whether to start with auto-approve enabled
        cwd: Current working directory to display
    """
    app = DeepAgentsApp(
        agent=agent,
        assistant_id=assistant_id,
        backend=backend,
        auto_approve=auto_approve,
        cwd=cwd,
    )
    app.run()


if __name__ == "__main__":
    run_textual_app()
