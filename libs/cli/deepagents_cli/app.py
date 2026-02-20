"""Textual UI application for deepagents-cli."""

from __future__ import annotations

import asyncio
import logging
import os

# S404: subprocess is required for user-initiated shell commands via ! prefix
import subprocess  # noqa: S404
import uuid
import webbrowser
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rich.text import Text
from textual.app import App
from textual.binding import Binding, BindingType
from textual.containers import Container, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_cli.clipboard import copy_selection_to_clipboard
from deepagents_cli.config import (
    DOCS_URL,
    SHELL_TOOL_NAMES,
    CharsetMode,
    _detect_charset_mode,
    build_langsmith_thread_url,
    create_model,
    detect_provider,
    is_shell_command_allowed,
    settings,
)
from deepagents_cli.model_config import ModelSpec, save_recent_model
from deepagents_cli.textual_adapter import TextualUIAdapter, execute_task_textual
from deepagents_cli.widgets.approval import ApprovalMenu
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.loading import LoadingWidget
from deepagents_cli.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    ErrorMessage,
    QueuedUserMessage,
    ToolCallMessage,
    UserMessage,
)
from deepagents_cli.widgets.model_selector import ModelSelectorScreen
from deepagents_cli.widgets.status import StatusBar
from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen
from deepagents_cli.widgets.welcome import WelcomeBanner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents.backends import CompositeBackend
    from deepagents.backends.sandbox import SandboxBackendProtocol
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.pregel import Pregel
    from textual.app import ComposeResult
    from textual.events import Click, MouseUp, Paste, Resize
    from textual.scrollbar import ScrollUp
    from textual.widget import Widget
    from textual.worker import Worker

# iTerm2 Cursor Guide Workaround
# ===============================
# iTerm2's cursor guide (highlight cursor line) causes visual artifacts when
# Textual takes over the terminal in alternate screen mode. We disable it at
# module load and restore on exit. Both atexit and exit() override are used
# for defense-in-depth: atexit catches abnormal termination (SIGTERM, unhandled
# exceptions), while exit() ensures restoration before Textual's cleanup.

# Detection: check env vars AND that stderr is a TTY (avoids false positives
# when env vars are inherited but running in non-TTY context like CI)
_IS_ITERM = (
    (
        os.environ.get("LC_TERMINAL", "") == "iTerm2"
        or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
    )
    and hasattr(os, "isatty")
    and os.isatty(2)
)

# iTerm2 cursor guide escape sequences (OSC 1337)
# Format: OSC 1337 ; HighlightCursorLine=<yes|no> ST
# Where OSC = ESC ] (0x1b 0x5d) and ST = ESC \ (0x1b 0x5c)
_ITERM_CURSOR_GUIDE_OFF = "\x1b]1337;HighlightCursorLine=no\x1b\\"
_ITERM_CURSOR_GUIDE_ON = "\x1b]1337;HighlightCursorLine=yes\x1b\\"


def _format_token_count(count: int) -> str:
    """Format a token count into a human-readable short string.

    Args:
        count: Number of tokens.

    Returns:
        Formatted string like `"12.5K"`, `"1.2M"`, or `"500"`.
    """
    if count >= 1_000_000:  # noqa: PLR2004
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:  # noqa: PLR2004
        return f"{count / 1000:.1f}K"
    return str(count)


def _write_iterm_escape(sequence: str) -> None:
    """Write an iTerm2 escape sequence to stderr.

    Silently fails if the terminal is unavailable (redirected, closed, broken
    pipe). This is a cosmetic feature, so failures should never crash the app.
    """
    if not _IS_ITERM:
        return
    try:
        import sys

        if sys.__stderr__ is not None:
            sys.__stderr__.write(sequence)
            sys.__stderr__.flush()
    except OSError:
        # Terminal may be unavailable (redirected, closed, broken pipe)
        pass


# Disable cursor guide at module load (before Textual takes over)
_write_iterm_escape(_ITERM_CURSOR_GUIDE_OFF)

if _IS_ITERM:
    import atexit

    def _restore_cursor_guide() -> None:
        """Restore iTerm2 cursor guide on exit.

        Registered with atexit to ensure the cursor guide is re-enabled
        when the CLI exits, regardless of how the exit occurs.
        """
        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    atexit.register(_restore_cursor_guide)


InputMode = Literal["normal", "bash", "command"]


@dataclass(frozen=True, slots=True)
class QueuedMessage:
    """Represents a queued user message awaiting processing.

    Attributes:
        text: The message text content.
        mode: The input mode that determines message routing.
    """

    text: str
    mode: InputMode


class TextualTokenTracker:
    """Token tracker that updates the status bar."""

    def __init__(
        self,
        update_callback: Callable[[int], None],
        hide_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize with callbacks to update the display."""
        self._update_callback = update_callback
        self._hide_callback = hide_callback
        self.current_context = 0

    def add(self, total_tokens: int, _output_tokens: int = 0) -> None:
        """Update token count from a response.

        Args:
            total_tokens: Total context tokens (input + output from usage_metadata)
            _output_tokens: Unused, kept for backwards compatibility
        """
        self.current_context = total_tokens
        self._update_callback(self.current_context)

    def reset(self) -> None:
        """Reset token count."""
        self.current_context = 0
        self._update_callback(0)

    def hide(self) -> None:
        """Hide the token display (e.g., during streaming)."""
        if self._hide_callback:
            self._hide_callback()

    def show(self) -> None:
        """Show the token display with current value (e.g., after interrupt)."""
        self._update_callback(self.current_context)


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
        self.thread_id = thread_id or uuid.uuid4().hex[:8]

    def reset_thread(self) -> str:
        """Reset to a new thread.

        Returns:
            The new thread_id.
        """
        self.thread_id = uuid.uuid4().hex[:8]
        return self.thread_id


_COMMAND_URLS: dict[str, str] = {
    "/changelog": "https://github.com/langchain-ai/deepagents/blob/main/libs/cli/CHANGELOG.md",
    "/docs": DOCS_URL,
    "/feedback": "https://github.com/langchain-ai/deepagents/issues/new/choose",
}

# Prompt for /remember command - triggers agent to review conversation and update
# memory/skills
REMEMBER_PROMPT = """Review our conversation and capture valuable knowledge. Focus especially on **best practices** we discussed or discovered—these are the most important things to preserve.

## Step 1: Identify Best Practices and Key Learnings

Scan the conversation for:

### Best Practices (highest priority)
- **Patterns that worked well** - approaches, techniques, or solutions we found effective
- **Anti-patterns to avoid** - mistakes, gotchas, or approaches that caused problems
- **Quality standards** - criteria we established for good code, documentation, or processes
- **Decision rationale** - why we chose one approach over another

### Other Valuable Knowledge
- Coding conventions and style preferences
- Project architecture decisions
- Workflows and processes we developed
- Tools, libraries, or techniques worth remembering
- Feedback I gave about your behavior or outputs

## Step 2: Decide Where to Store Each Learning

For each best practice or learning, choose the right destination:

### → Memory (AGENTS.md) for preferences and guidelines
Use memory when the knowledge is:
- A preference or guideline (not a multi-step process)
- Something to always keep in mind
- A simple rule or pattern

**Global** (`~/.deepagents/agent/AGENTS.md`): Universal preferences across all projects
**Project** (`.deepagents/AGENTS.md`): Project-specific conventions and decisions

### → Skill for reusable workflows and methodologies
**Create a skill when** we developed:
- A multi-step process worth reusing
- A methodology for a specific type of task
- A workflow with best practices baked in
- A procedure that should be followed consistently

Skills are more powerful than memory entries because they can encode **how** to do something well, not just **what** to remember.

## Step 3: Create Skills for Significant Best Practices

If we established best practices around a workflow or process, capture them in a skill.

**Example:** If we discussed best practices for code review, create a `code-review` skill that encodes those practices into a reusable workflow.

### Skill Location
`~/.deepagents/agent/skills/<skill-name>/SKILL.md`

### Skill Structure
```
skill-name/
├── SKILL.md          (required - main instructions with best practices)
├── scripts/          (optional - executable code)
├── references/       (optional - detailed documentation)
└── assets/           (optional - templates, examples)
```

### SKILL.md Format
```markdown
---
name: skill-name
description: "What this skill does AND when to use it. Include triggers like 'when the user asks to X' or 'when working with Y'. This description determines when the skill activates."
---

# Skill Name

## Overview
Brief explanation of what this skill accomplishes.

## Best Practices
Capture the key best practices upfront:
- Best practice 1: explanation
- Best practice 2: explanation

## Process
Step-by-step instructions (imperative form):
1. First, do X
2. Then, do Y
3. Finally, do Z

## Common Pitfalls
- Pitfall to avoid and why
- Another anti-pattern we discovered
```

### Key Principles
1. **Encode best practices prominently** - Put them near the top so they guide the entire workflow
2. **Concise is key** - Only include non-obvious knowledge. Every paragraph should justify its token cost.
3. **Clear triggers** - The description determines when the skill activates. Be specific.
4. **Imperative form** - Write as commands: "Create a file" not "You should create a file"
5. **Include anti-patterns** - What NOT to do is often as valuable as what to do

## Step 4: Update Memory for Simpler Learnings

For preferences, guidelines, and simple rules that don't warrant a full skill:

```markdown
## Best Practices
- When doing X, always Y because Z
- Avoid A because it leads to B
```

Use `edit_file` to update existing files or `write_file` to create new ones.

## Step 5: Summarize Changes

List what you captured and where you stored it:
- Skills created (with key best practices encoded)
- Memory entries added (with location)
"""  # noqa: E501


class DeepAgentsApp(App):
    """Main Textual application for deepagents-cli."""

    TITLE = "Deep Agents"
    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False

    # Scroll speed (default is 3 lines per scroll event)
    SCROLL_SENSITIVITY_Y = 1.0

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+c", "quit_or_interrupt", "Quit/Interrupt", show=False),
        Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
        Binding("ctrl+t", "toggle_auto_approve", "Toggle Auto-Approve", show=False),
        Binding(
            "shift+tab",
            "toggle_auto_approve",
            "Toggle Auto-Approve",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+e",
            "toggle_tool_output",
            "Toggle Tool Output",
            show=False,
            priority=True,
        ),
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
        backend: CompositeBackend | None = None,
        auto_approve: bool = False,
        cwd: str | Path | None = None,
        thread_id: str | None = None,
        initial_prompt: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tools: list[Callable[..., Any] | dict[str, Any]] | None = None,
        sandbox: SandboxBackendProtocol | None = None,
        sandbox_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Deep Agents application.

        Args:
            agent: Pre-configured LangGraph agent (optional for standalone mode)
            assistant_id: Agent identifier for memory storage
            backend: Backend for file operations
            auto_approve: Whether to start with auto-approve enabled
            cwd: Current working directory to display
            thread_id: Optional thread ID for session persistence
            initial_prompt: Optional prompt to auto-submit when session starts
            checkpointer: Checkpointer for session persistence (enables model hot-swap)
            tools: Tools used to create the agent (for model hot-swap)
            sandbox: Sandbox backend (for model hot-swap)
            sandbox_type: Type of sandbox provider (for model hot-swap)
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
        self._initial_prompt = initial_prompt
        # Store for model hot-swap
        self._checkpointer = checkpointer
        self._tools = tools or []
        self._sandbox = sandbox
        self._sandbox_type = sandbox_type
        self._status_bar: StatusBar | None = None
        self._chat_input: ChatInput | None = None
        self._quit_pending = False
        self._session_state: TextualSessionState | None = None
        self._ui_adapter: TextualUIAdapter | None = None
        self._pending_approval_widget: Any = None
        # Agent task tracking for interruption
        self._agent_worker: Worker[None] | None = None
        self._agent_running = False
        self._loading_widget: LoadingWidget | None = None
        self._token_tracker: TextualTokenTracker | None = None
        # User message queue for sequential processing
        self._pending_messages: deque[QueuedMessage] = deque()
        self._queued_widgets: deque[QueuedUserMessage] = deque()
        self._processing_pending = False
        # Message virtualization store
        self._message_store = MessageStore()
        # Lazily imported here to avoid pulling image dependencies into
        # argument parsing paths.
        from deepagents_cli.input import ImageTracker

        self._image_tracker = ImageTracker()

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        Yields:
            UI components for the main chat area and status bar.
        """
        # Main chat area with scrollable messages
        # VerticalScroll tracks user scroll intent for better auto-scroll behavior
        with VerticalScroll(id="chat"):
            yield WelcomeBanner(thread_id=self._lc_thread_id, id="welcome-banner")
            yield Container(id="messages")
            with Container(id="bottom-app-container"):
                yield ChatInput(
                    cwd=self._cwd,
                    image_tracker=self._image_tracker,
                    id="input-area",
                )
            yield Static(id="chat-spacer")  # Fills remaining space below input

        # Status bar at bottom
        yield StatusBar(cwd=self._cwd, id="status-bar")

    async def on_mount(self) -> None:
        """Initialize components after mount."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            chat = self.query_one("#chat", VerticalScroll)
            chat.styles.scrollbar_size_vertical = 0

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
        self._token_tracker = TextualTokenTracker(
            self._update_tokens, self._hide_tokens
        )

        # Create UI adapter if agent is provided
        if self._agent:
            self._ui_adapter = TextualUIAdapter(
                mount_message=self._mount_message,
                update_status=self._update_status,
                request_approval=self._request_approval,
                on_auto_approve_enabled=self._on_auto_approve_enabled,
                scroll_to_bottom=self._scroll_chat_to_bottom,
                set_spinner=self._set_spinner,
                set_active_message=self._set_active_message,
                sync_message_content=self._sync_message_content,
            )
            self._ui_adapter.set_token_tracker(self._token_tracker)

        # Focus the input (autocomplete is now built into ChatInput)
        self._chat_input.focus_input()

        # Warn about missing optional tools (advisory only — never block startup)
        try:
            from deepagents_cli.main import (
                check_optional_tools,
                format_tool_warning_tui,
            )
        except ImportError:
            logger.warning(
                "Could not import optional tools checker; skipping tool warnings",
                exc_info=True,
            )
        else:
            try:
                for tool in check_optional_tools():
                    self.notify(
                        format_tool_warning_tui(tool),
                        severity="warning",
                        timeout=15,
                    )
            except Exception:
                logger.debug("Failed to check for optional tools", exc_info=True)

        # Size the spacer to fill remaining viewport below input
        self.call_after_refresh(self._size_initial_spacer)

        # Auto-submit initial prompt if provided via -m flag.
        # This check must come first because _lc_thread_id and _agent are
        # always set (even for brand-new sessions), so an elif after the
        # thread-history branch would never execute.
        if self._initial_prompt and self._initial_prompt.strip():
            # Use call_after_refresh to ensure UI is fully mounted before submitting
            # Capture value for closure to satisfy type checker
            prompt = self._initial_prompt
            self.call_after_refresh(
                lambda: asyncio.create_task(self._handle_user_message(prompt))
            )
        # Load thread history if resuming a session (no initial prompt)
        elif self._lc_thread_id and self._agent:
            self.call_after_refresh(
                lambda: asyncio.create_task(self._load_thread_history())
            )

    def on_resize(self, _event: Resize) -> None:
        """Handle terminal resize to recalculate layout."""
        try:
            self.query_one("#chat-spacer", Static)
            # Spacer exists, recalculate its height
            self.call_after_refresh(self._size_initial_spacer)
        except NoMatches:
            pass  # Spacer already removed, no action needed

    def on_scroll_up(self, _event: ScrollUp) -> None:
        """Handle scroll up to check if we need to hydrate older messages."""
        self._check_hydration_needed()

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message."""
        if self._status_bar:
            self._status_bar.set_status_message(message)

    def _update_tokens(self, count: int) -> None:
        """Update the token count in status bar."""
        if self._status_bar:
            self._status_bar.set_tokens(count)

    def _hide_tokens(self) -> None:
        """Hide the token display during streaming."""
        if self._status_bar:
            self._status_bar.hide_tokens()

    def _scroll_chat_to_bottom(self) -> None:
        """Scroll chat to bottom using sticky scroll pattern.

        Only scrolls if user is already at/near the bottom.
        This prevents dragging the user back if they've scrolled up to read.
        """
        chat = self.query_one("#chat", VerticalScroll)

        # Nothing to scroll if content fits in viewport
        if chat.max_scroll_y <= 0:
            return

        # Sticky scroll: only scroll to bottom if user is near the bottom
        # "Near" means within 100 pixels of the bottom (about 6-7 lines)
        distance_from_bottom = chat.max_scroll_y - chat.scroll_y
        if distance_from_bottom < 100:  # noqa: PLR2004  # Token count threshold
            chat.scroll_end(animate=False)

    def _check_hydration_needed(self) -> None:
        """Check if we need to hydrate messages from the store.

        Called when user scrolls up near the top of visible messages.
        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration check: #chat container not found")
            return

        scroll_y = chat.scroll_y
        viewport_height = chat.size.height

        if self._message_store.should_hydrate_above(scroll_y, viewport_height):
            self.call_later(self._hydrate_messages_above)

    async def _hydrate_messages_above(self) -> None:
        """Hydrate older messages when user scrolls near the top.

        This recreates widgets for archived messages and inserts them
        at the top of the messages container.
        """
        if not self._message_store.has_messages_above:
            return

        try:
            chat = self.query_one("#chat", VerticalScroll)
        except NoMatches:
            logger.debug("Skipping hydration: #chat not found")
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping hydration: #messages not found")
            return

        to_hydrate = self._message_store.get_messages_to_hydrate()
        if not to_hydrate:
            return

        old_scroll_y = chat.scroll_y
        first_child = (
            messages_container.children[0] if messages_container.children else None
        )

        # Build widgets in chronological order, then mount in reverse so
        # each is inserted before the previous first_child, resulting in
        # correct chronological order in the DOM.
        hydrated_count = 0
        hydrated_widgets: list[tuple] = []  # (widget, msg_data)
        for msg_data in to_hydrate:
            try:
                widget = msg_data.to_widget()
                hydrated_widgets.append((widget, msg_data))
            except Exception:
                logger.warning(
                    "Failed to create widget for message %s",
                    msg_data.id,
                    exc_info=True,
                )

        for widget, msg_data in reversed(hydrated_widgets):
            try:
                if first_child:
                    await messages_container.mount(widget, before=first_child)
                else:
                    await messages_container.mount(widget)
                first_child = widget
                hydrated_count += 1
                # Render Markdown content for hydrated assistant messages
                if isinstance(widget, AssistantMessage) and msg_data.content:
                    await widget.set_content(msg_data.content)
            except Exception:
                logger.warning(
                    "Failed to mount hydrated widget %s",
                    widget.id,
                    exc_info=True,
                )

        # Only update store for the number we actually mounted
        if hydrated_count > 0:
            self._message_store.mark_hydrated(hydrated_count)

        # Adjust scroll position to maintain the user's view.
        # Widget heights aren't known until after layout, so we use a
        # heuristic. A more accurate approach would measure actual heights
        # via call_after_refresh.
        estimated_height_per_message = 5  # terminal rows, rough estimate
        added_height = hydrated_count * estimated_height_per_message
        chat.scroll_y = old_scroll_y + added_height

    async def _mount_before_queued(self, container: Container, widget: Widget) -> None:
        """Mount a widget in the messages container, before any queued widgets.

        Queued-message widgets must stay at the bottom of the container so
        they remain visually anchored below the current agent response.
        This helper inserts `widget` just before the first queued widget,
        or appends at the end when the queue is empty.

        Args:
            container: The `#messages` container to mount into.
            widget: The widget to mount.
        """
        first_queued = self._queued_widgets[0] if self._queued_widgets else None
        if first_queued is not None and first_queued.parent is container:
            try:
                await container.mount(widget, before=first_queued)
            except Exception:
                logger.warning(
                    "Stale queued-widget reference; appending at end",
                    exc_info=True,
                )
            else:
                return
        await container.mount(widget)

    def _is_spinner_at_correct_position(self, container: Container) -> bool:
        """Check whether the loading spinner is already correctly positioned.

        The spinner should be immediately before the first queued widget, or
        at the very end of the container when the queue is empty.

        Args:
            container: The `#messages` container.

        Returns:
            `True` if the spinner is already in the correct position.
        """
        children = list(container.children)
        if not children or self._loading_widget not in children:
            return False

        if self._queued_widgets:
            first_queued = self._queued_widgets[0]
            if first_queued not in children:
                return False
            return children.index(self._loading_widget) == (
                children.index(first_queued) - 1
            )

        return children[-1] == self._loading_widget

    async def _set_spinner(self, status: str | None) -> None:
        """Show, update, or hide the loading spinner.

        Args:
            status: The status text to display (e.g., "Thinking", "Summarizing"),
                or `None` to hide the spinner.
        """
        if status is None:
            # Hide
            if self._loading_widget:
                await self._loading_widget.remove()
                self._loading_widget = None
            return

        messages = self.query_one("#messages", Container)

        if self._loading_widget is None:
            # Create new
            self._loading_widget = LoadingWidget(status)
            await self._mount_before_queued(messages, self._loading_widget)
        else:
            # Update existing
            self._loading_widget.set_status(status)
            # Reposition if not already at the correct location
            if not self._is_spinner_at_correct_position(messages):
                await self._loading_widget.remove()
                await self._mount_before_queued(messages, self._loading_widget)
        # NOTE: Don't call _scroll_chat_to_bottom() here - it would re-anchor
        # and drag user back to bottom if they've scrolled away during streaming

    def _size_initial_spacer(self) -> None:
        """Size the spacer to fill remaining viewport below input."""
        try:
            chat = self.query_one("#chat", VerticalScroll)
            welcome = self.query_one("#welcome-banner", WelcomeBanner)
            input_container = self.query_one("#bottom-app-container", Container)
            spacer = self.query_one("#chat-spacer", Static)
            content_height = welcome.size.height + input_container.size.height + 4
            spacer_height = chat.size.height - content_height
            spacer.styles.height = max(0, spacer_height)
        except NoMatches:
            # Spacer may have been removed already (e.g., when resuming a session)
            pass

    async def _remove_spacer(self) -> None:
        """Remove the initial spacer when first message is sent."""
        try:
            spacer = self.query_one("#chat-spacer", Static)
            await spacer.remove()
        except NoMatches:
            pass

    async def _request_approval(
        self,
        action_requests: Any,  # noqa: ANN401  # ActionRequest uses dynamic typing
        assistant_id: str | None,
    ) -> asyncio.Future:
        """Request user approval inline in the messages area.

        Mounts ApprovalMenu in the messages area (inline with chat).
        ChatInput stays visible - user can still see it.

        If another approval is already pending, queue this one.

        Auto-approves shell commands that are in the configured allow-list.

        Args:
            action_requests: List of action request dicts to approve
            assistant_id: The assistant ID for display purposes

        Returns:
            A Future that resolves to the user's decision.
        """
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future = loop.create_future()

        # Check if ALL actions in the batch are auto-approvable shell commands
        if settings.shell_allow_list and action_requests:
            all_auto_approved = True
            approved_commands = []

            for req in action_requests:
                if req.get("name") in SHELL_TOOL_NAMES:
                    command = req.get("args", {}).get("command", "")
                    if is_shell_command_allowed(command, settings.shell_allow_list):
                        approved_commands.append(command)
                    else:
                        all_auto_approved = False
                        break
                else:
                    # Non-shell commands need normal approval
                    all_auto_approved = False
                    break

            if all_auto_approved and approved_commands:
                # Auto-approve all commands in the batch
                result_future.set_result({"type": "approve"})

                # Mount system messages showing the auto-approvals
                try:
                    messages = self.query_one("#messages", Container)
                    for command in approved_commands:
                        auto_msg = AppMessage(
                            f"✓ Auto-approved shell command (allow-list): {command}"
                        )
                        await self._mount_before_queued(messages, auto_msg)
                    self._scroll_chat_to_bottom()
                except Exception:  # noqa: S110, BLE001  # Resilient auto-message display
                    pass  # Don't fail if we can't show the message

                return result_future

        # If there's already a pending approval, wait for it to complete first
        if self._pending_approval_widget is not None:
            while self._pending_approval_widget is not None:  # noqa: ASYNC110  # Simple polling is sufficient here
                await asyncio.sleep(0.1)

        # Create menu with unique ID to avoid conflicts
        unique_id = f"approval-menu-{uuid.uuid4().hex[:8]}"
        menu = ApprovalMenu(action_requests, assistant_id, id=unique_id)
        menu.set_future(result_future)

        # Store reference
        self._pending_approval_widget = menu

        # Mount approval inline in messages area (not replacing ChatInput)
        try:
            messages = self.query_one("#messages", Container)
            await self._mount_before_queued(messages, menu)
            # Scroll to make approval visible (but don't re-anchor)
            self.call_after_refresh(menu.scroll_visible)
            # Focus approval menu
            self.call_after_refresh(menu.focus)
        except Exception as e:
            logger.exception(
                "Failed to mount approval menu (id=%s) in messages container",
                unique_id,
            )
            self._pending_approval_widget = None
            if not result_future.done():
                result_future.set_exception(e)

        return result_future

    def _on_auto_approve_enabled(self) -> None:
        """Handle auto-approve being enabled via the HITL approval menu.

        Called when the user selects "Auto-approve all" from an approval
        dialog. Syncs the auto-approve state across the app flag, status
        bar indicator, and session state so subsequent tool calls skip
        the approval prompt.
        """
        self._auto_approve = True
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=True)
        if self._session_state:
            self._session_state.auto_approve = True

    async def _process_message(self, value: str, mode: InputMode) -> None:
        """Route a message to the appropriate handler based on mode.

        Args:
            value: The message text to process.
            mode: The input mode that determines message routing.
        """
        if mode == "bash":
            await self._handle_bash_command(value.removeprefix("!"))
        elif mode == "command":
            await self._handle_command(value)
        elif mode == "normal":
            await self._handle_user_message(value)
        else:
            logger.warning("Unrecognized input mode %r, treating as normal", mode)
            await self._handle_user_message(value)

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle submitted input from ChatInput widget."""
        value = event.value
        mode: InputMode = event.mode  # type: ignore[assignment]  # Textual event mode is str at type level but InputMode at runtime

        # Reset quit pending state on any input
        self._quit_pending = False

        # If agent is running, enqueue message instead of processing immediately
        if self._agent_running:
            self._pending_messages.append(QueuedMessage(text=value, mode=mode))
            queued_widget = QueuedUserMessage(value)
            self._queued_widgets.append(queued_widget)
            await self._mount_message(queued_widget)
            return

        await self._process_message(value, mode)

    def on_chat_input_mode_changed(self, event: ChatInput.ModeChanged) -> None:
        """Update status bar when input mode changes."""
        if self._status_bar:
            self._status_bar.set_mode(event.mode)

    async def on_approval_menu_decided(
        self,
        event: Any,  # noqa: ARG002, ANN401  # Textual event handler signature
    ) -> None:
        """Handle approval menu decision - remove from messages and refocus input."""
        # Remove ApprovalMenu using stored reference
        if self._pending_approval_widget:
            await self._pending_approval_widget.remove()
            self._pending_approval_widget = None

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

        # Execute bash command (user explicitly requested via ! prefix)
        # S604: shell=True is intentional - user requested shell execution via ! prefix
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
            # text=True ensures stdout/stderr are str, not bytes
            stdout = result.stdout
            stderr = result.stderr
            if not isinstance(stdout, str):
                stdout = stdout.decode() if stdout else ""
            output = stdout.strip()
            if stderr:
                if not isinstance(stderr, str):
                    stderr = stderr.decode() if stderr else ""
                output += f"\n[stderr]\n{stderr.strip()}"

            if output:
                # Display output as assistant message (uses markdown for code blocks)
                msg = AssistantMessage(f"```\n{output}\n```")
                await self._mount_message(msg)
                await msg.write_initial_content()
            else:
                await self._mount_message(AppMessage("Command completed (no output)"))

            if result.returncode != 0:
                await self._mount_message(
                    ErrorMessage(f"Exit code: {result.returncode}")
                )

            # Scroll to show the output (user-initiated command, so scroll is expected)
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)

        except subprocess.TimeoutExpired:
            await self._mount_message(ErrorMessage("Command timed out (60s limit)"))
        except OSError as e:
            await self._mount_message(ErrorMessage(str(e)))

    async def _open_url_command(self, command: str, cmd: str) -> None:
        """Open a URL in the browser and display a clickable link.

        Args:
            command: The raw command text (displayed as user message).
            cmd: The normalized slash command used to look up the URL.
        """
        url = _COMMAND_URLS[cmd]
        await self._mount_message(UserMessage(command))
        webbrowser.open(url)
        link = Text(url, style="dim italic")
        link.stylize(f"link {url}", 0)
        await self._mount_message(AppMessage(link))

    @staticmethod
    async def _build_thread_message(prefix: str, thread_id: str) -> str | Text:
        """Build a thread status message, hyperlinking the ID when possible.

        Attempts to resolve the LangSmith thread URL with a short timeout.
        Falls back to plain text if tracing is not configured or resolution
        fails.

        Args:
            prefix: Label before the thread ID (e.g. `'Resumed thread'`).
            thread_id: The thread identifier.

        Returns:
            A Rich `Text` with a clickable thread ID, or a plain string.
        """
        try:
            url = await asyncio.wait_for(
                asyncio.to_thread(build_langsmith_thread_url, thread_id),
                timeout=2.0,
            )
        except (TimeoutError, Exception):  # noqa: BLE001  # Resilient non-interactive mode error handling
            url = None

        if url:
            return Text.assemble(
                f"{prefix}: ",
                (thread_id, f"link {url}"),
            )
        return f"{prefix}: {thread_id}"

    async def _handle_trace_command(self, command: str) -> None:
        """Open the current thread in LangSmith.

        Shows a hint if no conversation has been started yet or if LangSmith
        tracing is not configured. Otherwise, opens the thread URL in the
        default browser and displays a clickable link.

        Args:
            command: The raw command text (displayed as user message).
        """
        await self._mount_message(UserMessage(command))
        if not self._session_state:
            await self._mount_message(AppMessage("No active session."))
            return
        thread_id = self._session_state.thread_id
        try:
            url = await asyncio.to_thread(build_langsmith_thread_url, thread_id)
        except Exception:
            logger.exception("Failed to build LangSmith thread URL for %s", thread_id)
            await self._mount_message(
                AppMessage("Failed to resolve LangSmith thread URL.")
            )
            return
        if not url:
            await self._mount_message(
                AppMessage(
                    "LangSmith tracing is not configured. "
                    "Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to enable."
                )
            )
            return
        try:
            webbrowser.open(url)
        except Exception:
            logger.debug("Could not open browser for URL: %s", url, exc_info=True)
        link = Text(url, style="dim italic")
        link.stylize(f"link {url}", 0)
        await self._mount_message(AppMessage(link))

    async def _handle_command(self, command: str) -> None:
        """Handle a slash command.

        Args:
            command: The slash command (including /)
        """
        cmd = command.lower().strip()

        if cmd in {"/quit", "/q"}:
            self.exit()
        elif cmd == "/help":
            await self._mount_message(UserMessage(command))
            help_text = Text(
                "Commands: /quit, /clear, /model [--default], /remember, "
                "/tokens, /threads, /trace, /changelog, /docs, /feedback, /help\n\n"
                "Interactive Features:\n"
                "  Enter           Submit your message\n"
                "  Ctrl+J          Insert newline\n"
                "  Shift+Tab       Toggle auto-approve mode\n"
                "  @filename       Auto-complete files and inject content\n"
                "  /command        Slash commands (/help, /clear, /quit)\n"
                "  !command        Run bash commands directly\n\n"
                f"Docs: {DOCS_URL}",
                style="dim italic",
            )
            help_text.stylize(f"link {DOCS_URL}", help_text.plain.index(DOCS_URL))
            await self._mount_message(AppMessage(help_text))

        elif cmd in {"/changelog", "/docs", "/feedback"}:
            await self._open_url_command(command, cmd)
        elif cmd == "/version":
            await self._mount_message(UserMessage(command))
            # Show CLI and SDK package versions
            try:
                from deepagents_cli._version import (
                    __version__ as cli_version,
                )

                cli_line = f"deepagents-cli version: {cli_version}"
            except ImportError:
                logger.debug("deepagents_cli._version module not found")
                cli_line = "deepagents-cli version: unknown"
            except Exception:
                logger.warning("Unexpected error looking up CLI version", exc_info=True)
                cli_line = "deepagents-cli version: unknown"
            try:
                from importlib.metadata import (
                    PackageNotFoundError,
                    version as _pkg_version,
                )

                sdk_version = _pkg_version("deepagents")
                sdk_line = f"deepagents (SDK) version: {sdk_version}"
            except PackageNotFoundError:
                logger.debug("deepagents SDK package not found in environment")
                sdk_line = "deepagents (SDK) version: unknown"
            except Exception:
                logger.warning("Unexpected error looking up SDK version", exc_info=True)
                sdk_line = "deepagents (SDK) version: unknown"
            await self._mount_message(AppMessage(f"{cli_line}\n{sdk_line}"))
        elif cmd == "/clear":
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            if self._token_tracker:
                self._token_tracker.reset()
            # Clear status message (e.g., "Interrupted" from previous session)
            self._update_status("")
            # Reset thread to start fresh conversation
            if self._session_state:
                new_thread_id = self._session_state.reset_thread()
                try:
                    banner = self.query_one("#welcome-banner", WelcomeBanner)
                    banner.update_thread_id(new_thread_id)
                except NoMatches:
                    pass
                await self._mount_message(
                    AppMessage(f"Started new thread: {new_thread_id}")
                )
        elif cmd == "/threads":
            await self._show_thread_selector()
        elif cmd == "/trace":
            await self._handle_trace_command(command)
        elif cmd == "/tokens":
            await self._mount_message(UserMessage(command))
            if self._token_tracker and self._token_tracker.current_context > 0:
                count = self._token_tracker.current_context
                formatted = _format_token_count(count)

                model_name = settings.model_name
                context_limit = settings.model_context_limit

                if context_limit is not None:
                    limit_str = _format_token_count(context_limit)
                    pct = count / context_limit * 100
                    usage = f"{formatted} / {limit_str} tokens ({pct:.0f}%)"
                else:
                    usage = f"{formatted} tokens used"

                msg = f"{usage} · {model_name}" if model_name else usage

                await self._mount_message(AppMessage(msg))
            else:
                await self._mount_message(AppMessage("No token usage yet"))
        elif cmd == "/remember" or cmd.startswith("/remember "):
            # Extract any additional context after /remember
            additional_context = ""
            if cmd.startswith("/remember "):
                additional_context = command.strip()[len("/remember ") :].strip()

            # Build the final prompt
            if additional_context:
                final_prompt = (
                    f"{REMEMBER_PROMPT}\n\n"
                    f"**Additional context from user:** {additional_context}"
                )
            else:
                final_prompt = REMEMBER_PROMPT

            # Send as a user message to the agent
            await self._handle_user_message(final_prompt)
            return  # _handle_user_message already mounts the message
        elif cmd == "/model" or cmd.startswith("/model "):
            model_arg = None
            set_default = False
            if cmd.startswith("/model "):
                raw_arg = command.strip()[len("/model ") :].strip()
                if raw_arg.startswith("--default"):
                    set_default = True
                    model_arg = raw_arg[len("--default") :].strip() or None
                else:
                    model_arg = raw_arg

            if set_default:
                await self._mount_message(UserMessage(command))
                if model_arg == "--clear":
                    await self._clear_default_model()
                elif model_arg:
                    await self._set_default_model(model_arg)
                else:
                    await self._mount_message(
                        AppMessage(
                            "Usage: /model --default provider:model\n"
                            "       /model --default --clear"
                        )
                    )
            elif model_arg:
                # Direct switch: /model claude-sonnet-4-5
                await self._mount_message(UserMessage(command))
                await self._switch_model(model_arg)
            else:
                await self._show_model_selector()
        else:
            await self._mount_message(UserMessage(command))
            await self._mount_message(AppMessage(f"Unknown command: {cmd}"))

        # Scroll to bottom after command output is rendered.
        # Use call_after_refresh so the layout pass completes first;
        # otherwise max_scroll_y is still stale.
        def _scroll_after_command() -> None:
            try:
                chat = self.query_one("#chat", VerticalScroll)
                if chat.max_scroll_y > 0:
                    chat.scroll_end(animate=False)
            except NoMatches:
                pass

        self.call_after_refresh(_scroll_after_command)

    async def _handle_user_message(self, message: str) -> None:
        """Handle a user message to send to the agent.

        Args:
            message: The user's message
        """
        # Mount the user message
        await self._mount_message(UserMessage(message))

        # Scroll to bottom when user sends a new message
        try:
            chat = self.query_one("#chat", VerticalScroll)
            if chat.max_scroll_y > 0:
                chat.scroll_end(animate=False)
        except NoMatches:
            pass

        # Check if agent is available
        if self._agent and self._ui_adapter and self._session_state:
            self._agent_running = True

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
                AppMessage(
                    "Agent not configured. "
                    "Run with --agent flag or use standalone mode."
                )
            )

    async def _run_agent_task(self, message: str) -> None:
        """Run the agent task in a background worker.

        This runs in a worker thread so the main event loop stays responsive.
        """
        # Caller ensures _ui_adapter is set (checked in _handle_user_message)
        if self._ui_adapter is None:
            return
        try:
            await execute_task_textual(
                user_input=message,
                agent=self._agent,
                assistant_id=self._assistant_id,
                session_state=self._session_state,
                adapter=self._ui_adapter,
                backend=self._backend,
                image_tracker=self._image_tracker,
            )
        except Exception as e:  # noqa: BLE001  # Resilient tool rendering
            # Ensure any in-flight tool calls don't remain stuck in "Running..."
            # when streaming aborts before tool results arrive.
            if self._ui_adapter:
                self._ui_adapter.finalize_pending_tools_with_error(f"Agent error: {e}")
            await self._mount_message(ErrorMessage(f"Agent error: {e}"))
        finally:
            # Clean up loading widget and agent state
            await self._cleanup_agent_task()

    async def _process_next_from_queue(self) -> None:
        """Process the next message from the queue if any exist.

        Dequeues and processes the next pending message in FIFO order.
        Uses the `_processing_pending` flag to prevent reentrant execution.
        """
        if self._processing_pending or not self._pending_messages or self._exit:
            return

        self._processing_pending = True
        try:
            msg = self._pending_messages.popleft()

            # Remove the ephemeral queued-message widget
            if self._queued_widgets:
                widget = self._queued_widgets.popleft()
                await widget.remove()

            await self._process_message(msg.text, msg.mode)
        except Exception:
            logger.exception("Failed to process queued message")
            await self._mount_message(
                ErrorMessage(f"Failed to process queued message: {msg.text[:60]}")
            )
        finally:
            self._processing_pending = False

        # Bash/command mode messages complete synchronously without spawning
        # a worker, so _cleanup_agent_task won't fire again. Continue
        # draining the queue if no worker was started.
        if not self._agent_running and self._pending_messages:
            await self._process_next_from_queue()

    async def _cleanup_agent_task(self) -> None:
        """Clean up after agent task completes or is cancelled."""
        self._agent_running = False
        self._agent_worker = None

        # Remove spinner if present
        await self._set_spinner(None)

        if self._chat_input:
            self._chat_input.set_cursor_active(active=True)

        # Ensure token display is restored (in case of early cancellation)
        if self._token_tracker:
            self._token_tracker.show()

        # Process next message from queue if any
        await self._process_next_from_queue()

    @staticmethod
    def _convert_messages_to_data(messages: list[Any]) -> list[MessageData]:
        """Convert LangChain messages into lightweight `MessageData` objects.

        This is a pure function with zero DOM operations. Tool call matching
        happens here: `ToolMessage` results are matched by `tool_call_id` and
        stored directly on the corresponding `MessageData`.

        Args:
            messages: LangChain message objects from a thread checkpoint.

        Returns:
            Ordered list of `MessageData` ready for `MessageStore.bulk_load`.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        result: list[MessageData] = []
        # Maps tool_call_id -> index into result list
        pending_tool_indices: dict[str, int] = {}

        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                if content.startswith("[SYSTEM]"):
                    continue
                result.append(MessageData(type=MessageType.USER, content=content))

            elif isinstance(msg, AIMessage):
                # Extract text content
                content = msg.content
                text = ""
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                        elif isinstance(block, str):
                            text += block
                    text = text.strip()

                if text:
                    result.append(MessageData(type=MessageType.ASSISTANT, content=text))

                # Track tool calls for later matching
                for tc in getattr(msg, "tool_calls", []):
                    tc_id = tc.get("id")
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    data = MessageData(
                        type=MessageType.TOOL,
                        content="",
                        tool_name=name,
                        tool_args=args,
                        tool_status=ToolStatus.PENDING,
                    )
                    result.append(data)
                    if tc_id:
                        pending_tool_indices[tc_id] = len(result) - 1
                    else:
                        data.tool_status = ToolStatus.REJECTED

            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in pending_tool_indices:
                    idx = pending_tool_indices.pop(tc_id)
                    data = result[idx]
                    status = getattr(msg, "status", "success")
                    content = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    if status == "success":
                        data.tool_status = ToolStatus.SUCCESS
                    else:
                        data.tool_status = ToolStatus.ERROR
                    data.tool_output = content
                else:
                    logger.debug(
                        "ToolMessage with tool_call_id=%r could not be "
                        "matched to a pending tool call",
                        tc_id,
                    )

            else:
                logger.debug(
                    "Skipping unsupported message type %s during history conversion",
                    type(msg).__name__,
                )

        # Mark unmatched tool calls as rejected
        for idx in pending_tool_indices.values():
            result[idx].tool_status = ToolStatus.REJECTED

        return result

    async def _load_thread_history(self) -> None:
        """Load and render message history when resuming a thread.

        This retrieves the checkpoint state from the agent and converts stored
        messages into lightweight `MessageData` objects, then bulk-loads them
        into the `MessageStore`. Only the last `WINDOW_SIZE` messages are
        mounted as widgets, drastically reducing DOM operations for large
        threads.
        """
        if not self._agent or not self._lc_thread_id:
            return

        config: RunnableConfig = {"configurable": {"thread_id": self._lc_thread_id}}

        try:
            # 1. Fetch state (single network call)
            state = await self._agent.aget_state(config)
            if not state or not state.values:
                return

            messages = state.values.get("messages", [])
            if not messages:
                return

            # 2. Convert to data (pure computation, no DOM)
            all_data = self._convert_messages_to_data(messages)
            if not all_data:
                return

            # 3. Bulk load into store (sets visible window)
            _archived, visible = self._message_store.bulk_load(all_data)

            # 4. Remove spacer once
            await self._remove_spacer()

            # 5. Cache container ref (single query)
            try:
                messages_container = self.query_one("#messages", Container)
            except NoMatches:
                return

            # 6-7. Create and mount only visible widgets (max WINDOW_SIZE)
            for msg_data in visible:
                widget = msg_data.to_widget()
                await messages_container.mount(widget)
                # 8. Render content for AssistantMessage after mount
                if isinstance(widget, AssistantMessage) and msg_data.content:
                    await widget.set_content(msg_data.content)

            # 9. Add footer — "Resumed thread" message
            thread_msg = await self._build_thread_message(
                "Resumed thread", self._lc_thread_id
            )
            await self._mount_message(AppMessage(thread_msg))

            # 10. Scroll once
            def scroll_to_end() -> None:
                with suppress(NoMatches):
                    chat = self.query_one("#chat", VerticalScroll)
                    chat.scroll_end(animate=False, immediate=True)

            self.set_timer(0.1, scroll_to_end)

        except Exception as e:  # Resilient history loading
            logger.exception("Failed to load thread history for %s", self._lc_thread_id)
            await self._mount_message(AppMessage(f"Could not load history: {e}"))

    async def _mount_message(
        self, widget: Static | AssistantMessage | ToolCallMessage
    ) -> None:
        """Mount a message widget to the messages area.

        This method also stores the message data and handles pruning
        when the widget count exceeds the maximum.

        If the ``#messages`` container is not present (e.g. the screen has
        been torn down during an interruption), the call is silently skipped
        to avoid cascading `NoMatches` errors.

        Args:
            widget: The message widget to mount
        """
        await self._remove_spacer()

        try:
            messages = self.query_one("#messages", Container)
        except NoMatches:
            return

        # Store message data for virtualization
        message_data = MessageData.from_widget(widget)
        self._message_store.append(message_data)

        # Queued-message widgets must always stay at the bottom so they
        # remain visually anchored below the current agent response.
        if isinstance(widget, QueuedUserMessage):
            await messages.mount(widget)
        else:
            await self._mount_before_queued(messages, widget)

        # Prune old widgets if window exceeded
        await self._prune_old_messages()

        # Scroll to keep input bar visible
        try:
            input_container = self.query_one("#bottom-app-container", Container)
            input_container.scroll_visible()
        except NoMatches:
            pass

    async def _prune_old_messages(self) -> None:
        """Prune oldest message widgets if we exceed the window size.

        This removes widgets from the DOM but keeps data in MessageStore
        for potential re-hydration when scrolling up.
        """
        if not self._message_store.window_exceeded():
            return

        try:
            messages_container = self.query_one("#messages", Container)
        except NoMatches:
            logger.debug("Skipping pruning: #messages container not found")
            return

        to_prune = self._message_store.get_messages_to_prune()
        if not to_prune:
            return

        pruned_ids: list[str] = []
        for msg_data in to_prune:
            try:
                widget = messages_container.query_one(f"#{msg_data.id}")
                await widget.remove()
                pruned_ids.append(msg_data.id)
            except NoMatches:
                # Widget not found -- do NOT mark as pruned to avoid
                # desyncing the store from the actual DOM state
                logger.debug(
                    "Widget %s not found during pruning, skipping",
                    msg_data.id,
                )

        if pruned_ids:
            self._message_store.mark_pruned(pruned_ids)

    def _set_active_message(self, message_id: str | None) -> None:
        """Set the active streaming message (won't be pruned).

        Args:
            message_id: The ID of the active message, or None to clear.
        """
        self._message_store.set_active_message(message_id)

    def _sync_message_content(self, message_id: str, content: str) -> None:
        """Sync final message content back to the store after streaming.

        Called when streaming finishes so the store holds the full text
        instead of the empty string captured at mount time.

        Args:
            message_id: The ID of the message to update.
            content: The final content after streaming.
        """
        self._message_store.update_message(
            message_id,
            content=content,
            is_streaming=False,
        )

    async def _clear_messages(self) -> None:
        """Clear the messages area and message store."""
        # Clear the message store first
        self._message_store.clear()
        try:
            messages = self.query_one("#messages", Container)
            await messages.remove_children()
        except NoMatches:
            logger.warning(
                "Messages container (#messages) not found during clear; "
                "UI may be out of sync with message store"
            )

    def action_quit_or_interrupt(self) -> None:
        """Handle Ctrl+C - interrupt agent, reject approval, or quit on double press.

        Priority order:
        1. If agent is running, interrupt it (preserve input)
        2. If approval menu is active, reject it
        3. If double press (quit_pending), quit
        4. Otherwise show quit hint
        """
        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._pending_messages.clear()
            for w in self._queued_widgets:
                w.remove()
            self._queued_widgets.clear()
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
        """Handle escape key.

        Dismiss completion popup, dismiss modal, interrupt agent,
        or reject approval. This is the primary way to stop a
        running agent.
        """
        # If a modal screen is active, dismiss it
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return

        # Close completion popup before interrupting the agent
        if self._chat_input and self._chat_input.dismiss_completion():
            return

        # If agent is running, interrupt it and discard queued messages
        if self._agent_running and self._agent_worker:
            self._pending_messages.clear()
            for w in self._queued_widgets:
                w.remove()
            self._queued_widgets.clear()
            self._agent_worker.cancel()
            return

        # If approval menu is active, reject it
        if self._pending_approval_widget:
            self._pending_approval_widget.action_select_reject()

    def action_quit_app(self) -> None:
        """Handle quit action (Ctrl+D)."""
        self.exit()

    def exit(
        self,
        result: Any = None,  # noqa: ANN401  # Dynamic LangGraph stream result type
        return_code: int = 0,
        message: Any = None,  # noqa: ANN401  # Dynamic LangGraph message type
    ) -> None:
        """Exit the app, restoring iTerm2 cursor guide if applicable.

        Overrides parent to restore iTerm2's cursor guide before Textual's
        cleanup. The atexit handler serves as a fallback for abnormal
        termination.

        Args:
            result: Return value passed to the app runner.
            return_code: Exit code (non-zero for errors).
            message: Optional message to display on exit.
        """
        # Discard queued messages so _cleanup_agent_task won't try to
        # process them after the event loop is torn down.
        self._pending_messages.clear()
        for w in self._queued_widgets:
            w.remove()
        self._queued_widgets.clear()

        _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
        super().exit(result=result, return_code=return_code, message=message)

    def action_toggle_auto_approve(self) -> None:
        """Toggle auto-approve mode for the current session.

        When enabled, all tool calls (shell execution, file writes/edits,
        web search, URL fetch) run without prompting. Updates the status
        bar indicator and session state.
        """
        self._auto_approve = not self._auto_approve
        if self._status_bar:
            self._status_bar.set_auto_approve(enabled=self._auto_approve)
        if self._session_state:
            self._session_state.auto_approve = self._auto_approve

    def action_toggle_tool_output(self) -> None:
        """Toggle expand/collapse of the most recent tool output."""
        # Find all tool messages with output, get the most recent one
        # NoMatches is raised if no ToolCallMessage widgets exist
        with suppress(NoMatches):
            tool_messages = list(self.query(ToolCallMessage))
            # Find ones with output, toggle the most recent
            for tool_msg in reversed(tool_messages):
                if tool_msg.has_output:
                    tool_msg.toggle_output()
                    return

    # Approval menu action handlers (delegated from App-level bindings)
    # NOTE: These only activate when approval widget is pending
    # AND input is not focused
    def action_approval_up(self) -> None:
        """Handle up arrow in approval menu."""
        # Only handle if approval is active
        # (input handles its own up for history/completion)
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
        """Check if the chat input (or its text area) has focus.

        Returns:
            True if the input widget has focus, False otherwise.
        """
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

    def on_paste(self, event: Paste) -> None:
        """Route unfocused paste events to chat input for drag/drop reliability."""
        if not self._chat_input:
            return
        if self._pending_approval_widget or self._is_input_focused():
            return
        if self._chat_input.handle_external_paste(event.text):
            event.prevent_default()
            event.stop()

    def on_app_focus(self) -> None:
        """Restore chat input focus when the terminal regains OS focus.

        When the user opens a link via `webbrowser.open`, OS focus shifts to
        the browser. On returning to the terminal, Textual fires `AppFocus`
        (requires a terminal that supports FocusIn events). Re-focusing the chat
        input here keeps it ready for typing.
        """
        if not self._chat_input:
            return
        if isinstance(self.screen, ModalScreen):
            return
        if self._pending_approval_widget:
            return
        self._chat_input.focus_input()

    def on_click(self, _event: Click) -> None:
        """Handle clicks anywhere in the terminal to focus on the command line."""
        if not self._chat_input:
            return
        # Don't steal focus from approval widget
        if self._pending_approval_widget:
            return
        self.call_after_refresh(self._chat_input.focus_input)

    def on_mouse_up(self, event: MouseUp) -> None:  # noqa: ARG002  # Textual event handler signature
        """Copy selection to clipboard on mouse release."""
        copy_selection_to_clipboard(self)

    # =========================================================================
    # Model Switching
    # =========================================================================

    async def _show_model_selector(self) -> None:
        """Show interactive model selector as a modal screen."""

        def handle_result(result: tuple[str, str] | None) -> None:
            """Handle the model selector result."""
            if result is not None:
                model_spec, _ = result
                self.call_later(self._switch_model, model_spec)
            # Refocus input after modal closes
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ModelSelectorScreen(
            current_model=settings.model_name,
            current_provider=settings.model_provider,
        )
        self.push_screen(screen, handle_result)

    async def _show_thread_selector(self) -> None:
        """Show interactive thread selector as a modal screen."""
        current = self._session_state.thread_id if self._session_state else None

        def handle_result(result: str | None) -> None:
            """Handle the thread selector result."""
            if result is not None:
                self.call_later(self._resume_thread, result)
            if self._chat_input:
                self._chat_input.focus_input()

        screen = ThreadSelectorScreen(current_thread=current)
        self.push_screen(screen, handle_result)

    async def _resume_thread(self, thread_id: str) -> None:
        """Resume a previously saved thread.

        Clears the current conversation, switches to the selected thread,
        updates the welcome banner, and loads its message history.

        Args:
            thread_id: The thread ID to resume.
        """
        if not self._agent:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active agent")
            )
            return

        if not self._session_state:
            await self._mount_message(
                AppMessage("Cannot switch threads: no active session")
            )
            return

        # Skip if already on this thread
        if self._session_state.thread_id == thread_id:
            await self._mount_message(AppMessage(f"Already on thread: {thread_id}"))
            return

        # Save previous state for rollback on failure
        prev_thread_id = self._lc_thread_id
        prev_session_thread = self._session_state.thread_id

        try:
            # Clear conversation (similar to /clear, without creating a new thread)
            self._pending_messages.clear()
            self._queued_widgets.clear()
            await self._clear_messages()
            if self._token_tracker:
                self._token_tracker.reset()
            self._update_status("")

            # Switch to the selected thread
            self._session_state.thread_id = thread_id
            self._lc_thread_id = thread_id

            # Update welcome banner
            try:
                banner = self.query_one("#welcome-banner", WelcomeBanner)
                banner.update_thread_id(thread_id)
            except NoMatches:
                logger.debug(
                    "Welcome banner not found during thread switch to %s",
                    thread_id,
                )

            # Load thread history
            await self._load_thread_history()

        except Exception as exc:
            logger.exception("Failed to switch to thread %s", thread_id)
            # Restore previous thread IDs so the user can retry
            self._session_state.thread_id = prev_session_thread
            self._lc_thread_id = prev_thread_id
            try:
                banner = self.query_one("#welcome-banner", WelcomeBanner)
                banner.update_thread_id(prev_session_thread)
            except NoMatches:
                logger.warning(
                    "Welcome banner not found during rollback to thread %s; "
                    "banner may display stale thread ID",
                    prev_session_thread,
                )
            # Attempt to restore the previous thread's visible history
            try:
                await self._load_thread_history()
            except Exception:  # noqa: BLE001  # Resilient session state saving
                logger.debug(
                    "Could not restore previous thread history after "
                    "failed switch to %s",
                    thread_id,
                )
            await self._mount_message(
                AppMessage(
                    f"Failed to switch to thread {thread_id}: {exc}. "
                    "Use /threads to try again."
                )
            )

    async def _switch_model(self, model_spec: str) -> None:
        """Switch to a new model, preserving conversation history.

        Args:
            model_spec: The model specification to switch to.

                Can be in `provider:model` format
                (e.g., `'anthropic:claude-sonnet-4-5'`) or just the model name
                for auto-detection.
        """
        logger.info("Switching model to %s", model_spec)

        from deepagents_cli.agent import create_cli_agent
        from deepagents_cli.model_config import (
            ModelConfigError,
            get_credential_env_var,
            has_provider_credentials,
        )

        # Strip leading colon — treat ":claude-opus-4-6" as "claude-opus-4-6"
        model_spec = model_spec.removeprefix(":")

        parsed = ModelSpec.try_parse(model_spec)
        if parsed:
            provider: str | None = parsed.provider
            model_name = parsed.model
        else:
            model_name = model_spec
            provider = detect_provider(model_spec)

        # Check credentials
        if provider and has_provider_credentials(provider) is False:
            env_var = get_credential_env_var(provider)
            if env_var:
                detail = f"{env_var} is not set or is empty"
            else:
                detail = (
                    f"provider '{provider}' is not recognized. "
                    "Add it to ~/.deepagents/config.toml with an api_key_env field"
                )
            await self._mount_message(ErrorMessage(f"Missing credentials: {detail}"))
            return

        # Check if already using this exact model
        if model_name == settings.model_name and (
            not provider or provider == settings.model_provider
        ):
            current = f"{settings.model_provider}:{settings.model_name}"
            await self._mount_message(AppMessage(f"Already using {current}"))
            return

        # Check if we have what we need for hot-swap
        if not self._checkpointer:
            # No checkpointer means we can't hot-swap
            # Save the preference and notify user
            if save_recent_model(model_spec):
                await self._mount_message(
                    AppMessage(
                        f"Model preference set to {model_spec}. "
                        "Restart the CLI for the change to take effect."
                    )
                )
            else:
                await self._mount_message(
                    ErrorMessage(
                        "Could not save model preference. "
                        "Check permissions for ~/.deepagents/"
                    )
                )
            return

        try:
            result = create_model(model_spec)
        except ModelConfigError as e:
            await self._mount_message(ErrorMessage(str(e)))
            return
        except Exception as e:
            logger.exception("Failed to create model from spec %s", model_spec)
            await self._mount_message(ErrorMessage(f"Failed to create model: {e}"))
            return

        # When switching models, settings must be updated before
        # create_cli_agent because it builds the system prompt from global
        # settings (model name, provider, context limit). Otherwise the
        # prompt would describe the old model to the new one.
        #
        # Save previous values for rollback if agent creation fails.
        prev_name = settings.model_name
        prev_provider = settings.model_provider
        prev_context_limit = settings.model_context_limit
        result.apply_to_settings()

        try:
            new_agent, new_backend = create_cli_agent(
                model=result.model,
                assistant_id=self._assistant_id or "default",
                tools=self._tools,
                sandbox=self._sandbox,
                sandbox_type=self._sandbox_type,
                auto_approve=self._auto_approve,
                checkpointer=self._checkpointer,
            )
        except Exception as e:
            # Roll back settings so the running agent isn't misrepresented.
            settings.model_name = prev_name
            settings.model_provider = prev_provider
            settings.model_context_limit = prev_context_limit
            logger.exception("Failed to create agent for model switch")
            await self._mount_message(ErrorMessage(f"Model switch failed: {e}"))
            return

        # Swap agent
        self._agent = new_agent
        self._backend = new_backend

        # Post-swap: update UI and save config
        display = f"{settings.model_provider}:{settings.model_name}"
        if self._status_bar:
            self._status_bar.set_model(display)

        config_saved = save_recent_model(display)
        if config_saved:
            await self._mount_message(AppMessage(f"Switched to {display}"))
        else:
            await self._mount_message(
                AppMessage(
                    f"Switched to {display} (preference not saved - "
                    "check ~/.deepagents/ permissions)"
                )
            )

        logger.info("Model switched to %s", display)

        # Scroll to bottom so the confirmation message is visible
        def _scroll_after_switch() -> None:
            try:
                chat = self.query_one("#chat", VerticalScroll)
                if chat.max_scroll_y > 0:
                    chat.scroll_end(animate=False)
            except NoMatches:
                pass

        self.call_after_refresh(_scroll_after_switch)

    async def _set_default_model(self, model_spec: str) -> None:
        """Set the default model in config without switching the current session.

        Updates `[models].default` in `~/.deepagents/config.toml` so that
        future CLI launches use this model. Does not affect the running session.

        Args:
            model_spec: The model specification (e.g., `'anthropic:claude-opus-4-6'`).
        """
        from deepagents_cli.model_config import save_default_model

        model_spec = model_spec.removeprefix(":")

        parsed = ModelSpec.try_parse(model_spec)
        if not parsed:
            provider = detect_provider(model_spec)
            if provider:
                model_spec = f"{provider}:{model_spec}"

        if save_default_model(model_spec):
            await self._mount_message(AppMessage(f"Default model set to {model_spec}"))
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not save default model. Check permissions for ~/.deepagents/"
                )
            )

    async def _clear_default_model(self) -> None:
        """Remove the default model from config.

        After clearing, future launches fall back to `[models].recent` or
        environment auto-detection.
        """
        from deepagents_cli.model_config import clear_default_model

        if clear_default_model():
            await self._mount_message(
                AppMessage(
                    "Default model cleared. "
                    "Future launches will use recent model or auto-detect."
                )
            )
        else:
            await self._mount_message(
                ErrorMessage(
                    "Could not clear default model. "
                    "Check permissions for ~/.deepagents/"
                )
            )


@dataclass(frozen=True)
class AppResult:
    """Result from running the Textual application.

    Attributes:
        return_code: Exit code (0 for success, non-zero for error).
        thread_id: The final thread ID at shutdown. May differ from the
            initial thread ID if the user switched threads via `/threads`.
    """

    return_code: int
    thread_id: str | None


async def run_textual_app(
    *,
    agent: Pregel | None = None,
    assistant_id: str | None = None,
    backend: CompositeBackend | None = None,
    auto_approve: bool = False,
    cwd: str | Path | None = None,
    thread_id: str | None = None,
    initial_prompt: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    tools: list[Callable[..., Any] | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
) -> AppResult:
    """Run the Textual application.

    Args:
        agent: Pre-configured LangGraph agent (optional)
        assistant_id: Agent identifier for memory storage
        backend: Backend for file operations
        auto_approve: Whether to start with auto-approve enabled
        cwd: Current working directory to display
        thread_id: Optional thread ID for session persistence
        initial_prompt: Optional prompt to auto-submit when session starts
        checkpointer: Checkpointer for session persistence (enables model hot-swap)
        tools: Tools used to create the agent (for model hot-swap)
        sandbox: Sandbox backend (for model hot-swap)
        sandbox_type: Type of sandbox provider (for model hot-swap)

    Returns:
        An `AppResult` with the return code and final thread ID.
    """
    app = DeepAgentsApp(
        agent=agent,
        assistant_id=assistant_id,
        backend=backend,
        auto_approve=auto_approve,
        cwd=cwd,
        thread_id=thread_id,
        initial_prompt=initial_prompt,
        checkpointer=checkpointer,
        tools=tools,
        sandbox=sandbox,
        sandbox_type=sandbox_type,
    )
    await app.run_async()
    return AppResult(
        return_code=app.return_code or 0,
        thread_id=app._lc_thread_id,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_textual_app())
