"""Status bar widget for deepagents-cli."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from deepagents_cli.config import get_glyphs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult, RenderResult
    from textual.geometry import Size


class ModelLabel(Widget):
    """A label that displays a model name, right-aligned with smart truncation.

    When the full `provider:model` text doesn't fit, the provider is dropped
    first. If the bare model name still doesn't fit, it is left-truncated
    with a leading ellipsis so the most distinctive tail stays visible.
    """

    provider: reactive[str] = reactive("", layout=True)
    model: reactive[str] = reactive("", layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """Return the intrinsic width so `width: auto` works.

        Args:
            container: Size of the container.
            viewport: Size of the viewport.

        Returns:
            Character length of the full provider:model string.
        """
        if not self.model:
            return 0
        full = f"{self.provider}:{self.model}" if self.provider else self.model
        return len(full)

    def render(self) -> RenderResult:
        """Render the model label with width-aware truncation.

        Returns:
            Text content, truncated from the left when necessary.
        """
        width = self.content_size.width
        if not self.model or width <= 0:
            return ""
        full = f"{self.provider}:{self.model}" if self.provider else self.model
        if len(full) <= width:
            return Content(full)
        if len(self.model) <= width:
            return Content(self.model)
        if width > 1:
            return Content("\u2026" + self.model[-(width - 1) :])
        return Content("\u2026")


class CwdLabel(Widget):
    """A label that displays the current working directory with ellipsis truncation.

    When the full path doesn't fit, it is left-truncated with a leading
    ellipsis so the deepest directory segments stay visible.
    """

    cwd: reactive[str] = reactive("", layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """Return the intrinsic width so `width: auto` works.

        Args:
            container: Size of the container.
            viewport: Size of the viewport.

        Returns:
            Character length of the formatted path.
        """
        return len(self.cwd)

    def render(self) -> RenderResult:
        """Render the cwd label with width-aware truncation.

        Returns:
            Text content, left-truncated with ellipsis when necessary.
        """
        width = self.content_size.width
        if not self.cwd or width <= 0:
            return ""
        if len(self.cwd) <= width:
            return Content(self.cwd)
        if width > 1:
            return Content("\u2026" + self.cwd[-(width - 1) :])
        return Content("\u2026")


class BranchLabel(Widget):
    """A label that displays a git branch name with ellipsis truncation.

    When the full icon + branch text doesn't fit, the branch name is
    right-truncated with a trailing ellipsis so the prefix stays visible.
    """

    branch: reactive[str] = reactive("", layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """Return the intrinsic width so `width: auto` works.

        Args:
            container: Size of the container.
            viewport: Size of the viewport.

        Returns:
            Character length of the full icon + branch string.
        """
        if not self.branch:
            return 0
        icon = get_glyphs().git_branch
        return len(f"{icon} {self.branch}")

    def render(self) -> RenderResult:
        """Render the branch label with width-aware truncation.

        Returns:
            Text content, right-truncated with ellipsis when necessary.
        """
        width = self.content_size.width
        if not self.branch or width <= 0:
            return ""
        icon = get_glyphs().git_branch
        full = f"{icon} {self.branch}"
        if len(full) <= width:
            return Content(full)
        # Left-truncate: keep icon + trailing branch segment visible
        if width > 1:
            return Content("\u2026" + full[-(width - 1) :])
        return Content("\u2026")


class StatusBar(Horizontal):
    """Status bar showing mode, auto-approve, cwd, git branch, tokens, and model."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }

    StatusBar .status-mode {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-mode.normal {
        display: none;
    }

    StatusBar .status-mode.shell {
        background: $mode-bash;
        color: white;
        text-style: bold;
    }

    StatusBar .status-mode.command {
        background: $mode-command;
        color: white;
    }

    StatusBar .status-auto-approve {
        width: auto;
        padding: 0 1;
    }

    StatusBar .status-auto-approve.on {
        background: $success;
        color: $background;
    }

    StatusBar .status-auto-approve.off {
        background: $warning;
        color: $background;
    }

    StatusBar .status-message {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .status-message.thinking {
        color: $warning;
    }

    StatusBar CwdLabel {
        width: 1fr;
        min-width: 0;
        color: $text-muted;
    }

    StatusBar BranchLabel {
        width: 1fr;
        min-width: 0;
        color: $text-muted;
        padding: 0 1;
    }

    StatusBar .status-left-collapsible {
        width: 1fr;
        min-width: 0;
        height: 1;
        overflow-x: hidden;
    }

    StatusBar .status-tokens {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar ModelLabel {
        width: auto;
        padding: 0 2;
        color: $text-muted;
        text-align: right;
    }
    """
    """Mode badges and auto-approve pills use distinct colors for at-a-glance status."""

    mode: reactive[str] = reactive("normal", init=False)
    status_message: reactive[str] = reactive("", init=False)
    auto_approve: reactive[bool] = reactive(default=False, init=False)
    cwd: reactive[str] = reactive("", init=False)
    branch: reactive[str] = reactive("", init=False)
    tokens: reactive[int] = reactive(0, init=False)

    def __init__(self, cwd: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the status bar.

        Args:
            cwd: Current working directory to display
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        # Store initial cwd - will be used in compose()
        self._initial_cwd = str(cwd) if cwd else str(Path.cwd())

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """Compose the status bar layout.

        Yields:
            Widgets for mode, auto-approve, message, cwd, branch, tokens, and
                model display.
        """
        yield Static("", classes="status-mode normal", id="mode-indicator")
        yield Static(
            "manual | shift+tab to cycle",
            classes="status-auto-approve off",
            id="auto-approve-indicator",
        )
        with Horizontal(classes="status-left-collapsible"):
            yield Static("", classes="status-message", id="status-message")
            yield CwdLabel(id="cwd-display")
            yield BranchLabel(id="branch-display")
        yield Static("", classes="status-tokens", id="tokens-display")
        yield ModelLabel(id="model-display")

    _BRANCH_WIDTH_THRESHOLD = 100
    """Hide git branch display below this terminal width."""
    _CWD_WIDTH_THRESHOLD = 70
    """Hide cwd display below this terminal width."""

    def on_resize(self, event: events.Resize) -> None:
        """Manage visibility of status items based on terminal width.

        Priority (highest first): model, cwd, git branch.
        """
        width = event.size.width
        with suppress(NoMatches):
            self.query_one("#branch-display", BranchLabel).display = (
                width >= self._BRANCH_WIDTH_THRESHOLD
            )
        with suppress(NoMatches):
            self.query_one("#cwd-display", CwdLabel).display = (
                width >= self._CWD_WIDTH_THRESHOLD
            )

    def on_mount(self) -> None:
        """Set reactive values after mount to trigger watchers safely."""
        from deepagents_cli.config import settings

        self.cwd = self._initial_cwd
        # Set initial model display
        label = self.query_one("#model-display", ModelLabel)
        label.provider = settings.model_provider or ""
        label.model = settings.model_name or ""

    def watch_mode(self, mode: str) -> None:
        """Update mode indicator when mode changes."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("normal", "shell", "command")

        if mode == "shell":
            indicator.update("SHELL")
            indicator.add_class("shell")
        elif mode == "command":
            indicator.update("CMD")
            indicator.add_class("command")
        else:
            indicator.update("")
            indicator.add_class("normal")

    def watch_auto_approve(self, new_value: bool) -> None:
        """Update auto-approve indicator when state changes."""
        try:
            indicator = self.query_one("#auto-approve-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("on", "off")

        if new_value:
            indicator.update("auto | shift+tab to cycle")
            indicator.add_class("on")
        else:
            indicator.update("manual | shift+tab to cycle")
            indicator.add_class("off")

    def watch_cwd(self, new_value: str) -> None:
        """Update cwd display when it changes."""
        try:
            label = self.query_one("#cwd-display", CwdLabel)
        except NoMatches:
            return
        label.cwd = self._format_cwd(new_value)

    def watch_branch(self, new_value: str) -> None:
        """Update branch display when it changes."""
        try:
            label = self.query_one("#branch-display", BranchLabel)
        except NoMatches:
            return
        label.branch = new_value

    def watch_status_message(self, new_value: str) -> None:
        """Update status message display."""
        try:
            msg_widget = self.query_one("#status-message", Static)
        except NoMatches:
            return

        msg_widget.remove_class("thinking")
        if new_value:
            msg_widget.update(new_value)
            if "thinking" in new_value.lower() or "executing" in new_value.lower():
                msg_widget.add_class("thinking")
        else:
            msg_widget.update("")

    def _format_cwd(self, cwd_path: str = "") -> str:
        """Format the current working directory for display.

        Returns:
            Formatted path string, using ~ for home directory when possible.
        """
        path = Path(cwd_path or self.cwd or self._initial_cwd)
        try:
            # Try to use ~ for home directory
            home = Path.home()
            if path.is_relative_to(home):
                return "~/" + path.relative_to(home).as_posix()
        except (ValueError, RuntimeError):
            pass
        return str(path)

    def set_mode(self, mode: str) -> None:
        """Set the current input mode.

        Args:
            mode: One of "normal", "shell", or "command"
        """
        self.mode = mode

    def set_auto_approve(self, *, enabled: bool) -> None:
        """Set the auto-approve state.

        Args:
            enabled: Whether auto-approve is enabled
        """
        self.auto_approve = enabled

    def set_status_message(self, message: str) -> None:
        """Set the status message.

        Args:
            message: Status message to display (empty string to clear)
        """
        self.status_message = message

    def watch_tokens(self, new_value: int) -> None:
        """Update token display when count changes."""
        try:
            display = self.query_one("#tokens-display", Static)
        except NoMatches:
            return

        if new_value > 0:
            # Format with K suffix for thousands
            if new_value >= 1000:  # noqa: PLR2004  # Count formatting threshold
                display.update(f"{new_value / 1000:.1f}K tokens")
            else:
                display.update(f"{new_value} tokens")
        else:
            display.update("")

    def set_tokens(self, count: int) -> None:
        """Set the token count.

        Args:
            count: Current context token count
        """
        self.tokens = count

    def hide_tokens(self) -> None:
        """Hide the token display (e.g., during streaming)."""
        self.query_one("#tokens-display", Static).update("")

    def set_model(self, *, provider: str, model: str) -> None:
        """Update the model display text.

        Args:
            provider: Model provider name (e.g., `'anthropic'`).
            model: Model name (e.g., `'claude-sonnet-4-5'`).
        """
        label = self.query_one("#model-display", ModelLabel)
        label.provider = provider
        label.model = model
