"""Status bar widget."""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code._constants import FIREWORKS_MODEL_ID_PREFIXES
from deepagents_code._env_vars import HIDE_CWD, HIDE_GIT_BRANCH, is_env_truthy
from deepagents_code._session_stats import format_cost, format_token_count
from deepagents_code.config import get_glyphs
from deepagents_code.tui.widgets.loading import Spinner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult, RenderResult
    from textual.geometry import Size
    from textual.timer import Timer

PROVIDER_PREFIX_STRIPS: dict[str, tuple[str, ...]] = {
    "fireworks": FIREWORKS_MODEL_ID_PREFIXES,
}
"""Some providers (e.g. Fireworks) require fully-qualified IDs like
`accounts/fireworks/models/...` or `accounts/fireworks/routers/...` that crowd
out the rest of the status bar; strip the registered prefixes before display."""

ConnectionState = Literal["", "connecting", "reconnecting", "resuming"]
"""Connection states the status bar can display (`''` means cleared)."""

CONNECTION_STATES = frozenset(get_args(ConnectionState))
"""Runtime view of `ConnectionState` for `set_connection`'s defensive guard.

Derived from the `Literal` so the two can never drift."""

StatusMessageSource = Literal["agent", "hooks"]
"""Owners that may write the shared status-message slot."""


def _compact_tokens(count: int) -> str:
    """Format a token count without a trailing `.0`.

    Returns:
        Compact token count.
    """
    text = format_token_count(count)
    return text.replace(".0", "", 1) if ".0" in text else text


class ModelLabel(Widget):
    """A label that displays a model name with smart truncation.

    When the full `provider:model` text doesn't fit, the provider is dropped
    first. If the bare model name still doesn't fit, it is left-truncated
    with a leading ellipsis so the most distinctive tail stays visible.

    When a reasoning effort is set, its label is appended to the model and
    participates in the same ladder: the effort suffix is preserved (with the
    model left-truncated to make room) and is only dropped once even the
    left-truncated model plus effort cannot fit.
    """

    provider: reactive[str] = reactive("", layout=True)
    model: reactive[str] = reactive("", layout=True)
    effort: reactive[str] = reactive("", layout=True)

    def _clean_model(self) -> str:
        """Strip the provider's registered prefix so the status bar stays compact.

        Returns:
            Model name with the provider's registered prefix removed if present,
                otherwise the original name.
        """
        name = self.model
        if not name or not self.provider:
            return name
        # Match on normalized text but slice the original to preserve its casing.
        name_lower = name.lower()
        for prefix in PROVIDER_PREFIX_STRIPS.get(self.provider, ()):
            if name_lower.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _with_effort(self, text: str) -> str:
        """Append the reasoning effort label when one is set.

        Args:
            text: Base model display text.

        Returns:
            Model display text with the effort suffix (a per-session override or
                the provider default) when one is present, else
                `text` unchanged.
        """
        return f"{text} {self.effort}" if self.effort else text

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
        model = self._clean_model()
        full = f"{self.provider}:{model}" if self.provider else model
        return len(self._with_effort(full))

    def render(self) -> RenderResult:
        """Render the model label with width-aware truncation.

        Returns:
            Text content, truncated from the left when necessary.
        """
        width = self.content_size.width
        if not self.model or width <= 0:
            return ""
        model = self._clean_model()
        full = f"{self.provider}:{model}" if self.provider else model
        full_with_effort = self._with_effort(full)
        model_with_effort = self._with_effort(model)
        if len(full_with_effort) <= width:
            return Content(full_with_effort)
        if len(model_with_effort) <= width:
            return Content(model_with_effort)
        suffix = f" {self.effort}" if self.effort else ""
        if suffix and width > len(suffix) + 1:
            model_width = width - len(suffix)
            return Content(f"\u2026{model[-(model_width - 1) :]}{suffix}")
        if len(model) <= width:
            return Content(model)
        if width > 1:
            return Content("\u2026" + model[-(width - 1) :])
        return Content("\u2026")


class BranchLabel(Widget):
    """A label that displays the git branch with glyph-aware truncation.

    Unlike CSS `text-overflow: ellipsis` (which always uses the Unicode
    ellipsis character), this widget truncates manually in :meth:`render` using
    :func:`get_glyphs` so ASCII mode (`DEEPAGENTS_CODE_UI_CHARSET_MODE=ascii`)
    gets `"..."` instead of `"…"`.
    """

    branch: reactive[str] = reactive("", layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """Return the intrinsic width so the widget participates in flex layout.

        Args:
            container: Size of the container.
            viewport: Size of the viewport.

        Returns:
            Character length of the full branch string (icon + space + name),
                or `0` when the branch is empty.
        """
        if not self.branch:
            return 0
        icon = get_glyphs().git_branch
        return len(icon) + 1 + len(self.branch)

    def render(self) -> RenderResult:
        """Render the branch label, truncating with the configured glyph.

        Returns:
            Branch text (icon + name) truncated from the right with
                :func:`get_glyphs`'s ellipsis when it overflows the available
                width, or an empty string when no branch is set.
        """
        width = self.content_size.width
        if not self.branch or width <= 0:
            return ""
        icon = get_glyphs().git_branch
        full = f"{icon} {self.branch}"
        if len(full) <= width:
            return full
        ellipsis = get_glyphs().ellipsis
        if width <= len(ellipsis):
            return full[:width]
        return full[: width - len(ellipsis)] + ellipsis


class MetricsLine(Widget):
    """A bullet-separated chain of session metrics.

    Segments are supplied pre-styled and in priority order. When the chain is
    wider than the bar, trailing segments are dropped one at a time (so the
    least important metric goes first and the surviving segments never shift
    position), and a lone segment that still overflows is ellipsized.
    """

    segments: reactive[tuple[Content, ...]] = reactive((), layout=True)

    def _separator(self) -> Content:  # noqa: PLR6301 — reads the active glyph set
        """Return the styled separator drawn between two segments."""
        return Content(f" {get_glyphs().bullet} ")

    def _chain(self, count: int) -> Content:
        """Join the first `count` segments with bullet separators.

        Returns:
            Joined metric segments.
        """
        return self._separator().join(self.segments[:count])

    def get_content_width(self, container: Size, viewport: Size) -> int:  # noqa: ARG002
        """Return the intrinsic width of the full chain so `width: auto` works.

        Args:
            container: Size of the container.
            viewport: Size of the viewport.

        Returns:
            Cell width of every segment joined by the separator.
        """
        if not self.segments:
            return 0
        return self._chain(len(self.segments)).cell_length

    def render(self) -> RenderResult:
        """Render as many leading segments as the available width allows.

        Returns:
            The joined chain, shortened from the tail until it fits.
        """
        width = self.content_size.width
        if not self.segments or width <= 0:
            return Content("")
        for count in range(len(self.segments), 0, -1):
            chain = self._chain(count)
            if chain.cell_length <= width:
                return chain
        ellipsis = get_glyphs().ellipsis
        if width <= len(ellipsis):
            return Content("")
        first = self._chain(1).truncate(width - len(ellipsis))
        return first + ellipsis


class StatusBar(Vertical):
    """Two-line status bar for session identity and runtime metrics."""

    DEFAULT_CSS = """
    StatusBar {
        height: 2;
        dock: bottom;
        background: $background;
    }

    StatusBar .status-session,
    StatusBar .status-metrics {
        width: 1fr;
        height: 1;
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

    StatusBar .status-mode.shell-incognito {
        background: $mode-incognito;
        color: $background;
        text-style: bold;
    }

    StatusBar .status-auto-approve {
        width: auto;
        padding: 0 1;
        margin-right: 1;
    }

    StatusBar .status-auto-approve.yolo {
        background: $error;
        color: white;
        text-style: bold;
    }

    StatusBar .status-auto-approve.auto {
        background: $success;
        color: $background;
    }

    StatusBar .status-auto-approve.manual {
        background: $warning;
        color: $background;
    }

    StatusBar .status-connection {
        width: auto;
        padding: 0 1 0 0;
        color: $warning;
        text-style: bold;
    }

    StatusBar .status-message {
        width: auto;
        padding: 0 1 0 0;
        color: $text-muted;
    }

    StatusBar .status-message.thinking {
        color: $warning;
    }

    StatusBar .status-cwd {
        width: auto;
        max-width: 45%;
        padding: 0 1 0 0;
        color: $text-muted;
        overflow-x: hidden;
        text-overflow: ellipsis;
        text-wrap: nowrap;
    }

    StatusBar .status-branch {
        width: 1fr;
        min-width: 0;
        overflow-x: hidden;
        text-wrap: nowrap;
    }

    StatusBar .status-cache-line {
        width: 1fr;
        min-width: 0;
        padding: 0;
        color: $text-muted;
    }

    StatusBar .status-context-line {
        width: auto;
        max-width: 55%;
        min-width: 0;
        padding: 0 0 0 2;
        color: $text-muted;
        text-align: right;
    }

    StatusBar .status-rubric {
        width: auto;
        padding: 0 0 0 2;
        color: $success;
        text-style: bold;
    }

    StatusBar ModelLabel {
        width: auto;
        max-width: 40%;
        min-width: 0;
        padding: 0 0 0 2;
        color: $text-muted;
        text-align: right;
    }

    StatusBar BranchLabel {
        color: $text-muted;
        padding: 0;
    }
    """
    """Mode badges color the input mode; the approval mode is colored text."""

    mode: reactive[str] = reactive("normal", init=False)
    status_message: reactive[str] = reactive("", init=False)
    connection_state: reactive[ConnectionState] = reactive("", init=False)
    queued_count: reactive[int] = reactive(0, init=False)
    approval_mode: reactive[str] = reactive(default="manual", init=False)
    cwd: reactive[str] = reactive("", init=False)
    branch: reactive[str] = reactive("", init=False)
    tokens: reactive[int] = reactive(0, init=False)
    cost_usd: reactive[float] = reactive(0.0, init=False)
    rubric_label: reactive[str] = reactive("", init=False)

    def __init__(self, cwd: str | Path | None = None, **kwargs: Any) -> None:
        """Initialize the status bar.

        Args:
            cwd: Current working directory to display
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        # Store initial cwd - will be used in compose()
        self._initial_cwd = str(cwd) if cwd else str(Path.cwd())
        self._hide_cwd = is_env_truthy(HIDE_CWD)
        self._hide_git_branch = is_env_truthy(HIDE_GIT_BRANCH)
        self._spinner = Spinner()
        self._spinner_timer: Timer | None = None
        self._busy_message = ""
        self.context_limit: int | None = None
        self.cache_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self._status_by_source: dict[StatusMessageSource, str] = {
            "agent": "",
            "hooks": "",
        }

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """Compose the status bar layout.

        Yields:
            The model/workspace line followed by cache and context metrics.
        """
        with Horizontal(classes="status-session"):
            yield Static("", classes="status-mode normal", id="mode-indicator")
            yield Static(
                "manual",
                classes="status-auto-approve manual",
                id="auto-approve-indicator",
            )
            yield Static("", classes="status-cwd", id="cwd-display")
            yield BranchLabel(classes="status-branch", id="branch-display")
            yield Static("", classes="status-rubric", id="rubric-display")
            yield ModelLabel(id="model-display")
        with Horizontal(classes="status-metrics"):
            yield Static("", classes="status-connection", id="connection-indicator")
            yield Static("", classes="status-message", id="status-message")
            yield MetricsLine(classes="status-cache-line", id="cache-display")
            yield MetricsLine(classes="status-context-line", id="tokens-display")

    _CWD_WIDTH_THRESHOLD = 70
    """Hide cwd display below this terminal width."""

    def on_resize(self, event: events.Resize) -> None:
        """Hide the cwd on very narrow terminals.

        The git branch stays visible at any width (unless disabled via
        `HIDE_GIT_BRANCH`) and ellipsizes to fit; only the cwd is dropped
        outright to reclaim space when the terminal gets narrow.
        """
        width = event.size.width
        self._set_cwd_visible(not self._hide_cwd and width >= self._CWD_WIDTH_THRESHOLD)

    def _set_cwd_visible(self, visible: bool) -> None:
        """Show or hide the cwd."""
        with suppress(NoMatches):
            self.query_one("#cwd-display", Static).display = visible

    def on_unmount(self) -> None:
        """Stop the spinner timer so it can't tick on a detached widget."""
        self._stop_spinner()

    def on_mount(self) -> None:
        """Set reactive values after mount to trigger watchers safely."""
        from deepagents_code.config import settings

        self.cwd = self._initial_cwd
        if self._hide_cwd:
            self._set_cwd_visible(False)
        if self._hide_git_branch:
            with suppress(NoMatches):
                self.query_one("#branch-display", BranchLabel).display = False
        # Set initial model display
        label = self.query_one("#model-display", ModelLabel)
        label.provider = settings.model_provider or ""
        label.model = settings.model_name or ""
        self.set_context_limit(settings.model_context_limit)
        with suppress(NoMatches):
            self.query_one("#rubric-display", Static).display = False
        # Reactives are `init=False`, so the connection watcher never fires on
        # mount; render once to hide the empty indicator (and its padding).
        self._render_connection()
        self.watch_status_message(self.status_message)
        self._refresh_metrics()

    def watch_mode(self, mode: str) -> None:
        """Update mode indicator when mode changes."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("normal", "shell", "command", "shell-incognito")

        if mode == "shell":
            indicator.update("SHELL")
            indicator.add_class("shell")
        elif mode == "shell_incognito":
            indicator.update("SHELL")
            indicator.add_class("shell-incognito")
        elif mode == "command":
            indicator.update("CMD")
            indicator.add_class("command")
        else:
            indicator.update("")
            indicator.add_class("normal")

    def watch_approval_mode(self, new_value: str) -> None:
        """Update the three-state approval indicator."""
        try:
            indicator = self.query_one("#auto-approve-indicator", Static)
        except NoMatches:
            return
        indicator.remove_class("manual", "auto", "yolo")
        mode = new_value if new_value in {"manual", "auto", "yolo"} else "manual"
        indicator.update("YOLO" if mode == "yolo" else mode)
        indicator.add_class(mode)

    def watch_cwd(self, new_value: str) -> None:
        """Update cwd display when it changes."""
        try:
            display = self.query_one("#cwd-display", Static)
        except NoMatches:
            return
        display.update(self._format_cwd(new_value))

    def watch_branch(self, new_value: str) -> None:
        """Update branch display when it changes."""
        try:
            display = self.query_one("#branch-display", BranchLabel)
        except NoMatches:
            return
        display.branch = new_value

    def watch_status_message(self, new_value: str) -> None:
        """Update status message display."""
        if self._busy_message:
            # The busy indicator owns the status-message slot while active;
            # defer regular status updates until `set_busy("")` clears it.
            return
        try:
            msg_widget = self.query_one("#status-message", Static)
        except NoMatches:
            return

        msg_widget.remove_class("thinking")
        # Hide when empty so the widget's padding doesn't reserve a blank gap
        # in the footer (mirrors the connection indicator).
        msg_widget.display = bool(new_value)
        if new_value:
            # Plain Content: hook-configured statusMessage may contain brackets.
            msg_widget.update(Content(new_value))
            if "thinking" in new_value.lower() or "executing" in new_value.lower():
                msg_widget.add_class("thinking")
        else:
            msg_widget.update("")

    def watch_connection_state(self, _new_value: ConnectionState) -> None:
        """Start or stop the spinner and re-render when connection state changes."""
        self._sync_spinner()
        self._render_connection()

    def watch_queued_count(self, _new_value: int) -> None:
        """Re-render the connection indicator when the queued count changes."""
        self._render_connection()

    def _spinner_active(self) -> bool:
        """Whether any indicator (connection or busy) needs the shared spinner.

        Returns:
            `True` when a connection state or a busy message is active.
        """
        return bool(self.connection_state) or bool(self._busy_message)

    def _sync_spinner(self) -> None:
        """Start or stop the shared spinner to match connection/busy state."""
        if self._spinner_active():
            self._start_spinner()
        else:
            self._stop_spinner()

    def _start_spinner(self) -> None:
        """Begin cycling the shared spinner frames.

        No-op when not yet running (e.g. before mount) since `set_interval`
        requires a live event loop, or when an animation is already active.
        """
        if self._spinner_timer is not None or not self._running:
            return
        # 0.1s mirrors LoadingWidget so this spinner ticks in step with the
        # in-thread "Thinking" spinner.
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

    def _stop_spinner(self) -> None:
        """Stop the spinner animation and reset to the first frame."""
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self._spinner = Spinner()

    def _tick_spinner(self) -> None:
        """Advance the spinner frame and re-render the animated indicators."""
        self._spinner.next_frame()
        self._render_connection()
        self._render_busy()

    def _render_connection(self) -> None:
        """Render the combined connection + queued-count indicator text."""
        try:
            widget = self.query_one("#connection-indicator", Static)
        except NoMatches:
            return

        parts: list[str] = []
        if self.connection_state == "reconnecting":
            parts.append(f"{self._spinner.current_frame()} Reconnecting")
        elif self.connection_state == "resuming":
            parts.append(f"{self._spinner.current_frame()} Resuming")
        elif self.connection_state == "connecting":
            parts.append(f"{self._spinner.current_frame()} Connecting")
        if self.queued_count > 0:
            label = "message" if self.queued_count == 1 else "messages"
            parts.append(f"{self.queued_count} {label} queued")
        separator = f" {get_glyphs().bullet} "
        text = separator.join(parts)
        widget.display = bool(text)
        widget.update(text)

    def _render_busy(self) -> None:
        """Render the animated busy indicator into the status-message slot."""
        if not self._busy_message:
            return
        try:
            widget = self.query_one("#status-message", Static)
        except NoMatches:
            return
        widget.remove_class("thinking")
        widget.display = True
        frame = self._spinner.current_frame()
        widget.update(Content.assemble(frame, " ", Content(self._busy_message)))

    def set_busy(self, message: str) -> None:
        """Show or clear an animated busy indicator in the status-message slot.

        Reuses the shared status-bar spinner so heavier UI operations (e.g. a
        model switch that imports a provider package) show activity instead of
        appearing to hang.

        Args:
            message: Busy text to animate with a spinner, or empty string to
                clear it and restore the regular status message.
        """
        self._busy_message = message
        self._sync_spinner()
        if message:
            self._render_busy()
        else:
            self.watch_status_message(self.status_message)

    def set_connection(self, state: ConnectionState) -> None:
        """Set the connection indicator state.

        Args:
            state: One of `''` (clear), `'connecting'`, `'reconnecting'`, or
                `'resuming'`.

        Raises:
            ValueError: If `state` is not a recognized connection state.
        """
        if state not in CONNECTION_STATES:
            msg = f"Unknown connection state: {state!r}"
            raise ValueError(msg)
        self.connection_state = state

    def set_queued(self, count: int) -> None:
        """Set the number of messages waiting in the queue.

        Args:
            count: Count of queued messages (negative values clamp to `0`).
        """
        self.queued_count = max(count, 0)

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

    @property
    def auto_approve(self) -> bool:
        """Whether unrestricted compatibility mode is active."""
        return self.approval_mode == "yolo"

    @auto_approve.setter
    def auto_approve(self, enabled: bool) -> None:
        self.set_approval_mode("yolo" if enabled else "manual")

    def set_approval_mode(self, mode: str) -> None:
        """Set the approval mode.

        Args:
            mode: `manual`, `auto`, or `yolo`.
        """
        self.approval_mode = mode if mode in {"manual", "auto", "yolo"} else "manual"

    def set_auto_approve(self, *, enabled: bool) -> None:
        """Set the compatibility unrestricted state.

        Args:
            enabled: Whether unrestricted mode is enabled.
        """
        self.set_approval_mode("yolo" if enabled else "manual")

    def set_status_message(
        self,
        message: str,
        *,
        source: StatusMessageSource = "agent",
    ) -> None:
        """Set the status message with explicit source ownership.

        Each source stores its own message. Hooks take display priority while
        they have a non-empty message; clearing hooks restores any stored agent
        message instead of blanking the slot. Agent writes never erase an active
        hook status, and hook completion never erases a stored agent status.

        Args:
            message: Status message to display (empty string to clear).
            source: Subsystem that owns this write (`agent` or `hooks`).
        """
        self._status_by_source[source] = message
        self.status_message = (
            self._status_by_source["hooks"] or self._status_by_source["agent"]
        )

    _approximate: bool = False
    """Append "+" to the token count to signal that the displayed value is stale.

    (The actual context is larger because the generation was interrupted before
    the model reported final usage.)
    """
    _has_token_count: bool = False
    """Whether the status bar has displayed a real token count this session."""

    _tokens_pending: bool = False
    """Whether the accurate token count for the current turn is still pending.

    A cost update can arrive mid-turn, and it re-renders the shared token/cost
    slot. Without this flag that re-render would replace the `... tokens`
    placeholder with the *previous* turn's count -- the stale value the
    placeholder exists to hide.
    """

    def watch_tokens(self, new_value: int) -> None:
        """Update the combined token and cost display when tokens change."""
        self._render_tokens(new_value, approximate=self._approximate)

    def watch_cost_usd(self, _new_value: float) -> None:
        """Update the combined token and cost display when cost changes."""
        self._render_tokens(self.tokens, approximate=self._approximate)

    def _refresh_metrics(self) -> None:
        """Re-render the metrics line from the current reactive values."""
        self._render_tokens(self.tokens, approximate=self._approximate)

    def watch_rubric_label(self, new_value: str) -> None:
        """Update rubric display when active rubric state changes."""
        try:
            display = self.query_one("#rubric-display", Static)
        except NoMatches:
            return
        display.display = bool(new_value)
        display.update(new_value)

    _CONTEXT_WARNING_PERCENT = 60.0
    """Context usage at which the percentage turns from calm to caution."""

    _CONTEXT_CRITICAL_PERCENT = 80.0
    """Context usage at which the percentage turns to alert."""

    def _percent_color(self, percent: float) -> str:
        """Return the color that encodes how full the context window is."""
        colors = theme.get_theme_colors(self)
        if percent > self._CONTEXT_CRITICAL_PERCENT:
            return colors.error
        if percent > self._CONTEXT_WARNING_PERCENT:
            return colors.warning
        return colors.muted

    def _context_segment(self, count: int, *, approximate: bool = False) -> Content:
        """Build the context percentage and absolute-usage segment.

        Returns:
            Styled context usage.
        """
        pending = self._tokens_pending
        suffix = "+" if approximate else ""
        if pending:
            percent_content = Content("...")
            count_text = "..."
        elif self.context_limit is None:
            percent_content = Content("0%" if count <= 0 else "--")
            count_text = f"{_compact_tokens(count)}{suffix}"
        else:
            percent = min(100.0, max(0.0, count / self.context_limit * 100))
            percent_content = Content.styled(
                f"{percent:.0f}%", self._percent_color(percent)
            )
            count_text = f"{_compact_tokens(count)}{suffix}"
        muted = theme.get_theme_colors(self).muted
        return Content.assemble(
            Content.styled("Context:", muted),
            " ",
            percent_content,
            " / ",
            Content.styled("Tokens:", muted),
            f" {count_text}",
        )

    def _cache_segment(self) -> Content:
        """Build the active thread's cache segment.

        Returns:
            Styled cache usage.
        """
        colors = theme.get_theme_colors(self)
        hit_rate = Content("")
        if self.cache_input_tokens:
            cached = min(self.cache_read_tokens, self.cache_input_tokens)
            percent = cached / self.cache_input_tokens * 100
            if percent < 60.0:  # noqa: PLR2004  # cache alert threshold
                color = colors.error
            elif percent < 90.0:  # noqa: PLR2004  # cache warning threshold
                color = colors.warning
            else:
                color = colors.muted
            hit_rate = Content.styled(f"{percent:.0f}% hit", color)
        elif not self.cache_read_tokens and not self.cache_write_tokens:
            return Content("")
        details = (
            f"{_compact_tokens(self.cache_read_tokens)} read"
            f" / {_compact_tokens(self.cache_write_tokens)} write"
        )
        return Content.assemble(
            Content.styled("Cache", colors.muted),
            " ",
            hit_rate,
            f" {get_glyphs().bullet} " if hit_rate.plain else "",
            details,
        )

    def _cost_text(self) -> str:
        """Format cumulative cost, including the initial zero state.

        Returns:
            Formatted cost.
        """
        return format_cost(self.cost_usd)

    def _render_tokens(self, count: int, *, approximate: bool = False) -> None:
        """Render cache left and context/cost right on the metrics line."""
        try:
            cache_display = self.query_one("#cache-display", MetricsLine)
            context_display = self.query_one("#tokens-display", MetricsLine)
        except NoMatches:
            return

        cost = self._cost_text()
        context_segments = tuple(
            segment
            for segment in (
                self._context_segment(count, approximate=approximate),
                Content(cost) if cost else Content(""),
            )
            if segment.plain
        )
        cache = self._cache_segment()
        cache_display.segments = (cache,) if cache.plain else ()
        context_display.segments = context_segments

    def set_rubric_label(self, label: str) -> None:
        """Set the rubric status label.

        Args:
            label: Label to display, or empty string to hide the badge.
        """
        self.rubric_label = label

    def set_tokens(self, count: int, *, approximate: bool = False) -> None:
        """Set the token count.

        Forces a display refresh even when the value is unchanged. During
        streaming, `show_pending_tokens` replaces the widget text without
        changing the reactive token value, so a later update with the same
        count still needs to re-render the exact count.

        Args:
            count: Current context token count.
            approximate: Append "+" to indicate the count is stale.
        """
        self._approximate = approximate
        self._has_token_count = count > 0
        # The accurate count has arrived, so stop suppressing it.
        self._tokens_pending = False
        if self.tokens == count:
            # Reactive dedup would skip the watcher — call render directly.
            self._render_tokens(count, approximate=approximate)
        else:
            # Reactive assignment triggers watch_tokens, which reads
            # self._approximate for the suffix.
            self.tokens = count

    def set_context_limit(self, limit: int | None) -> None:
        """Set the active model's context limit."""
        self.context_limit = limit if isinstance(limit, int) and limit > 0 else None
        self._refresh_metrics()

    def set_cache_tokens(
        self,
        read_tokens: int,
        write_tokens: int,
        *,
        input_tokens: int = 0,
    ) -> None:
        """Set cumulative input and cache token counts for the active thread.

        Args:
            read_tokens: Input tokens served from the provider cache.
            write_tokens: Input tokens written to the provider cache.
            input_tokens: Inclusive input-token total used as the hit-rate
                denominator.
        """
        reads = max(read_tokens, 0)
        writes = max(write_tokens, 0)
        inputs = max(input_tokens, 0)
        self.cache_input_tokens = inputs
        self.cache_read_tokens = reads
        self.cache_write_tokens = writes
        self._refresh_metrics()

    def set_cost(self, cost_usd: float) -> None:
        """Set the cumulative thread cost shown beside context tokens.

        Args:
            cost_usd: Cumulative estimated cost in US dollars.
        """
        if self.cost_usd == cost_usd:
            self._refresh_metrics()
        else:
            self.cost_usd = cost_usd

    def show_pending_tokens(self) -> None:
        """Show pending tokens while preserving the cumulative cost."""
        if not self._has_token_count:
            return
        # Latch the placeholder so a mid-turn cost refresh keeps it instead of
        # re-rendering the previous turn's count.
        self._tokens_pending = True
        self._refresh_metrics()

    def set_model(self, *, provider: str, model: str, effort: str = "") -> None:
        """Update the model display text.

        Args:
            provider: Model provider name (e.g., `'anthropic'`).
            model: Model name (e.g., `'claude-sonnet-4-5'`).
            effort: Reasoning effort label to display (per-session override or
                provider default), or empty when none applies.
        """
        label = self.query_one("#model-display", ModelLabel)
        label.provider = provider
        label.model = model
        label.effort = effort
