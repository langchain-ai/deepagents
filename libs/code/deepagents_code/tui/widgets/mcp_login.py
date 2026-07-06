"""In-TUI MCP OAuth login modal.

`MCPLoginScreen` is both a Textual `ModalScreen` and an implementation of
`OAuthInteraction`. The login worker awaits its interaction methods while
the user sees and acts on the modal's widgets — authorize URLs become
clickable links, paste-back callback URLs go through an inline input row,
device-code instructions render inline, and the modal closes itself on
success.

The screen runs on the Textual event loop (same loop as the worker), so
methods called from the worker can `await` modal-bound futures directly
without `app.call_from_thread`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar, Literal

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
    MouseMove,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Input, Static

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.tui.widgets._links import open_style_link
from deepagents_code.tui.widgets.loading import Spinner

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.timer import Timer


LoginOutcome = Literal["success", "cancelled", "failed"]
"""Discriminator returned by the modal when it dismisses."""


class MCPLoginCancelledError(RuntimeError):
    """Raised by `MCPLoginScreen.action_cancel` when the user cancels the flow."""


_PROMPT_CALLBACK = "Paste the full callback URL after approving in the browser:"


class MCPLoginScreen(ModalScreen[LoginOutcome]):
    """Modal that renders the OAuth login flow and collects user input.

    Implements the `OAuthInteraction` Protocol structurally so a
    `mcp_auth.login(..., ui=screen)` call drives the same modal. Each
    interaction method updates a status line, a clickable link area, and
    an inline input prompt for the callback URL. Slack workspace selection
    is deferred to Slack's browser page rather than prompted inline.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "toggle_authorize_url", "Toggle URL", show=False),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]
    """Esc unblocks the worker; the worker performs the actual shutdown.

    Cancellation completes any outstanding input future with
    `MCPLoginCancelledError`. The worker (`_run_mcp_login_worker`) sees
    that exception, calls `finish(success=False)`, and tears down the
    OAuth handshake. Doing the teardown here would race the worker.
    """

    CSS = """
    MCPLoginScreen {
        align: center middle;
    }

    MCPLoginScreen > Vertical {
        width: 80;
        max-width: 92%;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MCPLoginScreen .ml-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    MCPLoginScreen .ml-status {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    MCPLoginScreen .ml-link {
        height: auto;
        color: $accent;
        margin-bottom: 1;
    }

    MCPLoginScreen .ml-history {
        height: auto;
        max-height: 8;
        background: $surface-lighten-1;
        margin-bottom: 1;
    }

    MCPLoginScreen .ml-history-line {
        height: auto;
        color: $text-muted;
        padding: 0 1;
    }

    MCPLoginScreen .ml-prompt {
        height: 1;
        color: $text;
    }

    MCPLoginScreen #ml-input {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    MCPLoginScreen #ml-input:focus {
        border: solid $primary;
    }

    MCPLoginScreen .ml-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, server_name: str) -> None:
        """Initialize a login modal for `server_name`.

        Args:
            server_name: MCP server name shown in the modal title.
        """
        super().__init__()
        self._server_name = server_name
        self._status = f"Starting OAuth login for {server_name}…"
        self._title_widget: Static | None = None
        self._status_widget: Static | None = None
        self._link_widget: Static | None = None
        self._history_widget: VerticalScroll | None = None
        self._prompt_widget: Static | None = None
        self._input_widget: Input | None = None
        self._help_widget: Static | None = None
        self._spinner = Spinner()
        self._spinner_timer: Timer | None = None
        self._authorize_url: str | None = None
        self._authorize_url_opened_in_browser = False
        self._authorize_url_expanded = False
        self._waiting_for_authorization = False

        # Each prompt method blocks on a fresh Future; cancellation completes
        # it with MCPLoginCancelledError so the worker unblocks rather than
        # hanging on a dismissed modal.
        self._pending_input: asyncio.Future[str] | None = None
        self._cancelled = False
        self._done = False
        self._outcome: LoginOutcome | None = None
        self._last_history_line: str | None = None

    @property
    def is_done(self) -> bool:
        """`True` once the modal has been told to finish (success or failure).

        Public accessor for callers that need to coordinate teardown from
        outside the screen, e.g. the worker's `BaseException` branch that
        unblocks dismiss without re-finishing.
        """
        return self._done

    def compose(self) -> ComposeResult:
        """Compose the modal body inside a `Vertical` container.

        Yields:
            Title, status, optional link, history, prompt label, input,
            and help footer widgets, all parented inside a `Vertical`.
        """
        with Vertical():
            self._title_widget = Static(
                Content.from_markup("MCP login: $name", name=self._server_name),
                classes="ml-title",
                markup=False,
            )
            yield self._title_widget
            self._status_widget = Static(
                self._status, classes="ml-status", markup=False
            )
            yield self._status_widget
            self._link_widget = Static("", classes="ml-link")
            self._link_widget.display = False
            yield self._link_widget
            self._history_widget = VerticalScroll(classes="ml-history")
            self._history_widget.display = False
            yield self._history_widget
            self._prompt_widget = Static("", classes="ml-prompt", markup=False)
            self._prompt_widget.display = False
            yield self._prompt_widget
            self._input_widget = Input(id="ml-input")
            self._input_widget.display = False
            yield self._input_widget
            self._help_widget = Static("Esc to cancel", classes="ml-help")
            yield self._help_widget

    def on_mount(self) -> None:
        """Apply ASCII border when configured and start the spinner ticker."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.primary)
        self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

    def on_click(self, event: Click) -> None:
        """Open style links, or expand/collapse the manual authorize URL."""
        if (
            event.widget is self._link_widget
            and self._authorize_url is not None
            and self._authorize_url_opened_in_browser
            and not event.style.link
        ):
            self._toggle_authorize_url()
            event.stop()
            return
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer over links and the manual URL disclosure row."""
        self.styles.pointer = (
            "pointer"
            if event.style.link or event.widget is self._link_widget
            else "default"
        )

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the modal."""
        self.styles.pointer = "default"

    # ------------------------------------------------------------------
    # OAuthInteraction implementation.
    # ------------------------------------------------------------------

    async def show_authorize_url(self, url: str, *, opened_in_browser: bool) -> None:
        """Render browser-open status and only reveal the URL when needed."""
        self._authorize_url = url
        self._authorize_url_opened_in_browser = opened_in_browser
        self._authorize_url_expanded = not opened_in_browser
        self._waiting_for_authorization = opened_in_browser
        if opened_in_browser:
            self._render_authorization_wait_status(self._spinner.next_frame())
        else:
            self._set_status(
                f"Open the authorization URL manually to connect {self._server_name}.",
            )
        self._render_authorize_url()

    async def request_callback_url(self) -> str:
        """Wait for the user to paste back the OAuth callback URL.

        Returns:
            The trimmed callback URL.
        """
        return await self._await_input(_PROMPT_CALLBACK)

    async def show_device_code(
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Render RFC 8628 device-code instructions inline."""
        self._waiting_for_authorization = False
        self._set_status(
            f"Visit the URL below and enter the code (expires in {expires_in}s):",
        )
        if self._link_widget is not None:
            self._link_widget.display = True
            self._link_widget.update(
                Content.assemble(
                    ("Verification URL: ", "bold"),
                    (verification_uri, TStyle(link=verification_uri, underline=True)),
                    ("\nUser code: ", "bold"),
                    (user_code, "bold"),
                ),
            )
        self._append_history(
            f"Device code: visit {verification_uri} and enter {user_code}",
        )

    async def show_success(self, message: str) -> None:
        """Render a success status line without leaving OAuth fallback UI behind."""
        self._waiting_for_authorization = False
        self._hide_authorize_url()
        self._set_status(message)

    async def show_notice(self, message: str) -> None:
        """Append a progress notice without disrupting the active prompt."""
        self._append_history(message)

    async def show_error(self, message: str) -> None:
        """Render a fatal (flow-ending) error status line and history entry."""
        self._waiting_for_authorization = False
        self._set_status(message)
        self._append_history(message)

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _hide_authorize_url(self) -> None:
        """Hide any OAuth authorization URL or fallback affordance."""
        self._authorize_url = None
        self._authorize_url_opened_in_browser = False
        self._authorize_url_expanded = False
        self._waiting_for_authorization = False
        self._render_authorize_url()
        self._set_help("Esc to cancel")

    def _toggle_authorize_url(self) -> None:
        """Toggle the manual authorize URL when the fallback affordance is active."""
        if (
            self._pending_input is not None
            or self._authorize_url is None
            or not self._authorize_url_opened_in_browser
        ):
            return
        self._authorize_url_expanded = not self._authorize_url_expanded
        self._render_authorize_url()

    def _render_authorize_url(self) -> None:
        """Render the manual authorize URL affordance."""
        if self._link_widget is None:
            return
        url = self._authorize_url
        if url is None:
            self._link_widget.display = False
            self._link_widget.update("")
            return
        self._link_widget.display = True
        glyphs = get_glyphs()
        if self._authorize_url_opened_in_browser and not self._authorize_url_expanded:
            self._link_widget.update(
                Content.assemble(
                    ("Having trouble? Show manual authorization URL ", "dim"),
                    (glyphs.cursor, "dim"),
                ),
            )
            self._set_help("Enter to show URL · Esc to cancel")
            return
        prefix = (
            f"Having trouble? Hide manual authorization URL {glyphs.arrow_down}\n"
            if self._authorize_url_opened_in_browser
            else "Authorization URL:\n"
        )
        self._link_widget.update(
            Content.assemble(
                (prefix, "bold"),
                (url, TStyle(link=url, underline=True)),
            ),
        )
        if self._authorize_url_opened_in_browser:
            self._set_help("Enter to hide URL · Esc to cancel")
        else:
            self._set_help("Esc to cancel")

    def _render_authorization_wait_status(self, frame: str) -> None:
        """Render the browser-opened waiting state with an animated status line."""
        self._set_status(
            f"We opened your browser to connect {self._server_name}.\n"
            "Complete authorization there to continue.\n\n"
            f"Status: {frame} Waiting…",
        )

    def _set_status(self, message: str) -> None:
        """Update the top status line."""
        self._status = message
        if self._status_widget is not None:
            self._status_widget.update(Content.from_markup("$msg", msg=message))

    def _set_help(self, message: str) -> None:
        """Update the footer help text."""
        if self._help_widget is not None:
            self._help_widget.update(message)

    def _hide_history(self) -> None:
        """Hide the history pane for clean terminal states."""
        if self._history_widget is not None:
            self._history_widget.display = False
        self._last_history_line = None

    def _append_history(self, line: str) -> None:
        """Append a line to the scrolling history pane."""
        if self._history_widget is None or line == self._last_history_line:
            return
        self._last_history_line = line
        self._history_widget.display = True
        self._history_widget.mount(
            Static(line, classes="ml-history-line", markup=False)
        )
        self._history_widget.scroll_end(animate=False)

    async def _await_input(self, prompt: str) -> str:
        """Show `prompt`, wait for `Enter`, and return the typed value.

        Returns:
            The raw input value the user submitted.

        Raises:
            MCPLoginCancelledError: When the modal is cancelled before or
                during submission.
            RuntimeError: When a concurrent prompt is already active.
        """
        if self._cancelled:
            msg = "MCP login was cancelled before the prompt could be shown."
            raise MCPLoginCancelledError(msg)
        if self._pending_input is not None:
            msg = (
                "MCP login modal cannot have two concurrent input prompts; "
                "the previous prompt was not resolved."
            )
            raise RuntimeError(msg)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_input = future
        if self._prompt_widget is not None:
            self._prompt_widget.display = True
            self._prompt_widget.update(prompt)
        if self._input_widget is not None:
            self._input_widget.value = ""
            self._input_widget.display = True
            self._input_widget.focus()
        try:
            return await future
        finally:
            self._pending_input = None
            if self._input_widget is not None:
                self._input_widget.display = False
            if self._prompt_widget is not None:
                self._prompt_widget.display = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Resolve the active prompt with the submitted value."""
        if event.input.id != "ml-input":
            return
        future = self._pending_input
        if future is not None and not future.done():
            future.set_result(event.value)

    def action_toggle_authorize_url(self) -> None:
        """Toggle the manual authorize URL fallback via Enter."""
        self._toggle_authorize_url()

    def action_cancel(self) -> None:
        """Cancel the login flow.

        Sets a cancelled flag, completes any outstanding prompt with
        `MCPLoginCancelledError` so the worker unblocks, and dismisses
        the modal with the `cancelled` outcome.
        """
        if self._done:
            return
        self._cancelled = True
        self._done = True
        self._outcome = "cancelled"
        future = self._pending_input
        if future is not None and not future.done():
            future.set_exception(
                MCPLoginCancelledError("MCP login was cancelled by the user.")
            )
        self._stop_spinner_timer()
        self.dismiss("cancelled")

    def finish(self, *, success: bool, message: str | None = None) -> None:
        """Close the modal from the worker, reporting the final outcome.

        Dismiss is deferred by 0.6s so the user sees the final status
        before the modal disappears.

        Args:
            success: `True` when login succeeded; drives the dismiss value.
            message: Optional final status line shown before close.
        """
        if self._done:
            return
        self._done = True
        self._outcome = "success" if success else "failed"
        self._stop_spinner_timer()
        self._waiting_for_authorization = False
        if success:
            self._hide_authorize_url()
            self._hide_history()
        if message is not None:
            self._set_status(message)
            if not success:
                self._append_history(message)
        glyphs = get_glyphs()
        marker = glyphs.checkmark if success else glyphs.error
        if self._title_widget is not None:
            self._title_widget.update(
                Content.from_markup(
                    "MCP login: $name $marker",
                    name=self._server_name,
                    marker=marker,
                )
            )

        def _deferred_dismiss() -> None:
            # Must be a def (not a lambda): Textual's `_invoke` auto-awaits
            # any awaitable return, and `dismiss()` returns `AwaitComplete`;
            # awaiting it inside the screen's own message pump raises
            # `ScreenError`. A `None`-returning def discards the awaitable.
            self.dismiss(self._outcome or "failed")

        self.set_timer(0.6, _deferred_dismiss)

    def _tick_spinner(self) -> None:
        """Advance the status-line spinner while waiting for browser auth."""
        if self._done or not self._waiting_for_authorization:
            return
        self._render_authorization_wait_status(self._spinner.next_frame())

    def _stop_spinner_timer(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
