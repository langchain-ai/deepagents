"""xAI OAuth sign-in screen, reachable via `/auth` -> `xai_oauth`.

Structurally this mirrors `codex_auth.py` (a modal that drives the sign-in
flow on a worker and dismisses on completion), but the flow itself is a
device-code grant rather than a browser-loopback PKCE exchange, so the modal
content mirrors `tui.widgets.mcp_login.MCPLoginScreen.show_device_code`
instead: a short user code and a verification URL the user visits and enters
it on, not an auto-opened browser tab.

Security notes:

- The verification URL and user code shown inline carry no secrets — the
    device code alone cannot mint a token without the user completing the
    approval step in a browser.
- The success / error messages reported back via `notify` never include the
    access token or refresh token.
- This OAuth surface is unofficial and undocumented by xAI (see
    `deepagents_code.integrations.xai_oauth`'s module docstring); the modal
    copy says so explicitly, and signing in also requires a SuperGrok / X
    Premium+ entitlement on the underlying xAI account.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.color import Color as TColor
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Static
from textual.worker import Worker, WorkerFailed, WorkerState

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.integrations import xai_oauth as xai_integration
from deepagents_code.model_config import clear_caches
from deepagents_code.tui.widgets._links import open_style_link

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click, MouseMove

logger = logging.getLogger(__name__)


class _ScreenInteraction(xai_integration.XaiLoginInteraction):
    """Bridge `XaiLoginInteraction` callbacks into the modal.

    `run_device_login` runs from a Textual *async* worker (not a thread), so
    it shares the app's event loop and these callbacks can mutate widgets
    directly — no `call_from_thread` round-trip required.
    """

    def __init__(self, screen: XaiOAuthAuthScreen) -> None:
        """Bind the interaction to the modal it should drive."""
        self._screen = screen

    async def show_device_code(
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Render the RFC 8628 device-code prompt inline."""
        self._screen.on_device_code(
            verification_uri=verification_uri,
            user_code=user_code,
            expires_in=expires_in,
        )

    async def show_success(self, message: str) -> None:
        """No-op: the worker's completion handles the success toast."""

    async def show_error(self, message: str) -> None:
        """No-op: the worker's error path handles the failure toast."""


class XaiOAuthAuthScreen(ModalScreen[bool]):
    """Run the xAI OAuth Device Authorization Grant inline.

    Dismissal value:

    - `True`: a token was saved (caller should refresh provider lists /
        retry the operation that needed the credential).
    - `False`: the user cancelled, or the flow failed irrecoverably.

    The flow lives in a worker so the modal stays responsive to the cancel
    keybinding while the device flow polls for up to the code's `expires_in`
    window; cancelling stops the worker's `asyncio` task, which interrupts
    the poll loop at its next `await` point (no separate cancel-event
    plumbing is needed the way the Codex loopback flow requires, since the
    device-flow poll has no blocking OS-level callback server to tear down).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    XaiOAuthAuthScreen {
        align: center middle;
    }

    XaiOAuthAuthScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    XaiOAuthAuthScreen .xai-auth-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    XaiOAuthAuthScreen .xai-auth-copy {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    XaiOAuthAuthScreen .xai-auth-status {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    XaiOAuthAuthScreen .xai-auth-code {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    XaiOAuthAuthScreen .xai-auth-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self) -> None:
        """Initialize with no active worker; the flow starts on mount."""
        super().__init__()
        self._worker: Worker[xai_integration.XaiOAuthStatus] | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual handler signature
        """Compose the modal layout.

        Yields:
            Title, copy, status line, device-code line, and help footer
            widgets.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Sign in with xAI",
                classes="xai-auth-title",
            )
            yield Static(
                Content.assemble(
                    "Authorize Deep Agents to call Grok models on your "
                    "behalf via xAI's device sign-in. This uses an "
                    "unofficial, undocumented-by-xAI OAuth surface and "
                    "requires a SuperGrok / X Premium+ entitlement — if "
                    "sign-in fails or is rejected, set XAI_API_KEY instead.",
                ),
                classes="xai-auth-copy",
            )
            yield Static(
                "Requesting a device code...",
                id="xai-auth-status",
                classes="xai-auth-status",
            )
            yield Static(
                "",
                id="xai-auth-code",
                classes="xai-auth-code",
            )
            yield Static(
                f"Esc cancel {glyphs.bullet} visit the URL below and enter the code",
                classes="xai-auth-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed and kick off the OAuth worker."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._worker = self.run_worker(
            self._run_login(),
            name="xai-oauth",
            exclusive=True,
            thread=False,
        )

    def on_click(self, event: Click) -> None:  # noqa: PLR6301 - Textual handler
        """Open the verification URL when the user clicks it."""
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer over the inline verification link."""
        self.styles.pointer = "pointer" if event.style.link else "default"

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the modal."""
        self.styles.pointer = "default"

    async def _run_login(self) -> xai_integration.XaiOAuthStatus:
        """Worker body: drive the device-code flow with our UI hooks.

        Returns:
            The fresh `XaiOAuthStatus` returned by `run_device_login`, used
                by `on_worker_state_changed` to render the success toast.
        """
        status = await xai_integration.run_device_login(_ScreenInteraction(self))
        clear_caches()
        return status

    def on_device_code(
        self, *, verification_uri: str, user_code: str, expires_in: int
    ) -> None:
        """Render the device-code prompt in the modal.

        Called on the event loop from the async sign-in worker (the worker
        is started with `thread=False`), so it can mutate widgets directly.
        """
        status = self.query_one("#xai-auth-status", Static)
        code_label = self.query_one("#xai-auth-code", Static)
        status.update(
            f"Visit the URL below and enter the code (expires in {expires_in}s):",
        )
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in {"ansi-dark", "ansi-light"}
        link_style: str | TStyle = (
            TStyle(bold=True, underline=True, link=verification_uri)
            if ansi
            else TStyle(
                foreground=TColor.parse(colors.primary),
                underline=True,
                link=verification_uri,
            )
        )
        # `Content.assemble` with a (text, style) tuple skips markup parsing,
        # so a URL or code containing `[` cannot crash the renderer.
        code_label.update(
            Content.assemble(
                ("Verification URL: ", "bold"),
                (verification_uri, link_style),
                ("\nCode: ", "bold"),
                (user_code, TStyle(bold=True)),
            ),
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """React to worker completion: notify, then dismiss the modal."""
        if event.worker is not self._worker:
            return
        state = event.state
        if state is WorkerState.SUCCESS:
            self.app.notify("Signed in to xAI.", markup=False)
            self.dismiss(True)
        elif state is WorkerState.CANCELLED:
            self.app.notify("Sign-in cancelled.", markup=False)
            self.dismiss(False)
        elif state is WorkerState.ERROR:
            error = event.worker.error
            inner = (
                getattr(error, "error", error)
                if isinstance(error, WorkerFailed)
                else error
            )
            # Do not log or display the raw error: it may contain sensitive
            # details (token fragments, request/response bodies, internal
            # URLs with query params). Use a fixed, token-safe summary and
            # only log the exception type name for debugging.
            logger.warning(
                "xAI OAuth sign-in failed: %s",
                type(inner).__name__ if inner else "unknown error",
            )
            self.app.notify(
                "Sign-in failed: xAI OAuth login failed. "
                "Check your network connection and try again.",
                severity="error",
                markup=False,
            )
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Cancel the sign-in flow and dismiss the modal."""
        if self._worker is not None:
            self._worker.cancel()
        else:
            self.dismiss(False)
        # `cancel()` triggers `WorkerState.CANCELLED` on the worker, which
        # `on_worker_state_changed` translates into the dismissal; don't
        # dismiss eagerly here in that branch or it would race a success/
        # error path that lands in the same tick.


class XaiOAuthSignedInAction(StrEnum):
    """Outcome of the `XaiOAuthSignedInScreen` quick-action overlay.

    Mirrors `codex_auth.CodexSignedInAction`.
    """

    SIGN_OUT = "signout"
    """Delete the stored xAI OAuth token."""

    REAUTH = "reauth"
    """Open the OAuth flow again (e.g., to switch account)."""


class XaiOAuthSignedInScreen(ModalScreen["XaiOAuthSignedInAction | None"]):
    """Quick-action overlay shown when `xai_oauth` is already signed in.

    Dismissal values:

    - `XaiOAuthSignedInAction.SIGN_OUT`: delete the stored token.
    - `XaiOAuthSignedInAction.REAUTH`: open the OAuth flow again.
    - `None`: close without changes.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("s", "signout", "Sign out", show=False, priority=True),
        Binding("r", "reauth", "Reauth", show=False, priority=True),
    ]

    CSS = """
    XaiOAuthSignedInScreen {
        align: center middle;
    }

    XaiOAuthSignedInScreen > Vertical {
        width: 64;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    XaiOAuthSignedInScreen .xai-signed-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    XaiOAuthSignedInScreen .xai-signed-copy {
        height: auto;
        margin-bottom: 1;
    }

    XaiOAuthSignedInScreen .xai-signed-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual handler signature
        """Compose the overlay.

        Yields:
            Title + body + key-hint widgets.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("xAI sign-in", classes="xai-signed-title")
            yield Static(
                "Signed in to xAI.",
                classes="xai-signed-copy",
            )
            yield Static(
                f"S sign out {glyphs.bullet} R sign in again {glyphs.bullet} Esc close",
                classes="xai-signed-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def action_signout(self) -> None:
        """Dismiss with `SIGN_OUT` so the manager deletes the stored token."""
        self.dismiss(XaiOAuthSignedInAction.SIGN_OUT)

    def action_reauth(self) -> None:
        """Dismiss with `REAUTH` so the manager kicks off a new flow."""
        self.dismiss(XaiOAuthSignedInAction.REAUTH)

    def action_cancel(self) -> None:
        """Close without changes."""
        self.dismiss(None)


__all__ = [
    "XaiOAuthAuthScreen",
    "XaiOAuthSignedInAction",
    "XaiOAuthSignedInScreen",
]
