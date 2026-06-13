"""ChatGPT OAuth sign-in screen, reachable via `/auth` -> `openai_codex`.

Mirrors the MCP loopback flow in `mcp_auth` from the user's POV: a modal
shows progress, surfaces the authorize URL inline (so headless / SSH users
can copy it when the browser launch fails), and dismisses once the OAuth
callback completes. The OAuth primitives themselves (PKCE, callback HTTP
server, token exchange, refresh, atomic file write) are delegated to
`langchain_openai.chatgpt_oauth` via the
`deepagents_code.integrations.openai_codex` adapter.

Security notes:

- The authorize URL displayed inline does not contain secrets — it carries
    only the PKCE *challenge* (the verifier never leaves this process).
- The success / error messages reported back via `notify` never include
    the access token, refresh token, or ID token.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.color import Color as TColor
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

from deepagents_code import theme
from deepagents_code.config import get_glyphs, is_ascii_mode
from deepagents_code.integrations import openai_codex as codex_integration
from deepagents_code.model_config import clear_caches
from deepagents_code.widgets._links import open_style_link

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

logger = logging.getLogger(__name__)


class _ScreenInteraction(codex_integration.CodexLoginInteraction):
    """Bridge `CodexLoginInteraction` callbacks into the modal.

    `run_browser_login` runs from a Textual *async* worker (not a thread),
    so it shares the app's event loop. That means the callbacks land on
    the same thread Textual renders from and can mutate widgets directly —
    no `call_from_thread` round-trip required (using it from the UI thread
    would raise).
    """

    def __init__(self, screen: CodexAuthScreen) -> None:
        """Bind the interaction to the modal it should drive."""
        self._screen = screen

    async def show_authorize_url(  # awaited by the interaction protocol
        self, url: str, *, opened_in_browser: bool
    ) -> None:
        self._screen.on_authorize_url(url, opened_in_browser)


class CodexAuthScreen(ModalScreen[bool]):
    """Run the ChatGPT OAuth Authorization Code Flow with PKCE inline.

    Dismissal value:

    - `True`: a token was saved (caller should refresh provider lists /
        retry the operation that needed the credential).
    - `False`: the user cancelled, or the flow failed irrecoverably.

    The flow lives in a worker so the modal stays responsive to the cancel
    keybinding while `_wait_for_oauth_callback` blocks for up to 5 minutes;
    pressing Esc sets the worker's `cancel_event`, which frees the loopback
    port within one poll interval rather than holding it until the timeout.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    CodexAuthScreen {
        align: center middle;
    }

    CodexAuthScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    CodexAuthScreen .codex-auth-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    CodexAuthScreen .codex-auth-copy {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    CodexAuthScreen .codex-auth-status {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    CodexAuthScreen .codex-auth-url {
        height: auto;
        color: $text;
        margin-bottom: 1;
        text-style: italic;
    }

    CodexAuthScreen .codex-auth-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self) -> None:
        """Initialize with no active worker; the flow starts on mount."""
        super().__init__()
        self._cancel_event = threading.Event()
        self._worker: Worker[codex_integration.CodexAuthStatus] | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual handler signature
        """Compose the modal layout.

        Yields:
            Title, copy, status line, URL line, and help footer widgets.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(
                "Sign in with ChatGPT",
                classes="codex-auth-title",
            )
            yield Static(
                Content.assemble(
                    "Authorize Deep Agents to call ChatGPT Codex models on "
                    "your behalf. We will open your default browser to "
                    "openai.com to sign in.",
                ),
                classes="codex-auth-copy",
            )
            yield Static(
                "Preparing OAuth flow...",
                id="codex-auth-status",
                classes="codex-auth-status",
            )
            yield Static(
                "",
                id="codex-auth-url",
                classes="codex-auth-url",
            )
            yield Static(
                f"Esc cancel {glyphs.bullet} a browser window will open shortly",
                classes="codex-auth-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed and kick off the OAuth worker."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._worker = self.run_worker(
            self._run_login(),
            name="codex-oauth",
            exclusive=True,
            thread=False,
        )

    def on_click(self, event: Click) -> None:  # noqa: PLR6301 - Textual handler
        """Open the authorize URL when the user clicks it."""
        open_style_link(event)

    async def _run_login(self) -> codex_integration.CodexAuthStatus:
        """Worker body: drive the upstream OAuth flow with our UI hooks.

        Returns:
            The fresh `CodexAuthStatus` returned by `run_browser_login`, used
                by `on_worker_state_changed` to render the success toast.

        Raises:
            CodexLoginCancelledError: Re-raised so the worker enters the
                ERROR state and `on_worker_state_changed` can translate it
                into a "cancelled" toast and modal dismissal.
        """
        try:
            status = await codex_integration.run_browser_login(
                _ScreenInteraction(self),
                cancel_event=self._cancel_event,
            )
        except codex_integration.CodexLoginCancelledError:
            logger.info("ChatGPT OAuth sign-in cancelled by user")
            raise
        clear_caches()
        return status

    def on_authorize_url(self, url: str, opened_in_browser: bool) -> None:
        """Render the authorize URL in the modal.

        Called on the event loop from the async sign-in worker (the worker is
        started with `thread=False`), so it can mutate widgets directly.
        """
        status = self.query_one("#codex-auth-status", Static)
        url_label = self.query_one("#codex-auth-url", Static)
        if opened_in_browser:
            status.update("Waiting for you to finish signing in...")
        else:
            status.update("Could not launch a browser — open this URL manually:")
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in {"ansi-dark", "ansi-light"}
        link_style: str | TStyle = (
            TStyle(bold=True, link=url)
            if ansi
            else TStyle(
                foreground=TColor.parse(colors.primary),
                link=url,
            )
        )
        # `Content.assemble` with a (text, style) tuple skips markup parsing,
        # so a URL containing `[` (rare but possible in state params) cannot
        # crash the renderer.
        url_label.update(Content.assemble((url, link_style)))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """React to worker completion: notify, then dismiss the modal."""
        if event.worker is not self._worker:
            return
        state = event.state
        if state is WorkerState.SUCCESS:
            result = event.worker.result
            detail = "Signed in to ChatGPT."
            if (
                isinstance(result, codex_integration.CodexAuthStatus)
                and result.plan_type
            ):
                detail = f"Signed in to ChatGPT ({result.plan_type})."
            self.app.notify(detail, markup=False)
            self.dismiss(True)
        elif state is WorkerState.CANCELLED:
            self.app.notify("Sign-in cancelled.", markup=False)
            self.dismiss(False)
        elif state is WorkerState.ERROR:
            error = event.worker.error
            if isinstance(error, WorkerCancelled):
                self.app.notify("Sign-in cancelled.", markup=False)
                self.dismiss(False)
                return
            # `WorkerFailed` wraps the real exception (surfaced via
            # `error.error`); unwrap it so we can both detect a cancellation
            # that arrived via ERROR rather than CANCELLED and render an
            # accurate message. Test `inner`, not the wrapper.
            inner = (
                getattr(error, "error", error)
                if isinstance(error, WorkerFailed)
                else error
            )
            if isinstance(inner, codex_integration.CodexLoginCancelledError):
                self.app.notify("Sign-in cancelled.", markup=False)
                self.dismiss(False)
                return
            detail = str(inner) if inner else "Sign-in failed."
            logger.warning("ChatGPT OAuth sign-in failed: %s", detail)
            self.app.notify(
                f"Sign-in failed: {detail}",
                severity="error",
                markup=False,
            )
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Cancel the sign-in flow and dismiss the modal."""
        self._cancel_event.set()
        if self._worker is not None:
            self._worker.cancel()
        # `cancel()` triggers `WorkerState.CANCELLED` on the worker, which
        # `on_worker_state_changed` translates into the dismissal. Don't
        # dismiss eagerly here — that would race the success / error path
        # if the callback already landed.
        # However, if the worker hasn't been created yet (mount race), make
        # sure the modal still goes away.
        if self._worker is None:
            self.dismiss(False)


class CodexSignedInAction(StrEnum):
    """Outcome of the `CodexSignedInScreen` quick-action overlay.

    Encoded as an enum (mirroring `AuthResult`) rather than bare strings so a
    typo in either the producing `action_*` method or the consuming dispatch
    is a type error, not a silent no-op.
    """

    SIGN_OUT = "signout"
    """Delete the stored ChatGPT token."""

    REAUTH = "reauth"
    """Open the OAuth flow again (e.g., to switch account)."""


class CodexSignedInScreen(ModalScreen["CodexSignedInAction | None"]):
    """Quick-action overlay shown when `openai_codex` is already signed in.

    Dismissal values:

    - `CodexSignedInAction.SIGN_OUT`: delete the stored token.
    - `CodexSignedInAction.REAUTH`: open the OAuth flow again.
    - `None`: close without changes.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("s", "signout", "Sign out", show=False, priority=True),
        Binding("r", "reauth", "Reauth", show=False, priority=True),
    ]

    CSS = """
    CodexSignedInScreen {
        align: center middle;
    }

    CodexSignedInScreen > Vertical {
        width: 64;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    CodexSignedInScreen .codex-signed-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    CodexSignedInScreen .codex-signed-copy {
        height: auto;
        margin-bottom: 1;
    }

    CodexSignedInScreen .codex-signed-help {
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
        status = codex_integration.get_status()
        if status.plan_type and status.account_id:
            body = (
                f"Signed in to ChatGPT ({status.plan_type}) as account "
                f"{status.account_id}."
            )
        elif status.plan_type:
            body = f"Signed in to ChatGPT ({status.plan_type})."
        else:
            body = "Signed in to ChatGPT."
        with Vertical():
            yield Static("ChatGPT sign-in", classes="codex-signed-title")
            yield Static(
                Content.from_markup("$body", body=body),
                classes="codex-signed-copy",
            )
            yield Static(
                f"S sign out {glyphs.bullet} R sign in again {glyphs.bullet} Esc close",
                classes="codex-signed-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def action_signout(self) -> None:
        """Dismiss with `SIGN_OUT` so the manager deletes the stored token."""
        self.dismiss(CodexSignedInAction.SIGN_OUT)

    def action_reauth(self) -> None:
        """Dismiss with `REAUTH` so the manager kicks off a new flow."""
        self.dismiss(CodexSignedInAction.REAUTH)

    def action_cancel(self) -> None:
        """Close without changes."""
        self.dismiss(None)


def open_chatgpt_login_url() -> bool:
    """Open the ChatGPT account page in the user's browser.

    Used by `/auth` to let signed-in users jump to chatgpt.com (e.g., to
    change account, manage billing).

    Returns:
        Whether a browser actually launched — callers can fall back to a
            manual-URL toast on `False`.
    """
    try:
        return webbrowser.open("https://chatgpt.com/")
    except (webbrowser.Error, OSError) as exc:
        # `OSError` (not just `webbrowser.Error`) escapes when a configured
        # launcher's binary is missing; treat it as "no browser launched" so
        # the caller can fall back to a manual-URL toast.
        logger.warning("Could not open chatgpt.com: %s", exc)
        return False


__all__ = [
    "CodexAuthScreen",
    "CodexSignedInAction",
    "CodexSignedInScreen",
    "open_chatgpt_login_url",
]
