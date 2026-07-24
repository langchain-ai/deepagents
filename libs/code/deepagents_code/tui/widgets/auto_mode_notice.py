"""First-enable confirmation modal for Auto mode.

Shown at most once per install (per notice version) after Auto successfully
becomes active. Enter keeps Auto and records the notice; Esc reverts to Manual
and leaves the notice unsaved so it can appear again next time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from deepagents_code.tui.widgets._links import open_checked_url_async

if TYPE_CHECKING:
    from textual.app import ComposeResult

AUTO_MODE_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/code/approval-modes"
)
"""Canonical docs page for Manual / Auto / YOLO behavior."""

AUTO_MODE_NOTICE_BODY = (
    "You switched to **Auto**. In this mode, the agent can approve "
    "**routine gated actions** without asking you first (for example ordinary "
    "source edits and read-only Git commands).\n\n"
    "Anything uncertain is reviewed by the **active model** against your "
    "**literal request** — not a separate security reviewer. After repeated "
    "denials or review failures, you get the normal approval prompt.\n\n"
    "This is **not a sandbox**. The agent still runs on this machine and can "
    "change files, run commands, and use tools when Auto allows them.\n\n"
    "This notice appears **once** on this machine after you continue.\n\n"
    f"[Learn more about approval modes]({AUTO_MODE_DOCS_URL})"
)
"""Default Markdown body shown on first successful Auto enable."""


class AutoModeNoticeScreen(ModalScreen[bool]):
    """In-TUI first-run notice describing what Auto mode does.

    Dismisses with `True` on Enter (keep Auto) and `False` on Esc (return to
    Manual). Programmatic dismiss may yield `None`; callers treat that like
    cancel so Auto is never left active without an explicit continue.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Keep Auto", show=False, priority=True),
        Binding("escape", "cancel", "Manual", show=False, priority=True),
    ]

    CSS = """
    AutoModeNoticeScreen {
        align: center middle;
    }

    AutoModeNoticeScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    AutoModeNoticeScreen .auto-mode-notice-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    AutoModeNoticeScreen .auto-mode-notice-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
        padding: 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-body > * {
        margin: 0 0 1 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-body > *:last-child {
        margin-bottom: 0;
    }

    AutoModeNoticeScreen .auto-mode-notice-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    # The screen must be the focus target for its own priority Enter/Esc
    # bindings to fire (see `on_mount`); without this the keys reach no handler.
    can_focus = True

    def __init__(self, body: str | None = None) -> None:
        """Initialize the notice.

        Args:
            body: Optional Markdown body under the title. Defaults to
                `AUTO_MODE_NOTICE_BODY`. Links open in a browser.
        """
        super().__init__()
        self._body = AUTO_MODE_NOTICE_BODY if body is None else body

    def on_mount(self) -> None:
        """Take focus so priority bindings receive Enter/Esc."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the Auto first-enable notice.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Auto mode",
                classes="auto-mode-notice-title",
                markup=False,
            )
            # open_links=False so we own the click path (toast feedback + shared
            # URL safety). Assistant message widgets use the same pattern.
            yield Markdown(
                self._body,
                classes="auto-mode-notice-body",
                open_links=False,
            )
            yield Static(
                "Enter to keep Auto · Esc for Manual",
                classes="auto-mode-notice-help",
                markup=False,
            )

    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open docs (or any body link) with the shared URL helper."""
        event.stop()
        await open_checked_url_async(event.href, app=self.app, notify_on_success=True)

    def action_confirm(self) -> None:
        """Keep Auto and mark the notice dismissed without re-showing."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Return to Manual without persisting the notice.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc handling.
        """
        self.dismiss(False)
