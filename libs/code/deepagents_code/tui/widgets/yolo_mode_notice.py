"""First-enable confirmation modal before Shift+Tab enters YOLO.

Shown when the user cycles into unrestricted YOLO without a persisted
acknowledgement. Enter acknowledges YOLO and records the policy version; Esc
keeps the previous approval mode and leaves the acknowledgement unsaved.
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

YOLO_MODE_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/code/approval-modes"
)
"""Canonical docs page for Manual / Auto / YOLO behavior."""

YOLO_MODE_NOTICE_BODY = (
    "You are about to enable **YOLO**. The agent will run **gated actions "
    "without asking first** — shell commands, file edits, network calls, and "
    "other tools on this machine.\n\n"
    "This is **not Auto**. There is **no classifier review** and **no approval "
    "prompt** while YOLO is active. Prefer Auto unless you intend full "
    "unrestricted execution.\n\n"
    "This notice appears **once** on this machine after you continue. You can "
    "leave YOLO any time with **Shift+Tab**.\n\n"
    f"[Learn more about approval modes]({YOLO_MODE_DOCS_URL})"
)
"""Default Markdown body shown before the first YOLO switcher enable."""


class YoloModeNoticeScreen(ModalScreen[bool]):
    """In-TUI acknowledgement shown before unrestricted YOLO becomes active.

    Dismisses with `True` on Enter (acknowledge and enable) and `False` on Esc
    (keep the previous mode). Programmatic dismiss may yield `None`; callers
    treat that like cancel so YOLO is never left active without an explicit
    acknowledge.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Enable YOLO", show=False, priority=True),
        Binding("escape", "cancel", "Keep previous", show=False, priority=True),
    ]

    CSS = """
    YoloModeNoticeScreen {
        align: center middle;
    }

    YoloModeNoticeScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $error;
        padding: 1 2;
    }

    YoloModeNoticeScreen .yolo-mode-notice-title {
        text-style: bold;
        color: $error;
        text-align: center;
        margin-bottom: 1;
    }

    YoloModeNoticeScreen .yolo-mode-notice-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
        padding: 0;
    }

    YoloModeNoticeScreen .yolo-mode-notice-body > * {
        margin: 0 0 1 0;
    }

    YoloModeNoticeScreen .yolo-mode-notice-body > *:last-child {
        margin-bottom: 0;
    }

    YoloModeNoticeScreen .yolo-mode-notice-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    # The screen must be the focus target for its own priority Enter binding to
    # fire (see `on_mount`). Esc does not depend on focus: the app owns a global
    # priority `escape` binding that routes to `action_cancel` for active modals.
    can_focus = True

    def __init__(self, body: str | None = None) -> None:
        """Initialize the notice.

        Args:
            body: Optional Markdown body under the title. Defaults to
                `YOLO_MODE_NOTICE_BODY`. Links open in a browser.
        """
        super().__init__()
        self._body = YOLO_MODE_NOTICE_BODY if body is None else body

    def on_mount(self) -> None:
        """Take focus so priority bindings receive Enter/Esc."""
        self.focus()

    def compose(self) -> ComposeResult:
        """Compose the YOLO acknowledgement notice.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "YOLO mode",
                classes="yolo-mode-notice-title",
                markup=False,
            )
            # open_links=False so we own the click path (toast feedback + shared
            # URL safety). Assistant message widgets use the same pattern.
            yield Markdown(
                self._body,
                classes="yolo-mode-notice-body",
                open_links=False,
            )
            yield Static(
                "Enter to acknowledge and enable YOLO · Esc to keep current mode",
                classes="yolo-mode-notice-help",
                markup=False,
            )

    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open docs (or any body link) with the shared URL helper."""
        event.stop()
        await open_checked_url_async(event.href, app=self.app, notify_on_success=True)

    def action_confirm(self) -> None:
        """Acknowledge YOLO and mark the notice dismissed without re-showing."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Keep the previous approval mode without persisting acknowledgement.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc handling.
        """
        self.dismiss(False)
