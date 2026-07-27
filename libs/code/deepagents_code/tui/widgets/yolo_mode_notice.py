"""First-enable confirmation modal before Shift+Tab enters YOLO.

Shown when the user cycles into unrestricted YOLO without a persisted
acknowledgement. Enter acknowledges YOLO and records the policy version; `m`
switches to Manual instead; Esc keeps the previous approval mode. Only Enter
persists the acknowledgement.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from deepagents_code.tui.widgets._links import open_checked_url_async

if TYPE_CHECKING:
    from textual.app import ComposeResult


class YoloModeNoticeResult(StrEnum):
    """Outcome of the YOLO first-enable notice.

    `ACKNOWLEDGE` enables YOLO (and records the acknowledgement); `MANUAL`
    switches to Manual instead; `CANCEL` keeps the previous mode. A
    programmatic dismiss may still yield `None`, which callers treat like
    `CANCEL` so YOLO is never left active without an explicit acknowledge.
    """

    ACKNOWLEDGE = "acknowledge"
    MANUAL = "manual"
    CANCEL = "cancel"


YOLO_MODE_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/code/approval-modes"
)
"""Canonical docs page for Manual / Auto / YOLO behavior."""

YOLO_MODE_NOTICE_BODY = (
    "You are about to enable **YOLO mode**. The agent may run shell commands, "
    "edit files, make network calls, and use other tools on this machine "
    "**without asking you first**.\n\n"
    "Only continue if you're comfortable letting it act unsupervised. Leave "
    "YOLO any time with **Shift+Tab**.\n\n"
    "This notice appears **once** on this machine.\n\n"
    f"[Learn more about approval modes]({YOLO_MODE_DOCS_URL})"
)
"""Default Markdown body shown before the first YOLO switcher enable."""


class YoloModeNoticeScreen(ModalScreen[YoloModeNoticeResult]):
    """In-TUI acknowledgement shown before unrestricted YOLO becomes active.

    Dismisses with `YoloModeNoticeResult.ACKNOWLEDGE` on Enter (acknowledge and
    enable), `YoloModeNoticeResult.MANUAL` on `m` (switch to Manual), and
    `YoloModeNoticeResult.CANCEL` on Esc (keep the previous mode). Programmatic
    dismiss may yield `None`; callers treat that like cancel so YOLO is never
    left active without an explicit acknowledge.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Enable YOLO", show=False, priority=True),
        Binding("m", "switch_to_manual", "Switch to Manual", show=False, priority=True),
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
                "Enter to enable YOLO · m for Manual · Esc to keep current mode",
                classes="yolo-mode-notice-help",
                markup=False,
            )

    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open docs (or any body link) with the shared URL helper."""
        event.stop()
        await open_checked_url_async(event.href, app=self.app, notify_on_success=True)

    def action_confirm(self) -> None:
        """Acknowledge YOLO and mark the notice dismissed without re-showing."""
        self.dismiss(YoloModeNoticeResult.ACKNOWLEDGE)

    def action_switch_to_manual(self) -> None:
        """Switch to Manual instead of YOLO, without persisting acknowledgement."""
        self.dismiss(YoloModeNoticeResult.MANUAL)

    def action_cancel(self) -> None:
        """Keep the previous approval mode without persisting acknowledgement.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc handling.
        """
        self.dismiss(YoloModeNoticeResult.CANCEL)
