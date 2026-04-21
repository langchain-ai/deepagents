"""Dedicated modal for the update-available notification.

Shown automatically at startup when a newer version of
`deepagents-cli` is available on PyPI. Surfaces the same actions the
notification center would offer for the update entry but with a
focused, single-purpose presentation instead of the generic
notification list.
"""

from __future__ import annotations

import asyncio
import webbrowser
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

    from deepagents_cli.notifications import NotificationAction, PendingNotification

from deepagents_cli import theme
from deepagents_cli._version import CHANGELOG_URL
from deepagents_cli.config import get_glyphs, is_ascii_mode
from deepagents_cli.notifications import ActionId


class ChangelogClicked(Message):
    """Posted when the changelog row is clicked with the mouse."""


class _ActionOption(Static):
    """Clickable single-line action row."""

    def __init__(self, action: NotificationAction, widget_id: str) -> None:
        """Initialize the action row widget.

        Args:
            action: The action this row represents.
            widget_id: DOM id assigned to the widget.
        """
        super().__init__(id=widget_id, classes="ua-action")
        self._action = action
        self._is_selected = False
        self.update(self._render())

    @property
    def action(self) -> NotificationAction:
        """Return the underlying action."""
        return self._action

    def set_selected(self, selected: bool) -> None:
        """Toggle selection styling.

        Args:
            selected: Whether this row is currently under the cursor.
        """
        if self._is_selected == selected:
            return
        self._is_selected = selected
        self.set_class(selected, "-selected")
        self.update(self._render())

    def _render(self) -> Content:
        glyphs = get_glyphs()
        cursor = glyphs.cursor if self._is_selected else " "
        text = f"{cursor} {self._action.label}"
        if self._action.primary:
            return Content.styled(text, "bold")
        return Content(text)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler
        """Swallow the click without activating.

        Clicks on action rows are intentionally disabled to prevent
        accidentally triggering `Install now`, `Skip this version`, or
        similar destructive/irreversible actions. Activation is
        keyboard-only (enter). The changelog row, which only opens a
        URL, still accepts clicks.
        """
        event.stop()


class _ChangelogOption(Static):
    """Secondary row that opens the changelog URL in a browser.

    Visually grouped with the action rows so Tab/Shift+Tab navigation
    covers it uniformly, but activating it does not dismiss the modal
    — it is a "view more info" action, not one of the three
    mutually-exclusive update dispositions.
    """

    def __init__(self) -> None:
        """Initialize the changelog row."""
        super().__init__(id="ua-changelog", classes="ua-action ua-changelog")
        self._is_selected = False
        self.update(self._render())

    def set_selected(self, selected: bool) -> None:
        """Toggle selection styling.

        Args:
            selected: Whether this row is currently under the cursor.
        """
        if self._is_selected == selected:
            return
        self._is_selected = selected
        self.set_class(selected, "-selected")
        self.update(self._render())

    def _render(self) -> Content:
        glyphs = get_glyphs()
        cursor = glyphs.cursor if self._is_selected else " "
        return Content(f"{cursor} View changelog")

    def on_click(self, event: Click) -> None:
        """Dispatch a click as a `ChangelogClicked` message."""
        event.stop()
        self.post_message(ChangelogClicked())


class UpdateAvailableScreen(ModalScreen[ActionId | None]):
    """Modal dedicated to the update-available notification.

    Renders the entry's title and body, a "View changelog" row, and
    one row per configured action. Dismisses with the selected
    `ActionId`, or `None` on Esc. Activating the changelog row opens
    the URL in a browser and leaves the modal open.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("tab", "move_down", "Next", show=False, priority=True),
        Binding("shift+tab", "move_up", "Previous", show=False, priority=True),
        Binding("enter", "activate", "Select", show=False, priority=True),
    ]

    CSS = """
    UpdateAvailableScreen {
        align: center middle;
    }

    UpdateAvailableScreen > Vertical {
        width: 68;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $success;
        padding: 1 2;
    }

    UpdateAvailableScreen .ua-title {
        text-style: bold;
        color: $text-success;
        text-align: center;
        margin-bottom: 1;
    }

    UpdateAvailableScreen .ua-body {
        color: $text-muted;
        margin-bottom: 1;
    }

    UpdateAvailableScreen .ua-action {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    UpdateAvailableScreen .ua-action.-selected {
        background: $surface-lighten-1;
    }

    UpdateAvailableScreen .ua-changelog {
        color: $text-muted;
        margin-bottom: 1;
    }

    UpdateAvailableScreen .ua-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, entry: PendingNotification) -> None:
        """Initialize the screen.

        Args:
            entry: Registered notification to render. Its actions
                drive the action rows; dismissing via an action
                returns the chosen `ActionId` to the caller.
        """
        super().__init__()
        self._entry = entry
        self._options: list[_ActionOption | _ChangelogOption] = []
        self._selected = 0

    def compose(self) -> ComposeResult:
        """Compose the modal layout.

        Yields:
            Title, optional body, the changelog row, one
            `_ActionOption` per action, and a help footer.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static(self._entry.title, classes="ua-title", markup=False)
            if self._entry.body:
                yield Static(self._entry.body, classes="ua-body", markup=False)
            changelog = _ChangelogOption()
            self._options.append(changelog)
            yield changelog
            for idx, action in enumerate(self._entry.actions):
                option = _ActionOption(action, f"ua-row-{idx}")
                self._options.append(option)
                yield option
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab navigate "
                f"{glyphs.bullet} Enter select "
                f"{glyphs.bullet} Esc close"
            )
            yield Static(help_text, classes="ua-help")

    def on_mount(self) -> None:
        """Apply ASCII borders and highlight the primary action by default."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        # The changelog row sits at index 0; offset the primary action
        # by 1 so the initial cursor still lands on "Install now".
        primary_idx = next(
            (i + 1 for i, a in enumerate(self._entry.actions) if a.primary),
            1 if self._entry.actions else 0,
        )
        self._set_selected(primary_idx)

    def _set_selected(self, new_index: int) -> None:
        """Move the selection cursor to *new_index*.

        Raises:
            IndexError: When *new_index* is outside `0..len(self._options)`.
        """
        if not self._options:
            return
        if not 0 <= new_index < len(self._options):
            msg = f"selection {new_index} out of range 0..{len(self._options)}"
            raise IndexError(msg)
        if new_index != self._selected:
            self._options[self._selected].set_selected(selected=False)
        self._selected = new_index
        self._options[new_index].set_selected(selected=True)

    def action_move_up(self) -> None:
        """Move the cursor up one row (wraps at the top)."""
        if not self._options:
            return
        self._set_selected((self._selected - 1) % len(self._options))

    def action_move_down(self) -> None:
        """Move the cursor down one row (wraps at the bottom)."""
        if not self._options:
            return
        self._set_selected((self._selected + 1) % len(self._options))

    def action_activate(self) -> None:
        """Fire the highlighted option.

        Activating an action row dismisses the modal with its
        `ActionId`. Activating the changelog row opens the URL and
        leaves the modal open so the user can still pick an action.
        """
        if not self._options:
            self.dismiss(None)
            return
        option = self._options[self._selected]
        if isinstance(option, _ChangelogOption):
            self._open_changelog()
            return
        self.dismiss(option.action.action_id)

    def action_cancel(self) -> None:
        """Close without firing any action."""
        self.dismiss(None)

    def on_changelog_clicked(self, message: ChangelogClicked) -> None:
        """Handle a mouse click on the changelog row."""
        message.stop()
        self._open_changelog()

    def _open_changelog(self) -> None:
        """Open `CHANGELOG_URL` in a browser without closing the modal."""
        self.run_worker(
            asyncio.to_thread(webbrowser.open, CHANGELOG_URL),
            exclusive=False,
            group="update-available-changelog",
        )
