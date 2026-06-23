"""Confirmation modals for `/update` dependency-refresh flows in the TUI.

When `deepagents-code` itself is already on the latest release, `/update` can
still re-resolve its dependencies to the newest versions allowed by the pinned
ranges (e.g. a new `langchain-openai`). That runs `uv tool upgrade` and can pull
in newer minor releases, so this non-blocking modal asks for explicit
confirmation first. `/update --deps` skips that prompt, but asks before taking an
available app update ahead of the dependency refresh for the current app version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class UpdateBeforeDependenciesConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay before `/update --deps` upgrades dcode itself.

    Dismisses with `True` when the user chooses the app update first and `False`
    when they prefer to refresh dependencies for the current app version.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Update", show=False, priority=True),
        Binding("escape", "cancel", "Refresh deps", show=False, priority=True),
    ]

    CSS = """
    UpdateBeforeDependenciesConfirmScreen {
        align: center middle;
    }

    UpdateBeforeDependenciesConfirmScreen > Vertical {
        width: 66;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    UpdateBeforeDependenciesConfirmScreen .update-confirm-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    UpdateBeforeDependenciesConfirmScreen .update-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    UpdateBeforeDependenciesConfirmScreen .update-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, *, current: str, latest: str) -> None:
        """Create the app-update confirmation dialog.

        Args:
            current: Currently running `deepagents-code` version.
            latest: Latest available `deepagents-code` version.
        """
        super().__init__()
        self._current = current
        self._latest = latest

    def compose(self) -> ComposeResult:
        """Compose the app-update confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Update dcode first?",
                classes="update-confirm-title",
                markup=False,
            )
            yield Static(
                f"A newer deepagents-code version is available ({self._current} -> "
                f"{self._latest}). Update dcode now, or refresh dependencies for "
                "the current version you already have.",
                classes="update-confirm-body",
                markup=False,
            )
            yield Static(
                "Enter to update dcode, Esc to refresh current dependencies",
                classes="update-confirm-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`."""
        self.dismiss(False)


class RefreshDependenciesConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay for a dependency refresh.

    Dismisses with `True` when the user confirms and `False` when the user
    cancels. Esc is treated as cancel so the user is never forced into a
    refresh they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Refresh", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    RefreshDependenciesConfirmScreen {
        align: center middle;
    }

    RefreshDependenciesConfirmScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    RefreshDependenciesConfirmScreen .refresh-confirm-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    RefreshDependenciesConfirmScreen .refresh-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    RefreshDependenciesConfirmScreen .refresh-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual override; the (self) -> ComposeResult signature is required
        """Compose the refresh confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Refresh dependencies?",
                classes="refresh-confirm-title",
                markup=False,
            )
            yield Static(
                "deepagents-code is already up to date, but its dependencies "
                "can be re-resolved to the newest compatible versions. This "
                "may pull in newer minor releases of packages like "
                "langchain-openai.",
                classes="refresh-confirm-body",
                markup=False,
            )
            yield Static(
                "Enter to refresh, Esc to cancel",
                classes="refresh-confirm-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc to a
        `None` dismiss instead of an explicit cancel.
        """
        self.dismiss(False)
