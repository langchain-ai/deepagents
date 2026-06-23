"""Confirmation modals for `/update` dependency-refresh flows in the TUI.

When `deepagents-code` itself is already on the latest release, `/update` can
still re-resolve its dependencies to the newest versions allowed by the pinned
ranges (e.g. a new `langchain-openai`). The already-current path dry-runs the
resolution first, then asks for explicit confirmation only when dependencies can
move. `/update --deps` skips that prompt, but asks before taking an available
app update ahead of the dependency refresh for the current app version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class _DependencyConfirmScreen(ModalScreen[bool]):
    """Shared base for the `/update` dependency-refresh confirmation modals.

    Subclasses supply only their bindings and the title/body/help text; the base
    owns the layout, styling, and the `True`/`False` dismissal contract. The
    `DEFAULT_CSS` type selector matches subclasses because Textual resolves type
    selectors against every class name in the MRO.
    """

    DEFAULT_CSS = """
    _DependencyConfirmScreen {
        align: center middle;
    }

    _DependencyConfirmScreen > Vertical {
        width: 66;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    _DependencyConfirmScreen .dependency-confirm-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    _DependencyConfirmScreen .dependency-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    _DependencyConfirmScreen .dependency-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, *, title: str, body: str, help_text: str) -> None:
        """Store the dialog copy for `compose`.

        Args:
            title: Bold heading shown at the top of the dialog.
            body: Explanatory paragraph beneath the title.
            help_text: Muted key-hint row at the bottom.
        """
        super().__init__()
        self._title = title
        self._body = body
        self._help = help_text

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                self._title,
                classes="dependency-confirm-title",
                markup=False,
            )
            yield Static(
                self._body,
                classes="dependency-confirm-body",
                markup=False,
            )
            yield Static(
                self._help,
                classes="dependency-confirm-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to `action_cancel`
        if present and otherwise falls through to `dismiss(None)`. Renaming this
        would silently regress Esc to a `None` dismiss instead of an explicit
        cancel.
        """
        self.dismiss(False)


class UpdateBeforeDependenciesConfirmScreen(_DependencyConfirmScreen):
    """Confirmation overlay before `/update --deps` upgrades dcode itself.

    Dismisses with `True` when the user chooses the app update first and `False`
    when they prefer to refresh dependencies for the current app version.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Update", show=False, priority=True),
        Binding("escape", "cancel", "Refresh deps", show=False, priority=True),
    ]

    def __init__(self, *, current: str, latest: str) -> None:
        """Create the app-update confirmation dialog.

        Args:
            current: Currently running `deepagents-code` version.
            latest: Latest available `deepagents-code` version.
        """
        super().__init__(
            title="Update dcode first?",
            body=(
                f"A newer deepagents-code version is available ({current} -> "
                f"{latest}). Update dcode now, or refresh dependencies for the "
                "current version you already have."
            ),
            help_text="Enter to update dcode, Esc to refresh current dependencies",
        )


class RefreshDependenciesConfirmScreen(_DependencyConfirmScreen):
    """Confirmation overlay for a dependency refresh.

    Dismisses with `True` when the user confirms and `False` when the user
    cancels. Esc is treated as cancel so the user is never forced into a refresh
    they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Refresh", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(self, *, planned_changes: str | None = None) -> None:
        """Create the dependency-refresh confirmation dialog.

        Args:
            planned_changes: Optional dry-run summary of dependency updates.
        """
        body = (
            "deepagents-code is already up to date, but compatible dependency "
            "updates are available. Refresh to apply these changes:\n\n"
            f"{planned_changes}"
            if planned_changes
            else (
                "deepagents-code is already up to date, but its dependencies "
                "can be re-resolved to the newest compatible versions. This may "
                "pull in newer minor releases of packages like langchain-openai."
            )
        )
        super().__init__(
            title="Refresh dependencies?",
            body=body,
            help_text="Enter to refresh, Esc to cancel",
        )
