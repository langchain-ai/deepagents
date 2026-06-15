"""Confirmation modal for `/install <package> --package` in the TUI.

Arbitrary packages have no curated allowlist to vet against, so installing
one pulls in third-party code. Rather than forcing the user to re-run with
`--force`, this non-blocking modal asks for explicit confirmation before the
install runs. `--force` (or `--yes`) still bypasses the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class InstallPackageConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay for installing an arbitrary `--package`.

    Dismisses with `True` when the user confirms and `False` when the user
    cancels. Esc is treated as cancel so the user is never forced into an
    install they did not explicitly choose.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Install", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    InstallPackageConfirmScreen {
        align: center middle;
    }

    InstallPackageConfirmScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    InstallPackageConfirmScreen .install-confirm-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }

    InstallPackageConfirmScreen .install-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    InstallPackageConfirmScreen .install-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, package: str) -> None:
        """Initialize the prompt.

        Args:
            package: The package name to install, surfaced in the body.
        """
        super().__init__()
        self._package = package

    def compose(self) -> ComposeResult:
        """Compose the install confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Install package?",
                classes="install-confirm-title",
                markup=False,
            )
            yield Static(
                Content.from_markup(
                    "Installing [bold]$name[/bold] runs third-party code in "
                    "the Deep Agents Code environment.",
                    name=self._package,
                ),
                classes="install-confirm-body",
                markup=False,
            )
            yield Static(
                "Enter to install, Esc to cancel",
                classes="install-confirm-help",
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


class InstallProviderConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay for installing a model provider's extra.

    Shown from the model selector when the user picks a model whose provider
    integration package is not installed. Dismisses with `True` to install and
    `False` to cancel; Esc cancels so the user is never forced into an install.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Install", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    InstallProviderConfirmScreen {
        align: center middle;
    }

    InstallProviderConfirmScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    InstallProviderConfirmScreen .install-confirm-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    InstallProviderConfirmScreen .install-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    InstallProviderConfirmScreen .install-confirm-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(self, provider: str, extra: str, model_spec: str) -> None:
        """Initialize the prompt.

        Args:
            provider: The provider whose integration is missing.
            extra: The `deepagents-code` extra that installs the provider.
            model_spec: The selected `provider:model` spec, surfaced in the body.
        """
        super().__init__()
        self._provider = provider
        self._extra = extra
        self._model_spec = model_spec

    def compose(self) -> ComposeResult:
        """Compose the provider-install confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(
                "Install provider?",
                classes="install-confirm-title",
                markup=False,
            )
            yield Static(
                Content.from_markup(
                    "[bold]$model[/bold] needs the [bold]$extra[/bold] "
                    "integration, which isn't installed yet. Install it now?",
                    model=self._model_spec,
                    extra=self._extra,
                ),
                classes="install-confirm-body",
                markup=False,
            )
            yield Static(
                "Enter to install, Esc to cancel",
                classes="install-confirm-help",
                markup=False,
            )

    def action_confirm(self) -> None:
        """Dismiss with `True`."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False`.

        The method name must stay `cancel` for the same reason as
        `InstallPackageConfirmScreen.action_cancel`: the app's priority
        `escape` binding dispatches to it for an active `ModalScreen`.
        """
        self.dismiss(False)
