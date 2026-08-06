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
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
    MouseMove,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Static

from deepagents_code.tui.widgets._links import event_targets_link, open_style_link

if TYPE_CHECKING:
    from textual.app import ComposeResult


def _package_link(package: str, *, hovered: bool = False) -> Content:
    """Render a package name as a bold PyPI link.

    Args:
        package: The package name to display and link.
        hovered: Whether to add a reverse-video highlight, matching the
            pointer cursor shown while the mouse is over the link.

    Returns:
        Styled content linking to the package's PyPI project page.
    """
    url = f"https://pypi.org/project/{package}/"
    style = TStyle(bold=True, underline=True, reverse=hovered, link=url)
    return Content.assemble((package, style))


class _LinkHoverMixin(ModalScreen[bool]):
    """Pointer-cursor, hover-highlight, and click-to-open for styled links.

    Subclasses define `_body_content(hovered: bool) -> Content` to rebuild
    their body text with the package link's highlight toggled, and the mixin's
    `_refresh_body(hovered: bool)` writes it into the `.install-confirm-body`
    widget. The one widget lookup is cached after first use. Inheriting
    `ModalScreen[bool]` here (rather than a bare mixin) lets the type checker
    see `styles`/`query_one`, and gives subclasses a single linear MRO.
    """

    _hovered: bool = False
    _body_widget: Static | None = None

    def _body_content(self, *, hovered: bool) -> Content:
        """Return the body `Content` with the link hover state applied."""
        raise NotImplementedError

    def _refresh_body(self, *, hovered: bool) -> None:
        """Rewrite the body widget with the link's hover highlight toggled.

        Args:
            hovered: Whether the link should render highlighted.
        """
        if self._body_widget is None:
            self._body_widget = self.query_one(".install-confirm-body", Static)
        self._body_widget.update(self._body_content(hovered=hovered))

    def on_click(self, event: Click) -> None:  # noqa: PLR6301  # Textual event handler convention
        """Open style-embedded hyperlinks on single click."""
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer cursor over the link and highlight it on hover."""
        over_link = event_targets_link(event)
        self.styles.pointer = "pointer" if over_link else "default"
        if over_link != self._hovered:
            self._hovered = over_link
            self._refresh_body(hovered=over_link)

    def on_leave(self) -> None:
        """Reset the cursor and clear any hover highlight."""
        self.styles.pointer = "default"
        if self._hovered:
            self._hovered = False
            self._refresh_body(hovered=False)


class InstallPackageConfirmScreen(_LinkHoverMixin):
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

    def _body_content(self, *, hovered: bool = False) -> Content:
        """Build the body text, toggling the link's hover highlight.

        Args:
            hovered: Whether the PyPI link should render highlighted.

        Returns:
            The body `Content` with the package link styled for `hovered`.
        """
        return Content.assemble(
            "Installing ",
            _package_link(self._package, hovered=hovered),
            " runs third-party code in the dcode environment.",
        )

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
                self._body_content(),
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


class InstallProviderConfirmScreen(_LinkHoverMixin):
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

    def __init__(
        self, provider: str, extra: str, model_spec: str | None = None
    ) -> None:
        """Initialize the prompt.

        Args:
            provider: The provider whose integration is missing.
            extra: The `deepagents-code` extra that installs the provider.
            model_spec: The selected `provider:model` spec, surfaced in the body
                by the model selector. Omitted by the `/auth` manager, which
                installs a provider so a key can be added rather than to switch
                to a specific model.
        """
        super().__init__()
        self._provider = provider
        self._extra = extra
        self._model_spec = model_spec

    def _body_content(self, *, hovered: bool = False) -> Content:
        """Build the body text, toggling the link's hover highlight.

        Args:
            hovered: Whether the PyPI link should render highlighted.

        Returns:
            The body `Content` with the package link styled for `hovered`.
        """
        # Reuse the auth UI's curated labels (e.g. `google_genai` -> "Google
        # Gemini") so the title reads naturally, falling back to a title-cased
        # provider key. Avoids the event-loop config read in
        # `provider_display_name`, which is overkill for a static title.
        from deepagents_code.config_manifest import provider_package_name
        from deepagents_code.tui.widgets.auth import PROVIDER_DISPLAY_NAMES

        provider = PROVIDER_DISPLAY_NAMES.get(
            self._provider, self._provider.replace("_", " ").title()
        )
        package = provider_package_name(self._provider) or self._extra
        package_link = _package_link(package, hovered=hovered)
        if self._model_spec is not None:
            return Content.assemble(
                "To use ",
                (self._model_spec, "bold"),
                ", dcode needs to install the ",
                package_link,
                " integration. This will add the provider package to your "
                "dcode environment.",
            )
        return Content.assemble(
            "To add a key for ",
            (provider, "bold"),
            ", dcode needs to install the ",
            package_link,
            " integration. This will add the provider package to your "
            "dcode environment.",
        )

    def compose(self) -> ComposeResult:
        """Compose the provider-install confirmation dialog.

        Yields:
            Title, body, and help-row widgets parented inside a `Vertical`.
        """
        from deepagents_code.tui.widgets.auth import PROVIDER_DISPLAY_NAMES

        provider = PROVIDER_DISPLAY_NAMES.get(
            self._provider, self._provider.replace("_", " ").title()
        )
        with Vertical():
            yield Static(
                f"Install {provider} support?",
                classes="install-confirm-title",
                markup=False,
            )
            yield Static(
                self._body_content(),
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
