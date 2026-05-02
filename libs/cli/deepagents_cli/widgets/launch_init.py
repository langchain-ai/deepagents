"""Launch initialization screens for the interactive CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode


def _normalize_name(value: str) -> str:
    """Normalize submitted launch names for display.

    Args:
        value: Raw submitted name.

    Returns:
        The stripped name, title-cased when it was entered in lowercase.
    """
    name = value.strip()
    if name.islower():
        return name.title()
    return name


class LaunchNameScreen(ModalScreen[str | None]):
    """First-step launch initialization screen that asks for the user's name."""

    AUTO_FOCUS = "#launch-name-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    LaunchNameScreen {
        align: center middle;
        background: transparent;
    }

    LaunchNameScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    LaunchNameScreen .launch-init-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    LaunchNameScreen .launch-init-copy {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    LaunchNameScreen #launch-name-input {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    LaunchNameScreen #launch-name-input:focus {
        border: solid $primary;
    }

    LaunchNameScreen .launch-init-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual override
        """Compose the name-entry screen.

        Yields:
            Widgets for the modal content.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Welcome to Deep Agents", classes="launch-init-title")
            yield Static(
                Content.assemble(
                    "Start this thread by choosing how Deep Agents should address you."
                ),
                classes="launch-init-copy",
            )
            yield Input(
                placeholder="Your name",
                id="launch-name-input",
            )
            yield Static(
                f"Enter to continue {glyphs.bullet} Esc cancel",
                classes="launch-init-help",
            )

    def on_mount(self) -> None:
        """Focus the name field and apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        name_input = self.query_one("#launch-name-input", Input)
        name_input.focus()
        self.call_after_refresh(name_input.focus)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Dismiss with the submitted name.

        Args:
            event: The input submission event.
        """
        event.stop()
        value = _normalize_name(event.value)
        if not value:
            self.notify(
                "Enter a name, or press Esc to cancel setup.",
                severity="warning",
                timeout=3,
                markup=False,
            )
            return
        self.dismiss(value)

    def action_cancel(self) -> None:
        """Cancel the launch initialization sequence."""
        self.dismiss(None)
