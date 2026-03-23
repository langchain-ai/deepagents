"""Interactive theme selector screen for /theme command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

logger = logging.getLogger(__name__)


class ThemeSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for theme selection with live preview.

    Displays available themes in an `OptionList`. Navigating the option list
    applies a live preview by swapping the app theme. Returns the selected
    theme name on Enter, or `None` on Esc (restoring the original theme).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    ThemeSelectorScreen {
        align: center middle;
        background: transparent;
    }

    ThemeSelectorScreen > Vertical {
        width: 50;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ThemeSelectorScreen .theme-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ThemeSelectorScreen OptionList {
        height: auto;
        max-height: 16;
        background: $background;
    }

    ThemeSelectorScreen .theme-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(self, current_theme: str) -> None:
        """Initialize the ThemeSelectorScreen.

        Args:
            current_theme: The currently active theme name (to highlight).
        """
        super().__init__()
        self._current_theme = current_theme
        self._original_theme = current_theme

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the theme selector UI.
        """
        glyphs = get_glyphs()
        options: list[Option] = []
        highlight_index = 0

        for i, (name, entry) in enumerate(theme.ThemeEntry.REGISTRY.items()):
            label = entry.label
            if name == self._current_theme:
                label = f"{label} (current)"
                highlight_index = i
            options.append(Option(label, id=name))

        with Vertical():
            yield Static("Select Theme", classes="theme-selector-title")
            option_list = OptionList(*options, id="theme-options")
            option_list.highlighted = highlight_index
            yield option_list
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} preview"
                f" {glyphs.bullet} Enter select"
                f" {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="theme-selector-help")

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Live-preview the highlighted theme.

        Args:
            event: The option highlighted event.
        """
        name = event.option.id
        if name is not None and name in theme.ThemeEntry.REGISTRY:
            try:
                self.app.theme = name
                # refresh_css only repaints the active (modal) screen's layout;
                # force the screen beneath us to repaint so the user sees the
                # preview through the transparent scrim.
                stack = self.app.screen_stack
                if len(stack) > 1:
                    stack[-2].refresh(layout=True)
            except Exception:
                logger.warning("Failed to preview theme '%s'", name, exc_info=True)
                if self._original_theme is not None:
                    self.app.theme = self._original_theme

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Commit the selected theme.

        Args:
            event: The option selected event.
        """
        name = event.option.id
        if name is not None and name in theme.ThemeEntry.REGISTRY:
            self.dismiss(name)
        else:
            logger.warning("Selected theme '%s' is no longer available", name)
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Restore the original theme and dismiss."""
        self.app.theme = self._original_theme
        self.dismiss(None)
