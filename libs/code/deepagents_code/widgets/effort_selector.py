"""Interactive reasoning effort selector for `/effort`."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_code import theme
from deepagents_code.config import Glyphs, get_glyphs, is_ascii_mode


class EffortSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for selecting a reasoning effort level."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("tab", "cursor_down", "Next", show=False, priority=True),
        Binding("shift+tab", "cursor_up", "Previous", show=False, priority=True),
    ]

    CSS = """
    EffortSelectorScreen {
        align: center middle;
        background: transparent;
    }

    EffortSelectorScreen > Vertical {
        width: 54;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    EffortSelectorScreen .effort-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    EffortSelectorScreen .effort-selector-subtitle {
        height: auto;
        color: $text-muted;
        text-align: center;
        margin-bottom: 1;
    }

    EffortSelectorScreen OptionList {
        height: auto;
        max-height: 10;
        background: $background;
    }

    EffortSelectorScreen .effort-selector-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        model_spec: str,
        efforts: tuple[str, ...],
        current_effort: str | None = None,
    ) -> None:
        """Initialize the effort selector.

        Args:
            model_spec: Active `provider:model` spec.
            efforts: Supported effort labels for `model_spec`.
            current_effort: Current per-session effort override, if any.
        """
        super().__init__()
        self._model_spec = model_spec
        self._efforts = efforts
        self._current_effort = current_effort

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the effort selector UI.
        """
        glyphs = get_glyphs()
        with Vertical():
            yield Static("Select Reasoning Effort", classes="effort-selector-title")
            yield Static(self._model_spec, classes="effort-selector-subtitle")
            option_list = OptionList(*self._build_options(), id="effort-options")
            option_list.highlighted = self._current_index()
            yield option_list
            yield Static(self._help_text(glyphs), classes="effort-selector-help")

    def _build_options(self) -> list[Option]:
        """Build effort option entries.

        Returns:
            One `Option` per supported effort.
        """
        return [
            Option(self._format_label(effort), id=effort) for effort in self._efforts
        ]

    def _format_label(self, effort: str) -> Content:
        """Render an effort label with a current marker.

        Args:
            effort: Effort label.

        Returns:
            Styled option label.
        """
        if effort == self._current_effort:
            return Content.from_markup("$effort [dim](current)[/dim]", effort=effort)
        return Content.from_markup("$effort", effort=effort)

    def _current_index(self) -> int:
        """Return the highlighted effort index."""
        if self._current_effort is None:
            return 0
        try:
            return self._efforts.index(self._current_effort)
        except ValueError:
            return 0

    @staticmethod
    def _help_text(glyphs: Glyphs) -> str:
        """Build the selector help text.

        Args:
            glyphs: Glyph set for the active terminal mode.

        Returns:
            Help text for navigation and dismissal.
        """
        return (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab switch"
            f" {glyphs.bullet} Enter select"
            f" {glyphs.bullet} Esc cancel"
        )

    def on_mount(self) -> None:
        """Apply ASCII border if needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the selected effort.

        Args:
            event: The option selected event.
        """
        effort = event.option.id
        self.dismiss(effort)

    def action_cancel(self) -> None:
        """Cancel without changing effort."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the option list cursor down."""
        option_list = self._option_list()
        if option_list is not None:
            option_list.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option list cursor up."""
        option_list = self._option_list()
        if option_list is not None:
            option_list.action_cursor_up()

    def _option_list(self) -> OptionList | None:
        """Return the option list if it is mounted."""
        try:
            return self.query_one("#effort-options", OptionList)
        except NoMatches:
            return None
