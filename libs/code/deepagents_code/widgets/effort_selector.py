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
from deepagents_code.config import get_glyphs, is_ascii_mode


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
        default_effort: str | None = None,
    ) -> None:
        """Initialize the effort selector.

        Args:
            model_spec: Active `provider:model` spec.
            efforts: Supported effort labels for `model_spec`.
            current_effort: Current per-session effort override, if any.
            default_effort: Provider default effort for `model_spec`, if known.
        """
        super().__init__()
        self._model_spec = model_spec
        self._efforts = efforts
        self._current_effort = current_effort
        self._default_effort = default_effort

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the effort selector UI.
        """
        glyphs = get_glyphs()
        options = [
            Option(self._format_label(effort), id=effort) for effort in self._efforts
        ]
        highlighted_effort = self._current_effort or self._default_effort
        try:
            highlighted = self._efforts.index(highlighted_effort)
        except ValueError:
            highlighted = 0
        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab switch"
            f" {glyphs.bullet} Enter select"
            f" {glyphs.bullet} Esc cancel"
        )
        with Vertical():
            yield Static("Select Reasoning Effort", classes="effort-selector-title")
            yield Static(self._model_spec, classes="effort-selector-subtitle")
            option_list = OptionList(*options, id="effort-options")
            option_list.highlighted = highlighted
            yield option_list
            yield Static(help_text, classes="effort-selector-help")

    def _format_label(self, effort: str) -> Content:
        """Render an effort label with a current marker.

        Args:
            effort: Effort label.

        Returns:
            Styled option label.
        """
        markers = []
        if effort == self._current_effort:
            markers.append("current")
        if effort == self._default_effort:
            markers.append("default")
        if markers:
            suffix = ", ".join(markers)
            return Content.from_markup(
                "$effort [dim]($suffix)[/dim]", effort=effort, suffix=suffix
            )
        return Content.from_markup("$effort", effort=effort)

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
