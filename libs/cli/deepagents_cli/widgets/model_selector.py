"""Interactive model selector screen for /model command."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

from deepagents_cli.config import CharsetMode, _detect_charset_mode, get_glyphs
from deepagents_cli.model_config import (
    get_curated_models,
    has_provider_credentials,
)


class ModelOption(Static):
    """A clickable model option in the selector."""

    def __init__(
        self,
        label: str,
        model_spec: str,
        provider: str,
        index: int,
        classes: str = "",
    ) -> None:
        """Initialize a model option.

        Args:
            label: The display text for the option.
            model_spec: The model specification (provider:model format).
            provider: The provider name.
            index: The index of this option in the filtered list.
            classes: CSS classes for styling.
        """
        super().__init__(label, classes=classes)
        self.model_spec = model_spec
        self.provider = provider
        self.index = index

    class Clicked(Message):
        """Message sent when a model option is clicked."""

        def __init__(self, model_spec: str, provider: str, index: int) -> None:
            """Initialize the Clicked message.

            Args:
                model_spec: The model specification.
                provider: The provider name.
                index: The index of the clicked option.
            """
            super().__init__()
            self.model_spec = model_spec
            self.provider = provider
            self.index = index

    def on_click(self, event: Click) -> None:
        """Handle click on this option.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(self.Clicked(self.model_spec, self.provider, self.index))


class ModelSelectorScreen(ModalScreen[tuple[str, str] | None]):
    """Full-screen modal for model selection.

    Displays available models grouped by provider with keyboard navigation
    and search filtering. Current model is highlighted.

    Returns (model_spec, provider) tuple on selection, or None on cancel.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("k", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("j", "move_down", "Down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    ModelSelectorScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ModelSelectorScreen .model-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ModelSelectorScreen #model-filter {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    ModelSelectorScreen #model-filter:focus {
        border: solid $primary;
    }

    ModelSelectorScreen .model-list {
        height: 20;
        scrollbar-gutter: stable;
    }

    ModelSelectorScreen #model-options {
        height: auto;
    }

    ModelSelectorScreen .model-provider-header {
        color: $primary;
        margin-top: 1;
    }

    ModelSelectorScreen .model-option {
        height: 1;
        padding: 0 1;
    }

    ModelSelectorScreen .model-option:hover {
        background: $surface-lighten-1;
    }

    ModelSelectorScreen .model-option-selected {
        background: $primary;
        text-style: bold;
    }

    ModelSelectorScreen .model-option-selected:hover {
        background: $primary-lighten-1;
    }

    ModelSelectorScreen .model-option-current {
        text-style: italic;
    }

    ModelSelectorScreen .model-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        current_model: str | None = None,
        current_provider: str | None = None,
    ) -> None:
        """Initialize the ModelSelectorScreen.

        Args:
            current_model: The currently active model name (to highlight).
            current_provider: The provider of the current model.
        """
        super().__init__()
        self._current_model = current_model
        self._current_provider = current_provider

        # Build list of (model_spec, provider) tuples from curated models
        self._all_models: list[tuple[str, str]] = []
        for provider, models in get_curated_models().items():
            for model in models:
                model_spec = f"{provider}:{model}"
                self._all_models.append((model_spec, provider))

        self._filtered_models: list[tuple[str, str]] = list(self._all_models)
        self._selected_index = 0
        self._options_container: Container | None = None
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the model selector UI.
        """
        glyphs = get_glyphs()

        with Vertical():
            # Title with current model in provider:model format
            if self._current_model and self._current_provider:
                current_spec = f"{self._current_provider}:{self._current_model}"
                title = f"Select Model (current: {current_spec})"
            elif self._current_model:
                title = f"Select Model (current: {self._current_model})"
            else:
                title = "Select Model"
            yield Static(title, classes="model-selector-title")

            # Search input
            yield Input(
                placeholder="Type to filter or enter provider:model...",
                id="model-filter",
            )

            # Scrollable model list
            with VerticalScroll(classes="model-list"):
                self._options_container = Container(id="model-options")
                yield self._options_container

            # Help text
            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate {glyphs.bullet} "
                f"Enter select {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="model-selector-help")

    async def on_mount(self) -> None:
        """Set up the screen on mount."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", "green")

        await self._update_display()

        # Focus the filter input
        filter_input = self.query_one("#model-filter", Input)
        filter_input.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter models as user types.

        Args:
            event: The input changed event.
        """
        self._filter_text = event.value.lower()
        self._update_filtered_list()
        self.call_after_refresh(self._update_display)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key when filter input is focused.

        Args:
            event: The input submitted event.
        """
        event.stop()
        self.action_select()

    def on_model_option_clicked(self, event: ModelOption.Clicked) -> None:
        """Handle click on a model option.

        Args:
            event: The click event with model info.
        """
        self._selected_index = event.index
        self.call_after_refresh(self._update_display)
        self.dismiss((event.model_spec, event.provider))

    def _update_filtered_list(self) -> None:
        """Update the filtered models based on search text."""
        if not self._filter_text:
            self._filtered_models = list(self._all_models)
        else:
            self._filtered_models = [
                (model_spec, provider)
                for model_spec, provider in self._all_models
                if self._filter_text in model_spec.lower()
            ]

        # Reset selection if out of bounds
        if self._selected_index >= len(self._filtered_models):
            self._selected_index = max(0, len(self._filtered_models) - 1)

    async def _update_display(self) -> None:
        """Render the model list grouped by provider."""
        if not self._options_container:
            return

        await self._options_container.remove_children()

        if not self._filtered_models:
            no_matches = Static("[dim]No matching models[/dim]")
            await self._options_container.mount(no_matches)
            return

        # Group by provider
        by_provider: dict[str, list[str]] = {}
        for model_spec, provider in self._filtered_models:
            by_provider.setdefault(provider, []).append(model_spec)

        glyphs = get_glyphs()
        flat_index = 0
        selected_widget: ModelOption | None = None

        # Build current model spec for comparison
        current_spec = None
        if self._current_model and self._current_provider:
            current_spec = f"{self._current_provider}:{self._current_model}"

        for provider, model_specs in by_provider.items():
            # Provider header with credential indicator
            has_creds = has_provider_credentials(provider)
            cred_indicator = glyphs.checkmark if has_creds else glyphs.warning
            header = Static(
                f"[bold]{provider}[/bold] {cred_indicator}",
                classes="model-provider-header",
            )
            await self._options_container.mount(header)

            for model_spec in model_specs:
                is_current = model_spec == current_spec
                is_selected = flat_index == self._selected_index

                cursor = f"{glyphs.cursor} " if is_selected else "  "
                current_mark = " [dim](current)[/dim]" if is_current else ""

                classes = "model-option"
                if is_selected:
                    classes += " model-option-selected"
                if is_current:
                    classes += " model-option-current"

                label = f"{cursor}{model_spec}{current_mark}"
                widget = ModelOption(
                    label=label,
                    model_spec=model_spec,
                    provider=provider,
                    index=flat_index,
                    classes=classes,
                )
                await self._options_container.mount(widget)

                if is_selected:
                    selected_widget = widget

                flat_index += 1

        # Scroll the selected item into view
        if selected_widget:
            selected_widget.scroll_visible()

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._filtered_models:
            count = len(self._filtered_models)
            self._selected_index = (self._selected_index - 1) % count
            self.call_after_refresh(self._update_display)

    def action_move_down(self) -> None:
        """Move selection down."""
        if self._filtered_models:
            count = len(self._filtered_models)
            self._selected_index = (self._selected_index + 1) % count
            self.call_after_refresh(self._update_display)

    def action_select(self) -> None:
        """Select the current model."""
        # Check if user typed a custom model spec in the filter
        filter_input = self.query_one("#model-filter", Input)
        custom_input = filter_input.value.strip()

        if custom_input and ":" in custom_input:
            # User typed a custom provider:model - use it directly
            provider = custom_input.split(":", 1)[0]
            self.dismiss((custom_input, provider))
            return

        if not self._filtered_models:
            return

        model_spec, provider = self._filtered_models[self._selected_index]
        self.dismiss((model_spec, provider))

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self.dismiss(None)
