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
    ModelConfig,
    clear_default_model,
    get_available_models,
    has_provider_credentials,
    save_default_model,
)


class ModelOption(Static):
    """A clickable model option in the selector."""

    def __init__(
        self,
        label: str,
        model_spec: str,
        provider: str,
        index: int,
        *,
        has_creds: bool | None = True,
        classes: str = "",
    ) -> None:
        """Initialize a model option.

        Args:
            label: The display text for the option.
            model_spec: The model specification (provider:model format).
            provider: The provider name.
            index: The index of this option in the filtered list.
            has_creds: Whether the provider has valid credentials. True if
                confirmed, False if missing, None if unknown.
            classes: CSS classes for styling.
        """
        super().__init__(label, classes=classes)
        self.model_spec = model_spec
        self.provider = provider
        self.index = index
        self.has_creds = has_creds

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
        Binding("tab", "move_down", "Down", show=False, priority=True),
        Binding("shift+tab", "move_up", "Up", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("ctrl+s", "set_default", "Set default", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    ModelSelectorScreen > Vertical {
        width: 80;
        max-width: 90%;
        height: 80%;
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
        height: 1fr;
        min-height: 5;
        scrollbar-gutter: stable;
        background: $background;
    }

    ModelSelectorScreen #model-options {
        height: auto;
    }

    ModelSelectorScreen .model-provider-header {
        color: $primary;
        margin-top: 1;
    }

    ModelSelectorScreen #model-options > .model-provider-header:first-child {
        margin-top: 0;
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

        # Build list from dynamically discovered models (falls back to defaults)
        self._all_models: list[tuple[str, str]] = []
        for provider, models in get_available_models().items():
            for model in models:
                model_spec = f"{provider}:{model}"
                self._all_models.append((model_spec, provider))

        self._filtered_models: list[tuple[str, str]] = list(self._all_models)
        self._selected_index = self._find_current_model_index()
        self._options_container: Container | None = None
        self._option_widgets: list[ModelOption] = []
        self._filter_text = ""
        self._rebuild_needed = True
        self._current_spec: str | None = None
        if current_model and current_provider:
            self._current_spec = f"{current_provider}:{current_model}"

        config = ModelConfig.load()
        self._default_spec: str | None = config.default_model

    def _find_current_model_index(self) -> int:
        """Find the index of the current model in the filtered list.

        Returns:
            Index of the current model, or 0 if not found.
        """
        if not self._current_model or not self._current_provider:
            return 0

        current_spec = f"{self._current_provider}:{self._current_model}"
        for i, (model_spec, _) in enumerate(self._filtered_models):
            if model_spec == current_spec:
                return i
        return 0

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
                f"{glyphs.arrow_up}/{glyphs.arrow_down}/tab navigate {glyphs.bullet} "
                f"Enter select {glyphs.bullet} Ctrl+S set default "
                f"{glyphs.bullet} Esc cancel"
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
        self._rebuild_needed = True
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
        self.dismiss((event.model_spec, event.provider))

    def _update_filtered_list(self) -> None:
        """Update the filtered models based on search text."""
        if not self._filter_text:
            self._filtered_models = list(self._all_models)
            # Re-select current model when filter is cleared
            self._selected_index = self._find_current_model_index()
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
        """Render the model list grouped by provider.

        Performs a full DOM rebuild (removes all children, re-mounts).
        Arrow-key navigation uses `_move_selection` instead to avoid
        the cost of a full rebuild.
        """
        if not self._options_container:
            return

        await self._options_container.remove_children()
        self._option_widgets = []
        self._rebuild_needed = False

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

        # Collect all widgets first, then batch-mount once to avoid
        # individual DOM mutations per widget
        all_widgets: list[Static] = []

        for provider, model_specs in by_provider.items():
            # Provider header with credential indicator
            has_creds = has_provider_credentials(provider)
            if has_creds is True:
                cred_indicator = glyphs.checkmark
            elif has_creds is False:
                cred_indicator = f"{glyphs.warning} missing credentials"
            else:
                cred_indicator = f"{glyphs.question} credentials unknown"
            all_widgets.append(
                Static(
                    f"[bold]{provider}[/bold] [dim]{cred_indicator}[/dim]",
                    classes="model-provider-header",
                )
            )

            for model_spec in model_specs:
                is_current = model_spec == current_spec
                is_selected = flat_index == self._selected_index

                classes = "model-option"
                if is_selected:
                    classes += " model-option-selected"
                if is_current:
                    classes += " model-option-current"

                label = self._format_option_label(
                    model_spec,
                    selected=is_selected,
                    current=is_current,
                    has_creds=has_creds,
                    is_default=model_spec == self._default_spec,
                )
                widget = ModelOption(
                    label=label,
                    model_spec=model_spec,
                    provider=provider,
                    index=flat_index,
                    has_creds=has_creds,
                    classes=classes,
                )
                all_widgets.append(widget)
                self._option_widgets.append(widget)

                if is_selected:
                    selected_widget = widget

                flat_index += 1

        await self._options_container.mount(*all_widgets)

        # Scroll the selected item into view without animation so the list
        # appears already scrolled to the current model on first paint.
        if selected_widget:
            if self._selected_index == 0:
                # First item: scroll to top so header is visible
                scroll_container = self.query_one(".model-list", VerticalScroll)
                scroll_container.scroll_home(animate=False)
            else:
                selected_widget.scroll_visible(animate=False)

    @staticmethod
    def _format_option_label(
        model_spec: str,
        *,
        selected: bool,
        current: bool,
        has_creds: bool | None,
        is_default: bool = False,
    ) -> str:
        """Build the display label for a model option.

        Args:
            model_spec: The `provider:model` string.
            selected: Whether this option is currently highlighted.
            current: Whether this is the active model.
            has_creds: Credential status (True/False/None).
            is_default: Whether this is the configured default model.

        Returns:
            Rich-markup label string.
        """
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "
        if not has_creds:
            spec_text = f"[yellow]{model_spec}[/yellow]"
        elif is_default:
            spec_text = f"[cyan]{model_spec}[/cyan]"
        else:
            spec_text = model_spec
        suffix = " [dim](current)[/dim]" if current else ""
        default_suffix = " [cyan](default)[/cyan]" if is_default else ""
        return f"{cursor}{spec_text}{suffix}{default_suffix}"

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, updating only the affected widgets.

        Args:
            delta: Number of positions to move (-1 for up, +1 for down).
        """
        if not self._filtered_models or not self._option_widgets:
            return

        count = len(self._filtered_models)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        # Update the previously selected widget
        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("model-option-selected")
        old_widget.update(
            self._format_option_label(
                old_widget.model_spec,
                selected=False,
                current=old_widget.model_spec == self._current_spec,
                has_creds=old_widget.has_creds,
                is_default=old_widget.model_spec == self._default_spec,
            )
        )

        # Update the newly selected widget
        new_widget = self._option_widgets[new_index]
        new_widget.add_class("model-option-selected")
        new_widget.update(
            self._format_option_label(
                new_widget.model_spec,
                selected=True,
                current=new_widget.model_spec == self._current_spec,
                has_creds=new_widget.has_creds,
                is_default=new_widget.model_spec == self._default_spec,
            )
        )

        # Scroll the selected item into view
        if new_index == 0:
            scroll_container = self.query_one(".model-list", VerticalScroll)
            scroll_container.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

    def action_move_up(self) -> None:
        """Move selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._move_selection(1)

    def _visible_page_size(self) -> int:
        """Return the number of model options that fit in one visual page.

        Returns:
            Number of model options per page, at least 1.
        """
        default_page_size = 10
        try:
            scroll = self.query_one(".model-list", VerticalScroll)
            height = scroll.size.height
        except Exception:  # noqa: BLE001  # Fallback to default page size on any widget query error
            return default_page_size
        if height <= 0:
            return default_page_size

        total_models = len(self._filtered_models)
        if total_models == 0:
            return default_page_size

        # Each provider header = 1 row + margin-top: 1 (first has margin 0)
        num_headers = len(self.query(".model-provider-header"))
        header_rows = max(0, num_headers * 2 - 1) if num_headers else 0
        total_rows = total_models + header_rows
        return max(1, int(height * total_models / total_rows))

    def action_page_up(self) -> None:
        """Move selection up by one visible page."""
        if not self._filtered_models:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """Move selection down by one visible page."""
        if not self._filtered_models:
            return
        count = len(self._filtered_models)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """Select the current model."""
        # If there are filtered results, always select the highlighted model
        if self._filtered_models:
            model_spec, provider = self._filtered_models[self._selected_index]
            self.dismiss((model_spec, provider))
            return

        # No matches - check if user typed a custom provider:model spec
        filter_input = self.query_one("#model-filter", Input)
        custom_input = filter_input.value.strip()

        if custom_input and ":" in custom_input:
            provider = custom_input.split(":", 1)[0]
            self.dismiss((custom_input, provider))
        elif custom_input:
            self.dismiss((custom_input, ""))

    def action_set_default(self) -> None:
        """Toggle the highlighted model as the default.

        If the highlighted model is already the default, clears it.
        Otherwise sets it as the new default.
        """
        if not self._filtered_models or not self._option_widgets:
            return

        model_spec, _provider = self._filtered_models[self._selected_index]
        help_widget = self.query_one(".model-selector-help", Static)

        if model_spec == self._default_spec:
            # Already default — clear it
            if clear_default_model():
                self._default_spec = None
                self._rebuild_needed = True
                self.call_after_refresh(self._update_display)
                help_widget.update("[bold]Default cleared[/bold]")
                self.set_timer(3.0, self._restore_help_text)
            else:
                help_widget.update("[bold red]Failed to clear default[/bold red]")
                self.set_timer(3.0, self._restore_help_text)
        elif save_default_model(model_spec):
            self._default_spec = model_spec
            self._rebuild_needed = True
            self.call_after_refresh(self._update_display)
            help_widget.update(f"[bold]Default set to {model_spec}[/bold]")
            self.set_timer(3.0, self._restore_help_text)
        else:
            help_widget.update("[bold red]Failed to save default[/bold red]")
            self.set_timer(3.0, self._restore_help_text)

    def _restore_help_text(self) -> None:
        """Restore the default help text after a temporary message."""
        glyphs = get_glyphs()
        help_text = (
            f"{glyphs.arrow_up}/{glyphs.arrow_down}/tab navigate {glyphs.bullet} "
            f"Enter select {glyphs.bullet} Ctrl+S set default "
            f"{glyphs.bullet} Esc cancel"
        )
        help_widget = self.query_one(".model-selector-help", Static)
        help_widget.update(help_text)

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self.dismiss(None)
