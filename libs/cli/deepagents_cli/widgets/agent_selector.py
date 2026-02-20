"""Interactive agent selector screen for /agents command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.events import Click

from deepagents_cli.config import CharsetMode, _detect_charset_mode, get_glyphs

logger = logging.getLogger(__name__)


def _discover_agents(agents_dir: Path) -> list[str]:
    """Return sorted agent names found in the given directory.

    An agent is a subdirectory that contains an `AGENTS.md` file.

    Args:
        agents_dir: Directory to scan for agent subdirectories.

    Returns:
        Sorted list of agent names.
    """
    if not agents_dir.exists() or not agents_dir.is_dir():
        return []
    return sorted(
        folder.name
        for folder in agents_dir.iterdir()
        if folder.is_dir() and (folder / "AGENTS.md").exists()
    )


class AgentOption(Static):
    """A clickable agent option in the selector."""

    def __init__(
        self,
        label: str,
        agent_name: str,
        index: int,
        *,
        classes: str = "",
    ) -> None:
        """Initialize an agent option.

        Args:
            label: The display text for the option.
            agent_name: The agent identifier.
            index: The index of this option in the list.
            classes: CSS classes for styling.
        """
        super().__init__(label, classes=classes)
        self.agent_name = agent_name
        self.index = index

    class Clicked(Message):
        """Message sent when an agent option is clicked."""

        def __init__(self, agent_name: str, index: int) -> None:
            """Initialize the Clicked message.

            Args:
                agent_name: The agent identifier.
                index: The index of the clicked option.
            """
            super().__init__()
            self.agent_name = agent_name
            self.index = index

    def on_click(self, event: Click) -> None:
        """Handle click on this option.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(self.Clicked(self.agent_name, self.index))


class AgentSelectorScreen(ModalScreen[str | None]):
    """Modal dialog for browsing and switching agents.

    Displays available agents with keyboard navigation. The current agent
    is pre-selected and visually marked.

    Returns an agent name string on selection, or `None` on cancel.
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
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    AgentSelectorScreen {
        align: center middle;
    }

    AgentSelectorScreen > Vertical {
        width: 60;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AgentSelectorScreen .agent-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AgentSelectorScreen .agent-list {
        height: auto;
        max-height: 20;
        min-height: 3;
        scrollbar-gutter: stable;
        background: $background;
    }

    AgentSelectorScreen .agent-option {
        height: 1;
        padding: 0 1;
    }

    AgentSelectorScreen .agent-option:hover {
        background: $surface-lighten-1;
    }

    AgentSelectorScreen .agent-option-selected {
        background: $primary;
        text-style: bold;
    }

    AgentSelectorScreen .agent-option-selected:hover {
        background: $primary-lighten-1;
    }

    AgentSelectorScreen .agent-option-current {
        text-style: italic;
    }

    AgentSelectorScreen .agent-selector-help {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    AgentSelectorScreen .agent-empty {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        current_agent: str | None = None,
        agents_dir: Path | None = None,
    ) -> None:
        """Initialize the `AgentSelectorScreen`.

        Args:
            current_agent: The currently active agent name (to highlight).
            agents_dir: Directory to scan for agents. Defaults to
                `~/.deepagents`.
        """
        super().__init__()
        self._current_agent = current_agent
        self._agents_dir = agents_dir
        self._agents: list[str] = []
        self._selected_index = 0
        self._option_widgets: list[AgentOption] = []

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the agent selector UI.
        """
        glyphs = get_glyphs()

        if self._current_agent:
            title = f"Select Agent (current: {self._current_agent})"
        else:
            title = "Select Agent"

        with Vertical():
            yield Static(title, classes="agent-selector-title")

            with VerticalScroll(classes="agent-list"):
                yield Static(
                    "[dim]Loading agents...[/dim]",
                    classes="agent-empty",
                    id="agent-loading",
                )

            help_text = (
                f"{glyphs.arrow_up}/{glyphs.arrow_down}/tab navigate "
                f"{glyphs.bullet} Enter select {glyphs.bullet} Esc cancel"
            )
            yield Static(help_text, classes="agent-selector-help")

    async def on_mount(self) -> None:
        """Discover agents and build the list."""
        if _detect_charset_mode() == CharsetMode.ASCII:
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", "green")

        if self._agents_dir is None:
            from deepagents_cli.config import settings

            self._agents_dir = settings.user_deepagents_dir

        self._agents = _discover_agents(self._agents_dir)

        for i, name in enumerate(self._agents):
            if name == self._current_agent:
                self._selected_index = i
                break

        await self._build_list()
        self.focus()

    async def _build_list(self) -> None:
        """Build the agent option widgets."""
        scroll = self.query_one(".agent-list", VerticalScroll)
        await scroll.remove_children()
        self._option_widgets = []

        if not self._agents:
            await scroll.mount(
                Static(
                    "[dim]No agents found in ~/.deepagents/[/dim]",
                    classes="agent-empty",
                )
            )
            return

        selected_widget: AgentOption | None = None

        for i, name in enumerate(self._agents):
            is_current = name == self._current_agent
            is_selected = i == self._selected_index

            classes = "agent-option"
            if is_selected:
                classes += " agent-option-selected"
            if is_current:
                classes += " agent-option-current"

            label = self._format_option_label(
                name, selected=is_selected, current=is_current
            )
            widget = AgentOption(
                label=label,
                agent_name=name,
                index=i,
                classes=classes,
            )
            self._option_widgets.append(widget)

            if is_selected:
                selected_widget = widget

        await scroll.mount(*self._option_widgets)

        if selected_widget:
            if self._selected_index == 0:
                scroll.scroll_home(animate=False)
            else:
                selected_widget.scroll_visible(animate=False)

    @staticmethod
    def _format_option_label(
        name: str,
        *,
        selected: bool,
        current: bool,
    ) -> str:
        """Build the display label for an agent option.

        Args:
            name: Agent name.
            selected: Whether this option is currently highlighted.
            current: Whether this is the active agent.

        Returns:
            Rich-markup label string.
        """
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "
        label = f"{cursor}{name}"
        if current:
            label += " [dim](current)[/dim]"
        return label

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, re-rendering only the old and new widgets.

        Args:
            delta: Positions to move (negative for up, positive for down).
        """
        if not self._agents or not self._option_widgets:
            return

        count = len(self._agents)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("agent-option-selected")
        old_widget.update(
            self._format_option_label(
                self._agents[old_index],
                selected=False,
                current=self._agents[old_index] == self._current_agent,
            )
        )

        new_widget = self._option_widgets[new_index]
        new_widget.add_class("agent-option-selected")
        new_widget.update(
            self._format_option_label(
                self._agents[new_index],
                selected=True,
                current=self._agents[new_index] == self._current_agent,
            )
        )

        if new_index == 0:
            scroll = self.query_one(".agent-list", VerticalScroll)
            scroll.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

    def action_move_up(self) -> None:
        """Move selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._move_selection(1)

    def _visible_page_size(self) -> int:
        """Return the number of agent options that fit in one visual page.

        Returns:
            Number of agent options per page, at least 1.
        """
        default_page_size = 10
        try:
            scroll = self.query_one(".agent-list", VerticalScroll)
            height = scroll.size.height
        except Exception:  # Fallback to default page size on any widget query error
            logger.debug(
                "Agent list widget not found in _visible_page_size; "
                "using default page size %d",
                default_page_size,
                exc_info=True,
            )
            return default_page_size
        if height <= 0:
            return default_page_size
        return max(1, height)

    def action_page_up(self) -> None:
        """Move selection up by one visible page."""
        if not self._agents:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """Move selection down by one visible page."""
        if not self._agents:
            return
        count = len(self._agents)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """Confirm the highlighted agent and dismiss the selector."""
        if self._agents:
            self.dismiss(self._agents[self._selected_index])

    def on_agent_option_clicked(self, event: AgentOption.Clicked) -> None:
        """Handle click on an agent option.

        Args:
            event: The clicked message with agent name and index.
        """
        if 0 <= event.index < len(self._agents):
            self._selected_index = event.index
            self.dismiss(event.agent_name)

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self.dismiss(None)
