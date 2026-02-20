"""Tests for AgentSelectorScreen."""

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.screen import ModalScreen

from deepagents_cli.widgets.agent_selector import (
    AgentSelectorScreen,
    _discover_agents,
)


def _make_agents_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create a temporary agents directory with the given agent names.

    Args:
        tmp_path: Pytest temporary directory.
        names: Agent names to create as subdirectories with AGENTS.md.

    Returns:
        Path to the temporary agents directory.
    """
    agents_dir = tmp_path / ".deepagents"
    agents_dir.mkdir()
    for name in names:
        agent_dir = agents_dir / name
        agent_dir.mkdir()
        (agent_dir / "AGENTS.md").write_text(f"# {name}\n")
    return agents_dir


class AgentSelectorTestApp(App):
    """Test app for AgentSelectorScreen."""

    def __init__(
        self,
        current_agent: str | None = "coder",
        agents_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False
        self._current_agent = current_agent
        self._agents_dir = agents_dir

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the agent selector screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        screen = AgentSelectorScreen(
            current_agent=self._current_agent,
            agents_dir=self._agents_dir,
        )
        self.push_screen(screen, handle_result)


class AppWithEscapeBinding(App):
    """Test app that has a conflicting escape binding."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self, agents_dir: Path | None = None) -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False
        self.interrupt_called = False
        self._agents_dir = agents_dir

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def action_interrupt(self) -> None:
        """Handle escape - dismiss modal if present, otherwise mark as called."""
        if isinstance(self.screen, ModalScreen):
            self.screen.dismiss(None)
            return
        self.interrupt_called = True

    def show_selector(self) -> None:
        """Show the agent selector screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        screen = AgentSelectorScreen(
            current_agent="coder",
            agents_dir=self._agents_dir,
        )
        self.push_screen(screen, handle_result)


class TestDiscoverAgents:
    """Tests for the _discover_agents helper."""

    def test_returns_sorted_names(self, tmp_path: Path) -> None:
        """Agent names are returned in alphabetical order."""
        agents_dir = _make_agents_dir(tmp_path, ["researcher", "coder", "writer"])
        assert _discover_agents(agents_dir) == ["coder", "researcher", "writer"]

    def test_ignores_dirs_without_agents_md(self, tmp_path: Path) -> None:
        """Directories without AGENTS.md are excluded."""
        agents_dir = tmp_path / ".deepagents"
        agents_dir.mkdir()
        (agents_dir / "no-file").mkdir()
        (agents_dir / "valid").mkdir()
        (agents_dir / "valid" / "AGENTS.md").write_text("# valid\n")
        assert _discover_agents(agents_dir) == ["valid"]

    def test_ignores_files(self, tmp_path: Path) -> None:
        """Regular files in the agents dir are ignored."""
        agents_dir = tmp_path / ".deepagents"
        agents_dir.mkdir()
        (agents_dir / "config.toml").write_text("")
        (agents_dir / "valid").mkdir()
        (agents_dir / "valid" / "AGENTS.md").write_text("# valid\n")
        assert _discover_agents(agents_dir) == ["valid"]

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        """A nonexistent directory returns an empty list."""
        assert _discover_agents(tmp_path / "nowhere") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """An empty agents directory returns an empty list."""
        agents_dir = tmp_path / ".deepagents"
        agents_dir.mkdir()
        assert _discover_agents(agents_dir) == []


class TestAgentSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    @pytest.mark.asyncio
    async def test_escape_dismisses_modal(self, tmp_path: Path) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        agents_dir = _make_agents_dir(tmp_path, ["coder", "researcher"])
        app = AgentSelectorTestApp(current_agent="coder", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None

    @pytest.mark.asyncio
    async def test_escape_with_conflicting_app_binding(self, tmp_path: Path) -> None:
        """ESC should dismiss modal even when app has its own escape binding."""
        agents_dir = _make_agents_dir(tmp_path, ["coder", "researcher"])
        app = AppWithEscapeBinding(agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is None
            assert app.interrupt_called is False


class TestAgentSelectorKeyboardNavigation:
    """Tests for keyboard navigation in the modal."""

    @pytest.mark.asyncio
    async def test_down_arrow_moves_selection(self, tmp_path: Path) -> None:
        """Down arrow should move selection down."""
        agents_dir = _make_agents_dir(tmp_path, ["alpha", "beta", "gamma"])
        app = AgentSelectorTestApp(current_agent="alpha", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            initial_index = screen._selected_index

            await pilot.press("down")
            await pilot.pause()

            assert screen._selected_index == initial_index + 1

    @pytest.mark.asyncio
    async def test_up_arrow_wraps_from_first(self, tmp_path: Path) -> None:
        """Up arrow at the first item wraps to the last item."""
        agents_dir = _make_agents_dir(tmp_path, ["alpha", "beta", "gamma"])
        app = AgentSelectorTestApp(current_agent="alpha", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            count = len(screen._agents)

            await pilot.press("up")
            await pilot.pause()

            assert screen._selected_index == count - 1

    @pytest.mark.asyncio
    async def test_enter_selects_agent(self, tmp_path: Path) -> None:
        """Enter should select the highlighted agent and dismiss."""
        agents_dir = _make_agents_dir(tmp_path, ["coder", "researcher"])
        app = AgentSelectorTestApp(current_agent="coder", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result is not None

    @pytest.mark.asyncio
    async def test_enter_selects_navigated_agent(self, tmp_path: Path) -> None:
        """Enter selects the agent the user navigated to, not the original."""
        agents_dir = _make_agents_dir(tmp_path, ["alpha", "beta", "gamma"])
        app = AgentSelectorTestApp(current_agent="alpha", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            # Navigate to the second agent
            await pilot.press("down")
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert app.dismissed is True
            assert app.result == "beta"


class TestAgentSelectorCurrentAgentPreselection:
    """Tests for pre-selecting the current agent when opening the selector."""

    @pytest.mark.asyncio
    async def test_current_agent_is_preselected(self, tmp_path: Path) -> None:
        """Opening the selector should pre-select the current agent."""
        agents_dir = _make_agents_dir(tmp_path, ["alpha", "beta", "gamma"])
        app = AgentSelectorTestApp(current_agent="beta", agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            expected_index = screen._agents.index("beta")
            assert screen._selected_index == expected_index

    @pytest.mark.asyncio
    async def test_no_current_agent_selects_first(self, tmp_path: Path) -> None:
        """Without a current agent the first item should be selected."""
        agents_dir = _make_agents_dir(tmp_path, ["alpha", "beta"])
        app = AgentSelectorTestApp(current_agent=None, agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            assert screen._selected_index == 0


class TestAgentSelectorEmptyState:
    """Tests for the empty-state when no agents are available."""

    @pytest.mark.asyncio
    async def test_empty_agents_dir_shows_message(self, tmp_path: Path) -> None:
        """An empty agents directory should show a 'no agents found' message."""
        agents_dir = tmp_path / ".deepagents"
        agents_dir.mkdir()
        app = AgentSelectorTestApp(current_agent=None, agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            assert screen._agents == []

    @pytest.mark.asyncio
    async def test_missing_agents_dir_shows_message(self, tmp_path: Path) -> None:
        """A missing agents directory should show a 'no agents found' message."""
        agents_dir = tmp_path / "nonexistent"
        app = AgentSelectorTestApp(current_agent=None, agents_dir=agents_dir)
        async with app.run_test() as pilot:
            app.show_selector()
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AgentSelectorScreen)
            assert screen._agents == []
