"""Tests for AgentSelectorScreen."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import OptionList

from deepagents_cli.widgets.agent_selector import AgentSelectorScreen

_AGENT_NAMES = ["agent", "coder", "researcher"]


class AgentSelectorTestApp(App):
    """Test app for AgentSelectorScreen."""

    def __init__(
        self,
        current_agent: str | None = "agent",
        agent_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._current = current_agent
        self._names = agent_names if agent_names is not None else list(_AGENT_NAMES)
        self.result: str | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Container(id="main")

    def show_selector(self) -> None:
        """Show the agent selector screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        screen = AgentSelectorScreen(
            current_agent=self._current,
            agent_names=self._names,
        )
        self.push_screen(screen, handle_result)


class TestAgentSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    async def test_escape_dismisses_with_none(self) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        async with AgentSelectorTestApp().run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.dismissed
            assert pilot.app.result is None

    async def test_escape_does_not_select_agent(self) -> None:
        """After ESC, no agent name should be returned."""
        async with AgentSelectorTestApp().run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.result is None


class TestAgentSelectorNavigation:
    """Tests for keyboard navigation."""

    async def test_enter_selects_highlighted_agent(self) -> None:
        """Pressing Enter should return the highlighted agent name."""
        async with AgentSelectorTestApp().run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert pilot.app.dismissed
            # The current agent ("agent") should be pre-selected at index 0
            assert pilot.app.result == "agent"

    async def test_down_arrow_moves_selection(self) -> None:
        """Pressing Down should move selection to the next agent."""
        async with AgentSelectorTestApp().run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert pilot.app.result == "coder"

    async def test_current_agent_is_preselected(self) -> None:
        """The current agent should be highlighted when the modal opens."""
        async with AgentSelectorTestApp(current_agent="coder").run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            option_list = pilot.app.screen.query_one("#agent-options", OptionList)
            # "coder" is at index 1 in sorted ["agent", "coder", "researcher"]
            assert option_list.highlighted == 1


class TestAgentSelectorEmptyList:
    """Tests for the empty-agents case."""

    async def test_no_agents_shows_placeholder(self) -> None:
        """When no agents exist, a placeholder message should be shown."""
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            # No OptionList should be present when there are no agents
            assert len(pilot.app.screen.query("#agent-options")) == 0

    async def test_escape_with_no_agents(self) -> None:
        """ESC should still dismiss correctly when no agents exist."""
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert pilot.app.dismissed
            assert pilot.app.result is None


class TestAgentSelectorCurrentLabel:
    """Tests for the (current) label on the active agent."""

    async def test_current_agent_label_includes_current(self) -> None:
        """The current agent option should show '(current)' in its label."""
        async with AgentSelectorTestApp(current_agent="researcher").run_test() as pilot:
            pilot.app.show_selector()
            await pilot.pause()
            option_list = pilot.app.screen.query_one("#agent-options", OptionList)
            # researcher is index 2; its label should contain "(current)"
            option = option_list.get_option_at_index(2)
            assert "(current)" in str(option.prompt)
