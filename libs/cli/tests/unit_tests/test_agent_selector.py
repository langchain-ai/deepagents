"""Tests for AgentSelectorScreen."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import OptionList

from deepagents_cli.widgets.agent_selector import AgentSelectorScreen

if TYPE_CHECKING:
    from textual.pilot import Pilot

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


def _app(pilot: Pilot[None]) -> AgentSelectorTestApp:
    """Narrow `pilot.app` to the concrete test-app type.

    `Pilot.app` is typed `App[Unknown]`, so ty can't see `show_selector`,
    `result`, or `dismissed`. A single `cast` per test keeps call sites
    typed without sprinkling `type: ignore`.
    """
    return cast("AgentSelectorTestApp", pilot.app)


class TestAgentSelectorEscapeKey:
    """Tests for ESC key dismissing the modal."""

    async def test_escape_dismisses_with_none(self) -> None:
        """Pressing ESC should dismiss the modal with None result."""
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.dismissed
            assert app.result is None

    async def test_escape_does_not_select_agent(self) -> None:
        """After ESC, no agent name should be returned."""
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is None


class TestAgentSelectorNavigation:
    """Tests for keyboard navigation."""

    async def test_enter_selects_highlighted_agent(self) -> None:
        """Pressing Enter should return the highlighted agent name."""
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.dismissed
            # The current agent ("agent") should be pre-selected at index 0
            assert app.result == "agent"

    async def test_down_arrow_moves_selection(self) -> None:
        """Pressing Down should move selection to the next agent."""
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert app.result == "coder"

    async def test_current_agent_is_preselected(self) -> None:
        """The current agent should be highlighted when the modal opens."""
        async with AgentSelectorTestApp(current_agent="coder").run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            # "coder" is at index 1 in sorted ["agent", "coder", "researcher"]
            assert option_list.highlighted == 1


class TestAgentSelectorEmptyList:
    """Tests for the empty-agents case."""

    async def test_no_agents_shows_placeholder(self) -> None:
        """When no agents exist, a placeholder message should be shown."""
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            # No OptionList should be present when there are no agents
            assert len(app.screen.query("#agent-options")) == 0

    async def test_escape_with_no_agents(self) -> None:
        """ESC should still dismiss correctly when no agents exist."""
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.dismissed
            assert app.result is None


class TestAgentSelectorCurrentLabel:
    """Tests for the (current) label on the active agent."""

    async def test_current_agent_label_includes_current(self) -> None:
        """The current agent option should show '(current)' in its label."""
        async with AgentSelectorTestApp(current_agent="researcher").run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            # researcher is index 2; its label should contain "(current)"
            option = option_list.get_option_at_index(2)
            assert "(current)" in str(option.prompt)


class TestAgentSelectorMarkupSafety:
    """Agent directory names containing Rich markup characters must render."""

    async def test_bracketed_agent_name_renders_without_error(self) -> None:
        """`my[agent]` is a legal directory name — OptionList must accept it."""
        names = ["safe", "my[agent]"]
        async with AgentSelectorTestApp(
            current_agent="safe", agent_names=names
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            # The bracket-bearing name must appear verbatim in the option
            # prompt — proof that markup parsing did not eat the brackets.
            names_seen = {
                str(option_list.get_option_at_index(i).prompt)
                for i in range(option_list.option_count)
            }
            assert any("my[agent]" in rendered for rendered in names_seen)

    async def test_bracketed_current_agent_selects_cleanly(self) -> None:
        """Selecting a bracketed current agent returns the raw directory name."""
        names = ["alpha", "my[agent]"]
        async with AgentSelectorTestApp(
            current_agent="my[agent]", agent_names=names
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.result == "my[agent]"


class TestAgentSelectorEmptyStateHelp:
    """The empty-state dialog should not advertise keys that do nothing."""

    async def test_empty_state_hides_enter_hint(self) -> None:
        """With zero agents, the help text should not promise 'Enter select'."""
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            statics = app.screen.query(".agent-selector-help")
            rendered = " ".join(str(s.render()) for s in statics)
            assert "Enter" not in rendered
            assert "Esc" in rendered
