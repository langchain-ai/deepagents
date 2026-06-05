"""Tests for AgentSelectorScreen."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import OptionList

from deepagents_code.widgets.agent_selector import AgentSelectorScreen

if TYPE_CHECKING:
    from textual.pilot import Pilot

_AGENT_NAMES = ["agent", "coder", "researcher"]


class AgentSelectorTestApp(App):
    """Test app for AgentSelectorScreen."""

    def __init__(
        self,
        current_agent: str | None = "agent",
        agent_names: list[str] | None = None,
        default_agent: str | None = None,
    ) -> None:
        super().__init__()
        self._current = current_agent
        self._names = agent_names if agent_names is not None else list(_AGENT_NAMES)
        self._default = default_agent
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
            default_agent=self._default,
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


class TestAgentSelectorDefaultLabel:
    """The persisted default agent should be marked `(default)` in the picker."""

    async def test_default_agent_label_includes_default(self) -> None:
        """The default agent option should show '(default)' in its label."""
        async with AgentSelectorTestApp(
            current_agent="agent", default_agent="researcher"
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            # researcher is index 2; its label should contain "(default)"
            option = option_list.get_option_at_index(2)
            assert "(default)" in str(option.prompt)

    async def test_current_and_default_combine(self) -> None:
        """When the same agent is current and default, both markers appear."""
        async with AgentSelectorTestApp(
            current_agent="coder", default_agent="coder"
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            option = option_list.get_option_at_index(1)
            prompt = str(option.prompt)
            assert "current" in prompt
            assert "default" in prompt

    async def test_help_text_advertises_set_default(self) -> None:
        """The help line should mention `Ctrl+S set default`."""
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            statics = app.screen.query(".agent-selector-help")
            rendered = " ".join(str(s.render()) for s in statics)
            assert "Ctrl+S" in rendered
            assert "set default" in rendered


class TestAgentSelectorSetDefault:
    """Ctrl+S should toggle the highlighted agent as the persisted default."""

    async def test_set_default_persists_via_save_function(self, monkeypatch) -> None:
        """Pressing Ctrl+S calls `save_default_agent` with the highlighted name."""
        save_calls: list[str] = []

        def fake_save(name: str) -> bool:
            save_calls.append(name)
            return True

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            fake_save,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent=None
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()

        # "coder" is the highlighted (current) agent, so Ctrl+S sets it default
        assert save_calls == ["coder"]

    async def test_set_default_updates_label_in_place(self, monkeypatch) -> None:
        """After Ctrl+S, the highlighted option's label gains `(default)`."""
        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            lambda _name: True,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent=None
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            assert option_list.highlighted == 1
            prompt = str(option_list.get_option_at_index(1).prompt)
            assert "default" in prompt

    async def test_set_default_clears_when_already_default(self, monkeypatch) -> None:
        """Pressing Ctrl+S on the existing default clears it."""
        clear_calls: list[None] = []

        def fake_clear() -> bool:
            clear_calls.append(None)
            return True

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.clear_default_agent",
            fake_clear,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent="coder"
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            prompt = str(option_list.get_option_at_index(1).prompt)
            assert "default" not in prompt

        assert len(clear_calls) == 1

    async def test_set_default_no_op_with_empty_list(self, monkeypatch) -> None:
        """Ctrl+S on an empty selector must not raise."""
        save_calls: list[str] = []
        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            lambda name: save_calls.append(name) or True,
        )
        async with AgentSelectorTestApp(agent_names=[]).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
        assert save_calls == []


class TestAgentSelectorSetDefaultErrorPaths:
    """Defensive guards around the Ctrl+S flow.

    The user's stated concern was silent persistence failures. These
    tests pin the behavior of the failure paths so a regression that
    accidentally turns them silent (or, worse, crashes the modal) is
    caught.
    """

    async def test_save_failure_shows_error_in_help(self, monkeypatch) -> None:
        """`save_default_agent` returning False updates the help line, no crash."""
        from textual.widgets import Static

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            lambda _name: False,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent=None
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            help_widget = app.screen.query_one("#agent-help", Static)
            rendered = str(help_widget.render())
            # Help line surfaces the failure — picker still alive.
            assert "Failed to save" in rendered
            # Modal is still mounted; user can dismiss with Esc cleanly.
            assert not app.dismissed

    async def test_save_unexpected_exception_treated_as_failure(
        self, monkeypatch
    ) -> None:
        """An unexpected exception in `save_default_agent` is caught.

        Without the `try/except Exception` guard in `action_set_default`,
        a `TypeError` from `tomli_w.dump` would bubble out and kill the
        modal. The fix treats it as `ok = False`.
        """
        from textual.widgets import Static

        def boom(_name: str) -> bool:
            msg = "unexpected"
            raise TypeError(msg)

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            boom,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent=None
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            help_widget = app.screen.query_one("#agent-help", Static)
            rendered = str(help_widget.render())
            assert "Failed to save" in rendered
            assert not app.dismissed

    async def test_clear_unexpected_exception_treated_as_failure(
        self, monkeypatch
    ) -> None:
        """Same defensive guard for the clear path."""
        from textual.widgets import Static

        def boom() -> bool:
            msg = "unexpected"
            raise TypeError(msg)

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.clear_default_agent",
            boom,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent="coder"
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            help_widget = app.screen.query_one("#agent-help", Static)
            rendered = str(help_widget.render())
            assert "Failed to clear" in rendered
            assert not app.dismissed

    async def test_save_failure_keeps_default_unchanged(self, monkeypatch) -> None:
        """A failed save must not update the in-memory `_default_agent`.

        Otherwise the picker's `(default)` marker would advertise a state
        that did not actually persist to disk.
        """
        from textual.widgets import OptionList

        monkeypatch.setattr(
            "deepagents_code.widgets.agent_selector.save_default_agent",
            lambda _name: False,
        )

        async with AgentSelectorTestApp(
            current_agent="coder", default_agent=None
        ).run_test() as pilot:
            app = _app(pilot)
            app.show_selector()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            option_list = app.screen.query_one("#agent-options", OptionList)
            for i in range(option_list.option_count):
                prompt = str(option_list.get_option_at_index(i).prompt)
                assert "(default)" not in prompt
                assert "default)" not in prompt
