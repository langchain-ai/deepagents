"""Tests for AgentSelectorScreen."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import OptionList

from deepagents_code.tui.widgets.agent_selector import AgentSelectorScreen

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


class TestAgentSelectorEmptyList:
    """Tests for the empty-agents case."""

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


class TestAgentSelectorEmptyStateHelp:
    """The empty-state dialog should not advertise keys that do nothing."""


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


class TestAgentSelectorSetDefault:
    """Ctrl+S should toggle the highlighted agent as the persisted default."""

    async def test_set_default_persists_via_save_function(self, monkeypatch) -> None:
        """Pressing Ctrl+S calls `save_default_agent` with the highlighted name."""
        save_calls: list[str] = []

        def fake_save(name: str) -> bool:
            save_calls.append(name)
            return True

        monkeypatch.setattr(
            "deepagents_code.tui.widgets.agent_selector.save_default_agent",
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


class TestAgentSelectorSetDefaultErrorPaths:
    """Defensive guards around the Ctrl+S flow.

    The user's stated concern was silent persistence failures. These
    tests pin the behavior of the failure paths so a regression that
    accidentally turns them silent (or, worse, crashes the modal) is
    caught.
    """

    async def test_save_failure_keeps_default_unchanged(self, monkeypatch) -> None:
        """A failed save must not update the in-memory `_default_agent`.

        Otherwise the picker's `(default)` marker would advertise a state
        that did not actually persist to disk.
        """
        from textual.widgets import OptionList

        monkeypatch.setattr(
            "deepagents_code.tui.widgets.agent_selector.save_default_agent",
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


class TestAgentSelectorBackdrop:
    """Tests for the modal's dimming backdrop."""

    async def test_dims_underlying_content(self) -> None:
        """The modal must inherit the translucent `ModalScreen` backdrop.

        Like the model and thread selectors, `/agents` should composite and
        dim the content underneath rather than render a fully transparent
        (non-dimming) overlay. The alpha is in (0, 1) only under a non-ansi
        theme, so pin `textual-dark`.
        """
        async with AgentSelectorTestApp().run_test() as pilot:
            app = _app(pilot)
            app.theme = "textual-dark"
            await pilot.pause()
            app.show_selector()
            await pilot.pause()
            assert 0 < app.screen.styles.background.a < 1
