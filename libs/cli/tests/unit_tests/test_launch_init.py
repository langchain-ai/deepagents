"""Tests for onboarding screens."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_cli.extras_info import ExtraDependencyStatus
from deepagents_cli.widgets.launch_init import (
    LaunchDependenciesScreen,
    LaunchNameScreen,
)


class LaunchNameTestApp(App[None]):
    """Test app for `LaunchNameScreen`."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        """Compose a minimal host app."""
        yield Container(id="main")

    def show_name_screen(self) -> None:
        """Open the launch name screen."""

        def handle_result(result: str | None) -> None:
            self.result = result
            self.dismissed = True

        self.push_screen(LaunchNameScreen(), handle_result)

    def show_dependencies_screen(
        self,
        statuses: tuple[ExtraDependencyStatus, ...],
        *,
        continue_screen: ModalScreen[Any] | None = None,
    ) -> None:
        """Open the launch dependency summary screen."""

        def handle_result(result: bool | None) -> None:
            self.result = None if result is None else str(result)
            self.dismissed = True

        self.push_screen(
            LaunchDependenciesScreen(statuses, continue_screen=continue_screen),
            handle_result,
        )


class DummyNextScreen(ModalScreen[None]):
    """Simple modal used to test dependency-screen transitions."""

    def compose(self) -> ComposeResult:
        """Compose a minimal next screen."""
        yield Static("Next")


class TestLaunchNameScreen:
    """Tests for launch name entry."""

    def test_uses_modal_backdrop(self) -> None:
        """The name screen should keep Textual's dimmed modal backdrop."""
        assert "background: transparent" not in LaunchNameScreen.CSS

    async def test_name_placeholder_marks_field_optional(self) -> None:
        """The name field should make optional entry clear."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            name_input = app.screen.query_one("#launch-name-input", Input)

        assert name_input.placeholder == "Your name (optional)"

    async def test_copy_explains_name_memory_and_skip(self) -> None:
        """The name screen should describe memory and skip semantics."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            copy = app.screen.query_one(".launch-init-copy", Static)
            help_text = app.screen.query_one(".launch-init-help", Static)

        assert "remembered for future sessions" in str(copy.content)
        assert "Esc skip setup" in str(help_text.content)

    async def test_submit_returns_normalized_name(self) -> None:
        """Submitting a name should dismiss with the trimmed, title-cased value."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("space", "a", "d", "a", "space", "enter")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "Ada"

    async def test_submit_title_cases_multiple_lowercase_words(self) -> None:
        """Lowercase full names should be returned in title case."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press(
                "a", "d", "a", "space", "l", "o", "v", "e", "l", "a", "c", "e", "enter"
            )
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "Ada Lovelace"

    async def test_submit_empty_name_continues(self) -> None:
        """Submitting an empty optional name should continue setup."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == ""

    async def test_escape_skips(self) -> None:
        """Escape should skip the setup flow."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result is None


class TestLaunchDependenciesScreen:
    """Tests for launch dependency summary."""

    _STATUSES = (
        ExtraDependencyStatus(
            name="anthropic",
            installed=(("langchain-anthropic", "1.4.0"),),
            missing=(),
        ),
        ExtraDependencyStatus(
            name="bedrock",
            installed=(),
            missing=("langchain-aws",),
        ),
        ExtraDependencyStatus(
            name="daytona",
            installed=(("langchain-daytona", "0.0.5"),),
            missing=(),
        ),
        ExtraDependencyStatus(
            name="runloop",
            installed=(),
            missing=("langchain-runloop",),
        ),
    )

    async def test_renders_installed_and_available_extras(self) -> None:
        """Dependency screen should summarize ready and addable integrations."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(self._STATUSES)
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        assert "Installed Integrations" in content
        assert "Ready now" in content
        assert "Model providers: anthropic" in content
        assert "Sandboxes: daytona" in content
        assert "Available to add" in content
        assert "Model providers: bedrock" in content
        assert "Sandboxes: runloop" in content
        assert "Esc skip setup" in content

    async def test_enter_continues(self) -> None:
        """Enter should continue to the next onboarding step."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(self._STATUSES)
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "True"

    async def test_enter_switches_to_continue_screen(self) -> None:
        """Enter should replace the dependency modal when a next screen exists."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(
                self._STATUSES,
                continue_screen=DummyNextScreen(),
            )
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            assert isinstance(app.screen, DummyNextScreen)
            assert app.dismissed is False

    async def test_escape_skips(self) -> None:
        """Escape should skip the remaining setup flow."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(self._STATUSES)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result is None
