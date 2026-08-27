"""Tests for onboarding screens."""

from __future__ import annotations

import re
from typing import Any

import pytest
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static

from deepagents_code._paths import PATHS
from deepagents_code.config import get_glyphs
from deepagents_code.extras_info import (
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    STANDALONE_EXTRAS,
    ExtraDependencyStatus,
)
from deepagents_code.tui.widgets.launch_init import (
    LaunchDependenciesScreen,
    LaunchGoalCriteriaPreferenceScreen,
    LaunchNameScreen,
    _normalize_name,
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


class TestLaunchGoalCriteriaPreferenceScreen:
    """Tests for the first-run Auto goal criteria choice."""

    async def test_escape_chooses_review(self) -> None:
        """Escape should fail closed to review rather than skipping the question."""
        app = LaunchNameTestApp()
        results: list[bool | None] = []
        async with app.run_test() as pilot:
            app.push_screen(LaunchGoalCriteriaPreferenceScreen(), results.append)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

        assert results == [False]


class TestLaunchNameScreen:
    """Tests for launch name entry."""

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

        glyphs = get_glyphs()
        assert "Installed Integrations" in content
        # Section titles carry a total count; the `(2)` suffix is distinctive
        # enough to prove the section header rendered (vs. matching the intro
        # copy, which also mentions "model providers and sandboxes").
        assert "Ready now (2)" in content
        assert "Available to add (2)" in content
        # Ready extras carry the checkmark glyph; addable ones the empty circle.
        assert f"{glyphs.checkmark} anthropic" in content
        assert f"{glyphs.checkmark} daytona" in content
        assert f"{glyphs.circle_empty} bedrock" in content
        assert f"{glyphs.circle_empty} runloop" in content
        # The screen points at how to act on the listed integrations.
        assert "/install" in content
        assert "Enter to continue" in content
        assert "Esc skip setup" not in content

    async def test_populated_screen_fits_standard_terminal_height(self) -> None:
        """A full dependency list should keep footer controls visible at 80x24."""
        statuses = tuple(
            ExtraDependencyStatus(
                name=name, installed=(), missing=(f"langchain-{name}",)
            )
            for name in sorted(
                MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS | STANDALONE_EXTRAS
            )
        )
        app = LaunchNameTestApp()
        async with app.run_test(size=(80, 24)) as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()
            await pilot.pause()

            container = app.screen.query_one(Vertical)
            body = app.screen.query_one("#launch-dependencies-body", VerticalScroll)
            help_text = app.screen.query_one(".launch-init-help", Static)

        assert container.region.y >= 0
        assert container.region.y + container.region.height <= app.size.height
        assert help_text.region.y + help_text.region.height <= app.size.height
        max_height = body.styles.max_height
        assert max_height is not None
        assert max_height.cells is not None
        assert max_height.cells < 16

    async def test_resize_shrinks_body_to_keep_footer_visible(self) -> None:
        """Shrinking the terminal refits the body so the footer stays visible."""
        statuses = tuple(
            ExtraDependencyStatus(
                name=name, installed=(), missing=(f"langchain-{name}",)
            )
            for name in sorted(
                MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS | STANDALONE_EXTRAS
            )
        )
        app = LaunchNameTestApp()
        async with app.run_test(size=(80, 40)) as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()
            await pilot.pause()

            # A tall terminal leaves room for the full cap.
            tall = app.screen.query_one(
                "#launch-dependencies-body", VerticalScroll
            ).styles.max_height
            assert tall is not None
            assert tall.cells == 16

            await pilot.resize_terminal(80, 16)
            await pilot.pause()
            await pilot.pause()

            body = app.screen.query_one("#launch-dependencies-body", VerticalScroll)
            help_text = app.screen.query_one(".launch-init-help", Static)
            short = body.styles.max_height

        assert short is not None
        assert short.cells is not None
        assert short.cells < 16
        assert help_text.region.y + help_text.region.height <= app.size.height

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

    async def test_empty_statuses_render_explanatory_message(self) -> None:
        """Empty statuses should explain the cause instead of "none detected" twice."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(())
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        assert "Could not read installed dependency metadata" in content
        # The misleading double "none detected" must not appear.
        assert content.count("none detected") == 0
        # Section labels from the populated path must not leak through.
        assert "Ready now" not in content
        assert "Available to add" not in content


class TestNormalizeName:
    """Direct unit tests for `_normalize_name`."""


class TestLaunchDependenciesScreenDefaultStatuses:
    """Constructor branch that fetches status when none is supplied."""
