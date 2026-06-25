"""Tests for onboarding screens."""

from __future__ import annotations

import re
from typing import Any

import pytest
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from deepagents_code.config import get_glyphs
from deepagents_code.extras_info import (
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    STANDALONE_EXTRAS,
    ExtraDependencyStatus,
)
from deepagents_code.widgets.launch_init import (
    LaunchDependenciesScreen,
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

    async def test_copy_prompts_for_name_and_hides_skip_hint(self) -> None:
        """The name screen prompts for a name without advertising the skip hint."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_name_screen()
            await pilot.pause()

            copy = app.screen.query_one(".launch-init-copy", Static)
            help_text = app.screen.query_one(".launch-init-help", Static)

        assert "What should Deep Agents call you?" in str(copy.content)
        assert "Enter to continue" in str(help_text.content)
        assert "Esc skip setup" not in str(help_text.content)

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

    async def test_submit_can_switch_directly_to_next_screen(self) -> None:
        """Submitting can replace the modal without exposing the base screen."""
        app = LaunchNameTestApp()
        continued: list[str] = []

        async with app.run_test() as pilot:
            app.push_screen(
                LaunchNameScreen(
                    continue_screen=DummyNextScreen(),
                    on_continue=continued.append,
                ),
                lambda result: setattr(app, "result", result),
            )
            await pilot.pause()

            await pilot.press("a", "d", "a", "enter")
            await pilot.pause()

            assert isinstance(app.screen, DummyNextScreen)

        assert continued == ["Ada"]
        assert app.result is None

    async def test_submit_switch_failure_dismisses_with_typed_name(self) -> None:
        """A `ScreenStackError` during switch should dismiss with the typed name.

        Unlike `LaunchDependenciesScreen`, the name screen emits no toast on this
        path: the name has already propagated via `on_continue` before the switch
        is attempted, so the fallback only needs to dismiss without dropping it.
        """
        app = LaunchNameTestApp()
        continued: list[str] = []

        async with app.run_test() as pilot:
            app.push_screen(
                LaunchNameScreen(
                    continue_screen=DummyNextScreen(),
                    on_continue=continued.append,
                ),
                lambda result: setattr(app, "result", result),
            )
            await pilot.pause()

            def fake_switch_screen(_screen: object) -> None:
                msg = "stack torn down"
                raise ScreenStackError(msg)

            app.switch_screen = fake_switch_screen  # ty: ignore

            await pilot.press("a", "d", "a", "enter")
            await pilot.pause()

        # `on_continue` fired before the failed switch, so the name propagated.
        assert continued == ["Ada"]
        # The fallback dismisses with the typed name rather than dropping it.
        assert app.result == "Ada"

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

    async def test_available_section_is_not_truncated(self) -> None:
        """Every addable extra is listed, with no "+N more" summary."""
        statuses = tuple(
            ExtraDependencyStatus(
                name=name, installed=(), missing=(f"langchain-{name}",)
            )
            for name in sorted(MODEL_PROVIDER_EXTRAS)
        )
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        # No "+N more" truncation summary (the old `_EXTRA_LIST_LIMIT` cap).
        assert re.search(r"\+\d+ more", content) is None
        for name in MODEL_PROVIDER_EXTRAS:
            assert name in content

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

    async def test_renders_other_category(self) -> None:
        """Compatibility standalone extras get their own category."""
        statuses = (
            ExtraDependencyStatus(
                name="quickjs", installed=(), missing=("langchain-quickjs",)
            ),
        )
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        assert "Other" in content
        assert "quickjs" in content

    async def test_empty_ready_section_shows_placeholder(self) -> None:
        """When nothing is installed, "Ready now" shows its placeholder."""
        statuses = tuple(
            ExtraDependencyStatus(
                name=name, installed=(), missing=(f"langchain-{name}",)
            )
            for name in sorted(MODEL_PROVIDER_EXTRAS)
        )
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        assert "Ready now (0)" in content
        assert "Nothing installed yet" in content
        # No extra is ready, so the checkmark glyph never appears.
        assert get_glyphs().checkmark not in content

    async def test_empty_available_section_shows_placeholder(self) -> None:
        """When everything is installed, "Available to add" shows its placeholder."""
        statuses = tuple(
            ExtraDependencyStatus(
                name=name,
                installed=((f"langchain-{name}", "1.0.0"),),
                missing=(),
            )
            for name in sorted(MODEL_PROVIDER_EXTRAS)
        )
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(statuses)
            await pilot.pause()

            content = "\n".join(
                str(widget.content) for widget in app.screen.query(Static)
            )

        assert "Available to add (0)" in content
        assert "All bundled integrations are installed." in content
        # Nothing is addable, so the empty-circle glyph never appears.
        assert get_glyphs().circle_empty not in content

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

    async def test_continue_screen_switch_failure_dismisses_with_toast(self) -> None:
        """A `ScreenStackError` during switch should dismiss and notify the user."""
        app = LaunchNameTestApp()
        async with app.run_test() as pilot:
            app.show_dependencies_screen(
                self._STATUSES,
                continue_screen=DummyNextScreen(),
            )
            await pilot.pause()

            notified: list[tuple[str, dict[str, Any]]] = []

            def fake_notify(message: str, **kwargs: Any) -> None:
                notified.append((message, kwargs))

            def fake_switch_screen(_screen: object) -> None:
                msg = "stack torn down"
                raise ScreenStackError(msg)

            app.switch_screen = fake_switch_screen  # ty: ignore
            app.notify = fake_notify  # ty: ignore

            await pilot.press("enter")
            await pilot.pause()

        assert app.dismissed is True
        assert app.result == "True"
        assert notified, "expected a toast on screen-switch failure"
        message, kwargs = notified[0]
        assert "model selector" in message.lower()
        assert kwargs.get("severity") == "warning"
        assert kwargs.get("markup") is False

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

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ada", "Ada"),
            ("ada lovelace", "Ada Lovelace"),
            ("  ada  ", "Ada"),
            ("Ada", "Ada"),
            ("ADA", "ADA"),
            ("aDa", "aDa"),
            ("Ada Lovelace", "Ada Lovelace"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        """Title-case lowercase input; preserve user-typed casing otherwise."""
        assert _normalize_name(raw) == expected


class TestLaunchDependenciesScreenDefaultStatuses:
    """Constructor branch that fetches status when none is supplied."""

    async def test_default_constructor_invokes_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When `statuses=None`, the screen calls `get_optional_dependency_status`."""
        calls = 0

        def fake_fetch() -> tuple[ExtraDependencyStatus, ...]:
            nonlocal calls
            calls += 1
            return ()

        from deepagents_code import extras_info

        monkeypatch.setattr(extras_info, "get_optional_dependency_status", fake_fetch)

        screen = LaunchDependenciesScreen()
        assert screen._statuses == ()
        assert calls == 1
