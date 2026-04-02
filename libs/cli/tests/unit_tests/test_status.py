"""Unit tests for the StatusBar widget."""

from __future__ import annotations

from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Size

from deepagents_cli.widgets.status import StatusBar


class StatusBarApp(App):
    """Minimal app that mounts a StatusBar for testing."""

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")


class TestBranchDisplay:
    """Tests for the git branch display in the status bar."""

    async def test_branch_display_empty_by_default(self) -> None:
        """Branch display should be empty when no branch is set."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            display = pilot.app.query_one("#branch-display")
            assert bar.branch == ""
            assert display.render() == ""

    async def test_branch_display_shows_branch_name(self) -> None:
        """Setting branch reactive should update the display widget."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            assert "main" in rendered

    async def test_branch_display_with_feature_branch(self) -> None:
        """Feature branch names with slashes should display correctly."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "feat/new-feature"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            assert "feat/new-feature" in rendered

    async def test_branch_display_clears_when_set_empty(self) -> None:
        """Setting branch to empty string should clear the display."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            bar.branch = ""
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            assert display.render() == ""

    async def test_branch_display_contains_git_icon(self) -> None:
        """Branch display should include the git branch glyph prefix."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "develop"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            from deepagents_cli.config import get_glyphs

            assert rendered.startswith(get_glyphs().git_branch)


class TestResizePriority:
    """Branch hides before cwd, cwd hides before model."""

    async def test_branch_hidden_on_narrow_terminal(self) -> None:
        """Branch display should be hidden when terminal width < 100."""
        async with StatusBarApp().run_test(size=(80, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is False

    async def test_branch_visible_on_wide_terminal(self) -> None:
        """Branch display should be visible when terminal width >= 100."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is True

    async def test_cwd_hidden_on_very_narrow_terminal(self) -> None:
        """Cwd display should be hidden when terminal width < 70."""
        async with StatusBarApp().run_test(size=(60, 24)) as pilot:
            cwd = pilot.app.query_one("#cwd-display")
            assert cwd.display is False

    async def test_cwd_visible_branch_hidden_at_medium_width(self) -> None:
        """Between 70-99 cols: cwd visible, branch hidden."""
        async with StatusBarApp().run_test(size=(85, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            cwd = pilot.app.query_one("#cwd-display")
            branch = pilot.app.query_one("#branch-display")
            assert cwd.display is True
            assert branch.display is False

    async def test_resize_restores_branch_visibility(self) -> None:
        """Widening terminal should restore branch display."""
        async with StatusBarApp().run_test(size=(80, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is False
            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert branch.display is True

    async def test_model_visible_at_narrow_width(self) -> None:
        """Model display should remain visible even at very narrow widths."""
        async with StatusBarApp().run_test(size=(40, 24)) as pilot:
            from deepagents_cli.widgets.status import ModelLabel

            model = pilot.app.query_one("#model-display", ModelLabel)
            model.provider = "anthropic"
            model.model = "claude-sonnet-4-5"
            await pilot.pause()
            assert model.display is True


class TestTokenDisplay:
    """Tests for the token count display in the status bar."""

    async def test_set_tokens_updates_display(self) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert "5.0K" in str(display.render())

    async def test_hide_tokens_clears_display(self) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.hide_tokens()
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert str(display.render()) == ""

    async def test_set_tokens_after_hide_restores_display(self) -> None:
        """Regression: set_tokens must refresh even when value is unchanged.

        hide_tokens clears the widget text without updating the reactive,
        so a subsequent set_tokens with the same value must still re-render.
        """
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.hide_tokens()
            await pilot.pause()
            # Same value — previously skipped by reactive dedup
            bar.set_tokens(5000)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert "5.0K" in str(display.render())

    async def test_approximate_appends_plus(self) -> None:
        """approximate=True should append '+' to the token count."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000, approximate=True)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            rendered = str(display.render())
            assert "5.0K+" in rendered

    async def test_approximate_after_hide_restores_with_plus(self) -> None:
        """Interrupted restore: same value + approximate should show count with '+'."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.hide_tokens()
            await pilot.pause()
            bar.set_tokens(5000, approximate=True)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            rendered = str(display.render())
            assert "5.0K+" in rendered

    async def test_exact_count_clears_plus(self) -> None:
        """A non-approximate set_tokens after an approximate one should drop '+'."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000, approximate=True)
            await pilot.pause()
            bar.set_tokens(8000)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            rendered = str(display.render())
            assert "8.0K" in rendered
            assert "+" not in rendered
