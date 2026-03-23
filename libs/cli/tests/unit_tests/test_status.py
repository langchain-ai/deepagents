"""Unit tests for the StatusBar widget."""

from __future__ import annotations

from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Size

from deepagents_cli.widgets.status import BranchLabel, CwdLabel, StatusBar


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
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            assert "main" in rendered

    async def test_branch_display_with_feature_branch(self) -> None:
        """Feature branch names with slashes should display correctly."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "feat/new-feature"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            assert "feat/new-feature" in rendered

    async def test_branch_display_clears_when_set_empty(self) -> None:
        """Setting branch to empty string should clear the display."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            bar.branch = ""
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            assert display.render() == ""

    async def test_branch_display_contains_git_icon(self) -> None:
        """Branch display should include the git branch glyph prefix."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "develop"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            from deepagents_cli.config import get_glyphs

            assert rendered.startswith(get_glyphs().git_branch)


class TestBranchTruncation:
    """Tests for BranchLabel ellipsis truncation."""

    async def test_long_branch_truncated_with_ellipsis(self) -> None:
        """A branch name that exceeds available width should end with ellipsis."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            label = pilot.app.query_one("#branch-display", BranchLabel)
            long_name = (
                "feature/very-long-branch-name-that-will"
                "-definitely-exceed-the-available-width"
            )
            bar.branch = long_name
            await pilot.pause()
            rendered = str(label.render())
            width = label.content_size.width
            from deepagents_cli.config import get_glyphs

            icon = get_glyphs().git_branch
            full = f"{icon} {long_name}"
            if len(full) > width > 0:
                assert rendered.startswith("\u2026")
                assert len(rendered) == width

    async def test_short_branch_not_truncated(self) -> None:
        """A short branch name should render without ellipsis."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            label = pilot.app.query_one("#branch-display", BranchLabel)
            bar.branch = "main"
            await pilot.pause()
            rendered = str(label.render())
            assert "main" in rendered
            assert "\u2026" not in rendered


class TestCwdTruncation:
    """Tests for CwdLabel ellipsis truncation."""

    async def test_long_cwd_truncated_with_leading_ellipsis(self) -> None:
        """A cwd path that exceeds available width should start with ellipsis."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            label = pilot.app.query_one("#cwd-display", CwdLabel)
            long_path = (
                "~/projects/very/deeply/nested/directory"
                "/structure/that/exceeds/available/width"
            )
            bar.cwd = long_path
            await pilot.pause()
            rendered = str(label.render())
            width = label.content_size.width
            if len(long_path) > width > 0:
                assert rendered.startswith("\u2026")
                assert len(rendered) == width
                # Tail of the path should be preserved
                assert rendered.endswith("width")

    async def test_short_cwd_not_truncated(self) -> None:
        """A short cwd path should render without ellipsis."""
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            label = pilot.app.query_one("#cwd-display", CwdLabel)
            bar.cwd = "/tmp"
            await pilot.pause()
            rendered = str(label.render())
            assert "\u2026" not in rendered


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
