"""Unit tests for the StatusBar widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Size

from deepagents_cli._env_vars import HIDE_CWD, HIDE_GIT_BRANCH
from deepagents_cli.widgets.status import ModelLabel, StatusBar

if TYPE_CHECKING:
    import pytest


class StatusBarApp(App):
    """Minimal app that mounts a StatusBar for testing."""

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")


class TestCwdDisplay:
    """Tests for the cwd display in the status bar."""

    async def test_hide_cwd_env_var_hides_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cwd display should stay hidden when the env var override is enabled."""
        monkeypatch.setenv(HIDE_CWD, "1")
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            cwd = pilot.app.query_one("#cwd-display")
            assert cwd.display is False
            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert cwd.display is False

    async def test_hide_cwd_env_var_keeps_branch_visible_at_medium_width(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hiding cwd should not hide the branch when there is enough space."""
        monkeypatch.setenv(HIDE_CWD, "1")
        async with StatusBarApp().run_test(size=(85, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            cwd = pilot.app.query_one("#cwd-display")
            branch = pilot.app.query_one("#branch-display")
            assert cwd.display is False
            assert branch.display is True


class TestBranchDisplay:
    """Tests for the git branch display in the status bar."""

    async def test_hide_git_branch_env_var_hides_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Branch display should stay hidden when the env var override is enabled."""
        monkeypatch.setenv(HIDE_GIT_BRANCH, "1")
        async with StatusBarApp().run_test(size=(120, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is False
            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert branch.display is False

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

    async def test_show_pending_tokens_shows_unknown_placeholder(self) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.show_pending_tokens()
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert str(display.render()) == "... tokens"

    async def test_show_pending_tokens_before_count_leaves_display_empty(self) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.show_pending_tokens()
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert str(display.render()) == ""

    async def test_set_tokens_after_pending_restores_display(self) -> None:
        """Regression: set_tokens must refresh even when value is unchanged.

        `show_pending_tokens` replaces the widget text without updating the
        reactive value, so a subsequent `set_tokens` with the same count must
        still re-render.
        """
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.show_pending_tokens()
            await pilot.pause()
            # Same value — previously skipped by reactive dedup
            bar.set_tokens(5000)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert "5.0K" in str(display.render())

    async def test_show_pending_tokens_after_count_change_keeps_placeholder(
        self,
    ) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.show_pending_tokens()
            await pilot.pause()
            bar.set_tokens(7500)
            await pilot.pause()
            bar.show_pending_tokens()
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert str(display.render()) == "... tokens"

    def test_show_pending_tokens_without_mount_is_noop(self) -> None:
        bar = StatusBar()
        bar.show_pending_tokens()

    async def test_approximate_appends_plus(self) -> None:
        """approximate=True should append '+' to the token count."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000, approximate=True)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            rendered = str(display.render())
            assert "5.0K+" in rendered

    async def test_approximate_after_pending_restores_with_plus(self) -> None:
        """Interrupted restore: same value + approximate should show count with '+'."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            bar.show_pending_tokens()
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


class TestModeIndicator:
    """Tests for the input-mode indicator in the status bar."""

    async def test_incognito_shell_mode_shows_indicator(self) -> None:
        """Incognito shell mode renders the SHELL indicator with its own class."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            indicator = pilot.app.query_one("#mode-indicator")

            bar.set_mode("shell_incognito")
            await pilot.pause()

            assert str(indicator.render()) == "SHELL"
            assert indicator.has_class("shell-incognito")

    async def test_mode_transition_clears_incognito_class(self) -> None:
        """Leaving `shell_incognito` must remove the badge class.

        Regression guard: a future change forgetting to clear
        `shell-incognito` on transition would leak the badge across modes.
        """
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            indicator = pilot.app.query_one("#mode-indicator")

            bar.set_mode("shell_incognito")
            await pilot.pause()
            assert indicator.has_class("shell-incognito")

            bar.set_mode("normal")
            await pilot.pause()
            assert not indicator.has_class("shell-incognito")

            bar.set_mode("shell_incognito")
            await pilot.pause()
            bar.set_mode("shell")
            await pilot.pause()
            assert not indicator.has_class("shell-incognito")
            assert indicator.has_class("shell")


class TestModelLabelPrefixStripping:
    """Tests for provider-specific model prefix stripping in ModelLabel."""

    async def test_fireworks_prefix_stripped(self) -> None:
        """End-to-end: the fireworks prefix is stripped before rendering."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            await pilot.pause()
            rendered = str(label.render())
            assert "fireworks:kimi-k2p6" in rendered
            assert "accounts/fireworks/models/" not in rendered

    async def test_get_content_width_uses_stripped_name(self) -> None:
        """`get_content_width` sizes to the stripped name, not the raw model."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            await pilot.pause()
            assert label.get_content_width(Size(0, 0), Size(0, 0)) == len(
                "fireworks:kimi-k2p6"
            )

    async def test_provider_dropped_when_full_overflows(self) -> None:
        """When the cleaned full string overflows, render drops the provider."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            # padding 0 2 -> content width = 9, fits "kimi-k2p6" but not the
            # full "fireworks:kimi-k2p6" (19 chars).
            label.styles.width = 13
            await pilot.pause()
            assert str(label.render()) == "kimi-k2p6"

    async def test_truncation_uses_stripped_name(self) -> None:
        """Ellipsis truncation slices the stripped name; the raw prefix never leaks."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            # padding 0 2 -> content width = 5, smaller than "kimi-k2p6" (9).
            label.styles.width = 9
            await pilot.pause()
            rendered = str(label.render())
            assert rendered == "…k2p6"
            assert "accounts" not in rendered

    async def test_unmatched_prefix_for_registered_provider(self) -> None:
        """Registered provider whose model doesn't match any prefix is unchanged."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "kimi-k2p6"
            await pilot.pause()
            rendered = str(label.render())
            assert "fireworks:kimi-k2p6" in rendered

    async def test_multiple_registered_prefixes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider may register multiple prefixes; each matches independently."""
        from deepagents_cli.widgets import status

        monkeypatch.setitem(
            status.PROVIDER_PREFIX_STRIPS,
            "fireworks",
            ("accounts/fireworks/models/", "models/"),
        )
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "models/foo-bar"
            await pilot.pause()
            assert label._clean_model() == "foo-bar"

    async def test_non_fireworks_prefix_preserved(self) -> None:
        """Other providers should not have prefixes stripped."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "openai"
            label.model = "gpt-4o"
            await pilot.pause()
            rendered = str(label.render())
            assert "openai:gpt-4o" in rendered

    async def test_no_provider_no_stripping(self) -> None:
        """Without a provider, the model name is passed through unchanged."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = ""
            label.model = "accounts/fireworks/models/kimi-k2p6"
            await pilot.pause()
            rendered = str(label.render())
            assert "accounts/fireworks/models/kimi-k2p6" in rendered
