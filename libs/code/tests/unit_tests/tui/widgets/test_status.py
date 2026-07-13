"""Unit tests for the StatusBar widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Size
from textual.widgets import Static

from deepagents_code._env_vars import HIDE_CWD, HIDE_GIT_BRANCH
from deepagents_code.config import reset_glyphs_cache
from deepagents_code.tui.widgets.status import BranchLabel, ModelLabel, StatusBar

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_glyphs_between_tests() -> Iterator[None]:
    """Clear process-global glyph detection before and after each test."""
    reset_glyphs_cache()
    yield
    reset_glyphs_cache()


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
            assert branch.styles.padding.left == 1

    async def test_cwd_owns_left_gap_after_auto_approve_pill(self) -> None:
        """The cwd keeps visible spacing when transient status slots are hidden."""
        async with StatusBarApp().run_test() as pilot:
            cwd = pilot.app.query_one("#cwd-display")
            connection = pilot.app.query_one("#connection-indicator")
            status = pilot.app.query_one("#status-message")

            assert connection.display is False
            assert status.display is False
            assert cwd.styles.padding.left == 1
            assert cwd.styles.padding.right == 1


class TestPermissionModeDisplay:
    """Tests for user-facing permission-mode terminology."""

    async def test_unrestricted_mode_uses_clear_copy(self) -> None:
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            display = pilot.app.query_one("#auto-approve-indicator", Static)

            bar.set_auto_approve(enabled=True)
            await pilot.pause()
            assert str(display.render()) == "unrestricted"

            bar.set_auto_approve(enabled=False)
            await pilot.pause()
            assert str(display.render()) == "manual"


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

    async def test_branch_display_shows_branch_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting branch reactive should update the display widget.

        `HIDE_CWD` removes the cwd from the layout so the branch region isn't
        starved of width by a deep pytest cwd -- otherwise this assertion flakes
        on the actual run directory's path length rather than any real behavior.
        """
        monkeypatch.setenv(HIDE_CWD, "1")
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            assert "main" in rendered

    async def test_branch_display_with_feature_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Feature branch names with slashes should display correctly.

        `HIDE_CWD` keeps the branch region wide enough regardless of the pytest
        cwd path length (see `test_branch_display_shows_branch_name`).
        """
        monkeypatch.setenv(HIDE_CWD, "1")
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

    @staticmethod
    def _visible_branch_text(display: BranchLabel) -> str:
        """Return the branch text as actually rendered to the terminal line."""
        from rich.segment import Segment

        return "".join(
            seg.text for seg in display.render_line(0) if isinstance(seg, Segment)
        )

    async def test_long_branch_name_truncates_with_ellipsis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A branch too long for the footer should render a trailing ellipsis.

        The branch widget's width is pinned directly (rather than relying on
        whole-status-bar layout arithmetic) so truncation is deterministic and
        independent of the other status items' sizes.
        """
        monkeypatch.setenv("UI_CHARSET_MODE", "unicode")
        reset_glyphs_cache()
        long_branch = "feature/some-really-long-descriptive-branch-name-here"
        async with StatusBarApp().run_test(size=(110, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = long_branch
            display = pilot.app.query_one("#branch-display", BranchLabel)
            # Force a box far narrower than the branch text so overflow applies.
            display.styles.width = 20
            await pilot.pause()
            visible = self._visible_branch_text(display)
            # A *trailing* ellipsis (glyph-aware truncation), not a leading
            # one, with the head of the name preserved.
            assert visible.rstrip().endswith("\u2026")
            assert "feature/" in visible

    async def test_long_branch_truncates_with_ellipsis_when_cwd_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Truncation still applies in the cwd-hidden layout.

        With `HIDE_CWD` set, the branch is shown at a lower width threshold
        and fills the collapsible region alone; a too-long name must still
        ellipsize rather than hard-clip.
        """
        monkeypatch.setenv(HIDE_CWD, "1")
        monkeypatch.setenv("UI_CHARSET_MODE", "unicode")
        reset_glyphs_cache()
        long_branch = "feature/some-really-long-descriptive-branch-name-here"
        async with StatusBarApp().run_test(size=(90, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = long_branch
            display = pilot.app.query_one("#branch-display", BranchLabel)
            display.styles.width = 20
            await pilot.pause()
            assert display.display is True
            visible = self._visible_branch_text(display)
            assert visible.rstrip().endswith("\u2026")
            assert "feature/" in visible

    async def test_short_branch_name_not_truncated(self) -> None:
        """A branch that fits should render in full with no ellipsis."""
        async with StatusBarApp().run_test(size=(150, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display", BranchLabel)
            visible = self._visible_branch_text(display)
            assert "\u2026" not in visible
            assert "main" in visible

    async def test_long_branch_truncates_with_ascii_ellipsis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In ASCII charset mode, truncation uses `"..."` not `"…"`.

        CSS `text-overflow: ellipsis` always emits the Unicode ellipsis
        character; `BranchLabel` truncates manually via `get_glyphs` so
        the configured glyph (ASCII `"..."` in ascii mode) is used instead.
        """
        monkeypatch.setenv("UI_CHARSET_MODE", "ascii")
        reset_glyphs_cache()
        long_branch = "feature/some-really-long-descriptive-branch-name-here"
        try:
            async with StatusBarApp().run_test(size=(110, 24)) as pilot:
                bar = pilot.app.query_one("#status-bar", StatusBar)
                bar.branch = long_branch
                display = pilot.app.query_one("#branch-display", BranchLabel)
                display.styles.width = 20
                await pilot.pause()
                visible = self._visible_branch_text(display)
                # ASCII ellipsis is three dots, not the Unicode character.
                assert visible.rstrip().endswith("...")
                assert "\u2026" not in visible
                assert "feature/" in visible
        finally:
            monkeypatch.delenv("UI_CHARSET_MODE", raising=False)
            reset_glyphs_cache()

    async def test_branch_display_contains_git_icon(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Branch display should include the git branch glyph prefix.

        `HIDE_CWD` keeps the branch region wide enough regardless of the pytest
        cwd path length (see `test_branch_display_shows_branch_name`).
        """
        monkeypatch.setenv(HIDE_CWD, "1")
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "develop"
            await pilot.pause()
            display = pilot.app.query_one("#branch-display")
            rendered = str(display.render())
            from deepagents_code.config import get_glyphs

            assert rendered.startswith(get_glyphs().git_branch)


class TestResizePriority:
    """The cwd hides on narrow terminals; the branch truncates but never hides."""

    async def test_branch_stays_visible_on_narrow_terminal(self) -> None:
        """The branch truncates rather than hiding on a narrow terminal."""
        async with StatusBarApp().run_test(size=(80, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is True

    async def test_branch_stays_visible_below_cwd_threshold(self) -> None:
        """Even below the cwd hide threshold the branch stays visible."""
        async with StatusBarApp().run_test(size=(50, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is True
            assert branch.styles.padding.left == 1

    async def test_branch_removes_fallback_gap_when_cwd_returns(self) -> None:
        """Resizing wide again restores branch spacing to the cwd-visible layout."""
        async with StatusBarApp().run_test(size=(50, 24)) as pilot:
            branch = pilot.app.query_one("#branch-display")
            await pilot.pause()
            assert branch.styles.padding.left == 1

            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert branch.styles.padding.left == 0

    async def test_branch_visible_on_wide_terminal(self) -> None:
        """Branch display should be visible on a wide terminal."""
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

    async def test_cwd_and_branch_visible_at_medium_width(self) -> None:
        """Between 70-99 cols: cwd visible and branch visible (truncating)."""
        async with StatusBarApp().run_test(size=(85, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            cwd = pilot.app.query_one("#cwd-display")
            branch = pilot.app.query_one("#branch-display")
            assert cwd.display is True
            assert branch.display is True

    async def test_resize_never_hides_branch(self) -> None:
        """Resizing must never toggle the branch off; it only truncates."""
        async with StatusBarApp().run_test(size=(80, 24)) as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.branch = "main"
            await pilot.pause()
            branch = pilot.app.query_one("#branch-display")
            assert branch.display is True
            await pilot.resize_terminal(120, 24)
            await pilot.pause()
            assert branch.display is True
            await pilot.resize_terminal(50, 24)
            await pilot.pause()
            assert branch.display is True

    async def test_model_visible_at_narrow_width(self) -> None:
        """Model display should remain visible even at very narrow widths."""
        async with StatusBarApp().run_test(size=(40, 24)) as pilot:
            from deepagents_code.tui.widgets.status import ModelLabel

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

    async def test_empty_tokens_hidden_on_mount(self) -> None:
        """An empty token slot is hidden so its padding adds no gap."""
        async with StatusBarApp().run_test() as pilot:
            display = pilot.app.query_one("#tokens-display")
            assert display.display is False

    async def test_set_tokens_shows_then_zero_hides(self) -> None:
        """A positive count reveals the token slot; zeroing hides it again."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_tokens(5000)
            await pilot.pause()
            display = pilot.app.query_one("#tokens-display")
            assert display.display is True
            bar.set_tokens(0)
            await pilot.pause()
            assert display.display is False


class TestStatusMessageVisibility:
    """The status-message slot hides when empty so its padding adds no gap."""

    async def test_empty_message_hidden_on_mount(self) -> None:
        """The status-message slot starts empty and is hidden on mount."""
        async with StatusBarApp().run_test() as pilot:
            msg = pilot.app.query_one("#status-message")
            assert msg.display is False

    async def test_setting_message_shows_then_clearing_hides(self) -> None:
        """Setting a message reveals the slot; clearing it hides it again."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_status_message("Thinking")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message")
            assert msg.display is True
            bar.set_status_message("")
            await pilot.pause()
            assert msg.display is False

    async def test_busy_shows_slot_and_clearing_hides(self) -> None:
        """A busy indicator reveals the slot; clearing busy (no message) hides it."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_busy("Switching model")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message")
            assert msg.display is True
            bar.set_busy("")
            await pilot.pause()
            assert msg.display is False


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

    async def test_fireworks_routers_prefix_stripped(self) -> None:
        """The fireworks routers prefix is stripped before rendering."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/routers/glm-5p1-fast"
            await pilot.pause()
            rendered = str(label.render())
            assert "fireworks:glm-5p1-fast" in rendered
            assert "accounts/fireworks/routers/" not in rendered

    async def test_fireworks_prefix_stripped_case_insensitively(self) -> None:
        """A mixed-case fireworks ID is stripped, preserving the tail's casing.

        `detect_provider` resolves mixed-case `accounts/fireworks/...` IDs to
        the `fireworks` provider, so the display layer strips the prefix
        case-insensitively too. The remaining model name keeps its original
        casing rather than being lowercased.
        """
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "Accounts/Fireworks/Models/Kimi-K2P6"
            await pilot.pause()
            assert label._clean_model() == "Kimi-K2P6"
            rendered = str(label.render())
            assert "fireworks:Kimi-K2P6" in rendered
            assert "Accounts/Fireworks/Models/" not in rendered

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
        from deepagents_code.tui.widgets import status

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
            label.model = "gpt-5.5"
            await pilot.pause()
            rendered = str(label.render())
            assert "openai:gpt-5.5" in rendered

    async def test_effort_suffix_rendered(self) -> None:
        """Active reasoning effort should be shown next to the model."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_model(provider="openai", model="gpt-5.5", effort="xhigh")
            await pilot.pause()
            label = pilot.app.query_one("#model-display", ModelLabel)
            assert str(label.render()) == "openai:gpt-5.5 xhigh"

    async def test_effort_suffix_survives_provider_drop(self) -> None:
        """When narrow, provider is dropped before the effort label."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "openai"
            label.model = "gpt-5.5"
            label.effort = "xhigh"
            label.styles.width = 18
            await pilot.pause()
            assert str(label.render()) == "gpt-5.5 xhigh"

    async def test_effort_suffix_left_truncates_model(self) -> None:
        """Overflowing model text is left-truncated while the effort stays."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "openai"
            label.model = "gpt-5.5-turbo-preview"
            label.effort = "high"
            label.styles.width = 15
            await pilot.pause()
            width = label.content_size.width
            rendered = str(label.render())
            # Starts with an ellipsis (left-truncated) yet retains the effort
            # label — the branch that keeps effort while dropping model chars.
            assert rendered.startswith("…")
            assert rendered.endswith(" high")
            assert "openai:" not in rendered
            assert len(rendered) <= width

    async def test_effort_suffix_dropped_when_only_bare_model_fits(self) -> None:
        """In the narrow window where effort can't fit, the bare model wins.

        When the width is too small for even the left-truncated `model effort`
        form but still fits the bare model, the effort suffix is dropped rather
        than the model — the last rung before ellipsis truncation.
        """
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = ""
            label.model = "o1"
            label.effort = "medium"
            # padding 0 2 -> content width = 6: too narrow for "o1 medium" (9)
            # and below len("medium") + 2, but wide enough for bare "o1".
            label.styles.width = 10
            await pilot.pause()
            assert str(label.render()) == "o1"

    async def test_no_provider_no_stripping(self) -> None:
        """Without a provider, the model name is passed through unchanged."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = ""
            label.model = "accounts/fireworks/models/kimi-k2p6"
            await pilot.pause()
            rendered = str(label.render())
            assert "accounts/fireworks/models/kimi-k2p6" in rendered


class TestConnectionIndicator:
    """Tests for the connection-state indicator in the status bar."""

    async def test_indicator_empty_by_default(self) -> None:
        """The connection indicator should render nothing before any state is set."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert bar.connection_state == ""
            assert str(indicator.render()) == ""

    async def test_set_connecting_shows_message(self) -> None:
        """`set_connection('connecting')` should surface a Connecting message."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "Connecting" in str(indicator.render())

    async def test_set_reconnecting_shows_message(self) -> None:
        """`set_connection('reconnecting')` should surface a Reconnecting message."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("reconnecting")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "Reconnecting" in str(indicator.render())

    async def test_set_resuming_shows_message(self) -> None:
        """`set_connection('resuming')` should surface a Resuming message."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("resuming")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "Resuming" in str(indicator.render())

    async def test_clearing_connection_clears_indicator(self) -> None:
        """Returning to the empty state should clear the indicator text."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("reconnecting")
            await pilot.pause()
            bar.set_connection("")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert str(indicator.render()) == ""

    async def test_empty_indicator_is_hidden(self) -> None:
        """An empty indicator should be `display: none` so its padding adds no gap.

        The widget carries `padding: 0 1`; left visible while empty it would
        wedge two blank columns between the auto-approve pill and the cwd.
        """
        async with StatusBarApp().run_test() as pilot:
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert indicator.display is False

    async def test_set_connection_shows_indicator(self) -> None:
        """Setting a connection state should make the indicator visible again."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert indicator.display is True
            bar.set_connection("")
            await pilot.pause()
            assert indicator.display is False

    async def test_queued_count_shows_indicator(self) -> None:
        """A queued count alone should also surface (and later hide) the indicator."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_queued(2)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert indicator.display is True
            bar.set_queued(0)
            await pilot.pause()
            assert indicator.display is False

    async def test_invalid_state_raises(self) -> None:
        """An unrecognized connection state should raise `ValueError`."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            with pytest.raises(ValueError, match="Unknown connection state"):
                # Deliberately invalid to exercise the runtime guard; the
                # Literal-typed signature rejects it statically, hence the ignore.
                bar.set_connection("bogus")  # ty: ignore[invalid-argument-type]

    async def test_animation_starts_and_stops(self) -> None:
        """The spinner timer should run while connecting and stop after."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            await pilot.pause()
            assert bar._spinner_timer is not None
            bar.set_connection("")
            await pilot.pause()
            assert bar._spinner_timer is None

    async def test_spinner_glyph_rendered(self) -> None:
        """A real spinner frame should prefix the connection text."""
        from deepagents_code.config import get_glyphs

        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("reconnecting")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            rendered = str(indicator.render())
            frame, _, label = rendered.partition(" ")
            assert frame in get_glyphs().spinner_frames
            assert label == "Reconnecting"

    async def test_unmount_stops_spinner(self) -> None:
        """Leaving the DOM must stop the timer so it can't tick detached."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            await pilot.pause()
            assert bar._spinner_timer is not None
            await bar.remove()
            await pilot.pause()
            assert bar._spinner_timer is None

    def test_start_spinner_before_mount_is_noop(self) -> None:
        """`_start_spinner` must no-op before a live loop exists.

        `set_interval` requires the widget to be running; calling it pre-mount
        would raise, so the `not self._running` guard returns early instead.
        """
        bar = StatusBar()
        bar._start_spinner()
        assert bar._spinner_timer is None


class TestBusyIndicator:
    """Tests for the animated busy indicator used during model switches."""

    async def test_set_busy_shows_message_and_spinner(self) -> None:
        """`set_busy` should render a spinner-prefixed message and run the timer."""
        from deepagents_code.config import get_glyphs

        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_busy("Switching model")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            rendered = str(msg.render())
            assert "Switching model" in rendered
            # A spinner frame prefixes the message. Don't pin frame[0]: the
            # 0.1s timer may have ticked during the pause, so accept any frame.
            assert any(frame in rendered for frame in get_glyphs().spinner_frames)
            assert bar._spinner_timer is not None

    async def test_set_busy_treats_bracket_text_as_literal(self) -> None:
        """A model spec with markup-like brackets must render verbatim, not crash."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_busy("Switching to openai:[00]")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            assert "Switching to openai:[00]" in str(msg.render())

    async def test_clear_busy_stops_spinner_and_clears_message(self) -> None:
        """Clearing the busy state should stop the timer and empty the slot."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_busy("Switching model")
            await pilot.pause()
            bar.set_busy("")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            assert str(msg.render()) == ""
            assert bar._spinner_timer is None

    async def test_clear_busy_restores_status_message(self) -> None:
        """A status message set before busy should reappear once busy clears."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_status_message("Thinking")
            await pilot.pause()
            bar.set_busy("Switching model")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            assert "Switching" in str(msg.render())
            bar.set_busy("")
            await pilot.pause()
            assert str(msg.render()) == "Thinking"

    async def test_status_message_deferred_while_busy(self) -> None:
        """Regular status updates must not clobber an active busy indicator."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_busy("Switching model")
            await pilot.pause()
            bar.set_status_message("Executing")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            assert "Switching" in str(msg.render())

    async def test_busy_keeps_spinner_running_while_connecting(self) -> None:
        """Clearing busy while connecting must leave the shared spinner running."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            bar.set_busy("Switching model")
            await pilot.pause()
            assert bar._spinner_timer is not None
            bar.set_busy("")
            await pilot.pause()
            assert bar._spinner_timer is not None

    async def test_clear_busy_while_connecting_restores_message(self) -> None:
        """Clearing busy mid-connection restores the message and keeps the spinner.

        Exercises the combined state the shared-spinner refactor targets: the
        busy slot and the independent connection indicator are both active, and
        clearing busy must repaint the deferred message without stopping the
        spinner that the still-active connection state owns.
        """
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("connecting")
            bar.set_status_message("Thinking")
            bar.set_busy("Switching model")
            await pilot.pause()
            msg = pilot.app.query_one("#status-message", Static)
            conn = pilot.app.query_one("#connection-indicator", Static)
            # Busy owns the message slot; the connection indicator is separate.
            assert "Switching model" in str(msg.render())
            assert "Connecting" in str(conn.render())
            bar.set_busy("")
            await pilot.pause()
            # Deferred message reappears; spinner keeps ticking for the
            # still-active connection state.
            assert str(msg.render()) == "Thinking"
            assert "Connecting" in str(conn.render())
            assert bar._spinner_timer is not None

    async def test_set_busy_before_mount_does_not_raise(self) -> None:
        """`set_busy` on an unmounted bar is a safe no-op (no widgets, no timer)."""
        bar = StatusBar()
        bar.set_busy("Switching model")
        assert bar._busy_message == "Switching model"
        assert bar._spinner_timer is None
        bar.set_busy("")
        assert bar._busy_message == ""
        assert bar._spinner_timer is None


class TestQueuedCount:
    """Tests for the queued-message count in the connection indicator."""

    async def test_queued_count_hidden_at_zero(self) -> None:
        """A zero queue depth should leave the indicator empty."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_queued(0)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert str(indicator.render()) == ""

    async def test_queued_count_singular(self) -> None:
        """A single queued message should read in the singular."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_queued(1)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "1 message queued" in str(indicator.render())

    async def test_queued_count_plural(self) -> None:
        """Multiple queued messages should read in the plural."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_queued(3)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "3 messages queued" in str(indicator.render())

    async def test_negative_count_clamped(self) -> None:
        """Negative counts should clamp to zero and render nothing."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_queued(-5)
            await pilot.pause()
            assert bar.queued_count == 0
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert str(indicator.render()) == ""

    async def test_reconnecting_and_queued_combined(self) -> None:
        """Reconnecting plus queued messages should render both, joined."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("reconnecting")
            bar.set_queued(2)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            rendered = str(indicator.render())
            assert "Reconnecting" in rendered
            assert "2 messages queued" in rendered

    async def test_combined_indicator_uses_ascii_separator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ASCII glyph mode should not leak Unicode in the combined indicator."""
        from deepagents_code.config import ASCII_GLYPHS, UNICODE_GLYPHS

        monkeypatch.setattr(
            "deepagents_code.tui.widgets.status.get_glyphs",
            lambda: ASCII_GLYPHS,
        )
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("reconnecting")
            bar.set_queued(2)
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            rendered = str(indicator.render())
            assert f" {ASCII_GLYPHS.bullet} " in rendered
            # Derive the forbidden separator from the Unicode glyph itself so the
            # guard can't drift to the wrong codepoint (the bullet is U+2022 `•`,
            # not the U+00B7 middle dot `·`).
            assert f" {UNICODE_GLYPHS.bullet} " not in rendered
