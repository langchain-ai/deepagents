"""Unit tests for the StatusBar widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import Static

from deepagents_code._env_vars import HIDE_CWD
from deepagents_code.config import ASCII_GLYPHS, reset_glyphs_cache
from deepagents_code.tui.widgets.status import (
    _PICKER_TARGET_META,
    BranchLabel,
    ModelLabel,
    StatusBar,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.style import Style
    from textual.pilot import Pilot


@pytest.fixture(autouse=True)
def reset_glyphs_between_tests() -> Iterator[None]:
    """Clear process-global glyph detection before and after each test."""
    reset_glyphs_cache()
    yield
    reset_glyphs_cache()


class StatusBarApp(App[None]):
    """Minimal app that mounts a StatusBar for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.opened_pickers: list[str] = []
        self.unhandled_clicks = 0

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")

    def action_open_model_selector(self) -> None:
        self.opened_pickers.append("model")

    def action_open_effort_selector(self) -> None:
        self.opened_pickers.append("effort")

    def on_click(self, event: events.Click) -> None:
        """Count clicks that reach the app, standing in for the real handler.

        `DeepAgentsApp.on_click` refocuses the chat input, so a picker click must
        not bubble here while a plain click must.
        """
        del event
        self.unhandled_clicks += 1


class TestApprovalModeDisplay:
    """Tests for the three-state approval indicator."""


class TestCwdDisplay:
    """Tests for the cwd display in the status bar."""


class TestBranchDisplay:
    """Tests for the git branch display in the status bar."""

    @staticmethod
    def _visible_branch_text(display: BranchLabel) -> str:
        """Return the branch text as actually rendered to the terminal line."""
        from rich.segment import Segment

        return "".join(
            seg.text for seg in display.render_line(0) if isinstance(seg, Segment)
        )

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


class TestEdgeAlignment:
    """Tests that the status bar spans the full terminal width."""


class TestTokenDisplay:
    """Tests for the token count display in the status bar."""

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
            assert "5K" in str(display.render())

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
            assert "5K+" in rendered


class TestCostDisplay:
    """Tests for cumulative cost rendered inline with context tokens."""


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

    async def test_truncation_uses_stripped_name(self) -> None:
        """Ellipsis truncation slices the stripped name; the raw prefix never leaks."""
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            # Two columns are padding, leaving five for truncated content.
            label.styles.width = 7
            await pilot.pause()
            rendered = str(label.render())
            assert rendered == "…k2p6"
            assert "accounts" not in rendered

    async def test_ascii_truncation_uses_ascii_ellipsis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.tui.widgets.status.get_glyphs", lambda: ASCII_GLYPHS
        )
        async with StatusBarApp().run_test() as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "fireworks"
            label.model = "accounts/fireworks/models/kimi-k2p6"
            label.styles.width = 9
            await pilot.pause()

            rendered = str(label.render())
            assert rendered.startswith(ASCII_GLYPHS.ellipsis)
            assert rendered.isascii()
            assert len(rendered) <= label.content_size.width


class TestPickerTargetRegistries:
    """Tests that the picker mappings stay total over `PickerTarget`."""


class TestModelLabelClickTargets:
    """Tests for the model and effort status-bar click targets."""

    @staticmethod
    def _offset_for_target(label: ModelLabel, target: str) -> Offset:
        x = label.content_region.x - label.region.x
        for segment in label.render_line(0):
            if (
                segment.style is not None
                and segment.style.meta.get(_PICKER_TARGET_META) == target
            ):
                return Offset(x, 0)
            x += segment.cell_length
        msg = f"No rendered segment for {target}"
        raise AssertionError(msg)

    @staticmethod
    def _style_at(label: ModelLabel, x: int) -> Style | None:
        """Return the style of the painted cell at `x`, as Textual would report it.

        `render_line` yields segments, not cells, so walk their widths rather
        than indexing the segment list.
        """
        cell = label.content_region.x - label.region.x
        for segment in label.render_line(0):
            if cell <= x < cell + segment.cell_length:
                return segment.style
            cell += segment.cell_length
        msg = f"No painted cell at x={x}"
        raise AssertionError(msg)

    @classmethod
    def _app_mouse_event(
        cls,
        event_type: type[events.MouseDown | events.MouseUp],
        label: ModelLabel,
        offset: Offset,
    ) -> events.MouseDown | events.MouseUp:
        """Build an app-level mouse event that follows Textual's real input path."""
        target = label.content_region.offset + offset
        return event_type(
            None,
            x=target.x,
            y=target.y,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=target.x,
            screen_y=target.y,
            style=cls._style_at(label, offset.x),
        )

    @staticmethod
    def _rendered_targets(label: ModelLabel) -> dict[str, tuple[str, bool]]:
        """Collect picker targets from painted output, not from `render`.

        Reading `render_line` is what makes the assertions fail if
        `_hovered_target` ever stops repainting the widget.
        """
        targets: dict[str, tuple[str, bool]] = {}
        for segment in label.render_line(0):
            style = segment.style
            if style is None:
                continue
            meta_target = style.meta.get(_PICKER_TARGET_META)
            if meta_target is None:
                continue
            # Adjacent segments can split one span, so accumulate by target.
            previous = targets.pop(meta_target, None)
            merged = (previous[0] if previous else "") + segment.text
            # `Style.underline` is tri-state; only "set" matters here.
            targets[meta_target] = (merged, bool(style.underline))
        return {
            span_text: (target, underline)
            for target, (span_text, underline) in targets.items()
        }

    async def test_picker_clicks_do_not_bubble(self) -> None:
        """A target click must not race the app's chat-input refocus handler."""
        app = StatusBarApp()
        async with app.run_test(size=(150, 24)) as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "openai"
            label.model = "gpt-5.5"
            label.effort = "high"
            await pilot.pause()

            model_offset = self._offset_for_target(label, "model")
            await pilot.click(label, offset=model_offset)
            await pilot.pause()

            assert app.unhandled_clicks == 0
            assert app.opened_pickers == ["model"]

    async def test_keyboard_refocus_does_not_block_click(self) -> None:
        """A refocus without an adjacent mouse press must not consume a later click."""
        app = StatusBarApp()
        async with app.run_test(size=(150, 24)) as pilot:
            label = pilot.app.query_one("#model-display", ModelLabel)
            label.provider = "openai"
            label.model = "gpt-5.5"
            await pilot.pause()

            app.post_message(events.AppBlur())
            await pilot.pause()
            app.post_message(events.AppFocus())
            await pilot.pause()
            await pilot.click(label, offset=self._offset_for_target(label, "model"))
            await pilot.pause()

            assert app.opened_pickers == ["model"]

    @classmethod
    async def _move(
        cls,
        pilot: Pilot[None],
        label: ModelLabel,
        offset: Offset,
    ) -> None:
        """Post a mouse move at `offset` within `label`."""
        target = label.content_region.offset + offset
        label.post_message(
            events.MouseMove(
                label,
                x=offset.x,
                y=offset.y,
                delta_x=0,
                delta_y=0,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=target.x,
                screen_y=target.y,
                style=cls._style_at(label, offset.x),
            )
        )
        await pilot.pause()


class TestConnectionIndicator:
    """Tests for the connection-state indicator in the status bar."""

    async def test_set_resuming_shows_message(self) -> None:
        """`set_connection('resuming')` should surface a Resuming message."""
        async with StatusBarApp().run_test() as pilot:
            bar = pilot.app.query_one("#status-bar", StatusBar)
            bar.set_connection("resuming")
            await pilot.pause()
            indicator = pilot.app.query_one("#connection-indicator", Static)
            assert "Resuming" in str(indicator.render())


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


class TestQueuedCount:
    """Tests for the queued-message count in the connection indicator."""

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
