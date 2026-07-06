"""Unit tests for the LoadingWidget."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult

from deepagents_code.tui.widgets.loading import LoadingWidget


class LoadingWidgetApp(App[None]):
    """Minimal app that mounts a LoadingWidget for testing."""

    def compose(self) -> ComposeResult:
        widget = LoadingWidget()
        widget.id = "loading"
        yield widget


class TestLoadingWidget:
    """Tests for LoadingWidget timer behavior."""

    async def test_stop_halts_animation_while_widget_remains_mounted(self) -> None:
        """Calling `stop()` should stop advancing the animation timer."""
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)

            await asyncio.sleep(0.25)
            await pilot.pause()

            widget.stop()
            position_after_stop = widget._spinner._position

            await asyncio.sleep(0.25)
            await pilot.pause()

            assert widget._spinner._position == position_after_stop

    async def test_unmount_stops_animation_timer(self) -> None:
        """Unmounting the widget should stop and clear the animation timer."""
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)

            assert widget._animation_timer is not None

            await widget.remove()
            await pilot.pause()

            assert widget._animation_timer is None
            assert not pilot.app.query("LoadingWidget")

    async def test_double_stop_is_safe(self) -> None:
        """Calling `stop()` then `remove()` should not raise."""
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)

            widget.stop()
            assert widget._animation_timer is None

            await widget.remove()
            await pilot.pause()

            assert widget._animation_timer is None
            assert not pilot.app.query("LoadingWidget")

    def test_pause_resume_excludes_paused_duration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Elapsed time should not include time spent paused for HITL approval."""
        now = 100.0

        def fake_time() -> float:
            return now

        monkeypatch.setattr("deepagents_code.tui.widgets.loading.time", fake_time)
        widget = LoadingWidget()
        widget._start_time = now

        now = 112.5
        widget.pause()

        now = 145.0
        widget.resume()

        assert widget._start_time == pytest.approx(132.5)
        assert not widget._paused

    def test_resume_when_not_paused_leaves_start_time_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`resume()` must be a no-op when the widget was never paused.

        The resume callback fires on every approval-future completion and as a
        `_set_spinner` backstop, so it can land on a never-paused (or
        replacement) widget. It must not rebase `_start_time` there.
        """
        monkeypatch.setattr(
            "deepagents_code.tui.widgets.loading.time",
            lambda: 999.0,
        )
        widget = LoadingWidget()
        widget._start_time = 100.0
        widget._paused_elapsed = 12.5  # stale value from a prior pause cycle

        widget.resume()

        assert widget._start_time == pytest.approx(100.0)
        assert not widget._paused

    def test_pause_without_start_time_does_not_raise(self) -> None:
        """`pause()` before the timer starts must not raise or fabricate time."""
        widget = LoadingWidget()
        assert widget._start_time is None

        widget.pause()

        assert widget._paused
        assert widget._paused_elapsed == pytest.approx(0.0)

    async def test_pause_hint_renders_whole_seconds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The paused hint shows whole seconds, matching the live counter."""
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)
            widget._start_time = 100.0
            monkeypatch.setattr(
                "deepagents_code.tui.widgets.loading.time",
                lambda: 112.7,
            )

            widget.pause()

            assert widget._hint_widget is not None
            assert str(widget._hint_widget.render()) == "(paused at 12s)"

    async def test_on_mount_preserves_start_time_across_remount(self) -> None:
        """`on_mount` must not reset `_start_time` when it is already set.

        The spinner is repositioned by reordering children (preserving state),
        but if any code path falls back to remove + re-mount, the elapsed-time
        counter must not restart — that would visibly jump the "(Ns)" hint
        back to 0s mid-stream.
        """
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)
            original_start = widget._start_time
            assert original_start is not None

            await widget.remove()
            await pilot.pause()

            await pilot.app.mount(widget)
            await pilot.pause()

            assert widget._start_time == original_start
