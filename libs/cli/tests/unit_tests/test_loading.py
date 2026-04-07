"""Unit tests for the LoadingWidget."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from deepagents_cli.widgets.loading import LoadingWidget


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
