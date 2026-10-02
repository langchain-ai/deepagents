"""Unit tests for the LoadingWidget."""

from __future__ import annotations

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

    @pytest.mark.parametrize("next_status", ["Thinking", "Offloading"])
    async def test_responding_freezes_display_without_losing_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        next_status: str,
    ) -> None:
        """Streaming freezes only the counter, then catches up on the next activity."""
        now = 112.7
        monkeypatch.setattr("deepagents_code.tui.widgets.loading.time", lambda: now)
        async with LoadingWidgetApp().run_test() as pilot:
            widget = pilot.app.query_one("#loading", LoadingWidget)
            widget._start_time = 100.0
            widget.set_status("Responding")
            assert widget._hint_widget is not None
            assert str(widget._hint_widget.render()) == "(12s, esc to interrupt)"

            now = 145.0
            widget.set_status("Responding")
            position = widget._spinner._position
            widget._update_animation()
            assert widget._spinner._position != position
            assert str(widget._hint_widget.render()) == "(12s, esc to interrupt)"

            widget.set_status(next_status)
            assert str(widget._hint_widget.render()) == "(45s, esc to interrupt)"

            now = 150.0
            widget.set_status("Responding")
            now = 155.0
            widget._update_animation()
            assert str(widget._hint_widget.render()) == "(50s, esc to interrupt)"

            widget.pause()
            now = 200.0
            widget.resume()
            assert str(widget._hint_widget.render()) == "(55s, esc to interrupt)"

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
