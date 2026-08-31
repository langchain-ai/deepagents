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
