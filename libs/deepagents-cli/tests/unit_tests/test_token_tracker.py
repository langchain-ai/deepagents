"""Tests for TextualTokenTracker."""

from deepagents_cli.app import TextualTokenTracker


class TestTextualTokenTracker:
    def test_add_updates_context_and_calls_callback(self):
        """Token add() should update current_context and call update callback."""
        called_with = []
        tracker = TextualTokenTracker(lambda x: called_with.append(x))

        tracker.add(1500, 200)

        assert tracker.current_context == 1500
        assert called_with == [1500]

    def test_reset_clears_context_and_calls_callback_with_zero(self):
        """Token reset() should set context to 0 and call callback with 0."""
        called_with = []
        tracker = TextualTokenTracker(lambda x: called_with.append(x))
        tracker.add(1500, 200)
        called_with.clear()

        tracker.reset()

        assert tracker.current_context == 0
        assert called_with == [0]

    def test_hide_calls_hide_callback(self):
        """Token hide() should call the hide callback."""
        hide_called = []
        tracker = TextualTokenTracker(
            lambda x: None, hide_callback=lambda: hide_called.append(True)
        )

        tracker.hide()

        assert hide_called == [True]

    def test_hide_without_callback_is_noop(self):
        """Token hide() should be safe when no hide callback provided."""
        tracker = TextualTokenTracker(lambda x: None)
        tracker.hide()  # Should not raise
