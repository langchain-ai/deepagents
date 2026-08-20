"""Tests for the YOLO first-enable notice body."""

from deepagents_code.tui.widgets.yolo_mode_notice import YOLO_MODE_NOTICE_BODY


def test_exit_hint_starts_on_a_new_line() -> None:
    """Keep the YOLO exit hint visually separate from the warning."""
    assert "\nLeave YOLO any time with **Shift+Tab**." in YOLO_MODE_NOTICE_BODY
