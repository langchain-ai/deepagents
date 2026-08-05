"""Unit tests for the shared click-to-copy span metadata helpers."""

from __future__ import annotations

from textual.style import Style as TStyle

from deepagents_code.tui.widgets._copy_spans import copy_span_style, copy_span_target


class TestCopySpanRoundTrip:
    """`copy_span_style` output is readable by `copy_span_target`."""

    def test_round_trip(self) -> None:
        style = copy_span_style("abc-123", "Thread ID")

        assert copy_span_target(style) == ("abc-123", "Thread ID")

    def test_combines_with_a_visual_style(self) -> None:
        style = TStyle(dim=True) + copy_span_style("abc-123", "Thread ID")

        assert style.dim is True
        assert copy_span_target(style) == ("abc-123", "Thread ID")


class TestCopySpanTargetRejects:
    """Styles without usable copy metadata resolve to `None`."""

    def test_style_without_meta(self) -> None:
        assert copy_span_target(TStyle(dim=True)) is None

    def test_style_string(self) -> None:
        assert copy_span_target("dim") is None

    def test_empty_text(self) -> None:
        assert copy_span_target(copy_span_style("", "Thread ID")) is None

    def test_empty_label(self) -> None:
        assert copy_span_target(copy_span_style("abc-123", "")) is None

    def test_non_string_values(self) -> None:
        style = TStyle.from_meta({"copy_text": 1, "copy_label": 2})

        assert copy_span_target(style) is None
