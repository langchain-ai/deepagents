"""Tests for the context usage visualization."""

from unittest.mock import patch

from textual.geometry import Size

from deepagents_code import theme
from deepagents_code.tui.widgets.context_usage import _ContextUsage


def test_usage_bars_use_full_cell_backgrounds() -> None:
    usage = _ContextUsage(
        context_tokens=50,
        conversation_tokens=30,
        context_limit=100,
        model_spec="model",
        approximate=False,
    )
    with (
        patch.object(_ContextUsage, "content_size", property(lambda _: Size(20, 5))),
        patch.object(theme, "get_theme_colors", return_value=theme.DARK_COLORS),
    ):
        segments = usage.render().render_segments()

    bars = [
        segment.text for segment in segments if segment.style and segment.style.reverse
    ]
    assert bars == [" " * 4, " " * 6, " " * 10, "  ", "  ", "  "]
