"""Responsive render widgets for the context-usage modal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.content import Content
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code._session_stats import format_token_count
from deepagents_code.config import get_glyphs

if TYPE_CHECKING:
    from textual.widget import Widget

    from deepagents_code.tui.widgets.context_usage._models import (
        _ColorRole,
        _Snapshot,
    )


def _compact_tokens(tokens: int) -> str:
    return format_token_count(tokens).replace(".0K", "K").replace(".0M", "M")


def _category_color(widget: Widget, role: _ColorRole) -> str:
    colors = theme.get_theme_colors(widget)
    return {
        "warning": colors.warning,
        "primary": colors.primary,
        "secondary": colors.secondary,
        "accent": colors.accent,
        "muted": colors.muted,
    }[role]


def _allocate_widths(snapshot: _Snapshot, width: int) -> list[int]:
    categories = snapshot.categories
    nonempty = [index for index, category in enumerate(categories) if category.tokens]
    widths = [0] * len(categories)
    if not nonempty or width <= 0:
        return widths
    if width <= len(nonempty):
        for index in nonempty[:width]:
            widths[index] = 1
        return widths

    remaining = width - len(nonempty)
    raw = [
        category.tokens / snapshot.scale_tokens * remaining for category in categories
    ]
    for index in nonempty:
        widths[index] = 1 + int(raw[index])
    remainder = width - sum(widths)
    order = sorted(nonempty, key=lambda index: raw[index] % 1, reverse=True)
    for offset in range(remainder):
        widths[order[offset % len(order)]] += 1
    return widths


def _scale_line(width: int, maximum: int) -> str:
    line = [" "] * width
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        label = _compact_tokens(round(maximum * fraction))
        start = min(round((width - 1) * fraction), max(0, width - len(label)))
        visible = label[: width - start]
        line[start : start + len(visible)] = visible
    return "".join(line)


class _ContextHeader(Static):
    def __init__(self, snapshot: _Snapshot) -> None:
        super().__init__()
        self._snapshot = snapshot

    def render(self) -> Content:
        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        model = self._snapshot.model_spec or "Unknown model"
        maximum = (
            f"{_compact_tokens(self._snapshot.context_limit)} Max"
            if self._snapshot.context_limit is not None
            else "Max unavailable"
        )
        left = Content.assemble(
            ("Context", f"bold {colors.primary}"),
            (f" {glyphs.bullet} ", colors.muted),
            model,
            " ",
            maximum,
        )
        used = _compact_tokens(self._snapshot.displayed_usage)
        prefix = (
            "~"
            if self._snapshot.approximate or self._snapshot.context_tokens is None
            else ""
        )
        right_text = f"{prefix}{used}"
        if self._snapshot.context_limit is not None:
            right_text += f" / {_compact_tokens(self._snapshot.context_limit)}"
        right = Content(right_text)
        if self._snapshot.context_tokens is not None and self._snapshot.context_limit:
            percent = self._snapshot.context_tokens / self._snapshot.context_limit * 100
            right = Content.assemble(right, (f"  {percent:.1f}%", colors.success))

        gap = self.content_size.width - left.cell_length - right.cell_length
        title = (
            Content.assemble(left, " " * gap, right)
            if gap > 0
            else Content.assemble(left, "\n", right)
        )
        subtitle = Content.styled("Current context usage by category.", colors.muted)
        return Content("\n").join((title, subtitle))


class _ContextBar(Static):
    def __init__(self, snapshot: _Snapshot) -> None:
        super().__init__()
        self._snapshot = snapshot

    def render(self) -> Content:
        width = max(self.content_size.width, 1)
        glyph = get_glyphs().box_horizontal
        segments = [
            Content.styled(glyph * segment_width, _category_color(self, category.color))
            for category, segment_width in zip(
                self._snapshot.categories,
                _allocate_widths(self._snapshot, width),
                strict=True,
            )
            if segment_width
        ]
        bar = Content.assemble(*segments)
        scale = Content.styled(
            _scale_line(width, self._snapshot.scale_tokens),
            theme.get_theme_colors(self).muted,
        )
        return Content("\n").join((bar, scale))


class _ContextLegend(Static):
    def __init__(self, snapshot: _Snapshot) -> None:
        super().__init__()
        self._snapshot = snapshot

    def render(self) -> Content:
        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        marker = glyphs.box_horizontal * 2
        width = max(self.content_size.width, 1)
        rows: list[Content] = []
        for category in self._snapshot.categories:
            color = _category_color(self, category.color)
            percent = category.tokens / self._snapshot.scale_tokens * 100
            value = (
                f"{_compact_tokens(category.tokens)}  {glyphs.bullet}  {percent:.1f}%"
            )
            label_style = (
                colors.muted if category.color == "muted" else colors.foreground
            )
            left = Content.assemble((marker, color), " ", (category.label, label_style))
            gap = max(1, width - left.cell_length - len(value))
            rows.append(Content.assemble(left, " " * gap, (value, colors.muted)))
        if not rows:
            rows.append(Content.styled("No context usage reported yet.", colors.muted))
        return Content("\n").join(rows)
