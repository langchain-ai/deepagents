"""Lay out tables like Textual's keylined `MarkdownTableContent` grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.cells import cell_len
from textual.content import Content

from deepagents_code.tui.widgets.final_markdown._rows import Row, stylize

if TYPE_CHECKING:
    from deepagents_code.tui.widgets.final_markdown._parse import Node

_KEYLINE = {
    "top": ("┌", "┬", "┐"),
    "mid": ("├", "┼", "┤"),
    "bottom": ("└", "┴", "┘"),
}
"""Thin keyline joints `GridLayout` draws around table cells.

Keylines come from Textual's canvas, not its border tables, so ASCII mode
does not change them.
"""


class Table:
    """Lay out a table like `MarkdownTableContent`'s keylined grid."""

    def __init__(self, node: Node, width: int) -> None:
        self.headers = node.headers
        self.body = [row for row in node.rows if row]
        self.width = width

    def columns(self) -> list[int]:
        """Resolve column widths with the grid's auto, expand, and shrink rules.

        Returns:
            Each column's width, including its one-cell side padding.
        """
        from textual._resolve import resolve  # noqa: PLC2701  # exact grid parity
        from textual.css.scalar import Scalar
        from textual.geometry import Size

        count = max(len(self.headers), 1)
        optimal = [0] * count
        minimal = [1] * count
        for row in (self.headers, *self.body):
            for index, cell in enumerate(row[:count]):
                lines = cell.plain.split("\n")
                optimal[index] = max(optimal[index], max(map(cell_len, lines)) + 2)
                words = cell.plain.split()
                widest = max(map(cell_len, words), default=0)
                minimal[index] = max(minimal[index], widest + 2)
        size = Size(max(self.width - 2, 0), 0)
        resolved = resolve(
            [Scalar.from_number(width) for width in optimal],
            size.width,
            1,
            size,
            size,
            expand=True,
            shrink=True,
            minimums=minimal,
        )
        return [length for _, length in resolved]

    def rows(self, background: str) -> list[Row]:
        """Render the keylined table; every row copies as drawn.

        Args:
            background: The table's background style, if any.

        Returns:
            The table rows.
        """
        widths = self.columns()
        rows = [self.rule("top", widths)]
        for index, cells in enumerate((self.headers, *self.body)):
            if index:
                rows.append(self.rule("mid", widths))
            rows.extend(self.cells(cells, widths, header=not index))
        rows.append(self.rule("bottom", widths))
        return stylize(rows, background)

    @staticmethod
    def rule(edge: str, widths: list[int]) -> Row:
        """Render a horizontal keyline.

        Returns:
            The keyline row.
        """
        left, join, right = _KEYLINE[edge]
        text = left + join.join("─" * width for width in widths) + right
        return Row(((Content.styled(text, ".markdown--table-keyline"), True),))

    @staticmethod
    def cells(cells: list[Content], widths: list[int], *, header: bool) -> list[Row]:
        """Render one table row, wrapping each cell inside its padding.

        Header cells align to the bottom of the row, as `content-align: left
        bottom` does in `MarkdownTableContent`.

        Returns:
            The row's lines, as tall as its tallest cell.
        """
        wrapped = [
            _cell_lines(cells[index] if index < len(cells) else Content(), width - 2)
            for index, width in enumerate(widths)
        ]
        height = max((len(lines) for lines in wrapped), default=0)
        if header:
            wrapped = [[Content()] * (height - len(lines)) + lines for lines in wrapped]
        bar = Content.styled("│", ".markdown--table-keyline")
        rows: list[Row] = []
        for y in range(height):
            parts = []
            for lines, width in zip(wrapped, widths, strict=True):
                text = lines[y] if y < len(lines) else Content()
                cell = Content.assemble(" ", text, " " * (width - 1 - text.cell_length))
                parts.append(
                    cell.stylize_before(".markdown--table-header") if header else cell
                )
            rows.append(Row(((Content.assemble(bar, bar.join(parts), bar), True),)))
        return rows


def _cell_lines(content: Content, width: int) -> list[Content]:
    """Wrap a table cell like a `Static` with `text-overflow: ellipsis`.

    Returns:
        The cell's lines; none when the column has no room for text.
    """
    if width <= 0:
        return []
    return content.wrap(width, overflow="ellipsis")
