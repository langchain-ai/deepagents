"""Lay out parsed blocks with the margins and styles of Textual's `Markdown`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.content import Content

from deepagents_code.tui.widgets.final_markdown._parse import LISTS, Node
from deepagents_code.tui.widgets.final_markdown._rows import (
    Block,
    Row,
    blank,
    pad,
    stack,
    stylize,
    wrap_prose,
)
from deepagents_code.tui.widgets.final_markdown._table import Table

if TYPE_CHECKING:
    from textual.highlight import HighlightTheme

_BULLETS = ("• ", "▪ ", "‣ ", "⭑ ", "◦ ")
"""Unordered list bullets by nesting depth, as in `Markdown.BULLETS`."""

_HEADING_MARGINS = {
    "h1": (2, 1),
    "h2": (2, 1),
    "h3": (1, 1),
    "h4": (1, 1),
    "h5": (1, 1),
    "h6": (1, 1),
}
"""(top, bottom) margins of each heading level in Textual's `Markdown` CSS."""


@dataclass(frozen=True, slots=True)
class Theme:
    """Theme facts that change block geometry, colors, or fence highlighting.

    Block backgrounds and borders with alpha are inline styles rather than
    component classes: a partial component style pre-blends its alpha
    against transparent black, which darkens a light theme's `white 30%`.
    """

    ansi: bool
    dark: bool
    quote_bar: str
    rule: str

    @property
    def fence(self) -> str:
        """`MarkdownFence` color and background, per its `:light`/`:ansi` CSS."""
        if self.ansi:
            return "rgb(210,210,210)"
        return f"rgb(210,210,210) on {'black 10%' if self.dark else 'white 30%'}"

    @property
    def quote_border(self) -> str:
        """`MarkdownBlockQuote` border color."""
        return "$text-primary 50%" if self.dark else "$text-secondary"

    @property
    def table(self) -> str:
        """`MarkdownTable` background, set only in light themes."""
        return "" if self.dark else "on white 30%"


class Layout:
    """Lay out parsed blocks at a width, following Textual's block CSS."""

    def __init__(self, theme: Theme) -> None:
        self.theme = theme

    def blocks(self, nodes: list[Node], width: int, depth: int = 0) -> list[Block]:
        """Lay out sibling blocks.

        Args:
            nodes: Sibling nodes.
            width: Cells available to them.
            depth: Unordered list items enclosing them, which picks the bullet.

        Returns:
            The laid-out blocks with their margins.
        """
        return [self.block(node, max(width, 1), depth) for node in nodes]

    def block(self, node: Node, width: int, depth: int) -> Block:
        """Lay out one block.

        Returns:
            The block's rows and margins.
        """
        kind = node.kind
        if kind in _HEADING_MARGINS:
            return self.heading(node, width)
        if kind == "paragraph":
            return Block(wrap_prose(node.inline, width))
        if kind == "fence":
            return self.fence(node, width)
        if kind == "hr":
            rule = Content.styled(self.theme.rule * width, ".markdown--hr")
            return Block(
                [blank(margin=False), Row(((rule, False),), joiner=None)], 0, 1
            )
        if kind == "blockquote":
            return self.quote(node, width, depth)
        if kind in LISTS:
            return self.list_block(node, width, depth)
        if kind == "table":
            return Block(Table(node, width).rows(self.theme.table), 0, 1)
        return Block(stack(self.blocks(node.children, width, depth)))

    def heading(self, node: Node, width: int) -> Block:  # noqa: PLR6301  # sibling of block layouts
        """Lay out a heading across the full width; `h1` is centered.

        Returns:
            The heading block.
        """
        rows = wrap_prose(node.inline, width)
        if node.kind == "h1":
            rows = _center(rows, width)
        top, bottom = _HEADING_MARGINS[node.kind]
        return Block(stylize(rows, f".markdown--{node.kind}"), top, bottom)

    def fence(self, node: Node, width: int) -> Block:
        """Lay out a code block, folding lines longer than the block.

        Folding matches `AssistantMessage`'s streaming fence CSS
        (`text-wrap: nowrap; text-overflow: fold`), so the height holds when
        the reply finishes.

        Returns:
            The fence block.
        """
        from textual.highlight import highlight

        padding = 0 if self.theme.ansi else 2
        inner = max(width - 2 * padding, 1)
        code = highlight(
            node.code, language=node.lexer or None, theme=self.highlight_theme()
        )
        pad = Content(" " * padding)
        rows = [blank(margin=False)]
        for line in code.split("\n", allow_blank=True):
            expanded = line.expand_tabs(8)
            cuts = list(range(0, expanded.cell_length, inner))[1:]
            # Textual cuts at cell positions but divides at character offsets,
            # so a wide-character line yields pieces wider than the block that
            # the fence label then crops. Crop them the same way.
            for index, piece in enumerate(expanded.divide(cuts)):
                piece = piece.truncate(inner)  # noqa: PLW2901  # see above
                rows.append(
                    Row(((pad, False), (piece, True)), joiner="" if index else "\n")
                )
        rows.append(blank(margin=False))
        margin = 0 if self.theme.ansi else 1
        return Block(stylize(rows, self.theme.fence), margin, margin)

    def highlight_theme(self) -> type[HighlightTheme]:
        """Return the highlight theme `MarkdownFence.highlight` would pick."""
        from textual import highlight

        if not self.theme.ansi:
            return highlight.HighlightTheme
        if self.theme.dark:
            return highlight.ANSIDarkHighlightTheme
        return highlight.ANSILightHighlightTheme

    def quote(self, node: Node, width: int, depth: int) -> Block:
        """Lay out a block quote: an outer left border and one cell of padding.

        Returns:
            The quote block.
        """
        inner = max(width - 3, 1)
        rows = stack(self.blocks(node.children, inner, depth))
        space = (Content(" "), False)
        body = stylize(
            [
                Row((space, *pad(row, inner)), (), row.joiner, row.margin)
                for row in rows
            ],
            "on $boost",
        )
        # Textual blends the border color over the background outside the
        # quote, so the border color goes beneath the quote background.
        bar = Content.styled(self.theme.quote_bar, self.theme.quote_border)
        bar = bar.stylize("on $boost")
        quoted = [
            Row(((bar, False), *row.parts), row.fill, row.joiner, row.margin)
            for row in body
        ]
        return Block(quoted, 1, 1)

    def list_block(self, node: Node, width: int, depth: int) -> Block:
        """Lay out a list: a bullet column beside each item's blocks.

        Returns:
            The list block. A list inside another list has no margin.
        """
        items = [child for child in node.children if child.kind == "list_item"]
        ordered = node.kind == "ordered_list"
        if ordered:
            symbols = _ordered_symbols(items)
        else:
            symbols = [_BULLETS[depth % len(_BULLETS)]] * len(items)
        item_depth = depth if ordered else depth + 1
        rows: list[Row] = []
        for item, symbol in zip(items, symbols, strict=True):
            body_width = width - len(symbol)
            body = stack(self.blocks(item.children, body_width, item_depth))
            # An empty item has no rows: its bullet gets no height either.
            if body:
                rows.extend(_bullet(body, symbol))
        return Block(rows, 0, 0 if node.nested else 1)


def _center(rows: list[Row], width: int) -> list[Row]:
    """Center rows as one block, like `content-align: center` (`Strip.align`).

    Returns:
        The rows, shifted right by half the space beside the widest row.
    """
    shift = max(0, width - max(row.cell_length for row in rows)) // 2
    pad = (Content(" " * shift), False)
    return [
        row
        if row.cell_length == width or not shift
        else Row((pad, *row.parts), joiner=row.joiner)
        for row in rows
    ]


def _ordered_symbols(items: list[Node]) -> list[str]:
    """Number ordered items the way `MarkdownOrderedList.compose` does.

    Returns:
        Right-aligned `N. ` markers, one cell wider than the longest.
    """
    first = items[0].number if items else ""
    start = int(first) if first.isdigit() else 1
    labels = [f"{number}. " for number in range(start, start + len(items))]
    size = max((len(label) for label in labels), default=0)
    return [label.rjust(size + 1) for label in labels]


def _bullet(rows: list[Row], symbol: str) -> list[Row]:
    """Put `symbol` beside the first row and indent the rest.

    The marker is selectable, so copied list items keep their bullets.

    Returns:
        The item rows.
    """
    marker = symbol.lstrip()
    indent = len(symbol) - len(marker)
    lead: tuple[tuple[Content, bool], ...] = (
        (Content.styled(marker, ".markdown--bullet"), True),
    )
    if indent:
        lead = ((Content(" " * indent), False), *lead)
    rest = ((Content(" " * len(symbol)), False),)
    head, *tail = rows
    return [
        Row((*lead, *head.parts), head.fill, head.joiner or "\n"),
        *(Row((*rest, *row.parts), row.fill, row.joiner, row.margin) for row in tail),
    ]


def layout(nodes: list[Node], width: int, theme: Theme) -> list[Row]:
    """Lay out a whole reply.

    Only top-level paragraphs get a bottom margin (`Markdown >
    MarkdownParagraph`). `AssistantMessage` sets `margin-bottom: 0` on the
    last block so a reply has no trailing blank row. In Textual that rule
    replaces the whole margin, so the last block loses its top margin too.

    Returns:
        The reply's rows.
    """
    blocks = Layout(theme).blocks(nodes, width)
    for node, block in zip(nodes, blocks, strict=True):
        if node.kind == "paragraph":
            block.bottom = 1
    if blocks:
        blocks[-1].top = blocks[-1].bottom = 0
    return stack(blocks)
