"""Rendered rows and the helpers that compose them."""

from __future__ import annotations

from dataclasses import dataclass

from textual.content import Content, Span


@dataclass(frozen=True, slots=True)
class Row:
    """One rendered line of the reply.

    `parts` holds `(content, selectable)` pairs. Selectable parts carry the
    selection `offset` meta, so they take the text pointer, mouse selection,
    and copy; the others (indents, padding, borders) do not. A `None` flag
    marks blank fill, which takes only the background of its styles.
    """

    parts: tuple[tuple[Content, bool | None], ...] = ()
    fill: tuple[str, ...] = ()
    """Styles, outermost first, whose background fills the row to its edge."""

    joiner: str | None = "\n"
    """Text placed before this row in copied text, or `None` to skip the row."""

    margin: bool = False
    """Whether the row is blank spacing between blocks."""

    @property
    def cell_length(self) -> int:
        """Cells the parts occupy."""
        return sum(part.cell_length for part, _ in self.parts)

    @property
    def text(self) -> str:
        """Plain selectable text, as indexed by selection offsets."""
        return "".join(part.plain for part, selectable in self.parts if selectable)


@dataclass(slots=True)
class Block:
    """Rows of a laid-out block plus the margins Textual's CSS gives it."""

    rows: list[Row]
    top: int = 0
    bottom: int = 0


def blank(*, margin: bool = True) -> Row:
    """Return an empty row: block spacing, or a decorative blank row."""
    return Row(margin=margin, joiner="\n" if margin else None)


def wrap_prose(content: Content, width: int) -> list[Row]:
    """Wrap inline content and record how copied text rejoins the lines.

    A soft wrap rejoins with the space it replaced, a fold inside a long word
    with nothing, and a hard break with a newline.

    Returns:
        One selectable row per wrapped line.
    """
    plain = content.plain
    position = 0
    rows: list[Row] = []
    for index, line in enumerate(content.wrap(width)):
        joiner = "\n"
        if index:
            start = position
            while position < len(plain) and plain[position] in " \n":
                position += 1
            skipped = plain[start:position]
            joiner = "\n" if "\n" in skipped else (" " if skipped else "")
        if plain.startswith(line.plain, position):
            position += len(line.plain)
        rows.append(Row(((line, True),), joiner=joiner))
    return rows


def stylize(rows: list[Row], style: str) -> list[Row]:
    """Layer `style` beneath every part and the fill of `rows`.

    Returns:
        The restyled rows.
    """
    if not style:
        return rows
    return [
        Row(
            tuple((part.stylize_before(style), sel) for part, sel in row.parts),
            (style, *row.fill),
            row.joiner,
            row.margin,
        )
        for row in rows
    ]


def pad(row: Row, width: int) -> tuple[tuple[Content, bool | None], ...]:
    """Return the row's parts with its fill drawn out to `width` cells."""
    gap = width - row.cell_length
    if gap <= 0:
        return row.parts
    fill = Content(" " * gap, [Span(0, gap, style) for style in row.fill])
    return (*row.parts, (fill, None))


def stack(blocks: list[Block]) -> list[Row]:
    """Stack blocks vertically, collapsing adjacent margins like Textual.

    The first block's top margin and the last block's bottom margin stay, as
    they do inside an auto-height Textual container.

    Returns:
        The combined rows.
    """
    rows: list[Row] = []
    previous: int | None = None
    for block in blocks:
        gap = block.top if previous is None else max(previous, block.top)
        rows.extend(blank() for _ in range(gap))
        rows.extend(block.rows)
        previous = block.bottom
    rows.extend(blank() for _ in range(previous or 0))
    return rows
