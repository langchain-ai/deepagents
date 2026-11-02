"""The `FinalMarkdown` widget and the visual that draws its rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.segment import Segment
from textual.strip import Strip
from textual.style import Style
from textual.visual import Visual
from textual.widget import Widget

from deepagents_code.tui.widgets._links import open_checked_url_async
from deepagents_code.tui.widgets.final_markdown._layout import Theme, layout
from deepagents_code.tui.widgets.final_markdown._parse import parse
from deepagents_code.tui.widgets.final_markdown._rows import pad

if TYPE_CHECKING:
    from textual.content import Content
    from textual.css.styles import RulesMap
    from textual.selection import Selection
    from textual.visual import RenderOptions

    from deepagents_code.tui.widgets.final_markdown._parse import Node
    from deepagents_code.tui.widgets.final_markdown._rows import Row

_LAYOUT_CACHE_SIZE = 4
"""Widths kept in the layout cache."""


class _MarkdownVisual(Visual):
    """A `Visual` that lays out the reply for whatever width it is given.

    Textual measures the height at a new width before it resizes the widget,
    so measuring here, rather than in `render`, never reports the height of a
    stale width.
    """

    def __init__(self, widget: FinalMarkdown) -> None:
        self._widget = widget

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:  # noqa: ARG002, PLR6301  # Visual protocol
        """Fill the container, like a `Markdown` block.

        Returns:
            The container width.
        """
        return container_width

    def get_height(self, rules: RulesMap, width: int) -> int:  # noqa: ARG002  # Visual protocol
        """Measure the reply at `width`.

        Returns:
            The number of rows the reply takes at `width`.
        """
        return len(self._widget.rows_for(width))

    def render_strips(
        self, width: int, height: int | None, style: Style, options: RenderOptions
    ) -> list[Strip]:
        """Render the rows laid out at `width`.

        Returns:
            One strip per row.
        """
        rows = self._widget.rows_for(width)
        if height is not None:
            rows = rows[:height]
        return [_strip(row, y, width, style, options) for y, row in enumerate(rows)]


def _strip(row: Row, y: int, width: int, style: Style, options: RenderOptions) -> Strip:
    """Render one row, tagging selectable text with its selection offset.

    Returns:
        The row's strip, filled to `width`.
    """
    selection, selection_style = options.selection, options.selection_style
    span = selection.get_span(y) if selection is not None else None
    segments: list[Segment] = []
    x = 0
    for part, selectable in pad(row, width):
        if selectable is None:
            segments.append(_fill_segment(part, style, options))
            continue
        content = part
        if selectable and span is not None and selection_style is not None:
            start, end = span
            end = x + len(part.plain) if end == -1 else end
            content = part.stylize(selection_style, max(start - x, 0), max(end - x, 0))
        for text, text_style in content.render(
            style, end="", parse_style=options.get_style
        ):
            combined = style + text_style
            if selectable:
                segments.append(Segment(text, combined.rich_style_with_offset(x, y)))
                x += len(text)
            else:
                segments.append(Segment(text, combined.rich_style))
    return Strip(segments).crop(0, width)


def _fill_segment(fill: Content, style: Style, options: RenderOptions) -> Segment:
    """Render blank fill with only the backgrounds of its styles.

    A block's background spans the full width in Textual, but its text style
    (an underlined heading, say) applies only to its text.

    Returns:
        A segment of spaces.
    """
    for span in fill.spans:
        background = options.get_style(span.style).background
        if background is not None:
            style += Style(background=background)
    return Segment(fill.plain, style.rich_style)


class FinalMarkdown(Widget):
    """A finished markdown reply drawn as one widget.

    Looks like Textual's `Markdown` (validated against Textual 8.2.8) but
    holds no child widgets. Long fence lines fold instead of scrolling.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "em",
        "strong",
        "s",
        "code_inline",
        "markdown--h1",
        "markdown--h2",
        "markdown--h3",
        "markdown--h4",
        "markdown--h5",
        "markdown--h6",
        "markdown--hr",
        "markdown--bullet",
        "markdown--table-header",
        "markdown--table-keyline",
    }
    """Inline classes match `MarkdownBlock`; block classes mirror block CSS."""

    DEFAULT_CSS = """
    FinalMarkdown {
        width: 1fr;
        height: auto;
        color: $foreground;
        &:dark > .code_inline {
            background: $warning 10%;
            color: $text-warning 95%;
        }
        &:light > .code_inline {
            background: $error 5%;
            color: $text-error 95%;
        }
        &:ansi > .code_inline { background: ansi_default; }
        & > .em { text-style: italic; }
        & > .strong { text-style: bold; }
        & > .s { text-style: strike; }
        & > .markdown--h1 {
            color: $markdown-h1-color;
            background: $markdown-h1-background;
            text-style: $markdown-h1-text-style;
        }
        & > .markdown--h2 {
            color: $markdown-h2-color;
            background: $markdown-h2-background;
            text-style: $markdown-h2-text-style;
        }
        & > .markdown--h3 {
            color: $markdown-h3-color;
            background: $markdown-h3-background;
            text-style: $markdown-h3-text-style;
        }
        & > .markdown--h4 {
            color: $markdown-h4-color;
            background: $markdown-h4-background;
            text-style: $markdown-h4-text-style;
        }
        & > .markdown--h5 {
            color: $markdown-h5-color;
            background: $markdown-h5-background;
            text-style: $markdown-h5-text-style;
        }
        & > .markdown--h6 {
            color: $markdown-h6-color;
            background: $markdown-h6-background;
            text-style: $markdown-h6-text-style;
        }
        & > .markdown--hr { color: $secondary; }
        & > .markdown--bullet { color: $text-primary; }
        &:light > .markdown--bullet { color: $text-secondary; }
        & > .markdown--table-header { color: $primary; }
        & > .markdown--table-keyline { color: $foreground 20%; }
    }
    """

    def __init__(
        self,
        source: str = "",
        *,
        id: str | None = None,  # noqa: A002  # Textual widget convention
        classes: str | None = None,
    ) -> None:
        """Create the widget.

        Args:
            source: Markdown source of the finished reply.
            id: The widget id.
            classes: Space-separated CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._source = source
        self._blocks_cache: list[Node] | None = None
        self._layouts: dict[int, list[Row]] = {}
        self._md_theme: Theme | None = None

    @property
    def source(self) -> str:
        """The markdown source."""
        return self._source

    def update(self, source: str) -> None:
        """Replace the markdown source and re-layout.

        Args:
            source: New markdown source.
        """
        if source == self._source:
            return
        self._source = source
        self._blocks_cache = None
        self._layouts.clear()
        self.refresh(layout=True)

    def render(self) -> Visual:
        """Return a visual that lays out rows at the width it renders at.

        Returns:
            The reply's visual.
        """
        return _MarkdownVisual(self)

    def rows_for(self, width: int) -> list[Row]:
        """Return the reply's rows at `width`, cached per width.

        Args:
            width: Width in cells.

        Returns:
            The laid-out rows.
        """
        rows = self._layouts.get(width)
        if rows is None:
            if self._blocks_cache is None:
                self._blocks_cache = parse(self._source)
            if len(self._layouts) >= _LAYOUT_CACHE_SIZE:
                self._layouts.clear()
            rows = layout(self._blocks_cache, max(width, 1), self._current_theme())
            self._layouts[width] = rows
        return rows

    def _current_theme(self) -> Theme:
        if self._md_theme is None:
            # Read borders at render time: ASCII mode patches this table.
            from textual._border import BORDER_CHARS  # noqa: PLC2701

            self._md_theme = Theme(
                ansi=self.app.native_ansi_color,
                dark=self.app.current_theme.dark,
                quote_bar=BORDER_CHARS["outer"][1][0],
                rule=BORDER_CHARS["solid"][2][1],
            )
        return self._md_theme

    def notify_style_update(self) -> None:
        """Re-layout when a theme change alters fence highlighting or spacing."""
        super().notify_style_update()
        if self._md_theme is None or not self.is_attached:
            return
        app = self.app
        current = (app.native_ansi_color, app.current_theme.dark)
        if (self._md_theme.ansi, self._md_theme.dark) != current:
            self._md_theme = None
            self._layouts.clear()
            self.refresh(layout=True)

    @property
    def selectable_text(self) -> str:
        """Text that selection offsets index into, one line per row."""
        return "\n".join(row.text for row in self.rows_for(self.size.width))

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Copy the selected text as the reply reads, not as it wraps.

        Soft-wrapped lines rejoin with a space, folded code lines with
        nothing. Indents, padding, and repeated blank lines drop out.

        Args:
            selection: The selected range.

        Returns:
            The copied text and the separator after it.
        """
        pieces: list[str] = []
        blank = True
        for y, row in enumerate(self.rows_for(self.size.width)):
            span = selection.get_span(y)
            if span is None or row.joiner is None:
                continue
            if row.margin:
                if not blank:
                    pieces.append("\n")
                blank = True
                continue
            start, end = span
            text = row.text[start : None if end == -1 else end].rstrip()
            pieces.append((row.joiner if pieces else "") + text)
            blank = False
        return "".join(pieces).rstrip("\n"), "\n"

    async def action_link(self, href: str) -> None:
        """Open a clicked link with the shared URL safety check.

        Args:
            href: The original link target, with URL encoding preserved.
        """
        await open_checked_url_async(href, app=self.app, notify_on_success=True)
