"""Color-coded context-window visualization for `/context`."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from deepagents_code import theme
from deepagents_code._session_stats import format_token_count
from deepagents_code.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult


class _ContextUsage(Static):
    def __init__(
        self,
        *,
        context_tokens: int | None,
        conversation_tokens: int | None,
        context_limit: int | None,
        model_spec: str | None,
        approximate: bool,
    ) -> None:
        super().__init__()
        self._total = None if context_tokens is None else max(0, context_tokens)
        self._conversation = (
            None if conversation_tokens is None else max(0, conversation_tokens)
        )
        self._limit = context_limit if context_limit and context_limit > 0 else None
        self._model = model_spec or "Unknown model"
        self._approximate = approximate

    def render(self) -> Content:
        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        usage = self._total if self._total is not None else self._conversation or 0
        maximum = format_token_count(self._limit) if self._limit else "unavailable"
        prefix = "~" if self._approximate or self._total is None else ""
        right = f"{prefix}{format_token_count(usage)} / {maximum}"
        if self._total is not None and self._limit:
            right += f"  {self._total / self._limit * 100:.1f}%"
        left = Content.assemble(
            ("Context", f"bold {colors.primary}"),
            (f" {glyphs.bullet} ", colors.muted),
            self._model,
        )
        gap = self.content_size.width - left.cell_length - len(right)
        header = Content.assemble(left, " " * max(1, gap), (right, colors.muted))

        categories: list[tuple[str, int, str]] = []
        if self._total is None:
            if self._conversation:
                categories.append(
                    ("Conversation estimate", self._conversation, colors.primary)
                )
        elif self._total:
            if self._conversation is None:
                categories.append(("Used context", self._total, colors.secondary))
            else:
                conversation = min(self._conversation, self._total)
                if overhead := self._total - conversation:
                    categories.append(
                        ("System prompt + tools", overhead, colors.warning)
                    )
                if conversation:
                    categories.append(("Conversation", conversation, colors.primary))
        if self._total is not None and self._limit:
            categories.append(
                ("Free space", max(0, self._limit - self._total), colors.muted)
            )

        scale = max(self._limit or 0, sum(tokens for _, tokens, _ in categories), 1)
        width = max(self.content_size.width, 1)
        used = 0
        segments: list[Content] = []
        for _label, tokens, color in categories:
            end = round(min(scale, used + tokens) / scale * width)
            start = round(min(scale, used) / scale * width)
            segments.append(
                Content.styled(glyphs.box_horizontal_heavy * (end - start), color)
            )
            used += tokens
        bar = Content.assemble(*segments)

        rows: list[Content] = []
        marker = glyphs.box_horizontal_heavy * 2
        for label, tokens, color in categories:
            percent = tokens / scale * 100
            value = f"{format_token_count(tokens)}  {glyphs.bullet}  {percent:.1f}%"
            item = Content.assemble((marker, color), " ", label)
            rows.append(
                Content.assemble(
                    item,
                    " " * max(1, width - item.cell_length - len(value)),
                    (value, colors.muted),
                )
            )
        if not rows:
            rows.append(Content.styled("No context usage reported yet.", colors.muted))
        elif self._total is None:
            rows.append(Content.styled("Total usage unavailable.", colors.muted))
        return Content("\n").join((header, bar, *rows))


class ContextUsageScreen(ModalScreen[None]):
    """Modal visualization of the current model context window."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False)
    ]
    CSS = """
    ContextUsageScreen { align: center middle; }
    ContextUsageScreen > VerticalScroll {
        width: 94%; max-width: 120; height: auto; max-height: 90%;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    ContextUsageScreen _ContextUsage { height: auto; }
    ContextUsageScreen .context-usage-help {
        height: 1; color: $text-muted; margin-top: 2;
    }
    """

    def __init__(
        self,
        *,
        context_tokens: int | None,
        conversation_tokens: int | None,
        context_limit: int | None,
        model_spec: str | None,
        approximate: bool,
    ) -> None:
        """Initialize the modal from the latest usage measurements."""
        super().__init__()
        self._usage = _ContextUsage(
            context_tokens=context_tokens,
            conversation_tokens=conversation_tokens,
            context_limit=context_limit,
            model_spec=model_spec,
            approximate=approximate,
        )

    def compose(self) -> ComposeResult:
        """Compose the context visualization and close hint.

        Yields:
            Widgets that make up the modal.
        """
        with VerticalScroll():
            yield self._usage
            yield Static("Esc to close", classes="context-usage-help")

    def on_mount(self) -> None:
        """Use an ASCII border when the terminal cannot render Unicode."""
        if is_ascii_mode():
            self.query_one(VerticalScroll).styles.border = (
                "ascii",
                theme.get_theme_colors(self).primary,
            )

    def action_close(self) -> None:
        """Dismiss the context visualization."""
        self.dismiss(None)
