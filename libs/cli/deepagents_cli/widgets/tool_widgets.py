"""Tool-specific approval widgets for HITL display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult

# Constants for display limits
_MAX_VALUE_LEN = 200
_PREVIEW_LINES = 10
_MAX_DIFF_LINES = 50
_MAX_PREVIEW_LINES = 20


class ToolApprovalWidget(Vertical):
    """Base class for tool approval widgets."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the tool approval widget with data."""
        super().__init__(classes="tool-approval-widget")
        self.data = data

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """Default compose - override in subclasses.

        Yields:
            Static widget with placeholder message.
        """
        yield Static("Tool details not available", classes="approval-description")

    @property
    def is_expandable(self) -> bool:
        """Return whether this widget has truncated content that can be expanded.

        Returns:
            `True` if the widget supports expand/collapse, `False` otherwise.
        """
        return False

    def toggle_expand(self) -> None:
        """Toggle between truncated and full content display.

        No-ops by default. Override in subclasses that support expansion.
        """


class GenericApprovalWidget(ToolApprovalWidget):
    """Generic approval widget for unknown tools."""

    def compose(self) -> ComposeResult:
        """Compose the generic tool display.

        Yields:
            Static widgets displaying each key-value pair from tool data.
        """
        for key, value in self.data.items():
            if value is None:
                continue
            value_str = str(value)
            if len(value_str) > _MAX_VALUE_LEN:
                hidden = len(value_str) - _MAX_VALUE_LEN
                value_str = value_str[:_MAX_VALUE_LEN] + f"... ({hidden} more chars)"
            yield Static(
                f"{key}: {value_str}", markup=False, classes="approval-description"
            )


class WriteFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for write_file - shows file content with syntax highlighting.

    Content longer than `_PREVIEW_LINES` lines is truncated by default and can
    be expanded via `toggle_expand`.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the write file approval widget.

        Args:
            data: Tool data containing `file_path`, `content`, and `file_extension`.
        """
        super().__init__(data)
        self._expanded = False
        self._content_widget: Markdown | None = None
        self._hint_widget: Static | None = None

    @property
    def is_expandable(self) -> bool:
        """Return whether the file content exceeds the preview line limit.

        Returns:
            `True` if content has more lines than `_PREVIEW_LINES`.
        """
        content = self.data.get("content", "")
        return len(content.split("\n")) > _PREVIEW_LINES

    def toggle_expand(self) -> None:
        """Toggle between truncated and full content display."""
        if not self.is_expandable:
            return
        self._expanded = not self._expanded
        self._update_content()

    def compose(self) -> ComposeResult:
        """Compose the file content display with syntax highlighting.

        Yields:
            Widgets displaying file path header and syntax-highlighted content.
        """
        file_path = self.data.get("file_path", "")
        content = self.data.get("content", "")
        file_extension = self.data.get("file_extension", "text")

        yield Static(f"File: {file_path}", markup=False, classes="approval-file-path")
        yield Static("")

        lines = content.split("\n")
        self._content_widget = Markdown(
            self._build_markdown(lines, file_extension, expanded=False)
        )
        yield self._content_widget

        if len(lines) > _PREVIEW_LINES:
            remaining = len(lines) - _PREVIEW_LINES
            self._hint_widget = Static(
                Content.styled(
                    f"... {remaining} more lines — press 'e' to expand", "dim"
                ),
                classes="approval-expand-hint",
            )
            yield self._hint_widget

    @staticmethod
    def _build_markdown(
        lines: list[str], file_extension: str, *, expanded: bool
    ) -> str:
        """Build the Markdown code block string for the given lines.

        Args:
            lines: Lines of file content.
            file_extension: Language hint for syntax highlighting.
            expanded: Whether to show all lines or only the preview.

        Returns:
            Markdown-formatted code block string.
        """
        shown = lines if expanded else lines[:_PREVIEW_LINES]
        return f"```{file_extension}\n{chr(10).join(shown)}\n```"

    def _update_content(self) -> None:
        """Refresh the content and hint widgets after expand/collapse."""
        if self._content_widget is None:
            return
        content = self.data.get("content", "")
        file_extension = self.data.get("file_extension", "text")
        lines = content.split("\n")
        self._content_widget.update(
            self._build_markdown(lines, file_extension, expanded=self._expanded)
        )
        if self._hint_widget is not None:
            if self._expanded:
                self._hint_widget.update(
                    Content.styled("press 'e' to collapse", "dim")
                )
            else:
                remaining = len(lines) - _PREVIEW_LINES
                self._hint_widget.update(
                    Content.styled(
                        f"... {remaining} more lines — press 'e' to expand", "dim"
                    )
                )


class EditFileApprovalWidget(ToolApprovalWidget):
    """Approval widget for edit_file - shows clean diff with colors.

    Diffs longer than `_PREVIEW_LINES` lines are truncated by default and can
    be expanded via `toggle_expand`.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the edit file approval widget.

        Args:
            data: Tool data containing `file_path`, `diff_lines`, `old_string`,
                and `new_string`.
        """
        super().__init__(data)
        self._expanded = False
        self._diff_container: Vertical | None = None
        self._hint_widget: Static | None = None

    @property
    def is_expandable(self) -> bool:
        """Return whether the diff exceeds the preview line limit.

        Returns:
            `True` if the visible diff line count exceeds `_PREVIEW_LINES`.
        """
        return self._count_visible_lines() > _PREVIEW_LINES

    def _count_visible_lines(self) -> int:
        """Count the number of visible (non-header) diff lines.

        Returns:
            Count of renderable diff lines.
        """
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")
        if diff_lines:
            return sum(
                1
                for line in diff_lines
                if not line.startswith(("@@", "---", "+++"))
            )
        lines = 0
        if old_string:
            lines += len(old_string.split("\n"))
        if new_string:
            lines += len(new_string.split("\n"))
        return lines

    def toggle_expand(self) -> None:
        """Toggle between truncated and full diff display."""
        if not self.is_expandable:
            return
        self._expanded = not self._expanded
        self.call_after_refresh(self._remount_diff)

    async def _remount_diff(self) -> None:
        """Remount diff content after expand/collapse toggle."""
        if self._diff_container is None:
            return
        await self._diff_container.remove_children()
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")
        widgets = list(
            self._build_diff_widgets(diff_lines, old_string, new_string)
        )
        await self._diff_container.mount(*widgets)
        if self._hint_widget is not None:
            if self._expanded:
                self._hint_widget.update(
                    Content.styled("press 'e' to collapse", "dim")
                )
            else:
                hidden = self._count_visible_lines() - _PREVIEW_LINES
                self._hint_widget.update(
                    Content.styled(
                        f"... {hidden} more lines — press 'e' to expand", "dim"
                    )
                )

    def compose(self) -> ComposeResult:
        """Compose the diff display with colored additions and deletions.

        Yields:
            Widgets displaying file path, stats, and colored diff lines.
        """
        file_path = self.data.get("file_path", "")
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")

        additions, deletions = self._count_stats(diff_lines, old_string, new_string)
        stats_str = self._format_stats(additions, deletions)
        yield Static(
            Content.assemble(
                Content.from_markup(
                    "[bold cyan]File:[/bold cyan] $path  ", path=file_path
                ),
                stats_str,
            )
        )
        yield Static("")

        self._diff_container = Vertical()
        yield self._diff_container

        if self._count_visible_lines() > _PREVIEW_LINES:
            hidden = self._count_visible_lines() - _PREVIEW_LINES
            self._hint_widget = Static(
                Content.styled(
                    f"... {hidden} more lines — press 'e' to expand", "dim"
                ),
                classes="approval-expand-hint",
            )
            yield self._hint_widget

    async def on_mount(self) -> None:
        """Mount the initial diff content into the diff container."""
        if self._diff_container is None:
            return
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")
        widgets = list(self._build_diff_widgets(diff_lines, old_string, new_string))
        if widgets:
            await self._diff_container.mount(*widgets)

    def _build_diff_widgets(
        self, diff_lines: list[str], old_string: str, new_string: str
    ) -> ComposeResult:
        """Build the diff content widgets respecting the current expand state.

        Args:
            diff_lines: Unified diff lines.
            old_string: Original string being replaced.
            new_string: Replacement string.

        Yields:
            Static widgets for the diff content.
        """
        if not diff_lines and not old_string and not new_string:
            yield Static("No changes to display", classes="approval-description")
        elif diff_lines:
            yield from self._render_diff_lines_only(diff_lines)
        else:
            yield from self._render_strings_only(old_string, new_string)

    @staticmethod
    def _count_stats(
        diff_lines: list[str], old_string: str, new_string: str
    ) -> tuple[int, int]:
        """Count additions and deletions from diff data.

        Returns:
            Tuple of (additions count, deletions count).
        """
        if diff_lines:
            additions = sum(
                1
                for line in diff_lines
                if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1
                for line in diff_lines
                if line.startswith("-") and not line.startswith("---")
            )
        else:
            additions = new_string.count("\n") + 1 if new_string else 0
            deletions = old_string.count("\n") + 1 if old_string else 0
        return additions, deletions

    @staticmethod
    def _format_stats(additions: int, deletions: int) -> Content:
        """Format addition/deletion stats as styled Content.

        Returns:
            Styled Content showing additions and deletions.
        """
        parts: list[str | tuple[str, str] | Content] = []
        if additions:
            if parts:
                parts.append(" ")
            parts.append((f"+{additions}", "green"))
        if deletions:
            if parts:
                parts.append(" ")
            parts.append((f"-{deletions}", "red"))
        return Content.assemble(*parts) if parts else Content("")

    def _render_diff_lines_only(self, diff_lines: list[str]) -> ComposeResult:
        """Render unified diff lines without returning stats.

        Yields:
            Static widgets for each diff line with appropriate styling.
        """
        lines_shown = 0
        limit = len(diff_lines) if self._expanded else _PREVIEW_LINES

        for line in diff_lines:
            if lines_shown >= limit:
                break

            if line.startswith(("@@", "---", "+++")):
                continue

            widget = self._render_diff_line(line)
            if widget:
                yield widget
                lines_shown += 1

    def _render_strings_only(self, old_string: str, new_string: str) -> ComposeResult:
        """Render old/new strings without returning stats.

        Yields:
            Static widgets showing removed and added content with styling.
        """
        if old_string:
            yield Static(Content.styled("Removing:", "bold red"))
            yield from self._render_string_lines(
                old_string, is_addition=False
            )
            yield Static("")

        if new_string:
            yield Static(Content.styled("Adding:", "bold green"))
            yield from self._render_string_lines(
                new_string, is_addition=True
            )

    @staticmethod
    def _render_diff_line(line: str) -> Static | None:
        """Render a single diff line with appropriate styling.

        Returns:
            Static widget with styled diff line, or None for empty/skipped lines.
        """
        raw = line[1:] if len(line) > 1 else ""

        if line.startswith("-"):
            return Static(
                Content.from_markup(
                    "[on #4a2020][#ff8787]- $text[/#ff8787][/on #4a2020]", text=raw
                )
            )
        if line.startswith("+"):
            return Static(
                Content.from_markup(
                    "[on #1e4620][#8ce99a]+ $text[/#8ce99a][/on #1e4620]", text=raw
                )
            )
        if line.startswith(" "):
            return Static(Content.from_markup("[#aaaaaa]  $text[/#aaaaaa]", text=raw))
        if line.strip():
            return Static(line, markup=False)
        return None

    def _render_string_lines(self, text: str, *, is_addition: bool) -> ComposeResult:
        """Render lines from a string with appropriate styling.

        Yields:
            Static widgets for each line with addition or deletion styling.
        """
        lines = text.split("\n")
        limit = len(lines) if self._expanded else _MAX_PREVIEW_LINES
        style = "[on #1e4620][#8ce99a]+" if is_addition else "[on #4a2020][#ff8787]-"
        end_style = (
            "[/#8ce99a][/on #1e4620]" if is_addition else "[/#ff8787][/on #4a2020]"
        )

        for line in lines[:limit]:
            yield Static(Content.from_markup(f"{style} $text{end_style}", text=line))

        if not self._expanded and len(lines) > _MAX_PREVIEW_LINES:
            remaining = len(lines) - _MAX_PREVIEW_LINES
            yield Static(Content.styled(f"... ({remaining} more lines)", "dim"))
