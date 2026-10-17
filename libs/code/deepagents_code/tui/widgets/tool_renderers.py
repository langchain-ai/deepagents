"""Tool renderers for approval widgets - registry pattern."""

from __future__ import annotations

import difflib
import logging
from typing import TYPE_CHECKING, Any

from deepagents_code.diff_utils import split_diff_lines
from deepagents_code.file_ops import (
    build_approval_preview,
    format_display_path,
    is_sensitive_file_path,
)
from deepagents_code.tui.widgets.tool_widgets import (
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
    format_display_content,
)

if TYPE_CHECKING:
    from deepagents_code.tui.widgets.tool_widgets import ToolApprovalWidget

logger = logging.getLogger(__name__)


class ToolRenderer:
    """Strategy for building a tool's HITL approval widget.

    Each renderer maps a tool name to a `(widget_class, data)` pair that
    controls what the user sees in the approval box. Tools not registered
    in `_RENDERER_REGISTRY` fall through to the default, which dumps all
    args as `key: value` lines via `GenericApprovalWidget`.
    """

    @staticmethod
    def get_approval_widget(
        tool_args: dict[str, Any],
        assistant_id: str | None = None,  # noqa: ARG004
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """Get the approval widget class and data for this tool.

        Args:
            tool_args: The tool arguments from action_request.
            assistant_id: Optional assistant identifier for resolving virtual paths.

        Returns:
            Tuple of (widget_class, data_dict)
        """
        return GenericApprovalWidget, tool_args


class WriteFileRenderer(ToolRenderer):
    """Renderer for write_file tool - shows full file content."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],
        assistant_id: str | None = None,  # noqa: ARG004
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        # Extract file extension for syntax highlighting
        file_path = tool_args.get("file_path", "")
        content = format_display_content(tool_args.get("content", ""))

        # Get file extension
        file_extension = "text"
        if "." in file_path:
            file_extension = file_path.rsplit(".", 1)[-1]

        data = {
            "file_path": file_path,
            "content": content,
            "file_extension": file_extension,
        }
        return WriteFileApprovalWidget, data


class TaskRenderer(ToolRenderer):
    """Renderer for task tool — interrupt description provides full context."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],  # noqa: ARG004  # Unused; interrupt description already formats task args
        assistant_id: str | None = None,  # noqa: ARG004
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        return GenericApprovalWidget, {}


class DeleteFileRenderer(ToolRenderer):
    """Renderer for delete tool - shows removed file content when available."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],
        assistant_id: str | None = None,
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        path = str(tool_args.get("file_path") or tool_args.get("path") or "")
        preview = build_approval_preview(
            "delete", {"file_path": path}, assistant_id=assistant_id
        )
        if preview is None:
            # `build_approval_preview` always returns a preview for "delete";
            # this guards its `ApprovalPreview | None` contract defensively.
            return GenericApprovalWidget, tool_args
        if preview.diff:
            return EditFileApprovalWidget, {
                "file_path": format_display_path(path),
                # `split_diff_lines`, not `splitlines()`: the diff was joined
                # with `"\n"`, and splitting on every boundary `splitlines()`
                # recognizes would cut a deleted line into an unmarked tail
                # fragment shown as neutral metadata.
                "diff_lines": split_diff_lines(preview.diff),
                "old_string": "",
                "new_string": "",
                # The preview body is clipped at 100 lines; these counts are
                # not. Without them the prompt for deleting a 5,000-line file
                # reads "-96".
                "stats": preview.stats,
            }
        data: dict[str, Any] = {"file_path": format_display_path(path)}
        details = [
            detail for detail in preview.details if not detail.startswith("File:")
        ]
        if details:
            data["details"] = "\n".join(details)
        if preview.error:
            data["error"] = preview.error
        return GenericApprovalWidget, data


class EditFileRenderer(ToolRenderer):
    """Renderer for edit_file tool - shows unified diff."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],
        assistant_id: str | None = None,
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        file_path = tool_args.get("file_path", "")
        old_arg = tool_args.get("old_string", "")
        new_arg = tool_args.get("new_string", "")

        # Non-string args (e.g. a model passing a dict) have no meaningful file
        # replacement to preview — fall through to the fragment diff, whose
        # `format_display_content` coercion is what keeps them displayable.
        if isinstance(old_arg, str) and isinstance(new_arg, str):
            # The preview reads the file back at approval time, so a diff here
            # describes the current file state, not the state the model read.
            # That is the desired behavior on this surface: when the two have
            # drifted, `preview.diff` is `None` and the fallback fragment diff
            # below shows the requested swap rather than numbers that no longer
            # describe anything that will happen.
            preview = build_approval_preview(
                "edit_file",
                {
                    "file_path": file_path,
                    "old_string": old_arg,
                    "new_string": new_arg,
                    "replace_all": bool(tool_args.get("replace_all")),
                },
                assistant_id=assistant_id,
            )
            if (
                preview is not None
                and preview.diff is not None
                # Requiring the sources keeps the type checker's guarantee:
                # a future producer setting `diff` without them must fail
                # here, not silently skip highlighting (empty code exits
                # `_highlighted_rows` before its drift warning).
                and preview.before is not None
                and preview.after is not None
            ):
                return EditFileApprovalWidget, {
                    "file_path": format_display_path(file_path),
                    "diff_lines": split_diff_lines(preview.diff),
                    # Highlight from the full before/after sources, not the
                    # fragments: the renderer locates rows by hunk line number,
                    # which this diff numbers in the file, so a fragment would
                    # leave changed rows deeper in the file plain and match
                    # low-line edits against the wrong fragment line.
                    "old_string": preview.before,
                    "new_string": preview.after,
                    "stats": preview.stats,
                    "show_numbers": True,
                }
            if preview is not None and preview.error:
                # The edit is known not to apply — e.g. `old_string` no longer
                # matches, or matches 40 times under `replace_all=False`.
                # Rendering the fragment diff here would show a confident swap
                # that the tool then rejects after approval.
                return GenericApprovalWidget, {
                    "file_path": format_display_path(file_path),
                    "error": preview.error,
                }
            if preview is not None and not is_sensitive_file_path(file_path):
                # A diffless, errorless preview for a file the user will see a
                # fragment diff for: oversized (skipped on the message pump)
                # or an empty/no-op edit. The fragment fallback is the right
                # render, but the preview's silence must not hide which.
                logger.debug(
                    "edit_file preview produced no diff for %s; "
                    "falling back to the fragment diff",
                    file_path,
                )

        old_string = format_display_content(old_arg)
        new_string = format_display_content(new_arg)

        # Fallback: generate a unified diff from the replacement fragments
        # alone, when the file cannot be read at approval time (sandbox-backed
        # session, unreadable path) or the preview was skipped. Its line
        # numbers are fragment-relative, so the widget hides the gutter — see
        # `EditFileApprovalWidget.compose`.
        diff_lines = EditFileRenderer._generate_diff(old_string, new_string)

        data = {
            "file_path": format_display_path(file_path),
            "diff_lines": diff_lines,
            "old_string": old_string,
            "new_string": new_string,
        }
        return EditFileApprovalWidget, data

    @staticmethod
    def _generate_diff(old_string: str, new_string: str) -> list[str]:
        """Generate unified diff lines from old and new strings.

        Returns:
            List of diff lines without the file headers.
        """
        if not old_string and not new_string:
            return []

        # `splitlines()`, matching `compute_unified_diff` and the source split in
        # `_highlight_source_prefix`. Splitting on `"\n"` alone leaves `\r`,
        # U+2028 and the rest inside a diff line, so the highlighter — which
        # splits on all of them — lines up against a different set of lines and
        # reports every row as drifted.
        old_lines = old_string.splitlines() if old_string else []
        new_lines = new_string.splitlines() if new_string else []

        # Generate unified diff
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=3,  # Context lines
        )

        # Skip the first two header lines (--- and +++)
        diff_list = list(diff)
        return diff_list[2:] if len(diff_list) > 2 else diff_list  # noqa: PLR2004  # Column count threshold


_RENDERER_REGISTRY: dict[str, type[ToolRenderer]] = {
    "task": TaskRenderer,
    "write_file": WriteFileRenderer,
    "edit_file": EditFileRenderer,
    "delete": DeleteFileRenderer,
}
"""Registry mapping tool names to renderers

Note: bash/shell/execute use minimal approval (no renderer) — see
ApprovalMenu._MINIMAL_TOOLS
"""


def get_renderer(tool_name: str) -> ToolRenderer:
    """Get the renderer for a tool by name.

    Args:
        tool_name: The name of the tool

    Returns:
        The appropriate ToolRenderer instance
    """
    renderer_class = _RENDERER_REGISTRY.get(tool_name, ToolRenderer)
    return renderer_class()
