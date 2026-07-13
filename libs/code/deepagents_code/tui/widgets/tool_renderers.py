"""Tool renderers for approval widgets - registry pattern."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from deepagents_code.file_ops import build_approval_preview, format_display_path
from deepagents_code.tui.widgets.tool_widgets import (
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
    format_display_content,
)

if TYPE_CHECKING:
    from deepagents_code.tui.widgets.tool_widgets import ToolApprovalWidget


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
                "diff_lines": preview.diff.splitlines(),
                "old_string": "",
                "new_string": "",
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
        assistant_id: str | None = None,  # noqa: ARG004
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        file_path = tool_args.get("file_path", "")
        old_string = format_display_content(tool_args.get("old_string", ""))
        new_string = format_display_content(tool_args.get("new_string", ""))

        # Generate unified diff
        diff_lines = EditFileRenderer._generate_diff(old_string, new_string)

        data = {
            "file_path": file_path,
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

        old_lines = old_string.split("\n") if old_string else []
        new_lines = new_string.split("\n") if new_string else []

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
