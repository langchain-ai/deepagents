"""Tool renderers for approval widgets - registry pattern."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from deepagents_cli.widgets.tool_widgets import (
    BashApprovalWidget,
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
)

if TYPE_CHECKING:
    from deepagents_cli.widgets.tool_widgets import ToolApprovalWidget


class ToolRenderer:
    """Base renderer for tool approval widgets."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """Get the approval widget class and data for this tool.

        Args:
            tool_args: The tool arguments from action_request

        Returns:
            Tuple of (widget_class, data_dict)
        """
        return GenericApprovalWidget, tool_args


class WriteFileRenderer(ToolRenderer):
    """Renderer for write_file tool - shows full file content."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        # Extract file extension for syntax highlighting
        file_path = tool_args.get("file_path", "")
        content = tool_args.get("content", "")

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


class EditFileRenderer(ToolRenderer):
    """Renderer for edit_file tool - shows unified diff."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        file_path = tool_args.get("file_path", "")
        old_string = tool_args.get("old_string", "")
        new_string = tool_args.get("new_string", "")

        # Generate unified diff
        diff_lines = self._generate_diff(old_string, new_string)

        data = {
            "file_path": file_path,
            "diff_lines": diff_lines,
            "old_string": old_string,
            "new_string": new_string,
        }
        return EditFileApprovalWidget, data

    def _generate_diff(self, old_string: str, new_string: str) -> list[str]:
        """Generate unified diff lines from old and new strings."""
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
        return diff_list[2:] if len(diff_list) > 2 else diff_list


class BashRenderer(ToolRenderer):
    """Renderer for bash/shell tool - shows command."""

    def get_approval_widget(
        self, tool_args: dict[str, Any]
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        data = {
            "command": tool_args.get("command", ""),
            "description": tool_args.get("description", ""),
        }
        return BashApprovalWidget, data


# Registry mapping tool names to renderers
_RENDERER_REGISTRY: dict[str, type[ToolRenderer]] = {
    "write_file": WriteFileRenderer,
    "edit_file": EditFileRenderer,
    "bash": BashRenderer,
    "shell": BashRenderer,
}


def get_renderer(tool_name: str) -> ToolRenderer:
    """Get the renderer for a tool by name.

    Args:
        tool_name: The name of the tool

    Returns:
        The appropriate ToolRenderer instance
    """
    renderer_class = _RENDERER_REGISTRY.get(tool_name, ToolRenderer)
    return renderer_class()
