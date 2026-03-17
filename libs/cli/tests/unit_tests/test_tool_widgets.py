"""Unit tests for tool approval widgets - truncation and expand behavior."""

from __future__ import annotations

import pytest

from deepagents_cli.widgets.tool_widgets import (
    _PREVIEW_LINES,
    EditFileApprovalWidget,
    WriteFileApprovalWidget,
)


def _make_content(n_lines: int) -> str:
    return "\n".join(f"line {i}" for i in range(n_lines))


class TestWriteFileApprovalWidgetIsExpandable:
    """Tests for `WriteFileApprovalWidget.is_expandable`."""

    def test_short_content_not_expandable(self) -> None:
        """Content at or below the preview limit is not expandable."""
        content = _make_content(_PREVIEW_LINES)
        widget = WriteFileApprovalWidget(
            {"file_path": "a.py", "content": content, "file_extension": "py"}
        )
        assert widget.is_expandable is False

    def test_long_content_is_expandable(self) -> None:
        """Content exceeding the preview limit is expandable."""
        content = _make_content(_PREVIEW_LINES + 1)
        widget = WriteFileApprovalWidget(
            {"file_path": "a.py", "content": content, "file_extension": "py"}
        )
        assert widget.is_expandable is True

    def test_empty_content_not_expandable(self) -> None:
        """Empty content is never expandable."""
        widget = WriteFileApprovalWidget(
            {"file_path": "a.py", "content": "", "file_extension": "py"}
        )
        assert widget.is_expandable is False


class TestWriteFileApprovalWidgetToggleExpand:
    """Tests for `WriteFileApprovalWidget.toggle_expand`."""

    def test_toggle_changes_expanded_state(self) -> None:
        """Toggling flips `_expanded` for expandable content."""
        content = _make_content(_PREVIEW_LINES + 5)
        widget = WriteFileApprovalWidget(
            {"file_path": "a.py", "content": content, "file_extension": "py"}
        )
        assert widget._expanded is False
        widget.toggle_expand()
        assert widget._expanded is True
        widget.toggle_expand()
        assert widget._expanded is False

    def test_toggle_no_op_for_short_content(self) -> None:
        """Toggling does nothing for content within the preview limit."""
        content = _make_content(_PREVIEW_LINES)
        widget = WriteFileApprovalWidget(
            {"file_path": "a.py", "content": content, "file_extension": "py"}
        )
        widget.toggle_expand()
        assert widget._expanded is False


class TestWriteFileApprovalWidgetBuildMarkdown:
    """Tests for `WriteFileApprovalWidget._build_markdown`."""

    def test_preview_shows_only_first_lines(self) -> None:
        """Preview mode includes only up to `_PREVIEW_LINES` lines."""
        lines = [f"line {i}" for i in range(_PREVIEW_LINES + 10)]
        result = WriteFileApprovalWidget._build_markdown(lines, "py", expanded=False)
        for i in range(_PREVIEW_LINES):
            assert f"line {i}" in result
        assert f"line {_PREVIEW_LINES}" not in result

    def test_expanded_shows_all_lines(self) -> None:
        """Expanded mode includes all lines."""
        lines = [f"line {i}" for i in range(_PREVIEW_LINES + 10)]
        result = WriteFileApprovalWidget._build_markdown(lines, "py", expanded=True)
        for i in range(_PREVIEW_LINES + 10):
            assert f"line {i}" in result

    def test_file_extension_used_in_code_fence(self) -> None:
        """File extension appears in the Markdown code fence."""
        result = WriteFileApprovalWidget._build_markdown(
            ["x"], "typescript", expanded=False
        )
        assert "```typescript" in result


class TestEditFileApprovalWidgetIsExpandable:
    """Tests for `EditFileApprovalWidget.is_expandable`."""

    def test_short_diff_not_expandable(self) -> None:
        """Diff at or below the preview limit is not expandable."""
        diff_lines = [f"+line {i}" for i in range(_PREVIEW_LINES)]
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": diff_lines,
                "old_string": "",
                "new_string": "",
            }
        )
        assert widget.is_expandable is False

    def test_long_diff_is_expandable(self) -> None:
        """Diff exceeding the preview limit is expandable."""
        diff_lines = [f"+line {i}" for i in range(_PREVIEW_LINES + 1)]
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": diff_lines,
                "old_string": "",
                "new_string": "",
            }
        )
        assert widget.is_expandable is True

    def test_empty_diff_not_expandable(self) -> None:
        """Empty diff is not expandable."""
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": [],
                "old_string": "",
                "new_string": "",
            }
        )
        assert widget.is_expandable is False

    def test_header_lines_excluded_from_count(self) -> None:
        """Diff header lines (@@, ---, +++) do not count toward the limit."""
        diff_lines = (
            ["--- a", "+++ b"]
            + [f"@@ -{i},1 +{i},1 @@" for i in range(5)]
            + [f"+line {i}" for i in range(_PREVIEW_LINES)]
        )
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": diff_lines,
                "old_string": "",
                "new_string": "",
            }
        )
        assert widget.is_expandable is False

    def test_string_based_diff_counts_both_sides(self) -> None:
        """When no diff_lines, old+new string line counts are summed."""
        old = _make_content(_PREVIEW_LINES)
        new = _make_content(1)
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": [],
                "old_string": old,
                "new_string": new,
            }
        )
        assert widget.is_expandable is True


class TestEditFileApprovalWidgetToggleExpand:
    """Tests for `EditFileApprovalWidget.toggle_expand`."""

    def test_toggle_changes_expanded_state(self) -> None:
        """Toggling flips `_expanded` for expandable diffs."""
        diff_lines = [f"+line {i}" for i in range(_PREVIEW_LINES + 5)]
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": diff_lines,
                "old_string": "",
                "new_string": "",
            }
        )
        assert widget._expanded is False
        widget.toggle_expand()
        assert widget._expanded is True
        widget.toggle_expand()
        assert widget._expanded is False

    def test_toggle_no_op_for_short_diff(self) -> None:
        """Toggling does nothing for diffs within the preview limit."""
        diff_lines = [f"+line {i}" for i in range(_PREVIEW_LINES)]
        widget = EditFileApprovalWidget(
            {
                "file_path": "a.py",
                "diff_lines": diff_lines,
                "old_string": "",
                "new_string": "",
            }
        )
        widget.toggle_expand()
        assert widget._expanded is False


class TestApprovalMenuExpandableContent:
    """Tests for `ApprovalMenu._check_expandable_content`."""

    def test_write_file_is_expandable_content(self) -> None:
        """write_file tool is recognized as expandable content."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        menu = ApprovalMenu(
            {"name": "write_file", "args": {"file_path": "a.py", "content": "x"}}
        )
        assert menu._check_expandable_content() is True

    def test_edit_file_is_expandable_content(self) -> None:
        """edit_file tool is recognized as expandable content."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        menu = ApprovalMenu({"name": "edit_file", "args": {"file_path": "a.py"}})
        assert menu._check_expandable_content() is True

    def test_shell_is_not_expandable_content(self) -> None:
        """Shell tools are not recognized as expandable file content."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        menu = ApprovalMenu({"name": "shell", "args": {"command": "echo hi"}})
        assert menu._check_expandable_content() is False

    def test_multiple_requests_not_expandable_content(self) -> None:
        """Batch requests are never treated as expandable content."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        menu = ApprovalMenu(
            [
                {"name": "write_file", "args": {}},
                {"name": "write_file", "args": {}},
            ]
        )
        assert menu._check_expandable_content() is False
