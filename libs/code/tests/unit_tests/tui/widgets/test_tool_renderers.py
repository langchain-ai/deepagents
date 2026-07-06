from pathlib import Path

from deepagents_code.tui.widgets.tool_renderers import get_renderer
from deepagents_code.tui.widgets.tool_widgets import (
    EditFileApprovalWidget,
    GenericApprovalWidget,
)


def test_delete_renderer_shows_removed_file_diff(tmp_path: Path) -> None:
    target = tmp_path / "old.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    widget_class, data = get_renderer("delete").get_approval_widget(
        {"file_path": str(target)}
    )

    assert widget_class is EditFileApprovalWidget
    assert data["file_path"] == "old.txt"
    assert "-alpha" in data["diff_lines"]
    assert "-beta" in data["diff_lines"]


def test_delete_renderer_flags_directories_without_diff(tmp_path: Path) -> None:
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "child.txt").write_text("data\n", encoding="utf-8")

    widget_class, data = get_renderer("delete").get_approval_widget(
        {"file_path": str(target)}
    )

    assert widget_class is GenericApprovalWidget
    assert data["file_path"] == "subdir"
    assert "Contents: directory or unreadable file" in data["details"]


def test_delete_renderer_surfaces_unresolvable_path_error() -> None:
    """An empty path yields a resolution error shown in the approval widget."""
    widget_class, data = get_renderer("delete").get_approval_widget({"file_path": ""})

    assert widget_class is GenericApprovalWidget
    assert data["error"] == "Unable to resolve file path."
