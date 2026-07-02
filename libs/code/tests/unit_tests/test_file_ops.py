import shutil
import textwrap
from pathlib import Path

from langchain_core.messages import ToolMessage

from deepagents_code.file_ops import FileOpTracker, build_approval_preview


def test_tracker_records_read_lines(tmp_path: Path) -> None:
    tracker = FileOpTracker(assistant_id=None)
    path = tmp_path / "example.py"

    tracker.start_operation(
        "read_file",
        {"file_path": str(path), "offset": 0, "limit": 100},
        "read-1",
    )

    message = ToolMessage(
        content="    1\tline one\n    2\tline two\n",
        tool_call_id="read-1",
        name="read_file",
    )
    record = tracker.complete_with_message(message)

    assert record is not None
    assert record.metrics.lines_read == 2
    assert record.metrics.start_line == 1
    assert record.metrics.end_line == 2


def test_tracker_records_write_diff(tmp_path: Path) -> None:
    tracker = FileOpTracker(assistant_id=None)
    file_path = tmp_path / "created.txt"

    tracker.start_operation(
        "write_file",
        {"file_path": str(file_path)},
        "write-1",
    )

    file_path.write_text("hello world\nsecond line\n")

    message = ToolMessage(
        content=f"Updated file {file_path}",
        tool_call_id="write-1",
        name="write_file",
    )
    record = tracker.complete_with_message(message)

    assert record is not None
    assert record.metrics.lines_written == 2
    assert record.metrics.lines_added == 2
    assert record.diff is not None
    assert "+hello world" in record.diff


def test_tracker_records_edit_diff(tmp_path: Path) -> None:
    tracker = FileOpTracker(assistant_id=None)
    file_path = tmp_path / "functions.py"
    file_path.write_text(
        textwrap.dedent(
            """\
        def greet():
            return "hello"
        """
        )
    )

    tracker.start_operation(
        "edit_file",
        {"file_path": str(file_path)},
        "edit-1",
    )

    file_path.write_text(
        textwrap.dedent(
            """\
        def greet():
            return "hi"

        def wave():
            return "wave"
        """
        )
    )

    message = ToolMessage(
        content=f"Successfully replaced 1 instance(s) of the string in '{file_path}'",
        tool_call_id="edit-1",
        name="edit_file",
    )
    record = tracker.complete_with_message(message)

    assert record is not None
    assert record.metrics.lines_added >= 1
    assert record.metrics.lines_removed >= 1
    assert record.diff is not None
    assert '-    return "hello"' in record.diff
    assert '+    return "hi"' in record.diff


def test_tracker_records_delete_diff(tmp_path: Path) -> None:
    tracker = FileOpTracker(assistant_id=None)
    file_path = tmp_path / "old.txt"
    file_path.write_text("alpha\nbeta\n")

    tracker.start_operation("delete", {"file_path": str(file_path)}, "delete-1")
    file_path.unlink()

    message = ToolMessage(
        content=f"Deleted {file_path}", tool_call_id="delete-1", name="delete"
    )
    record = tracker.complete_with_message(message)

    assert record is not None
    assert record.status == "success"
    assert record.metrics.lines_removed == 2
    assert record.diff is not None
    assert "-alpha" in record.diff
    assert "-beta" in record.diff


def test_build_approval_preview_generates_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")

    preview = build_approval_preview(
        "edit_file",
        {
            "file_path": str(target),
            "old_string": "beta",
            "new_string": "gamma",
            "replace_all": False,
        },
        assistant_id=None,
    )

    assert preview is not None
    assert preview.diff is not None
    assert "+gamma" in preview.diff


def test_build_delete_approval_preview_shows_removed_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta\n")

    preview = build_approval_preview(
        "delete",
        {"file_path": str(target)},
        assistant_id=None,
    )

    assert preview is not None
    assert preview.title == "Delete notes.txt"
    assert "Action: Delete file or directory" in preview.details
    assert "Lines to delete: 2" in preview.details
    assert preview.diff is not None
    assert "-alpha" in preview.diff


def test_tracker_records_directory_delete(tmp_path: Path) -> None:
    """A recursive directory delete is tracked as a success without a diff."""
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "child.txt").write_text("data\n")

    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("delete", {"file_path": str(target)}, "delete-dir")
    # Directory has no readable text content, so no before/after to diff.
    shutil.rmtree(target)

    message = ToolMessage(
        content=f"Deleted {target}", tool_call_id="delete-dir", name="delete"
    )
    record = tracker.complete_with_message(message)

    assert record is not None
    assert record.status == "success"
    assert record.metrics.lines_removed == 0
    assert not record.diff


def test_build_delete_approval_preview_for_directory(tmp_path: Path) -> None:
    """The delete preview flags directories instead of rendering a diff."""
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "child.txt").write_text("data\n")

    preview = build_approval_preview(
        "delete",
        {"file_path": str(target)},
        assistant_id=None,
    )

    assert preview is not None
    assert preview.title == "Delete subdir"
    assert "Contents: directory or unreadable file" in preview.details
    assert preview.diff is None


def test_build_delete_approval_preview_unresolvable_path() -> None:
    """An empty path yields an explicit resolution error, not a blank preview."""
    preview = build_approval_preview("delete", {"file_path": ""}, assistant_id=None)

    assert preview is not None
    assert preview.error == "Unable to resolve file path."
