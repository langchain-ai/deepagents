import textwrap
from pathlib import Path

from langchain_core.messages import ToolMessage

from deepagents_cli.file_ops import FileOpTracker, build_approval_preview


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
