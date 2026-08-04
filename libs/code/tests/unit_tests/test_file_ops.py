import shutil
import textwrap
from pathlib import Path
from typing import cast
from unittest import mock

import pytest
from langchain_core.messages import ToolMessage

from deepagents_code.file_ops import (
    FileOpTracker,
    build_approval_preview,
    is_sensitive_file_path,
)


def test_file_not_found_matches_sdk() -> None:
    """`_constants.FILE_NOT_FOUND` must not drift from the SDK sentinel."""
    from deepagents.backends.protocol import FILE_NOT_FOUND as SDK_FILE_NOT_FOUND

    from deepagents_code._constants import FILE_NOT_FOUND

    assert FILE_NOT_FOUND == SDK_FILE_NOT_FOUND


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "/home/user/project/.env",
        "config/.ENV",
        "credentials",
        "~/.aws/credentials",
        "credentials.json",
        "TOKEN.JSON",
        "~/.deepagents/.state/auth.json",
        ".git-credentials",
        ".netrc",
        "_netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        ".htpasswd",
        "id_rsa",
        "id_ed25519",
        "server.pem",
        "private.KEY",
        "cert.pfx",
        "store.p12",
        "app.keystore",
        "release.jks",
    ],
)
def test_is_sensitive_file_path_matches_credentials(path: str) -> None:
    assert is_sensitive_file_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "",
        None,
        "main.py",
        "README.md",
        "src/app.ts",
        "environment.py",
        "keyboard.json",
        ".envision",
    ],
)
def test_is_sensitive_file_path_ignores_regular_files(path: str | None) -> None:
    assert is_sensitive_file_path(path) is False


def test_is_sensitive_file_path_fails_closed_on_unparseable_path() -> None:
    """A path that cannot be parsed is treated as sensitive, not rendered.

    The wrong runtime type is the point of the test: it drives the defensive
    branch that keeps a malformed `file_path` from crashing `compose()` and
    from leaking as a non-sensitive file.
    """
    assert is_sensitive_file_path(cast("str", 123)) is True


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


def test_diff_counts_are_computed_before_truncation(tmp_path: Path) -> None:
    """Large changes retain their true counts when the rendered diff is truncated."""
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"old {index}" for index in range(1000)))
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("edit_file", {"file_path": str(path)}, "large-edit")
    path.write_text("\n".join(f"new {index}" for index in range(1000)))
    record = tracker.complete_with_message(
        ToolMessage(content="Updated file", tool_call_id="large-edit", name="edit_file")
    )

    assert record is not None
    assert record.diff is not None
    assert record.diff.endswith("...")
    assert (record.metrics.lines_added, record.metrics.lines_removed) == (1000, 1000)


def test_unreadable_before_content_is_flagged(tmp_path: Path) -> None:
    """A pre-image we could not read must not masquerade as an empty file.

    Otherwise the diff renders the whole file as additions (or, for an
    unchanged file, as "no changes") with no signal that it is unreliable.
    """
    path = tmp_path / "locked.txt"
    path.write_text("alpha\nbeta\n")
    tracker = FileOpTracker(assistant_id=None)

    with mock.patch("deepagents_code.file_ops._safe_read", return_value=None):
        tracker.start_operation("edit_file", {"file_path": str(path)}, "locked-1")

    record = tracker.active["locked-1"]
    assert record.before_unreadable is True
    assert record.before_content == ""


def test_missing_before_content_is_not_flagged_as_unreadable(tmp_path: Path) -> None:
    """Creating a new file has no pre-image; that is normal, not a failure."""
    path = tmp_path / "brand-new.txt"
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("write_file", {"file_path": str(path)}, "new-1")

    record = tracker.active["new-1"]
    assert record.before_unreadable is False


def test_backend_file_not_found_is_not_flagged_as_unreadable() -> None:
    """Backends report a missing pre-image with the `FILE_NOT_FOUND` sentinel.

    Creating a file through a backend (state, store, sandbox) always answers
    the pre-edit download with `file_not_found`; that is the known empty
    pre-image of the create case, not a read failure, so the diff of the whole
    file as additions is trustworthy and must not be suppressed.
    """
    from deepagents.backends.protocol import FileDownloadResponse

    backend = mock.Mock()
    backend.download_files.return_value = [
        FileDownloadResponse(path="/new.txt", content=None, error="file_not_found")
    ]
    tracker = FileOpTracker(assistant_id=None, backend=backend)

    tracker.start_operation(
        "write_file", {"file_path": "/new.txt", "content": "hello"}, "new-2"
    )

    record = tracker.active["new-2"]
    assert record.before_unreadable is False
    assert record.before_content == ""


def test_backend_read_failure_is_flagged_as_unreadable() -> None:
    """Backend errors other than `FILE_NOT_FOUND` lose the pre-image.

    `permission_denied` means the file may exist and differ from the stand-in
    empty string, so any diff against it is suspect and must be flagged.
    """
    from deepagents.backends.protocol import FileDownloadResponse

    backend = mock.Mock()
    backend.download_files.return_value = [
        FileDownloadResponse(
            path="/locked.txt", content=None, error="permission_denied"
        )
    ]
    tracker = FileOpTracker(assistant_id=None, backend=backend)

    tracker.start_operation("edit_file", {"file_path": "/locked.txt"}, "locked-2")

    record = tracker.active["locked-2"]
    assert record.before_unreadable is True
    assert record.before_content == ""


def test_backend_response_without_content_or_error_is_flagged_as_unreadable() -> None:
    """Neither content nor an error breaks the backend contract both ways.

    `content=None` means failure and `error=None` means success, so this
    response asserts both at once. Treating it as an absent file would hand the
    diff a fabricated empty pre-image with nothing marking it untrustworthy —
    the whole file would render as a confident insertion.
    """
    from deepagents.backends.protocol import FileDownloadResponse

    backend = mock.Mock()
    backend.download_files.return_value = [FileDownloadResponse(path="/x.txt")]
    tracker = FileOpTracker(assistant_id=None, backend=backend)

    tracker.start_operation("edit_file", {"file_path": "/x.txt"}, "contract-1")

    record = tracker.active["contract-1"]
    assert record.before_unreadable is True
    assert record.before_content == ""


def test_empty_backend_response_list_is_flagged_as_unreadable() -> None:
    """No response at all is a lost pre-image, not an absent file."""
    backend = mock.Mock()
    backend.download_files.return_value = []
    tracker = FileOpTracker(assistant_id=None, backend=backend)

    tracker.start_operation("edit_file", {"file_path": "/y.txt"}, "contract-2")

    record = tracker.active["contract-2"]
    assert record.before_unreadable is True
    assert record.before_content == ""


def test_trailing_newline_only_edit_is_not_reported_as_unchanged(
    tmp_path: Path,
) -> None:
    """A change `splitlines()` erases is still a change.

    `compute_unified_diff` compares line lists, so adding a trailing newline
    produces no diff. Left unflagged, the edit row is superseded by a diff
    header reading "no changes" — the file changed, and the tool's own output
    saying so has been hidden.
    """
    path = tmp_path / "eof.txt"
    path.write_text("alpha\nbeta")
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("edit_file", {"file_path": str(path)}, "eof-1")
    path.write_text("alpha\nbeta\n")

    record = tracker.complete_with_message(
        ToolMessage(content="Updated file", tool_call_id="eof-1", name="edit_file")
    )

    assert record is not None
    assert record.diff is None
    assert record.change_invisible_to_line_diff is True


def test_genuine_noop_edit_is_not_flagged_as_an_invisible_change(
    tmp_path: Path,
) -> None:
    """An edit that truly changed nothing must stay eligible for "no changes"."""
    path = tmp_path / "same.txt"
    path.write_text("alpha\nbeta\n")
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("edit_file", {"file_path": str(path)}, "same-1")

    record = tracker.complete_with_message(
        ToolMessage(content="Updated file", tool_call_id="same-1", name="edit_file")
    )

    assert record is not None
    assert record.diff is None
    assert record.change_invisible_to_line_diff is False


def test_unreadable_after_content_sets_its_own_flag(tmp_path: Path) -> None:
    """Succeeded-but-undisplayable must be distinguishable from a tool error.

    Both set `status == "error"`; only this one means the operation itself
    landed, so only this one may tell the user it succeeded.
    """
    path = tmp_path / "vanishing.txt"
    path.write_text("alpha\n")
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("edit_file", {"file_path": str(path)}, "vanish-1")
    path.unlink()

    record = tracker.complete_with_message(
        ToolMessage(content="Updated file", tool_call_id="vanish-1", name="edit_file")
    )

    assert record is not None
    assert record.status == "error"
    assert record.after_unreadable is True


def test_tool_reported_error_does_not_set_after_unreadable(tmp_path: Path) -> None:
    """A genuine tool failure must not be reported as "succeeded, but…"."""
    path = tmp_path / "f.txt"
    path.write_text("alpha\n")
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("edit_file", {"file_path": str(path)}, "err-1")

    record = tracker.complete_with_message(
        ToolMessage(
            content="Error: string not found", tool_call_id="err-1", name="edit_file"
        )
    )

    assert record is not None
    assert record.status == "error"
    assert record.after_unreadable is False


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
