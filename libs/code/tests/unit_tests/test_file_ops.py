import shutil
import textwrap
from pathlib import Path
from typing import cast

import pytest
from langchain_core.messages import ToolMessage

from deepagents_code.file_ops import (
    FileOpTracker,
    build_approval_preview,
    is_sensitive_file_path,
    redact_secret_file_content,
)


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


def test_redact_secret_file_content_hides_values_keeps_keys() -> None:
    """A .env value is redacted while the key name and structure survive."""
    content = "API_TOKEN=secretvalue\n# comment\nDEBUG=true\n"
    redacted = redact_secret_file_content(content)

    assert "API_TOKEN=" in redacted
    assert "secretvalue" not in redacted
    assert "<redacted:11 chars>" in redacted
    assert "# comment" in redacted


def test_redact_secret_file_content_handles_read_file_gutter_and_quotes() -> None:
    """Redaction survives the read_file line gutter and quoted values."""
    content = '     1\tSECRET_KEY="hunter2"\n     2\texport TOKEN=abc\n'
    redacted = redact_secret_file_content(content)

    assert "hunter2" not in redacted
    assert "abc" not in redacted
    assert "SECRET_KEY=" in redacted
    assert "export TOKEN=" in redacted


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
