import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import mock

import pytest
from langchain_core.messages import ToolMessage

from deepagents_code.diff_utils import DiffStats

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


from deepagents_code.file_ops import (
    FileOperationRecord,
    FileOpTracker,
    build_approval_preview,
    display_caveat,
    is_sensitive_file_path,
    record_display_caveat,
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


def test_is_sensitive_file_path_fails_closed_on_unparseable_path() -> None:
    """A path that cannot be parsed is treated as sensitive, not rendered.

    The wrong runtime type is the point of the test: it drives the defensive
    branch that keeps a malformed `file_path` from crashing `compose()` and
    from leaking as a non-sensitive file.
    """
    assert is_sensitive_file_path(cast("str", 123)) is True


def test_unreadable_before_content_is_flagged(tmp_path: Path) -> None:
    """A pre-image we could not read must not masquerade as an empty file.

    Otherwise the diff renders the whole file as additions (or, for an
    unchanged file, as "no changes") with no signal that it is unreliable.
    """
    path = tmp_path / "locked.txt"
    path.write_text("alpha\nbeta\n")
    tracker = FileOpTracker(assistant_id=None)

    with mock.patch(
        "deepagents_code.file_ops._read_with_reason",
        return_value=(None, "Permission denied"),
    ):
        tracker.start_operation("edit_file", {"file_path": str(path)}, "locked-1")

    record = tracker.active["locked-1"]
    assert record.diff_outcome == "untrusted_before"
    assert record.before_content == ""


def test_a_real_binary_pre_image_is_flagged_without_mocking_the_read(
    tmp_path: Path,
) -> None:
    """The other flag tests all stub the read, so none pins what raises.

    `read_text(encoding="utf-8")` is the call that makes a binary file a lost
    pre-image. Switching it to `errors="replace"` would silently disable the
    whole flow — every mocked test would still pass, because they never exercise
    the decode.
    """
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01")
    tracker = FileOpTracker(assistant_id=None)

    tracker.start_operation("edit_file", {"file_path": str(path)}, "binary-1")

    record = tracker.active["binary-1"]
    assert record.diff_outcome == "untrusted_before"
    assert record.before_content == ""


def test_missing_before_content_is_not_flagged_as_unreadable(tmp_path: Path) -> None:
    """Creating a new file has no pre-image; that is normal, not a failure."""
    path = tmp_path / "brand-new.txt"
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation("write_file", {"file_path": str(path)}, "new-1")

    record = tracker.active["new-1"]
    assert record.diff_outcome != "untrusted_before"


def test_absent_local_pre_image_is_flagged_for_an_edit(tmp_path: Path) -> None:
    """An edit whose pre-image is simply gone still lost the pre-image.

    Gating the flag on `exists()` meant a path that diverged from the backend's,
    a broken symlink, or a file replaced mid-operation produced
    `before_content == ""` with no caveat — which renders a three-line edit as a
    confident whole-file insertion, and lets the tool row be hidden behind it.
    """
    path = tmp_path / "vanished.txt"
    tracker = FileOpTracker(assistant_id=None)

    tracker.start_operation("edit_file", {"file_path": str(path)}, "gone-1")

    record = tracker.active["gone-1"]
    assert record.diff_outcome == "untrusted_before"
    assert record.before_content == ""


def test_absent_local_pre_image_is_flagged_for_a_delete(tmp_path: Path) -> None:
    """Same rule for a delete: no pre-image means nothing to show as removed."""
    path = tmp_path / "vanished.txt"
    tracker = FileOpTracker(assistant_id=None)

    tracker.start_operation("delete", {"file_path": str(path)}, "gone-2")

    assert tracker.active["gone-2"].diff_outcome == "untrusted_before"


def test_an_unresolvable_delete_does_not_claim_a_verified_zero() -> None:
    """No backend and no physical path means nothing about the file is known.

    `start_operation` had no `else` for this case, so the outcome stayed at its
    `shown` default. `write_file`/`edit_file` recover downstream, where the
    post-read hits the same missing path — but `delete` synthesizes an empty
    post-image instead of reading one back, so empty-against-empty yields no
    diff and the record finished as a confident `+0 -0` about a file whose
    contents were never seen.
    """
    tracker = FileOpTracker(assistant_id=None)

    with mock.patch(
        "deepagents_code.file_ops.resolve_physical_path", return_value=None
    ):
        tracker.start_operation("delete", {"file_path": "/nul\x00/x"}, "lost-1")

    assert tracker.active["lost-1"].diff_outcome == "untrusted_before"

    record = tracker.complete_with_message(
        SimpleNamespace(content="Deleted", tool_call_id="lost-1", status="success")
    )

    assert record is not None
    assert record.diff_outcome == "untrusted_before"
    assert record.diff_stats is None
    assert display_caveat(record.diff_outcome, record.tool_name) != ""


def test_a_malformed_backend_response_reports_a_shape_not_an_attribute_error(
    tmp_path: Path,
) -> None:
    """A contract violation must not present as a broken workspace.

    Catching `AttributeError` alongside the read errors made a local bug — a
    renamed field, a `None` where a response was expected — read as "could not
    read file", degrading every operation in the session with nothing to say
    which it was.
    """
    backend = mock.MagicMock()
    backend.download_files.return_value = [SimpleNamespace(content="str", error=None)]
    tracker = FileOpTracker(assistant_id=None, backend=backend)

    tracker.start_operation("edit_file", {"file_path": str(tmp_path / "a.py")}, "bad-1")

    record = tracker.active["bad-1"]
    assert record.diff_outcome == "untrusted_before"
    assert record.before_content == ""


def test_failed_read_back_still_reports_what_the_request_knew(
    tmp_path: Path,
) -> None:
    """A write that landed must not report zero lines to session accounting.

    The early return for an unreadable post-image skipped every metric, so real
    work came out as a plausible-looking zero. `write_file` carries its full
    result in its args, so that much is knowable without the read-back — and the
    reason for the failure has to reach the caller.
    """
    path = tmp_path / "written.txt"
    content = "alpha\nbeta\ngamma\n"
    tracker = FileOpTracker(assistant_id=None)
    tracker.start_operation(
        "write_file", {"file_path": str(path), "content": content}, "w-1"
    )

    with mock.patch(
        "deepagents_code.file_ops._read_with_reason",
        return_value=(None, "Permission denied"),
    ):
        record = tracker.complete_with_message(
            SimpleNamespace(content="Wrote file", tool_call_id="w-1", status="success")
        )

    assert record is not None
    assert record.diff_outcome == "unreadable_after"
    assert record.after_read_error == "Permission denied"
    assert record.metrics.lines_written == 3
    assert record.metrics.bytes_written == len(content.encode("utf-8"))


def test_unreadable_existing_file_is_flagged_even_for_write_file(
    tmp_path: Path,
) -> None:
    """Absence is the create case for `write_file`; a failed read is not.

    The file is there and we could not read it, so whatever diff follows is
    against a pre-image we do not have.
    """
    path = tmp_path / "locked.txt"
    path.write_text("alpha\n")
    tracker = FileOpTracker(assistant_id=None)

    with mock.patch(
        "deepagents_code.file_ops._read_with_reason",
        return_value=(None, "Permission denied"),
    ):
        tracker.start_operation("write_file", {"file_path": str(path)}, "locked-2")

    assert tracker.active["locked-2"].diff_outcome == "untrusted_before"


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
    assert record.diff_outcome != "untrusted_before"
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
    assert record.diff_outcome == "untrusted_before"
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
    assert record.diff_outcome == "untrusted_before"
    assert record.before_content == ""


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


def test_delete_preview_says_so_when_it_cannot_read_the_file(tmp_path: Path) -> None:
    """A preview that cannot reach the file must not render as an empty one.

    The pre-image is read from the local filesystem, so a session whose files
    live on a backend (sandbox, store, LangSmith) resolves to a path that is
    not there. Without an explicit detail the prompt carries no diff, no
    counts, and no error — asking the user to approve destroying a 5,000-line
    file on a screen identical to destroying an empty one.
    """
    absent = tmp_path / "not-here.py"

    preview = build_approval_preview("delete", {"file_path": str(absent)}, None)

    assert preview is not None
    assert preview.diff is None
    assert preview.stats is None
    assert any("could not be read" in detail for detail in preview.details), (
        f"the prompt showed nothing about what it will delete: {preview.details}"
    )


def test_write_preview_distinguishes_an_unreadable_file_from_a_new_one(
    tmp_path: Path,
) -> None:
    """`before or ""` collapsed "exists but unreadable" into the create case.

    Both the delete and edit branches handle an unreadable pre-image
    explicitly; only `write_file` degraded silently, dropping the overwrite
    warning and rendering the payload as a pure insertion against nothing.
    """
    target = tmp_path / "locked.py"
    target.write_text("secret = 1\n", encoding="utf-8")

    with mock.patch("deepagents_code.file_ops._safe_read", return_value=None):
        preview = build_approval_preview(
            "write_file", {"file_path": str(target), "content": "new = 2\n"}, None
        )

    assert preview is not None
    assert any("could not be read" in detail for detail in preview.details), (
        f"an unreadable existing file was described as a create: {preview.details}"
    )


def _tool_message(content: str, tool_call_id: str) -> object:
    """Build the minimal shape `complete_with_message` reads.

    Returns:
        An object exposing `content`, `status`, and `tool_call_id`.
    """
    return SimpleNamespace(content=content, status="success", tool_call_id=tool_call_id)


class TestUntrustedBeforeThroughCompletion:
    """The lost-pre-image gate, exercised past `start_operation`.

    Every other test for this outcome stops at `tracker.active` and never calls
    `complete_with_message`, so the guard that keeps fictional counts out of
    `diff_stats` *and* out of session accounting has no coverage at all.
    """

    @staticmethod
    def _complete(tmp_path: Path) -> FileOperationRecord:
        target = tmp_path / "a.py"
        target.write_text("value = 1\n" * 400, encoding="utf-8")
        tracker = FileOpTracker(assistant_id=None)

        with mock.patch(
            "deepagents_code.file_ops._read_with_reason",
            return_value=(None, "Permission denied"),
        ):
            tracker.start_operation("edit_file", {"file_path": str(target)}, "t-1")
        record = tracker.complete_with_message(_tool_message("Updated file", "t-1"))
        assert record is not None
        return record

    def test_counts_are_left_unknown(self, tmp_path: Path) -> None:
        """`None` is the only way this says unknown; `DiffStats(0, 0)` is a zero."""
        record = self._complete(tmp_path)

        assert record.diff_outcome == "untrusted_before"
        assert record.diff_stats is None

    def test_fictional_counts_never_reach_session_accounting(
        self, tmp_path: Path
    ) -> None:
        """A diff against a stand-in empty file is a whole-file insertion.

        Booking those lines into `metrics` reports 400 added lines for a
        one-line edit, in a place where nothing marks the number unreliable.
        """
        record = self._complete(tmp_path)

        assert record.metrics.lines_added == 0
        assert record.metrics.lines_removed == 0


class TestBackendReadBack:
    """The backend branch of `_populate_after_content`.

    Its pre-image counterparts each have a test; the post-image ones had none,
    so a read-back failure that left `after_read_error` unset would degrade the
    user-facing caveat to "the reason was not reported" — the exact tautology
    `_read_with_reason` exists to avoid.
    """

    @staticmethod
    def _complete(backend: mock.Mock) -> FileOperationRecord:
        tracker = FileOpTracker(
            assistant_id=None, backend=cast("BackendProtocol", backend)
        )
        tracker.start_operation("edit_file", {"file_path": "/x.txt"}, "b-1")
        record = tracker.complete_with_message(_tool_message("Updated file", "b-1"))
        assert record is not None
        return record

    @staticmethod
    def _backend(pre: list[object], post: list[object]) -> mock.Mock:
        backend = mock.Mock()
        backend.download_files.side_effect = [pre, post]
        return backend

    @staticmethod
    def _found(content: bytes) -> list[object]:
        from deepagents.backends.protocol import FileDownloadResponse

        return [FileDownloadResponse(path="/x.txt", content=content, error=None)]

    def test_an_error_response_carries_its_reason(self) -> None:
        from deepagents.backends.protocol import FileDownloadResponse

        record = self._complete(
            self._backend(
                self._found(b"value = 1\n"),
                [
                    FileDownloadResponse(
                        path="/x.txt", content=None, error="permission_denied"
                    )
                ],
            )
        )

        assert record.diff_outcome == "unreadable_after"
        assert record.after_read_error == "permission_denied"

    def test_a_backend_raising_outside_oserror_does_not_abort_the_turn(self) -> None:
        """Real backends raise well outside `OSError`.

        The store backend base64-decodes (`binascii.Error`, a `ValueError`) and
        the LangSmith backend lets transport errors through. Catching only
        `OSError`/`AttributeError` let a transient sandbox blip kill the turn
        and drop every remaining tool's hooks — the outcome the handlers exist
        to prevent.
        """
        backend = mock.Mock()
        backend.download_files.side_effect = [
            self._found(b"value = 1\n"),
            ValueError("Invalid base64-encoded string"),
        ]

        record = self._complete(backend)

        assert record.diff_outcome == "unreadable_after"
        assert record.after_read_error == "Invalid base64-encoded string"

    def test_a_raising_pre_image_read_is_a_lost_pre_image_not_a_crash(self) -> None:
        """Same guarantee on the pre-operation read, which runs on the turn loop."""
        backend = mock.Mock()
        backend.download_files.side_effect = RuntimeError("sandbox unreachable")
        tracker = FileOpTracker(
            assistant_id=None, backend=cast("BackendProtocol", backend)
        )

        tracker.start_operation("edit_file", {"file_path": "/x.txt"}, "raise-1")

        record = tracker.active["raise-1"]
        assert record.diff_outcome == "untrusted_before"
        assert record.before_content == ""


class TestOutcomeInvariants:
    """`_finalize` is the one funnel where outcome/payload pairings can be checked.

    Every "outcome implies payload" rule lives in `DiffOutcome`'s docstring and
    is written from three separate branches, so nothing but this enforces them.
    """

    @staticmethod
    def _finalized(**overrides: object) -> FileOperationRecord:
        """Push a hand-built record through the tracker's completion funnel.

        Returns:
            The record after invariant enforcement.
        """
        record = FileOperationRecord(
            tool_name="edit_file",
            display_path="a.py",
            physical_path=None,
            tool_call_id="inv-1",
            status="success",
            tool_succeeded=True,
            **overrides,  # ty: ignore
        )
        FileOpTracker(assistant_id=None)._finalize(record)
        return record

    def test_counts_are_dropped_under_an_untrusted_pre_image(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`diff_stats` here would be counted against a stand-in empty file.

        The body is already suppressed for this outcome, but `diff_stats` is
        read directly by the adapter and the message store — so a stale pair
        still puts a fabricated `+200` on screen.
        """
        with caplog.at_level(logging.ERROR):
            record = self._finalized(
                diff_outcome="untrusted_before",
                diff_stats=DiffStats(additions=200, deletions=0),
            )

        assert record.diff_stats is None
        assert "untrusted pre-image" in caplog.text

    def test_a_mismatched_read_error_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`after_read_error` is the payload for `unreadable_after` and only it."""
        with caplog.at_level(logging.WARNING):
            self._finalized(diff_outcome="shown", after_read_error="permission_denied")

        assert "after_read_error" in caplog.text


class TestDisplayCaveat:
    """The caveat is the only account of a change the transcript cannot show."""

    def test_a_failed_operation_never_claims_success(self) -> None:
        """An untrusted pre-image does not override the tool's error result."""
        record = FileOperationRecord(
            tool_name="edit_file",
            display_path="missing.py",
            physical_path=None,
            tool_call_id="failed-1",
            status="error",
            tool_succeeded=False,
            diff_outcome="untrusted_before",
        )

        assert record_display_caveat(record) == ""

    def test_an_unreadable_read_back_names_the_reason(self) -> None:
        """Without the reason the sentence restates the problem."""
        caveat = display_caveat("unreadable_after", "edit_file", "Permission denied")

        assert "Permission denied" in caveat

    def test_a_lost_pre_image_says_the_change_cannot_be_shown(self) -> None:
        """Pin the wording contract directly.

        Otherwise it is only implied by adapter tests matching this prose.
        """
        caveat = display_caveat("untrusted_before", "delete")

        assert "`delete`" in caveat
        assert "prior contents could not be read" in caveat
