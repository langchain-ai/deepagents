"""Unit tests for input parsing utilities."""

import errno
import os
from pathlib import Path

import pytest

from deepagents_code.input import (
    ParsedPastedPathPayload,
    ProbeFailure,
    dropped_payload_paths,
    extract_leading_pasted_entry_path,
    extract_leading_pasted_file_path,
    normalize_pasted_path,
    parse_file_mentions,
    parse_pasted_any_entry_paths,
    parse_pasted_file_paths,
    parse_pasted_path_payload,
    parse_single_pasted_file_path,
    select_probe_failure_for,
    track_probe_failures,
)


def test_parse_file_mentions_with_chinese_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure `@file` parsing terminates at non-path characters such as CJK text."""
    file_path = tmp_path / "input.py"
    file_path.write_text("print('hello')")

    monkeypatch.chdir(tmp_path)
    text = f"你分析@{file_path.name}的代码就懂了"

    _, files = parse_file_mentions(text)

    assert files == [file_path.resolve()]


def test_parse_file_mentions_handles_multiple_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure multiple `@file` mentions are extracted from a single input."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("1")
    second.write_text("2")

    monkeypatch.chdir(tmp_path)
    text = f"读一下@{first.name}，然后看看@{second.name}。"

    _, files = parse_file_mentions(text)

    assert files == [first.resolve(), second.resolve()]


def test_parse_file_mentions_with_escaped_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure escaped spaces in paths are handled correctly."""
    spaced_dir = tmp_path / "my folder"
    spaced_dir.mkdir()
    file_path = spaced_dir / "test.py"
    file_path.write_text("content")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("@my\\ folder/test.py")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_warns_for_nonexistent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure non-existent files are excluded and warning is printed."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    _, files = parse_file_mentions("@nonexistent.py")

    assert files == []
    mock_console.print.assert_called_once()
    assert "nonexistent.py" in mock_console.print.call_args[0][0]


def test_parse_file_mentions_ignores_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure directories are not included in file list."""
    dir_path = tmp_path / "mydir"
    dir_path.mkdir()
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    _, files = parse_file_mentions("@mydir")

    assert files == []
    mock_console.print.assert_called_once()
    assert "mydir" in mock_console.print.call_args[0][0]


def test_parse_file_mentions_with_no_mentions() -> None:
    """Ensure text without mentions returns empty file list."""
    _, files = parse_file_mentions("just some text without mentions")
    assert files == []


def test_parse_file_mentions_handles_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure path traversal sequences are resolved to actual paths."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file_path = tmp_path / "test.txt"
    file_path.write_text("content")
    monkeypatch.chdir(subdir)

    _, files = parse_file_mentions("@../test.txt")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_with_absolute_path(tmp_path: Path) -> None:
    """Ensure absolute paths are resolved correctly without cwd changes."""
    file_path = tmp_path / "test.py"
    file_path.write_text("content")

    _, files = parse_file_mentions(f"@{file_path}")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_handles_multiple_in_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure multiple `@mentions` within a sentence are each parsed separately."""
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("1")
    second.write_text("2")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("compare @a.py and @b.py")

    assert files == [first.resolve(), second.resolve()]


def test_parse_file_mentions_adjacent_looks_like_email(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Adjacent `@mentions` without space look like emails and are skipped.

    `@a.py@b.py` - the second `@` is preceded by `y` which looks like
    an email username, so `@b.py` is skipped. This is expected behavior
    to avoid false positives on email addresses.
    """
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("1")
    second.write_text("2")
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    _, files = parse_file_mentions("@a.py@b.py")

    # Only first file is parsed; second looks like email and is skipped
    assert files == [first.resolve()]
    mock_console.print.assert_not_called()


def test_parse_file_mentions_handles_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure `OSError` during path resolution is handled gracefully."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")
    mocker.patch("pathlib.Path.resolve", side_effect=OSError("Permission denied"))

    _, files = parse_file_mentions("@somefile.py")

    assert files == []
    mock_console.print.assert_called_once()
    call_arg = mock_console.print.call_args[0][0]
    assert "somefile.py" in call_arg
    assert "Invalid path" in call_arg


def test_parse_file_mentions_skips_email_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure email addresses are not parsed as file mentions.

    Email addresses like `user@example.com` should be silently skipped
    because the `@` is preceded by email-like characters.
    """
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    _, files = parse_file_mentions("contact me at user@example.com")

    # Email addresses should be silently skipped (no warning, no files)
    assert files == []
    mock_console.print.assert_not_called()


def test_parse_file_mentions_skips_various_email_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure various email formats are all skipped."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    emails = [
        "test@domain.com",
        "user.name@company.org",
        "first+tag@example.io",
        "name_123@test.co",
        "a@b.c",
    ]

    for email in emails:
        _, files = parse_file_mentions(f"Email: {email}")
        assert files == [], f"Expected {email} to be skipped"

    mock_console.print.assert_not_called()


def test_parse_file_mentions_works_after_cjk_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure `@file` mentions work after CJK text (not email-like)."""
    file_path = tmp_path / "test.py"
    file_path.write_text("content")
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    # CJK character before @ is not email-like, so this should parse
    _, files = parse_file_mentions("查看@test.py")

    assert files == [file_path.resolve()]
    mock_console.print.assert_not_called()


def test_parse_file_mentions_handles_bad_tilde_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure `~nonexistentuser` paths produce a warning instead of crashing.

    `Path.expanduser()` raises `RuntimeError` when the username does not
    exist. This must be caught gracefully rather than propagating up.
    """
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_code.input.console")

    _, files = parse_file_mentions("@~nonexistentuser12345/file.py")

    assert files == []
    mock_console.print.assert_called_once()
    call_arg = mock_console.print.call_args[0][0]
    assert "nonexistentuser12345" in call_arg


def test_parse_pasted_file_paths_with_quoted_paths(tmp_path: Path) -> None:
    """Quoted dropped paths should resolve correctly."""
    img = tmp_path / "my image.png"
    img.write_bytes(b"img")

    result = parse_pasted_file_paths(f'"{img}"')

    assert result == [img.resolve()]


def test_parse_pasted_file_paths_with_file_url(tmp_path: Path) -> None:
    """`file://` dropped paths should be URL-decoded and resolved."""
    img = tmp_path / "space name.png"
    img.write_bytes(b"img")

    result = parse_pasted_file_paths(f"file://{str(img).replace(' ', '%20')}")

    assert result == [img.resolve()]


def test_parse_pasted_file_paths_with_multiple_lines(tmp_path: Path) -> None:
    """Multiple dropped paths separated by newlines should all resolve."""
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    result = parse_pasted_file_paths(f"{first}\n{second}")

    assert result == [first.resolve(), second.resolve()]


def test_parse_pasted_file_paths_returns_empty_for_text_payload() -> None:
    """Normal prose should not be interpreted as dropped file paths."""
    assert parse_pasted_file_paths("please inspect this image") == []


def test_parse_pasted_file_paths_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """Missing dropped files should fall back to regular text paste."""
    missing = tmp_path / "missing.png"
    assert parse_pasted_file_paths(str(missing)) == []


def test_parse_pasted_file_paths_returns_empty_for_empty_string() -> None:
    """Empty string should return an empty list."""
    assert parse_pasted_file_paths("") == []


def test_parse_pasted_file_paths_returns_empty_for_whitespace() -> None:
    """Whitespace-only payloads should return an empty list."""
    assert parse_pasted_file_paths("   \n\t  ") == []


def test_parse_pasted_file_paths_handles_angle_bracket_wrapped_path(
    tmp_path: Path,
) -> None:
    """Angle-bracket wrapped paths (e.g. from some terminals) should resolve."""
    img = tmp_path / "bracketed.png"
    img.write_bytes(b"img")

    result = parse_pasted_file_paths(f"<{img}>")

    assert result == [img.resolve()]


def test_parse_pasted_file_paths_still_rejects_a_directory(tmp_path: Path) -> None:
    """The narrow parser must not widen to folders alongside the entry parser.

    `parse_pasted_file_paths` feeds the attachment/image flows, so a folder
    reaching it would be read as something to attach.
    """
    folder = tmp_path / "assets"
    folder.mkdir()

    assert parse_pasted_file_paths(str(folder)) == []


def test_parse_pasted_file_paths_prefers_a_file_over_a_space_variant_dir(
    tmp_path: Path,
) -> None:
    """The Unicode-space fallback must keep its file-only preference."""
    folder = tmp_path / "my assets"
    folder.mkdir()

    # Non-breaking space in the payload, ASCII space on disk: the variant walk
    # finds the directory, and the file-only predicate must still reject it.
    # Written as an escape so the whitespace-normalizing hook cannot silently
    # turn this back into an ASCII space and retire the case it covers.
    payload = f"'{tmp_path}/my\u00a0assets'"
    assert parse_pasted_file_paths(payload) == []


def test_parse_pasted_any_entry_paths_resolves_dropped_folder(tmp_path: Path) -> None:
    """A dragged folder payload should resolve to the directory."""
    folder = tmp_path / "assets"
    folder.mkdir()

    assert parse_pasted_any_entry_paths(str(folder)) == [folder.resolve()]


def test_parse_pasted_any_entry_paths_resolves_quoted_folder(tmp_path: Path) -> None:
    """Quoted folder payloads should resolve like quoted file payloads."""
    folder = tmp_path / "my assets"
    folder.mkdir()

    assert parse_pasted_any_entry_paths(f"'{folder}'") == [folder.resolve()]


def test_parse_pasted_any_entry_paths_rejects_unquoted_folder_with_spaces(
    tmp_path: Path,
) -> None:
    """Shell tokenization splits unquoted spaces; the leading extractor covers it."""
    folder = tmp_path / "my assets" / "raw images"
    folder.mkdir(parents=True)

    assert parse_pasted_any_entry_paths(str(folder)) == []

    result = extract_leading_pasted_entry_path(str(folder))
    assert result is not None
    assert result[0] == folder.resolve()


def test_parse_pasted_any_entry_paths_resolves_file_url(tmp_path: Path) -> None:
    """`file://` folder payloads should be URL-decoded and resolved."""
    folder = tmp_path / "space name"
    folder.mkdir()

    payload = f"file://{str(folder).replace(' ', '%20')}"

    assert parse_pasted_any_entry_paths(payload) == [folder.resolve()]


def test_parse_pasted_any_entry_paths_resolves_multiple_folders(tmp_path: Path) -> None:
    """Dropping several folders at once should resolve every path."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    result = parse_pasted_any_entry_paths(f"{first}\n{second}")

    assert result == [first.resolve(), second.resolve()]


def test_parse_pasted_any_entry_paths_ignores_missing_folder(tmp_path: Path) -> None:
    """Missing directories should fall back to regular text paste."""
    assert parse_pasted_any_entry_paths(str(tmp_path / "missing")) == []


@pytest.mark.parametrize("payload", ["", "   \n\t  ", "/help", "please inspect this"])
def test_parse_pasted_any_entry_paths_ignores_non_path_payloads(payload: str) -> None:
    """Prose and slash commands must not be read as dropped entries."""
    assert parse_pasted_any_entry_paths(payload) == []


def test_parse_pasted_any_entry_paths_resolves_mixed_folder_then_file(
    tmp_path: Path,
) -> None:
    """A folder followed by a file resolves as a mixed drop."""
    folder = tmp_path / "assets"
    folder.mkdir()
    note = tmp_path / "note.txt"
    note.write_text("hi")

    result = parse_pasted_any_entry_paths(f"{folder} {note}")

    assert result == [folder.resolve(), note.resolve()]


def test_parse_pasted_any_entry_paths_resolves_mixed_file_then_folder(
    tmp_path: Path,
) -> None:
    """A file followed by a folder resolves as a mixed drop."""
    note = tmp_path / "note.txt"
    note.write_text("hi")
    folder = tmp_path / "assets"
    folder.mkdir()

    result = parse_pasted_any_entry_paths(f"{note} {folder}")

    assert result == [note.resolve(), folder.resolve()]


def test_parse_pasted_any_entry_paths_rejects_missing_token(tmp_path: Path) -> None:
    """Any unresolvable token rejects the whole payload as ordinary text."""
    folder = tmp_path / "assets"
    folder.mkdir()

    assert parse_pasted_any_entry_paths(f"{folder} {tmp_path / 'missing'}") == []


def test_extract_leading_pasted_entry_path_accepts_folder_then_prose(
    tmp_path: Path,
) -> None:
    """A leading folder followed by a question resolves like a leading file."""
    folder = tmp_path / "assets"
    folder.mkdir()

    result = extract_leading_pasted_entry_path(f"{folder} what is in here")

    assert result is not None
    path, token_end = result
    assert path == folder.resolve()
    assert token_end == len(str(folder))


def test_extract_leading_pasted_entry_path_accepts_file_then_prose(
    tmp_path: Path,
) -> None:
    """The file case keeps working through the shared extractor."""
    note = tmp_path / "note.txt"
    note.write_text("hi")

    result = extract_leading_pasted_entry_path(f"{note} explain this")

    assert result is not None
    assert result[0] == note.resolve()


def test_extract_leading_pasted_entry_path_handles_unquoted_spaces(
    tmp_path: Path,
) -> None:
    """A folder whose name contains spaces still splits from trailing prose."""
    folder = tmp_path / "my assets"
    folder.mkdir()

    result = extract_leading_pasted_entry_path(f"{folder} what is in here")

    assert result is not None
    assert result[0] == folder.resolve()


def test_extract_leading_pasted_entry_path_rejects_prose(tmp_path: Path) -> None:
    """Text with no resolvable leading path stays ordinary prose."""
    assert extract_leading_pasted_entry_path("please inspect this") is None
    assert extract_leading_pasted_entry_path(f"{tmp_path / 'missing'} hi") is None


def test_normalize_pasted_path_rejects_mixed_payload() -> None:
    """Single-path normalizer should reject path+prose mixed payloads."""
    assert normalize_pasted_path("'/tmp/a.png' what's this") is None


def test_normalize_pasted_path_accepts_windows_drive_payload() -> None:
    """Unquoted Windows drive path with spaces should parse as one path token."""
    payload = r"C:\Users\Alice\My Pictures\example image.png"
    result = normalize_pasted_path(payload)
    assert result == Path(payload)


def test_parse_single_pasted_file_path_resolves_unicode_space_variant(
    tmp_path: Path,
) -> None:
    """ASCII-space paste should resolve files with lookalike Unicode spaces."""
    unicode_name = "Screenshot 2026-02-26 at 2.02.42\u202fAM.png"
    img = tmp_path / unicode_name
    img.write_bytes(b"img")

    pasted_path = str(img).replace("\u202f", " ")
    pasted = f"'{pasted_path}'"
    resolved = parse_single_pasted_file_path(pasted)

    assert resolved == img.resolve()


def test_parse_single_pasted_file_path_unquoted_posix_path_with_spaces(
    tmp_path: Path,
) -> None:
    """Raw POSIX absolute paths with spaces should resolve as one file path."""
    img = tmp_path / "Screenshot 1.png"
    img.write_bytes(b"img")

    resolved = parse_single_pasted_file_path(str(img))

    assert resolved == img.resolve()


def test_parse_pasted_path_payload_single_path(tmp_path: Path) -> None:
    """Payload parser should resolve path-only payloads."""
    img = tmp_path / "one.png"
    img.write_bytes(b"img")

    parsed = parse_pasted_path_payload(str(img))

    assert parsed is not None
    assert parsed.paths == [img.resolve()]
    assert parsed.token_end is None


def test_parse_pasted_path_payload_leading_path_with_suffix(tmp_path: Path) -> None:
    """Payload parser should extract leading path when enabled."""
    img = tmp_path / "my image.png"
    img.write_bytes(b"img")
    payload = f"'{img}' what's in this image?"

    assert parse_pasted_path_payload(payload) is None

    parsed = parse_pasted_path_payload(payload, allow_leading_path=True)

    assert parsed is not None
    assert parsed.paths == [img.resolve()]
    assert parsed.token_end is not None
    assert payload[parsed.token_end :] == " what's in this image?"


def test_dropped_payload_paths_resolves_image(tmp_path: Path) -> None:
    """A dropped image path is resolved."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(str(img)) == [img.resolve()]


def test_dropped_payload_paths_resolves_video(tmp_path: Path) -> None:
    """A dropped video path is resolved."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"vid")

    assert dropped_payload_paths(str(clip)) == [clip.resolve()]


def test_dropped_payload_paths_resolves_non_media(tmp_path: Path) -> None:
    """Classification is left to the caller, so non-media paths resolve too."""
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")

    assert dropped_payload_paths(str(doc)) == [doc.resolve()]


def test_dropped_payload_paths_resolves_multiple(tmp_path: Path) -> None:
    """A multi-file drop returns every resolved path."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")

    assert dropped_payload_paths(f"{img} {doc}") == [img.resolve(), doc.resolve()]


def test_dropped_payload_paths_resolves_file_url(tmp_path: Path) -> None:
    """A `file://` drop payload resolves like a plain path."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(img.as_uri()) == [img.resolve()]


@pytest.mark.parametrize("wrap", ["'{}'", '"{}"', "<{}>"])
def test_dropped_payload_paths_resolves_quoted_payload(
    tmp_path: Path, wrap: str
) -> None:
    """Quoted and bracketed drops resolve, since terminals wrap paths that way.

    The shape guard strips leading `<`, `'`, and `"` for exactly this reason;
    without that strip every quoted drop would look like typed text. A quoted
    path is also the designed burst shape — see `PASTE_BURST_START_CHARS`.
    """
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(wrap.format(img)) == [img.resolve()]


@pytest.mark.parametrize("template", ["{}", "'{}'", '"{}"'])
def test_dropped_payload_paths_resolves_space_bearing_filename(
    tmp_path: Path, template: str
) -> None:
    """A filename containing spaces resolves escaped or quoted.

    This is the modal real-world drop: macOS screenshots are named
    `Screenshot ... at ....png`.
    """
    img = tmp_path / "my shot.png"
    img.write_bytes(b"img")
    raw = str(img)
    payload = template.format(raw if template != "{}" else raw.replace(" ", r"\ "))

    assert dropped_payload_paths(payload) == [img.resolve()]


def test_dropped_payload_paths_resolves_home_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `~/`-shaped drop resolves, matching the shape guard's `~/` arm."""
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths("~/shot.png") == [img.resolve()]


@pytest.mark.parametrize(
    "payload",
    [
        "/usr/local is where it lives",
        "/ is the root directory",
        "~/ is my home",
    ],
)
def test_dropped_payload_paths_ignores_prose_that_passes_shape_guard(
    payload: str,
) -> None:
    """Prose starting with a path-shaped token is still text, not a drop.

    The shape guard only inspects the leading token, so rejection here depends
    on the parser refusing directories and multi-token text. Free-text prompts
    rely on this: a swallowed answer is worse than an inserted path.
    """
    assert dropped_payload_paths(payload) == []


def test_dropped_payload_paths_accepts_windows_drive_shape(mocker) -> None:
    """Windows drive-letter drops pass the shape guard and get parsed."""
    resolved = Path(r"C:\Users\Alice\shot.png")
    mocker.patch(
        "deepagents_code.input.parse_pasted_path_payload",
        return_value=ParsedPastedPathPayload(paths=[resolved]),
    )

    assert dropped_payload_paths(r"C:\Users\Alice\shot.png") == [resolved]


def test_dropped_payload_paths_accepts_windows_unc_shape(mocker) -> None:
    """Windows UNC drops pass the shape guard and get parsed."""
    resolved = Path(r"\\server\share\shot.png")
    mocker.patch(
        "deepagents_code.input.parse_pasted_path_payload",
        return_value=ParsedPastedPathPayload(paths=[resolved]),
    )

    assert dropped_payload_paths(r"\\server\share\shot.png") == [resolved]


def test_track_probe_failures_records_unreadable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that raises is recorded, not silently folded into 'not a path'."""
    folder = tmp_path / "assets"
    folder.mkdir()

    def _deny(self: Path, *args: object, **kwargs: object) -> object:  # noqa: ARG001  # Replaces the stat probe signature
        msg = "permission denied"
        raise PermissionError(msg)

    monkeypatch.setattr(Path, "stat", _deny)

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(str(folder)) == []

    # The Unicode-space fallback re-walks parent segments, so more than one
    # probe can fail; what matters is that the caller learns at least one did.
    assert failures
    assert all("permission denied" in failure.describe() for failure in failures)


def test_track_probe_failures_records_os_rejected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path the OS refuses outright is a failure, not a clean miss.

    Python 3.14 routes `Path.exists()` through `os.path.*`, which swallows
    every `OSError` — including an `ENAMETOOLONG` — and answers `False`. The
    probes must stat directly so such rejections still reach the tracker
    instead of looking like "not a path".

    The rejection is injected rather than provoked with an overlong component:
    `NAME_MAX` is a filesystem property, so a real 300-character segment tests
    the host rather than the code.
    """
    overlong = "/" + "x" * 300

    def _too_long(self: Path, *args: object, **kwargs: object) -> object:  # noqa: ARG001  # Replaces the stat probe signature
        raise OSError(errno.ENAMETOOLONG, "File name too long")

    monkeypatch.setattr(Path, "stat", _too_long)

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(overlong) == []

    assert failures
    assert all("File name too long" in failure.describe() for failure in failures)


@pytest.mark.parametrize("clean_errno", [errno.ENOENT, errno.ENOTDIR])
def test_track_probe_failures_empty_for_clean_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_errno: int
) -> None:
    """`ENOENT` and `ENOTDIR` are clean negatives, not probe failures.

    Both mean "there is definitively nothing here". Recording either would make
    ordinary prose shaped like a path — `/nope/nope` — warn the user and count
    as a drop.
    """

    def _miss(self: Path, *args: object, **kwargs: object) -> object:  # noqa: ARG001  # Replaces the stat probe signature
        raise OSError(clean_errno, os.strerror(clean_errno))

    monkeypatch.setattr(Path, "stat", _miss)

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(str(tmp_path / "missing")) == []

    assert failures == []


def test_track_probe_failures_empty_for_merely_missing_path(tmp_path: Path) -> None:
    """A path that simply does not exist is a clean negative, not a failure."""
    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(str(tmp_path / "missing")) == []

    assert failures == []


def test_track_probe_failures_records_dangling_symlink_target_as_clean_miss(
    tmp_path: Path,
) -> None:
    """A dangling symlink is a miss, not a refusal — its target is just absent."""
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nowhere")

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(str(link)) == []

    assert failures == []


def test_track_probe_failures_records_symlink_loop(tmp_path: Path) -> None:
    """A symlink loop is a refusal, not a miss.

    Python 3.11 and 3.12 raise `RuntimeError` from non-strict `resolve()`, while
    newer versions continue to the following stat and raise `ELOOP`. Either way
    the caller must learn the path could not be probed rather than reading it as
    ordinary text.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.symlink_to(second)
    second.symlink_to(first)

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(str(first)) == []

    assert failures
    assert any(
        isinstance(failure.error, RuntimeError)
        or getattr(failure.error, "errno", None) == errno.ELOOP
        for failure in failures
    )


def test_track_probe_failures_nests_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each block sees only the failures raised inside it.

    The inner block must raise a *real* failure: a merely-missing path records
    nothing, so both lists would be empty however badly nesting leaked.
    """
    folder = tmp_path / "assets"
    folder.mkdir()

    with track_probe_failures() as outer:
        real_stat = Path.stat

        def _deny(self: Path, *args: object, **kwargs: object) -> object:  # noqa: ARG001  # Replaces the stat probe signature
            raise PermissionError(errno.EACCES, "Permission denied")

        with track_probe_failures() as inner:
            monkeypatch.setattr(Path, "stat", _deny)
            assert parse_pasted_any_entry_paths(str(folder)) == []
            monkeypatch.setattr(Path, "stat", real_stat)
        assert inner
        # The inner failures must not have leaked outward.
        assert outer == []


def test_unreadable_working_directory_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused `Path.cwd()` must not escape the guarded resolution helpers.

    A Windows-shaped payload is drop-shaped but relative to `Path`, so it reaches
    the `Path.cwd()` call in the Unicode-space fallback. If the working directory
    has been deleted, the `OSError` would otherwise propagate out of a
    synchronous text-change handler and take down input handling.
    """

    def _deny() -> Path:
        raise OSError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(Path, "cwd", _deny)

    with track_probe_failures() as failures:
        assert parse_pasted_any_entry_paths(r"C:\Users\x\shot.png") == []

    assert failures
    assert isinstance(failures[0].error, OSError)


def test_select_probe_failure_for_returns_none_without_failures() -> None:
    """No recorded failures means there is nothing to report."""
    assert select_probe_failure_for([], "/srv/assets") is None


def test_select_probe_failure_for_prefers_longest_mentioned_path() -> None:
    """The dropped entry must win over the ancestor the fallback also probed.

    The Unicode-space fallback walks ancestor segments, so the first recorded
    failure is often a parent the user never dropped.
    """
    parent = ProbeFailure(Path("/srv"), PermissionError("denied"))
    entry = ProbeFailure(Path("/srv/assets"), PermissionError("denied"))

    assert select_probe_failure_for([parent, entry], "/srv/assets") is entry
    # Recording order must not decide the answer.
    assert select_probe_failure_for([entry, parent], "/srv/assets") is entry


def test_select_probe_failure_for_falls_back_to_first_when_unmentioned() -> None:
    """A failure whose path is absent from the payload is still reported.

    A `~/` or percent-encoded drop records the expanded path, which is not a
    substring of what the user dropped. Returning `None` there would silently
    drop the warning this machinery exists to produce.
    """
    first = ProbeFailure(Path("/Users/x/assets"), PermissionError("denied"))
    second = ProbeFailure(Path("/Users/x"), PermissionError("denied"))

    assert select_probe_failure_for([first, second], "~/assets") is first


def test_dropped_payload_paths_ignores_plain_text() -> None:
    """Ordinary typed text that is not an existing path yields nothing."""
    assert dropped_payload_paths("just some words") == []


def test_dropped_payload_paths_ignores_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path is typed text, not a drop, so it is not resolved.

    Terminals deliver a dragged file as an absolute path, so accepting relative
    tokens here only misfires: it would swallow a hand-typed `assets/logo.png`
    that resolves against the working directory.
    """
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.png").write_bytes(b"img")
    monkeypatch.chdir(tmp_path)

    assert dropped_payload_paths("assets/logo.png") == []


def test_dropped_payload_paths_ignores_leading_path_with_suffix(
    tmp_path: Path,
) -> None:
    """`<path> <question>` is out of scope, matching drop-time chat-input calls."""
    img = tmp_path / "shot.png"
    img.write_bytes(b"img")

    assert dropped_payload_paths(f"{img} what's in this image?") == []


@pytest.mark.parametrize(
    "payload",
    [
        "/tmp/a\x00b.png",
        "file://[::1/x.png",
        "file://[bad",
    ],
)
def test_dropped_payload_paths_tolerates_unparseable_payloads(payload: str) -> None:
    """Payloads the OS or URL parser rejects fall back to text, never raise.

    `Path.resolve` raises `ValueError` on an embedded NUL and `urlparse` raises
    on a malformed authority, neither of which is an `OSError`.
    """
    assert dropped_payload_paths(payload) == []


@pytest.mark.parametrize(
    "payload",
    [
        "/tmp/a\x00b.png",
        "file://[::1/x.png",
    ],
)
def test_parse_pasted_file_paths_tolerates_unparseable_payloads(payload: str) -> None:
    """The strict parser's documented "returns an empty list" contract holds."""
    assert parse_pasted_file_paths(payload) == []


def test_extract_leading_pasted_file_path_with_trailing_text(tmp_path: Path) -> None:
    """Leading path token should be extracted while preserving trailing text."""
    img = tmp_path / "my image.png"
    img.write_bytes(b"img")
    payload = f"'{img}' what's in this image?"

    result = extract_leading_pasted_file_path(payload)

    assert result is not None
    resolved, end = result
    assert resolved == img.resolve()
    assert payload[end:] == " what's in this image?"


def test_extract_leading_pasted_file_path_unquoted_path_with_spaces(
    tmp_path: Path,
) -> None:
    """Unquoted absolute paths with spaces should be extracted from leading text."""
    img = tmp_path / "Screenshot 1.png"
    img.write_bytes(b"img")
    payload = f"{img} what's in this"

    result = extract_leading_pasted_file_path(payload)

    assert result is not None
    resolved, end = result
    assert resolved == img.resolve()
    assert payload[end:] == " what's in this"


def test_parse_pasted_file_paths_handles_overlong_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-long path component must not crash path probing.

    Regression test: holding a key floods the input with a single long token.
    Resolving it against the cwd produces a path whose component exceeds the
    filesystem name limit, so `os.stat` raises `OSError` (`ENAMETOOLONG`). On
    Python <=3.13 `pathlib` lets that propagate (the original crash); on 3.14
    it is swallowed, so this asserts the no-match contract on every version.
    The version-independent guard is `*_handles_oserror_on_probe` below.
    """
    monkeypatch.chdir(tmp_path)
    overlong = "a" * 5000

    assert parse_pasted_file_paths(overlong) == []


def test_parse_pasted_path_payload_handles_overlong_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dropped-path entrypoint must not crash on an over-long token.

    This mirrors the exact path that crashed the TUI: `on_text_area_changed`
    routes freshly typed text through `parse_pasted_path_payload`.
    """
    monkeypatch.chdir(tmp_path)
    overlong = "a" * 5000

    assert parse_pasted_path_payload(overlong) is None


def test_parse_pasted_file_paths_handles_oserror_on_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """`OSError` raised by a stat probe must be swallowed, not propagated.

    Guards against platforms/filesystems that surface the limit at a different
    probe than `resolve`, ensuring the fix does not rely on a specific errno.
    """
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.txt"
    target.write_text("hi")
    mocker.patch("pathlib.Path.stat", side_effect=OSError(63, "File name too long"))

    assert parse_pasted_file_paths(str(target)) == []


def test_parse_pasted_file_paths_handles_oserror_in_unicode_variant(
    tmp_path: Path, mocker
) -> None:
    """An `OSError` probe inside the Unicode-space fallback must not propagate.

    Exercises the `_resolve_with_unicode_space_variants` traversal: the on-disk
    name carries a narrow no-break space while the paste uses an ASCII space,
    forcing the `iterdir`-match branch where component stat probes run. The
    path is quoted so it stays a single token instead of being split.
    """
    unicode_name = "Screenshot 2026-02-26 at 2.02.42 AM.png"
    img = tmp_path / unicode_name
    img.write_bytes(b"img")
    ascii_name = unicode_name.replace(chr(0x202F), " ")
    pasted = f"'{str(img).replace(unicode_name, ascii_name)}'"
    mocker.patch("pathlib.Path.stat", side_effect=OSError(63, "File name too long"))

    assert parse_pasted_file_paths(pasted) == []
