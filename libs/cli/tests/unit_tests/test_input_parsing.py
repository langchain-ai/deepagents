"""Unit tests for input parsing utilities."""

from pathlib import Path

import pytest

from deepagents_cli.input import (
    parse_file_mentions,
    parse_pasted_file_paths,
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
    mock_console = mocker.patch("deepagents_cli.input.console")

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
    mock_console = mocker.patch("deepagents_cli.input.console")

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
    mock_console = mocker.patch("deepagents_cli.input.console")

    _, files = parse_file_mentions("@a.py@b.py")

    # Only first file is parsed; second looks like email and is skipped
    assert files == [first.resolve()]
    mock_console.print.assert_not_called()


def test_parse_file_mentions_handles_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure `OSError` during path resolution is handled gracefully."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_cli.input.console")
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
    mock_console = mocker.patch("deepagents_cli.input.console")

    _, files = parse_file_mentions("contact me at user@example.com")

    # Email addresses should be silently skipped (no warning, no files)
    assert files == []
    mock_console.print.assert_not_called()


def test_parse_file_mentions_skips_various_email_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    """Ensure various email formats are all skipped."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_cli.input.console")

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
    mock_console = mocker.patch("deepagents_cli.input.console")

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
    mock_console = mocker.patch("deepagents_cli.input.console")

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
