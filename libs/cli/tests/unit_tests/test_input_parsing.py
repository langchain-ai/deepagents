from prompt_toolkit.document import Document

from deepagents_cli.config import COLORS
from deepagents_cli.input import MentionHighlightLexer, parse_file_mentions


def test_parse_file_mentions_with_chinese_sentence(tmp_path, monkeypatch):
    """Ensure @file parsing stops before Chinese text or punctuation."""
    file_path = tmp_path / "input.py"
    file_path.write_text("print('hello')")

    monkeypatch.chdir(tmp_path)
    text = f"你分析@{file_path.name}的代码就懂了"

    _, files = parse_file_mentions(text)

    assert files == [file_path.resolve()]


def test_parse_file_mentions_handles_multiple_mentions(tmp_path, monkeypatch):
    """Ensure multiple @file mentions are extracted from a single input."""
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("1")
    second.write_text("2")

    monkeypatch.chdir(tmp_path)
    text = f"读一下@{first.name}，然后看看@{second.name}。"

    _, files = parse_file_mentions(text)

    assert files == [first.resolve(), second.resolve()]


def test_highlighter_marks_file_mentions():
    """Ensure @file mentions are highlighted with the correct style."""
    lexer = MentionHighlightLexer()
    document = Document("请阅读@README.md然后继续", cursor_position=0)
    tokens = lexer.lex_document(document)(0)

    default_style = f"fg:{COLORS['user']}"
    path_style = f"bold fg:{COLORS['primary']}"

    assert tokens == [
        (default_style, "请阅读"),
        (path_style, "@README.md"),
        (default_style, "然后继续"),
    ]


def test_highlighter_skips_email_addresses():
    """Ensure email addresses are not highlighted as file mentions."""
    lexer = MentionHighlightLexer()
    document = Document("contact user@example.com please", cursor_position=0)
    tokens = lexer.lex_document(document)(0)

    default_style = f"fg:{COLORS['user']}"

    # Entire line should be default style (no highlighting)
    assert tokens == [(default_style, "contact user@example.com please")]


def test_highlighter_highlights_file_but_not_email():
    """Ensure @file is highlighted but email in same line is not."""
    lexer = MentionHighlightLexer()
    document = Document("read @file.py and email me@test.com", cursor_position=0)
    tokens = lexer.lex_document(document)(0)

    default_style = f"fg:{COLORS['user']}"
    path_style = f"bold fg:{COLORS['primary']}"

    assert tokens == [
        (default_style, "read "),
        (path_style, "@file.py"),
        (default_style, " and email me@test.com"),
    ]


def test_highlighter_marks_slash_only_on_first_line():
    """Ensure slash commands are only highlighted on the first line."""
    lexer = MentionHighlightLexer()
    document = Document("/help 命令\n/ignore", cursor_position=0)

    first_line = lexer.lex_document(document)(0)
    second_line = lexer.lex_document(document)(1)

    command_style = f"bold fg:{COLORS['tool']}"
    default_style = f"fg:{COLORS['user']}"

    assert first_line[0] == (command_style, "/help")
    assert second_line[0] == (default_style, "/ignore")


def test_parse_file_mentions_with_escaped_spaces(tmp_path, monkeypatch):
    """Ensure escaped spaces in paths are handled correctly."""
    spaced_dir = tmp_path / "my folder"
    spaced_dir.mkdir()
    file_path = spaced_dir / "test.py"
    file_path.write_text("content")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("@my\\ folder/test.py")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_warns_for_nonexistent_file(tmp_path, monkeypatch, mocker):
    """Ensure non-existent files are excluded and warning is printed."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_cli.input.console")

    _, files = parse_file_mentions("@nonexistent.py")

    assert files == []
    mock_console.print.assert_called_once()
    assert "nonexistent.py" in mock_console.print.call_args[0][0]


def test_parse_file_mentions_ignores_directories(tmp_path, monkeypatch):
    """Ensure directories are not included in file list."""
    dir_path = tmp_path / "mydir"
    dir_path.mkdir()
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("@mydir")

    assert files == []


def test_parse_file_mentions_with_no_mentions():
    """Ensure text without mentions returns empty file list."""
    _, files = parse_file_mentions("just some text without mentions")
    assert files == []


def test_parse_file_mentions_handles_path_traversal(tmp_path, monkeypatch):
    """Ensure path traversal sequences are resolved to actual paths."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file_path = tmp_path / "test.txt"
    file_path.write_text("content")
    monkeypatch.chdir(subdir)

    _, files = parse_file_mentions("@../test.txt")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_with_absolute_path(tmp_path):
    """Ensure absolute paths are resolved correctly without cwd changes."""
    file_path = tmp_path / "test.py"
    file_path.write_text("content")

    _, files = parse_file_mentions(f"@{file_path}")

    assert files == [file_path.resolve()]


def test_parse_file_mentions_handles_multiple_in_sentence(tmp_path, monkeypatch):
    """Ensure multiple @mentions with spaces are parsed as separate files."""
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("1")
    second.write_text("2")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("compare @a.py and @b.py")

    assert files == [first.resolve(), second.resolve()]


def test_parse_file_mentions_adjacent_looks_like_email(tmp_path, monkeypatch, mocker):
    """Adjacent @mentions without space look like emails and are skipped.

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


def test_parse_file_mentions_handles_oserror(tmp_path, monkeypatch, mocker):
    """Ensure OSError during path resolution is handled gracefully."""
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_cli.input.console")
    mocker.patch("pathlib.Path.resolve", side_effect=OSError("Permission denied"))

    _, files = parse_file_mentions("@somefile.py")

    assert files == []
    mock_console.print.assert_called_once()
    call_arg = mock_console.print.call_args[0][0]
    assert "somefile.py" in call_arg
    assert "Invalid path" in call_arg


def test_parse_file_mentions_skips_email_addresses(tmp_path, monkeypatch, mocker):
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


def test_parse_file_mentions_skips_various_email_formats(tmp_path, monkeypatch, mocker):
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


def test_parse_file_mentions_works_after_cjk_text(tmp_path, monkeypatch, mocker):
    """Ensure @file mentions work after CJK text (not email-like)."""
    file_path = tmp_path / "test.py"
    file_path.write_text("content")
    monkeypatch.chdir(tmp_path)
    mock_console = mocker.patch("deepagents_cli.input.console")

    # CJK character before @ is not email-like, so this should parse
    _, files = parse_file_mentions("查看@test.py")

    assert files == [file_path.resolve()]
    mock_console.print.assert_not_called()
