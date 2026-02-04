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


def test_parse_file_mentions_handles_adjacent_mentions(tmp_path, monkeypatch):
    """Ensure adjacent @mentions (no space) are parsed as separate files."""
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("1")
    second.write_text("2")
    monkeypatch.chdir(tmp_path)

    _, files = parse_file_mentions("@a.py@b.py")

    assert files == [first.resolve(), second.resolve()]
