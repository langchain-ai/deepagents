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
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("1")
    second.write_text("2")

    monkeypatch.chdir(tmp_path)
    text = f"读一下@{first.name}，然后看看@{second.name}。"

    _, files = parse_file_mentions(text)

    assert files == [first.resolve(), second.resolve()]


def test_highlighter_marks_file_mentions():
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
    lexer = MentionHighlightLexer()
    document = Document("/help 命令\n/ignore", cursor_position=0)

    first_line = lexer.lex_document(document)(0)
    second_line = lexer.lex_document(document)(1)

    command_style = f"bold fg:{COLORS['tool']}"
    default_style = f"fg:{COLORS['user']}"

    assert first_line[0] == (command_style, "/help")
    assert second_line[0] == (default_style, "/ignore")
