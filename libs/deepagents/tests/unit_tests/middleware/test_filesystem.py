"""Unit tests for filesystem middleware functions."""

from pathlib import Path

from langchain.tools import ToolRuntime

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import MAX_LINE_LENGTH, _read_file_tool_generator


def _write_file(p: Path, content: str) -> None:
    """Helper to write a file with parent directory creation."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_truncate_lines(tmp_path: Path) -> None:
    """Test truncation with mixed short and long lines."""
    root = tmp_path
    fp = root / "mixed.txt"

    # Mix of short and long lines
    line1 = "normal line"
    line2 = "x" * 3000  # Very long
    line3 = "another normal line"
    line4 = "y" * 2100  # Also long
    line5 = "final normal line"

    content = f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n"
    _write_file(fp, content)

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    read_tool = _read_file_tool_generator(backend)

    runtime = ToolRuntime(state={}, context=None, tool_call_id="test_call", store=None, stream_writer=lambda _: None, config={})
    result = read_tool.invoke({"file_path": "/mixed.txt", "runtime": runtime})

    # Normal lines should be present
    assert "normal line" in result
    assert "another normal line" in result
    assert "final normal line" in result

    # Long lines should be truncated with "..."
    x_lines = [line for line in result.split("\n") if "xxx" in line]
    assert len(x_lines) > 0
    assert any(line.rstrip().endswith("...") for line in x_lines)
    assert all(len(line) <= MAX_LINE_LENGTH for line in x_lines)

    y_lines = [line for line in result.split("\n") if "yyy" in line]
    assert len(y_lines) > 0
    assert any(line.rstrip().endswith("...") for line in y_lines)
    assert all(len(line) <= MAX_LINE_LENGTH for line in y_lines)


def test_truncate_lines_preserves_newlines(tmp_path: Path) -> None:
    """Test that newlines are preserved correctly."""
    root = tmp_path
    fp = root / "newlines.txt"

    # Create content with different newline patterns
    long_line = "b" * 2500
    content = f"line1\n{long_line}\nline3"
    _write_file(fp, content)

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    read_tool = _read_file_tool_generator(backend)

    runtime = ToolRuntime(state={}, context=None, tool_call_id="test_call", store=None, stream_writer=lambda _: None, config={})
    result = read_tool.invoke({"file_path": "/newlines.txt", "runtime": runtime})

    # Should have multiple lines
    expected_min_lines = 3
    lines = result.split("\n")
    assert len(lines) >= expected_min_lines

    # Check that line1 and line3 are present
    assert any("line1" in line for line in lines)
    assert any("line3" in line for line in lines)


def test_truncate_lines_empty_file(tmp_path: Path) -> None:
    """Test reading an empty file."""
    root = tmp_path
    fp = root / "empty.txt"
    _write_file(fp, "")

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    read_tool = _read_file_tool_generator(backend)

    runtime = ToolRuntime(state={}, context=None, tool_call_id="test_call", store=None, stream_writer=lambda _: None, config={})
    result = read_tool.invoke({"file_path": "/empty.txt", "runtime": runtime})

    # Empty file should return empty or minimal content
    # (FilesystemBackend might add warnings or format)
    assert isinstance(result, str)
