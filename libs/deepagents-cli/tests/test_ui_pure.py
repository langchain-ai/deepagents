"""Tests for pure UI utility functions (no mocks needed)."""

from deepagents_cli.ui import (
    _format_line_span,
    format_tool_display,
    format_tool_message_content,
    truncate_value,
)


class TestTruncateValue:
    """Tests for truncate_value function."""

    def test_truncate_short_string(self) -> None:
        """Test that short strings are not truncated."""
        result = truncate_value("hello", max_length=10)
        assert result == "hello"

    def test_truncate_exact_length(self) -> None:
        """Test string exactly at max length."""
        result = truncate_value("1234567890", max_length=10)
        assert result == "1234567890"

    def test_truncate_long_string(self) -> None:
        """Test that long strings are truncated with ellipsis."""
        result = truncate_value("1234567890abc", max_length=10)
        assert result == "1234567890..."
        assert len(result) == 13  # 10 chars + "..."

    def test_truncate_default_max_length(self) -> None:
        """Test truncate uses default MAX_ARG_LENGTH."""
        from deepagents_cli.config import MAX_ARG_LENGTH

        long_string = "x" * (MAX_ARG_LENGTH + 100)
        result = truncate_value(long_string)
        assert result.endswith("...")
        assert len(result) == MAX_ARG_LENGTH + 3

    def test_truncate_empty_string(self) -> None:
        """Test empty string returns empty."""
        result = truncate_value("")
        assert result == ""

    def test_truncate_with_special_characters(self) -> None:
        """Test truncation preserves special characters."""
        result = truncate_value("hello🌟world", max_length=8)
        assert result == "hello🌟wo..."


class TestFormatToolDisplay:
    """Tests for format_tool_display function."""

    def test_read_file_with_file_path(self) -> None:
        """Test read_file shows abbreviated path."""
        result = format_tool_display("read_file", {"file_path": "/long/path/to/file.py"})
        assert "read_file" in result
        assert "file.py" in result

    def test_write_file_with_path(self) -> None:
        """Test write_file with 'path' argument."""
        result = format_tool_display("write_file", {"path": "config.yaml"})
        assert "write_file" in result
        assert "config.yaml" in result

    def test_edit_file(self) -> None:
        """Test edit_file formatting."""
        result = format_tool_display("edit_file", {"file_path": "main.py"})
        assert result == "edit_file(main.py)"

    def test_web_search(self) -> None:
        """Test web_search shows query."""
        result = format_tool_display("web_search", {"query": "how to code in python"})
        assert result == 'web_search("how to code in python")'

    def test_web_search_long_query(self) -> None:
        """Test web_search truncates long queries."""
        long_query = "x" * 150
        result = format_tool_display("web_search", {"query": long_query})
        assert "..." in result
        # Should be truncated to 100 chars + "..."
        assert len(result) <= len('web_search("")') + 100 + 3 + 10

    def test_grep_with_pattern(self) -> None:
        """Test grep shows search pattern."""
        result = format_tool_display("grep", {"pattern": "def.*main"})
        assert result == 'grep("def.*main")'

    def test_shell_command(self) -> None:
        """Test shell shows command."""
        result = format_tool_display("shell", {"command": "git status"})
        assert result == 'shell("git status")'

    def test_shell_long_command(self) -> None:
        """Test shell truncates long commands."""
        long_cmd = "echo " + "x" * 200
        result = format_tool_display("shell", {"command": long_cmd})
        assert "..." in result

    def test_ls_with_path(self) -> None:
        """Test ls with directory path."""
        result = format_tool_display("ls", {"path": "/home/user"})
        assert "ls(" in result
        assert "home" in result or "user" in result

    def test_ls_without_path(self) -> None:
        """Test ls without path shows empty parens."""
        result = format_tool_display("ls", {})
        assert result == "ls()"

    def test_glob_with_pattern(self) -> None:
        """Test glob shows pattern."""
        result = format_tool_display("glob", {"pattern": "**/*.py"})
        assert result == 'glob("**/*.py")'

    def test_http_request_get(self) -> None:
        """Test HTTP request formatting."""
        result = format_tool_display(
            "http_request", {"method": "GET", "url": "https://example.com/api"}
        )
        assert "http_request" in result
        assert "GET" in result
        assert "example.com" in result

    def test_http_request_post(self) -> None:
        """Test HTTP POST request."""
        result = format_tool_display("http_request", {"method": "post", "url": "https://api.test"})
        assert "POST" in result  # Should be uppercase

    def test_http_request_long_url(self) -> None:
        """Test HTTP request with long URL."""
        long_url = "https://example.com/" + "path/" * 50
        result = format_tool_display("http_request", {"method": "GET", "url": long_url})
        assert "..." in result

    def test_task_with_description(self) -> None:
        """Test task formatting."""
        result = format_tool_display("task", {"description": "Install dependencies"})
        assert result == 'task("Install dependencies")'

    def test_write_todos(self) -> None:
        """Test write_todos shows count."""
        todos = [{"content": "Task 1"}, {"content": "Task 2"}, {"content": "Task 3"}]
        result = format_tool_display("write_todos", {"todos": todos})
        assert result == "write_todos(3 items)"

    def test_write_todos_single_item(self) -> None:
        """Test write_todos with single item."""
        result = format_tool_display("write_todos", {"todos": [{"content": "Task"}]})
        assert result == "write_todos(1 items)"

    def test_write_todos_empty_list(self) -> None:
        """Test write_todos with empty list."""
        result = format_tool_display("write_todos", {"todos": []})
        assert result == "write_todos(0 items)"

    def test_unknown_tool_fallback(self) -> None:
        """Test unknown tool uses fallback formatting."""
        result = format_tool_display("unknown_tool", {"arg1": "value1", "arg2": "value2"})
        assert "unknown_tool" in result
        assert "arg1" in result
        assert "value1" in result

    def test_tool_with_no_args(self) -> None:
        """Test tool with no arguments."""
        result = format_tool_display("some_tool", {})
        assert result == "some_tool()"

    def test_read_file_simple_filename(self) -> None:
        """Test read_file with simple filename (no path)."""
        result = format_tool_display("read_file", {"file_path": "README.md"})
        assert result == "read_file(README.md)"


class TestFormatToolMessageContent:
    """Tests for format_tool_message_content function."""

    def test_none_content(self) -> None:
        """Test None returns empty string."""
        result = format_tool_message_content(None)
        assert result == ""

    def test_string_content(self) -> None:
        """Test string content is returned as-is."""
        result = format_tool_message_content("hello world")
        assert result == "hello world"

    def test_list_of_strings(self) -> None:
        """Test list of strings joined with newlines."""
        result = format_tool_message_content(["line1", "line2", "line3"])
        assert result == "line1\nline2\nline3"

    def test_list_with_dicts(self) -> None:
        """Test list containing dicts converted to JSON."""
        content = [{"key": "value"}, {"foo": "bar"}]
        result = format_tool_message_content(content)
        assert '{"key": "value"}' in result
        assert '{"foo": "bar"}' in result
        assert "\n" in result

    def test_list_mixed_types(self) -> None:
        """Test list with mixed types."""
        content = ["text", {"dict": "value"}, 123]
        result = format_tool_message_content(content)
        assert "text" in result
        assert '"dict"' in result
        assert "123" in result

    def test_number_content(self) -> None:
        """Test number is converted to string."""
        result = format_tool_message_content(42)
        assert result == "42"

    def test_dict_content(self) -> None:
        """Test dict is converted to string."""
        result = format_tool_message_content({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_empty_list(self) -> None:
        """Test empty list returns empty string."""
        result = format_tool_message_content([])
        assert result == ""

    def test_list_with_none_values(self) -> None:
        """Test list containing None values."""
        result = format_tool_message_content([None, "text", None])
        # Should handle None gracefully
        assert "text" in result


class TestFormatLineSpan:
    """Tests for _format_line_span function."""

    def test_both_none(self) -> None:
        """Test when both start and end are None."""
        result = _format_line_span(None, None)
        assert result == ""

    def test_only_start(self) -> None:
        """Test when only start is provided."""
        result = _format_line_span(10, None)
        assert result == "(starting at line 10)"

    def test_only_end(self) -> None:
        """Test when only end is provided."""
        result = _format_line_span(None, 20)
        assert result == "(through line 20)"

    def test_same_line(self) -> None:
        """Test when start equals end."""
        result = _format_line_span(5, 5)
        assert result == "(line 5)"

    def test_line_range(self) -> None:
        """Test when start and end are different."""
        result = _format_line_span(1, 10)
        assert result == "(lines 1-10)"

    def test_large_range(self) -> None:
        """Test with large line numbers."""
        result = _format_line_span(1000, 2000)
        assert result == "(lines 1000-2000)"

    def test_single_digit_lines(self) -> None:
        """Test with single digit line numbers."""
        result = _format_line_span(1, 9)
        assert result == "(lines 1-9)"
