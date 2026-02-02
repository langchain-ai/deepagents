"""Unit tests for UI rendering utilities."""

from deepagents_cli.ui import _format_timeout, format_tool_display, truncate_value


class TestFormatTimeout:
    """Tests for `_format_timeout`."""

    def test_seconds(self) -> None:
        """Test formatting values under 60 as seconds."""
        assert _format_timeout(30) == "30s"
        assert _format_timeout(59) == "59s"

    def test_minutes(self) -> None:
        """Test formatting round minute values."""
        assert _format_timeout(60) == "1m"
        assert _format_timeout(300) == "5m"
        assert _format_timeout(600) == "10m"

    def test_hours(self) -> None:
        """Test formatting round hour values."""
        assert _format_timeout(3600) == "1h"
        assert _format_timeout(7200) == "2h"

    def test_odd_values_as_seconds(self) -> None:
        """Test that non-round values show as seconds."""
        assert _format_timeout(90) == "90s"  # 1.5 minutes
        assert _format_timeout(3700) == "3700s"  # not round hours

    def test_likely_milliseconds_shown_as_seconds(self) -> None:
        """Test that large values (likely ms confusion) still show with unit."""
        # 120000 looks like milliseconds for 120 seconds
        assert _format_timeout(120000) == "120000s"


class TestTruncateValue:
    """Tests for `truncate_value`."""

    def test_short_string_unchanged(self) -> None:
        """Test that short strings are not truncated."""
        result = truncate_value("hello", max_length=10)
        assert result == "hello"

    def test_long_string_truncated(self) -> None:
        """Test that long strings are truncated with ellipsis."""
        result = truncate_value("hello world", max_length=5)
        assert result == "hello..."

    def test_exact_length_unchanged(self) -> None:
        """Test that strings at exact max length are unchanged."""
        result = truncate_value("hello", max_length=5)
        assert result == "hello"


class TestFormatToolDisplayShell:
    """Tests for `format_tool_display` with shell tool."""

    def test_shell_command_only(self) -> None:
        """Test shell display with command only."""
        result = format_tool_display("shell", {"command": "echo hello"})
        assert result == 'shell("echo hello")'

    def test_shell_with_timeout_minutes(self) -> None:
        """Test shell display formats timeout in minutes when appropriate."""
        result = format_tool_display("shell", {"command": "make test", "timeout": 300})
        assert result == 'shell("make test", timeout=5m)'

    def test_shell_with_timeout_seconds(self) -> None:
        """Test shell display formats timeout in seconds for small values."""
        result = format_tool_display("shell", {"command": "make test", "timeout": 30})
        assert result == 'shell("make test", timeout=30s)'

    def test_shell_with_timeout_hours(self) -> None:
        """Test shell display formats timeout in hours when appropriate."""
        result = format_tool_display("shell", {"command": "make test", "timeout": 3600})
        assert result == 'shell("make test", timeout=1h)'

    def test_shell_with_none_timeout(self) -> None:
        """Test shell display excludes timeout when `None`."""
        result = format_tool_display("shell", {"command": "echo hello", "timeout": None})
        assert result == 'shell("echo hello")'

    def test_shell_with_default_timeout_hidden(self) -> None:
        """Test shell display excludes timeout when it equals the default (120s)."""
        result = format_tool_display("shell", {"command": "echo hello", "timeout": 120})
        assert result == 'shell("echo hello")'

    def test_shell_long_command_truncated(self) -> None:
        """Test that long shell commands are truncated."""
        long_cmd = "x" * 200
        result = format_tool_display("shell", {"command": long_cmd})
        assert "..." in result
        assert len(result) < 200


class TestFormatToolDisplayOther:
    """Tests for `format_tool_display` with other tools."""

    def test_read_file(self) -> None:
        """Test read_file display shows filename."""
        result = format_tool_display("read_file", {"file_path": "/path/to/file.py"})
        assert "file.py" in result

    def test_web_search(self) -> None:
        """Test web_search display shows query."""
        result = format_tool_display("web_search", {"query": "how to code"})
        assert result == 'web_search("how to code")'

    def test_grep(self) -> None:
        """Test grep display shows pattern."""
        result = format_tool_display("grep", {"pattern": "TODO"})
        assert result == 'grep("TODO")'

    def test_unknown_tool_fallback(self) -> None:
        """Test unknown tools use generic formatting."""
        result = format_tool_display("custom_tool", {"arg1": "val1", "arg2": "val2"})
        assert "custom_tool(" in result
        assert "arg1=" in result
        assert "arg2=" in result
