"""Unit tests for UI rendering utilities."""

from deepagents_cli.config import get_glyphs
from deepagents_cli.tool_display import (
    _format_content_block,
    _format_timeout,
    format_tool_display,
    format_tool_message_content,
    truncate_value,
)


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
        assert result == f"hello{get_glyphs().ellipsis}"

    def test_exact_length_unchanged(self) -> None:
        """Test that strings at exact max length are unchanged."""
        result = truncate_value("hello", max_length=5)
        assert result == "hello"


class TestFormatToolDisplayExecute:
    """Tests for `format_tool_display` with execute tool."""

    def test_execute_command_only(self) -> None:
        """Test execute display with command only."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("execute", {"command": "echo hello"})
        assert result == f'{prefix} execute("echo hello")'

    def test_execute_with_timeout_minutes(self) -> None:
        """Test execute display formats timeout in minutes when appropriate."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display(
            "execute", {"command": "make test", "timeout": 300}
        )
        assert result == f'{prefix} execute("make test", timeout=5m)'

    def test_execute_with_timeout_seconds(self) -> None:
        """Test execute display formats timeout in seconds for small values."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("execute", {"command": "make test", "timeout": 30})
        assert result == f'{prefix} execute("make test", timeout=30s)'

    def test_execute_with_timeout_hours(self) -> None:
        """Test execute display formats timeout in hours when appropriate."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display(
            "execute", {"command": "make test", "timeout": 3600}
        )
        assert result == f'{prefix} execute("make test", timeout=1h)'

    def test_execute_with_none_timeout(self) -> None:
        """Test execute display excludes timeout when `None`."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display(
            "execute", {"command": "echo hello", "timeout": None}
        )
        assert result == f'{prefix} execute("echo hello")'

    def test_execute_with_default_timeout_hidden(self) -> None:
        """Test execute display excludes timeout when it equals the default (120s)."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display(
            "execute", {"command": "echo hello", "timeout": 120}
        )
        assert result == f'{prefix} execute("echo hello")'

    def test_execute_long_command_truncated(self) -> None:
        """Test that long execute commands are truncated."""
        long_cmd = "x" * 200
        result = format_tool_display("execute", {"command": long_cmd})
        assert get_glyphs().ellipsis in result
        assert len(result) < 200


class TestFormatToolDisplayOther:
    """Tests for `format_tool_display` with other tools."""

    def test_read_file(self) -> None:
        """Test read_file display shows filename with icon."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("read_file", {"file_path": "/path/to/file.py"})
        assert result.startswith(f"{prefix} read_file(")
        assert "file.py" in result

    def test_web_search(self) -> None:
        """Test web_search display shows query."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("web_search", {"query": "how to code"})
        assert result == f'{prefix} web_search("how to code")'

    def test_grep(self) -> None:
        """Test grep display shows pattern."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("grep", {"pattern": "TODO"})
        assert result == f'{prefix} grep("TODO")'

    def test_unknown_tool_fallback(self) -> None:
        """Test unknown tools use generic formatting."""
        prefix = get_glyphs().tool_prefix
        result = format_tool_display("custom_tool", {"arg1": "val1", "arg2": "val2"})
        assert f"{prefix} custom_tool(" in result
        assert "arg1=" in result
        assert "arg2=" in result


class TestFormatToolMessageContent:
    """Tests for `format_tool_message_content`."""

    def test_none_returns_empty_string(self) -> None:
        """Test that None content returns empty string."""
        assert format_tool_message_content(None) == ""

    def test_plain_string_returned_as_is(self) -> None:
        """Test that a plain string is returned unchanged."""
        assert format_tool_message_content("hello") == "hello"

    def test_list_of_strings_joined(self) -> None:
        """Test that a list of strings is joined with newlines."""
        assert format_tool_message_content(["a", "b"]) == "a\nb"

    def test_list_with_dict_uses_json(self) -> None:
        """Test that dicts in a list are serialized as JSON."""
        result = format_tool_message_content([{"key": "val"}])
        assert '"key"' in result
        assert '"val"' in result

    def test_list_mixed_types(self) -> None:
        """Test a list with both strings and dicts."""
        result = format_tool_message_content(["text", {"k": 1}])
        lines = result.split("\n")
        assert lines[0] == "text"
        assert '"k"' in lines[1]

    def test_non_serializable_falls_back_to_str(self) -> None:
        """Test that non-JSON-serializable items fall back to str()."""
        obj = object()
        result = format_tool_message_content([obj])
        assert "object" in result

    def test_integer_content(self) -> None:
        """Test that non-string, non-list content is stringified."""
        assert format_tool_message_content(42) == "42"

    def test_image_block_shows_placeholder(self) -> None:
        """Test that image content blocks show a placeholder instead of base64."""
        content = [{"type": "image", "base64": "A" * 4000, "mime_type": "image/png"}]
        result = format_tool_message_content(content)
        assert "Image" in result
        assert "image/png" in result
        assert "KB" in result
        # Must NOT contain raw base64
        assert "AAAA" not in result

    def test_image_block_without_mime_type(self) -> None:
        """Test image block falls back to generic 'image' when mime_type missing."""
        content = [{"type": "image", "base64": "data"}]
        result = format_tool_message_content(content)
        assert "Image" in result
        assert "image" in result

    def test_mixed_list_with_strings_and_image_blocks(self) -> None:
        """Test that mixed string/image list preserves ordering."""
        content = [
            "Here is the screenshot:",
            {"type": "image", "base64": "A" * 4000, "mime_type": "image/png"},
            "Analysis complete.",
        ]
        result = format_tool_message_content(content)
        lines = result.split("\n")
        assert lines[0] == "Here is the screenshot:"
        assert "Image" in lines[1]
        assert "AAAA" not in lines[1]
        assert lines[2] == "Analysis complete."


class TestFormatContentBlock:
    """Tests for `_format_content_block`."""

    def test_image_block_placeholder(self) -> None:
        """Test image block returns a human-readable placeholder."""
        block = {
            "type": "image",
            "base64": "A" * 40000,
            "mime_type": "image/jpeg",
        }
        result = _format_content_block(block)
        assert result == "[Image: image/jpeg, ~29KB]"

    def test_non_image_dict_returns_json(self) -> None:
        """Test that non-image dicts are still JSON-serialized."""
        block = {"type": "text", "content": "hello"}
        result = _format_content_block(block)
        assert '"type"' in result
        assert '"text"' in result

    def test_image_block_without_base64_returns_json(self) -> None:
        """Test that image blocks missing base64 key fall back to JSON."""
        block = {"type": "image", "url": "https://example.com/img.png"}
        result = _format_content_block(block)
        assert '"url"' in result

    def test_image_block_none_base64_returns_json(self) -> None:
        """Test that image block with None base64 falls through to JSON."""
        block = {"type": "image", "base64": None, "mime_type": "image/png"}
        result = _format_content_block(block)
        assert '"type"' in result
        assert "Image" not in result

    def test_image_block_non_string_base64_returns_json(self) -> None:
        """Test that image block with non-string base64 falls through to JSON."""
        block = {"type": "image", "base64": 12345}
        result = _format_content_block(block)
        assert "12345" in result
        assert "Image" not in result

    def test_image_block_empty_base64(self) -> None:
        """Test that empty base64 string produces a 0KB placeholder."""
        block = {"type": "image", "base64": "", "mime_type": "image/png"}
        result = _format_content_block(block)
        assert result == "[Image: image/png, ~0KB]"

    def test_non_serializable_dict_falls_back_to_str(self) -> None:
        """Test that dicts with non-serializable values fall back to str()."""
        block = {"type": "data", "value": object()}
        result = _format_content_block(block)
        assert "type" in result
