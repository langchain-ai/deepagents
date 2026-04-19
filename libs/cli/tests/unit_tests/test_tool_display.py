"""Unit tests for deepagents_cli/tool_display.py.

All functions under test are pure (no I/O, no async, no TUI), so no fixtures
or mocks are needed beyond patching ``get_glyphs()`` to force a deterministic
ASCII prefix and ``DEFAULT_EXECUTE_TIMEOUT`` to a known value.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deepagents_cli.config import ASCII_GLYPHS
from deepagents_cli.tool_display import (
    _HIDDEN_CHAR_MARKER,
    _coerce_timeout_seconds,
    _format_content_block,
    _format_timeout,
    _sanitize_display_value,
    format_tool_display,
    format_tool_message_content,
    truncate_value,
)

# Deterministic prefix for all format_tool_display tests.
_PREFIX = ASCII_GLYPHS.tool_prefix  # "(*)
_ELLIPSIS = ASCII_GLYPHS.ellipsis  # "..."


# ---------------------------------------------------------------------------
# _format_timeout
# ---------------------------------------------------------------------------


class TestFormatTimeout:
    """Tests for _format_timeout()."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            pytest.param(0, "0s", id="zero"),
            pytest.param(1, "1s", id="one-second"),
            pytest.param(59, "59s", id="59-seconds"),
            pytest.param(60, "1m", id="exact-one-minute"),
            pytest.param(120, "2m", id="exact-two-minutes"),
            pytest.param(3540, "59m", id="exact-59-minutes"),
            pytest.param(3600, "1h", id="exact-one-hour"),
            pytest.param(7200, "2h", id="exact-two-hours"),
            pytest.param(90, "90s", id="irregular-90s"),
            pytest.param(3601, "3601s", id="irregular-3601s"),
        ],
    )
    def test_format_timeout(self, seconds: int, expected: str) -> None:
        assert _format_timeout(seconds) == expected


# ---------------------------------------------------------------------------
# _coerce_timeout_seconds
# ---------------------------------------------------------------------------


class TestCoerceTimeoutSeconds:
    """Tests for _coerce_timeout_seconds()."""

    def test_int_passthrough(self) -> None:
        assert _coerce_timeout_seconds(120) == 120

    def test_int_zero(self) -> None:
        assert _coerce_timeout_seconds(0) == 0

    def test_valid_string(self) -> None:
        assert _coerce_timeout_seconds("300") == 300

    def test_string_with_whitespace(self) -> None:
        assert _coerce_timeout_seconds("  60  ") == 60

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_timeout_seconds("") is None

    def test_whitespace_only_string_returns_none(self) -> None:
        assert _coerce_timeout_seconds("   ") is None

    def test_invalid_string_returns_none(self) -> None:
        assert _coerce_timeout_seconds("abc") is None

    def test_float_string_returns_none(self) -> None:
        # Only integer strings are accepted
        assert _coerce_timeout_seconds("1.5") is None

    def test_none_returns_none(self) -> None:
        assert _coerce_timeout_seconds(None) is None

    def test_float_type_returns_none(self) -> None:
        # Only exact int type passes (type(x) is int check)
        assert _coerce_timeout_seconds(1.5) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# truncate_value
# ---------------------------------------------------------------------------


class TestTruncateValue:
    """Tests for truncate_value()."""

    def test_short_value_unchanged(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            assert truncate_value("hello", max_length=10) == "hello"

    def test_exactly_at_limit_unchanged(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            value = "a" * 10
            assert truncate_value(value, max_length=10) == value

    def test_over_limit_truncated_with_ellipsis(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            value = "a" * 15
            result = truncate_value(value, max_length=10)
            assert result == "a" * 10 + _ELLIPSIS

    def test_empty_string_unchanged(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            assert truncate_value("", max_length=10) == ""


# ---------------------------------------------------------------------------
# _sanitize_display_value
# ---------------------------------------------------------------------------


class TestSanitizeDisplayValue:
    """Tests for _sanitize_display_value()."""

    def test_clean_value_returned_as_is(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            assert _sanitize_display_value("hello world") == "hello world"

    def test_hidden_unicode_stripped_and_marker_appended(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            # U+200B is a zero-width space — stripped by strip_dangerous_unicode
            result = _sanitize_display_value("hello\u200bworld")
            assert "helloworld" in result
            assert _HIDDEN_CHAR_MARKER in result

    def test_long_clean_value_truncated(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            long_value = "x" * 200
            result = _sanitize_display_value(long_value, max_length=50)
            assert len(result) == 50 + len(_ELLIPSIS)
            assert result.endswith(_ELLIPSIS)

    def test_non_string_value_coerced(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            assert _sanitize_display_value(42) == "42"

    def test_none_value_coerced(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            assert _sanitize_display_value(None) == "None"


# ---------------------------------------------------------------------------
# format_tool_display — per-tool branches
# ---------------------------------------------------------------------------


class TestFormatToolDisplay:
    """Tests for format_tool_display() — one test per tool branch."""

    @pytest.fixture(autouse=True)
    def _patch_glyphs(self) -> None:
        with patch("deepagents_cli.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
            yield

    # --- file tools ---

    @pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file"])
    def test_file_tool_with_file_path(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {"file_path": "/tmp/test.py"})
        assert result.startswith(_PREFIX)
        assert tool_name in result
        assert "test.py" in result

    @pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file"])
    def test_file_tool_with_path_key(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {"path": "/tmp/test.py"})
        assert "test.py" in result

    @pytest.mark.parametrize("tool_name", ["read_file", "write_file", "edit_file"])
    def test_file_tool_missing_path_falls_back_to_generic(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {})
        assert _PREFIX in result
        assert tool_name in result

    # --- web_search ---

    def test_web_search_shows_query(self) -> None:
        result = format_tool_display("web_search", {"query": "how to code"})
        assert 'web_search("how to code")' in result

    def test_web_search_missing_query_falls_back(self) -> None:
        result = format_tool_display("web_search", {})
        assert "web_search" in result

    # --- grep ---

    def test_grep_shows_pattern(self) -> None:
        result = format_tool_display("grep", {"pattern": "def foo"})
        assert 'grep("def foo")' in result

    # --- execute ---

    def test_execute_shows_command(self) -> None:
        # DEFAULT_EXECUTE_TIMEOUT is a deferred import inside the function body;
        # patch it at the source module.
        with patch("deepagents.backends.DEFAULT_EXECUTE_TIMEOUT", 120):
            result = format_tool_display("execute", {"command": "ls -la"})
        assert 'execute("ls -la")' in result

    def test_execute_shows_timeout_when_non_default(self) -> None:
        with patch("deepagents.backends.DEFAULT_EXECUTE_TIMEOUT", 120):
            result = format_tool_display(
                "execute", {"command": "sleep 5", "timeout": 300}
            )
        assert "timeout=5m" in result

    def test_execute_omits_timeout_when_default(self) -> None:
        with patch("deepagents.backends.DEFAULT_EXECUTE_TIMEOUT", 120):
            result = format_tool_display(
                "execute", {"command": "ls", "timeout": 120}
            )
        assert "timeout" not in result

    # --- ls ---

    def test_ls_with_path(self) -> None:
        result = format_tool_display("ls", {"path": "/tmp"})
        assert "ls(" in result
        assert "tmp" in result

    def test_ls_without_path(self) -> None:
        result = format_tool_display("ls", {})
        assert "ls()" in result

    # --- glob ---

    def test_glob_shows_pattern(self) -> None:
        result = format_tool_display("glob", {"pattern": "**/*.py"})
        assert 'glob("**/*.py")' in result

    # --- fetch_url ---

    def test_fetch_url_shows_url(self) -> None:
        result = format_tool_display("fetch_url", {"url": "https://example.com"})
        assert 'fetch_url("https://example.com")' in result

    # --- task ---

    def test_task_with_subagent_type(self) -> None:
        result = format_tool_display("task", {"subagent_type": "code-review"})
        assert "task [code-review]" in result

    def test_task_without_subagent_type(self) -> None:
        result = format_tool_display("task", {})
        assert result.endswith("task")

    # --- ask_user ---

    def test_ask_user_singular(self) -> None:
        result = format_tool_display("ask_user", {"questions": ["What?"]})
        assert "1 question" in result

    def test_ask_user_plural(self) -> None:
        result = format_tool_display("ask_user", {"questions": ["Q1", "Q2", "Q3"]})
        assert "3 questions" in result

    def test_ask_user_missing_questions_falls_back(self) -> None:
        result = format_tool_display("ask_user", {})
        assert "ask_user" in result

    # --- compact_conversation ---

    def test_compact_conversation(self) -> None:
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result

    # --- write_todos ---

    def test_write_todos_shows_count(self) -> None:
        result = format_tool_display(
            "write_todos", {"todos": ["task1", "task2", "task3"]}
        )
        assert "3 items" in result

    # --- generic fallback ---

    def test_unknown_tool_generic_fallback(self) -> None:
        result = format_tool_display("custom_tool", {"key": "value"})
        assert "custom_tool" in result
        assert "key" in result
        assert "value" in result

    def test_unknown_tool_no_args(self) -> None:
        result = format_tool_display("my_tool", {})
        assert "my_tool()" in result

    # --- Unicode sanitization in tool args ---

    def test_hidden_unicode_in_command_stripped(self) -> None:
        with patch("deepagents.backends.DEFAULT_EXECUTE_TIMEOUT", 120):
            result = format_tool_display(
                "execute", {"command": "echo he\u200bllo"}
            )
        assert _HIDDEN_CHAR_MARKER in result

    def test_hidden_unicode_in_file_path_stripped(self) -> None:
        result = format_tool_display(
            "read_file", {"file_path": "/tmp/fi\u200ble.py"}
        )
        assert _HIDDEN_CHAR_MARKER in result


# ---------------------------------------------------------------------------
# _format_content_block
# ---------------------------------------------------------------------------


class TestFormatContentBlock:
    """Tests for _format_content_block()."""

    def test_image_block_with_base64(self) -> None:
        # ~100 bytes of base64 → ~75 decoded bytes → 0 KB
        b64 = "A" * 100
        result = _format_content_block(
            {"type": "image", "base64": b64, "mime_type": "image/png"}
        )
        assert result.startswith("[Image: image/png")
        assert "KB]" in result

    def test_video_block_with_base64(self) -> None:
        b64 = "A" * 400
        result = _format_content_block(
            {"type": "video", "base64": b64, "mime_type": "video/mp4"}
        )
        assert result.startswith("[Video: video/mp4")

    def test_file_block_with_base64(self) -> None:
        b64 = "A" * 400
        result = _format_content_block(
            {"type": "file", "base64": b64, "mime_type": "application/pdf"}
        )
        assert result.startswith("[File: application/pdf")

    def test_image_block_missing_mime_defaults(self) -> None:
        result = _format_content_block({"type": "image", "base64": "AAAA"})
        assert "[Image: image," in result

    def test_plain_dict_serialized_as_json(self) -> None:
        result = _format_content_block({"type": "text", "text": "hello"})
        assert "hello" in result

    def test_non_serialisable_falls_back_to_str(self) -> None:
        obj = object()
        result = _format_content_block({"type": "custom", "data": obj})
        # Falls back to str() since object() is not JSON serialisable
        assert isinstance(result, str)

    def test_preserves_non_ascii_in_json(self) -> None:
        result = _format_content_block({"type": "text", "text": "日本語"})
        assert "日本語" in result


# ---------------------------------------------------------------------------
# format_tool_message_content
# ---------------------------------------------------------------------------


class TestFormatToolMessageContent:
    """Tests for format_tool_message_content()."""

    def test_none_returns_empty_string(self) -> None:
        assert format_tool_message_content(None) == ""

    def test_plain_string_returned_as_is(self) -> None:
        assert format_tool_message_content("ok") == "ok"

    def test_integer_coerced_to_string(self) -> None:
        assert format_tool_message_content(42) == "42"

    def test_list_of_strings_joined_by_newline(self) -> None:
        result = format_tool_message_content(["line1", "line2", "line3"])
        assert result == "line1\nline2\nline3"

    def test_list_with_dict_items_serialized(self) -> None:
        result = format_tool_message_content([{"type": "text", "text": "hi"}])
        assert "hi" in result

    def test_list_with_image_block_shows_placeholder(self) -> None:
        result = format_tool_message_content(
            [{"type": "image", "base64": "AAAA", "mime_type": "image/png"}]
        )
        assert "[Image:" in result

    def test_mixed_list_string_and_dict(self) -> None:
        result = format_tool_message_content(
            ["prefix", {"type": "text", "text": "body"}]
        )
        assert "prefix" in result
        assert "body" in result

    def test_list_with_non_serialisable_item(self) -> None:
        result = format_tool_message_content([object()])
        assert isinstance(result, str)

    def test_preserves_non_ascii(self) -> None:
        assert format_tool_message_content("日本語") == "日本語"

    def test_empty_list_returns_empty_string(self) -> None:
        assert format_tool_message_content([]) == ""