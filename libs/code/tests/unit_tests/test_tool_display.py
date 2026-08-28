"""Unit tests for deepagents_code/tool_display.py.

All functions under test are pure (no I/O, no async, no TUI). A single
module-level autouse fixture pins `get_glyphs()` to `ASCII_GLYPHS` so
assertions are deterministic regardless of terminal configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT

from deepagents_code.config import ASCII_GLYPHS, MAX_ARG_LENGTH
from deepagents_code.tool_display import (
    _HIDDEN_CHAR_MARKER,
    _coerce_timeout_seconds,
    _format_content_block,
    _format_timeout,
    _sanitize_display_value,
    format_tool_display,
    format_tool_message_content,
    truncate_value,
)

_PREFIX = ASCII_GLYPHS.tool_prefix
_ELLIPSIS = ASCII_GLYPHS.ellipsis


@pytest.fixture(autouse=True)
def _pin_ascii_glyphs() -> Generator[None, None, None]:
    with patch("deepagents_code.tool_display.get_glyphs", return_value=ASCII_GLYPHS):
        yield


# ---------------------------------------------------------------------------
# _format_timeout
# ---------------------------------------------------------------------------


class TestFormatTimeout:
    """Tests for _format_timeout()."""


# ---------------------------------------------------------------------------
# _coerce_timeout_seconds
# ---------------------------------------------------------------------------


class TestCoerceTimeoutSeconds:
    """Tests for _coerce_timeout_seconds()."""


# ---------------------------------------------------------------------------
# truncate_value
# ---------------------------------------------------------------------------


class TestTruncateValue:
    """Tests for truncate_value()."""


# ---------------------------------------------------------------------------
# _sanitize_display_value
# ---------------------------------------------------------------------------


class TestSanitizeDisplayValue:
    """Tests for _sanitize_display_value()."""

    def test_hidden_unicode_stripped_and_marker_appended(self) -> None:
        # U+200B is a zero-width space — stripped by strip_dangerous_unicode.
        result = _sanitize_display_value("hello\u200bworld")
        assert "helloworld" in result
        assert _HIDDEN_CHAR_MARKER in result


# ---------------------------------------------------------------------------
# format_tool_display — per-tool branches
# ---------------------------------------------------------------------------


class TestFormatToolDisplay:
    """Tests for format_tool_display()."""

    # --- file tools ---

    @pytest.mark.parametrize(
        "tool_name", ["read_file", "write_file", "edit_file", "delete"]
    )
    def test_file_tool_with_file_path(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {"file_path": "/tmp/test.py"})
        assert result.startswith(_PREFIX)
        assert tool_name in result
        assert "test.py" in result

    @pytest.mark.parametrize(
        "tool_name", ["read_file", "write_file", "edit_file", "delete"]
    )
    def test_file_tool_with_path_key(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {"path": "/tmp/test.py"})
        assert "test.py" in result

    @pytest.mark.parametrize(
        "tool_name", ["read_file", "write_file", "edit_file", "delete"]
    )
    def test_file_tool_missing_path_falls_back_to_generic(self, tool_name: str) -> None:
        result = format_tool_display(tool_name, {})
        assert _PREFIX in result
        assert tool_name in result

    # --- web_search ---

    # --- grep ---

    def test_grep_shows_scoped_path(self) -> None:
        abs_path = str(Path.cwd() / "subdir")
        result = format_tool_display("grep", {"pattern": "def foo", "path": abs_path})
        assert 'grep("def foo" in subdir)' in result

    def test_grep_omits_default_root_path(self) -> None:
        result = format_tool_display("grep", {"pattern": "def foo", "path": "/"})
        assert 'grep("def foo")' in result
        assert " in " not in result

    def test_grep_omits_empty_path(self) -> None:
        result = format_tool_display("grep", {"pattern": "def foo", "path": ""})
        assert 'grep("def foo")' in result
        assert " in " not in result

    def test_grep_omits_none_path(self) -> None:
        result = format_tool_display("grep", {"pattern": "def foo", "path": None})
        assert 'grep("def foo")' in result
        assert " in " not in result

    def test_grep_shows_out_of_cwd_path(self) -> None:
        # A path outside cwd cannot be made relative; it must still render.
        result = format_tool_display(
            "grep", {"pattern": "def foo", "path": "/etc/nginx"}
        )
        assert " in /etc/nginx" in result

    def test_grep_scoped_path_strips_dangerous_unicode(self) -> None:
        # A zero-width space in the path is stripped and flagged for the user.
        abs_path = str(Path.cwd() / "subdir") + "\u200b"
        result = format_tool_display("grep", {"pattern": "def foo", "path": abs_path})
        assert " in subdir" in result
        assert _HIDDEN_CHAR_MARKER in result

    # --- execute ---

    # --- js_eval ---

    # --- ls ---

    # --- glob ---

    def test_glob_shows_scoped_path(self) -> None:
        abs_path = str(Path.cwd() / "subdir")
        result = format_tool_display("glob", {"pattern": "**/*.py", "path": abs_path})
        assert 'glob("**/*.py" in subdir)' in result

    def test_glob_omits_default_root_path(self) -> None:
        result = format_tool_display("glob", {"pattern": "**/*.py", "path": "/"})
        assert 'glob("**/*.py")' in result
        assert " in " not in result

    def test_glob_distinguishes_scoped_from_unscoped(self) -> None:
        # The two calls from the LangSmith trace must render differently.
        unscoped = format_tool_display("glob", {"pattern": "**/*.py"})
        scoped = format_tool_display(
            "glob", {"pattern": "**/*.py", "path": str(Path.cwd() / "langchain")}
        )
        assert unscoped != scoped

    def test_glob_omits_empty_path(self) -> None:
        result = format_tool_display("glob", {"pattern": "**/*.py", "path": ""})
        assert 'glob("**/*.py")' in result
        assert " in " not in result

    def test_glob_omits_none_path(self) -> None:
        result = format_tool_display("glob", {"pattern": "**/*.py", "path": None})
        assert 'glob("**/*.py")' in result
        assert " in " not in result

    def test_glob_shows_out_of_cwd_path(self) -> None:
        # A path outside cwd cannot be made relative; it must still render.
        result = format_tool_display(
            "glob", {"pattern": "**/*.py", "path": "/etc/nginx"}
        )
        assert " in /etc/nginx" in result

    def test_glob_scoped_path_strips_dangerous_unicode(self) -> None:
        # A zero-width space in the path is stripped and flagged for the user.
        abs_path = str(Path.cwd() / "subdir") + "\u200b"
        result = format_tool_display("glob", {"pattern": "**/*.py", "path": abs_path})
        assert " in subdir" in result
        assert _HIDDEN_CHAR_MARKER in result

    # --- fetch_url ---

    # --- task ---

    # --- ask_user ---

    # --- compact_conversation ---

    # --- write_todos ---

    # --- generic fallback ---

    # --- Unicode sanitization in tool args ---


# ---------------------------------------------------------------------------
# _format_content_block
# ---------------------------------------------------------------------------


class TestFormatContentBlock:
    """Tests for _format_content_block()."""

    @pytest.mark.parametrize(
        ("b64_len", "expected_kb"),
        [
            pytest.param(100, 0, id="sub-kb-rounds-down"),
            pytest.param(1400, 1, id="just-over-1kb"),
            pytest.param(8192, 6, id="8kb-payload"),
        ],
    )
    def test_image_block_size_formula(self, b64_len: int, expected_kb: int) -> None:
        # size_kb = len(b64) * 3 // 4 // 1024 (approx decoded size).
        result = _format_content_block(
            {"type": "image", "base64": "A" * b64_len, "mime_type": "image/png"}
        )
        assert result == f"[Image: image/png, ~{expected_kb}KB]"

    def test_video_block_with_base64(self) -> None:
        result = _format_content_block(
            {"type": "video", "base64": "A" * 400, "mime_type": "video/mp4"}
        )
        assert result.startswith("[Video: video/mp4")

    def test_file_block_with_base64(self) -> None:
        result = _format_content_block(
            {"type": "file", "base64": "A" * 400, "mime_type": "application/pdf"}
        )
        assert result.startswith("[File: application/pdf")

    def test_image_block_missing_mime_defaults(self) -> None:
        result = _format_content_block({"type": "image", "base64": "AAAA"})
        assert "[Image: image," in result

    @pytest.mark.parametrize(
        ("block", "expected_fragment"),
        [
            ({"type": "image", "base64": 123}, '"base64": 123'),
            (
                {"type": "image", "url": "https://example.com/image.png"},
                '"url": "https://example.com/image.png"',
            ),
        ],
    )
    def test_image_block_without_string_base64_falls_through_to_json(
        self, block: dict[str, object], expected_fragment: str
    ) -> None:
        result = _format_content_block(block)
        assert "[Image" not in result
        assert '"type": "image"' in result
        assert expected_fragment in result

    def test_plain_dict_serialized_as_json(self) -> None:
        result = _format_content_block({"type": "text", "text": "hello"})
        assert "hello" in result

    def test_non_serializable_falls_back_to_str(self) -> None:
        obj = object()
        result = _format_content_block({"type": "custom", "data": obj})
        # json.dumps raises TypeError for `object()` → falls back to `str(block)`,
        # which renders the repr including "object at 0x...".
        assert "object" in result

    def test_preserves_non_ascii_in_json(self) -> None:
        result = _format_content_block({"type": "text", "text": "日本語"})
        assert "日本語" in result


# ---------------------------------------------------------------------------
# format_tool_message_content
# ---------------------------------------------------------------------------


class TestFormatToolMessageContent:
    """Tests for format_tool_message_content()."""

    def test_list_with_image_block_shows_placeholder(self) -> None:
        result = format_tool_message_content(
            [{"type": "image", "base64": "A" * 4000, "mime_type": "image/png"}]
        )
        assert "[Image:" in result
        assert "AAAA" not in result
