"""Tests for exception handling improvements in CLI modules.

These tests verify that:
1. Exceptions are properly logged at DEBUG level
2. Specific exception types are caught instead of bare Exception
3. The code behaves correctly when exceptions occur
4. Tavily-specific exceptions are handled in web_search
"""

import ast
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from tavily import BadRequestError, InvalidAPIKeyError, UsageLimitExceededError
from tavily.errors import TimeoutError as TavilyTimeoutError

from deepagents_cli.clipboard import (
    copy_selection_to_clipboard,
    logger as clipboard_logger,
)
from deepagents_cli.file_ops import FileOpTracker, _safe_read
from deepagents_cli.image_utils import (
    _get_clipboard_via_osascript,
    _get_macos_clipboard_image,
    logger as image_utils_logger,
)
from deepagents_cli.tools import http_request, web_search


class TestToolsExceptionHandling:
    """Test exception handling in CLI tools."""

    def test_http_request_handles_json_decode_error(self):
        """Test that http_request catches JSONDecodeError properly."""
        # Mock a response that returns invalid JSON
        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.url = "http://example.com"
            mock_response.text = "not valid json"
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_request.return_value = mock_response

            result = http_request("http://example.com")

        # Should succeed and return text content
        assert result["success"] is True
        assert result["content"] == "not valid json"

    def test_http_request_handles_requests_json_decode_error(self):
        """Test that http_request also catches requests.exceptions.JSONDecodeError."""
        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.url = "http://example.com"
            mock_response.text = "plain text response"
            mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
                "Expecting value", "doc", 0
            )
            mock_request.return_value = mock_response

            result = http_request("http://example.com")

        assert result["success"] is True
        assert result["content"] == "plain text response"

    def test_web_search_handles_tavily_usage_limit_error(self):
        """Test that web_search catches Tavily UsageLimitExceededError."""
        mock_client = MagicMock()
        mock_client.search.side_effect = UsageLimitExceededError("Rate limit")
        with patch("deepagents_cli.tools._get_tavily_client", return_value=mock_client):
            result = web_search("test query")

        assert "error" in result
        assert "Rate limit" in result["error"]
        assert result["query"] == "test query"

    def test_web_search_handles_tavily_invalid_api_key(self):
        """Test that web_search catches Tavily InvalidAPIKeyError."""
        mock_client = MagicMock()
        mock_client.search.side_effect = InvalidAPIKeyError("Invalid key")
        with patch("deepagents_cli.tools._get_tavily_client", return_value=mock_client):
            result = web_search("test query")

        assert "error" in result
        assert "Invalid key" in result["error"]

    def test_web_search_handles_tavily_bad_request(self):
        """Test that web_search catches Tavily BadRequestError."""
        mock_client = MagicMock()
        mock_client.search.side_effect = BadRequestError("Bad request")
        with patch("deepagents_cli.tools._get_tavily_client", return_value=mock_client):
            result = web_search("test query")

        assert "error" in result
        assert "Bad request" in result["error"]

    def test_web_search_handles_tavily_timeout(self):
        """Test that web_search catches Tavily TimeoutError."""
        mock_client = MagicMock()
        mock_client.search.side_effect = TavilyTimeoutError(30.0)
        with patch("deepagents_cli.tools._get_tavily_client", return_value=mock_client):
            result = web_search("test query")

        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestFileOpsExceptionHandling:
    """Test exception handling in file_ops."""

    def test_file_op_tracker_handles_backend_failure(self, caplog):
        """Test that FileOpTracker logs backend failures."""
        # Create tracker with a mock backend that fails
        mock_backend = MagicMock()
        mock_backend.download_files.side_effect = OSError("Backend error")

        tracker = FileOpTracker(assistant_id=None, backend=mock_backend)

        with caplog.at_level(logging.DEBUG):
            tracker.start_operation(
                "write_file",
                {"file_path": "/test.txt", "content": "test"},
                "tool_call_123",
            )

        # Should have recorded the operation (with empty before_content due to failure)
        assert "tool_call_123" in tracker.active
        record = tracker.active["tool_call_123"]
        assert record.before_content == ""

        # Verify the error was logged
        assert "Failed to read before_content" in caplog.text
        assert "Backend error" in caplog.text

    def test_file_op_tracker_handles_attribute_error(self, caplog):
        """Test that FileOpTracker handles AttributeError properly."""
        # Create tracker with a mock backend that raises AttributeError
        mock_backend = MagicMock()
        mock_backend.download_files.side_effect = AttributeError("Missing attribute")

        tracker = FileOpTracker(assistant_id=None, backend=mock_backend)

        with caplog.at_level(logging.DEBUG):
            tracker.start_operation(
                "edit_file",
                {"file_path": "/test.txt", "old_string": "a", "new_string": "b"},
                "tool_call_456",
            )

        # Should have recorded the operation with empty before_content
        assert "tool_call_456" in tracker.active
        record = tracker.active["tool_call_456"]
        assert record.before_content == ""

        # Verify the error was logged
        assert "Failed to read before_content" in caplog.text
        assert "Missing attribute" in caplog.text

    def test_file_op_tracker_handles_unicode_decode_error(self, caplog):
        """Test that FileOpTracker handles UnicodeDecodeError for binary files."""
        # Create tracker with a mock backend that returns binary data
        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"\xff\xfe\x00\x01"  # Invalid UTF-8
        mock_response.error = None
        mock_backend.download_files.return_value = [mock_response]

        tracker = FileOpTracker(assistant_id=None, backend=mock_backend)

        with caplog.at_level(logging.DEBUG):
            tracker.start_operation(
                "write_file",
                {"file_path": "/test.bin", "content": "test"},
                "tool_call_789",
            )

        # Should have recorded the operation with empty before_content
        assert "tool_call_789" in tracker.active
        record = tracker.active["tool_call_789"]
        assert record.before_content == ""

        # Verify the error was logged
        assert "Failed to read before_content" in caplog.text

    def test_safe_read_logs_on_failure(self, caplog, tmp_path):
        """Test that _safe_read logs when file read fails."""
        # Test with non-existent file
        nonexistent = tmp_path / "does_not_exist.txt"

        with caplog.at_level(logging.DEBUG):
            result = _safe_read(nonexistent)

        assert result is None
        assert "Failed to read file" in caplog.text


class TestClipboardExceptionHandling:
    """Test exception handling in clipboard utilities."""

    def test_copy_handles_widget_selection_failures(self, caplog):
        """Test that copy_selection_to_clipboard handles widget failures gracefully."""
        # Create a mock app with widgets
        mock_app = MagicMock()
        mock_widget = MagicMock()
        mock_widget.text_selection = MagicMock()
        mock_widget.get_selection.side_effect = AttributeError("No selection")

        mock_app.query.return_value = [mock_widget]

        with caplog.at_level(logging.DEBUG):
            # Should not raise
            copy_selection_to_clipboard(mock_app)

        # Verify the error was logged
        assert "Failed to get selection from widget" in caplog.text
        assert "No selection" in caplog.text

    def test_clipboard_logger_exists(self):
        """Test that clipboard module has proper logging configured."""
        assert clipboard_logger is not None
        assert clipboard_logger.name == "deepagents_cli.clipboard"


class TestImageUtilsExceptionHandling:
    """Test exception handling in image utilities."""

    def test_image_utils_logger_exists(self):
        """Test that image_utils module has proper logging configured."""
        assert image_utils_logger is not None
        assert image_utils_logger.name == "deepagents_cli.image_utils"

    def test_image_utils_exception_types(self):
        """Test that image_utils uses proper exception types."""
        # Read the source file and check exception handling
        source_path = (
            Path(__file__).parent.parent.parent / "deepagents_cli" / "image_utils.py"
        )
        source = source_path.read_text()
        tree = ast.parse(source)

        # Find all except handlers - bare excepts have type=None
        bare_excepts = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]

        # Should have no bare excepts after our fix
        assert len(bare_excepts) == 0, f"Found bare except at lines: {bare_excepts}"

    def test_pngpaste_timeout_logs_and_returns_none(self, caplog):
        """Test that pngpaste timeout is logged and function falls back."""
        with (
            patch("deepagents_cli.image_utils._get_executable") as mock_exec,
            patch("subprocess.run") as mock_run,
            patch(
                "deepagents_cli.image_utils._get_clipboard_via_osascript"
            ) as mock_osascript,
        ):
            mock_exec.return_value = "/usr/local/bin/pngpaste"
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="pngpaste", timeout=2)
            mock_osascript.return_value = None

            with caplog.at_level(logging.DEBUG):
                result = _get_macos_clipboard_image()

            assert result is None
            assert "pngpaste timed out" in caplog.text

    def test_pngpaste_not_found_logs_and_falls_back(self, caplog):
        """Test that FileNotFoundError for pngpaste is logged."""
        with (
            patch("deepagents_cli.image_utils._get_executable") as mock_exec,
            patch("subprocess.run") as mock_run,
            patch(
                "deepagents_cli.image_utils._get_clipboard_via_osascript"
            ) as mock_osascript,
        ):
            mock_exec.return_value = "/usr/local/bin/pngpaste"
            mock_run.side_effect = FileNotFoundError("pngpaste")
            mock_osascript.return_value = None

            with caplog.at_level(logging.DEBUG):
                result = _get_macos_clipboard_image()

            assert result is None
            assert "pngpaste not found" in caplog.text

    def test_osascript_timeout_logs_and_returns_none(self, caplog):
        """Test that osascript timeout is logged."""
        with (
            patch("deepagents_cli.image_utils._get_executable") as mock_exec,
            patch("subprocess.run") as mock_run,
            patch("tempfile.mkstemp") as mock_mkstemp,
            patch("os.close"),
        ):
            mock_exec.return_value = "/usr/bin/osascript"
            mock_mkstemp.return_value = (5, "/tmp/test.png")
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=2)

            with caplog.at_level(logging.DEBUG):
                result = _get_clipboard_via_osascript()

            assert result is None
            assert "osascript timed out" in caplog.text
