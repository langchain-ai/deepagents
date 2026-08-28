"""Tests for exception handling improvements in CLI modules.

These tests verify that:
1. Exceptions are properly logged at DEBUG level
2. Specific exception types are caught instead of bare Exception
3. The code behaves correctly when exceptions occur
4. Tavily-specific exceptions are handled in web_search
"""

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from tavily import BadRequestError, InvalidAPIKeyError, UsageLimitExceededError
from tavily.errors import TimeoutError as TavilyTimeoutError

from deepagents_code.file_ops import FileOpTracker, _safe_read
from deepagents_code.media_utils import (
    _get_clipboard_via_osascript,
    _get_macos_clipboard_image,
)
from deepagents_code.tools import web_search


class TestToolsExceptionHandling:
    """Test exception handling in CLI tools."""


class TestFileOpsExceptionHandling:
    """Test exception handling in file_ops."""

    def test_file_op_tracker_handles_backend_failure(self, caplog):
        """Test that FileOpTracker logs backend failures."""
        # Create tracker with a mock backend that fails
        mock_backend = MagicMock()
        mock_backend.download_files.side_effect = OSError("Backend error")

        tracker = FileOpTracker(assistant_id=None, backend=mock_backend)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            tracker.start_operation(
                "write_file",
                {"file_path": "/test.txt", "content": "test"},
                "tool_call_123",
            )

        # Should have recorded the operation (with empty before_content due to failure)
        assert "tool_call_123" in tracker.active
        record = tracker.active["tool_call_123"]
        assert record.before_content == ""
        # The empty string is a stand-in, not the file's real prior state; the
        # flag is what stops downstream renderers presenting it as fact.
        assert record.diff_outcome == "untrusted_before"

        # Verify the error was logged loudly enough to notice in the field.
        assert "Could not read pre-edit content" in caplog.text
        assert "Backend error" in caplog.text
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_file_op_tracker_handles_unicode_decode_error(self, caplog):
        """Test that FileOpTracker handles UnicodeDecodeError for binary files."""
        # Create tracker with a mock backend that returns binary data
        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b"\xff\xfe\x00\x01"  # Invalid UTF-8
        mock_response.error = None
        mock_backend.download_files.return_value = [mock_response]

        tracker = FileOpTracker(assistant_id=None, backend=mock_backend)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            tracker.start_operation(
                "write_file",
                {"file_path": "/test.bin", "content": "test"},
                "tool_call_789",
            )

        # Should have recorded the operation with empty before_content
        assert "tool_call_789" in tracker.active
        record = tracker.active["tool_call_789"]
        assert record.before_content == ""
        # A binary pre-image is unreadable, not empty — the diff must not
        # present the write as if it created the file from nothing.
        assert record.diff_outcome == "untrusted_before"

        # Verify the error was logged
        assert "Could not read pre-edit content" in caplog.text


class TestMediaUtilsExceptionHandling:
    """Test exception handling in media utilities."""
