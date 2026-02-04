"""Tests for exception handling improvements in CLI modules.

These tests verify that:
1. Silent exceptions are now properly logged
2. Specific exception types are caught instead of bare Exception
3. The code behaves correctly when exceptions occur
"""

import logging
from unittest.mock import MagicMock, patch

import pytest


class TestToolsExceptionHandling:
    """Test exception handling in CLI tools."""

    def test_http_request_handles_json_decode_error(self):
        """Test that http_request catches JSONDecodeError properly."""
        from deepagents_cli.tools import http_request

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
        import requests

        from deepagents_cli.tools import http_request

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


class TestFileOpsExceptionHandling:
    """Test exception handling in file_ops."""

    def test_file_op_tracker_handles_backend_failure(self, caplog):
        """Test that FileOpTracker logs backend failures."""
        from deepagents_cli.file_ops import FileOpTracker

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
        from deepagents_cli.file_ops import FileOpTracker

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


class TestClipboardExceptionHandling:
    """Test exception handling in clipboard utilities."""

    def test_copy_handles_widget_selection_failures(self, caplog):
        """Test that copy_selection_to_clipboard handles widget failures gracefully."""
        from deepagents_cli.clipboard import copy_selection_to_clipboard

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
        from deepagents_cli.clipboard import logger

        assert logger is not None
        assert logger.name == "deepagents_cli.clipboard"


class TestImageUtilsExceptionHandling:
    """Test exception handling in image utilities."""

    def test_image_utils_logger_exists(self):
        """Test that image_utils module has proper logging configured."""
        from deepagents_cli.image_utils import logger

        assert logger is not None
        assert logger.name == "deepagents_cli.image_utils"

    def test_image_utils_exception_types(self):
        """Test that image_utils uses proper exception types."""
        import ast
        from pathlib import Path

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
