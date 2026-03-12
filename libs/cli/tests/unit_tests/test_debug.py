"""Tests for _debug.configure_debug_logging."""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

from deepagents_cli._debug import configure_debug_logging


class TestConfigureDebugLogging:
    def test_noop_when_env_unset(self) -> None:
        """No handlers should be added when DEEPAGENTS_DEBUG is unset."""
        logger = logging.getLogger("test.debug.noop")
        original_count = len(logger.handlers)
        with patch.dict(os.environ, {}, clear=True):
            configure_debug_logging(logger)
        assert len(logger.handlers) == original_count

    def test_adds_handler_when_env_set(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.add")
        log_file = tmp_path / "debug.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_DEBUG": "1", "DEEPAGENTS_DEBUG_FILE": str(log_file)},
        ):
            configure_debug_logging(logger)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert logger.level == logging.DEBUG
        # Cleanup
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)

    def test_custom_path_used(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.custom_path")
        log_file = tmp_path / "custom.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_DEBUG": "1", "DEEPAGENTS_DEBUG_FILE": str(log_file)},
        ):
            configure_debug_logging(logger)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1
        assert str(log_file) in file_handlers[-1].baseFilename
        # Cleanup
        for h in file_handlers:
            h.close()
            logger.removeHandler(h)

    def test_bad_path_prints_warning_no_crash(self, capsys) -> None:
        """Invalid log path should print warning to stderr, not crash."""
        logger = logging.getLogger("test.debug.bad_path")
        original_count = len(logger.handlers)
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_DEBUG": "1",
                "DEEPAGENTS_DEBUG_FILE": "/nonexistent_dir/debug.log",
            },
        ):
            configure_debug_logging(logger)
        assert len(logger.handlers) == original_count
        captured = capsys.readouterr()
        assert "Warning" in captured.err
