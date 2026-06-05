"""Tests for _debug.configure_debug_logging."""

from __future__ import annotations

import importlib
import logging
import os
from unittest.mock import patch

import deepagents_code
from deepagents_code._debug import configure_debug_logging


class TestConfigureDebugLogging:
    def test_noop_when_env_unset(self) -> None:
        """No handlers should be added when DEEPAGENTS_CODE_DEBUG is unset."""
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
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
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
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
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

    def test_repeated_configuration_is_idempotent(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.idempotent")
        log_file = tmp_path / "debug.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
        ):
            configure_debug_logging(logger)
            configure_debug_logging(logger)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            assert len(file_handlers) == 1
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_changed_path_swaps_handler(self, tmp_path) -> None:
        """Re-configuring with a new path replaces the stale handler, not stacks."""
        logger = logging.getLogger("test.debug.swap")
        first = tmp_path / "first.log"
        second = tmp_path / "second.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(first)},
        ):
            configure_debug_logging(logger)
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(second)},
        ):
            configure_debug_logging(logger)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            assert len(file_handlers) == 1
            assert str(second) in file_handlers[0].baseFilename
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_untagged_handler_does_not_block_configuration(self, tmp_path) -> None:
        """A foreign FileHandler on the same path must not suppress our handler."""
        logger = logging.getLogger("test.debug.untagged")
        log_file = tmp_path / "debug.log"
        foreign = logging.FileHandler(str(log_file), mode="a")
        logger.addHandler(foreign)
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
        ):
            configure_debug_logging(logger)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            # Both the pre-existing foreign handler and our tagged handler remain.
            assert foreign in file_handlers
            assert any(
                getattr(h, "_deepagents_code_debug_handler", False)
                for h in file_handlers
            )
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_child_logger_propagates_to_configured_parent(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.parent")
        child = logging.getLogger("test.debug.parent.child")
        log_file = tmp_path / "debug.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
        ):
            configure_debug_logging(logger)

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            child.warning("child warning")
            for h in file_handlers:
                h.flush()
            assert "test.debug.parent.child child warning" in log_file.read_text()
        finally:
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
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_FILE": "/nonexistent_dir/debug.log",
            },
        ):
            configure_debug_logging(logger)
        assert len(logger.handlers) == original_count
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_package_import_configures_package_logger(self, tmp_path) -> None:
        logger = logging.getLogger("deepagents_code")
        original_handlers = list(logger.handlers)
        original_level = logger.level
        log_file = tmp_path / "package.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG": "1", "DEEPAGENTS_CODE_DEBUG_FILE": str(log_file)},
        ):
            importlib.reload(deepagents_code)

        new_handlers = [h for h in logger.handlers if h not in original_handlers]
        try:
            child = logging.getLogger("deepagents_code.test_child")
            child.warning("package child warning")
            for h in new_handlers:
                h.flush()
            assert "deepagents_code.test_child package child warning" in (
                log_file.read_text()
            )
        finally:
            for h in new_handlers:
                h.close()
                logger.removeHandler(h)
            logger.setLevel(original_level)
            # Reload with the debug env cleared so cleanup never re-attaches a
            # handler to the real package logger (e.g. when a developer runs the
            # suite with DEEPAGENTS_CODE_DEBUG exported in their shell).
            with patch.dict(os.environ, {}, clear=True):
                importlib.reload(deepagents_code)
