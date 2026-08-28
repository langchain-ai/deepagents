"""Tests for _debug.configure_debug_logging."""

from __future__ import annotations

import importlib
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import deepagents_code
from deepagents_code import _debug
from deepagents_code._debug import (
    bind_debug_logging_to_thread,
    configure_debug_logging,
    installed_debug_log_path,
    resolve_log_level,
)


def _icacls_entries(path) -> list[str]:
    """Return one string per access-control entry on `path`, via `icacls`.

    `icacls` prints `<path> <first ACE>`, then one indented ACE per line, then
    a blank line and a summary. Only the ACE text is returned.
    """
    completed = subprocess.run(
        ["icacls", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    head = completed.stdout.split("\n\n")[0]
    first, *rest = head.splitlines()
    entries = [first.removeprefix(str(path)), *rest]
    return [entry.strip() for entry in entries if entry.strip()]


class TestResolveLogLevel:
    def test_defaults_to_debug_when_debug_enabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_log_level(debug_enabled=True) == logging.DEBUG

    def test_defaults_to_info_when_debug_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_log_level(debug_enabled=False) == logging.INFO

    def test_empty_value_falls_back(self) -> None:
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": ""}, clear=True):
            assert resolve_log_level(debug_enabled=False) == logging.INFO

    def test_whitespace_value_falls_back(self) -> None:
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "   "}, clear=True):
            assert resolve_log_level(debug_enabled=True) == logging.DEBUG

    def test_value_is_case_insensitive(self) -> None:
        with patch.dict(
            os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "warning"}, clear=True
        ):
            assert resolve_log_level(debug_enabled=False) == logging.WARNING

    def test_explicit_level_overrides_debug_fallback(self) -> None:
        """An explicit level wins over the debug-derived default."""
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "ERROR"}, clear=True):
            assert resolve_log_level(debug_enabled=True) == logging.ERROR

    def test_reads_debug_env_when_flag_omitted(self) -> None:
        """With no explicit flag, the truthiness of the env var decides."""
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_DEBUG": "1"}, clear=True):
            assert resolve_log_level() == logging.DEBUG
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_log_level() == logging.INFO


class TestConfigureDebugLogging:
    def test_noop_when_env_unset(self) -> None:
        """No handlers should be added when DEEPAGENTS_CODE_DEBUG is unset."""
        logger = logging.getLogger("test.debug.noop")
        original_count = len(logger.handlers)
        with patch.dict(os.environ, {}, clear=True):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert len(logger.handlers) == original_count

    def test_adds_handler_when_env_set(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.add")
        log_file = tmp_path / "test-thread.log"
        log_file.touch(mode=0o644)
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert logger.level == logging.DEBUG
        if os.name != "nt":
            assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
        # Cleanup
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL hardening")
    def test_debug_file_dacl_grants_current_user_only(self, tmp_path) -> None:
        """On Windows the debug file DACL is restricted to the current user.

        A file that inherits its parent's DACL carries several entries
        (`SYSTEM`, `Administrators`, the user). Exactly one entry, naming the
        current user, is what proves the replacement DACL was applied and
        marked protected so inherited entries were dropped.
        """
        log_file = tmp_path / "test-thread.log"
        log_file.touch()

        _debug._prepare_debug_file(log_file)

        aces = _icacls_entries(log_file)
        assert len(aces) == 1, f"expected a single ACE, got {aces}"
        assert os.environ["USERNAME"].lower() in aces[0].lower()

    def test_no_file_handler_when_hardening_fails(self, tmp_path, capsys) -> None:
        """A file that cannot be secured gets no handler at all.

        Captured MCP server stderr can carry credentials, so failing to
        restrict the file must disable file logging rather than fall through
        and write to it anyway.
        """
        logger = logging.getLogger("test.debug.harden_fail")
        logger.handlers = []
        log_file = tmp_path / "test-thread.log"
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPAGENTS_CODE_DEBUG": "1",
                    "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
                },
            ),
            patch.object(_debug, "_prepare_debug_file", side_effect=OSError("nope")),
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert "Warning" in capsys.readouterr().err

    @pytest.mark.skipif(os.name == "nt", reason="POSIX O_NOFOLLOW refusal")
    def test_symlinked_debug_directory_is_refused(self, tmp_path, capsys) -> None:
        logger = logging.getLogger("test.debug.symlink_directory")
        logger.handlers = []
        directory = tmp_path / "directory"
        directory.mkdir()
        link = tmp_path / "link"
        link.symlink_to(directory, target_is_directory=True)
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(link),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert "Warning" in capsys.readouterr().err
        assert list(directory.iterdir()) == []

    @pytest.mark.skipif(os.name == "nt", reason="POSIX O_NOFOLLOW refusal")
    def test_symlinked_debug_file_is_refused(self, tmp_path, capsys) -> None:
        """A symlink at the debug path is refused, not followed.

        `/tmp` is the default location, so a planted symlink would otherwise
        redirect captured MCP stderr into a file the attacker chose.
        """
        logger = logging.getLogger("test.debug.symlink")
        logger.handlers = []
        victim = tmp_path / "victim.log"
        victim.touch()
        link = tmp_path / "test-thread.log"
        link.symlink_to(victim)
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(tmp_path),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert "Warning" in capsys.readouterr().err
        logger.warning("must not be written through the symlink")
        assert victim.read_text() == ""

    def test_log_level_debug_enables_debug_without_file_handler(self) -> None:
        logger = logging.getLogger("test.debug.level_only")
        logger.handlers = []
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "DEBUG"}, clear=True):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert logger.level == logging.DEBUG
        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_debug_file_can_use_info_runtime_level(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.file_info_level")
        log_file = tmp_path / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
                "DEEPAGENTS_CODE_LOG_LEVEL": "INFO",
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            assert logger.level == logging.INFO
            assert file_handlers
            assert file_handlers[-1].level == logging.INFO
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_invalid_log_level_warns_and_uses_default(self, capsys) -> None:
        logger = logging.getLogger("test.debug.bad_level")
        with patch.dict(os.environ, {"DEEPAGENTS_CODE_LOG_LEVEL": "TRACE"}, clear=True):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert logger.level == logging.INFO
        captured = capsys.readouterr()
        assert "DEEPAGENTS_CODE_LOG_LEVEL" in captured.err

    def test_thread_id_cannot_escape_directory(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.unsafe_thread")
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(tmp_path),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("../../victim")
        handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        try:
            assert len(handlers) == 1
            path = Path(handlers[0].baseFilename)
            assert path.parent == tmp_path
            assert path.name.startswith("thread-")
        finally:
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)

    def test_custom_path_used(self, tmp_path) -> None:
        logger = logging.getLogger("test.debug.custom_path")
        log_file = tmp_path / "custom" / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
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
        log_file = tmp_path / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            assert len(file_handlers) == 1
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_changed_thread_swaps_handler(self, tmp_path) -> None:
        """Binding a new thread replaces the stale handler, not stacks."""
        logger = logging.getLogger("test.debug.swap")
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(tmp_path),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("first")
            bind_debug_logging_to_thread("second")

        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        try:
            assert len(file_handlers) == 1
            assert str(tmp_path / "second.log") == file_handlers[0].baseFilename
        finally:
            for h in file_handlers:
                h.close()
                logger.removeHandler(h)

    def test_failed_thread_rotation_removes_stale_handler(
        self, tmp_path, capsys
    ) -> None:
        logger = logging.getLogger("test.debug.failed_rotation")
        directory = tmp_path / "debug"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(directory),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("first")
            assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            with patch.object(
                _debug, "_prepare_debug_directory", side_effect=OSError("nope")
            ):
                bind_debug_logging_to_thread("second")

        assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        logger.warning("must not reach the first thread")
        assert (
            "must not reach the first thread"
            not in (directory / "first.log").read_text()
        )
        assert "Warning" in capsys.readouterr().err

    def test_legacy_debug_file_uses_parent_directory(self, tmp_path) -> None:
        legacy_file = tmp_path / "legacy" / "debug.log"
        with patch.dict(
            os.environ,
            {"DEEPAGENTS_CODE_DEBUG_FILE": str(legacy_file)},
            clear=True,
        ):
            assert _debug._debug_directory() == legacy_file.parent

    def test_debug_directory_overrides_legacy_file(self, tmp_path) -> None:
        directory = tmp_path / "current"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(directory),
                "DEEPAGENTS_CODE_DEBUG_FILE": str(tmp_path / "legacy" / "debug.log"),
            },
            clear=True,
        ):
            assert _debug._debug_directory() == directory

    def test_legacy_debug_file_config_uses_parent_directory(self, tmp_path) -> None:
        legacy_file = tmp_path / "legacy" / "debug.log"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "deepagents_code.config_manifest.load_config_toml",
                return_value={"debug": {"file": str(legacy_file)}},
            ),
        ):
            assert _debug._debug_directory() == legacy_file.parent

    def test_untagged_handler_does_not_block_configuration(self, tmp_path) -> None:
        """A foreign FileHandler on the same path must not suppress our handler."""
        logger = logging.getLogger("test.debug.untagged")
        log_file = tmp_path / "test-thread.log"
        foreign = logging.FileHandler(str(log_file), mode="a")
        logger.addHandler(foreign)
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")

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
        log_file = tmp_path / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")

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
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": "/nonexistent_dir/debug.log",
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        assert len(logger.handlers) == original_count
        captured = capsys.readouterr()
        assert "Warning" in captured.err


class TestInstalledDebugLogPath:
    def test_returns_none_when_no_handler(self) -> None:
        """Absent a tagged handler, the helper reports no log file."""
        logger = logging.getLogger("deepagents_code")
        original = list(logger.handlers)
        for h in logger.handlers[:]:
            if getattr(h, "_deepagents_code_debug_handler", False):
                logger.removeHandler(h)
        try:
            assert installed_debug_log_path() is None
        finally:
            for h in logger.handlers[:]:
                if h not in original:
                    logger.removeHandler(h)
            for h in original:
                if h not in logger.handlers:
                    logger.addHandler(h)

    def test_returns_path_when_handler_installed(self, tmp_path) -> None:
        """The helper returns the path of the actually-installed handler."""
        logger = logging.getLogger("deepagents_code")
        log_file = tmp_path / "installed" / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        installed = [
            h
            for h in logger.handlers
            if getattr(h, "_deepagents_code_debug_handler", False)
        ]
        try:
            assert installed_debug_log_path() == log_file
        finally:
            for h in installed:
                h.close()
                logger.removeHandler(h)

    def test_ignores_untagged_file_handler(self, tmp_path) -> None:
        """A foreign FileHandler does not count as an installed debug log.

        Mirrors the divergence the helper exists to catch: a truthy
        `DEEPAGENTS_CODE_DEBUG` set after import (e.g. via `.env`) never installs
        our tagged handler, so the helper must report `None` regardless of any
        unrelated handlers present.
        """
        logger = logging.getLogger("deepagents_code")
        pre_existing = [
            h
            for h in logger.handlers
            if getattr(h, "_deepagents_code_debug_handler", False)
        ]
        for h in pre_existing:
            logger.removeHandler(h)
        foreign = logging.FileHandler(str(tmp_path / "foreign.log"), mode="a")
        logger.addHandler(foreign)
        try:
            with patch.dict(os.environ, {"DEEPAGENTS_CODE_DEBUG": "1"}, clear=True):
                # Env is truthy but no tagged handler was installed.
                assert installed_debug_log_path() is None
        finally:
            foreign.close()
            logger.removeHandler(foreign)
            for h in pre_existing:
                logger.addHandler(h)

    def test_package_import_configures_package_logger(self, tmp_path) -> None:
        logger = logging.getLogger("deepagents_code")
        original_handlers = list(logger.handlers)
        original_level = logger.level
        log_file = tmp_path / "package" / "test-thread.log"
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(log_file.parent),
            },
        ):
            importlib.reload(deepagents_code)
            bind_debug_logging_to_thread("test-thread")

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


class TestSweepDebugHandlers:
    """Tests for the conftest sweep that closes leaked debug handlers.

    The negative case is the one worth pinning: every other test in this file
    installs its handlers inside the test body, i.e. after the setup-only
    `_close_leaked_debug_handlers` fixture has already run, so none of them
    would notice if the sweep started closing handlers it did not own.
    """

    def test_leaves_untagged_file_handler_attached_and_open(
        self, tmp_path, sweep_debug_handlers
    ) -> None:
        """An unrelated `FileHandler` must survive the sweep untouched."""
        logger = logging.getLogger("test.sweep.untagged")
        foreign = logging.FileHandler(tmp_path / "foreign.log")
        logger.addHandler(foreign)
        try:
            sweep_debug_handlers()

            assert foreign in logger.handlers
            assert foreign.stream is not None
            assert not foreign.stream.closed
        finally:
            logger.removeHandler(foreign)
            foreign.close()

    def test_closes_and_detaches_tagged_file_handler(
        self, tmp_path, sweep_debug_handlers
    ) -> None:
        """A `configure_debug_logging` handler is removed and closed."""
        logger = logging.getLogger("test.sweep.tagged")
        with patch.dict(
            os.environ,
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_DEBUG_DIRECTORY": str(tmp_path),
            },
        ):
            configure_debug_logging(logger)
            bind_debug_logging_to_thread("test-thread")
        tagged = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert tagged, "configure_debug_logging installed no FileHandler"

        sweep_debug_handlers()

        assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)], (
            "sweep left a tagged handler attached"
        )
        for handler in tagged:
            assert handler.stream is None or handler.stream.closed
