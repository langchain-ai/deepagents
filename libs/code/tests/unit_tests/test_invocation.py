"""Unit tests for `deepagents_code._invocation`."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pytest

from deepagents_code._env_vars import DEBUG, INVOKED_AS
from deepagents_code._invocation import (
    DEFAULT_INVOKED_NAME,
    STANDARD_INVOKED_NAMES,
    invoked_name,
    log_nonstandard_invoked_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_invoked_name_cache() -> Iterator[None]:
    """Drop the process-lifetime caches around each test.

    Clearing afterwards too keeps a patched `argv[0]` from leaking into other
    modules' tests through the caches.
    """
    invoked_name.cache_clear()
    log_nonstandard_invoked_name.cache_clear()
    yield
    invoked_name.cache_clear()
    log_nonstandard_invoked_name.cache_clear()


class TestInvokedNameFromArgv:
    """Resolution of the launch name from `sys.argv[0]`."""

    @pytest.mark.parametrize(
        "argv0",
        [
            "dcode",
            "/usr/local/bin/dcode",
            "./.superset/bin/dcode",
        ],
    )
    def test_reports_console_script_name(
        self, monkeypatch: pytest.MonkeyPatch, argv0: str
    ) -> None:
        """The basename of `argv[0]` is the command the user typed."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", [argv0, "-r", "thread-1"])

        assert invoked_name() == "dcode"

    def test_reports_alias_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The longer `deepagents-code` console script is reported verbatim."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", ["/usr/local/bin/deepagents-code"])

        assert invoked_name() == "deepagents-code"

    def test_reports_shim_name_not_symlink_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A renamed shim reports its own name, not the script it points at.

        Mirrors the per-worktree setup where `~/.local/bin/abc` is a symlink to
        a checkout's `bin/dcode`: the kernel hands the interpreter the pathname
        passed to `execve`, so `argv[0]` keeps the shim name.
        """
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", ["/home/user/.local/bin/abc"])

        assert invoked_name() == "abc"

    def test_module_entry_point_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`python -m deepagents_code` reports `__main__.py`, which is unusable."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(
            sys, "argv", ["/venv/lib/deepagents_code/__main__.py", "-r", "t"]
        )

        assert invoked_name() == DEFAULT_INVOKED_NAME

    @pytest.mark.parametrize(
        "argv0",
        [
            "",
            "-c",
            "python3.13",
            "/usr/bin/python",
            "run dcode",
            "dcode; rm -rf /",
            "dcode$(id)",
            "a" * 65,
        ],
    )
    def test_implausible_names_fall_back(
        self, monkeypatch: pytest.MonkeyPatch, argv0: str
    ) -> None:
        """Anything that is not a plain command name falls back to the default.

        `argv[0]` is supplied by whatever started the process and is rendered
        into a copy-pasteable command, so interpreter names, shell
        metacharacters, whitespace, and absurd lengths are rejected.
        """
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", [argv0])

        assert invoked_name() == DEFAULT_INVOKED_NAME

    def test_empty_argv_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An embedded interpreter may leave `sys.argv` empty."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", [])

        assert invoked_name() == DEFAULT_INVOKED_NAME

    def test_windows_executable_suffix_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows console scripts are `.exe` wrappers; users type the stem."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", ["dcode.exe"])

        assert invoked_name() == "dcode"

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The name is fixed for the process, so later argv edits are ignored."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", ["abc"])
        assert invoked_name() == "abc"

        monkeypatch.setattr(sys, "argv", ["dcode"])

        assert invoked_name() == "abc"


class TestInvokedNameFromEnv:
    """The re-exec sentinel takes priority over `argv[0]`."""

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After the auto-update re-exec, only the sentinel knows the name."""
        monkeypatch.setenv(INVOKED_AS, "abc")
        monkeypatch.setattr(
            sys, "argv", ["/venv/lib/deepagents_code/__main__.py", "-r", "t"]
        )

        assert invoked_name() == "abc"

    def test_implausible_override_falls_back_to_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A junk sentinel must not shadow a usable `argv[0]`."""
        monkeypatch.setenv(INVOKED_AS, "rm -rf /")
        monkeypatch.setattr(sys, "argv", ["/usr/local/bin/dcode"])

        assert invoked_name() == "dcode"

    def test_empty_override_falls_back_to_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty sentinel is treated as absent."""
        monkeypatch.setenv(INVOKED_AS, "")
        monkeypatch.setattr(sys, "argv", ["/usr/local/bin/abc"])

        assert invoked_name() == "abc"


class TestLogNonstandardInvokedName:
    """The once-per-process Debug Console note for shim/alias launches."""

    def test_logs_at_info_when_debug_mode_off(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With debug off, INFO passes the package logger's buffer floor.

        A plain DEBUG record would be filtered before reaching the in-memory
        buffer that backs the Debug Console, so the note would be lost.
        """
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.delenv(DEBUG, raising=False)
        monkeypatch.setattr(sys, "argv", ["/home/user/.local/bin/abc"])

        with caplog.at_level(logging.INFO, logger="deepagents_code._invocation"):
            log_nonstandard_invoked_name()

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.INFO
        assert "'abc'" in record.getMessage()

    def test_logs_at_debug_when_debug_mode_on(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With debug on, the note uses DEBUG so it joins the debug log file."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setenv(DEBUG, "1")
        monkeypatch.setattr(sys, "argv", ["/home/user/.local/bin/abc"])

        with caplog.at_level(logging.DEBUG, logger="deepagents_code._invocation"):
            log_nonstandard_invoked_name()

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.DEBUG
        assert "'abc'" in record.getMessage()

    @pytest.mark.parametrize("argv0", ["dcode", "deepagents-code"])
    def test_standard_names_log_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        argv0: str,
    ) -> None:
        """The shipped console scripts are ordinary launches; no note."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", [f"/usr/local/bin/{argv0}"])

        with caplog.at_level(logging.DEBUG, logger="deepagents_code._invocation"):
            log_nonstandard_invoked_name()

        assert caplog.records == []

    def test_logs_only_once_per_process(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The `invoked_name` cache suppresses repeat calls in one process."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.delenv(DEBUG, raising=False)
        monkeypatch.setattr(sys, "argv", ["/home/user/.local/bin/abc"])

        with caplog.at_level(logging.INFO, logger="deepagents_code._invocation"):
            log_nonstandard_invoked_name()
            log_nonstandard_invoked_name()

        assert len(caplog.records) == 1

    def test_fallback_name_logs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Falling back to the default is a standard name; no note."""
        monkeypatch.delenv(INVOKED_AS, raising=False)
        monkeypatch.setattr(sys, "argv", ["python3.13"])

        with caplog.at_level(logging.DEBUG, logger="deepagents_code._invocation"):
            log_nonstandard_invoked_name()

        assert caplog.records == []


def test_standard_names_match_pyproject_scripts() -> None:
    """Drift guard: `STANDARD_INVOKED_NAMES` covers `[project.scripts]`.

    The set is hand-maintained in `_invocation` (the module must stay
    import-light), so a console script added to `pyproject.toml` without a
    matching entry would start logging a spurious "non-standard" note for a
    shipped command.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    scripts = set(tomllib.loads(pyproject.read_text())["project"]["scripts"])

    assert scripts == set(STANDARD_INVOKED_NAMES)
