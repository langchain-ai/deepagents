"""Unit tests for terminal multiplexer detection and option probing."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from deepagents_code import multiplexer
from deepagents_code.multiplexer import (
    ALLOW_PASSTHROUGH,
    FOCUS_EVENTS,
    SET_CLIPBOARD,
    inside_tmux,
    parse_tmux_options,
    query_tmux_status,
)

if TYPE_CHECKING:
    import pytest

_SHOW_OPTIONS_OUTPUT = """\
activity-action other
focus-events off
set-clipboard external
default-command ''
allow-passthrough off
"""


class TestInsideTmux:
    """Tests for tmux pane detection."""

    def test_true_when_socket_exported(self) -> None:
        """Tmux exports `TMUX` into every pane it owns."""
        assert inside_tmux({"TMUX": "/tmp/tmux-0/default,1,0"}) is True

    def test_false_when_absent(self) -> None:
        """A bare terminal has no `TMUX`."""
        assert inside_tmux({"TERM": "xterm-256color"}) is False

    def test_false_when_empty(self) -> None:
        """An exported-but-empty `TMUX` does not indicate a pane."""
        assert inside_tmux({"TMUX": ""}) is False


class TestParseTmuxOptions:
    """Tests for `show-options` output parsing."""

    def test_extracts_reported_options(self) -> None:
        """Only the options that change app behavior are kept."""
        assert parse_tmux_options(_SHOW_OPTIONS_OUTPUT) == {
            FOCUS_EVENTS: "off",
            SET_CLIPBOARD: "external",
            ALLOW_PASSTHROUGH: "off",
        }

    def test_omits_options_tmux_does_not_know(self) -> None:
        """An older tmux without `allow-passthrough` simply omits the line."""
        parsed = parse_tmux_options("focus-events on\nset-clipboard on\n")
        assert ALLOW_PASSTHROUGH not in parsed

    def test_ignores_valueless_lines(self) -> None:
        """A line with no separator is not an option assignment."""
        assert parse_tmux_options("focus-events\n") == {}

    def test_empty_output_yields_no_options(self) -> None:
        """An empty answer is a valid answer with nothing in it."""
        assert parse_tmux_options("") == {}


class TestQueryTmuxStatus:
    """Tests for the end-to-end probe."""

    def test_returns_none_outside_tmux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe must not shell out when there is no pane to ask about."""
        monkeypatch.delenv("TMUX", raising=False)
        assert query_tmux_status() is None

    def test_reports_version_and_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cooperative server yields both facts."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,1,0")

        def fake_run(args: list[str]) -> str:
            return "tmux 3.5a\n" if tuple(args) == ("-V",) else _SHOW_OPTIONS_OUTPUT

        monkeypatch.setattr(multiplexer, "_run_tmux", fake_run)

        status = query_tmux_status()
        assert status is not None
        assert status.version == "tmux 3.5a"
        assert status.options == {
            FOCUS_EVENTS: "off",
            SET_CLIPBOARD: "external",
            ALLOW_PASSTHROUGH: "off",
        }

    def test_failed_probe_reports_none_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed query must stay distinguishable from an empty answer."""
        monkeypatch.setenv("TMUX", "/tmp/tmux-0/default,1,0")
        monkeypatch.setattr(multiplexer, "_run_tmux", lambda _args: None)

        status = query_tmux_status()
        assert status is not None
        assert status.version is None
        assert status.options is None


class TestRunTmux:
    """Tests for the subprocess wrapper's failure handling."""

    def test_missing_binary_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A machine without tmux on `PATH` should not raise."""

        def raise_missing(*_args: object, **_kwargs: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_missing)
        assert multiplexer._run_tmux(["-V"]) is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wedged server must not hang `doctor`."""

        def raise_timeout(*_args: object, **_kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="tmux", timeout=1)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert multiplexer._run_tmux(["-V"]) is None

    def test_nonzero_exit_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rejected command yields no facts rather than partial output."""
        completed = subprocess.CompletedProcess(
            args=["tmux"], returncode=1, stdout="usage: tmux\n", stderr=""
        )
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed)
        assert multiplexer._run_tmux(["show-options", "-s"]) is None

    def test_success_returns_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful call hands back raw stdout for the caller to parse."""
        completed = subprocess.CompletedProcess(
            args=["tmux"], returncode=0, stdout="tmux 3.5a\n", stderr=""
        )
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed)
        assert multiplexer._run_tmux(["-V"]) == "tmux 3.5a\n"
