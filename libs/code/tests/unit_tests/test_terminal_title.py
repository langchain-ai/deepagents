"""Terminal title lifecycle and escape-sequence safety."""

from __future__ import annotations

import io

import pytest

from deepagents_code.terminal_title import TerminalTitle


class TerminalStream(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> TerminalStream:
    stream = TerminalStream()
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stderr__", stream)
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stdout__", io.StringIO())
    monkeypatch.setattr(
        "deepagents_code.terminal_title.invoked_name", lambda: "dcode-dev"
    )
    return stream


def test_title_lifecycle_deduplicates_and_restores(terminal: TerminalStream) -> None:
    title = TerminalTitle("{app_name} - {thread_name} | {cwd} [{branch}]")
    title.update(thread_name="Not started")
    assert terminal.getvalue() == ""
    title.start()
    title.start()
    title.update(thread_name="Cache repair", cwd="/work", branch="main")
    title.update(thread_name="Cache repair", cwd="/work", branch="main")
    title.update(thread_name="New name", cwd="/work", branch="main")
    title.restore()
    title.restore()
    title.update(thread_name="Already stopped")
    assert terminal.getvalue() == (
        "\x1b[22;0t"
        "\x1b]0;dcode-dev - Cache repair | /work [main]\x07"
        "\x1b]0;dcode-dev - New name | /work [main]\x07"
        "\x1b[23;0t"
    )


def test_terminal_control_characters_are_never_emitted_in_title(
    terminal: TerminalStream,
) -> None:
    title = TerminalTitle("\x1b{thread_name}\n{cwd}\x9c{branch}")
    title.start()
    title.update(
        thread_name="Cache\x07\x1b]2;Injected\x9c",
        cwd="/work\r\n\x00",
        branch="main\x1b\\",
    )
    output = terminal.getvalue()
    assert output.startswith("\x1b[22;0t\x1b]0;")
    payload = output.removeprefix("\x1b[22;0t\x1b]0;").removesuffix("\x07")
    assert payload.isprintable()
    assert output.count("\x07") == 1
    assert output.count("\x1b") == 2


@pytest.mark.parametrize(
    "template",
    [
        "{unknown}",
        "{thread_name.__class__}",
        "{thread_name[0]}",
        "{thread_name!r}",
        "{thread_name:999999999}",
        "{thread_name",
        "thread_name}",
    ],
)
def test_invalid_template_uses_safe_default(
    terminal: TerminalStream, template: str
) -> None:
    title = TerminalTitle(template)
    title.start()
    title.update(thread_name="Cache repair")
    assert terminal.getvalue() == "\x1b[22;0t\x1b]0;dcode-dev\x07"


def test_literal_template_braces(terminal: TerminalStream) -> None:
    title = TerminalTitle("{{{thread_name}}}")
    title.start()
    title.update(thread_name="Cache")
    assert terminal.getvalue() == "\x1b[22;0t\x1b]0;{Cache}\x07"


def test_title_length_is_bounded(terminal: TerminalStream) -> None:
    title = TerminalTitle("{cwd}")
    title.start()
    title.update(cwd="x" * 10000)
    assert terminal.getvalue() == "\x1b[22;0t\x1b]0;" + "x" * 512 + "\x07"


def test_redirected_streams_receive_no_escape_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr, stdout = io.StringIO(), io.StringIO()
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stderr__", stderr)
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stdout__", stdout)
    title = TerminalTitle("{thread_name}")
    title.start()
    title.update(thread_name="Cache")
    title.restore()
    assert stderr.getvalue() == ""
    assert stdout.getvalue() == ""


def test_redirected_stderr_falls_back_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stderr, stdout = io.StringIO(), TerminalStream()
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stderr__", stderr)
    monkeypatch.setattr("deepagents_code.terminal_title.sys.__stdout__", stdout)
    title = TerminalTitle("{thread_name}")
    title.start()
    title.update(thread_name="Cache")
    title.restore()
    assert stderr.getvalue() == ""
    assert stdout.getvalue() == "\x1b[22;0t\x1b]0;Cache\x07\x1b[23;0t"


def test_closed_terminal_does_not_break_cleanup(terminal: TerminalStream) -> None:
    title = TerminalTitle("{thread_name}")
    title.start()
    terminal.close()
    title.update(thread_name="Cache")
    title.restore()
    title.restore()
