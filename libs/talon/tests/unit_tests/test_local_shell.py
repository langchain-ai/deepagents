"""Behavior of Talon's default local command execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from deepagents_talon.runtime import _default_backend

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_timeout_terminates_children_and_reports_incomplete_execution(
    tmp_path: Path, *, asynchronous: bool
) -> None:
    backend = _default_backend({"DEEPAGENTS_TALON_WORKSPACE": str(tmp_path)}, tmp_path)
    command = "(sleep 2; printf survived > survived) & printf started > started; wait"
    if asynchronous:
        result = await backend.aexecute(command, timeout=1)
    else:
        result = backend.execute(command, timeout=1)

    assert (tmp_path / "started").read_text() == "started"
    assert result.exit_code == 124
    assert "timed out after 1 seconds" in result.output
    assert "process group was terminated" in result.output
    assert "execution is incomplete" in result.output
    await asyncio.sleep(1.2)
    assert not (tmp_path / "survived").exists()
    assert backend.execute("printf recovered", timeout=3).output == "recovered"


@pytest.mark.parametrize(
    ("command", "output", "exit_code"),
    [
        ("printf hello", "hello", 0),
        ("true", "<no output>", 0),
        ("printf error >&2; exit 7", "[stderr] error\n\nExit code: 7", 7),
        ("read value; printf done", "done", 0),
        ("", "Error: Command must be a non-empty string.", 1),
    ],
)
def test_command_results(tmp_path: Path, command: str, output: str, exit_code: int) -> None:
    backend = _default_backend({"DEEPAGENTS_TALON_WORKSPACE": str(tmp_path)}, tmp_path)
    result = backend.execute(command)
    assert result.output == output
    assert result.exit_code == exit_code
    assert not result.truncated


def test_command_output_limit_and_workspace(tmp_path: Path) -> None:
    backend = _default_backend({"DEEPAGENTS_TALON_WORKSPACE": str(tmp_path)}, tmp_path)
    (tmp_path / "large").write_text("x" * 100_001)
    result = backend.execute("cat large")
    assert result.exit_code == 0
    assert result.truncated
    assert result.output == "x" * 100_000 + "\n\n... Output truncated at 100000 bytes."


def test_invalid_timeout(tmp_path: Path) -> None:
    backend = _default_backend({}, tmp_path)
    with pytest.raises(ValueError, match="timeout must be positive"):
        backend.execute("true", timeout=0)


def test_execution_failure_is_reported(tmp_path: Path) -> None:
    backend = _default_backend({}, tmp_path)
    result = backend.execute("invalid\0command")
    assert result.exit_code == 1
    assert "Error executing command (ValueError)" in result.output
